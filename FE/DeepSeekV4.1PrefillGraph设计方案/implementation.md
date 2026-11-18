# 实施拆分、测试与验收

基线：`sglang@dsv41-a5@9898f7ce9a`。本文把 [design.md](design.md) 落到具体代码触点和验证步骤。路径均相对 `sglang/`。

## 1. PR 拆分

### PR-0：代际闸门（安全基线）

**目标**：在实现 V4.1 契约之前，杜绝 V4.1 进入 V4-only 容器。

改动：

| 文件 | 改动 |
| --- | --- |
| `python/sglang/srt/hardware_backend/npu/attention/ascend_dsv4_backend.py` | 把 `_supports_dsv4_prefill_graph_capture()` 拆成 V4/V4.1 两个判定；V4.1 在契约未完成时返回 False；增加 `_dsv4_is_v41_model and (has_c4 or has_c128)` 的互斥断言；日志说明“V4.1 prefill graph not enabled, sources/attention stay eager breaks” |
| `test/registered/unit/npu/attention/test_npu_ascend_dsv4_backend.py` | V4.1 low-ratio 的 capability 返回 False；V4 路径不变；draft worker False；CP/DCP 仍显式报错 |
| `test/registered/unit/npu/graph_runner/test_dsv4_prefill_graph_metadata.py` | 保持 V4 用例通过 |

不改变：runner、factory、segment、MoE、DSpark、decode graph。

完成条件：

- V4.1 + `--cuda-graph-backend-prefill=breakable` 启动后，`use_captured_forward_metadata_for_breakable_cuda_graph=False`；
- source 与 attention 都走 `bcg_` eager break；
- 现有 V4/V4.1 CPU 回归全过。

### PR-1：V4.1 固定 metadata 与纯 helper

**目标**：在关闭图契约的状态下，把 V4.1 所需的状态全部纳入固定容器，并能通过 CPU 生命周期测试。

改动：

| 文件/函数 | 改动 |
| --- | --- |
| `dsv4_prefill_graph.py` | 新增共享基类 `_Dsv4PrefillGraphMetadataBase`（allocate/fill/refresh、共享序列标量、kernel metadata 拷贝、topk staging）；`Dsv4PrefillGraphMetadata` 保持 V4，新增 `Dsv41PrefillGraphMetadata` 完整承载 V4.1 字段（见 design.md §5.2）；backend 用 `_dsv4_is_v41_model` 显式选择类 |
| `ascend_dsv4_backend.py` `_dsv41_window_indices` / `_dsv41_cmp_topk_length` | 抽出纯 helper `_dsv41_fill_window_buffers` / `_dsv41_fill_cmp_lengths` / `_dsv41_fill_ratio_rows`：显式 `positions/live_mask/capacity` 输入，返回容量形状 buffer；eager 包装保留 |
| `ascend_dsv4_backend.py` `_dsv41_attention_kernel_metadata` / `_dsv41_index_kernel_metadata` | 抽成 `*_dispatch_metadata(*, cu, lengths, kv_lens, positions, ...)`；eager 与 refresh 共用 |
| `ascend_dsv4_backend.py` `_validate_v41_candidate_spec()` | 硬性校验 block size > 0 与 candidate block 数；记录 `cdiv(index_topk, block)` 与 `candidate_topk_blocks` 的覆盖关系，配置不满足时告警；页行整除由 `_dsv41_qsli_cache` 在使用时硬性检查 |
| `ascend_dsv4_backend.py` `can_run_prefill_cuda_graph` / `vote_prefill_cuda_graph` | 拒绝 MIXED、超 context；DP vote 与本地一致 |
| `test/registered/unit/npu/attention/test_npu_dsv41_low_ratio_backend.py` | 新增 `TestV41PrefillGraphMetadata` 等用例；见 §2 |
| `test/registered/unit/npu/graph_runner/test_dsv4_prefill_graph_metadata.py` | 补 V4 容器不含 V4.1 字段的隔离断言 |

PR-0 的闸门就是 `_supports_dsv41_prefill_graph_capture()` 本身：未实现时返回 False。最终代码不再保留额外的全局布尔开关；需要回退时用 CLI 关闭整张 prefill graph，或让该 predicate 返回 False。

### PR-2：原子启用 source + attention 入图

**目标**：在同一提交里打开 metadata 契约、source 路由和 attention 路由。

改动：

| 文件/函数 | 改动 |
| --- | --- |
| `ascend_dsv4_backend.py` `_supports_dsv41_prefill_graph_capture` | 返回 True（仅当 arch35/PP/candidate 等条件满足） |
| `ascend_dsv4_backend.py` `LowRatioCompressAscendBackendMixin` | 实例属性 `low_ratio_prefill_graph = _dsv4_is_v41_model and use_captured_forward_metadata`；更新过时注释 |
| `ascend_dsv4_backend.py` `_low_ratio_index_topk` | 通过 `graph_state.stage_topk(layer_id, topk)` 绑定逐层输出 |
| `ascend_dsv4_backend.py` `_forward_low_ratio_attention` | 有 graph_state 时从 `topk_by_layer[layer_id]` 取本层 topk |
| `deepseek_v4.py` | 无需改路由；`_low_ratio_sources_break` 自动读 backend 属性 |
| `prefill_cuda_graph_runner.py` | 仅当 bucket 大于 context 或 batch 超出容量时给出明确错误；不改变主流程 |
| `test/registered/unit/npu/attention/test_npu_dsv41_low_ratio_backend.py` | 增加 helper、路由配对、逐层 topk、refresh 闭环和准入用例 |
| `test/registered/unit/npu/graph_runner/test_dsv4_prefill_graph_metadata.py` | 保持 V4 生命周期回归 |

完成条件（CPU）：

- V4.1 图契约开启时，source 不再 break、attention 走裸函数；
- 契约关闭时，两者都走 break；
- metadata 刷新不重绑任何字段；
- 所有新增与既有 CPU 回归通过。

### PR-3：设备验证后的修正

设备组合探针暴露的问题在这一阶段修：

- metadata/ABI 形状；
- 死行/尾部 sentinel；
- bank/页隔离；
- MoE/Engram/mHC 顺序；
- DSpark 输出。

不预留“先桥接再删除”的代码路径。

### PR-4：整模与性能

- 打开首批 A5 配置；
- 整模正确性 + 性能 A/B；
- 记录 bucket、capture memory、hit/fallback；
- 更新模型 cookbook（若启动参数或推荐配置变化）。

## 2. CPU 测试

测试文件优先复用现有子系统；新增只放在已有对应文件或必要的独立 metadata 文件，避免每文件一个解释器。

### 2.1 闸门与路由

| 用例 | 防住的错误 | 类别 |
| --- | --- | --- |
| V4.1 contract=False 时 `use_captured...=False`、`low_ratio_prefill_graph=False` | 当前代码把 V4.1 错路由进 V4 容器 | bug regression |
| contract=True 时两者同时为 True | 半捕获（source eager + attention captured） | derived property |
| V4 (0/4/128) 仍走原 V4 字段与 key | 代际回归 | derived property |
| CP/DCP/TBO/PP!=1 的启动拒绝 | 不支持配置静默 replay | negative branch |
| MIXED/超 context 的 `can_run_prefill_cuda_graph` 返回 False | MIXED 被映射成 EXTEND replay | derived property |

### 2.2 metadata 生命周期

| 用例 | 防住的错误 | 类别 |
| --- | --- | --- |
| `allocate` 后所有 V4.1 字段形状/dtype/别名关系正确；页表列数等于 `req_to_token.shape[1] // page_size` | 字段缺失、页表未共享或用 context_len 少算池 extra 行 | bookkeeping |
| capture seed -> 修改 live -> `refresh`，所有 data_ptr 不变、内容更新 | capture 后重绑 | bug regression |
| ratio1 行序不 compact；ratio2 稳定 compact | 行序错位导致 loc/topk/attention 错配 | derived property |
| `T_live < T_cap`：尾部 0/-1、mask false、长度 0 | 串读上一轮尾部 | derived property |
| page/index 页表越界抛错，不截断 | 静默丢数据 | negative branch |
| `kernel_metadata` key 集合变化抛错 | 缺键补零导致非法 schedule | negative branch |
| 逐层 topk map 不互相覆盖 | 共享属性只记住最后一层 | bug regression |
| idle 防御路径：cu `[0,1]`、loc=-1、长度 0 | idle replay 写坏状态 | derived property |
| `_fill_idle` 不触碰真实 bank/页 | 隔离失败 | bug regression |

### 2.3 纯 helper

对 `_dsv41_fill_window_buffers`、`_dsv41_fill_cmp_lengths`、`_dsv41_fill_ratio_rows`、`_dsv41_attention_dispatch_metadata`、`_dsv41_index_dispatch_metadata` 的 CPU 测试使用已有的 fake `ds41` stub：

- 容量输入 + live cu：死行长度为 0；
- ratio1/2 的 position/loc 公式与 eager `_dsv41_ratio_row_values` 一致；
- candidate 可见块公式在边界 prefix/ratio 上正确；
- metadata tensor 整块覆盖，不保留上一轮尾部。

### 2.4 既有回归

必须保持：

- `test/registered/unit/npu/attention/test_npu_ascend_dsv4_backend.py`（V4 attention 63 项）
- `test/registered/unit/npu/attention/test_npu_dsv41_low_ratio_backend.py`（V4.1 low-ratio 101 项）
- `test/registered/unit/npu/graph_runner/test_dsv4_prefill_graph_metadata.py`
- `test/registered/unit/npu/graph_runner/test_breakable_npu_graph_backend.py`
- `test/registered/unit/moe/test_ascend_ops.py`
- `test/registered/unit/layers/test_engram_history.py`
- 模型平台/mHC 分流测试

CPU 只证明协议与生命周期，不证明 NPU aclgraph 能执行组合图。

## 3. A5 组合探针（实现后第一批设备工作）

现有脚本 `FE/DeepSeekV4.1PrefillGraph设计方案/scripts/verify_dsv41_prefill_graph.py` 已覆盖单算子。本方案不重复跑全部单算子，新增以下组合场景。

### 3.1 P1：一层完整 source + attention 同段捕获

这是最关键的设备缺口。

配置：T=64 或 128，B=1，full page=128，index page=64，candidate block=8，main heads=64，head_dim=512，ratios 1/2 各一组。

每个 case 执行：

```text
eager 参考：真实 source -> attention
capture：
  compressor -> indexer_prologue_k(mode 0/1) -> QLI/QSLI(+candidate) -> MQSMLA
replay：
  刷新 loc/page/index/length/bank -> 同一张图 replay
对照：
  输出、compressed KV 行、index K 行、compressor state bank、candidate blocks
```

覆盖：

| 维度 | 取值 |
| --- | --- |
| live T | 64 / 31 / 16 / 61 / 64（同 bucket） |
| prefix | 0 / 1 / 128-1 / 4*128+1 |
| ratio | 1 / 2；candidate source=1，consumer=1/2 |
| bank | 0（capture）/1/2；ring wrap |
| 跨 chunk | 奇数 prefix -> 下一 chunk；pending pair |
| 输出 | attention、压缩 KV、index K、state、candidate |

判定：

- 同一 bucket 内改变 live T/prefix/page 后，graph 与 eager 逐元素一致；
- 所有固定输入 data_ptr 不变；
- capture 前后真实 bank/页内容不变；
- writer 写出的缓存被图内 MQSMLA 消费（已在 run4 单链验证，这里要求同段）。

失败处理：若组合图不可捕获或语义不符，**不实现半捕获**；保持 contract=False，记录失败点，回到设计评估。

### 3.2 P2：真实 metadata 容器刷新

不手工构造 metadata，而是使用 `Dsv41PrefillGraphMetadata`：

1. capture 一个 bucket；
2. 用 5 组 live batch（长度/prefix/page 不同）调用 `refresh`；
3. 断言固定字段地址、尾部、dispatch metadata、页表/索引表全部正确；
4. replay 图，输出与 eager 一致。

重点：ratio1 行序、ratio2 compact、死行零长度、index 页表扩展、cycle bank 切换。

### 3.3 P3：allocator 隔离

- 启动真实 `ReqToTokenPool`，确认 `free_slots` 不含 0；
- 分配两个真实请求（slot 1/2），写入哨兵值；
- 执行 capture/warmup；
- 比较 bank 1/2、真实压缩页、index 页、SWA 页，必须不变；
- replay 切到 slot 1，下一 chunk 继续，再切到 slot 2；
- 释放/复用 slot 后再次 replay。

### 3.4 P4：外围组合

| 组件 | 验证 |
| --- | --- |
| MoE | prefill graph scope 走 AllGather；固定行数；无 capture 期 host 计数；TP 组输出正确 |
| Engram | hash break 每 replay 执行一次；lookup 写固定 out；history 不重复提交 |
| mHC | premix → post 的 stats 依赖；Engram 改写 residual 后不复用旧 stats |
| DSpark | target prefill 的 aux hidden/token indices；draft KV 注入；下一轮生成 |
| logprob | prompt logprob 走 eager tail，与 eager 一致 |

## 4. 整模验收

### 4.1 配置记录

每次结果记录：模型/checkpoint revision、主仓 SHA、A5 型号和卡数、CANN/driver/torch/torch_npu/sgl-kernel-npu、SDK 算子包、TP/EP/DP/CP/DCP/PP、量化、KV dtype、Engram、DSpark、prefill buckets、request/context 上限、完整启动命令。

启动配置以模型 cookbook（`docs/cookbook/autoregressive/DeepSeek/DeepSeek-V4_1.mdx`）和本分支实际 A5 配置为准；`test/` 配置只作为可复现 CI 配置并注明差异。

### 4.2 正确性矩阵

| 维度 | 用例 |
| --- | --- |
| 长度 | bucket 原长、可准入的非整 bucket、压缩边界前后、context 上限及超限回退 |
| 历史 | 无 prefix、有 prefix、连续 chunk、prefix 复用、新请求替换旧 slot |
| replay 顺序 | 大→小、小→大、两个不同 prompt 同 bucket、多轮 eager/replay 交替 |
| 请求数 | 首版 B=1；B>1 必须被拒绝 |
| 并行 | 固定 TP/EP；DP 每 rank 有 token；idle rank 回退 eager |
| 输出 | 首 token/logits、prompt logprob、hidden states、后续 decode、完整响应 |
| 投机 | 无投机与 DSpark target prefill；draft 侧 eager |
| 回退 | 无 bucket、超 capacity、MIXED、未验证拓扑：各 rank 一致拒绝或 eager |

准确率使用 `sgl-eval`，同一 harness、同一采样参数；测试内走 `sglang.test.sgl_eval_utils.run_sgl_eval`，服务外部 CLI 使用 `test` extra 固定版本。graph/eager 只改变 prefill graph 开关；不能用手写评分或旧 `run_eval`。

### 4.3 性能 A/B

唯一变量：V4.1 prefill graph 开关。记录：

- prefill forward 时延分布、CPU 下发开销；
- TTFT、prefill 吞吐、后续 decode 指标；
- graph hit/fallback 次数与原因、每轮 live/bucket 行数、padding 放大；
- 图数量、segment 数、eager break 清单、图内 compressor/indexer/attention/MoE 证据；
- capture memory、workspace、隔离槽开销；
- 小 chunk、满 bucket、明显 padding 三类输入的分布。

首轮可选 512/1024/2048 chunk 对照，但 bucket 最终按 A5 工作负载和 memory budget 决定。

## 5. 排障映射

| 现象 | 优先检查 |
| --- | --- |
| capture 时 `win_metadata/c1a_metadata` 缺失 | 是否仍进入 V4 容器；V4.1 seed key 集合是否被正确采集 |
| 同 bucket 较小 live T 输出错乱 | 尾部 loc/length 是否清零；ratio2 compact 尾部是否 `-1`；live mask 是否来自 `cu[-1]` |
| 第二个请求读到上一个请求的 KV/topk | 页表/index 表是否按本轮刷新；capture 是否污染了真实 bank/页；`dsv4_cycle_state_block_table` 是否切换 |
| 连续 chunk 后精度漂移 | ratio-2 `start_pos`、pending pair、ring wrap、capture bank 0 是否与 live bank 混用 |
| 多 rank hang | MoE 固定行数、DP vote、bucket 选择是否各 rank 一致；是否有人走了半捕获 |
| capture 直接报错或回退 | source/attention op 是否有 host sync/动态输出；组合图 ABI 是否支持 |
| 整模首 token 正确、后续 decode 错 | target prefill 的 aux hidden/token indices、draft KV 注入、Engram history |
| graph 命中但无收益 | 段内是否仍有 host 读取/小算子；dispatch metadata 构建占比；padding 放大 |

## 6. 完成判定

同时满足：

1. 组合图 P1/P2/P3 通过；
2. 同 bucket 多轮 replay 的数值、缓存、state 与 eager 一致；
3. capture/warmup 不污染真实请求；
4. MIXED、B>1、CP/DCP/PP/TBO、draft、多模态等有明确一致的回退；
5. 整模 logits/生成/后续 decode 与 eager 对齐；
6. Engram/mHC/DSpark 的边界行为正确；
7. 性能与内存数据完整，收益与成本如实记录；
8. 独立精度议题单独保留。

## 7. 回退与发布

- 发布粒度：**整个 V4.1 prefill 契约**，按模型/硬件配置启用；不支持则 eager。
- 任一设备验证失败：把契约完成标志置 False，source/attention 回到 eager break；不保留半捕获代码。
- 新增或修改 SDK/CANN/torch_npu 后，重跑 P1/P2/P3 的受影响子集；不能只凭 Python 源码 hash 相同推广。
- 主仓稳定回归进入对应测试子系统；一次性证据放 `validation/` 并带环境与 SHA。
