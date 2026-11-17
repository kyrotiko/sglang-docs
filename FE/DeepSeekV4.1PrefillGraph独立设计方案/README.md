# DeepSeek V4.1 NPU Prefill 入图：独立设计方案

日期：2026-11-17
主仓基线：`sglang` 分支 `dsv41-a5`，HEAD `9898f7ce9a`
关联需求：在 Ascend A5 上把 DeepSeek V4.1 的 target prefill 放进 breakable NPU graph，降低 prefill 的 CPU 算子下发开销和 kernel 启动开销，同时保持 eager 路径的数值、KV 写入和跨 chunk 状态语义。

本文目录与 [参考设计目录](../DeepSeekV4.1PrefillGraph设计方案/README.md) 相互独立。参考目录中的单算子设备证据、提交映射和部分结论被采用；**架构选择、片内数据流边界、metadata 组织形式和验收顺序由本文独立给出**，分歧集中记录在 [design.md 第 10 节](design.md#10-与参考方案的关键分歧)。

## 1. 一句话结论

**复用现有 breakable NPU prefill graph 的工厂、segment、capture/replay 生命周期，不新建 runner；为 V4.1 增加一套代际独占的固定 metadata 契约，并把 ratio-1/2 的 source（compressor + index K + QLI/QSLI/candidate）与 MQSMLA attention 作为同一个原子数据流一起入图。**

首版不采用“source 留 eager、attention 入图”的半捕获桥接：QLI/QSLI 的 topk 输出没有可写的固定输出参数，桥接只能每层每轮复制 `T_cap x index_topk` 的 int32 张量，代价与分片协议复杂度都不划算；而且一旦 source 与 consumer 分处图的边界两侧，Python 属性赋值、候选块归属和逐层 topk 地址都会成为新的正确性面。

如果某类 source 在当前 SDK 上被证明不可捕获，**整个 V4.1 prefill graph 契约关闭**（source 与 attention 一起走既有 eager break），而不是只捕获一半。

## 2. 当前代码事实（基线 `9898f7ce9a`）

以下结论均已在本仓源码核对，行号为该提交：

| 事实 | 代码位置 | 影响 |
| --- | --- | --- |
| NPU prefill 只接受 `breakable` backend | `python/sglang/srt/model_executor/runner_backend/utils.py:146-159` | 不新增 backend 名称 |
| NPU segment 使用 NPUGraph，MoE 固定 AllGather 行数，构造期拒绝 ascend_ops AllToAllV | `hardware_backend/npu/graph_runner/breakable_npu_graph_backend.py:42-94` | 并行与 MoE 边界沿用 |
| prefill runner 已有 capture/replay metadata hooks | `model_executor/runner/prefill_cuda_graph_runner.py:1332-1416` | 不需要改 runner 主流程 |
| DSV4 captured-metadata 只对 Breakable 生效 | 同文件 `:700-711` | V4.1 可沿用同一开关 |
| V4 固定 metadata 容器只为 ratio 4/128 分配，kernel key 写死四个 V4 名字 | `hardware_backend/npu/attention/dsv4_prefill_graph.py:43-48,119-121` | **V4.1 不能直接复用**，当前会错路由 |
| NPU capability 未区分 V4/V4.1，low-ratio 也会返回 True | `hardware_backend/npu/attention/ascend_dsv4_backend.py:2645-2662` | 第一步必须加代际闸门 |
| V4.1 目前声明 `low_ratio_prefill_graph = False` | 同文件 `:1224` | source 在 BCG 下走 eager break |
| 捕获 metadata 为 True 时，attention 走不带 `bcg_` 包装的路径 | `models/deepseek_v4.py:2961-2984` | attention 能否留在图内取决于该开关与 backend 契约 |
| V4 已有逐层 topk staging | `ascend_dsv4_backend.py:1191-1196`、`dsv4_prefill_graph.py:191-201` | V4.1 同样需要逐层 topk 绑定 |
| V4.1 的 c2 使用 arch35 fused compressor + cycle request-bank table | `ascend_dsv4_backend.py:1281-1295`、`:284-290` | 不使用 V4 的 explicit state table |
| c2 pair state 每个 request slot 一个 physical bank | `hardware_backend/npu/dsv4/dsv4_memory_pool.py:546-568` | capture 用槽 0 即 bank 0 |
| `ReqToTokenPool` 槽 0 保留，真实分配从 1 开始 | `mem_cache/memory_pool.py:414-424` | capture 的请求槽隔离有现成机制 |
| Engram hash 是 eager break，lookup 也是 eager break | `models/deepseek_v4.py:4167-4171`、`layers/engram.py:1640-1645` | 二者都留在图外，写固定输出 |
| mHC A5 用单次 premix（含 previous-pre 折叠） | `models/deepseek_v4_mhc.py:546-580`、`:665-676` | 无多流，premix→post 依赖天然可捕获 |
| prefill runner 在 capture body 前清 extend 标记；ascend_ops 用 prefill graph scope 兜底 | `prefill_cuda_graph_runner.py:935-943`、`layers/moe/token_dispatcher/ascend_ops.py:175-190` | MoE prefill 路由沿用 |

### 2.1 当前代码对 V4.1 的真实行为

`_supports_dsv4_prefill_graph_capture()` 对 low-ratio 模型也会返回 True；随后：

1. `init_forward_metadata_for_breakable_cuda_graph_capture()` 调用 `Dsv4PrefillGraphMetadata.allocate()`。
2. 该容器的 `ratios` 过滤后为空，只绑定 V4 字段，并把 `kernel_metadata` 换回 `c1a/c4a/c128a/li_quant` 四个 V4 key。
3. V4.1 的 attention/source 需要 `win_metadata`、`c1a_metadata`、`c2a_metadata`、`c*_qli_*`、`dsv41_*` 等字段；这些字段要么缺失，要么仍是 capture 时 eager 分配的临时张量的地址，replay 不刷新。

因此**在实现 V4.1 契约之前，必须让 low-ratio 模型显式退出该容器**。这是 PR-0 的唯一目标，也是后续所有工作的安全基线。

## 3. 首版交付范围

| 维度 | 首版边界 | 不在首版 |
| --- | --- | --- |
| 硬件 | Ascend 950 / A5，`is_npu_arch35()` 为真；单机 | arch22 |
| 模型 | DeepSeek V4.1，`compress_ratios` 含 0/1/2；满足每 rank 64 query heads | V4 (0/4/128) 复用本契约 |
| 执行阶段 | target 普通 `EXTEND`、chunked prefill、有 prefix 的 prefill | decode/verify、draft extend |
| batch 轴 | 每次 replay 只允许 1 个请求（`MAX_PREFILL_GRAPH_REQS=1`） | B>1、多请求 bucket |
| 并行 | CP=1、DCP=1、PP=1、无 TBO；DP 每 rank 都有 token | DP idle rank 非 eager、CP/DCP 组合 |
| 通信 | ascend_ops prefill=allgather，固定行数布局 | AllToAllV、其他未验证 MoE backend |
| 投机 | DSpark target prefill；draft 侧保持 eager | draft prefill 入图、其他投机算法 |
| 图形式 | 显式 `--cuda-graph-backend-prefill=breakable` | Full graph、torch.compile |
| 输入 | 文本 prefill；无 LoRA、无多模态输入 | 多模态、LoRA、MIXED 批次 |
| 输出 | logits / 后续 decode / prompt logprob（走 eager tail） | 改变计算行数的 bounded replay |

## 4. 设计要点（展开见 design.md）

1. **代际独占契约**：V4 与 V4.1 各有一套固定字段和 kernel metadata key 集合；一个 backend 实例只能命中其中一套。
2. **原子捕获边界**：同一层的 compressor/index K/QLI/QSLI/candidate 与 MQSMLA 在同一 captured segment 内；禁止 source eager + attention captured 的半捕获。
3. **容量形状执行**：所有固定 buffer 按 bucket `T_cap` 分配。replay 的真实 token 数、prefix 和页表由刷新后的 `cu/seqused/live mask` 表达；死行长度 0、写回 loc 用 `-1`。这与 A5 单算子探针已经验证过的“固定 shape + live cu + 死行零长度”模式一致。
4. **metadata 与存储分离**：把 V4.1 的 dispatch metadata 计算重构成显式输入/输出的纯 helper；eager、capture seed、replay refresh 三处共用，避免 replay 依赖 live `ForwardMetadata` 的副作用。
5. **只原地刷新**：capture 后任何被图读到的 tensor 不得重绑；刷新函数必须校验 shape/numel/key 集合并报错，不能截断或静默换 buffer。
6. **逐层 topk 绑定**：每个低比例层的 indexer topk 输出在图内分配、图内消费；用 `topk_by_layer[layer_id]` 保留该层 capture 时的输出对象，避免共享属性只记住最后一层。
7. **槽 0 / bank 0 隔离**：capture 和 warmup 使用保留请求槽 0、ratio-2 的 bank 0、压缩页 0；真实请求从槽 1 开始，不新增 capture-only bank。
8. **失败必须显式**：不支持的配置在启动期拒绝或让整个契约关闭回 eager，不能在图内静默走另一条算法，也不能以“capture 成功”代替正确性。

## 5. 文档索引

| 文档 | 内容 |
| --- | --- |
| [design.md](design.md) | 完整设计：路由、固定 metadata 表、刷新顺序、数据流、隔离、外围组件、风险、与参考方案的分歧 |
| [implementation.md](implementation.md) | 代码触点、PR 拆分、CPU 测试、A5 组合探针、整模与性能验收 |

## 6. 已采用的外部事实与本文的独立判断

证据来源：[第三轮 A5 findings](../DeepSeekV4.1PrefillGraph设计方案/validation/a5-20261117-run3/findings.md)、[第四轮 A5 findings](../DeepSeekV4.1PrefillGraph设计方案/validation/a5-20261117-run4/findings.md)、[验证脚本](../DeepSeekV4.1PrefillGraph设计方案/scripts/verify_dsv41_prefill_graph.py)。以下“设备已验证”均指这些材料中的单卡、合成状态范围，不外推到整模或多卡。

| 主题 | 参考目录已有事实 | 本文处理 |
| --- | --- | --- |
| NPU graph 框架 | factory / NPUGraph segment / fixed AllGather 已迁入并通过 CPU 回归 | 直接采用，不重写 |
| 单算子可捕获性 | compressor、writer、QLI→QSLI、MQSMLA、mHC premix、Engram break 已有 A5 图回放证据 | 采用为“实现可行性证据”，但不等于组合图和整模通过 |
| 容量 + live cu 的语义 | 探针在 T=64 固定输入下用 live 31/16/61 等 replay，结果 exact | 采用为固定 buffer 协议的基础，并在真实 metadata 容器上重验 |
| 固定请求槽 | 文档指出槽 0 为 padding row | 补充源码依据（`free_slots=range(1,size+1)`），作为隔离前提 |
| source 留 eager 的过渡 | 参考方案允许用固定输出桥接 | **不采用为交付路径**；只保留“整个契约关闭”的回退 |
| V4.1 固定状态 | 参考方案计划扩展同一容器 | 采用，但要求代际分支、动态 kernel key、纯 helper 重构 |
| 验收 | 已列 CPU/A5/整模/性能四层 | 重整顺序：先组合图探针，再做 metadata 容器闭环，最后整模 |

## 7. 当前状态声明

- 本设计已在 `sglang` 分支 `dsv41-a5-graph-deepseek` 落地：代际闸门、V4.1 固定 metadata 容器、纯 metadata helper、逐层 topk 绑定、source/attention 成对路由和准入检查。
- 整体回滚开关是 `ascend_dsv4_backend.py` 的 `_DSV41_PREFILL_GRAPH_CONTRACT_READY`；置 False 时 source 与 attention 一起回到 eager break，不做半捕获。
- CPU 回归：新增 14 个 V4.1 metadata/准入/契约用例；`test_dsv4_prefill_graph_metadata.py`、`test_npu_dsv41_low_ratio_backend.py`、`test_npu_ascend_dsv4_backend.py` 共 182 项通过；NPU graph/attention + backend resolution + MoE/Engram 合并 248 项通过（1 skipped）。CPU 结果只证明协议与生命周期。
- A5 证据边界：单算子/单消费链已验证；compressor+QLI/QSLI+MQSMLA 的**同一 segment 组合图**、真实 allocator 下的 bank/slot 生命周期、整模数值与性能均未验收。下一步执行 `implementation.md` §3 的 P1/P2/P3。
- 参考目录中独立 CPU FP32 参考的 attention 差异仍作为单独议题保留，不能用图回放 exact 覆盖，也不能因此放宽默认阈值。
