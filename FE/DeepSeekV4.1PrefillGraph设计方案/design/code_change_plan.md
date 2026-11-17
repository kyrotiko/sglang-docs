# V4.1 Prefill Graph 本轮代码改动清单

基线：`sglang/dsv41-a5@9898f7ce9a`；2026-11-17。实现分支：`dsv41-a5-graph-gpt`，实现提交：`d3700bf0a9edad40dc30b3cc4e6ea9f810822f34`；独立 worktree：`D:/workspace/SGLang/sglang-dsv41-a5-graph-gpt`。本方案对应的代码已落地并通过 287 项 CPU 回归，尚未做本版 A5 整模验收。本文只跟踪该分支，源码链接指向该独立 worktree。[实现与 CPU 回归记录](../validation/gpt-implementation-20261117.md)。

本轮目标：让 V4.1 target 普通 EXTEND 的 source、indexer 和 attention 读取按 token bucket 固定的存储，并在 replay 前原地刷新本次请求的长度、位置、页表和 request-bank。复用已迁入的 breakable NPU graph，不重做 runner/backend，也不把“metadata 已分配”当成完整实现。

[第三轮](../validation/a5-20261117-run3/findings.md)和[第四轮](../validation/a5-20261117-run4/findings.md)已提供当前环境的基元及 writer 消费链证据，无需先重复相同探针。writer 在第四轮仍是图外准备，完整 source 链尚须在本轮接入后验证；CPU FP32 attention reference 的原 FAIL 继续保留。

## 1. 本轮范围

- target 普通 EXTEND，B=1，覆盖无 prefix、有 prefix 和连续 chunk；保留完整 decoder rows。
- A5，CP=DCP=PP=1，无 TBO；每 rank 64 query heads。先验收固定的 TP/EP、MoE AllGather 部署组合，复用现有并行限制。
- 使用已有 bucket、请求数/context admission 和 eager tail；idle rank replay 先取得设备证据，再决定协议与回退策略，不能把未验证直接等同于不支持。首轮 A5 反馈及当前验证入口见[记录第 4 节](../validation/gpt-implementation-20261117.md#4-首轮-a5-runtime-反馈与修复方向)。
- draft prefill、decode、target verify 继续使用各自已有协议；DSpark 作为 target prefill 输出及后续生成回归项。
- encoder/decoder SWA bounded replay 不在本轮入图范围，不捕获其按行裁剪或跳过 compressor 的另一套计算分支。

## 2. 文件与函数清单

本轮明确需要修改 **2 个生产文件**，测试扩展在现有文件中。公共 base hook 和 runner 的 warmup 清理改动降为条件项，触发条件见 2.3。`deepseek_v4.py` 已有所需路由，预计无需改计算代码；若接入发现额外调用点，先在本文补明原因和范围。

| 文件 | 拟修改位置 | 具体变化 |
| --- | --- | --- |
| [`dsv4_prefill_graph.py`](../../../../sglang-dsv41-a5-graph-gpt/python/sglang/srt/hardware_backend/npu/attention/dsv4_prefill_graph.py) | `Dsv4PrefillGraphMetadata.allocate/fill/refresh/_bind`、copy helpers | 增加 V4.1 固定存储分支，覆盖 c1/c2、window/index metadata、cycle-bank、SWA loc；保留 V4 的 c4/c128 分支 |
| [`ascend_dsv4_backend.py`](../../../../sglang-dsv41-a5-graph-gpt/python/sglang/srt/hardware_backend/npu/attention/ascend_dsv4_backend.py) | 构造期能力选择、`_supports_dsv4_prefill_graph_capture`、两个 BCG metadata hooks、low-ratio metadata helpers | 同时使能固定 metadata 与 source 入图；在 bucket 几何下重建并原地刷新 metadata，限定 capture 使用保留 bank 0 |

条件改动：只有证据表明必须在每次 warmup 与实际 capture 之间清理 NPU 状态时，才修改 [`base_attn_backend.py`](../../../../sglang-dsv41-a5-graph-gpt/python/sglang/srt/layers/attention/base_attn_backend.py) 和 [`prefill_cuda_graph_runner.py`](../../../../sglang-dsv41-a5-graph-gpt/python/sglang/srt/model_executor/runner/prefill_cuda_graph_runner.py)。目前不新增 `reset_prefill_graph_capture_state()` 公共 hook，也不调整 runner 清理回调。

相关现有调用点：[`deepseek_v4.py`](../../../../sglang-dsv41-a5-graph-gpt/python/sglang/srt/models/deepseek_v4.py) 的 `_low_ratio_sources_break()` 和 `MQALayer.forward` 分别读取 source 开关与 captured-metadata 开关；[`deepseek_v4_memory_pool.py`](../../../../sglang-dsv41-a5-graph-gpt/python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py) 已有 `clear_request_scoped_state(0)`，直接复用，不新增 pool 或分配额外 bank。

### 2.1 固定容器：`dsv4_prefill_graph.py`

沿用同一生命周期类，按后端实际 compression ratios 选择具体字段；V4.1 不进入 V4 explicit-state-table 分配，也不把 metadata 字典替换成 V4 四个固定 key。

| 固定字段 | 几何 / 本轮刷新规则 |
| --- | --- |
| query cu、KV lengths、`start_pos`、`seqused` | `B_cap=1`，cu 为 `[2]`；query cu 表达真实 query 数，不能等于 bucket padding 数；KV length 含 prefix |
| `actual_seq_lengths_q_pa_cpu`、必要的 CPU 长度镜像 | 按 bucket 生命周期维护，CPU/GPU cu 内容一致；不把 decode/verify 的共享镜像当成 prefill bucket 私有状态 |
| `dsv41_live_mask_buf` | `[T_cap]` bool；真实行 True，其余 False。普通 EXTEND 从 `extend_seq_lens_cpu` 得到真实行数，不能使用可能已经被 DP padding 扩大的 `extend_num_tokens` |
| `dsv41_win_sparse_indices`、`dsv41_win_topk_length` | `[T_cap,1,W]`、`[T_cap,1]`；padding 长度为 0，索引按现有 MQSMLA 协议填充 |
| `dsv41_cmp_topk_length_c1/c2` | `[T_cap,1]`；由本轮 positions、ratio 和 live mask 更新，padding 长度为 0 |
| full / c1 / c2 page table | `[1,P_cap]`；c1/c2 绑定同一个固定 full table，保留现有别名语义，尾列清 0 |
| SWA page table | 独立固定容量；从本轮 full→SWA 页映射刷新，尾列清 0 |
| c1/c2 index page table | 按 full page、ratio 和 `DSV41_INDEX_PAGE_SIZE` 推导列数；复用 `expand_index_page_table`，然后写回固定表。candidate block 8 不参与 PA page 几何 |
| c1 positions/loc | `[T_cap]`；保留 token 行序，padding positions 为 0、loc 为 `-1`，不 compact |
| c2 positions/loc | `[min(T_cap,T_cap//2+B_cap)]`；复用稳定 compact 规则，padding positions 为 0、loc 为 `-1` |
| `dsv4_cycle_state_block_table` | `[B_cap]`；capture 为保留槽 0，replay 原地切换到本轮真实 request slot |
| `swa_loc` | `[T_cap]`；真实行按本轮 full→SWA mapping 翻译，padding 沿现有保留 dummy page 协议。与 c1/c2 的 `-1` 规则分别处理 |
| `kernel_metadata` | 按当前 V4.1 planner 产生的 key 和每个 tensor 的实际 shape/dtype 分配；逐 key 原地更新，不能统一改成 V4 key 或强行扩成任意长度 |

具体实现约束：

1. 每个 bucket 分配一次。`_bind` 只在 capture 建立固定引用，replay 使用 `copy_` / 原地算子更新内容；CPU 生命周期测试同时检查 tensor identity 和 `data_ptr()`。
2. common 请求轴逻辑共用，V4/V4.1 几何使用短小的私有 helper 分开，不引入 owner/variant/registry。
3. V4.1 replay 的固定 key 集合、shape/dtype 不允许因 live T 改变。key 缺失、额外 key、shape 不兼容应报协议错误，不能静默沿用旧调度表；若 SDK ABI 改变，重新 capture/验证。
4. 所有 live 容量先检查再复制，不能用 `min(src,dst)` 截断。现有 `_copy_1d/_copy_cu` 的截断行为一起收紧并回归 V4；loc 的尾部填充值显式区分 V4 的 0 和 V4.1 的 `-1`。
5. context 页表首版按 `backend.max_context_len` 分配；请求超出图可用范围应在现有 admission 回退。容器内 overflow 检查仅用于发现协议错误，不充当正常 admission。

### 2.2 后端接入：`ascend_dsv4_backend.py`

**能力选择与模型路由配对。** 构造期只计算一次本轮固定能力：普通 target + breakable prefill + A5 + 本轮已允许的配置。V4.1 的两个开关从同一判断得到：

```python
use_captured_forward_metadata_for_breakable_cuda_graph = supports_prefill_capture
low_ratio_prefill_graph = has_low_ratio and supports_prefill_capture
```

该关系使 `_low_ratio_sources_break()` 选择普通 source 调用，同时 `MQALayer.forward` 选择图内 attention。实现和测试完成后一起开放；不提交“captured consumer 已开启，producer 仍留 eager 且无固定桥接”的半闭合状态。

能力和 admission 的处理位置：

- 保留现有 draft-worker 排除、CP/DCP 拒绝、B=1 上限，以及 runtime TBO 检查。
- V4.1 的 PP/TBO、NPU encoder SWA bounded replay 已在 `validate_deepseek_v41_features()` 拒绝，不再复制整套配置校验。
- V4.1 显式启用 decoder SWA bounded replay 与本轮 NPU prefill graph 同时出现时，在后端能力检查明确拒绝；读现有 resolved feature，不增加环境变量或 CLI 开关。
- B>1、没有可用 bucket、超过已配置图 context 上限、DP idle 等，复用 runner/scheduler 的同源 admission 与集体决策，不新增一套本地投票。
- AllToAllV 和 torch.compile 继续由现有 NPU breakable backend 拒绝。

**Capture hook。** `init_forward_metadata_for_breakable_cuda_graph_capture()` 使用现有 eager builder 获得 planner schema，随后分配/绑定固定 V4.1 存储。按 dummy batch 的 bucket positions、固定 cu/live mask 生成最终 window/cmp/kernel metadata。capture 的 cycle bank 固定为 0，c1/c2 写入 loc 使用 skip，SWA 写入限定在保留 dummy 区域。

**Replay hook。** `prepare_forward_metadata_for_breakable_cuda_graph_replay()` 开始使用已经传入、当前尚未使用的 `static_forward_batch`。顺序固定为：

1. 从 live batch 的 CPU extend/prefix 长度和 live eager metadata 取得本轮真实请求信息，校验 T/B/context 和字段容量。
2. 原地刷新固定 cu、KV lengths、`start_pos/seqused`、live mask、cycle-bank、页表、c1/c2 positions/loc 和 SWA loc；清除上一轮较长输入留下的尾部。
3. 将 backend 的 `forward_metadata` 设为该 bucket 的 captured object。
4. 用 `static_forward_batch.positions` 的 `[T_cap]` 形状和固定 live mask 运行现有 window/cmp helpers；用本轮固定 cu/KV lengths 规划 MQSMLA、QLI 和 QSLI metadata，逐 key 写回固定存储。
5. runner replay；图内 source 产生本轮 compressed/index KV、candidate 和 top-k，后续对应 attention 消费。

不能直接把 live `[T_live]` 的 window/cmp planner 输出换绑给 captured `[T_cap]` 字段。QLI 的 query cu、compressed KV length、ratio residual 也必须对应本轮 prefix/extend；QSLI 的调度长度要有 bucket 行数，并遵守有效 query 范围，不能把 padding positions 误当成有效候选行。

**Source/indexer 的既有自然依赖。** 保留 `_low_ratio_compress()` → index-K store → QLI/candidate → QSLI → attention 的计算路径，以及模型已有 KV source / index source 映射。`c{ratio}_topk_indices`、`dsv41_candidate_blocks` 在 capture 时建立 producer/consumer 的 tensor 依赖；图 replay 会执行产生它们的 kernel，无需重新执行 Python 属性赋值。

warmup 完成 RoPE table、QSLI cache 等既有懒初始化。跨 source group 的正确性通过真实生产者/消费者验证；不为它另建通用 staging registry，也不把 V4 的 `stage_topk` 逐层桥接照搬到已完整捕获的 V4.1 数据流。若仍有 producer 必须 eager，届时针对该边界补明确的固定输出桥接并重新审阅路由。

在该文件的相关改动处修正“prefill graph / `is_in_breakable_cuda_graph()` 仅 CUDA”的陈旧注释。对 V4.1 eager、decode 和 verify helpers 的改动限制在共享计算规则，不重写现有图初始化/刷新协议。

### 2.3 Capture bank 隔离；warmup 清理扩展为条件项

ratio-2 compressor 会读写 pending-pair state，warmup/capture 因此会改变 dummy bank。必须保证的协议是：B=1 capture 仅使用保留槽 0，真实 allocator 不分配该槽；replay 前把固定 cycle table 刷新为真实 request bank，真实 pending pair 跨 chunk 保留。清理 dummy state 不能代替这组地址隔离约束。

若需要为每个 bucket 提供已知初始 dummy state，可在 NPU 现有 `init_forward_metadata_for_breakable_cuda_graph_capture()` 中、图外调用已有 `clear_request_scoped_state(0)`。该位置在本 bucket 的 warmup 前，不需要修改公共 base 或 runner；下一 bucket 也可在同一入口初始化。调用须与之前设备写入按现有捕获流和同步边界有序。

原提案为每次 warmup 后和 capture 后增加 reset，是为了避免 dummy pending pair 在重复执行中累积；但目前尚未证明这个额外约束是整模正确性的必要条件。已有 `probe_compressor()` 在初始化后连续执行两次 warmup 再 capture，中间不 reset，并取得对应 capture/replay 对照证据；它不替代整模验证，但不支持把公共 hook 预先列成必改项。

因此本轮先保持 `base_attn_backend.py` 和 `prefill_cuda_graph_runner.py` 不变。整模接入时检查非零 request bank 在 capture 前后不变、跨 bucket capture 不受旧 dummy state 影响，以及实际 replay 切换 bank 后的结果/缓存正确性。只有出现可复现的 warmup 间状态依赖，且 bucket 入口初始化不足以解决时，才考虑增加 prefill 专用 reset hook。

届时 runner 通过已有 `post_warmup_hook` 在图外调度，并与 Mamba cleanup 组合。现有 BCG 继续屏蔽通用 `on_after_cuda_graph_warmup()`：CUDA DSV4 的该 hook 可能把 metadata 恢复成 decode 留下的旧对象，不能为 NPU 清理需求直接取消屏蔽。任何 reset 都不得在正常 replay 时清理真实请求的 pending-pair state。

## 3. 测试落点与可阻止的回归

本轮先在既有文件加 CPU 契约测试。使用真实 torch 几何和实际 metadata helper；DSL stub 只用于隔离依赖/记录实参，不以 mock capture 成功证明 NPU 可入图。

| 现有测试文件 | 新增/扩展的性质或生命周期检查 |
| --- | --- |
| `test/registered/unit/npu/graph_runner/test_dsv4_prefill_graph_metadata.py` | 同 bucket 大→小→大，固定地址和别名不变、尾部清空；c1 保序、c2 奇偶 prefix/chunk 边界；真实 bank 切换；metadata key/shape 与容量越界拒绝；保留 V4 sentinel/explicit table 回归 |
| `test/registered/unit/npu/attention/test_npu_dsv41_low_ratio_backend.py` | 实际 BCG hook 联合刷新 `[T_cap]` window/cmp/QLI/QSLI metadata；DP padded token 数与真实 extend 数不同；图内 source 与 captured metadata 的配对能力；若在 NPU capture 入口初始化 slot 0，检查不影响真实 pending state |
| `test/registered/unit/model_executor/test_prefill_cuda_graph_runner.py` | 本轮先运行已有回归；仅触发 2.3 的 runner 扩展时，新增 reset 时序、与 Mamba cleanup 共存、通用 warmup hook 仍被屏蔽及正常 replay 不清状态的测试 |
| `test/registered/unit/models/test_deepseek_v4_platform_dispatch.py` | 仅在现有测试未覆盖时，补模型 source/attention 两个选择器对 NPU enabled/disabled 的配对路由检查 |

生命周期测试必须能防住“加了字段却忘记 replay 更新”“较短请求读到旧尾部”“新 source 覆盖前一组”“capture 后忘记切换 bank”等静默错误。新增用例按仓库 unit-test admission 归类；声称真实 bug 回归的用例要验证修复前失败、修复后通过。

运行上先验证修改的 metadata、low-ratio 和 runner 文件，再各独立 Python 进程执行现有相关 12 文件回归，避免 stub 污染。AST/编译和 diff whitespace 一并检查。CPU 测试不能证明设备图执行或整模精度。

## 4. 实施顺序与交付检查

1. 先完成固定容器及 CPU 生命周期测试，保证 V4 和 V4.1 的字段/几何分支清晰。
2. 接入 capture/replay hooks、bank 0 隔离和实际 helper 联合测试；必要的 bucket 初始清理留在 NPU capture 入口，公共 runner 扩展按 2.3 的证据条件处理。
3. 同一可审阅改动中配对开放 V4.1 两个使能开关，完成 source/index/attention 路由检查和相关 CPU 回归。无需为过渡 eager 基线单独再跑一轮已完成探针。
4. 交付代码、变更说明和针对本轮新接入点的 A5 验证步骤，再做真实模型证据收集；CPU 通过不能宣称本轮配置已支持。

设备首先验证同 bucket 不同 prompt/长度、奇偶 prefix 与连续 chunk、真实 request-slot 更换/复用、两个 source group、capture 对非零 bank 的隔离，以及下一次 decode；核查 compressor、writer、indexer、attention 的实际图内 kernel 和多流顺序。之后验证 DSpark target→draft 的 KV/aux hidden 传递与完整生成，再按[验收计划](../validation/acceptance_plan.md)做准确率和性能 A/B。独立 CPU reference 的精度问题单独跟踪，不放宽原阈值来标记通过。

## 5. 保持复用的部分

本轮沿用 graph factory、NPU segment、breakable backend capture/replay 主体、MoE scope/AllGather、Engram hash/history eager break、mHC 当前计算、模型 eager tail 和现有 KV writer/layout。`ModelRunner`、`ScheduleBatch`、真实请求 allocator、speculative runner 及只读上游仓不在预计代码改动清单中。

不新增环境变量、CLI 开关、通用 `graph.update` 通道或 source-owner 框架。可选 B>1、DP idle 入图、CP/DCP、bounded replay 和 bucket 调优留到后续独立验收。
