# 基于已迁入框架的 V4.1 Prefill 入图方案

开发基线：`sglang/dsv41-a5@9898f7ce9a`；2026-11-17。已保留先前 11 个入图提交并合入 randgun 的 13 个新提交，281 项 CPU 回归全部通过；本次变化和定向设备复验见 [合并记录](../analysis/randgun_merge_20261117.md)。writer/mHC 已迁到公开 SDK wrapper，但 V4.1 prefill 固定状态和下面的使能闸门仍未实现，新主仓尚未做 A5 整模验证。

前两轮 A5 [run1](../validation/a5-20261116-run1/findings.md) / [run2](../validation/a5-20261116-run2/findings.md)已验证几何、metadata、prologue、Engram 和 QLI/QSLI/MQSMLA 对照。[第三轮](../validation/a5-20261117-run3/findings.md)验证当前 950DT/SDK 的 compressor、公开 writer、mHC premix；[第四轮](../validation/a5-20261117-run4/findings.md)验证 writer→attention 的 15 个 graph/eager case exact，dtype/storage 相容且诊断延续原精度差异。单算子复验已可收敛到实现阶段，继续固定 metadata/模型数据流接入；独立 FP32 精度、真实模型状态和多卡/整模仍单独验收。

本轮拟改文件、函数、capture/replay 顺序及测试落点见[代码改动清单](code_change_plan.md)。该清单为待审阅方案，主仓尚未实施；本轮直接闭合固定 metadata 与 source/index/attention，第 2 节的 eager 过渡仅在完整路径暂不能闭合时采用，不作为又一轮重复探针的前置。

## 1. 复用边界与首版范围

直接复用现有 `PrefillCudaGraphRunner`、NPU breakable backend、NPUGraph segment、metadata hooks、MoE prefill scope、固定 AllGather 和 eager tail。保留已有 device-neutral graph 基类与生命周期。

首版固定 B=1、CP=1、DCP=1、PP=1、无 TBO、完整 decoder rows，满足每 rank 64 query heads 及所选 MoE 的 TP/EP 约束。选择固定 A5 部署拓扑，暂不扩展多请求和 DP idle 非 eager 模式。

新增状态优先扩展同文件中的 `Dsv4PrefillGraphMetadata`：共享 allocate/fill/refresh/bind 生命周期，按 V4/V4.1 分配具体字段。只有实现中确实出现独立职责，才在同文件拆出小 helper 或 sibling class。不要求生命周期对象改为 `msgspec.Struct`；新增纯数据结构按仓库约定处理，不增加 dataclass、通用 owner、variant 或 registry。

复用 V4.1 decode 图已有的字段布局、`_dsv41_ratio_row_values`、`_stable_compact_1d`、`expand_index_page_table` 和 window/cmp 长度计算。decode allocator 使用 `T=B*tokens_per_req`，不能直接作为任意 extend 的 allocator；prefill 的容量和存储生命周期保持独立。

## 2. 先隔离 V4-only 协议，再验证现有 break 闭环

当前 `_supports_dsv4_prefill_graph_capture()` 对 V4.1 也可能返回 True，但 `Dsv4PrefillGraphMetadata` 只分配 ratio 4/128，且会用 V4 metadata keys 替换原字典。首先为 low-ratio target 隔离这条协议。

首个候选实现：当 `_dsv4_has_low_ratio` 时返回 False，使 `use_captured_forward_metadata_for_breakable_cuda_graph=False`。这个 False 表示不采用当前 V4-only captured metadata，不等于关闭整张 prefill graph。配合既有 `low_ratio_prefill_graph=False`，source 和 attention 都沿现有 eager break 执行，其他可捕获 body 仍使用现有图。

两条选择必须配对核验：

- `_low_ratio_sources_break()` 控制 low-ratio sources 的 eager break。
- `MQALayer.forward` 根据 `use_captured_forward_metadata_for_breakable_cuda_graph` 选择普通 attention 调用或 `bcg_deepseek_v4_attention_with_output`。

只修改一个布尔值不能证明完整模型已支持。先核对 replay 的 live metadata、输入/输出行数、SWA 写入、source/cache loc 和 eager tail，再在 A5 验证同 bucket 不同 live T、prefix、连续 chunk 和后续 decode。若暂不能形成合法闭环，对相应配置明确拒绝启用，不能继续进入 V4-only 容器或宣称支持。

这是过渡基线，完成条件仅为“现有 break 路径多轮数值和缓存状态正确”；完整交付仍要求 compressor/indexer/attention 的真实图内证据。

## 3. 固定 metadata 与数据流一起交付

### 3.1 容量和 admission

| 容量 | 来源 | 可执行约束 |
| --- | --- | --- |
| `T_cap` | 现有 token bucket | live 行数不超 bucket，padding 行独立处理 |
| `B_cap` | 现有 `MAX_PREFILL_GRAPH_REQS=1` | 调度和执行侧均限制请求数 |
| `P_cap` | 首版沿 `backend.max_context_len` 分配页表；需要更小上限时沿现有 `prefill.max_seq_len` | 有效 context 上限必须落到 `can_replay_locally` 和调度投票，不能大于实际页表容量 |
| `C_cap(1/2)` | ratio 1 保留 `T_cap` 行；ratio 2 按 `min(T_cap,T_cap//2+B_cap)` 规划 | 核对 A5 输出 ABI、cu 尾部与有效压缩数 |
| candidate/top-k | 当前模型配置和 source/group | 固定每组必要 shape，检查候选上限及 consumer 归属 |

`P_cap` 首版使用已有模型/池容量，不为此新建配置项，也不直接设置当前 NPU backend 尚不支持的 `max_context_size` capability。若采用更小 context 容量，要在首次 replay 前落实相应上限；不能等 buffer copy 溢出才报错。模型本身的 context 拒绝和图容量导致的 eager 回退分别验证。

### 3.2 最小固定状态清单

| 字段/存储 | 固定几何与刷新规则 |
| --- | --- |
| query offsets、KV lengths、`start_pos/seqused` | 请求轴按 `B_cap`；cu 按 `B_cap+1`；每轮从 live 请求布局刷新，空槽重复合法 cu 尾值 |
| `dsv41_live_mask_buf` | `[T_cap]`；按真实 token/request 布局生成，padding 为 False，不能沿用 decode/verify mask 假设 |
| `dsv41_win_sparse_indices` / `dsv41_win_topk_length` | `[T_cap,1,W]` / `[T_cap,1]`；无效窗口长度为 0，索引使用算子接受的 skip 值 |
| `dsv41_cmp_topk_length_c1/c2` | `[T_cap,1]`；按位置、ratio 和 live mask 更新，尾部为 0 |
| `dsv4_cycle_state_block_table` | `[B_cap]`；每轮绑定本次 request-bank，不能遗留 capture 的 bank 0 |
| full/c1/c2 page table | 共享同一份固定 full block table 的别名关系，或明确原地复制；不能仍指向 eager 每轮新建的 table |
| c1/c2 index page table | 复用已有 index-page expansion，刷新固定表并清除旧列 |
| c1 positions/loc | 保持 token 行序，不 compact；loc 无效行使用现有 `-1` skip 约定 |
| c2 positions/loc | 按请求和位置稳定 compact；容量由 bucket 几何决定，尾部 loc 为 `-1` |
| window/c1/c2/indexer kernel metadata | 按当前 DSL schema 建键；同 bucket 动态长度变化时 shape 必须匹配并原地更新，不能替换回 V4 四键 |
| candidate / c1/c2 top-k | 按 source/group 明确 producer/consumer；只有被 captured consumer 绑定的存储需要对应稳定地址 |
| compressed/index-K/QSLI cache | 沿已有池和 source 映射；metadata 页表、写入 loc 和本轮请求一致 |

V4 仍沿用自己的 padding/loc 语义，不能将其 0 sentinel 直接套到 V4.1。上述 V4.1 `-1`、零有效长度和容量公式是当前 Python 协议；部署 A5 算子是否正确接受仍需探针。

capture 使用 bucket 几何分配固定存储；replay 可以用 eager planner 生成 live 结果后原地复制，也可以用显式容量参数规划。禁止根据 live T/B 重分配被图绑定的 tensor。window/cmp 长度 helper 应读取 bucket-padded 输入和显式 live mask，或将 live 结果复制到对应固定 buffer，不能仅复用其遇到 shape 不同就分配的默认分支。

### 3.3 Producer/consumer 闭环与退出条件

固定 metadata、source/index/attention 数据流及其使能选择构成同一交付单元，不能把“metadata 分配完成”当成可 replay 的完成条件。

A5 探针允许时，优先将 source/index/attention 作为完整捕获数据流一并闭合。若某 producer 必须暂留 eager，再用具体的固定输出 buffer 桥接到 captured consumer。桥接是备选，不强制先实现一套过渡桥接、随后再删除重做。

过渡 eager 边界的退出条件：每个 captured consumer 读取的 top-k、candidate、长度和 cache 映射都由本轮对应 producer 更新；相关 storage 地址稳定，内容在同 shape 不同输入、跨 source/group、多轮 replay 中保持正确。全部算子进入同一捕获数据流时，保留图内自然的张量依赖，不为 Python 属性赋值额外建立一层通用 owner。

## 4. Capture 状态隔离和既有模型逻辑

### 4.1 优先复用保留的请求槽 0

已核实 `ReqToTokenPool` 为槽 0 留 padding row，真实请求从 1 开始分配；现有单请求 capture 通过 `arange(B)` 使用槽 0；ratio-2 pair pool 按 request-bank 分配。这为首版隔离提供了已有机制，“没有 dummy_state_loc”本身不证明需要新建 bank。

首版先核实并复用 bank 0：

1. 明确 B=1 capture/warmup 使用 `req_pool_indices=[0]`，cycle table 对应 bank 0；检查 pair pool 覆盖该 bank 且真实 allocator 永不分配槽 0。
2. 在 warmup/capture 的约定边界清理或恢复 bank 0 的 pending-pair 状态及相关 padding history，避免跨 bucket 干扰；不清理真实请求 bank。
3. replay 前将 cycle table 原地刷新为 live request slots，复核 compressed/SWA loc 的 skip 和 dummy 行规则。
4. 对真实 bank 做 capture 前后状态比较，并验证 slot 复用和下一 chunk。若实际 A5 路径不能按上述规则隔离，再设计 capture-only bank 或快照恢复，不能先增加一套分配机制。

多请求 capture 的 `arange(B)` 会涉及非零请求槽，B>1 不能直接沿用本方案，后续扩展必须重新验证隔离。

### 4.2 Engram、mHC、DSpark

复用现有 Engram hash eager break、`new_hash_ids` 固定输出、history 提交和 runner 的 live history/staged rows 传递。`is_in_breakable_cuda_graph()` 是设备共享 scope，函数名不代表仅 CUDA；NPU backend 中“never true here”的注释需在相关实现变更时修正。

不新增 NPU 专用 hash 边界。核验当前 lookup 模式、padding、不同 prompt/history、请求槽复用及提交次数；只有当前模式确有未覆盖的动态部分时才增加具体边界。hash/history 保持 eager 不阻塞首版正确性或完整 attention 入图。

mHC 保持上一子层 pre、本子层 stats 和 residual 的生产消费关系；Engram 改写 residual 后不得复用旧 stats。基础 prefill 先不启用投机建立对照；启用 DSpark 时，packed aux hidden、有效行和 token indices 必须可被正确注入 draft KV，单独验收。

## 5. 实施顺序与验收边界

| 阶段 | 工作 | 完成条件 |
| --- | --- | --- |
| A：闸门与现有 break 基线 | 隔离 V4-only metadata；配对已有 source/attention 路由，复用 Engram break | B=1 多轮 replay、prefix/chunk 和后续 decode 对齐；未支持配置明确拒绝，不宣称完整 attention 已入图 |
| B：决定实现方向的 A5 探针 | ratio-2 容量/state、QLI/QSLI/candidate、MQSMLA、metadata、sentinel 和 bank 0 隔离 | 固定 shape 改输入/长度/loc 后输出及状态正确；确认是否有 host tiling 依赖 |
| C：V4.1 固定状态与计算闭环 | 一份具体 metadata 扩展，与 source/index/attention 数据流和使能条件一起实现 | 地址/shape、live mask、尾部、source/group、bank 和完整 epilog 均正确；不需要时省去过渡桥接 |
| D：首批整模正确性及收益 | 保留真实 Engram/mHC；验证 DSpark 适用配置、既有 MoE、eager tail、admission | 多轮数值、缓存、后续生成、图内覆盖和首批配置性能证据完整 |
| E：后续范围 | B>1、DP idle 非 eager、CP/DCP/TBO/AllToAllV、其他图模式和 bucket/内存优化 | 每种新增组合独立验证，不成为首批配置交付前置 |

沿用已迁入 MoE scope 和固定 AllGather；首版固定实际 DP idle 策略并记录，不要求先跑完所有开关组合。性能 A/B 在首批配置正确性通过后执行，作为该配置最终验收；bucket 调优、workspace 优化和广泛性能矩阵可后置。

`graph.update` 仅在 A5 ABI 探针证明 device metadata 不足以表达动态值时设计，届时明确具体算子的 payload、workspace 和流依赖。验证条目与执行优先级见 [验收计划](../validation/acceptance_plan.md)。
