# 迁移前 dsv41-a5 的 Prefill Graph 差距分析

本文是 `3cc67a7544` 的历史快照。入图提交已于 2026-11-16 迁入，后续开发基线为 `5633d68b04`；最新能力、提交映射和 V4.1 剩余增量见 [集成记录](prefill_graph_integration.md)。下文第 2/3 节的 NPU 框架和 MoE scope 缺口已由迁移覆盖，第 4 节保留迁移前模型状态的源码证据，行号继续固定在原提交。

分析基线：`sglang@3cc67a7544ce5819170235d364e8ecd14c8e9e73`；日期：2026-11-16。本文中的源码路径均相对 `sglang/`，行号针对该提交。仅采用当前跟踪的源码；`outputs/production/` 和 `__pycache__/` 中的旧副本不作为当前能力证据。

## 1. 迁移前结论

当前分支不能通过单纯开启 `--cuda-graph-backend-prefill=breakable` 完成 DeepSeek V4.1 的 NPU prefill 入图。至少需要补齐 NPU 分段执行框架、V4.1 捕获 metadata 协议和模型状态的数据流，以及 `ascend_ops` 的图内 prefill 路由。

已有 NPU decode graph 或某个算子的 CUDA graph 支持不能证明 prefill 可 replay。prefill 同 bucket 中的 live token、prefix 和请求布局变化，必须由本轮 metadata 表达；capture 时执行的 Python 选择不会在 replay 自动重算。

## 2. 图框架与入口：已确认的缺口

| 位置 | 当前代码事实 | 对本需求的影响 |
| --- | --- | --- |
| `model_executor/cuda_graph_config.py:119` | `default_prefill_backend()` 仅在 CUDA 默认启用 breakable | NPU 首版应显式 opt-in，默认关闭不能误报为“自动支持” |
| `model_executor/runner_backend/utils.py:82` | decode 有 NPU 分支，使用 `NPUCudaGraphBackend` | 这是 decode 的路由；不能推导 prefill 复用同一套协议 |
| `model_executor/runner_backend/utils.py:136` | prefill factory 仅特殊处理 XPU full，其余返回通用 CUDA breakable/full backend | 缺少 NPU prefill backend 解析、能力边界和不支持配置的拒绝 |
| `model_executor/runner_backend_utils/breakable_cuda_graph/breakable_cuda_graph.py:95` | portable capture-status 分支包含 HIP/MUSA/XPU，不含 NPU | NPU 会落到 `cuda-python` capture-status 查询，需按设备修正 |
| 同文件 `:434` | segment class 为 XPUGraph 或 CUDAGraph，未选择 NPUGraph | 不补此处就没有真正的 NPU segment capture |
| `model_executor/runner/prefill_cuda_graph_runner.py:126` | transformer-body backend 分类依赖具体 CUDA/XPU 类型 | 独立 NPU backend 需要参与相同协议，不能只改 factory |
| 同文件 `:700` | captured attention metadata 只对 `BreakableCudaGraphBackend` 启用 | backend 类型选择必须保证 V4.1 capture/replay hooks 真正被调用 |
| 同文件 `:1332`、`:1356` | 已存在 per-ShapeKey capture metadata 和 replay 原地刷新的 hook 调用 | 可以复用协议，无需重写 prefill runner |
| 同文件 `:1921` | BCG 与其他 backend 使用不同 post-warmup hook | 复用时核对 warmup 对 compressor/cache state 和当前 metadata 的影响 |
| 同文件 `:2086` | replay 构造新的 static `ForwardBatch` | V4.1 新增字段必须进入这份 view 或通过 backend-owned 固定状态传递 |
| 同文件 `:2161` | 已传递 ngram/Engram history、staged rows 和 encoder replay 相关字段 | 已有通用生命周期应复用；这些传递本身不等于内部张量可 capture |

上表路径的简写 `model_executor/` 均位于 `python/sglang/srt/`。

启动还有两层门槛需要一起核对：

- `model_executor/model_runner_components/cuda_graph_setup.py:691` 检查模型的 hybrid graph capability 和实际 layer handlers，否则要求所有层满足 Standard GQA；新增 backend 不代表一定能进入 capture。
- `arg_groups/cuda_graph_hook.py:136` 保留显式 prefill backend 选择；默认兼容性规则在 `:193`、`:223` 会限制 compressed-KV 和未验证 MoE backend。显式开关可以绕过默认禁用，但不能绕过真实算子约束。

当前已存在 `model_executor/runner_backend/base_breakable_graph_backend.py:23` 这一语义基类。通用分段行为可以基于它表达；是否保留 NPU 对 CUDA backend 的薄继承，应根据父类的 CUDA 专属行为决定，不能为了通过 `isinstance` 就忽略这些行为。

## 3. MoE：直接启用会选错路由

`python/sglang/srt/model_executor/runner/prefill_cuda_graph_runner.py:942` 在 prefill capture body 前调用 `set_is_extend_in_batch(False)`。

`python/sglang/srt/layers/moe/token_dispatcher/ascend_ops.py:176` 用 `get_is_extend_in_batch()` 决定是否走 prefill AllGather/AllToAllV；为 False 时在 `:185` 走 decode low-latency 路径。当前文件没有图 scope 的优先分支。

因此，框架接通后仍不能把这个标记当作图内 prefill 的唯一判据。应在 prefill graph scope 中明确选择合法的 prefill 算法，普通 eager 继续沿用 extend/decode 分流。

还要处理两类 host 数据依赖：

| 位置 | 行为 | 必要改造 |
| --- | --- | --- |
| `ascend_ops.py:417` | AllGather 交换行数后对每个结果调用 `item()` | 捕获采用固定行数协议，不在图内 host 读计数 |
| `ascend_ops.py:426` | 优先复用 `get_forward().ascend_allgather_row_sizes`，不满足时重新交换 | 图中必须明确几何来源，不能在某 rank 自行改走另一 collective |
| `ascend_ops.py:440` | 根据 Python row_sizes 选择 equal/unequal 布局和切片 | 固定 shape 的 capture/replay 不能假装这些 live Python 分支每轮重算 |
| `ascend_ops.py:508` | combine 根据相同 row_sizes 组织 ReduceScatter 和输出切片 | dispatch/combine 必须共享一致、固定且正确的布局 |
| `ascend_ops.py:569` | AllToAllV 的 split sizes 通过 `.cpu().tolist()` 得到 | 数据相关动态通信不能直接按固定 AllGather 方案支持；首版明确拒绝或统一 eager |

`python/sglang/srt/models/deepseek_common/moe_transport.py:294` 仅在普通 extend scope 中准备上述 row layout；其 `:314` 又按 CP/DP/attention-TP 情形推导行数。图中的 MoE 输入形状必须以实际 gather/shard 后的张量为准，不能无条件等于原始 attention token bucket。

`python/sglang/srt/model_executor/forward_batch_info.py:318` 的 SUM_LEN 容忍判断目前只为 MegaMoE 的特定情形返回 True；DP gather 是否被录图由 `layers/dp_attention.py:970` 记录。新实现需要维护通信几何和多 rank admission，不能只在本地 attention backend 加一条允许 replay 的分支。

## 4. V4.1 模型与 Ascend 路径

### 4.1 已有模型能力与真正的入口问题

`python/sglang/srt/configs/deepseek_v41.py:38` 将 V4.1 架构归一到 `DeepseekV4ForCausalLM`；V4.1 在模型层按配置分流，并非需要从零增加模型类。

NPU 的 `dsv4` attention 路由位于 `python/sglang/srt/layers/attention/attention_registry.py:209`，选择 `DeepseekV4AscendAttnBackend`。当前 backend 的 V4.1 ratio 0/1/2 路径已存在；旧资料中“Ascend 仅支持 0/4/128”不适用于这个基线。

以下现有入口仍以 CUDA breakable scope 为条件：

| 位置 | 行为 | 对 NPU 接入的要求 |
| --- | --- | --- |
| `models/deepseek_v4.py:931` | `_low_ratio_sources_break()` 决定 source 是否在 eager break 运行 | 明确 NPU 的 capability 和边界，不能假定旧装饰器会自动生效 |
| `hardware_backend/npu/attention/ascend_dsv4_backend.py:1194` | NPU low-ratio 路径明确未建模 CUDA 的 CP/prefill-graph 分支，`low_ratio_prefill_graph=False` | 需要独立接入 V4.1 graph 协议，不能只把这个属性改为 True |
| `models/deepseek_v4.py:2963` | 只有 CUDA breakable EXTEND 走 attention eager wrapper；其他情形直接调用 backend | NPU 初始分段和最终 attention 入图都需要明确路由 |
| `models/deepseek_v4.py:4160` | Engram hash eager break 也以 CUDA breakable scope 为条件 | NPU 需要接通每轮 hash/history 更新边界 |

本节表格路径均相对 `python/sglang/srt/`。

模型没有声明 `supports_hybrid_prefill_graph`，但这不是已确认的启动阻塞：每层的 `self_attn.attn_mqa` 是 `RadixAttention`，`model_executor/model_runner_components/layer_setup.py:43` 的收集器识别它，能够满足 Standard GQA 的层数检查。不过 `MQALayer.forward` 直接调用 backend（`:2979`），绕过 `RadixAttention.forward`，所以“收集成功”不代表通用 attention wrapper 已为它创建断点。

当前 NPU V4.1 对每 rank 的 query heads 有 64 的硬检查（`ascend_dsv4_backend.py:2584`）；不能照搬 attention-TP 头分片部署。首个拓扑需要满足这一约束，同时满足所选 MoE backend 的 TP/EP 条件。

### 4.2 Prefill metadata 尚未接入

Ascend backend 尚未实现 breakable prefill 专用 capture/replay hooks。`layers/attention/base_attn_backend.py:251` 默认关闭 captured metadata，`:342`、`:349` 的默认 hook 为未实现；因此仅打开 capability 标记不能完成适配。CUDA V4 backend 的 `layers/attention/deepseek_v4_backend.py:3324`、`:3427` 可参考接口契约，不可整段复用设备实现。

当前 NPU graph metadata 针对另一种几何：

- `ascend_dsv4_backend.py:3073` 对普通 decode 使用每请求 1 token，对 verify/draft 使用固定 token 数，再用 `arange` 建 query offsets。
- 同文件 `:3846`、`:3887` 的行刷新处理 decode/target-verify；不能自动覆盖每请求 extend 长度可变的 prefill。
- 同文件 `:2142` 的 ratio-2 metadata 按 `min(n, n // 2 + B)` 分配压缩行，因此固定 token 数、改变请求数也会改变输出形状。prefill 必须同时固定或区分 `T_cap` 与 `B_cap`。

需要覆盖的状态至少包括 query offsets/KV lengths、`start_pos`/`seqused`、request-bank、SWA/压缩 loc、full/压缩/index page tables、live mask、window/压缩 top-k lengths 及 kernel metadata。每轮刷新必须检查 shape/address 和尾部清理；不能用默认 decode offsets 或只替换一个 metadata 对象。

### 4.3 c1/c2、source、candidate 的所有权

| 状态 | 当前源码事实 | 图协议要求 |
| --- | --- | --- |
| ratio 1 | `ascend_dsv4_backend.py:1255` 逐 token 投影，无 gate/pairing；`:2134` 保留 token 行序，只 mask 无效行 | 不可套用 ratio 2 的 compact 行序；无效 loc 保持 skip sentinel |
| ratio 2 | `:1259` 要求 arch35 和独立 wkv/wgate；`:1275` 使用 `seqused/start_pos/cu` 和请求级 cycle state-bank | 保留跨 chunk 的 pending pair；capture/dummy 不能污染真实请求 state |
| 压缩与 index 存储 | `models/deepseek_v4.py:1493` 按 source 配置创建模块；`mem_cache/deepseek_v4_memory_pool.py:2012` 将消费层映射到最近同 ratio 的 KV source | 以 source 为存储所有者，不能每层独立分配或错误复用 source 组 |
| top-k / candidate | `ascend_dsv4_backend.py:1323` 发布本轮 top-k；`:1590` 检查 QSLI candidate 的存在和行数 | 形状相同不证明属于本轮 forward；靠明确的 producer/consumer 顺序和固定 storage 保证 |
| QSLI K cache | `:1421` 按 `(ratio, source slot)` 惰性分配，`:1555` 写入对应 index record | warmup 覆盖实际 source 和 optional 路径，capture 后不再首次分配 |
| 压缩行选择 | `:2100` 共享 eager/replay 行公式；`:3250` 有 `_stable_compact_1d` | 可参考稳定 compact 实现，避免动态 NonZero；ratio 1 仍不能 compact |

本节表格路径均相对 `python/sglang/srt/`。

source 的 Python 属性赋值不是天然错误：producer 与 consumer 同在捕获数据流内时，固定输出可以每轮更新。风险发生在 eager source 每次生成新 tensor，而 captured consumer 仍绑定旧地址，或者某 source 未执行却沿用同 shape 的旧 candidate。过渡阶段若让 source eager，相关 consumer 应一起留在同一显式边界，或通过固定 output buffer 桥接；最终完整 attention/indexer 入图仍需闭合这条链。

### 4.4 Engram、mHC 和 DSpark

**Engram 有已确认的 batch 动态依赖。** `layers/engram.py:435` 的 extend 分支按请求长度做 `repeat_interleave`，`output_size` 来自 CPU extend 长度之和；`:499` 按本轮 request slots 提交 history。相同 padded T 下，live token 数和请求划分仍能改变这段执行。首版建议将 hash/history 提交置于明确的 NPU eager break，写入固定 hash buffer，再验证 lookup、projection 和 gate 入图；不能通过关闭 Engram 代替模型正确性。

Engram verify forward 与接受后的 history 提交是不同职责（`layers/engram.py:589`），新 prefill 路径不能重复提交或提前提交 verify history。

**mHC 已有 NPU 实现，需保护现有状态依赖。** `models/deepseek_v4_mhc.py:546` 的 NPU combine 使用上一子层的 pre 系数，并生成本子层 stats；`:665` 的 post 消费相应系数；`:231` 的 `with_residual()` 在 residual 改变后丢弃旧 input/stats、保留 pre。图边界、多流及 Engram residual 更新必须保留这个关系，不能仅看 tensor shape 相同就复用旧 stats。V4.1 的该模型层循环还要求 PP=1，首版不扩展 PP。

**DSpark 的交付不止 logits。** `models/deepseek_v4.py:4355` 在指定层 attention 输入处采集 residual 的 mHC mean；`layers/aux_hidden_states.py:29` 组织 packed aux hidden。`speculative/dspark_components/dspark_worker_v2.py:815`、`:839` 在 target prefill 前规划 placement、完成后向 draft KV 注入 hidden。图输出需要保留本轮正确 aux tensor、有效行和 token indices，不能只验证首 token。

### 4.5 首版范围与设备未知项

首版建议 CP=1、DCP=1、PP=1、无 TBO，保持完整 decoder rows；实际 attention-TP/DP 配置需满足每 rank 64 heads 的现有约束。NPU CP 的 metadata 拼接仍有 V4 假设（`ascend_dsv4_backend.py:2883`、`:3984`），CUDA decoder tail graph 的创建也有 `is_cuda()` 条件（`models/deepseek_v4_replay_tail.py:156`），不应在本轮顺带放开。

上述结论表明 Python 编排和状态协议存在缺口，但不能证明 `compressor`、QLI/QSLI、MQSMLA、MX quant、KV epilog 或 mHC 算子不能入图。实际 A5 包的 device-length 读取、metadata 输出 shape、workspace、skip sentinel 和多轮 replay 正确性仍需设备探针。


## 5. 历史工作能复用到哪里

下列历史提交可在同一个主仓通过 `git show` 查看；它们不在当前基线中提供完整能力。迁移前按当前接口重新对齐，不整体合并旧工作分支。

| 历史提交 | 可参考内容 | 当前必须重核的边界 |
| --- | --- | --- |
| `f71ce61842` | `BreakableNPUGraphBackend`、NPUGraph segment class、portable capture-status、prefill backend 解析 | 当前 runner 的语义基类、warmup、pool、输出和各类具体类型判断 |
| `708b953bb9` | 固定 metadata、bucket 内原地刷新、loc/state 隔离、完整 V4 attention 数据流 | 提交明确限定 A3、单请求、无 CP/DCP/attention-TP 分片；ratio 4/128 不等于 V4.1 的 1/2 |
| `5c88c18cca`、`53c83ddd34`、`85378bf71e` | prefill graph 中的固定容量 AllGather、通过 graph scope 分流 | 当前 transport、量化、DP geometry 和 padding 的语义 |
| `4cafabd1a3` | 在 capture 前拒绝尚无方案的 AllToAllV | 当前配置/启动组件已调整，拒绝点需要放到实际能执行的路径 |

旧文档 `learn/sglang-deepseek-v4.1.md` 的基线是 `4cafabd1a3`。架构背景可参考，旧分支的 Ascend 支持范围不应套用到当前 `dsv41-a5`。

## 6. 尚不能从静态代码得出的结论

- 部署环境的 DSL/custom op 是否在固定 tensor shape 下重读所有动态 metadata，还是冻结了部分 host tiling/标量。
- padded token、零有效输出和 dummy request/state-bank 是否被每个算子合法接受且不会写真实缓存。
- Engram 的目标放置和 lookup 通信是否能被当前 A5 环境 capture。
- 图池固定占用、实际 workspace、启动耗时、图命中率及最终 prefill 收益。

这些问题分别进入 [实现方案](../design/implementation_plan.md) 和 [验收计划](../validation/acceptance_plan.md)。本轮不将源码存在、CPU mock 测试或旧 A3 实验记为 A5 整模通过。
