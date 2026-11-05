**Ascend Ops prefill MoE 入图方案：定容 AllToAllV + 初始化期 padding**

基线：`sglang` `98141575f3`（DSV4 prefill graph cleanup 之后）。
范围：`--moe-a2a-backend ascend_ops` 的 **prefill** 阶段，让 MoE 的 dispatch / 专家计算 / combine 进入 breakable NPU prefill graph，不再有 eager 断点。decode 路径不在此方案内。
方法：源码静态分析。第 6 节的显存/性能数字需要 P1/P3 实测；本地只有 CPU 环境。

---

## 0. 对"padding 在初始化时做好、在补图环节做"的理解

原话：**"主要就是 padding 部分的处理，不能在这块计算的时候去做，要在初始化的时候就做好，补图环节做。"**

拆开是三条约束：

1. **"padding 部分的处理"** —— 指 prefill alltoallv 为了让通信成立而临时算出来的那些东西：`_alltoall_counts` 的 split（`ascend_ops.py:526-567`，含两次 `.cpu().tolist()`）、`_alltoall_single` 按 `sum(output_splits)` 现分配（`:576-578`）、`recv_counts` 驱动的 `repeat_interleave` 重排（`moe_runner/ascend.py:447-453`）。
2. **"不能在计算的时候做"** —— 这些物料不能出现在被捕获的那段图里。图内 Python 只执行一次，任何 D2H 同步或依赖真实 token 数的形状计算，在 replay 时要么失效、要么直接非法。
3. **"初始化时做好 / 补图环节做"** —— 容量、split、buffer 的形状在初始化阶段定死；与具体 batch 无关的常量在 capture 时固化进图；图内剩下的"动态"只有**设备侧的数据**（每 peer 实到多少行），没有主机参与、没有形状变化。

注意：这条要求**不是**"把 padding 挪到 replay 前用 Python 补"——那仍然是主机逻辑。真正的形态是：buffer 永远是容量大小，padding 体现为 buffer 里的无效行，而"哪些行有效"由图内的 device 计数表达。

---

## 1. 目标与不变量

| 编号 | 不变量 | 检查方式 |
| --- | --- | --- |
| I1 | 图内 forward 无主机侧计算：无 `.item()` / `.cpu()` / `tolist()`，无依赖张量形状的 Python 分支 | capture 期断言；profile 看有无 D2H |
| I2 | 图内所有张量的地址、形状、dtype 在 capture 后不变 | 单测比对 `data_ptr()` / `shape` |
| I3 | 通信 split 是常量，只由 bucket `T_cap` 决定 | 同一 bucket 用不同真实 token 数 replay，split 不变 |
| I4 | eager prefill 行为完全不变（alltoallv / allgather 保持现状） | 关图回归 |

---

## 2. 现状与 gap

| 环节 | 现状 | 为什么不能入图 |
| --- | --- | --- |
| 路径选择 | `ascend_ops.py:179` 读 `get_is_extend_in_batch()` | capture 时被 runner 置 False（`prefill_cuda_graph_runner.py:936`），分支会被固化 |
| split 计算 | `ascend_ops.py:543-566` `histc` → `all_gather_into_tensor` → `.cpu().tolist()` ×2 | D2H 同步 |
| 接收缓冲 | `ascend_ops.py:576-578` `new_empty(sum(output_splits))` | 形状依赖通信结果 |
| 二次重排 | `moe_runner/ascend.py:447-453` `repeat_interleave(local_ids, local_counts)` | 输出长度依赖路由结果 |
| 配置闸门 | `cuda_graph_hook.py:223-228` 白名单不含 `ascend_ops` | 直接关图 |

**为什么不能照搬 decode 的 MC2**：MC2 是 low-latency（小 batch、fullmesh）算法，容量按 `global_bs = T_cap × ep_size` 开，4096 token/rank 时通信 buffer 与算法都不匹配；它是 decode 的形态，不是 prefill 的形态。本方案的 prefill 继续走 AllToAllV。

**关于 allgather**：目标配置是 `--moe-ascend-ops-config '{"prefill":"allgather","decode":"low_latency"}'`。allgather 在形状上是**天然定容**的（每个 rank 恒为 `T_cap` 行，`_allgather_row_sizes` 的 `.item()` 在图里根本不需要），改造量远小于 AllToAllV，没有 split、没有容量溢出、没有 slack，见 §3.7。它唯一的门槛是显存：`_allgather_weighted_sum`（`ascend.py:682-686`）的 fp32 `[world × T_cap × K, H]`，16 rank × 4096 × 8 × 7168 × 4B ≈ 15 GB。**这个数字是方案选型的分水岭**，先用 `verify_ascend_ops_prefill_graph.py --mode account` 在机器上量。

---

## 3. 方案：定容 AllToAllV

### 3.1 核心变换

变长 split 的 all-to-all 换成**定长 split** 的 all-to-all：

```
今天：  每个 rank 发给 peer p 的行数 = 路由结果的函数  →  split 要主机算
目标：  每个 rank 固定发给 peer p 恰好 C 行           →  split = [C] * ep，常量
```

其中 `C` 在初始化时确定：

```
C = ceil(T_cap * K / ep * slack)        # slack 建议 1.25，见 §5.1
```

代价是每 peer 多传 `C - 实际行数` 行 padding（约 10~25%），换来的是：**split 变成常量，所有 buffer 形状静态，图内零主机**。

### 3.2 发送侧：专家序 → 固定槽位

AllToAllV 的前提是"发给同一个 peer 的行连续"。当前实现依赖 `npu_moe_init_routing` 产出的**全局专家序**（专家在 rank 上连续放置，所以同一 peer 的专家区间也是连续的），再按每 peer 的计数切段。

定容化后只需要在每个 peer 的段尾补空隙：

```
routed  = npu_moe_init_routing(hidden_states, topk_ids, num_experts)   # 全局专家序，形状 [T_cap*K, H]
peer    = expert_id // num_local_experts                               # 每个 route 的目标 rank
counts  = bincount(peer, minlength=ep)                                 # device
start   = cumsum(counts) - counts                                      # device
within  = arange(T_cap*K) - start[peer]                                # device
slot    = peer * C + within                                            # device
send    = zeros((ep*C, H)); send.index_copy_(0, slot, routed.hs)       # device
```

全部是 device 侧元素级运算，形状恒定。`topk_ids` / scale 的搬运用同一套 `slot`。

### 3.3 接收侧：定容 buffer + 哨兵行

接收缓冲固定为 `[ep*C, H]`，其中每个 sender 块内的有效行数由 `recv_counts` 表达（device：`all_gather_into_tensor(local_counts)` 已经在做，`ascend_ops.py:557`，形状 `[ep, num_experts]` 静态）。

从"每 sender 块内按专家排"到"按本地专家聚合"需要一次重排，当前是 `repeat_interleave`（变长）。定容版本：

- 构造 `[ep*C, 1]` 的 `local_ids`：有效行填本地专家 id，**gap 行填哨兵 `num_local_experts`**（`ascend.py:499-508` 已经在用这个哨兵，语义是"丢弃但保持计数正确"）；
- `npu_moe_init_routing(received, local_ids, num_local_experts, active_num=ep*C)` —— `active_num` 给定后输出是容量形状；
- `expert_tokens` 是 device 张量，直接喂 gmm（`group_list_type=1`，与 decode 一致）。

这样接收侧的每一步都是 `[ep*C, ...]` 的定容形状。

### 3.4 Combine：同一套槽位反向

返回侧用同一组固定 split `[C]*ep` 发回；发送方再用 `expanded_row_idx`（`npu_moe_finalize_routing`，`ascend_ops.py:644-648`）还原成 `[T_cap, H]`。`topk_weights` 的加权在 finalize 里完成，形状静态。

### 3.5 三条时间线的分工

| 阶段 | 做什么 | 产物 |
| --- | --- | --- |
| 初始化 | 算 `C`、`global` 常量、`[C]*ep` split 列表；预取 comm name；固化 exec_mode | 常量 |
| 补图（capture） | 从图池分配 send/recv `[ep*C, H]`、`local_ids [ep*C,1]`、`slot/counts/start` 等 index buffer；跑一次带固定 split 的 a2a 完成捕获 | 定容 buffer + 段 |
| replay | 什么都不做 | — |

与 `dsv4_prefill_graph.py` 的关系：那里因为 attention metadata 的内容每步都变，需要 `refresh()`；这里**连刷新都不需要**——所有动态内容都由图内 kernel 从 `topk_ids` 现算。这是本方案比"定容 + 主机刷新"更彻底的地方，也正好对应"补图环节做"。

### 3.6 路径选择

`AscendOpsDispatcher.dispatch()` 不再读运行时 flag，改成读当前 batch 上的图状态：

```python
state = get_forward_batch().moe_prefill_graph_state   # None = eager
if state is None:
    return self._dispatch_eager(...)                   # 现有 alltoallv/allgather 原样
return self._dispatch_alltoallv_capacity(state, hidden_states, topk_output)
```

`get_forward_batch()` 由 prefill runner 在 `_prefill_forward_context` 里发布（`runner_utils/forward_batch.py:41`），与 DSV4 attention 读 `dsv4_prefill_graph_state` 是同一模式。

### 3.7 目标配置：`prefill=allgather` 变体

配置：`--moe-a2a-backend ascend_ops --moe-ascend-ops-config '{"prefill":"allgather","decode":"low_latency"}'`。

allgather 比 AllToAllV 简单一个数量级，因为**它根本没有 split**：每个 rank 把所有 `T_cap` 行广播给所有 peer，形状只由 bucket 决定。

需要改的地方（对照现状）：

| 位置 | 现状 | 图内形态 |
| --- | --- | --- |
| `_allgather_row_sizes`（`ascend_ops.py:419-427`） | `dist.all_gather` + 逐 rank `.item()` | **删掉**：每个 rank 恒为 `T_cap` 行 |
| `gather_rows`（`:452-468`） | `equal_rows` 主机分支 + `torch.cat` | 只保留等长分支，`all_gather_into_tensor` 静态 |
| `_pre_permute_allgather`（`ascend.py:492-538`） | `active_num=local_ids.numel()` | 已经是定容，device 侧 mask，无需改动 |
| `_allgather_weighted_sum`（`ascend.py:663-700`） | `torch.zeros((num_routes, H), fp32)` | 形状静态，但**这是显存大头** |
| `_combine_allgather`（`ascend_ops.py:502-522`） | `output[: sizes[rank]]` | `[:T_cap]` 恒等切片 |

前置条件（`ascend_ops.py:159-168` 已有校验，不满足直接抛错）：`tp_size == moe_ep_size`、专家连续放置、无融合共享专家。

"所有 rank 跑同一个 bucket"这条前提**已经由 runner 保证**：prefill CUDA graph 下 `ForwardBatch` 强制 `MAX_LEN` 并把 `global_num_tokens` 对齐到同一个 bucket（`forward_batch_info.py:1737-1763`，注释里明确写了"BCG replay requires a common padded shape for MoE collectives"），`prefill_graph_tolerates_sum_len()` 只对 megamoe 返回 True。allgather 需要的等长条件因此天然成立。

**优点**：没有 split 常量、没有容量溢出、没有 slack、replay 阶段零刷新——比 §3.2~3.5 的 AllToAllV 方案干净得多，是**首选**。

**唯一门槛**：显存。`_allgather_weighted_sum` 的 fp32 `[world*T_cap*K, H]` 随 `world × T_cap` 线性增长。若在目标 bucket 上放不下，有三条路：

1. 降 bucket（最直接）；
2. 把 `expert_output` 拆成 `world` 段、每段 `[T_cap*K, H]` 分批处理（循环次数静态，仍可入图），代价是 finalize 的 `expanded_row_idx` 也要按段切分，改造量不小；
3. 把 fp32 中间结果降到 bf16（省一半，但 `ascend.py:693-695` 注释说明 fp32 是为了 reduce_scatter 前不丢精度，需要数值验证）；
4. 退回 §3 的定容 AllToAllV。

先用 `verify_ascend_ops_prefill_graph.py --mode account` 把这个数字测出来，再决定走哪条。

---

## 4. 数据契约

| 张量 | 形状 | 分配 | 写入方 | 说明 |
| --- | --- | --- | --- | --- |
| routed.hidden_states | `[T_cap*K, H]` | 图池 | 图内 routing | 全局专家序 |
| counts / start / slot | `[ep]`, `[ep]`, `[T_cap*K]` | capture | 图内 | device 索引 |
| send / recv | `[ep*C, H]` | capture | 图内 | 定容通信缓冲 |
| local_ids | `[ep*C, 1]` | capture | 图内 | gap 填哨兵 |
| expert_tokens | `[num_local_experts]` | 图内 | routing op | gmm group_list |
| combined | `[T_cap, H]` | 图内 | combine | — |
| split 列表 `[C]*ep` | 常量 | init | — | Python list，capture 固化 |

---

## 5. 正确性论证

### 5.1 容量溢出

定容 split 的唯一风险是"某个 peer 实际收到的行数 > C"。路由分布由 top-k 加负载均衡偏置（`e_score_correction_bias`）约束，按二项分布估计：`N = T_cap*K` 个 route、`1/ep` 的落点概率，`C = N/ep * 1.25` 对应约 15σ 以上，实际溢出概率可忽略但**不是零**。

设计上必须显式处理，不能静默：

1. 图内维护一个 device 计数 `overflow = max(counts - C, 0).sum()`；
2. **默认**：调试/验证阶段每步读回（图外一次 D2H），超过阈值即 assert；
3. **生产**：周期性采样读回，或提供 `slack` 参数按实测分布调整；
4. **保守兜底**：`slack` 可配到 `ep`（即 `C = T_cap*K`，无溢出），只用于对正确性零容忍的场景，显存换正确性。

### 5.2 尾行与 gap 行

- 图本来就按 bucket 跑满 `T_cap` 行（`prefill_cuda_graph_runner.py:863-874` 在 NPU 上强制 `zero_padded_tail=True`），尾行是零；它们的路由结果只影响自己那一行，随后被图的输出切片丢弃。
- gap 行填哨兵专家 id，`npu_moe_init_routing` 会给出 0 权重，`_allgather_weighted_sum`（`ascend.py:688-692`）里"未使用容量行清零"的处理方式可以照搬到 alltoallv。

### 5.3 split 的静态性

`dist.all_to_all_single(output, input, output_split_sizes=[C]*ep, input_split_sizes=[C]*ep)` 的 split 是 Python 常量列表，capture 时固化，replay 不变——这是整个方案能成立的关键，也是相对现状唯一的语义变化（变长 → 定长）。

### 5.4 量化路径

`npu_moe_init_routing` 的 `quant_mode` 与 scale 搬运沿用现有分支（`ascend_ops.py:595-607`）；scale 使用同一套 `slot`，不引入额外形状。

---

## 6. 显存与性能预算

**通信量**：`ep*C` 行/层，相对现状 `T_cap*K` 行多出约 `slack` 系数。以 T_cap=4096、K=8、ep=16、H=7168、bf16 计：

- 现状：32768 行 ≈ 470 MB/层；
- 定容（slack=1.25）：40960 行 ≈ 587 MB/层；
- send + recv 两份，峰值约 1.2 GB，**长期钉在图池里**（跨层按活跃区间复用，峰值约一层）。

这与之前 `aclnnCamMoeDispatchNormal … Failed to execute tiling function` 同源：MoE 通信 workspace 跟着 bucket 放大。

**bucket 上限反推**：

```
T_cap ≤ (图池可用显存 - 其他捕获段占用) / (2 * K * slack * H * sizeof(act))
```

建议 P3 先在 2048 上量，再决定是否上 4096。

---

## 7. 分阶段落地

| 阶段 | 内容 | 判据 |
| --- | --- | --- |
| P0 | 纯函数化：`C`、slot、local_ids 的构造抽成不依赖 NPU 的函数 + CPU 单测（参照 `test_dsv4_prefill_graph_metadata.py`） | 单测覆盖：非均匀分布、全部落一个 peer、空 batch |
| P1 | 512 bucket 打通 capture/replay；验证 HCCL `all_to_all_single` 在图内可用 | capture 成功；输出与 eager 逐元素一致；溢出计数为 0 |
| P2 | 接线 runner（`moe_prefill_graph_state`）+ 放开白名单 + 去掉 flag 依赖 | 64K 上下文 prefill 结果正确 |
| P3 | 2048/4096 的性能与显存 | 段耗时 ≥ eager-alltoallv；显存不超预算 |
| P4 | 文档、回归、回退开关 | 关图路径行为不变 |

---

## 8. 风险与回退

| 编号 | 风险 | 缓解 / 回退 |
| --- | --- | --- |
| R1 | 容量溢出导致静默错算 | §5.1 的计数 + 断言；`slack` 可配 |
| R2 | HCCL `all_to_all_single` 在 NPU 图内不可捕获 | P1 第一件事就验；不可用则退到"分段 + 单段捕获"或评估 fused op |
| R3 | 设备侧索引（bincount/cumsum/index_copy_）的图内开销 | P1 量；必要时合并成单个自定义 kernel |
| R4 | 图池显存 | §6 公式限制 T_cap；保留 `--cuda-graph-backend-prefill disabled` |
| R5 | `set_is_extend_in_batch(False)`（`prefill_cuda_graph_runner.py:936`）本身是隐患 | 本方案不依赖该 flag；建议单独修，不要和本方案绑在一起 |
| R6 | 实现量与 fused op 重叠（MC2 / `ascend_fuseep` 的 `fused_deep_moe` 都自带定容语义） | 先做一个 spike 对比两者的 capture 可行性与性能，再决定是否自建 |

---

## 9. 验收判据

1. **入图证据**：profiling 里 MoE 的通信/计算不出现在段间 eager 调用，`[dsv4-prefill-graph]` 日志的段数相应减少；
2. **数值**：同 batch 下 graph 与 eager 的输出 `max |diff|` 在量化误差内；
3. **容量**：长时间压测下溢出计数恒为 0；
4. **性能**：同 batch、同 prefix 下 prefill 段耗时不低于 eager-alltoallv；
5. **显存**：capture 前后 `avail mem` 差值在预算内。
