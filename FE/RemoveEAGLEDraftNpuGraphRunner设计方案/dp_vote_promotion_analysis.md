# can_run_graph DP 投票上移公共基类：逻辑解释与跨设备风险分析

> 配套《[去除 EAGLEDraftNpuGraphRunner 设计方案](remove_eagle_draft_npu_graph_runner_design.md)》§4.3 的展开论证。
> 结论先行：**投票块内没有任何 NPU 专有依赖，搬移是纯位置变更；对 CUDA/XPU/MUSA 的影响是"触发组合下正确性严格变好、每轮多一次单元素 CPU all_reduce"，且决策只能从 graph 退向 eager、不可能反向引入挂死。**

---

## 1. 这段代码在防什么

问题链（每一环都设备无关）：

1. GLM-5.2 一类 DSA-MTP 模型的 draft 步复用上一轮 draft-extend 抓取的 indexer top-k（IndexShare seed，`eagle_worker_v2.py:273-293`）。该 seed 在 PD disagg 场景下**异步传输**，就绪与否是**请求局部**的：同一时刻，一个 rank 的 seed 可能已到、另一个 rank 还没到，空批 rank 甚至没有请求。
2. draft forward 内含 **TP/EP collectives**（MoE all-to-all 等）。这要求同一轮里**所有 rank 走同一条路径**（graph 或 eager）——一个 rank 进 graph、另一个走 eager，collective 参与者凑不齐，直接挂死。
3. PR #30839 在 worker 层加了**本 rank** 的降级（`eagle_worker_v2.py:617-623`：seed 缺失则本 rank 不走 graph）。但它管不了"别的 rank 怎么想"。PR #32210（commit `db9143ee08`）为此在 NPU 子类的 `can_run_graph` 尾部补了**跨 rank 投票**。

注意第 1、2 环没有任何 NPU 专属成分：异步传输时序、collective 失配挂死，在 CUDA 上同样成立。投票当初落在 NPU 子类里，只是因为它随一个 NPU 性能修复一起提交，不是因为问题本身 NPU 特有。

## 2. 代码逐段解释

搬移后的基类代码（`eagle_draft_cuda_graph_runner.py:368-394`）：

```python
if (
    not self.eagle_worker.seed_dsa_topk_from_draft_extend
    or self.attn_dp_size <= 1
):
    return is_bs_supported

# A PD-disagg IndexShare seed is transferred asynchronously, so seed
# availability is request-local: a real rank can miss it while idle
# ranks have no request at all. The draft forward contains TP/EP
# collectives, so all ranks must still choose the same graph/eager
# path or the mismatched collectives hang. Reduce the final decision
# across the model TP group, which contains all attention-DP ranks
# for this pipeline stage.
seed_ready = forward_batch.forward_mode.is_idle() or (
    spec_info is not None and spec_info.dsa_topk_indices is not None
)
decision = torch.tensor(
    int(is_bs_supported and seed_ready),
    dtype=torch.int32,
    device="cpu",
)
torch.distributed.all_reduce(
    decision,
    op=torch.distributed.ReduceOp.MIN,
    group=get_parallel().tp_group.cpu_group,
)
return bool(decision.item())
```

| 段 | 语义 |
| --- | --- |
| 门控（前 4 行） | 模型不带 IndexShare seed 机制，或没有开 attention DP，则维持原本地决策——**绝大多数模型/配置在这里原样返回，投票代码根本不执行** |
| `seed_ready` | 本 rank 的 seed 就绪位。`is_idle()` 豁免空批 rank（没有请求，视为就绪，否则全员永远投不出 True） |
| `is_bs_supported` | 门控之前算好的本地 graph 可行性（bs 档位、`require_mlp_sync` 合取等，基类原有逻辑，未改动） |
| `all_reduce(MIN)` | **一票否决**：组内任一 rank 投 0，全员得 0。等价于"最保守的 rank 替所有人做决定" |
| `cpu_group` | TP 组的 **gloo CPU 副组**（见 §3.3），刻意不走 device 通信后端 |

一句话总结：**把"我能不能走 graph"从本 rank 私有判断，升级为组内取最小值的全局一致判断**。

## 3. 为什么可以挪到公共代码：设备无关性审计

### 3.1 逐变量审计

| 变量 / 调用 | 来源 | 设备相关性 | rank 间一致性 |
| --- | --- | --- | --- |
| `self.eagle_worker.seed_dsa_topk_from_draft_extend` | worker `_init_dsa_index_share_state`（`eagle_worker_v2.py:291`），由 hf_config + topk 派生 | 无（纯 config） | **全 rank 一致** |
| `self.attn_dp_size` | `deployment_attn_dp_size()` 运行时上下文（`:120`） | 无（纯 config） | **全 rank 一致** |
| `forward_batch.forward_mode.is_idle()` | 调度器批状态 | 无 | rank 局部（正是投票要收敛的对象之一） |
| `spec_info.dsa_topk_indices` | draft-extend 抓取的 seed 张量 | 无（普通 tensor） | rank 局部（**投票的核心输入**） |
| `is_bs_supported` | 基类原有逻辑 | 无 | rank 局部（投票的另一输入） |
| `torch.tensor(..., device="cpu")` + `.item()` | 纯 CPU tensor 操作 | 无 | — |
| `torch.distributed.all_reduce(MIN)` | torch.distributed 通用 API；gloo 后端支持 MIN | 无 | — |
| `get_parallel().tp_group.cpu_group` | gloo 副组，见 §3.3 | 无 | — |

**没有任何一行触碰 NPU graph API、torch_npu、Ascend 符号。** 对照佐证：旧文件 `eagle_draft_npu_graph_runner.py` 的 NPU 性质全部来自其它方法（`_cache_loc_dtype`、`_get_update_attr_*`、`_replay_graph`），投票块在旧文件里（`:60-88`）就已经只用了上表这些设备无关设施。搬移前后该块的唯一差异是变量名（旧代码 `can_run_graph = super().can_run_graph(...)`，新代码内联为 `is_bs_supported`——喂给投票的值语义相同，基类判定逻辑本身未动）。

### 3.2 论点一（死锁安全）：门控与所有前置早退都是 rank 一致条件

`all_reduce` 的参与对称性是唯一可能"搬出新问题"的地方：若一个 rank 进了投票、另一个在门控前返回，后者永远等不到前者。逐条核查 `can_run_graph` 在投票之前的**每一个退出点**：

| 退出点 | 条件性质 | 结论 |
| --- | --- | --- |
| 宽度检查（`:336-341`，`num_tokens_per_req != captured_req_width`） | 轮次级状态（每轮所有请求统一 draft 宽度）+ config 派生的捕获宽度 | rank 一致 |
| 投票门控（`:368-371`） | 两个 config 派生量（§3.1 前两行） | rank 一致 |

即**全员同进同退**：要么所有 rank 都投，要么都不投。rank 局部的量（`is_bs_supported`、`seed_ready`）全部出现在投票**内部**，是待收敛的输入而非退出条件。另外，draft worker 只在最后一个 PP stage 上运行（`eagle_worker_v2.py:1258`），而投票用的组是该 stage 自己的 `tp_group`——组成员恰好就是全部投票参与者，不存在组内有人缺席。

### 3.3 论点二（通信原语）：`cpu_group` 是全平台无条件存在的 gloo 组

`parallel_state.py:414-428`：每个并行组创建 device 组的**同时**无条件创建一个 gloo 副组，注释原话 "a group with `gloo` backend, to allow direct coordination between processes through the CPU"；`:432` 的 `assert self.cpu_group is not None` 兜底。CUDA/XPU/MUSA 部署里这个组同样存在，且早已被**所有设备共用的** DP-attention 调度同步（`dp_attn.py:614`）日常使用。`ReduceOp.MIN` + int32 CPU tensor 是 gloo 的标准能力。选 CPU 组而非 device 组本身也是设备无关的设计（NCCL 一类后端不接受 CPU tensor）。

### 3.4 论点三（决策单调性）：只能否决，不能放行

投票结果 `= min(所有 rank 的本地可行性)`，语义上**只能把"本 rank 想走 graph"否决成"全员 eager"，永远不会把 eager 放行成 graph**。因此：

- 它**不可能引入**正确性错误或新的挂死（挂死恰恰来自路径不一致，投票只会把路径收敛得更一致）；
- 最坏代价是**性能**：本可走 graph 的轮次被拖去 eager。

风险分布高度不对称：漏掉投票 → 挂死（不可恢复）；多出投票 → 个别轮次变慢（可观测、可回退）。这决定了把它放到所有设备上是净收益方向。

## 4. 跨设备影响分析

### 4.1 触发面：哪些配置会真正执行投票

三个条件同时成立（缺一即在门控处原样返回，行为与搬移前**逐字节相同**）：

1. 模型 checkpoint 带 `index_share_for_mtp_iteration` + `index_topk`（GLM-5.2、Qwen4-Exp 一类 DSA-MTP）；
2. `--speculative-eagle-topk 1`；
3. `--attn-dp-size ≥ 2`。

对 CUDA 而言这不是假想配置：cookbook 里 GLM-5.2 的 **balanced** 配方（H200 FP8，`--attn-dp-size 8` + MTP 1-1-2，已验证）就满足全部三条。XPU/MUSA/ROCm 同理论证——它们跑不了这些模型时触发面为零，跑得了时收益与代价同 CUDA。

### 4.2 CUDA 上的行为变化（拆成正确性与性能两栏）

**正确性：严格变好。** 搬移前 CUDA 在触发组合下是"各 rank 本地决策"，两个具体的不一致场景现在被投票收敛：

- **PD seed 异步窗口**：rank A 的 seed 已到（走 graph）、rank B 未到（走 eager）→ collective 失配挂死。这正是 NPU 侧 PR #32210 修的问题，场景本身没有 NPU 专有成分。
- **bs 分量的 rank 局部性**：`require_mtp_tp_gather` 为 False 时 `cuda_graph_bs` 取本 rank 的 `batch_size`（`:346-347`）；attention DP 下各 rank 请求数天然不同，一个 rank 超 `max_bs` 而另一个没超 → 同样失配。投票把该分量也统一成组内最小。

**性能：触发组合下每轮 draft 多一次单元素 CPU all_reduce + 一次 `.item()` 阻塞。** 量级上是 µs 级的 gloo 单元素归约，对 ms 级的 draft 轮占比很小；但它是一个**新增的每轮 CPU 同步点**，在 overlap 调度的流水线里会收紧一点松弛度。非触发组合（所有非 IndexShare 模型、所有 `attn-dp-size 1` 部署）零开销——门控在最前面短路。

### 4.3 与 scheduler 层既有 min-reduce 的关系（不是重复建设）

DP-attention 调度同步已经有一个跨 rank 取 min 的机制：`spec_info.cuda_graph_compatible`（`eagle_worker_v2.py:1226-1228` 写入）经 `MLPSyncBatchInfo` 全组 min-reduce 后落到 `can_run_dp_draft_cuda_graph`（`dp_attn.py:83-99, :348`），并已包含在投票的输入 `is_bs_supported` 里。两者**时点不同**：

- scheduler 层在**批准备期**，读的是上一轮 draft-extend 写入的"计划兼容位"；
- 投票在 **draft 消费期**，读的是此刻 `dsa_topk_indices` 的实际存在性。

PD 异步传输恰好会在这两个时点之间落地——scheduler 层可能标了"计划兼容"而 seed 实际未到。投票是消费前的**最终裁决**，语义更紧；它依赖前者的结果（作为输入之一）而非取代它。

## 5. 验证与回退

- **NPU（已做）**：GLM-5.2 + `--attn-dp-size 2` + NEXTN 3-1-4，探针确认投票触发、两 DP 组 rank 成对参与、混合轮全员降级 eager、无卡死、输出正常。
- **单测（已做）**：`test_npu_cudagraph_backend.py` 4 项白盒测试保持绿（投票块本身在 `can_run_graph`，不在被测的 `replay` 路径上）。
- **CUDA 回归（待做）**：GLM-5.2 + attn-DP（cookbook balanced 配方）跑通 + accept-length 对比；无环境则以单测 mock `tp_group` 验证门控与投票路径。
- **回退方案（一行改动）**：若 CUDA 侧回归发现不可接受的开销，加 backend 属性（如 `needs_uniform_graph_decision`，NPU 为 True）门控投票块，其余设备行为退回搬移前。

## 6. 结论

| 维度 | 判定 |
| --- | --- |
| 代码本身 | 零 NPU 依赖，纯 torch.distributed + config 派生量，搬移是位置变更不是语义变更 |
| 参与对称性 | 门控与前置早退全部 rank 一致，全员同进同退，不引入新死锁 |
| 通信原语 | gloo `cpu_group` 全平台无条件创建，MIN/CPU tensor 为通用能力 |
| CUDA 影响 | 触发面窄（GLM-5.2 类 + topk=1 + attn-DP≥2）；正确性严格变好；代价为每轮一次 µs 级 CPU 同步 |
| 其他设备 | 同一套论证，跑不了触发模型则零影响 |
| 回退 | 单属性门控，一行 |
