**Ascend Ops prefill MoE 入图方案（`prefill=allgather`）**

基线：`sglang` `98141575f3`。
目标配置：`--moe-a2a-backend ascend_ops --moe-ascend-ops-config '{"prefill":"allgather","decode":"low_latency"}'`。
范围：prefill 阶段让 MoE 的 dispatch / 专家计算 / combine 进入 breakable NPU prefill graph，不再有 eager 断点；decode 路径不动。
验证环境：Ascend910_9362 × 16，空载 free 60.88 GiB，bf16，topk=8，hidden=7168，experts=256。

---

## 0. 结论摘要

| 项 | 结论 |
| --- | --- |
| 可行性 | 可行。集合通信、routing、finalize、整链在 `NPUGraph` 内全部捕获成功，replay 与 eager **数值完全一致（diff=0）** |
| 显存 | 单 bucket 峰值：1024→5.93 GiB，2048→11.87 GiB，4096→23.74 GiB（空载 free 60.88 GiB） |
| 性能 | 图内 replay ≈ eager（collective 略快 7%），收益来自去掉断点与主机同步，不来自算子 |
| 改动量 | **约 150 行**，4 个文件 + 1 个新小模块 + 单测；核心改动只有"跳过主机侧行数交换"和"显式 exec 模式" |
| 不需要改 | `_pre_permute_allgather`、`_allgather_weighted_sum`、`_combine_allgather`、gmm 路径——它们本来就是定容的 |

---

## 1. 架构师意见的落地

原话：**"padding 部分的处理，不能在这块计算的时候去做，要在初始化的时候就做好，补图环节做。"**

`prefill=allgather` 恰好让这句话退化成一件很简单的事：**padding 就是 bucket 本身**。

- prefill 图按 bucket 跑满 `T_cap` 行，尾部零填（`prefill_cuda_graph_runner.py:863-874`，NPU 上强制 `zero_padded_tail=True`）；
- 所有 rank 跑同一个 bucket 已由 runner 保证（`forward_batch_info.py:1737-1763` 强制 `MAX_LEN` 并把 `global_num_tokens` 对齐到同一 bucket；`prefill_graph_tolerates_sum_len()` 只对 megamoe 返回 True）；
- 因此 `_dispatch_allgather` 里"各 rank 行数可能不同"的假设在图内不成立，那段主机侧交换（`ascend_ops.py:419-427`）可以直接跳过。

于是：**图内没有任何 padding 计算，也没有任何主机同步**；`T_cap` 在 capture 时写进状态对象，replay 不再执行 Python。

前置条件（`ascend_ops.py:159-168` 已有校验，不满足直接抛错）：`tp_size == moe_ep_size`、专家连续放置、无融合共享专家。目标配置 TP16 = EP16 满足。

---

## 2. 目标与不变量

| 编号 | 不变量 | 检查方式 |
| --- | --- | --- |
| I1 | 图内 forward 无主机侧计算：无 `.item()` / `.cpu()` / `tolist()` | capture 期断言；profile 无 D2H |
| I2 | 图内张量地址、形状、dtype 在 capture 后不变 | 单测比对 `data_ptr()` |
| I3 | 所有形状只由 `T_cap` 决定 | 同 bucket 换不同真实 token 数 replay，结果一致 |
| I4 | eager prefill 行为完全不变 | 关图回归；非图路径走原有动态逻辑 |

---

## 3. 实测基线

`verify_ascend_ops_prefill_graph.py`（16 rank，纯 torch_npu 复刻 allgather 链）的结果：

| bucket | gathered | routed | expert_output(fp32) | 峰值 |
| --- | --- | --- | --- | --- |
| 512 | 0.11 | 0.88 | 1.75 | 2.97 GiB |
| 1024 | 0.22 | 1.75 | 3.50 | 5.93 GiB |
| 2048 | 0.44 | 3.50 | 7.00 | 11.87 GiB |
| 4096 | 0.88 | 7.00 | 14.00 | 23.74 GiB |

捕获与数值（bucket 1024 / 2048）：

| 项 | 1024 | 2048 |
| --- | --- | --- |
| `all_gather` + `reduce_scatter` | PASS, diff=0 | PASS, diff=0 |
| `init_routing` + `finalize` | PASS, rel=0，输出 `(131072, 7168)` = 16×1024×8（容量形状） | PASS, rel=0，输出 `(262144, 7168)` |
| 整链 e2e | PASS, diff=0 | SKIP（脚本预算限制，未跑） |
| collective 耗时 | eager 2.47 ms / replay 2.29 ms | 4.74 / 4.37 ms |
| routing 耗时 | 10.56 / 10.58 ms | 20.98 / 21.03 ms |

两点必须扣的折扣：

1. 显存是**空载**空闲内存；生产要对比 capture 时刻日志里的 `avail mem`。
2. 图池被所有 bucket 共享，但每个被捕获的 graph 持有自己的池内存，**总占用≈所捕获 bucket 峰值之和**。建议只捕获 1024 + 2048（≈18 GiB），4096 视余量再加。

---

## 4. 现状 gap（allgather 路径逐项）

| 位置 | 现状 | 图内问题 |
| --- | --- | --- |
| `ascend_ops.py:179` | `dispatch()` 读 `get_is_extend_in_batch()` | capture 时 runner 置 False（`prefill_cuda_graph_runner.py:936`），分支被固化 |
| `ascend_ops.py:419-427` | `_allgather_row_sizes`：`dist.all_gather` + 逐 rank `.item()` | D2H 同步，每层每步一次 |
| `ascend_ops.py:450` | `equal_rows` 主机分支 + `torch.cat` | 形状/分支依赖数据 |
| `ascend_ops.py:458` | `new_empty((world*max_rows, ...))` | 形状依赖 `max_rows`（数据） |
| `ascend_ops.py:521` | `output[: sizes[rank]]` | 依赖主机值 |
| `cuda_graph_hook.py:223-228` | BCG 白名单不含 `ascend_ops` | 直接关图 |

注意：**去掉 `_allgather_row_sizes` 本身对 eager 也是净收益**——它现在是每层每步一次主机同步，61 层就是 61 次。

---

## 5. 改动清单

### 5.1 总览

| # | 文件 | 改动 | 约行数 |
| --- | --- | --- | --- |
| 1 | `layers/moe/token_dispatcher/ascend_ops.py` | `dispatch()` 增加图模式；`_dispatch_allgather` 增加 `capacity` 参数 | ~18 |
| 2 | `hardware_backend/npu/moe/prefill_graph.py`（新） | 状态对象 + 安装/读取函数 | ~55 |
| 3 | `model_executor/forward_batch_info.py` | `ForwardBatch` 增加一个字段 | 3 |
| 4 | `model_executor/runner_utils/forward_batch.py` | 增加非抛异常访问器 | ~6 |
| 5 | `model_executor/runner/prefill_cuda_graph_runner.py` | `capture_prepare` 末尾安装状态 | ~3 |
| 6 | `test/registered/unit/npu/moe/test_ascend_ops_prefill_graph.py`（新） | CPU 单测 | ~80 |

> `arg_groups/cuda_graph_hook.py:223-228` 的白名单**本次不改**，见 §5.6。

### 5.2 `ascend_ops.py` —— 显式执行模式

当前（`:175-188`）：

```python
def dispatch(self, hidden_states, topk_output):
    topk_weights, topk_ids, _ = topk_output
    if get_is_extend_in_batch():
        if self.moe_ascend_ops_config.prefill == "allgather":
            return self._dispatch_allgather(hidden_states, topk_weights, topk_ids, self.tp_group)
        return self._dispatch_alltoallv(hidden_states, topk_weights, topk_ids, self.tp_group)
    return self._dispatch_low_latency(hidden_states, topk_weights, topk_ids)
```

目标：

```python
def dispatch(self, hidden_states, topk_output):
    topk_weights, topk_ids, _ = topk_output
    # 图状态存在即代表"这次 forward 正在被捕获"，与运行期 flag 无关。
    if get_moe_prefill_graph_state() is not None:
        return self._dispatch_allgather(
            hidden_states, topk_weights, topk_ids, self.tp_group,
            capacity=hidden_states.shape[0],
        )
    if get_is_extend_in_batch():
        ...  # 原样保留
```

`_dispatch_allgather`（`:429`）只加一个关键字参数：

```python
def _dispatch_allgather(self, hidden_states, topk_weights, topk_ids, group, *, capacity=None):
    ...
    if capacity is None:
        row_sizes = self._allgather_row_sizes(hidden_states.shape[0], group, hidden_states.device)
    else:
        # 图内每个 rank 行数相同；无主机交换，且 equal_rows 恒真。
        row_sizes = [capacity] * dist.get_world_size(group)
```

其余代码不用动：`equal_rows` 为真时 `gather_rows` 直接走 `dist.all_gather_into_tensor(gathered, padded)`（`:452-468`），形状 `[world*capacity, ...]` 静态。

### 5.3 新模块 `hardware_backend/npu/moe/prefill_graph.py`

```python
class MoePrefillGraphState(msgspec.Struct, frozen=True):
    """纯标记：外层 forward 正在被捕获。"""
    # 宽度不放在这里：用层自己的输入行数，capture 时各 rank 必然相同，
    # 而且它已经把 MoE 之前的 DP gather 算进去了。

def install_capture_state(forward_batch) -> None:
    """capture 前调用；非 ascend_ops / 非 allgather 时不动。"""
    if not get_moe_a2a_backend().is_ascend_ops():
        return
    if get_moe_ascend_ops_config().prefill != "allgather":
        return
    forward_batch.moe_prefill_graph_state = MoePrefillGraphState()

def get_moe_prefill_graph_state() -> Optional[MoePrefillGraphState]:
    batch = try_get_forward_batch()
    return None if batch is None else batch.moe_prefill_graph_state
```

要点：**这个变体不需要 mask、不需要 replay 刷新、不需要弱引用**——状态本身只是一个空标记（宽度取层自己的输入行数）。这是 allgather 相对 AllToAllV 最大的优势。

### 5.4 `ForwardBatch` 与访问器

- `forward_batch_info.py`：在 `full_prefill_cuda_graph_capture`（`:570`）旁加

  ```python
  # capture 期绑定的 MoE 执行模式；None 表示走层自身的运行期分发。
  moe_prefill_graph_state: Optional[Any] = None
  ```

- `runner_utils/forward_batch.py`：`get_forward_batch()`（`:32-38`）在 eager 服务路径下会抛异常（`set_forward_batch` 只在 `prefill_cuda_graph_runner.py:885, 1024` 使用），因此加一个非抛异常版本：

  ```python
  def try_get_forward_batch() -> "ForwardBatch | None":
      return _current_batch.get()
  ```

### 5.5 runner 安装点

`prefill_cuda_graph_runner.py:1790` 返回前一行：

```python
install_capture_state(forward_batch)
return forward_batch, self.model_runner.attn_backend
```

`_prefill_forward_context`（`:876-885`）已经发布了这个 batch，所以 `get_moe_prefill_graph_state()` 在 capture 的 forward 里读得到。**replay 不执行 Python，因此不需要在 `load_batch` 里做任何刷新**。

### 5.6 白名单：本次不改

`cuda_graph_hook.py:223-228` 的 `("none", "deepep", "megamoe", "flashinfer")` **保持不变**。

原因：白名单只参与"自动关闭"决策，而 `apply_cuda_graph_compatibility` 的第一行就是

```python
if (Phase.PREFILL, "backend") in server_args._cuda_graph_config_locked:
    return
```

只要显式传了 `--cuda-graph-backend-prefill breakable`，backend 这个 key 就被锁定，整条兼容性规则（包括 a2a 白名单和 DSV4 的显存压力规则）都会被跳过。所以手动开图完全够用，也不存在别的硬校验会拒绝 `ascend_ops`（`ALLOWED_BACKENDS_PER_PHASE` 只校验 backend 名称，不校验 a2a backend）。

保持不动的另一个好处：**不显式开图的人仍然自动落到 eager**，在 P3 性能结论出来之前这是更保守的默认。等验证完成再把这行加上，让 `ascend_ops` 变成默认可用。

### 5.7 不需要改的部分（附论证）

| 位置 | 为什么不用改 |
| --- | --- |
| `ascend.py:492-538` `_pre_permute_allgather` | `active_num=local_ids.numel()` 已经给定容语义；实测输出 `(16×T×K, H)` |
| `ascend.py:663-700` `_allgather_weighted_sum` | `torch.zeros((num_routes, H))` 形状静态；`unused_rows` 是 device 计算 |
| `ascend_ops.py:502-522` `_combine_allgather` | `sizes` 全相等时 `padded = contributions`、`output[:T]` 是恒等切片 |
| gmm / finalize 路径 | 形状由 `expert_tokens` 决定，device 侧；decode 图已验证可捕获 |
| `set_is_extend_in_batch(False)`（`prefill_cuda_graph_runner.py:936`） | 本方案不依赖该 flag；建议单独修（见 R4），不与本方案绑定 |

### 5.8 单测（CPU，无 NPU）

- `install_capture_state` 的三种输入（非 ascend_ops / 非 allgather / 正常）分别是否落状态；
- `get_moe_prefill_graph_state` 在没有 batch、有 batch 但字段为 None、有状态时三种返回；
- `_dispatch_allgather` 的 `capacity` 分支不调用 `_allgather_row_sizes`（用 mock 断言）；
- `row_sizes = [capacity]*world` 时 `equal_rows` 为真、走等长分支。

---

## 6. 数据契约

| 张量 | 形状 | 分配 | 说明 |
| --- | --- | --- | --- |
| gathered hidden | `[world*T_cap, H]` | 图池（段内） | `all_gather_into_tensor` 输出 |
| gathered topk ids / weights | `[world*T_cap, K]` | 图池 | int32 / fp32 |
| routed hidden | `[world*T_cap*K, H]` | 图池 | `init_routing_v2` 容量形 |
| expert_output | `[world*T_cap*K, H]` fp32 | 图池 | 显存大头 |
| finalize 输出 | `[world*T_cap, H]` | 图池 | reduce_scatter 输入 |
| combined | `[T_cap, H]` | 图池 | reduce_scatter 输出 |
| `moe_prefill_graph_state` | 无字段标记对象 | Python | 仅表示"正在捕获" |

---

## 7. 正确性论证

1. **等长前提**：见 §1 引用的 `MAX_LEN` 强制逻辑。dispatch 直接用层自己的输入行数当容量（而不是 runner 的 bucket），这样即使 MoE 之前存在 DP gather，宽度也自动是对的；前提只剩下"各 rank 的 bucket 一致"，由 runner 保证。若该前提被绕过，`all_gather_into_tensor` 的跨 rank 尺寸不一致会直接报错，不会静默算错。
2. **尾行**：图内尾部是零，路由结果只影响尾行自身，最终被图的输出切片丢弃；MoE 逐 token 独立，不污染有效行。
3. **空 batch**：`real_tokens == 0` 的 idle DP rank 仍按 bucket 参与通信（这正是 MAX_LEN 的目的），不会破坏集合通信的对称性。
4. **eager 回退**：任何不是图 replay 的 forward（超过最大 bucket、关闭图、非 prefill）都拿不到 state，走原有动态路径。

---

## 8. 显存与性能

- 单 bucket 峰值见 §3。建议捕获 1024 + 2048（≈18 GiB），且与 `--mem-fraction-static` 一起试。
- 图池按 bucket 累加，这是和之前 `aclnnCamMoeDispatchNormal` 显存问题同源的约束。
- 若显存吃紧，按顺序：降 bucket → `expert_output` 分段处理（需改 finalize 索引）→ fp32 降 bf16（需数值验证，`ascend.py:693-695` 注释说明 fp32 是为了 reduce_scatter 前的精度）。
- 性能预期：去掉了每层一次的主机同步（`_allgather_row_sizes`）和段间 Python 调用；实测 replay ≈ eager，所以**收益量级取决于 MoE 层数 × 同步开销**，需要用真机 A/B 确认，不要预设。

---

## 9. 分阶段与验收

| 阶段 | 内容 | 判据 |
| --- | --- | --- |
| P1 | 5.3~5.6 接线，1024 单 bucket 跑通 | capture 成功；prefill 输出与 eager 逐元素一致 |
| P2 | 加 2048，跑通 64K 上下文 | 数值一致；无 hang |
| P3 | 真机 A/B（BCG vs eager） | prefill 段耗时 ≥ eager；显存不超预算 |
| P4 | 文档、回归、回退开关；性能达标后再放开 `cuda_graph_hook.py` 白名单 | 关图路径行为不变 |

验收判据：

1. profiling 中 MoE 不再出现在段间 eager 调用；`[dsv4-prefill-graph]` 日志段数减少；
2. 同 batch 下 graph 与 eager 输出 `max |diff|` 在量化误差内（实测可直接要求 0）；
3. eager 路径逐位不变（关图回归）。

---

## 10. 风险与回退

| 编号 | 风险 | 缓解 / 回退 |
| --- | --- | --- |
| R1 | 显存：bucket 累加 | 只捕获 1–2 个 bucket；`--cuda-graph-backend-prefill disabled` 回退 |
| R2 | 等长前提被打破 | `assert` 兜底（§7.1） |
| R3 | 真实 gmm 在容量形状下的性能（脚本用占位算子） | P3 真机 A/B；必要时补 `--with-gmm` 再测 |
| R4 | `set_is_extend_in_batch(False)` 隐患（对 deepep AUTO 也成立） | 单独修，不与本方案绑定 |
| R5 | 量化路径（INT8 / MXFP8）未在脚本覆盖 | P1 用实际权重格式跑一遍 |

---

## 附录 A：备选方案——定容 AllToAllV

若 `prefill` 需要保留 `alltoallv`（例如 allgather 显存不可接受），则要把变长 split 换成定长 split：

- 每个 peer 固定 `C = ceil(T_cap*K/ep*slack)` 行；
- 发送侧用 `bincount/cumsum/gather/index_copy_` 把专家序的行搬到固定槽位；
- 接收侧定容 `[ep*C, H]`，gap 行填哨兵 `num_local_experts`，再走 `active_num=ep*C` 的 `init_routing`；
- 代价：约 25% 通信 padding、容量溢出需要显式处理（图内计数 + 读回断言）、改动量远大于 allgather。

现状中对应的阻塞点：`ascend_ops.py:543-567`（`.cpu().tolist()` ×2）、`:576`（`new_empty(sum(output_splits))`）、`moe_runner/ascend.py:447-453`（`repeat_interleave`）。
