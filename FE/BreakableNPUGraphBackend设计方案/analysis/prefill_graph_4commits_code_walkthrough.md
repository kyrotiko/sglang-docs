# 4 个 commit 代码修改说明（main 基线）

**分支**：`prefill_graph_ascend_ops_on_main`（`origin/main` = `704681229f` + 4 个 commit）。
**用途**：逐文件、逐 hunk 说明这 4 个 commit 改了什么、每段代码是什么意思、为什么必须这么改。
**注意**：行号会漂移，以函数名/内容为准；本机没有可用的 Python/NPU，本文是静态阅读结论，未在本机跑测试。

| commit | 标题 | 文件数 |
| --- | --- | --- |
| `12667c59dc` | `[Feat] [NPU] prefill: breakable NPU graph backend (P1 wiring)` | 5 |
| `e834d6fa61` | `[Feat] [NPU] DSV4 prefill graph: capture attention and compressor in the segment` | 4 |
| `4c3a237e38` | `[Feat] [NPU] ascend_ops: capture prefill MoE in the breakable graph` | 6 |
| `3d44122dd3` | `[Fix] [NPU] prefill graph: keep SUM_LEN when a DP rank is idle` | 1 |

15 个文件（`forward_batch_info.py` 被 commit 3/4 各改一次）：

```
python/sglang/srt/hardware_backend/npu/graph_runner/breakable_npu_graph_backend.py   (新)
python/sglang/srt/model_executor/runner_backend/utils.py
python/sglang/srt/model_executor/runner_backend_utils/breakable_cuda_graph/breakable_cuda_graph.py
test/registered/unit/model_executor/runner_backend/test_backend_resolution.py
test/registered/unit/npu/graph_runner/test_breakable_npu_graph_backend.py            (新)

python/sglang/srt/hardware_backend/npu/attention/dsv4_prefill_graph.py               (新)
python/sglang/srt/hardware_backend/npu/attention/ascend_dsv4_backend.py
python/sglang/srt/models/deepseek_v4.py
test/registered/unit/npu/graph_runner/test_dsv4_prefill_graph_metadata.py            (新)

python/sglang/srt/hardware_backend/npu/moe/prefill_graph.py                          (新)
python/sglang/srt/layers/moe/token_dispatcher/ascend_ops.py
python/sglang/srt/model_executor/forward_batch_info.py
python/sglang/srt/model_executor/runner/prefill_cuda_graph_runner.py
python/sglang/srt/model_executor/runner_utils/forward_batch.py
test/registered/unit/moe/test_ascend_ops.py

python/sglang/srt/model_executor/forward_batch_info.py   (第二次)
```

---

## 0. 先理解这 4 个 commit 在做什么

Ascend NPU 的 prefill 图沿用 CUDA 的 **breakable CUDA graph（BCG）** 框架：

- capture 时，普通 Python 代码执行一次，它 launch 的 device kernel 被录进 segment；
- `@eager_on_graph` 装饰的函数**不会执行**，只记录成 replay 时的回调（eager break）；
- replay 时只重放 segment + 回调。**Python 里写死的 tensor 地址、Python 标量、`if` 分支不会重新计算。**

由此得到这 4 个 commit 的分工：

| commit | 解决的问题 |
| --- | --- |
| P1 wiring | 让 BCG 在 NPU 上能跑：每个 segment 用 `torch.npu.NPUGraph`，NPU 上只接受 `breakable` 后端 |
| DSV4 | attention / compressor / indexer 原来在 eager break 里，现在留在 segment 内；方法是给它们**固定容量、原地刷新的 metadata** |
| ascend_ops | MoE dispatch 原来要读 host 上的 allgather 行数，无法 capture；改成 capture 时用固定宽度 `capacity` 跳过 host 交换 |
| SUM_LEN fix | 零 token 的 DP rank 会把整组 padding 到 bucket，图又 replay 不了，白算一整桶；改成保留 SUM_LEN 走 eager |

三个贯穿全文的容量轴：

| 名称 | 含义 | 由谁决定 |
| --- | --- | --- |
| `T_cap` | token bucket，`positions`/Q/输出的行数 | `forward_batch.positions.shape[0]` |
| `B_cap` | request 槽数，cu/页表/state table 第一维 | `MAX_PREFILL_GRAPH_REQS`（当前 = 1） |
| `C_cap(r)` | 压缩输出容量 | `min(T_cap, T_cap // r + B_cap)` |

图内 kernel 只认**地址**：capture 分配的 tensor 地址必须一直有效，replay 只改内容（`copy_`/`fill_`），不能换对象。

---

## 1. commit 1：P1 wiring（`12667c59dc`）

### 1.1 新文件 `breakable_npu_graph_backend.py`（60 行）

文件 docstring：

```python
"""Breakable NPU graph backend: one torch.npu.NPUGraph per segment,
attention (and other host-input ops) eager between segments.
...
Segment graphs are captured with bare capture_begin/capture_end — the
graph.update channel is not used because every op that consumes host-side
values runs at the eager break points, outside the captured segments.
"""
```

含义：

- BCG 的 **capture 上下文、capture 循环、replay 路径全部复用父类**；NPU 只换“每段用哪个 graph 类”。
- 段图用裸 `capture_begin/capture_end` 捕获，不走 `graph.update`：因为吃 host 值的算子在 eager 断点里，段内不需要动态参数更新。

类本体只有一个 `__init__`：

```python
class BreakableNPUGraphBackend(BreakableCudaGraphBackend):
    """NPU prefill graphs split at eager_on_graph break points."""

    def __init__(self, cuda_graph_runner, *, enable_memory_saver=False, debug_eager=False):
        super().__init__(cuda_graph_runner,
                         enable_memory_saver=enable_memory_saver,
                         debug_eager=debug_eager)
        assert not get_flags().capture.enable_torch_compile, (
            "breakable NPU prefill graph does not support torch.compile; "
            "disable --enable-torch-compile or prefill graph"
        )
```

逐行：

- `BreakableCudaGraphBackend` 是公共 BCG backend；NPU 子类**不重写任何方法**，只做能力门禁。
- `enable_memory_saver` / `debug_eager` 原样透传，保证和 CUDA 路径行为一致。
- `assert not enable_torch_compile`：`torch.compile` 会把 Python 函数编译掉，而 BCG 的 eager 断点（`eager_on_graph` 回调）依赖 Python 在 replay 时执行；两者不能同时开。注意 NPU **decode** 仍然可以走 `NPUGraphRunner` 的 compile 路径，这里只限制 prefill breakable。

### 1.2 `runner_backend/utils.py`：`resolve_prefill_backend` 加 NPU 分支

原来 NPU 会落到最后的 `BreakableCudaGraphBackend`（CUDA 实现），capture 时用 `torch.cuda.CUDAGraph` 直接崩。新增：

```python
enable_memory_saver = get_exec().features.enable_memory_saver

if model_runner.device == "npu":
    if backend_name != Backend.BREAKABLE:
        raise ValueError(
            f"NPU prefill graph only supports 'breakable', got '{backend_name}'"
        )
    from ...breakable_npu_graph_backend import BreakableNPUGraphBackend
    return BreakableNPUGraphBackend(
        cuda_graph_runner,
        enable_memory_saver=enable_memory_saver,
        debug_eager=get_exec().graph.debug_cuda_graph,
    )
```

要点：

- **先校验再 import**：`backend_name != BREAKABLE` 直接 `ValueError`，不静默 fallback。NPU 上 `full` 后端（固定 request slot 的整图）尚未适配，静默走 CUDA 实现只会在 capture 时崩，错误信息也没有指向性。
- lazy import：只有 NPU 才加载 NPU 模块，CUDA/XPU 环境不会因为 import 触发 NPU 依赖。
- `enable_memory_saver` 提到函数开头复用，后面 CUDA 的两个分支也改用它，行为不变。

### 1.3 `breakable_cuda_graph.py`：两处设备相关点

**(a) `_is_stream_capturing` 增加 `is_npu()`**

```python
if is_npu() or is_hip() or is_musa() or _is_xpu:
    with get_device_module().stream(stream):
        return get_device_module().is_current_stream_capturing()
return (... cuda-python 路径 ...)
```

- 这个函数在 capture 期间跟踪 side stream 的 fork/join，决定 `capture_end` 前是否还有别的流参与。
- NVIDIA 上保留 cuda-python 直接查询（历史上 `torch.cuda.is_current_stream_capturing()` 不可靠）；NPU 上必须走 `torch.npu` 的 portable API，不能走 cuda-python。

**(b) 新增 `_device_graph_cls()`**

```python
def _device_graph_cls():
    if is_npu():
        return torch.npu.NPUGraph
    if _is_xpu:
        return torch.xpu.XPUGraph
    return torch.cuda.CUDAGraph
```

`_begin_new_segment` 原来写死：

```python
graph_cls = torch.xpu.XPUGraph if _is_xpu else torch.cuda.CUDAGraph
```

改成 `graph_cls = _device_graph_cls()`。这样每个 segment 在 NPU 上实例化 `torch.npu.NPUGraph`，其余逻辑（pool、`capture_begin/end`、replay）不变。

### 1.4 测试

- `test_backend_resolution.py`：补 `TestResolvePrefillBackendNPU`
  - NPU + breakable → 返回 `BreakableNPUGraphBackend`，并且透传 `enable_memory_saver/debug_eager`；
  - NPU + full → `ValueError`，且不会调用 CUDA `FullCudaGraphBackend`。
- 新文件 `test_breakable_npu_graph_backend.py`（197 行）：
  - 构造后端时不重写父类、`_cuda_graph_runner` 指向传入 runner；
  - `enable_torch_compile=True` 时 assert；
  - `_device_graph_cls()` 按设备返回 NPU/CUDA graph 类；
  - 用假 graph 类验证：两个 `eager_on_graph` 断点把 capture 切成 3 段，每段 `capture_begin` 恰好一次、`capture_end` 恰好一次；replay 时 segment 和断点严格交替；
  - `_is_stream_capturing` 在 NPU 上走 portable 查询，不落到 cuda-python。

---

## 2. commit 2：DSV4 attention / compressor 入图（`e834d6fa61`）

### 2.1 整体数据流

```
capture（bucket T_cap）:
  runner.capture_prepare
    -> backend.init_forward_metadata_for_breakable_cuda_graph_capture(fb)
       -> self.init_forward_metadata(fb)            # 现有 eager builder，生成 seed
       -> Dsv4PrefillGraphMetadata.allocate(seed)   # 分配固定容量 buffer
       -> allocate 内部 fill(capture) + _bind(seed) # 把固定 buffer 绑到 seed
  warmup + capture
    captured segment 按地址读 state.*
    attention body 留在段内（见 2.4）

replay（live batch）:
  runner.load_batch
    -> backend.prepare_forward_metadata_for_breakable_cuda_graph_replay(captured, live)
       -> self.init_forward_metadata(live)          # 按当次请求规划
       -> state.refresh(live)                       # copy_ 进固定 buffer
       -> self.forward_metadata = captured          # eager 断点也读固定 buffer
    -> replay segment
```

### 2.2 新文件 `dsv4_prefill_graph.py`（337 行）

#### 模块 docstring（L14-L25）

- captured segment **按 tensor 地址**读 attention/compressor metadata；
- 每个 token bucket 分配一份固定容量对象，replay 原地刷新；
- **capture 之后不允许替换任何 tensor 对象**（否则图里还是旧地址）；
- 当前 `MAX_PREFILL_GRAPH_REQS = 1`；多请求 replay 需要 fused indexer / sparse attention 的 metadata 算子支持零长度 request slot，尚未验证，所以 runner 会把多请求 batch 送回 eager。

#### 常量

```python
MAX_PREFILL_GRAPH_REQS = 1

_KERNEL_METADATA_KEYS = ("c1a_metadata", "c4a_metadata", "c128a_metadata", "li_quant_metadata")
_MIN_KERNEL_METADATA_INTS = 1024
```

- `MAX_PREFILL_GRAPH_REQS`：一张图最多几个 request 槽。取 1 是保守值，不是算子限制。
- 四个 key 对应 sparse attention / lightning indexer 的 metadata 算子输出。
- `1024` 是 decode graph 已经用过的固定 int32 数量；allocate 时取 `max(1024, capture 实际 numel)`。

#### `compress_capacity(num_tokens, ratio, max_reqs)`

```python
return min(num_tokens, num_tokens // ratio + max_reqs)
```

- compressor 输出的**容量**上界：每个 request 最多贡献 `extend//r + 1` 个新压缩块，总上界约 `T//r + B`，再用 `T` 封顶。
- 与 eager builder 里的 `padding_size = min(t, t // ratio + bs)` 一致。
- 这是容量，不是本次有效块数 `N_r`；`N_r` 只在图外规划。

#### `_copy_1d(dst, src)`

```python
if src is None:
    dst.zero_()
    return
flat = src.reshape(-1)
n = min(flat.numel(), dst.numel())
dst[:n].copy_(flat[:n].to(dtype=dst.dtype))
if n < dst.numel():
    dst[n:].zero_()
```

- 用于 1D metadata：`seqused`、`start_pos`、`positions_cmp_padding_*`、`c*_loc`。
- `src is None`：整个 buffer 清零（capture 期的 loc 走这里 → 写保留 slot 0）。
- 尾部清零：防止“大 batch → 小 batch”读到上一轮残留。
- `.to(dst.dtype)`：live 可能是 int32，buffer 可能是 int64（loc）。

#### `_copy_cu(dst, src)`

```python
flat = src.reshape(-1)
n = min(flat.numel(), dst.numel())
dst[:n].copy_(flat[:n].to(dtype=dst.dtype))
if n < dst.numel():
    # Spare slots are zero-length requests, not a second copy of request 0.
    dst[n:].copy_(flat[n - 1].expand(dst.numel() - n))
```

- 用于 `actual_seq_lengths_q` / `actual_seq_lengths_q_pa`（cumsum）。
- 尾部**不能填 0**：cu 必须非递减。把最后一个值重复，表示后面的槽位是**零长度请求**。
- 例：live `[0, 64]`，容量 3 → `[0, 64, 64]`。
- 这依赖算子接受零长度 request slot；`B_cap=1` 时走不到尾部。

#### `_copy_table(dst, src, name)`

```python
dst.zero_()
if src.shape[0] > dst.shape[0] or src.shape[1] > dst.shape[1]:
    raise RuntimeError(f"DSV4 prefill graph {name} overflow: ...")
rows, cols = src.shape
dst[:rows, :cols].copy_(src[:rows, :cols].to(dtype=dst.dtype))
```

- 用于 `swa_page_table` / `c4_page_table` / `c128_page_table`。
- 先清零：无效行列指向 page 0（allocator 的保留 padded slot）。
- 溢出直接报错，不静默截断。

#### `__init__` 字段

```python
self.metadata = None          # 绑定的 ForwardMetadata
self.num_tokens = 0           # T_cap
self.max_reqs = 0             # B_cap
self.ratios = []              # 需要处理的 4/128
self.positions = {}           # ratio -> positions_cmp_padding_c*
self.locs = {}                # ratio -> c*_loc
self.state_tables = {}        # ratio -> explicit state table
self.dummy_state = {}         # ratio -> dummy_state_loc
self.topk_by_layer = {}       # layer_id -> captured indexer 输出
```

这些字段都是**固定 tensor 的所有者**；外部只通过 `_bind` 拿到引用。`topk_by_layer` 不预分配，见 `stage_topk`。

#### `allocate`

```python
state.metadata = seed
state.num_tokens = num_tokens
state.max_reqs = MAX_PREFILL_GRAPH_REQS
state.ratios = [r for r in backend._dsv4_unique_compress_ratios if r in (4, 128)]
```

- `seed` 是 capture hook 里用现有 eager builder 生成的一份 metadata，只作为**数据来源**。
- `ratios` 只保留 4/128；ratio 0 层没有 compressor/indexer。

```python
state.actual_q = torch.zeros(max_reqs, **int32)
state.actual_q_pa = torch.zeros(max_reqs + 1, **int32)
state.actual_kv = torch.zeros(max_reqs, **int32)
state.seqused = torch.zeros(max_reqs, **int32)
state.start_pos = torch.zeros(max_reqs, **int32)
```

- `actual_q`：`cumsum(extend_lens)`，没有前导 0，长度 B。
- `actual_q_pa`：有前导 0，长度 B+1；compressor 和 attention 共用。
- `actual_kv`：每请求总 KV 长度。
- `seqused`：每请求本次有效 token 数，compressor 用它限制有效计算。
- `start_pos`：每请求本 chunk 的全局起点（prefix 长度）。

```python
for ratio in state.ratios:
    capacity = compress_capacity(num_tokens, ratio, max_reqs)
    state.positions[ratio] = torch.zeros(capacity, dtype=torch.int64, device=device)
    state.locs[ratio] = torch.zeros(capacity, dtype=torch.int64, device=device)
```

- `positions_cmp_padding_c*`：压缩块的全局位置，长度固定 `C_cap(r)`。
- `c*_loc`：压缩 KV 的写回槽，长度固定 `C_cap(r)`。
- capture 时全 0：epilog 把所有容量行写到 slot 0（保留槽），不碰真实请求。
- replay 时前 `N_r` 个是真实 loc，尾部保持 0。

```python
swa_cols = max(1, (context_len + page_size - 1) // page_size)
state.swa_page_table = torch.zeros((max_reqs, swa_cols), **int32)
```

- SWA 页表宽度按**最大上下文**固定，不是 capture 时的短 prefix；replay 可能命中长 prefix。

```python
if backend._dsv4_has_c4:
    state.c4_page_table = torch.zeros((max_reqs, swa_cols), **int32)
if backend._dsv4_has_c128:
    c128_page_size = backend.req_to_token_pool.c128_page_size
    groups = max(1, ((context_len // 128) + c128_page_size - 1) // c128_page_size)
    state.c128_page_table = torch.zeros((max_reqs, groups), **int32)
```

- c4 页表列数口径容易写错：builder 按 token 轴以 `page_size` 步进，所以列数和 SWA 页表一样，**不是压缩块数**。
- c128 页表列数 = `ceil((context_len // 128) / c128_page_size)`，与 `_compute_compress_locs` 的 `n_groups` 公式一致。

```python
metadata_ints = _MIN_KERNEL_METADATA_INTS
for tensor in seed.kernel_metadata.values():
    metadata_ints = max(metadata_ints, tensor.numel())
state.kernel_metadata = {key: torch.zeros(metadata_ints, **int32) for key in _KERNEL_METADATA_KEYS}
```

- 四个 metadata buffer 都按 `max(1024, capture 实际大小)` 分配。
- replay 的 live metadata 不应超过 capture 大小；超过会报错。

```python
for ratio in state.ratios:
    pool = backend._dsv4_state_pools_by_ratio[ratio]
    history = (2 if ratio == 4 else 1) * ratio
    state.dummy_state[ratio] = int(pool.dummy_state_loc)
    state.state_tables[ratio] = torch.full(
        (max_reqs, history + num_tokens), state.dummy_state[ratio], **int32
    )
```

- A3 compressor 需要 explicit state table：`(B_cap, history + T_cap)`。
- `history = coff * ratio`；ratio 4 时 coff=2（1 + overlap），ratio 128 时 coff=1。
- capture 时全部填 `dummy_state_loc`：compressor 的 state 写入只落在 dummy 行，不污染真实 state。
- replay 时先填 dummy，再覆盖有效列（见 `_refresh_state_tables`）。

```python
state.fill(backend=backend, live=seed, forward_batch=None)
state._bind(seed)
return state
```

- `forward_batch=None` 即 capture：把 seed 的 cu/kv/positions/页表/kernel metadata 拷进固定 buffer；loc 不读 seed，保持 0。
- `_bind`：把固定 buffer 挂到 `seed` 上，`forward_metadata` 从此指向它们。
- 返回的 `seed` 就是 runner 保存的 `capture_metadata`。

#### `refresh`

```python
def refresh(self, *, backend, live, forward_batch):
    self.fill(backend=backend, live=live, forward_batch=forward_batch)
```

replay 前调用；`forward_batch` 非空即 replay：会读 live 的 `c*_loc`，并重建 state table。

#### `stage_topk`

```python
self.topk_by_layer[layer_id] = topk_indices
return topk_indices
```

- 每层 c4 indexer 跑完都会调用。
- **不拷贝**：直接保存 capture 时 kernel 产生的 tensor 对象。这个 tensor 是 captured indexer 的输出，replay 时同一 kernel 会往同一地址写；Python 引用保证 storage 不被释放。
- 共享属性 `forward_metadata.c4_topk_indices` 只记得最后一层；真正按层消费时用 `topk_by_layer[layer_id]`。
- 不预分配每层 `[T_cap, topk]`，否则 30 个 c4 层会非常占内存。

#### `fill`

```python
capture = forward_batch is None
real_tokens = None
if forward_batch is not None:
    real_tokens = forward_batch.global_num_token_non_padded_cpu
idle = forward_batch is not None and (
    forward_batch.forward_mode.is_idle() or real_tokens == 0
)
if idle:
    self._fill_idle(backend)
    return
```

- `capture` 由 `forward_batch is None` 推出。
- idle DP rank：没有真实 attention，但不能让 captured compressor/indexer 用非法零长度启动 → 填一套“合法但无效”的 metadata。

```python
_copy_cu(self.actual_q_pa, live.actual_seq_lengths_q_pa)
_copy_cu(self.actual_q, live.actual_seq_lengths_q)
_copy_1d(self.actual_kv, live.actual_seq_lengths_kv)
_copy_1d(self.seqused, live.seqused)
_copy_1d(self.start_pos, live.start_pos)
```

用 live metadata 刷新 cu / kv / seqused / start_pos。`actual_q_pa` 同时是 compressor 的输入 cu 和 attention 的 query cu（B=1 时一致）。

```python
for ratio in self.ratios:
    _copy_1d(self.positions[ratio], getattr(live, f"positions_cmp_padding_c{ratio}"))
    _copy_1d(self.locs[ratio], None if capture else getattr(live, f"c{ratio}_loc"))
```

- capture：positions 用 seed 的值；loc 不读 seed，保持 0。
- replay：positions 和 loc 都从 live 读。
- 尾部填 0：positions 尾部 0 是合法位置；loc 尾部 0 是保留 slot。
- 动态名字用 `getattr` 但**不给默认值**：eager builder 一定会设置它们，给 `None` 默认只会把契约破坏变成静默填 0。

```python
_copy_table(self.swa_page_table, live.swa_page_table, "swa_page_table")
if self.c4_page_table is not None:
    _copy_table(self.c4_page_table, live.c4_page_table, "c4_page_table")
if self.c128_page_table is not None:
    _copy_table(self.c128_page_table, live.c128_page_table, "c128_page_table")
```

页表按 live 的 max context 刷新；宽度不够直接报错；先清零再拷，保证尾部不残留。

```python
self._copy_kernel_metadata(live.kernel_metadata)
if not capture:
    self._refresh_state_tables(backend, forward_batch)
```

- kernel metadata（c1a/c4a/c128a/li_quant）从 live 的 metadata 算子输出拷入固定 buffer。
- reclaim state table 只在 replay 做；capture 保持全 dummy。

#### `_fill_idle`

```python
self.actual_q.fill_(1)
self.actual_q_pa.zero_()
self.actual_q_pa[-1:].fill_(1)
self.actual_kv.fill_(1)
self.seqused.zero_()
self.start_pos.zero_()
```

- 造一个“1 个 dummy query、0 个有效 token”的合法 batch。
- `actual_q_pa = [0, 1]`：kernel 至少有一行可处理，不会因为零长度被算子拒绝。
- `seqused = 0`：compressor 不产生有效输出。
- `actual_kv = 1`：key 长度至少 1，page 0 是保留页。

```python
for ratio in self.ratios:
    self.positions[ratio].zero_()
    self.locs[ratio].zero_()
self.swa_page_table.zero_()
...
for ratio, table in self.state_tables.items():
    table.fill_(self.dummy_state[ratio])
```

所有写回都指向保留区，不碰真实请求的 KV/state。

```python
rebuilt = backend._kernel_metadata_from_parts(
    bs=self.max_reqs,
    actual_seq_lengths_q_pa=self.actual_q_pa,
    actual_seq_lengths_kv=self.actual_kv,
)
self._copy_kernel_metadata(rebuilt)
```

idle 时 live metadata 是空的（`init_forward_metadata` 提前 return），所以用合成长度重新跑一次 metadata 算子，保证 captured kernel 有合法 metadata。

#### `_refresh_state_tables`

```python
batch_size = forward_batch.batch_size
if batch_size > self.max_reqs:
    raise RuntimeError(...)
```

防御性检查：replay 的请求数不能超过 capture 容量。

```python
table = backend._build_dsv4_prefill_graph_state_table(
    ratio=ratio,
    req_pool_indices=forward_batch.req_pool_indices,
    start_pos=self.start_pos[:batch_size],
    cu_seqlens=self.actual_q_pa[: batch_size + 1],
    seqused=self.seqused[:batch_size],
    max_input_capacity=self.num_tokens,
)
dst.fill_(self.dummy_state[ratio])
rows, cols = table.shape
if cols > dst.shape[1]:
    raise RuntimeError(...)
dst[:rows, :cols].copy_(table.to(dtype=dst.dtype))
```

- 图外用 live 的 req_pool_indices / start_pos / cu / seqused 重建 A3 explicit state table。
- `max_input_capacity=self.num_tokens` 固定表宽为 `history + T_cap`。
- 这个调用会 launch 一些 device op，但在图外，安全。
- 先全部填 dummy，再把 live 表拷进去；表宽不够直接报错。

#### `_copy_kernel_metadata`

```python
for key, dst in self.kernel_metadata.items():
    src = source.get(key)
    if src is None:
        dst.zero_()
        continue
    if src.numel() > dst.numel():
        raise RuntimeError(...)
    dst[: src.numel()].copy_(src.to(dtype=dst.dtype))
    if src.numel() < dst.numel():
        dst[src.numel():].zero_()
```

逐 key 拷 metadata 算子输出；live 没有某个 key（例如模型没有 c128）就填 0；live 超过 capture 大小直接报错，避免截断。

#### `_bind`

```python
metadata.actual_seq_lengths_q = self.actual_q
metadata.actual_seq_lengths_q_pa = self.actual_q_pa
metadata.actual_seq_lengths_kv = self.actual_kv
metadata.seqused = self.seqused
metadata.start_pos = self.start_pos

for ratio in self.ratios:
    setattr(metadata, f"positions_cmp_padding_c{ratio}", self.positions[ratio])
    setattr(metadata, f"c{ratio}_loc", self.locs[ratio])

metadata.swa_page_table = self.swa_page_table
...
metadata.kernel_metadata = self.kernel_metadata
metadata.c4_topk_indices = None
metadata.dsv4_explicit_state_block_tables = self.state_tables
metadata.dsv4_max_input_capacity = self.num_tokens
metadata.dsv4_prefill_graph_state = self
```

- 把固定 buffer 挂到 `ForwardMetadata` 对象上，后续 compressor/attention 直接读。
- 字段名与 eager builder 保持一致（动态名字用 `setattr`）。
- `c4_topk_indices = None`：每层的 topk 由 `stage_topk` 单独保存，不挂在共享属性上。
- `dsv4_explicit_state_block_tables`：A3 explicit state table。
- `dsv4_max_input_capacity = T_cap`：eager builder 把它当“每请求最大 chunk”；单请求图里等于 bucket。
- `dsv4_prefill_graph_state = self`：后面 `forward_c4_indexer` 和 `_forward_compressed` 用它判断“当前在读图内 metadata”。

### 2.3 `ascend_dsv4_backend.py`

#### (a) import

新增 `MAX_PREFILL_GRAPH_REQS`、`Dsv4PrefillGraphMetadata`，以及 `Backend/Phase/check_cuda_graph_backend`（判断当前 prefill 是否 breakable）。

#### (b) `forward_c4_indexer`：多流改成调用方显式请求 + 按层保存 topk

```python
if enable_multi_stream and self._can_use_indexer_multi_stream():
    q, weights = self._forward_prepare_multi_stream(...)
else:
    q, weights = self._forward_prepare(...)
```

- 原来只要 `_can_use_indexer_multi_stream()` 为真就走多流；现在必须调用方传 `enable_multi_stream=True`。
- 含义：NPU indexer 默认单流运行；capture 时不会有 side stream/event 进入段内。需要 overlap 的 eager 调用方自己显式开。

```python
topk_idxs = self._forward_indexer(...)
graph_state = getattr(self.forward_metadata, "dsv4_prefill_graph_state", None)
if graph_state is not None:
    topk_idxs = graph_state.stage_topk(c4_indexer.layer_id, topk_idxs)
else:
    compress_layer_id = self.token_to_kv_pool.layer_mapping[c4_indexer.layer_id].compress_layer_id
    maybe_capture_indexer_topk(compress_layer_id, topk_idxs)
self.forward_metadata.c4_topk_indices = topk_idxs
```

- 图内路径：indexer 输出是 captured region 里分配的（storage 属于 graph pool），用 `stage_topk` 按 layer 保存引用；不能走 `maybe_capture_indexer_topk`（那是 eager 的 state capturer，会读 host/做额外动作）。
- eager 路径：保持原来的 `maybe_capture_indexer_topk(compress_layer_id, topk_idxs)`。
- 两条路径最后都更新共享属性 `c4_topk_indices`（给不支持按层读取的旧消费者兜底）。

#### (c) `__init__`：打开 captured-metadata 契约

```python
self.use_captured_forward_metadata_for_breakable_cuda_graph = (
    self._supports_dsv4_prefill_graph_capture()
)
if self.use_captured_forward_metadata_for_breakable_cuda_graph:
    self.prefill_cuda_graph_max_batch_size = MAX_PREFILL_GRAPH_REQS
    logger.info(...)
```

- `use_captured_forward_metadata_for_breakable_cuda_graph` 是 BCG 框架认的能力位：置 True 后，runner 会调 backend 的两个 hook（capture/replay），并且模型层可以选择把 attention 留在段内。
- `prefill_cuda_graph_max_batch_size = 1`：告诉 runner 这张图最多容纳 1 个 request；超过就回 eager。

#### (d) `_supports_dsv4_prefill_graph_capture`

```python
if not check_cuda_graph_backend(Phase.PREFILL, Backend.BREAKABLE):
    return False
if self.is_draft_worker:
    return False
parallel = get_parallel()
if is_npu_arch35() or parallel.attn_cp_size != 1 or parallel.attn_dcp_size != 1:
    raise ValueError("... requires an A3 model without CP/DCP ...")
return True
```

- 只有 breakable prefill 才尝试。
- draft worker 保持 eager metadata（图覆盖由 speculative runner 管）。
- A3 only：A5 的 compressor state 布局不同；CP/DCP 会改变 token 布局和 metadata 语义，当前不支持。**直接 raise**，而不是静默退回，避免用户以为开了图。

#### (e) capture hook

```python
def init_forward_metadata_for_breakable_cuda_graph_capture(self, forward_batch):
    if forward_batch.batch_size > MAX_PREFILL_GRAPH_REQS:
        raise RuntimeError("... Reduce the prefill token bucket or raise MAX_PREFILL_GRAPH_REQS ...")
    if forward_batch.can_run_tbo:
        raise RuntimeError("... does not support TBO ...")
    self.init_forward_metadata(forward_batch)
    return Dsv4PrefillGraphMetadata.allocate(
        backend=self,
        seed=self.forward_metadata,
        num_tokens=int(forward_batch.positions.shape[0]),
    ).metadata
```

- 先检查请求数：capture dummy batch 是用最少请求凑 token bucket；如果 bucket 太大、必须多个请求才能凑满，就超过 `MAX_REQS=1`，直接报错（提示降 bucket，或在验证零长度 slot 后 raise MAX_REQS）。
- TBO（two-batch overlap）不支持：capture/replay 的 metadata 假设单批，TBO 会把 batch 拆成两个子批。
- `init_forward_metadata` 是现有 eager builder，生成 seed；`allocate` 把 seed 的值拷进固定 buffer，并把固定 buffer 绑回 seed；返回的 seed 就是 runner 保存的 `capture_metadata`。

#### (f) replay hook

```python
def prepare_forward_metadata_for_breakable_cuda_graph_replay(self, capture_metadata, forward_batch, *, static_forward_batch=None):
    state = getattr(capture_metadata, "dsv4_prefill_graph_state", None)
    if state is None:
        raise RuntimeError("... missing its captured metadata")
    if forward_batch.can_run_tbo:
        raise RuntimeError("... does not support TBO")
    self.init_forward_metadata(forward_batch)
    state.refresh(backend=self, live=self.forward_metadata, forward_batch=forward_batch)
    self.forward_metadata = capture_metadata
```

- 从 capture metadata 上取回 state；没有就是契约破坏，直接报错。
- 先跑 live 的 eager builder（`init_forward_metadata`），得到当次请求的 lengths/locs/page tables/state table 规划。
- `state.refresh` 把 live 值原地拷进固定 buffer。
- 最后把 `self.forward_metadata` 指回 capture metadata：这样图外的 eager attention 断点读到的也是固定、已刷新的 buffer。

#### (g) `_build_dsv4_prefill_graph_state_table`

```python
return _build_explicit_state_block_table(
    compress_ratio=ratio,
    coff=2 if ratio == 4 else 1,
    state_pool=self._dsv4_state_pools_by_ratio[ratio],
    token_to_kv_pool=self.token_to_kv_pool,
    req_to_token=self.req_to_token,
    req_pool_indices=req_pool_indices,
    start_pos=start_pos,
    cu_seqlens=cu_seqlens,
    seqused=seqused,
    max_input_capacity=max_input_capacity,
)
```

- 复用现有 A3 explicit state table builder。
- `coff=2` 只给 ratio 4（compressor 有 1 个 overlap 窗口）；ratio 128 用 1。

#### (h) `_forward_compressed`：按层取 topk + 切 live 行 + 错误上下文

```python
topk = None
attn_kwargs["cmp_sparse_indices"] = None
if compress_ratio == 4:
    topk = fm.c4_topk_indices
    graph_state = getattr(fm, "dsv4_prefill_graph_state", None)
    if graph_state is not None:
        topk = graph_state.topk_by_layer[layer.layer_id]
        real_tokens = forward_batch.global_num_token_non_padded_cpu
        if real_tokens is not None and 0 < real_tokens < topk.shape[0]:
            topk = topk[:real_tokens]
    attn_kwargs["cmp_sparse_indices"] = topk.view(-1, 1, topk.shape[-1])
```

- c4 用 indexer topk 做稀疏 attention；c128 读完整压缩历史，不走 topk。
- 图内路径：共享属性只记得最后一层，必须按 `layer.layer_id` 从 `topk_by_layer` 取。
- captured buffer 是容量大小（bucket）；如果本次 replay 的真实 token 数小于容量，就让 topk 跟着 q 切到 live 行（q 已经在断点处切过）。
- `cmp_sparse_indices` 统一 reshape 成 `[T, 1, topk]`。

```python
try:
    out, _ = attn_op(q_arg, **attn_kwargs)
except RuntimeError as exc:
    raise RuntimeError(
        f"DSV4 sparse attention failed: layer={layer.layer_id} "
        f"ratio={compress_ratio} q={tuple(q_arg.shape)} "
        f"cmp_kv={tuple(cmp_kv.shape)} "
        f"topk={None if topk is None else tuple(topk.shape)}"
    ) from exc
```

- 算子自己的报错不含 layer/ratio/shape；这里补上，方便定位是哪个压缩层、topk 形状不对还是 KV 形状不对。

### 2.4 `deepseek_v4.py`：attention 留在段内还是做 eager break

```python
attention_with_output = (
    deepseek_v4_attention_with_output
    if _is_npu
    and attn_backend.use_captured_forward_metadata_for_breakable_cuda_graph
    else bcg_deepseek_v4_attention_with_output
)
attention_with_output(attn_q, attn_k, o, self.attn_mqa.layer_id, self.compress_ratio, attn_sink, save_kv_cache)
```

- `bcg_deepseek_v4_attention_with_output = eager_on_graph(True)(deepseek_v4_attention_with_output)`：显式断点，capture 时只记录、replay 时执行。
- NPU + 能力位为 True 时，直接调原始函数 `deepseek_v4_attention_with_output`，它内部的 `attn_backend.forward` 就留在 captured segment 里。
- 其他平台/其他后端保持 `bcg_` 断点，行为不变。
- 前提：`deepseek_v4_attention_with_output` 在 capture 时读到的是 dummy batch（`real_num_tokens` = bucket），`query[:real]`/`out_cache_loc[:real]` 都是满 bucket 的切片；replay 时图内 kernel 读刷新后的 capacity-sized metadata，按真实长度计算，padding 行不参与输出。

### 2.5 测试

`test_dsv4_prefill_graph_metadata.py`（4 个 case，纯 CPU）：

1. **capture 分配固定隔离 buffer**：`allocate` 后 `state.metadata is seed`、`seed.dsv4_prefill_graph_state is state`；capture 期 `c4_loc` 形状 = `compress_capacity(...)` 且全 0（不写真实 slot）；state table 全 dummy；页表宽度符合 builder 口径（SWA/c4 列数相同，c128 用 group 数）。
2. **replay 原地刷新**：refresh 前后固定 tensor 的 `id()` 不变；cu/kv/seqused 更新为 live 值；state table 有效列是 live 表、尾部保持 dummy。
3. **idle replay 合法**：`actual_q_pa=[0,1]`、`actual_kv=[1]`、`seqused=[0]`，state table 全 dummy。
4. **topk 按层保存**：`stage_topk(3, first)` / `stage_topk(5, second)` 分别保留，共享属性仍为 None。

---

## 3. commit 3：ascend_ops MoE 入图（`4c3a237e38`）

### 3.1 新文件 `hardware_backend/npu/moe/prefill_graph.py`（65 行）

模块 docstring：

- captured prefill segment 里每个 MoE 层都按 bucket 宽度跑，replay 没有 Python；
- dispatcher 无法从 live tensor 读 allgather 行数；
- marker 绑到 batch 上时，dispatcher 走**固定宽度 allgather**，不需要 host 行数交换；
- 与 DSV4 metadata 不同，这个 state 不持有 tensor，replay 不需要刷新。

```python
class MoePrefillGraphState(msgspec.Struct, frozen=True):
    """Marker: the enclosing forward is being captured into a prefill graph.

    The dispatch width is the layer's own input height, not a value carried
    here: capture makes that height identical on every rank, and it already
    accounts for any DP gather ahead of the MoE.
    """
```

- 空结构体，只作为 marker；宽度不存这里，而是取 `hidden_states.shape[0]`（层输入高度）。
- capture 时每个 rank 的行数都等于 bucket（MAX_LEN 已经对齐），并且 DP gather 之后的 MoE 输入已经包含了 gather 后的行数。

```python
def install_capture_state(forward_batch):
    if not get_moe_a2a_backend().is_ascend_ops():
        return
    if get_moe_ascend_ops_config().prefill != "allgather":
        return
    forward_batch.moe_prefill_graph_state = MoePrefillGraphState()
```

- 只有 `ascend_ops` + `prefill == "allgather"` 才绑定；其他 MoE 后端/配置完全不变。
- 在 `capture_prepare` 里对 dummy batch 调用一次；eager batch 不会走到这里。

```python
def get_moe_prefill_graph_state():
    batch = try_get_forward_batch()
    return None if batch is None else batch.moe_prefill_graph_state
```

- 从 runner 发布的 ContextVar 取当前 batch；不在 runner forward 里（例如单测直接调 dispatcher）返回 None，不抛异常。

### 3.2 `token_dispatcher/ascend_ops.py`

#### (a) `dispatch`：marker 检查必须在 `is_extend_in_batch` 之前

```python
topk_weights, topk_ids, _ = topk_output
# Must precede the extend check: the prefill runner forces
# is_extend_in_batch False while capturing, so only the capture marker
# keeps this on allgather instead of the decode MC2 path.
if get_moe_prefill_graph_state() is not None:
    return self._dispatch_allgather(
        hidden_states, topk_weights, topk_ids, self.tp_group,
        capacity=hidden_states.shape[0],
    )
if get_is_extend_in_batch():
    ...
```

- capture 期间 runner 会把 `is_extend_in_batch` 强制成 False（为了让 capture 走和 decode 不同的路径）；如果没有 marker，dispatcher 会掉进 decode 的 MC2 分支，录错 kernel。
- 所以 marker 检查放在 `get_is_extend_in_batch()` 之前。
- `capacity=hidden_states.shape[0]`：宽度取层输入高度，capture 时每 rank 相同，且已经包含 DP gather 后的行数。

#### (b) `_dispatch_allgather`：新增 `capacity` 分支

```python
def _dispatch_allgather(self, hidden_states, topk_weights, topk_ids, group, *, capacity=None):
    """Gather the full batch on every TP rank.

    ``capacity`` is the captured bucket width: the graph runs every rank at
    that width, so the host row-count exchange is skipped.
    """
    ...
    if capacity is not None:
        # Captured graph: every rank runs the same bucket width, and a
        # captured segment cannot read live row counts.
        row_sizes = [capacity] * dist.get_world_size(group)
    else:
        # CP and DP can split a batch unevenly ...
        row_sizes = self._allgather_row_sizes(hidden_states.shape[0], group, hidden_states.device)
```

- **graph 路径**：所有 rank 的 `row_sizes` 都是 bucket，`max_rows == capacity`，`equal_rows=True`，后面的 `gather_rows` 走 contiguous + `all_gather_into_tensor`，不做 host 行数交换（`dist.all_gather` + `.item()` 会 D2H 同步，capture 不了）。
- **eager 路径**：main 现在的行为——实际交换每 rank 行数。CP/DP 可能切得不均匀（interleaved/zigzag），从 DP metadata 推行数在 CP 下不正确。
- 注意：这个分支在本分支上和原 commit 有一处差异，见 3.7。

后面的代码不变：`max_rows = max(row_sizes)`、`equal_rows = min == max`、`gather_rows` 先 pad 再 allgather、最后按 `row_sizes` 切回实际行数。

### 3.3 `forward_batch_info.py`：新字段

```python
# Marks a captured prefill forward so the MoE dispatcher takes its
# fixed-width path; None keeps the layer's own runtime dispatch.
moe_prefill_graph_state: Optional["MoePrefillGraphState"] = None
```

- 类型只在 `TYPE_CHECKING` 下 import，运行时不会加载 NPU 模块。
- 默认 None：普通 eager batch 行为不变。
- 只有 `install_capture_state` 对 capture dummy batch 设置它。

### 3.4 `prefill_cuda_graph_runner.py`：capture 时安装 marker

```python
# Local import keeps this shared runner free of the NPU-only module;
# the call is a no-op for every backend that binds no state.
from sglang.srt.hardware_backend.npu.moe.prefill_graph import install_capture_state
install_capture_state(forward_batch)
return forward_batch, self.model_runner.attn_backend
```

- 位置：`capture_prepare` 末尾，dummy batch 已经构造、TBO 已经处理，capture forward 马上要用这个 batch。
- 局部 import：共享的 prefill runner 不常驻 NPU-only 依赖；非 NPU/非 ascend_ops 下 `install_capture_state` 直接 no-op。
- 只在 capture 的 dummy batch 上装 marker；replay 时图内不跑 Python，不需要 marker。

### 3.5 `runner_utils/forward_batch.py`：不抛异常的取 batch

```python
def try_get_forward_batch() -> "ForwardBatch | None":
    """Return the published batch, or None outside a runner-owned forward."""
    return _current_batch.get()
```

- 原来的 `get_forward_batch()` 在没有 batch 时抛异常；`get_moe_prefill_graph_state()` 需要在“不在 runner forward 里”时安静返回 None（例如 CPU 单测直接调 dispatcher），所以加一个非抛异常版本。

### 3.6 测试

`test_ascend_ops.py` 新增 4 个 case：

1. **capture marker 选择 capacity allgather**：即使 `is_extend_in_batch=False`，有 marker 也走 `_dispatch_allgather(..., capacity=hidden_states.shape[0])`，不走 `_dispatch_low_latency`。
2. **capacity 模式不做行数交换**：`_allgather_row_sizes` 被 patch 成一旦调用就 assert；`dist.get_world_size` / `all_gather_into_tensor` 用假实现；最终 `row_sizes == [capacity]*world`、`hidden_states.shape[0] == world*capacity`。
3. **marker 只在 ascend_ops + allgather 绑定**：deepEP、alltoallv 都不绑定；ascend_ops + allgather 才绑定。
4. **没有 forward batch 时返回 None**：`try_get_forward_batch` 返回 None → `get_moe_prefill_graph_state()` 返回 None，不抛异常。

### 3.7 和原 commit 的差异（main 基线冲突）

原 commit `cc223cf15e` 的 eager 路径是“优先复用 DP/CP token metadata，否则行数交换”。但 `origin/main` 上 `_dispatch_allgather` 已经改成**无条件行数交换**（CP/DP 可以切得不均匀，从 metadata 推假定均分，CP 下不正确）。

cherry-pick 到 main 时这里冲突，解决方式是：

- 保留 main 的无条件 `_allgather_row_sizes` 作为 eager 路径；
- 只加 commit 需要的 `capacity` 图路径分支。

因此新分支的 eager MoE 仍然每层做一次行数 all_gather；图路径完全跳过它。旧分支基线 `30ff2df417` 上的 DP/CP metadata 复用优化**不在新分支里**，这是 main 基线的行为差异，不是 cherry-pick 丢代码。

---

## 4. commit 4：零 token rank 保留 SUM_LEN（`3d44122dd3`）

改的是 `forward_batch_info.py` 里 `prepare_mlp_sync_batch` 的 MAX_LEN 强制条件，只加了一行代码 + 4 行注释：

```python
# A zero-token rank also keeps SUM_LEN: padding it to the bucket would
# make the whole group do bucket-sized work for no tokens, while the
# graph runner rejects the replay once any rank is inactive. Eager then
# computes only the rows that exist.
prefill_cg = get_exec().graph.cuda_graph_config.prefill
if (
    self.can_run_dp_prefill_cuda_graph
    and self.is_extend_in_batch
    and prefill_cg.bs
    and max(global_num_tokens) <= max(prefill_cg.bs)
    and min(global_num_tokens) > 0          # ← 新增
    and not prefill_graph_tolerates_sum_len()
):
    dp_padding_mode = DpPaddingMode.MAX_LEN
    bucket = min(size for size in prefill_cg.bs if size >= max(global_num_tokens))
    global_num_tokens = [bucket] * sync_group_size
```

逐行解释：

- `self.can_run_dp_prefill_cuda_graph`：调度期 DP 全体 vote（min-reduce）的结果，表示这一步“可能”跑 prefill 图。
- `self.is_extend_in_batch`：这一步有 prefill。
- `prefill_cg.bs`：capture 了桶；没 capture 就不需要强制。
- `max(global_num_tokens) <= max(prefill_cg.bs)`：超过最大桶就回 eager，不强制。
- **`min(global_num_tokens) > 0`（新增）**：只要有一个 DP rank 是 0 token，就不强制 MAX_LEN。
- `not prefill_graph_tolerates_sum_len()`：ascend_ops/deepep/CP 不允许 SUM_LEN；megamoe 允许，不受这段影响。

为什么必须加：

1. 没有这行时，零 token rank 也会被 `MAX_LEN` 补到 bucket：**整组 16 个 rank 都按 bucket 宽度跑**，而真实 token 只有少数 rank 的少量行。
2. 图 runner 本来有 `_has_inactive_dp_rank` 会拒绝 replay，但 `prepare_mlp_sync_batch` 后面会把 `global_num_tokens_cpu` 覆盖成 padding 后的列表，导致这个检查看不到真实的 0，形同虚设。
3. 于是这一步既做了 bucket 大小的 eager 计算，又没跑成图——纯亏。

加上这行后：

- 零 token rank → 条件不成立 → 保持 `SUM_LEN` → 图 runner 读到真实的 0 → 拒绝 replay → 这一步走 eager，按各 rank 真实行数计算。
- 全组都有 token → 条件成立 → 仍然 `MAX_LEN` + 统一 bucket → 图路径行为完全不变。
- 代价：稀疏 step 不跑图（本来也 replay 不了），但不再白算一整桶；这是当前最优解。

---

## 附录 A：关键契约速查

| 契约 | 位置 | 规则 |
| --- | --- | --- |
| 固定地址 | `dsv4_prefill_graph.py` 模块 docstring | capture 后只能 `copy_`/`fill_`，不能换 tensor 对象 |
| 三个容量 | `allocate` | `T_cap` = positions 行数；`B_cap` = 1；`C_cap(r) = min(T, T//r + B)` |
| cu 尾部 | `_copy_cu` | 重复最后一个值（零长度请求），不能填 0 |
| loc 尾部 | `_copy_1d` | 填 0（保留 slot 0），不能指向真实请求 |
| state table | `allocate` / `_refresh_state_tables` | capture 全 dummy；replay 先 dummy 再覆盖有效列 |
| kernel metadata | `_copy_kernel_metadata` | live 大小不得超过 capture 大小；缺失 key 填 0 |
| topk | `stage_topk` / `_forward_compressed` | 按 layer_id 取；容量 buffer 按 `real_tokens` 切 |
| MoE marker | `moe/prefill_graph.py` | 只在 capture dummy batch 上；宽度 = 层输入高度 |
| 图路径行数 | `ascend_ops._dispatch_allgather` | `capacity` 非空 → 不做 host 行数交换 |
| 零 token rank | `prepare_mlp_sync_batch` | `min > 0` 才强制 MAX_LEN，否则 SUM_LEN → eager |

## 附录 B：与旧分支/旧文档的差异

- 本文对应新分支 `prefill_graph_ascend_ops_on_main`；旧分支 `prefill_graph_ascend_ops` 的 4 个 commit hash 是 `be5fbe4525` / `317f99b1f4` / `cc223cf15e` / `393da680d6`，内容基本相同。
- 旧分支基线 `30ff2df417`（`liwansi/feat/dev_ep_eplb`）带 EPLB/PP/CP 改动和 DP/CP metadata 复用；新分支基线是 `origin/main`，这些都不在。
- `ascend_ops._dispatch_allgather` 的 eager 路径因此改成了 main 的无条件行数交换（见 3.7）。
- 旧文档 `dsv4_prefill_graph_code_walkthrough.md` 对 DSV4 metadata 的逐行解释仍然适用；本文按新分支的实际代码重新核对过。

## 附录 C：测试覆盖

| 测试文件 | 覆盖点 |
| --- | --- |
| `test_backend_resolution.py` | NPU 后端解析、full 后端拒绝、参数透传 |
| `test_breakable_npu_graph_backend.py` | 后端构造、torch.compile 拒绝、graph 类分发、段切分/replay 顺序、portable capture 查询 |
| `test_dsv4_prefill_graph_metadata.py` | 固定 buffer、原地刷新、idle 合法、按层 topk |
| `test_ascend_ops.py` | marker 选 capacity allgather、不做行数交换、marker 绑定条件、无 batch 返回 None |

所有测试都是 CPU 单测，不需要 NPU；NPU 上的 capture/replay 正确性仍需真机验证。
