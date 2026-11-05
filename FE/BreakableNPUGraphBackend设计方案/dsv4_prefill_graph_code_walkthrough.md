# DSV4 Breakable Prefill Graph 代码逐段说明

基线：`sglang` 分支 `prefill_graph_deepseek`，基线 commit `92b02637f7`。

本文对应到 `HEAD` 的以下提交：

| 提交 | 内容 |
| --- | --- |
| `6f9a263ada` | 主功能：compressor/indexer 入图 + 固定 metadata（含 scatter guard 移除、topk 切 live 行、idle 兜底） |
| `859cfc07f4` | 精简：删除死分支、冗余参数和重复赋值；本文按此版本核对过 |
| `723e4a245d` | 再精简：删除恒真/恒假分支；replay hook 改为读 live batch 判断 TBO |

文件清单：

| 文件 | 作用 |
| --- | --- |
| `python/sglang/srt/hardware_backend/npu/attention/dsv4_prefill_graph.py` | 固定容量 metadata 容器与刷新逻辑（348 行） |
| `python/sglang/srt/hardware_backend/npu/attention/ascend_dsv4_backend.py` | 打开 captured-metadata 契约、两个 hook、按层 topk |
| `test/registered/unit/npu/graph_runner/test_dsv4_prefill_graph_metadata.py` | CPU 单测（4 个 case） |

---

## 0. 先理解整体数据流

BCG（breakable CUDA graph）的规则：

- 普通 Python 代码在 capture 时执行一次，它 launch 的 device op 被录进 segment；
- `eager_on_graph` 装饰的函数不会被执行，只记录成 replay 时的回调；
- replay 时只重放 segment + 回调，**Python 里写死的 tensor 地址和 Python 标量不会重新计算**。

所以要把 compressor/indexer 留在图内，必须满足：

1. 图里读到的每个 metadata tensor 地址固定；
2. replay 前把这些 tensor 的**内容**原地刷新；
3. 所有 Python 分支（`if` 用到的标量）在 capture 和 replay 下走同一条路径。

整体流程：

```
capture:
  runner.capture_prepare(bucket T)
    -> backend.init_forward_metadata_for_breakable_cuda_graph_capture(fb)
       -> self.init_forward_metadata(fb)          # 现有 eager builder，作为 seed
       -> Dsv4PrefillGraphMetadata.allocate(seed) # 分配固定 buffer，capture 期 loc/state 用 dummy
       -> allocate 原地把固定 buffer 绑到 seed 上并返回 seed.metadata
    -> warmup x2 + capture
       captured segment 按地址读 state.*
       attention 仍然走 eager break（capture 时只记录）

replay:
  runner.load_batch(live fb)
    -> backend.prepare_forward_metadata_for_breakable_cuda_graph_replay(captured, live)
       -> self.init_forward_metadata(live)        # 按当次请求重新规划
       -> state.refresh(live)                     # copy_ 进固定 buffer
       -> self.forward_metadata = captured
    -> backend.replay(shape_key)                  # 重放 captured segment
    -> eager attention break 读 captured metadata
```

三个容量轴：

| 名称 | 含义 | 由谁决定 |
| --- | --- | --- |
| `T_cap` | token bucket，`positions`/`X`/Q 的行数 | `forward_batch.positions.shape[0]` |
| `B_cap` | request 槽数，cu/页表/state table 第一维 | `MAX_PREFILL_GRAPH_REQS` |
| `C_cap(r)` | 压缩输出容量 | `min(T_cap, T_cap // r + B_cap)` |
| `N_r` | 本次有效新增压缩块数 | `sum(floor((p_i+e_i)/r) - floor(p_i/r))`，只用于图外规划 |

`N_r <= C_cap(r)`，图内不做 `N_r` 的 Python 切片。

---

## 1. `dsv4_prefill_graph.py`

### 1.1 模块 docstring

```python
"""Fixed-capacity metadata for DSV4 prefill inside breakable NPU graphs.
...
Only one request per replay is admitted for now (``MAX_PREFILL_GRAPH_REQS``).
...
"""
```

- L16-L19：说明为什么需要固定容量：captured segment 按地址读 metadata，replay 只能原地刷新，不能替换 tensor 对象。
- L21-L24：说明当前只支持单请求；多请求需要先验证 fused indexer 对零长度 request slot 的行为。这是后面 `B_cap=1` 的依据。

### 1.2 `MAX_PREFILL_GRAPH_REQS`

```python
MAX_PREFILL_GRAPH_REQS = 1
```

- 一张 prefill 图最多容纳多少个 request 槽。
- 取 1 不是算子限制，而是“零长度槽位未验证”时的保守值。
- 如果改成 2/4/8：
  - `B_cap` 变大，`C_cap(r)` 跟着变大；
  - capture 的尾部槽会出现零长度 cu；
  - 需要先做第 1 条验证（见文末“假设”）。

### 1.3 `_KERNEL_METADATA_KEYS` / `_MIN_KERNEL_METADATA_INTS`

```python
_KERNEL_METADATA_KEYS = (
    "c1a_metadata",
    "c4a_metadata",
    "c128a_metadata",
    "li_quant_metadata",
)
_MIN_KERNEL_METADATA_INTS = 1024
```

- 四个 key 对应 sparse attention / lightning indexer 的 metadata 算子输出。
- `1024` 是 decode graph 已经使用的固定 int32 数量（`_init_dsv4_graph_buffers` 里也是 1024）。这里取 `max(1024, capture 时实际 numel)`，防止 prefill 的 metadata 比 1024 大。

### 1.4 `compress_capacity`

```python
def compress_capacity(num_tokens: int, ratio: int, max_reqs: int) -> int:
    """Output rows the fused compressor reserves for one token bucket."""
    return min(num_tokens, num_tokens // ratio + max_reqs)
```

- 这是 TH 形态下 compressor 的输出上界：
  - 每个 request 最多贡献 `extend//r + 1` 个新块；
  - 总上界约 `T//r + B`；
  - 再用 `T` 封顶。
- 它与现有 eager builder 里的 `padding_size = min(t, t // ratio + bs)` 完全一致。
- 注意：这是**容量**，不是有效块数 `N_r`。

### 1.5 `_copy_1d`

```python
def _copy_1d(dst, src):
    if src is None:
        dst.zero_()
        return
    flat = src.reshape(-1)
    n = min(flat.numel(), dst.numel())
    dst[:n].copy_(flat[:n].to(dtype=dst.dtype))
    if n < dst.numel():
        dst[n:].zero_()
```

- 用于 1D metadata：`seqused`、`start_pos`、`positions_cmp_padding`、`c*_loc`。
- `src is None`：整个 buffer 清零（capture 期的 loc 走这里 → 写 slot 0）。
- `n`：live 长度和容量取小，避免越界。
- 尾部清零：保证“大 batch → 小 batch”不会读到上一轮残留。
- `.to(dtype=dst.dtype)`：live 可能是 int32，buffer 可能是 int64（loc）。
- （精简掉了原先恒为 0 的 `fill` 参数。）

### 1.6 `_copy_cu`

```python
def _copy_cu(dst, src):
    flat = src.reshape(-1)
    n = min(flat.numel(), dst.numel())
    dst[:n].copy_(flat[:n].to(dtype=dst.dtype))
    if n < dst.numel():
        # Spare slots are zero-length requests, not a second copy of request 0.
        dst[n:].copy_(flat[n - 1].expand(dst.numel() - n))
```

- 用于 `actual_seq_lengths_q_pa` / `actual_seq_lengths_q`（cumsum）。
- 尾部不是填 0：cu 必须非递减。把最后一个值重复，表示后面槽位都是**零长度请求**。
- 例：live `[0, 64]`，容量 3 → `[0, 64, 64]`。
- 注意：这依赖算子接受零长度 request slot；`B_cap=1` 时不会走到尾部。

### 1.7 `_copy_table`

```python
def _copy_table(dst, src, name):
    dst.zero_()
    if src.shape[0] > dst.shape[0] or src.shape[1] > dst.shape[1]:
        raise RuntimeError(...)
    rows, cols = src.shape
    dst[:rows, :cols].copy_(...)
```

- 用于 `swa_page_table` / `c4_page_table` / `c128_page_table`。
- 先整体清零：无效列/行指向 page 0；page 0 是 allocator 保留的 padded slot（`free_pages = arange(1, ...)`）。
- 溢出直接报错，不静默截断。
- 行=request 槽，列=上下文宽度。

### 1.8 `__init__`

```python
self.metadata = None          # 绑定的 ForwardMetadata
self.num_tokens = 0           # T_cap
self.max_reqs = 0             # B_cap
self.ratios = []              # 需要处理的 4/128
self.positions = {}           # ratio -> positions_cmp_padding_c*
self.locs = {}                # ratio -> c*_loc
self.state_tables = {}        # ratio -> explicit state table
self.dummy_state = {}         # ratio -> dummy_state_loc
self.topk_by_layer = {}       # layer_id -> captured topk tensor
```

- 这些字段都是**固定 tensor 的所有者**；外部只通过 `_bind` 拿到引用。
- `topk_by_layer` 不预分配，见 `stage_topk`。

### 1.9 `allocate`

逐段：

```python
state = cls()
state.metadata = seed
state.num_tokens = num_tokens
state.max_reqs = MAX_PREFILL_GRAPH_REQS
state.ratios = [r for r in backend._dsv4_unique_compress_ratios if r in (4, 128)]
```

- `seed` 是 capture hook 里用现有 eager builder 生成的一份 metadata，只作为**数据来源**。
- `ratios` 是 unique 后的 `[4, 128]` 子集；ratio 0 层不需要 compressor/indexer。

```python
device = backend.device
max_reqs = state.max_reqs
int32 = {"dtype": torch.int32, "device": device}

state.actual_q = torch.zeros(max_reqs, **int32)
state.actual_q_pa = torch.zeros(max_reqs + 1, **int32)
state.actual_kv = torch.zeros(max_reqs, **int32)
state.seqused = torch.zeros(max_reqs, **int32)
state.start_pos = torch.zeros(max_reqs, **int32)
```

- `actual_q`：`cumsum(extend_lens)`，**没有**前导 0，长度 B。
- `actual_q_pa`：有前导 0，长度 B+1，compressor 和 attention 的 cu。
- `actual_kv`：每请求总 KV 长度。
- `seqused`：每请求本次有效 token 数，compressor 用它限制有效计算。
- `start_pos`：每请求本 chunk 的全局起点（prefix）。

```python
for ratio in state.ratios:
    capacity = compress_capacity(num_tokens, ratio, max_reqs)
    state.positions[ratio] = torch.zeros(capacity, dtype=torch.int64, device=device)
    state.locs[ratio] = torch.zeros(capacity, dtype=torch.int64, device=device)
```

- `positions_cmp_padding_c*`：压缩块的全局位置，长度固定为 `C_cap(r)`。
- `c*_loc`：压缩 KV 的写回槽，长度固定为 `C_cap(r)`。
- capture 时先全 0：epilog 会把所有容量行写到 slot 0（保留槽），不碰真实请求。
- replay 时前 `N_r` 个是真实 loc，尾部保持 0。

```python
page_size = backend.page_size
context_len = backend.max_context_len
swa_cols = max(1, (context_len + page_size - 1) // page_size)
state.swa_page_table = torch.zeros((max_reqs, swa_cols), **int32)
```

- SWA 页表宽度按 **最大上下文** 固定，而不是 capture 时的短 prefix。
- replay 可以命中长 prefix，宽度必须够。

```python
if backend._dsv4_has_c4:
    state.c4_page_table = torch.zeros((max_reqs, swa_cols), **int32)
```

- c4 页表的列数口径容易写错：
  - `_compute_compress_locs` 里是 `torch.arange(0, n_c_tokens * ratio, page_size)`；
  - 列数 = `ceil((n_c_tokens * ratio) / page_size)`；
  - 和 SWA 页表一样按 page 步进，**不是**压缩块数。
- 这里直接用 `swa_cols`，是保守上界。

```python
if backend._dsv4_has_c128:
    c128_page_size = backend.req_to_token_pool.c128_page_size
    groups = max(1, ((context_len // 128) + c128_page_size - 1) // c128_page_size)
    state.c128_page_table = torch.zeros((max_reqs, groups), **int32)
```

- c128 页表来自 `req_to_c128_sidecar`；
- 列数 = `ceil((context_len // 128) / c128_page_size)`；
- 与 `_compute_compress_locs` 的 `n_groups` 公式一致。

```python
metadata_ints = _MIN_KERNEL_METADATA_INTS
for tensor in seed.kernel_metadata.values():
    metadata_ints = max(metadata_ints, tensor.numel())
state.kernel_metadata = {
    key: torch.zeros(metadata_ints, **int32) for key in _KERNEL_METADATA_KEYS
}
```

- 四个 metadata buffer 都按 `max(1024, capture 实际大小)` 分配。
- replay 时 live metadata 应该不超过 capture 的大小（capture 用的是最大 bucket），超过会报错。

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
- `history` = `coff * ratio`，ratio 4 时 coff=2，ratio 128 时 coff=1。
- capture 时全部填 `dummy_state_loc`：compressor 的 state 写入只落在 dummy 行，不污染真实 state。
- replay 时先填 dummy，再覆盖前 `N_r` 个有效列（见 `_refresh_state_tables`）。

> A5 的 cycle table 分支已删除：`_supports_dsv4_prefill_graph_capture` 在 A5 上直接 raise，
> `allocate` 永远走不到，那个 buffer 恒为 `None`。

```python
state.fill(backend=backend, live=seed, forward_batch=None)
state._bind(seed)
return state
```

- `forward_batch=None` 即 capture：把 seed 的 cu/kv/positions/页表/kernel metadata 拷进固定 buffer；loc 不读 seed，保持 0。
- `_bind`：把固定 buffer 挂到 `seed` 上，`forward_metadata` 从此指向它们。
- 返回的 `seed` 就是 runner 保存的 `capture_metadata`。

### 1.10 `refresh`

```python
self.fill(backend=backend, live=live, forward_batch=forward_batch)
```

- replay 前调用。
- `forward_batch` 非空即 replay：会读 live 的 `c*_loc`，并重建 state table。

### 1.11 `stage_topk`

```python
self.topk_by_layer[layer_id] = topk_indices
return topk_indices
```

- 每层 c4 indexer 跑完都会调用。
- **不拷贝**：直接保存 capture 时 kernel 产生的 tensor 对象。
- 共享属性 `forward_metadata.c4_topk_indices` 由调用方（`forward_c4_indexer`）负责写，
  这里不重复赋值。
- 原因：这个 tensor 是 captured indexer 的输出，replay 时同一 kernel 会往同一地址写；Python 引用保证 storage 不被释放。
- `metadata.c4_topk_indices` 只是共享属性，capture 结束时只保留最后一层；真正的消费在 `_forward_compressed` 里按 `layer_id` 查 `topk_by_layer`。
- 不拷贝的好处：不需要为每层额外分配 `[T_cap, topk]`，否则 30 个 c4 层会非常占内存。

### 1.12 `fill`

```python
capture = forward_batch is None
idle = forward_batch is not None and (
    forward_batch.forward_mode.is_idle() or real_tokens == 0
)
if idle:
    self._fill_idle(backend)
    return
```

- `capture` 由 `forward_batch is None` 推出，不再单独传参（两个参数只编码了一个 bit）。
- idle DP rank：没有真实 attention，但不能让 captured compressor/indexer 用非法零长度启动。
- 走 `_fill_idle` 填一套“合法但无效”的 metadata。
- `real_tokens` 只有在 `forward_batch` 非空时才可能取值；`None == 0` 本身就是 False，不需要额外的 `is not None`。

```python
_copy_cu(self.actual_q_pa, live.actual_seq_lengths_q_pa)
_copy_cu(self.actual_q, live.actual_seq_lengths_q)
_copy_1d(self.actual_kv, live.actual_seq_lengths_kv)
_copy_1d(self.seqused, live.seqused)
_copy_1d(self.start_pos, live.start_pos)
```

- 用 live metadata 刷新 cu / kv / seqused / start_pos。
- `actual_q_pa` 同时是 compressor 的输入 cu 和 attention 的 query cu（B=1 时两者一致）。

```python
for ratio in self.ratios:
    _copy_1d(self.positions[ratio], getattr(live, f"positions_cmp_padding_c{ratio}"))
    _copy_1d(self.locs[ratio], None if capture else getattr(live, f"c{ratio}_loc"))
```

- capture：positions 用 seed 的值；loc 不读 seed，保持 0。
- replay：positions 和 loc 都从 live 读。
- `seqused` / `start_pos` / `positions_cmp_padding_*` 直接取属性（动态名字用 `getattr` 但**不给默认值**）：
  eager prefill builder 一定会设置它们，给 `None` 默认只会把契约破坏变成静默填 0。
- 尾部填 0：
  - positions 尾部 0 是合法位置；
  - loc 尾部 0 是保留 slot。

```python
_copy_table(self.swa_page_table, live.swa_page_table, "swa_page_table")
if self.c4_page_table is not None:
    _copy_table(self.c4_page_table, live.c4_page_table, "c4_page_table")
if self.c128_page_table is not None:
    _copy_table(self.c128_page_table, live.c128_page_table, "c128_page_table")
```

- 页表按 live 的 max context 刷新，宽度不够直接报错。
- 清零后再拷，保证尾部列/行不会读到上一轮残留。

```python
self._copy_kernel_metadata(live.kernel_metadata)
if not capture:
    self._refresh_state_tables(backend, forward_batch)
```

- kernel metadata（c1a/c4a/c128a/li_quant）从 live 的 metadata 算子输出拷入固定 buffer。
- reclaim state table 只在 replay 做；capture 保持全 dummy。

### 1.13 `_fill_idle`

```python
self.actual_q.fill_(1)
self.actual_q_pa.zero_()
self.actual_q_pa[-1:].fill_(1)
self.actual_kv.fill_(1)
self.seqused.zero_()
self.start_pos.zero_()
```

- 造一个“1 个 dummy query、0 个有效 token”的合法 batch。
- `actual_q_pa = [0, 1]`：kernel 至少有一行可以处理，不会因为零长度被算子拒绝。
- `seqused = 0`：compressor 不产生有效输出。
- `actual_kv = 1`：key 长度至少 1，page 0 是保留页。

```python
for ratio in self.ratios:
    self.positions[ratio].zero_()
    self.locs[ratio].zero_()
self.swa_page_table.zero_()
if self.c4_page_table is not None: self.c4_page_table.zero_()
if self.c128_page_table is not None: self.c128_page_table.zero_()
for ratio, table in self.state_tables.items():
    table.fill_(self.dummy_state[ratio])
```

- 所有写回都指向保留区；不会碰真实请求的 KV/state。

```python
rebuilt = backend._kernel_metadata_from_parts(
    bs=self.max_reqs,
    actual_seq_lengths_q_pa=self.actual_q_pa,
    actual_seq_lengths_kv=self.actual_kv,
)
self._copy_kernel_metadata(rebuilt)
```

- idle 时 live metadata 是空的（`init_forward_metadata` 提前 return），所以用合成长度重新跑一次 metadata 算子，保证 captured kernel 有合法 metadata。

### 1.14 `_refresh_state_tables`

```python
batch_size = forward_batch.batch_size
if batch_size > self.max_reqs:
    raise RuntimeError(...)
```

- 防御性检查：replay 的请求数不能超过 capture 容量。

```python
for ratio, dst in self.state_tables.items():
    table = backend._build_dsv4_prefill_graph_state_table(
        ratio=ratio,
        req_pool_indices=forward_batch.req_pool_indices,
        start_pos=self.start_pos[:batch_size],
        cu_seqlens=self.actual_q_pa[: batch_size + 1],
        seqused=self.seqused[:batch_size],
        max_input_capacity=self.num_tokens,
    )
```

- 图外用 live 的 req_pool_indices / start_pos / cu / seqused 重建 A3 explicit state table。
- `max_input_capacity=self.num_tokens` 固定表宽为 `history + T_cap`。
- 这个调用会 launch 一些 device op，但在图外，安全。

```python
dst.fill_(self.dummy_state[ratio])
rows, cols = table.shape
if cols > dst.shape[1]: raise ...
dst[:rows, :cols].copy_(table.to(dtype=dst.dtype))
```

- 先全部填 dummy，再把 live 表拷进去。
- 表宽必须一致；不一致直接报错。

### 1.15 `_copy_kernel_metadata`

```python
for key, dst in self.kernel_metadata.items():
    src = source.get(key)
    if src is None:
        dst.zero_()
        continue
    if src.numel() > dst.numel(): raise ...
    dst[: src.numel()].copy_(src.to(dtype=dst.dtype))
    if src.numel() < dst.numel():
        dst[src.numel():].zero_()
```

- 逐 key 拷 metadata 算子输出。
- live 可能没有某个 key（例如模型没有 c128），填 0。
- live 超过 capture 大小直接报错，避免截断。

### 1.16 `_bind`

```python
metadata.actual_seq_lengths_q = self.actual_q
metadata.actual_seq_lengths_q_pa = self.actual_q_pa
metadata.actual_seq_lengths_kv = self.actual_kv
metadata.seqused = self.seqused
metadata.start_pos = self.start_pos
```

- 把固定 buffer 挂到 `ForwardMetadata` 对象上；后续 `forward_compress` / `_forward_npu_fused` 直接读。

```python
for ratio in self.ratios:
    setattr(metadata, f"positions_cmp_padding_c{ratio}", self.positions[ratio])
    setattr(metadata, f"c{ratio}_loc", self.locs[ratio])
```

- 与现有 eager builder 的字段名保持一致。

```python
metadata.swa_page_table = self.swa_page_table
if self.c4_page_table is not None: metadata.c4_page_table = self.c4_page_table
if self.c128_page_table is not None: metadata.c128_page_table = self.c128_page_table
```

- 图内/attention break 都读这些页表。

```python
metadata.kernel_metadata = self.kernel_metadata
metadata.c4_topk_indices = None
metadata.dsv4_explicit_state_block_tables = self.state_tables
metadata.dsv4_max_input_capacity = self.num_tokens
metadata.dsv4_prefill_graph_state = self
```

- `kernel_metadata` 用固定 dict；`forward_compress` 看到 ratio 已存在就不会再惰性建表。
- `c4_topk_indices` 初始为 None；capture 时每层 indexer 会写它，replay 时不再写，
  attention 按 `topk_by_layer` 查本层 buffer。
- `dsv4_max_input_capacity` 固定为 `T_cap`：eager builder 把它当“单请求最大 chunk”，
  `B_cap=1` 时正好等于 bucket 大小。
- 不再写 `dsv4_cycle_state_block_table`：A5 分支已删，A3 下该字段本来就是 None。
- `dsv4_prefill_graph_state` 是外部识别“当前是 captured metadata”的标记。

---

## 2. `ascend_dsv4_backend.py`

### 2.1 imports

- 新增 `Dsv4PrefillGraphMetadata` / `MAX_PREFILL_GRAPH_REQS`。
- 新增 `Backend` / `Phase` / `check_cuda_graph_backend`，用于判断 prefill 是否真的是 breakable。
- 新增 `ForwardMetadata` 用于 hook 返回类型。

### 2.2 indexer 多流

```python
if enable_multi_stream and self._can_use_indexer_multi_stream():
```

- 原来只看 env `SGLANG_NPU_USE_MULTI_STREAM`，prefill 也会走侧流。
- 现在必须调用方显式传 `enable_multi_stream=True`。
- prefill 的 model 路径不传，所以 prefill 强制单流；decode 的 multi-stream 调用仍传 True，行为不变。
- 这样 captured segment 内的 indexer 与 compressor 顺序固定，避免跨流依赖。

### 2.3 `forward_c4_indexer` 的 topk 交接

```python
graph_state = getattr(self.forward_metadata, "dsv4_prefill_graph_state", None)
if graph_state is not None:
    topk_idxs = graph_state.stage_topk(c4_indexer.layer_id, topk_idxs)
else:
    compress_layer_id = ...compress_layer_id
    maybe_capture_indexer_topk(compress_layer_id, topk_idxs)
self.forward_metadata.c4_topk_indices = topk_idxs
```

- eager 路径不变：仍然调用 `maybe_capture_indexer_topk`。
- captured 路径：
  - 把每层 topk 输出登记到 `topk_by_layer[layer_id]`；
  - 跳过 `maybe_capture_indexer_topk`，因为 replay 时 Python 不执行，喂给 capturer 的数据会过期。
- 共享属性 `c4_topk_indices` 仍设置，作为兜底；真正消费按 layer_id 查。

### 2.4 打开契约 + 支持性检查

```python
self.use_captured_forward_metadata_for_breakable_cuda_graph = (
    self._supports_dsv4_prefill_graph_capture()
)
if self.use_captured_forward_metadata_for_breakable_cuda_graph:
    self.prefill_cuda_graph_max_batch_size = MAX_PREFILL_GRAPH_REQS
    logger.info(...)
```

- 只有当前配置是 `Phase.PREFILL == breakable` 时才打开。
- `prefill_cuda_graph_max_batch_size` 是 runner 的 admission 上限，也是 DP 投票的依据。
- 日志打印 `pool_tokens` / `req_slots`，方便对比 KV 池大小。

```python
def _supports_dsv4_prefill_graph_capture(self) -> bool:
    if not check_cuda_graph_backend(Phase.PREFILL, Backend.BREAKABLE):
        return False
    if self.is_draft_worker:
        return False
    parallel = get_parallel()
    if is_npu_arch35() or parallel.attn_cp_size != 1 or parallel.attn_dcp_size != 1:
        raise ValueError(...)
    return True
```

- draft worker：不打开；draft 的 prefill graph 由 speculative runner 处理。
- A5：当前实现只验证了 A3，直接报错。
- CP/DCP：metadata 的 local/full 切换未验证，直接报错。
- 注意：这里**没有**检查 `attn_tp_sequence_sharded`；它只是 MLP/DP gather bookkeeping，不代表行轴被切（见 `eb05020fd5`）。

### 2.5 capture hook

```python
if forward_batch.batch_size > MAX_PREFILL_GRAPH_REQS:
    raise RuntimeError(...)
```

- capture batch 的请求数必须能被固定槽位装下。
- 正常 prefill bucket ≤ context，`capture_prepare` 只会合成 1 个请求，所以不触发。
- bucket > context 或 `--cuda-graph-prefill-max-context` 很小时会触发；此时应该调大 `MAX_PREFILL_GRAPH_REQS` 或减小 bucket。

```python
if forward_batch.can_run_tbo:
    raise RuntimeError(...)
```

- TBO 会把 batch 拆成 child，每个 child 需要独立 metadata；当前只支持一份，所以拒绝。
- TBO 默认关闭。

```python
self.init_forward_metadata(forward_batch)
# allocate() refreshes the eager metadata in place, so this object is
# also the one the capture forward reads.
return Dsv4PrefillGraphMetadata.allocate(
    backend=self,
    seed=self.forward_metadata,
    num_tokens=int(forward_batch.positions.shape[0]),
).metadata
```

- 先跑现有 eager builder，得到 capture 的“数据来源”。
- `num_tokens` 是 bucket 行数 `T_cap`。
- `allocate` 分配固定 buffer 并原地绑定到 seed，所以不需要再给 `self.forward_metadata` 赋值。
- 返回的 metadata 被 runner 存进 `attn_metadata_buffers[shape_key]`。

### 2.6 replay hook

```python
state = getattr(capture_metadata, "dsv4_prefill_graph_state", None)
if state is None:
    raise RuntimeError(...)
```

- runner 传回的 captured metadata 必须带状态对象。

```python
if forward_batch.can_run_tbo:
    raise RuntimeError(...)
```

- 必须读 live batch：`static_forward_batch` 是 runner 新建的，不带 `tbo_split_seq_index`，
  在它上面判断 `can_run_tbo` 恒为 False。
- capture 时的合成 batch 也不带 TBO，所以这个检查是 TBO 的唯一兜底。

```python
self.init_forward_metadata(forward_batch)
state.refresh(backend=self, live=self.forward_metadata, forward_batch=forward_batch)
```

- **关键**：用 live `forward_batch`（不是 `static_forward_batch`）重建 live metadata。
  - `static_forward_batch` 的 positions 是 bucket 长度，不含 live prefix/extend 的真实口径；
  - live batch 才有 `out_cache_loc_dsv4`、真实 `extend_prefix_lens`、`seq_lens`。
- `state.refresh` 把 live 值原地拷进固定 buffer。

```python
self.forward_metadata = capture_metadata
```

- 最后把 `forward_metadata` 指回 captured 对象：
  - captured segment 里的 kernel 读固定地址；
  - eager attention break 也读这个对象，看到同样的刷新值。

### 2.7 `_build_dsv4_prefill_graph_state_table`

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

- 复用现有 A3 state table 构造器。
- `coff`：ratio 4 有 overlap，coff=2；ratio 128 是 1。
- 传入固定 `max_input_capacity=T_cap`，保证表宽与 capture 时一致。
- 这个函数在图外调用；返回的临时表再拷进固定表。

### 2.8 `_forward_compressed` 的 topk + 错误上下文

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

- `topk_by_layer`：按 `layer_id` 直接下标取本层 indexer 的输出。原来用 `.get(id, fm.c4_topk_indices)`
  兜底，但 replay 时 `fm.c4_topk_indices` 已被 `_bind` 置 None，兜底只会把“缺层”变成更远的
  `AttributeError`；缺层现在直接抛 `KeyError`。
- `real_tokens`：live 全局真实 token 数（host int，不需要 device sync）。
- `topk[:real_tokens]`：
  - captured indexer 输出按 `T_cap` 行分配；
  - eager attention break 的 `q` 已经切到 live 行；
  - 不切 topk 会把 padding 行（capture 旧值/越界值）传给 sparse attention，导致 `aclnnSparseAttnSharedkv failed`。
- 切片只发生在 captured 路径；eager 路径的 topk 本来就是 live 行数。

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

- 算子自身的报错（`aclnnSparseAttnSharedkv failed`）不带 layer 和 sparse metadata，
  这里补上 `layer / ratio / q / cmp_kv / topk` 的形状用于定位。
- 只打 shape：`tolist()` / `min()` 会在这条错误路径上做 device→host 同步，
  且 metadata 为 None 时会二次报错、盖掉原始异常。

---

## 3. 单测 `test_dsv4_prefill_graph_metadata.py`

| case | 守什么 |
| --- | --- |
| `test_capture_allocates_fixed_isolated_buffers` | capture 时 `c4_loc` 全 0、state table 全 dummy、页表宽度口径（c4 用 SWA page 数，不是块数） |
| `test_replay_refreshes_in_place` | replay 刷新后 tensor **对象 id 不变**，值更新；这是 captured kernel 地址稳定的核心契约 |
| `test_idle_replay_stays_legal` | idle 时 `q_pa=[0,1]`、`kv=[1]`、`seqused=0`、state 全 dummy，避免非法零长度启动 |
| `test_topk_is_staged_per_layer` | 每层 topk 独立登记；`stage_topk` 不碰共享属性，消费方必须按 `layer_id` 查 |

测试用 `SimpleNamespace` 模拟 backend/seed，不需要 NPU。

---

## 4. 必须成立的不变量

1. **地址稳定**：capture 之后，`dsv4_prefill_graph_state` 里的 tensor 对象不能替换，只能 `copy_`/`fill_`。
2. **capture 隔离**：
   - 压缩 KV 写 slot 0（paged allocator 保留页）；
   - state 写 `dummy_state_loc`；
   - 无效行始终写保留区。
3. **两套口径分开**：
   - compressor 用 live cu + seqused；
   - attention break 用 live q_pa；
   - `C_cap(r)` 是容量，不是 `N_r`。
4. **页表宽度**：SWA/c4/c128 都按 max context 固定，不能按 capture 时的短 prefix 固定。
5. **topk 按层保存**：`c4_topk_indices` 只是兜底，真正消费按 `layer_id`。
6. **B_cap=1**：多请求 replay 会被 runner admission 拒绝并回退 eager。

## 5. 当前限制

- 只支持 A3；A5/CP/DCP 在 backend init 报错。
- TBO 在 capture 时报错。
- `MAX_PREFILL_GRAPH_REQS=1`；多请求需要先验证零长度 request slot。
- attention 仍是每层 eager break；4096 token 一块约 1.9s，图只省 launch 开销，
  实测 eager / breakable 的 prefill 单步差异在 1–3% 以内。
- 运行期没有常驻诊断日志；临时 profiling 代码不随该提交引入。
