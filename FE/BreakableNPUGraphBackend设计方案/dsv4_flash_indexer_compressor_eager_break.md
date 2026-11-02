# DSV4 Flash 在 NPU breakable prefill graph 下 capture 崩溃：分析与方案 A / B

- 代码基线：sglang `5257b6d5`（[Feat] [NPU] prefill: breakable NPU graph backend (P1 wiring)）。第 2 ~ 4 节的行号都指这个提交。服务器 traceback 的行号与它有偏差（如 `deepseek_v4.py:2303` 对应 `2797`），函数与逻辑一致
- 实现：采用方案 B，提交 `sglang@d6c477d013`，位于 `support_breakable_graph_back` 分支，在 [GLM-5.2 DSA indexer 断点](glm52_dsa_indexer_eager_break.md)（`6ceb3d9fa7`）之后
- 状态：CPU 单测通过，**尚未在 NPU 上运行**。方案 A 未实现，保留作对比

## 1. 现象

`--cuda-graph-backend-prefill breakable` 启动 DSV4 Flash（DP16/TP16/EP16），capture 第一个 shape（2048）时所有 rank 报错：

```
set_index_k_scale: loc_long = loc.view(-1, 1).long()
AttributeError: 'NoneType' object has no attribute 'view'
```

调用链：`DeepseekV4Attention._forward_prepare` → `Indexer.forward` → `forward_c4_indexer` → `compressor` → `forward_compress` → `_compressor_epilog_npu` → `set_compress_buffer(loc=None)`。

## 2. capture 崩溃的根因

崩溃发生在 `BreakableCudaGraphBackend.capture_one` 的 warmup 阶段（`breakable_cuda_graph_backend.py:141`，还没开始录图），所以与 NPUGraph 录制机制无关，是 metadata 缺字段：

1. **capture 用的 dummy batch 没有 `out_cache_loc_dsv4`。** `PrefillCudaGraphRunner.capture_prepare`（`prefill_cuda_graph_runner.py:1617`）构造 `ForwardBatch` 时不设这个字段；它只由调度器给真实 batch 分配 KV 时写入（`mem_cache/static_kv_storage.py:223/267`）。
2. **NPU DSV4 后端走的是 eager metadata 初始化。** `DeepseekV4AscendAttnBackend` 继承 `AscendAttnBackend`，没有继承 CUDA 的 `DeepseekV4AttnBackend`，`use_captured_forward_metadata_for_breakable_cuda_graph` 是基类默认的 `False`（`base_attn_backend.py:228`）。因此 capture 时 runner 调普通 `init_forward_metadata`（`prefill_cuda_graph_runner.py:1264`）。
3. **NPU prefill compress metadata 只从 bundle 取 loc。** `_build_npu_compress_metadata_prefill`（`ascend_dsv4_backend.py:318-331`）在 bundle 为 `None` 时把 `c4_loc` / `c128_loc` 设为 `None`。函数开头的注释写着 "eager-only: prefill is never graph-captured"（`:245`），这次 commit 打破了这个前提。
4. **`forward_compress` 没拦住 `None`。** `loc is None` 时跳过长度检查；`graph_mode=False` 且 `cmp_kv.shape[0] > 0`，照常进入 epilog，最终 `loc.view` 崩溃。

## 3. replay 的问题：只补 loc 不够

补上 loc 后 capture 能过，但 replay 结果会错。原因是 NPU 的 indexer 和 compressor 没有 `eager_on_graph` 断点，会被录进 segment。

### 3.1 一层 DSV4 attention 在 NPU prefill BCG 下的执行位置

| 步骤 | 代码位置 | 修复前状态 |
| --- | --- | --- |
| wq_a / wkv / norm / RoPE | `deepseek_v4.py:2432` 起 NPU 分支 | 录进 segment，读静态 `positions` slot，replay 正确 |
| SWA `store_cache` | `ascend_dsv4_backend.py:2224` | 录进 segment，读静态 `out_cache_loc` slot 经 `translate_loc_from_full_to_swa` 映射，replay 正确 |
| c4 indexer（含 indexer 自带的 compressor） | `self.indexer(...)` → `forward_c4_indexer` | **录进 segment，有问题** |
| 主 compressor（c4 / c128 层） | `attn_backend.forward_core_compressor` | **录进 segment，有问题** |
| sparse attention | `bcg_deepseek_v4_attention_with_output`（`deepseek_v4.py:840`） | 已经是 eager 断点，replay 读当次 metadata |

在 NPU 上 `is_unified_kv_triton()` 恒为 False（只在 HIP 上为 True），mega attention 只在 CUDA 上启用，所以 prefill attention 一定走上表最后一行的断点。

### 3.2 indexer / compressor 录进图后的后果

replay 时 runner 对 live batch 重新调用 `init_forward_metadata`（`prefill_cuda_graph_runner.py:1320`），生成一整套**新**的 metadata 张量，capture 时的旧张量随之被释放。而录进图的 kernel 仍引用旧地址：

- **读到已释放的显存。** `forward_compress` 读 `positions_cmp_padding_c*`、`start_pos`、`seqused`、`actual_seq_lengths_q_pa`、state block table、`c*_loc`；lightning indexer 读 `actual_seq_lengths_q/kv`、`c4_page_table`、`kernel_metadata["li_quant_metadata"]`。这些张量都分配在图的 pool 之外，replay 时已经释放，可能已被其他分配复用。这类错误不报错，只会让结果出错。
- **capture 时的 host 值被固化。** 例如 `dsv4_max_input_capacity` 来自 `cu.cpu().tolist()`，决定 state block table 的形状；`cmp_kv[: loc.numel()]` 的截断长度；`if loc is not None` 等 Python 分支。这些都按 capture 时的 dummy batch 定死了。
- **c4 层的 attention 拿不到 topk。** indexer 把结果赋给 `self.forward_metadata.c4_topk_indices`（`ascend_dsv4_backend.py:918`），这是 Python 属性赋值，只在 capture 时执行一次，而且赋给的是 capture 时的 fm 对象。replay 时 attention 断点读的是新 fm，上面没有 `c4_topk_indices`。预期会在 `_forward_compressed` 的 `topk.view`（`:2194`）处报错。

结论：indexer 和 compressor 必须做成 eager 断点。这两步依赖 host 侧的 metadata，本来也无法放进图里。CUDA 侧有同样的先例：ratio 1/2 的 indexer 和 compressor 由 `bcg_deepseek_v4_low_ratio_sources`（`deepseek_v4.py:915`）作为断点执行，注释写的是 "The compressor and prefill indexer sync with the host, like the attention."。

### 3.3 断点能保证 replay 正确的依据

- **capture pass 不执行断点函数体。** 它只结束当前 segment，并把 `functools.partial(run, *captured_args)` 记进 `_break_fns`；replay 时直接调用这个 partial（`breakable_cuda_graph.py:306-332`）。所以 replay 每次都在 eager 下执行 indexer 和 compressor，读的是 `_prepare_forward_metadata_for_replay` 刚为 live batch 建好的 metadata，走的就是已经验证过的 eager prefill 路径。chunked prefill（prefix > 0）的 loc 也照常从 bundle 取。
- **断点函数体内部取 batch 的方式与 attention 断点相同。** 用 `get_tc_piecewise_forward_context().forward_batch` 取 batch：warmup 时拿到的是 dummy batch，replay 时是 `static_forward_batch`（`prefill_cuda_graph_runner.py:2268`）。再用 `global_num_token_non_padded_cpu` 把 `x` / `q_lora` 切到真实行数，`deepseek_v4_attention_with_output`（`deepseek_v4.py:791`）也是这样做的。
- **x 和 q_lora 的地址在 replay 时有效。** 断点的输入是上一个 segment 在图 pool 里产出的张量，以 weak ref 保存，由 pool 的 use-count 保活（`breakable_cuda_graph.py:311-315`）；每次 replay 时 segment 会在同一地址重写它们。
- **断点之间的顺序保证 attention 读到当次的 topk。** indexer 断点写 `fm.c4_topk_indices`，attention 断点随后读，两者都在 eager 下执行，读的是同一个 live fm。

还要处理一个细节：indexer 内部的 `_compute_q_npu` 读 `forward_batch.positions` 算 RoPE，而 `static_forward_batch.positions` 按 bucket 补齐了长度。断点里 q 已经切到真实行数，positions 也要切成同样长度，具体做法见 5.1。

## 4. 方案 A：最小修复，只改新增文件（未实现）

**改动范围**：只改 `hardware_backend/npu/graph_runner/breakable_npu_graph_backend.py`（本次 commit 新增的文件）。其他文件一行不动，公共代码零改动。

**思路**：在 `BreakableNPUGraphBackend` 里覆写 `capture_one`，做两件事：

1. 进入 warmup 之前，为 capture 用的 metadata 补上占位 loc。runner 在调用 `capture_one` 之前已经完成 metadata 初始化（`prefill_cuda_graph_runner.py:1813`），之后在 warmup 与 capture 之间不会再重建。
2. 在 capture 期间，把 attention 后端实例上的 `forward_c4_indexer` 和 `forward_core_compressor` 临时替换成 `eager_on_graph` 断点，`finally` 里恢复。替换只需要在 capture 期间生效：capture pass 记下的断点函数在 replay 时被直接调用，不再经过这两个实例属性。

```python
from contextlib import contextmanager

import torch

from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.breakable_cuda_graph import (
    eager_on_graph,
)
from sglang.srt.model_executor.runner_backend_utils.tc_piecewise_cuda_graph import (
    get_tc_piecewise_forward_context,
)


def _dsv4_backend_cls():
    # Lazy: the DSV4 backend module imports torch_npu kernels at import time.
    from sglang.srt.hardware_backend.npu.attention.ascend_dsv4_backend import (
        DeepseekV4AscendAttnBackend,
    )

    return DeepseekV4AscendAttnBackend


@contextmanager
def _live_positions(forward_batch, num_tokens: int):
    # The static batch keeps bucket-sized positions; the indexer's RoPE must
    # match the live rows it is given.
    positions = forward_batch.positions
    forward_batch.positions = positions[:num_tokens]
    try:
        yield
    finally:
        forward_batch.positions = positions


def _dsv4_c4_indexer(backend, x, q_lora, c4_indexer, skip_compressor) -> None:
    # The indexer and its compressor read host-built metadata, like the attention.
    forward_batch = get_tc_piecewise_forward_context().forward_batch
    num_tokens = forward_batch.global_num_token_non_padded_cpu
    if num_tokens == 0:
        return
    with _live_positions(forward_batch, num_tokens):
        _dsv4_backend_cls().forward_c4_indexer(
            backend,
            x=x[:num_tokens],
            q_lora=q_lora[:num_tokens],
            forward_batch=forward_batch,
            c4_indexer=c4_indexer,
            skip_compressor=skip_compressor,
        )


def _dsv4_core_compressor(backend, x, layer_id, compressor) -> None:
    forward_batch = get_tc_piecewise_forward_context().forward_batch
    num_tokens = forward_batch.global_num_token_non_padded_cpu
    if num_tokens == 0:
        return
    _dsv4_backend_cls().forward_core_compressor(
        backend, x[:num_tokens], forward_batch, layer_id, compressor
    )


_bcg_dsv4_c4_indexer = eager_on_graph(_dsv4_c4_indexer)
_bcg_dsv4_core_compressor = eager_on_graph(_dsv4_core_compressor)


@contextmanager
def _dsv4_sources_as_breaks(backend):
    # Only the capture pass needs the routing: replay calls the recorded
    # break functions directly.
    def forward_c4_indexer(
        *, x, q_lora, forward_batch, c4_indexer=None, skip_compressor=False, **_
    ):
        _bcg_dsv4_c4_indexer(backend, x, q_lora, c4_indexer, skip_compressor)

    def forward_core_compressor(x, forward_batch, layer_id, compressor):
        _bcg_dsv4_core_compressor(backend, x, layer_id, compressor)

    backend.forward_c4_indexer = forward_c4_indexer
    backend.forward_core_compressor = forward_core_compressor
    try:
        yield
    finally:
        del backend.forward_c4_indexer
        del backend.forward_core_compressor


def _fill_capture_compress_locs(backend) -> None:
    # The capture batch carries no out_cache_loc_dsv4 bundle, so the eager
    # prefill metadata leaves c4/c128 loc None. Route the warmup's compressed
    # writes to padding slot 0, as decode graph capture does.
    fm = backend.forward_metadata
    seq_lens = fm.actual_seq_lengths_q_pa_cpu[1:] - fm.actual_seq_lengths_q_pa_cpu[:-1]
    for ratio in backend._dsv4_unique_compress_ratios:
        if ratio not in (4, 128) or getattr(fm, f"c{ratio}_loc", None) is not None:
            continue
        # Capture requests have no prefix: each completes seq_len // ratio blocks.
        num_blocks = int((seq_lens // ratio).sum())
        setattr(
            fm,
            f"c{ratio}_loc",
            torch.zeros(num_blocks, dtype=torch.int32, device=backend.device),
        )


class BreakableNPUGraphBackend(BreakableCudaGraphBackend):
    ...

    def capture_one(
        self, shape_key, forward_fn, capture_inputs=None, post_warmup_hook=None
    ) -> None:
        backend = self._model_runner.attn_backend
        if not isinstance(backend, _dsv4_backend_cls()):
            super().capture_one(
                shape_key,
                forward_fn,
                capture_inputs=capture_inputs,
                post_warmup_hook=post_warmup_hook,
            )
            return
        _fill_capture_compress_locs(backend)
        with _dsv4_sources_as_breaks(backend):
            super().capture_one(
                shape_key,
                forward_fn,
                capture_inputs=capture_inputs,
                post_warmup_hook=post_warmup_hook,
            )
```

**占位 loc 为什么这样取：**

- **长度。** 每个请求取 `seq_len // ratio`，与真实 prefill 时 bundle 给出的块数一致（`ascend_dsv4_backend.py:274-282`，prefix 为 0）。这个长度不会超过 `cmp_kv` 的行数，会走 `forward_compress` 里已有的截断分支（`:520`），不触发 mismatch 报错。bucket 为 16 时 c128 的块数为 0，cmp_kv 截断成 0 行后跳过 epilog。
- **值与 dtype。** 取全 0、int32。decode graph capture 的 `c4_loc` 也是 `torch.zeros`（`:1358`）；slot 0 不会分给真实请求，依据见 5.2。int32 与 eager prefill 从 bundle 转出来的类型一致（`:329`）。
- **生效范围。** 占位 loc 只在 warmup 时被写入：capture pass 不执行断点函数体，replay 用的是 live bundle。

**代价：**

- 后端实例上的方法在 capture 期间被临时替换。这是隐式耦合：将来模型如果改用别的后端入口调用 indexer 或 compressor，这里不会报错，而是悄悄把它录进图。
- graph backend 需要知道 DSV4 metadata 的字段名，分层不干净。
- CP 路径用的 `forward_indexer_compressor` 别名不会被替换（CP 不在本次范围内）。

## 5. 方案 B：按职责归属放代码（已实现）

**原则**：DSV4 相关的知识放在 DSV4 NPU 后端里，graph backend 保持与模型无关；不做运行时的方法替换；公共代码零改动。

**改动**（`sglang@d6c477d013`）：

| 文件 | 性质 | 改动 |
| --- | --- | --- |
| `hardware_backend/npu/attention/dsv4_prefill_graph.py` | **新增**，69 行 | 两个断点函数和 `in_prefill_graph` 判定 |
| `hardware_backend/npu/attention/ascend_dsv4_backend.py` | NPU 专属，+49 / -2 | 两个入口分发到断点；indexer 函数体拆出；capture 时补占位 loc；修正 "eager-only" 注释 |
| `test/registered/unit/npu/attention/test_npu_ascend_dsv4_backend.py` | 已有测试文件，+118 | 3 个用例，见 5.4 |
| `breakable_npu_graph_backend.py` 和公共代码（`models/`、`model_executor/`、`layers/`） | — | 不改 |

### 5.1 断点：`dsv4_prefill_graph.py`

```python
def in_prefill_graph(forward_batch: ForwardBatch) -> bool:
    # Same predicate as the attention break in models/deepseek_v4.py.
    return forward_batch.forward_mode.is_extend() and is_in_breakable_cuda_graph()


@eager_on_graph
def bcg_dsv4_c4_indexer(
    *,
    backend: C4IndexerAscendBackendMixin,
    x: torch.Tensor,
    q_lora: torch.Tensor,
    c4_indexer,
    skip_compressor: bool,
) -> None:
    forward_batch = get_tc_piecewise_forward_context().forward_batch
    num_tokens = forward_batch.global_num_token_non_padded_cpu
    if num_tokens == 0:
        return
    # The static batch keeps bucket-sized positions; the indexer RoPE reads one
    # per live row.
    live_batch = copy.copy(forward_batch)
    live_batch.positions = forward_batch.positions[:num_tokens]
    backend.forward_c4_indexer_eager(
        x=x[:num_tokens],
        q_lora=q_lora[:num_tokens],
        forward_batch=live_batch,
        c4_indexer=c4_indexer,
        skip_compressor=skip_compressor,
    )


@eager_on_graph
def bcg_dsv4_compressor(*, x: torch.Tensor, compressor) -> None:
    forward_batch = get_tc_piecewise_forward_context().forward_batch
    num_tokens = forward_batch.global_num_token_non_padded_cpu
    if num_tokens == 0:
        return
    compressor(x[:num_tokens], forward_batch)
```

与初稿不同的两处：

- **positions 的切法。** 初稿用 context manager 临时改写 replay batch 的 `positions`，`finally` 里恢复。实现改成在 batch 的浅拷贝上切片，replay batch 本身不被修改，后面的 attention 断点看到的仍是 bucket 长度的 positions。浅拷贝与 runner 构造 `padded_view` / `tail_batch` 的写法相同（`prefill_cuda_graph_runner.py:1299/2245`）。indexer 写结果用的是后端上的 `forward_metadata`，不写 batch，所以浅拷贝不会丢结果；RoPE 的 per-forward 缓存按 positions 对象匹配，切片后会按真实 positions 重算。
- **断点参数改为 keyword-only**，按仓库代码规范（2 个以上参数用关键字传）。

### 5.2 后端入口：`ascend_dsv4_backend.py`

两个入口在 prefill BCG 下改走断点。indexer 原来的函数体拆成 `forward_c4_indexer_eager`，断点调用的是它，不会再走回入口导致递归（warmup 和 replay 执行断点函数体时 `is_in_breakable_cuda_graph()` 仍为 True）。下面的签名省略了类型标注：

```python
def forward_core_compressor(self, x, forward_batch, layer_id, compressor) -> None:
    if forward_batch.forward_mode.is_idle():
        return
    if in_prefill_graph(forward_batch):
        bcg_dsv4_compressor(x=x, compressor=compressor)
        return
    compressor(x, forward_batch)


def forward_c4_indexer(self, *, x, q_lora, forward_batch, c4_indexer=None,
                       alt_streams=None, enable_multi_stream=False,
                       q_lora_ready=None, skip_compressor=False) -> None:
    if forward_batch.forward_mode.is_idle():
        return
    if in_prefill_graph(forward_batch):
        # A break runs on the replay stream after the preceding segment,
        # so q_lora is ready without the capture-time event.
        bcg_dsv4_c4_indexer(
            backend=self,
            x=x,
            q_lora=q_lora,
            c4_indexer=c4_indexer,
            skip_compressor=skip_compressor,
        )
        return
    self.forward_c4_indexer_eager(
        x=x,
        q_lora=q_lora,
        forward_batch=forward_batch,
        c4_indexer=c4_indexer,
        q_lora_ready=q_lora_ready,
        skip_compressor=skip_compressor,
    )
```

`forward_indexer_compressor` 是 `forward_core_compressor` 的类级别名，也会走断点。

`_build_npu_compress_metadata_prefill` 在 bundle 缺失时，只在 prefill graph capture 期间生成占位 loc，其他情况保持原样：

```python
            if bundle_loc is None and get_flags().dp.capturing_prefill_graph:
                # The capture batch has no allocation bundle; route its compressed
                # writes to padding slot 0, as decode graph capture does.
                bundle_loc = torch.zeros(
                    sum(blocks.numel() for blocks in ratio_lists[ratio]),
                    dtype=torch.int32,
                    device=device,
                )
```

- **块数**直接用同一函数里刚算好的 `ratio_lists`，与 `positions_cmp_padding_c*` 一定一致，不用另写公式。bucket 为 16 时 c128 块数为 0，cmp_kv 截断成 0 行后跳过 epilog。
- **判断条件用 `capturing_prefill_graph`**，而不是 `is_in_breakable_cuda_graph()`。后者在 replay 时也是 True；如果 replay 时意外缺 bundle，占位 loc 会把真实请求的数据悄悄写进 slot 0。用 capture 标志能保证 replay 时缺 bundle 依旧报错。这个标志虽然挂在 `dp` 下，但 `PrefillCudaGraphRunner._capture` 不论是否开 DP 都会设置它（`prefill_cuda_graph_runner.py:1711`），`deepseek_v4_replay_tail.py:748` 也把它当通用的 capture 标志来用。它在 `self.warmup()` 之后才置位，不影响 capture 前的 kernel warmup。
- **slot 0 不会分给真实请求**（初稿列为待确认，已从代码确认）：
  - DSV4 在 NPU 上的 `page_size` 固定为 128（`arg_groups/model_overrides/deepseek_v4.py:57`）。
  - full KV 和 c128 的分配器都是 `NPUPagedTokenToKVPoolAllocator`（`mem_cache/allocator/swa.py:94-98`、`dsv4_allocator.py:128-140`），它继承的 `clear()` 让空闲页从 1 开始，注释写明 slot 0 留给 padding 写入（`mem_cache/allocator/paged.py:395-399`）。
  - c4 loc 由 full loc 推出（`full_loc // 4`，`dsv4_allocator.py:180-181`）。full page 0 的 slot 0 ~ 127 不会分配，对应的 c4 slot 0 ~ 31 也就不会分给真实请求。
- `:245` 的注释改为说明：这个函数在 prefill graph 下也会被调用，但它产出的 metadata 只被 eager 断点读取。

### 5.3 为什么不采用 CUDA 那套 captured-metadata 契约

CUDA 的 DSV4 后端打开了 `use_captured_forward_metadata_for_breakable_cuda_graph`。capture 时它构建一份地址稳定的 metadata，replay 时原地刷新（`deepseek_v4_backend.py:3167/3270`）。这套契约的前提是：segment 里**有 kernel 读这份 metadata**。NPU 的 compressor、lightning indexer 和 sparse attention 都要依赖 host 侧的值（`kernel_metadata` 由 host 算子生成，state block table 的宽度来自 host 上的长度），本来就不能进图。

所以在 NPU 上照搬这套契约，只是把 metadata 改写成可原地刷新的 buffer。工作量与 decode graph 的 `_init_dsv4_graph_metadata` / `_refresh_graph_*` 相当，有数百行，换不来任何算子进图。只有将来 compressor 能拿到纯 device 侧的接口、可以进图时，这套契约才值得做。

### 5.4 测试与 CPU 验证结果

新增用例：

- `test_capture_routes_every_completed_block_to_slot_zero`：两个请求，长度 130 和 7。capture 期间缺 bundle 时，c4 占位 33 个（130 // 4 + 7 // 4），c128 占位 1 个，全 0，int32。
- `test_missing_bundle_outside_capture_stays_none`：不在 capture 期间时缺 bundle，loc 仍为 `None`。这个用例防止占位逻辑在 replay 或 eager 时生效、把真实数据写进 slot 0。
- `test_sources_run_at_replay_on_live_rows`：用真实的 `eager_on_graph` 和 `BreakableCUDAGraph` 走一遍 capture 和 replay，图类和设备模块用 mock。capture 期间 indexer 函数体和 compressor 都不执行，只记下 2 个断点。replay 时换成 8 行 bucket、5 个真实行、positions 从 100 开始的 batch：indexer 拿到前 5 行的 x / q_lora 和 positions 100 ~ 104，compressor 拿到前 5 行；replay batch 本身的 positions 仍是 8 行。

结果（`sglang-npu-dev` 容器，CPU torch，无 `torch_npu`）：

- `test_npu_ascend_dsv4_backend.py` 共 67 项全部通过。rebase 到 `6ceb3d9fa7` 后，连同 `test_npu_dsa_indexer_break.py` 共 74 项通过。
- 把新用例放到修复前的 `ascend_dsv4_backend.py` 上运行，2 项失败，与线上一致：占位用例报 `c4_loc` 为 `None`；断点用例报 indexer 函数体在 capture 期间被直接执行。负向分支用例在修复前后都通过，这是预期的。
- isort 7.0.0 和 ruff 0.15.1（与 pre-commit 锁定版本一致）检查三个文件，均通过。

已存在的失败，与本次改动无关：

- `test_breakable_npu_graph_backend.py::test_capture_splits_at_breaks_and_replays_in_order` 在 `5257b6d5` 上就失败：测试给 `BreakableCUDAGraphCapture` 传了它不接受的 `barrier_fn` 参数。
- 一次性收集整个 `test/registered/unit/npu` 时，`test_npu_cudagraph_backend.py` 和 `test_npu_graph_runner.py` 报 `cannot import name 'get_draft_input_from_target_hidden_dim' from 'sglang.srt.speculative.eagle_utils' (unknown location)`。在干净的 `5257b6d5` 上同样出现，两个文件单独运行都能通过。`unknown location` 说明导入到的是 `sys.modules` 里没有源文件的 stub，很可能是 `test_npu_ascend_dsv4_backend.py` 放进去的那个，它只定义了 `per_step_draft_out_cache_loc`。

## 6. A / B 对比

| 维度 | 方案 A | 方案 B |
| --- | --- | --- |
| 状态 | 未实现 | 已提交 `d6c477d013`，未上设备 |
| 改动文件 | 1 个（本次新增的文件） | 1 个新增 + 1 个 NPU 专属文件 |
| 公共代码 | 不改 | 不改 |
| 断点机制 | `eager_on_graph`，capture 期间临时替换实例方法 | `eager_on_graph`，后端入口显式分发 |
| DSV4 相关知识放在哪 | graph backend（分层不干净） | DSV4 NPU 后端 |
| 占位 loc 的块数 | 在外部用 `cu` 重新推算 | 直接用函数内的 `ratio_lists` |
| 隐式耦合 | 依赖模型只经这两个入口调用 indexer / compressor | 无 |
| 对 eager prefill 的影响 | 无（替换只在 capture 期间生效） | 每次调用多一次判定 |
| replay 正确性 | 两者相同：indexer 和 compressor 在 eager 下读当次 metadata | 同左 |
| 代码量 | 约 90 行（估计） | 新增文件 69 行，后端 +49 / -2，测试 +118 |

实现采用 B。A 的主要优势是只动一个文件、回退简单；B 是单个提交，回退只需 revert `d6c477d013`，这点优势已不重要。

## 7. NPU 验证计划

CPU 侧结果见 5.4。以下需要在 NPU 上进行：

1. **启动**：8 个 bucket 都能 capture 成功，日志里没有 `NoneType` 报错或 capture_end 报错。
2. **replay 对齐**：同一批 prompt 分别在 `--cuda-graph-backend-prefill breakable` 和 `disabled` 下运行，greedy 解码，比较输出 token，并抽样比较首 token 的 logits。用例需要覆盖：
   - 长度不是 4 或 128 整数倍的请求，以及多请求混合；
   - 真实 token 数小于 bucket，以覆盖 padding 行；
   - 有 prefix 的 chunked prefill；
   - 同一 bucket 连续 replay 不同 batch，排查读到上一次残留数据的问题。
3. **精度**：两种设置都用 sgl-eval 跑 gsm8k，采样参数取模型推荐值；汇报分数时注明所用设置。
4. **DP idle rank**：某个 DP rank 没有请求时（`global_num_token_non_padded_cpu == 0`），replay 能走通，且不写 KV。
5. **性能**：每个 c4 层多出 2 个断点，c128 层多出 1 个。indexer 断点和主 compressor 断点之间没有被录制的算子，会产生空 segment。[v2_spike_two_segments.py](v2_spike_two_segments.py) 的 V6 项验证过空 segment 可以录制和 replay，但每次 replay 的开销需要用 profiling 确认。如果开销明显，后续可以在模型层把 indexer 和 compressor 合成一个断点，仿照 `bcg_deepseek_v4_low_ratio_sources`，代价是要改共享的模型文件。

## 8. 未覆盖的风险

- **`SGLANG_NPU_USE_MULTI_STREAM=1`**：打开后，`deepseek_v4.py:2621` 的 `enable_multi_stream` 在 prefill BCG 下为真，attention 会进入 `_forward_prepare_multi_stream_npu`。这条路径先用 `wait_stream` 把 KV 和 Q 分到侧流，BCG 的 hook（`breakable_cuda_graph.py:126`）能跟踪这次分流，indexer 断点结束 segment 时会自动 join（`:469-478`）；indexer 自带的多流准备现在只在断点里 eager 执行。代码上看能走通，但没有在设备上验证。建议先关掉验证，通过后再单独打开测一次。官方部署文档（`docs/docs/hardware-platforms/ascend-npus/model-deployment/best-practices/deepseek_v4_flash.mdx`）只在 decode 节点设置了这个开关。
- **attn-TP 序列切分**：断点按 `global_num_token_non_padded_cpu` 切行，与现有 attention 断点一致，前提是 `attn_tp_size == 1`（本次部署 DP16 满足）。
- **CP 和 MoE**：CP、NPU MoE 和 DP gather 在 BCG 下的行为没有分析。DSV4 不在 `PREFILL_CP_BCG_SUPPORTED_COMBINATIONS` 里（`layers/cp/bcg/__init__.py:21-26`），CP 的 BCG capture 不会为它打开；CP batch 是否会去 replay 非 CP 的图没有分析。这些路径里如果有 host 同步，capture 会直接报错，不会悄悄算错。
