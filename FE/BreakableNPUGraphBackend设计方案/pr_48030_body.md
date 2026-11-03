<!-- PR body for sgl-project/sglang#48030
     [Refactor] [NPU] graph: Enable breakable prefill graphs on Ascend NPU
     Paste the two sections below into the PR template. Two things to check first:
       1. The GLM-5.2 sentence in Motivation reports "works with no model-side
          change" from your runs, not from anything measured in this repo.
          Reword it to match what you actually ran.
       2. Accuracy Tests is intentionally left empty: there is no breakable-vs-
          disabled gsm8k comparison yet, and the checklist asks for one.
-->

## Motivation

Prefill CUDA graphs had no working path on Ascend NPU. The gate and `PrefillCudaGraphRunner` are already device-agnostic, but `resolve_prefill_backend()` had no NPU branch: an NPU run that passed `--cuda-graph-backend-prefill breakable` received the CUDA `BreakableCudaGraphBackend` and crashed during segment capture on `torch.cuda.CUDAGraph`.

Breakable is the backend that fits, rather than full. NPU attention ops take host-side IntArray sequence lengths, which are frozen into the graph at capture. The breakable backend already runs designated sources eagerly between captured segments, so those ops never enter a segment and nothing needs `graph.update()`. Holding them inside a segment instead would require `torch.npu.graph(..., auto_dispatch_capture=True)` — max workspace pre-allocation, `.out` overloads, record-count alignment — which is out of scope here.

With the device plumbing in place, GLM-5.2 (DSA) captures and replays with no model-side change. DeepSeek V4 Flash needs the per-model break adaptation that forms the second half of this PR; it has been run with DP16/TP16/EP16 and `--page-size 128`, both capture and online requests.

## Modifications

### 1. Device-agnostic breakable graph plumbing

`runner_backend_utils/breakable_cuda_graph/breakable_cuda_graph.py`

- Extract `_device_graph_cls()` and use it in `_begin_new_segment()`, where the segment graph class was hardcoded to `torch.cuda.CUDAGraph` / `torch.xpu.XPUGraph`. `NPUGraph`'s raw `capture_begin` / `capture_end` match the CUDA signature, so the parent's capture code is reused unchanged: no  capture subclass, and none of the per-segment `synchronize()` +  `empty_cache()` that the `torch.npu.graph` context manager would add.
- `_is_stream_capturing()`: add NPU to the portable branch that calls `is_current_stream_capturing()` under the device stream. The CUDA branch reads   `stream.cuda_stream` through cuda-python; NPU has neither.

`model_executor/runner_backend/utils.py`

- `resolve_prefill_backend()` gets an NPU branch returning `BreakableNPUGraphBackend`, and raises `ValueError` for any other prefill backend name on NPU instead of silently returning a CUDA backend whose capture would crash.

`hardware_backend/npu/graph_runner/breakable_npu_graph_backend.py` (new, 60 lines)

- A subclass that overrides no parent method; it supplies the NPU identity and a `torch.compile` assertion. `capture_session` is inherited — the parent already goes through `device_module.graph_pool_handle()`.

### 2. DeepSeek V4 Flash adaptation (NPU-only files, no shared code touched)

`hardware_backend/npu/attention/dsv4_prefill_graph.py` (new, 69 lines)

- `in_prefill_graph()` — the same predicate as the attention break in `models/deepseek_v4.py`, so every break in a layer turns on together.
- `bcg_dsv4_c4_indexer` and `bcg_dsv4_compressor`, both `@eager_on_graph`. They read the batch from `get_tc_piecewise_forward_context()` at call time rather than from captured arguments, return early when `global_num_token_non_padded_cpu == 0` (idle DP rank), and slice their inputs to the live row count.
- The indexer break passes a `copy.copy` of the batch with `positions` sliced to the live rows: `_compute_q_npu` reads `positions` off the batch to build RoPE cos/sin, while the replay batch keeps it at the bucket-padded length. A shallow copy avoids mutate-and-restore on an object that the other breaks in the same forward also read.

Neither source can live inside a captured segment. They consume host-built metadata that is rebuilt per forward outside the graph pool: the state block table width comes from `cu.cpu().tolist()`, the compressor rope table rows from `min(t, t // ratio + bs)`, and `li_quant_metadata` from a host-side metadata op. The CUDA path reaches the same conclusion for ratio 1/2 through `bcg_deepseek_v4_low_ratio_sources` ("The compressor and prefill indexer sync with the host, like the attention.").

`hardware_backend/npu/attention/ascend_dsv4_backend.py` (+53 / -4)

- `forward_c4_indexer` and `forward_core_compressor` dispatch to the breaks under `in_prefill_graph()`. The indexer body is split out as `forward_c4_indexer_eager` so a break does not recurse back through the entry point — the breakable-graph flag is still set while a break runs.
- Capture-time placeholder for `c{ratio}_loc`: a capture batch carries no `out_cache_loc_dsv4` bundle, so `loc` was `None` and the compressor epilog crashed in `set_index_k_scale`. The placeholder is sized from the `ratio_lists` computed in the same function, and gated on `get_flags().dp.capturing_prefill_graph` rather than `is_in_breakable_cuda_graph()` — the latter is also true at replay, where a missing bundle must not silently route real data to slot 0. Slot 0 is never allocated to a real request, so the dummy writes land in the padding page.
- The compress metadata token count now comes from `cu` instead of `positions`: `t = int(cu_cpu[bs])`. The fused compressor derives its own `tokenSize` from `cu_seqlens`, while graph replay keeps `positions` at the bucket-padded length, so a 64-token request replayed in a 128-token bucket built a 33-row rope table where the op wanted 17 (`[compressor] ropeSin shape dim 0, should be equal to min(tokenSize, tokenSize/cmpRatio+batchSize): 17, but got 33`). Eager never exposed this because `positions.shape[0] == cu[-1]` holds there.
- Replace a comment that justified the host read (`cu.cpu().tolist()`) with "prefill is never graph-captured", a premise this PR invalidates. The read is still safe for a different reason: the function runs outside captured segments.

### 3. Tests

All CPU-only unit tests; none require an NPU.

- `test/registered/unit/npu/graph_runner/test_breakable_npu_graph_backend.py` (new, 197 lines): construction reuses the parent backend, `torch.compile` is rejected, `_device_graph_cls()` dispatches per device, segment capture splits at breaks and replays segments and breaks in order, and `_is_stream_capturing()` uses the portable query on NPU.
- `test_backend_resolution.py` (+53 / -2): NPU + `breakable` resolves to `BreakableNPUGraphBackend` and never constructs the CUDA class; NPU + `full` raises `ValueError`.
- `test_npu_ascend_dsv4_backend.py` (+129): capture routes every completed block to slot 0 while a missing bundle outside capture stays `None`; the rope table length follows `cu` rather than the padded positions (regression test for the error above); the breaks run at replay against live rows.

### Known limitations

- Decode is unchanged — it has its own full-graph backend. EAGLE / draft prefill
  and segment dedup are not supported.
- `SGLANG_NPU_USE_MULTI_STREAM=1` is code-path compatible but unverified on
  device.
- `docs/docs/advanced_features/breakable_cuda_graph.mdx` still lists NPU as
  unsupported in its platform table.
- Plain MLA models other than GLM-5.2 and DSV4 Flash are unverified.
