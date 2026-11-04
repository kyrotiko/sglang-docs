"""Does the NPU fused compressor re-read cu_seqlens on every replay?

This is the one question that decides whether DSV4 prefill can go full-graph.
Everything else in that plan is transcription work: CUDA's DSV4 backend already
has the captured-metadata contract (deepseek_v4_backend.py:1263/3239/3342) and
NPU decode already captures torch.ops.npu.compressor into a full graph and
refreshes its device-tensor inputs in place (ascend_dsv4_backend.py:1681/1882).

But decode never rewrites cu. Per request it is exactly one token, so
actual_seq_lengths_q_pa is a constant (built once at :1376, never copy_'d).
Prefill must rewrite it: the op derives its own tokenSize from cu_seqlens, and
that value changes with every batch.

So: the op demonstrably READS cu at execution time -- the ropeSin error proves
it cross-checks cu against the rope table row count. What is unverified is
whether a CAPTURED op re-reads the buffer's contents at replay, or whether
tiling freezes tokenSize at capture.

                    capture with cu=[0,64]
                              |
                    replay 1, cu=[0,64]   -> out_a
                    replay 2, cu=[0,32]   -> out_b     (same buffer, new content)
                              |
            out_a != out_b  ->  the op re-reads cu.  Full-graph prefill is viable.
            out_a == out_b  ->  tokenSize is frozen.  Needs CANN support.

Usage (on the NPU host, inside the normal serving env):

    SGLANG_CU_REBIND_SPIKE=1 python -m sglang.launch_server ... \
        --cuda-graph-backend-prefill breakable --cuda-graph-bs-prefill 128

Import this module from anywhere that runs before the first prefill (e.g. add
``import cu_rebind_spike`` to a sitecustomize, or call install() from a
breakpoint). It patches the compressor op, captures the first real call it
sees, and reports. It does not change serving behaviour: the real call still
runs, and the spike only adds its own extra graph.
"""

from __future__ import annotations

import logging
import os

import torch

logger = logging.getLogger(__name__)

_RESULT_BANNER = "=" * 72


def _clone_args(args, kwargs):
    """Deep-copy the tensors so the spike never aliases live serving buffers."""
    c_args = tuple(a.clone() if isinstance(a, torch.Tensor) else a for a in args)
    c_kwargs = {
        k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in kwargs.items()
    }
    return c_args, c_kwargs


def _halve_cu(cu: torch.Tensor) -> torch.Tensor:
    """Build a second cu describing half the tokens, same shape and dtype.

    cu is [bs+1] cumulative query lengths starting at 0, so halving every
    non-zero entry keeps it monotonic and keeps cu[0] == 0. Round down to a
    multiple of 128 where possible: the sparse-attn metadata op rejects token
    counts below the compression granularity, and we want this spike to fail
    on the question being asked, not on that limit.
    """
    halved = (cu // 2).to(cu.dtype)
    if int(cu[-1]) >= 256:
        halved = ((cu // 2) // 128 * 128).to(cu.dtype)
        if int(halved[-1]) == 0:
            halved = (cu // 2).to(cu.dtype)
    halved[0] = 0
    return torch.maximum(halved, torch.zeros_like(halved))


def _run_probe(op, args, kwargs) -> None:
    """Capture one graph around op(*args), then replay it twice with
    different cu contents written into the SAME buffer."""
    import torch_npu  # noqa: F401

    args, kwargs = _clone_args(args, kwargs)
    cu = kwargs["cu_seqlens"]
    cu_full = cu.clone()
    cu_half = _halve_cu(cu_full)

    logger.warning(
        "[cu-spike] cu_full=%s cu_half=%s", cu_full.tolist(), cu_half.tolist()
    )
    if torch.equal(cu_full, cu_half):
        logger.warning("[cu-spike] ABORT: cu_half == cu_full, nothing to distinguish")
        return

    # Warm up twice so kernel load and any one-time setup happen before capture,
    # mirroring NPUCudaGraphBackend.capture_one.
    for _ in range(2):
        torch.npu.synchronize()
        op(*args, **kwargs)

    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        captured_out = op(*args, **kwargs)

    def replay_with(cu_value: torch.Tensor) -> torch.Tensor:
        # Write into the captured buffer: same address, new content. This is
        # exactly what a full-graph prefill replay would do.
        cu.copy_(cu_value)
        torch.npu.synchronize()
        graph.replay()
        torch.npu.synchronize()
        out = captured_out[0] if isinstance(captured_out, (tuple, list)) else captured_out
        return out.clone()

    out_a = replay_with(cu_full)
    out_b = replay_with(cu_half)
    out_c = replay_with(cu_full)  # back to the original: guards against drift

    _report(out_a, out_b, out_c, cu_full, cu_half)


def _report(out_a, out_b, out_c, cu_full, cu_half) -> None:
    same_ab = torch.equal(out_a, out_b)
    same_ac = torch.equal(out_a, out_c)
    n_full, n_half = int(cu_full[-1]), int(cu_half[-1])

    # Only the rows the shorter cu does not cover should differ. Compare the
    # common prefix separately: a compressor that re-reads cu should leave
    # rows below n_half/ratio alone and change (or stop writing) the rest.
    rows = out_a.shape[0]
    cut = max(1, rows * n_half // max(n_full, 1))
    prefix_same = torch.equal(out_a[:cut], out_b[:cut])

    logger.warning("%s", _RESULT_BANNER)
    logger.warning("[cu-spike] out shape=%s rows=%d", tuple(out_a.shape), rows)
    logger.warning("[cu-spike] cu[-1]: full=%d half=%d", n_full, n_half)
    logger.warning("[cu-spike] replay(full) == replay(half) : %s", same_ab)
    logger.warning("[cu-spike] replay(full) == replay(full)': %s  (want True)", same_ac)
    logger.warning("[cu-spike] common prefix [:%d] equal    : %s", cut, prefix_same)

    if not same_ac:
        logger.warning(
            "[cu-spike] INCONCLUSIVE: two replays with identical cu disagree, so "
            "something other than cu is varying (state cache carry-over is the "
            "likely cause -- the compressor mutates state_cache in place). Re-run "
            "with a c128 layer, or snapshot/restore state_cache around each replay."
        )
    elif same_ab:
        logger.warning(
            "[cu-spike] NEGATIVE: output did not change when cu changed. tokenSize "
            "is frozen at capture. Full-graph prefill needs CANN support; this run "
            "is the minimal repro to hand over."
        )
    else:
        logger.warning(
            "[cu-spike] POSITIVE: a captured compressor re-reads cu_seqlens at "
            "replay. Full-graph prefill is viable -- proceed with the captured-"
            "metadata contract (CUDA deepseek_v4_backend.py:3239/3342 as the "
            "blueprint)."
        )
    logger.warning("%s", _RESULT_BANNER)


def install() -> None:
    """Wrap the compressor op so the first real prefill call drives the probe.

    torch.ops.<ns>.<op> resolves lazily through C++, so the op object cannot be
    replaced in place. forward_compress reads ``torch.ops`` through the backend
    module's globals on every call, so shadowing that name on the module is
    enough and leaves the real torch.ops untouched for everyone else.
    """
    from sglang.srt.hardware_backend.npu.attention import ascend_dsv4_backend as mod

    state = {"done": False}
    real_ops = mod.torch.ops

    class _NsProxy:
        def __init__(self, ns):
            self._ns = ns

        def __getattr__(self, op_name):
            op = getattr(self._ns, op_name)
            if op_name != "compressor":
                return op

            def probing(*args, **kwargs):
                out = op(*args, **kwargs)
                if not state["done"] and "cu_seqlens" in kwargs:
                    state["done"] = True  # one shot, set before the probe can throw
                    try:
                        _run_probe(op, args, kwargs)
                    except Exception:
                        logger.exception("[cu-spike] probe raised; serving continues")
                return out

            return probing

    class _OpsProxy:
        def __getattr__(self, ns_name):
            ns = getattr(real_ops, ns_name)
            return _NsProxy(ns) if ns_name in ("npu", "custom") else ns

    # Shadow only the one global the op lookup goes through. Patching `mod.torch`
    # wholesale would route all 252 torch.* calls in that module through a proxy;
    # a module-level `ops` name shadows the attribute lookup on `torch` for the
    # compressor call site alone, because forward_compress writes
    # `torch.ops.custom.compressor` / `torch.ops.npu.compressor` and Python
    # resolves `torch` from module globals first.
    mod.torch = _ShadowTorch(real_torch=mod.torch, ops=_OpsProxy())
    logger.warning("[cu-spike] installed; probing the first compressor call")


class _ShadowTorch:
    """`torch` stand-in whose only override is `.ops`.

    __getattr__ runs solely for names absent from the instance dict, so `ops`
    resolves to the proxy and every other attribute reaches the real module
    with no wrapping and no per-call overhead beyond one attribute lookup.
    """

    def __init__(self, real_torch, ops):
        self.__dict__["ops"] = ops
        self.__dict__["_real"] = real_torch

    def __getattr__(self, name):
        return getattr(self.__dict__["_real"], name)


if os.environ.get("SGLANG_CU_REBIND_SPIKE") == "1":
    try:
        install()
    except Exception:
        logger.exception("[cu-spike] install failed")
