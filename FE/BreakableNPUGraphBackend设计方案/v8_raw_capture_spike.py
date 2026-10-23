"""V8 spike: is bare capture_begin/capture_end equivalent to torch.npu.graph?

Design B drops the `torch.npu.graph` context manager and calls the raw
NPUGraph API from the parent capture class, exactly like the CUDA path:

    graph = torch.npu.NPUGraph()
    graph.capture_begin(pool=pool, capture_error_mode="global")
    ...
    graph.capture_end()

What that skips, per torch_npu (torch_npu/npu/graphs.py:898-907):
`torch.npu.synchronize()`, `torch.npu.empty_cache()`, `host_empty_cache()`,
and the stream context (the parent enters the stream itself).

Sections:
  A  vocabulary check: which raw methods exist on NPUGraph
  B  two segments via raw API, sync before each capture, eager break between;
     replay with a fresh input -> err must be 0
  C  same, but no sync before capture and a kernel still pending on the
     capture stream (mimics capture_one: warmups then capture immediately)
  D  empty segment (back-to-back breaks) via raw API
  E  repeated replays with new inputs (graph liveness / address stability)

Run on an NPU host:
    python v8_raw_capture_spike.py

Interpretation:
  - B PASS is the baseline: the graph class plug-in works.
  - C is the real question. PASS/FAIL here decides design B vs the
    fallback (torch.npu.graph + NPU capture subclass + _make_capture_ctx).
  - D/E are regression guards; a failure there also blocks design B.
"""

import traceback

import torch
import torch_npu  # noqa: F401

DEV = "npu:0"
N = 1024
RESULTS = {}


def report(name: str, ok: bool, detail: str = "") -> None:
    RESULTS[name] = ok
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))


def new_buffers():
    return (
        torch.randn(N, device=DEV),   # in
        torch.empty(N, device=DEV),   # out1
        torch.empty(N, device=DEV),   # bridge (eager break)
        torch.empty(N, device=DEV),   # out2
    )


def capture_two_segments(pool, stream, sync_before: bool, pending_kernel: bool):
    """Raw-API capture of seg1 -> eager break -> seg2 on one stream/pool."""
    inp, out1, bridge, out2 = new_buffers()

    def seg1():
        torch.mul(inp, 2.0, out=out1)

    def break_fn():
        torch.mul(out1, 3.0, out=bridge)

    def seg2():
        torch.add(bridge, 1.0, out=out2)

    # Warmups, like BreakableCudaGraphBackend.capture_one.
    for _ in range(2):
        with torch.npu.stream(stream):
            seg1()
            break_fn()
            seg2()
    torch.npu.synchronize()

    addr_before = tuple(t.data_ptr() for t in (inp, out1, bridge, out2))

    if sync_before:
        torch.npu.synchronize()

    scratch = torch.empty(N, device=DEV)

    with torch.npu.stream(stream):
        if pending_kernel:
            # Still in flight on the capture stream when capture begins.
            torch.add(inp, 7.0, out=scratch)
            assert not torch.npu.is_current_stream_capturing()

        g1 = torch.npu.NPUGraph()
        g1.capture_begin(pool=pool, capture_error_mode="global")
        assert torch.npu.is_current_stream_capturing()
        seg1()
        g1.capture_end()

        # Eager break: runs outside any capture.
        assert not torch.npu.is_current_stream_capturing()
        break_fn()

        g2 = torch.npu.NPUGraph()
        g2.capture_begin(pool=pool, capture_error_mode="global")
        seg2()
        g2.capture_end()

    torch.npu.synchronize()
    addr_after = tuple(t.data_ptr() for t in (inp, out1, bridge, out2))
    return g1, g2, (inp, out1, bridge, out2), addr_before, addr_after


def replay_with_break(g1, g2, bufs, rounds: int = 3) -> float:
    inp, out1, bridge, out2 = bufs
    worst = 0.0
    for _ in range(rounds):
        new_in = torch.randn(N, device=DEV)
        inp.copy_(new_in)
        g1.replay()
        torch.mul(out1, 3.0, out=bridge)  # the eager break fn
        g2.replay()
        torch.npu.synchronize()
        ref = new_in * 2.0 * 3.0 + 1.0
        worst = max(worst, (out2 - ref).abs().max().item())
    return worst


print("== A. raw API on NPUGraph ==")
g = torch.npu.NPUGraph()
have = {name: hasattr(g, name) for name in ("capture_begin", "capture_end", "replay")}
print("methods:", have)
report("A.raw-api-exists", all(have.values()), str(have))
del g

pool = torch.npu.graph_pool_handle()
stream = torch.npu.Stream()

print("== B. two segments via raw API (sync before capture) ==")
try:
    g1, g2, bufs, a_before, a_after = capture_two_segments(
        pool, stream, sync_before=True, pending_kernel=False
    )
    stable = a_before == a_after
    err = replay_with_break(g1, g2, bufs)
    print(f"addresses stable: {stable}; max abs err = {err:.3e}")
    report("B.capture-ok", True)
    report("B.address-stable", stable, f"{a_before} -> {a_after}")
    report("B.replay-correct", err < 1e-5, f"err={err:.3e}")
    del g1, g2
except Exception:
    traceback.print_exc()
    report("B.capture-ok", False, "exception during raw capture")

print("== C. no sync before capture, kernel pending on capture stream ==")
try:
    g1, g2, bufs, a_before, a_after = capture_two_segments(
        pool, stream, sync_before=False, pending_kernel=True
    )
    stable = a_before == a_after
    err = replay_with_break(g1, g2, bufs)
    print(f"addresses stable: {stable}; max abs err = {err:.3e}")
    report("C.capture-ok", True)
    report("C.address-stable", stable)
    report("C.replay-correct", err < 1e-5, f"err={err:.3e}")
    del g1, g2
except Exception:
    traceback.print_exc()
    report("C.capture-ok", False, "exception during unsynced capture")

print("== D. empty segment via raw API ==")
try:
    with torch.npu.stream(stream):
        ge = torch.npu.NPUGraph()
        ge.capture_begin(pool=pool, capture_error_mode="global")
        ge.capture_end()
    torch.npu.synchronize()
    ge.replay()
    torch.npu.synchronize()
    report("D.empty-segment", True)
    del ge
except Exception:
    traceback.print_exc()
    report("D.empty-segment", False, "exception on empty segment")

print("== E. repeated replays on the B-style graphs ==")
try:
    g1, g2, bufs, _, _ = capture_two_segments(
        pool, stream, sync_before=True, pending_kernel=False
    )
    err = replay_with_break(g1, g2, bufs, rounds=5)
    report("E.repeat-replay", err < 1e-5, f"worst err={err:.3e}")
    del g1, g2
except Exception:
    traceback.print_exc()
    report("E.repeat-replay", False, "exception during repeated replay")

print()
print("summary:", RESULTS)
failed = [k for k, ok in RESULTS.items() if not ok]
print("RESULT:", "PASS (design B)" if not failed else f"FAIL -> fallback to torch.npu.graph ({failed})")
