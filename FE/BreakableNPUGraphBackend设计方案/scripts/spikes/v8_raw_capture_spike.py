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

Pool lifetime gotcha (found the hard way): a mempool token can only be
reused while at least one graph created with it is still alive. Once every
such graph is destroyed, `releasePool` drops use_count to 0 and a later
capture_begin on the same token trips
`create_or_incref_pool: use_count > 0` (NPUCachingAllocator.cpp:2232).
So every section below takes a FRESH pool, and sections that may poison the
process are last.

Sections:
  A  vocabulary check: which raw methods exist on NPUGraph
  B  two segments via raw API, sync before capture, eager break between;
     replay 5x with fresh inputs -> err must be 0
  C  empty segment (back-to-back breaks) via raw API
  D  no sync, warmup kernels still pending on ANOTHER stream (what
     capture_one actually looks like: warmups on the default stream,
     capture on the side capture stream)
  E  no sync, kernel pending on the CAPTURE stream itself (harsher)

Run on an NPU host:
    python v8_raw_capture_spike.py

Interpretation:
  - B PASS is the baseline: the graph class plug-in works.
  - D is the real question: does capture_begin need a device-wide sync?
    A failure here means design B must add a synchronize() before
    capture_begin (or fall back to the torch.npu.graph context).
  - E is informational; it is harsher than the real capture path.
  - If D/E raise the allocator assert, the process is poisoned: later
    sections will be skipped, earlier results stand.
"""

import traceback

import torch
import torch_npu  # noqa: F401

DEV = "npu:0"
N = 1024
RESULTS = {}
POISONED = False


def report(name: str, ok: bool, detail: str = "") -> None:
    RESULTS[name] = ok
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))


def section(name: str) -> bool:
    """Start a section unless an earlier failure poisoned the process."""
    global POISONED
    print(f"== {name} ==")
    if POISONED:
        print("skipped: the process is poisoned by an earlier capture failure")
        RESULTS[name] = False
        return False
    return True


def new_pool():
    # A fresh token per section; see the pool lifetime gotcha above.
    return torch.npu.graph_pool_handle()


def new_buffers():
    return (
        torch.randn(N, device=DEV),   # in
        torch.empty(N, device=DEV),   # out1 (segment 1 output)
        torch.empty(N, device=DEV),   # bridge (eager break output)
        torch.empty(N, device=DEV),   # out2 (segment 2 output)
    )


def capture_two_segments(pool, stream, *, sync_before: bool, busy: str | None):
    """Raw-API capture of seg1 -> eager break -> seg2 on one stream/pool.

    busy: None | "other" (pending work on the default stream) |
          "capture" (pending work on the capture stream itself).
    """
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
    scratch = torch.empty(N, device=DEV)

    if sync_before:
        torch.npu.synchronize()
    elif busy == "other":
        # Warmup-shaped work still in flight on the default stream, capture
        # starts on the side stream without waiting for it.
        torch.add(inp, 7.0, out=scratch)
    elif busy == "capture":
        with torch.npu.stream(stream):
            torch.add(inp, 7.0, out=scratch)

    with torch.npu.stream(stream):
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


def replay_with_break(g1, g2, bufs, rounds: int = 5) -> float:
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


def run_pair(name: str, **kwargs) -> None:
    global POISONED
    try:
        g1, g2, bufs, a_before, a_after = capture_two_segments(**kwargs)
    except Exception:
        traceback.print_exc()
        report(f"{name}.capture-ok", False, "exception during capture_begin/end")
        POISONED = True
        return
    report(f"{name}.capture-ok", True)
    stable = a_before == a_after
    report(f"{name}.address-stable", stable, "" if stable else f"{a_before} -> {a_after}")
    err = replay_with_break(g1, g2, bufs)
    report(f"{name}.replay-correct", err < 1e-5, f"err={err:.3e}")
    del g1, g2


print("== A. raw API on NPUGraph ==")
g = torch.npu.NPUGraph()
have = {name: hasattr(g, name) for name in ("capture_begin", "capture_end", "replay")}
print("methods:", have)
report("A.raw-api-exists", all(have.values()), str(have))
del g

stream = torch.npu.Stream()

if section("B. two segments via raw API (sync before capture)"):
    run_pair("B", pool=new_pool(), stream=stream, sync_before=True, busy=None)

if section("C. empty segment via raw API (fresh pool)"):
    try:
        with torch.npu.stream(stream):
            ge = torch.npu.NPUGraph()
            ge.capture_begin(pool=new_pool(), capture_error_mode="global")
            ge.capture_end()
        torch.npu.synchronize()
        ge.replay()
        torch.npu.synchronize()
        report("C.empty-segment", True)
        del ge
    except Exception:
        traceback.print_exc()
        report("C.empty-segment", False, "exception on empty segment")
        POISONED = True

if section("D. no sync, pending work on ANOTHER stream (capture_one shape)"):
    run_pair("D", pool=new_pool(), stream=stream, sync_before=False, busy="other")

if section("E. no sync, pending work on the CAPTURE stream (harsher)"):
    run_pair("E", pool=new_pool(), stream=stream, sync_before=False, busy="capture")

print()
print("summary:", RESULTS)
failed = [k for k, ok in RESULTS.items() if not ok]
print("RESULT:", "PASS (design B)" if not failed else f"FAIL -> fallback / add sync ({failed})")
