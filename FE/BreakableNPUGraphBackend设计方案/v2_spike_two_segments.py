"""V2 spike: capture two NPU graph segments sharing one pool, with an eager
break in between, then replay with changed input and check correctness.

Run on an Ascend NPU host:
    python v2_spike_two_segments.py

What it verifies:
  1. Two torch.npu.graph captures on the same stream + same pool are legal.
  2. Tensors written eagerly between segments keep stable addresses.
  3. Replaying both segments reads the current input value (not capture-time).
  Also incidentally checks V6: an empty segment (all ops eager) is tolerated.
"""

import torch
import torch_npu  # noqa: F401

DEV = "npu:0"
N = 1024

pool = torch.npu.graph_pool_handle()
stream = torch.npu.Stream()

# Static buffers: these addresses are what the captured graphs will use.
seg1_in = torch.randn(N, device=DEV)
seg1_out = torch.empty(N, device=DEV)   # produced by segment 1
bridge = torch.empty(N, device=DEV)     # written by the eager break fn
seg2_out = torch.empty(N, device=DEV)   # produced by segment 2


def seg1_fn(scale: float) -> None:
    torch.mul(seg1_in, scale, out=seg1_out)


def break_fn() -> None:
    # Runs eagerly between segments; its output is the bridge buffer that
    # segment 2 reads. Addresses must be stable across replays.
    torch.mul(seg1_out, 2.0, out=bridge)


def seg2_fn() -> None:
    torch.add(bridge, 1.0, out=seg2_out)


def run_sequence() -> None:
    seg1_fn(3.0)
    break_fn()
    seg2_fn()


def addr(t: torch.Tensor) -> int:
    return t.data_ptr()


print("warmup ...")
for _ in range(2):
    with torch.npu.stream(stream):
        run_sequence()
torch.npu.synchronize()

print("capture ...")
g1 = torch.npu.NPUGraph()
g2 = torch.npu.NPUGraph()
addr_before = (addr(seg1_in), addr(seg1_out), addr(bridge), addr(seg2_out))

with torch.npu.stream(stream):
    # Segment 1
    with torch.npu.graph(g1, pool=pool, stream=stream, auto_dispatch_capture=True):
        seg1_fn(3.0)
    # Eager break between the two captured segments
    break_fn()
    # Segment 2
    with torch.npu.graph(g2, pool=pool, stream=stream, auto_dispatch_capture=True):
        seg2_fn()
torch.npu.synchronize()

addr_after = (addr(seg1_in), addr(seg1_out), addr(bridge), addr(seg2_out))
assert addr_before == addr_after, f"address moved: {addr_before} -> {addr_after}"

print("replay with new input ...")
new_in = torch.randn(N, device=DEV)
seg1_in.copy_(new_in)
g1.replay()
break_fn()
g2.replay()
torch.npu.synchronize()

ref = new_in * 3.0 * 2.0 + 1.0
err = (seg2_out - ref).abs().max().item()
print(f"max abs err = {err:.3e}")
assert err < 1e-3, "MISMATCH: captured segments read stale data"

# V6 side check: capturing a segment whose ops are all eager (empty graph).
print("empty-segment capture ...")
g_empty = torch.npu.NPUGraph()
try:
    with torch.npu.stream(stream):
        with torch.npu.graph(
            g_empty, pool=pool, stream=stream, auto_dispatch_capture=True
        ):
            pass
    torch.npu.synchronize()
    g_empty.replay()
    torch.npu.synchronize()
    print("empty segment: OK")
except Exception as e:  # noqa: BLE001 - spike script, report and continue
    print(f"empty segment: FAILED ({type(e).__name__}: {e})")

print("RESULT: V2 PASS (shared pool + stable addresses + fresh input)")
