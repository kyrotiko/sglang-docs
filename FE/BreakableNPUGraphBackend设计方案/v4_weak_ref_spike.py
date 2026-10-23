"""V4 spike: weak-ref + segmented-pool liveness on Ascend NPU.

Question: after dropping the only strong Python reference to a tensor
produced inside a captured segment, does a weak view stay valid across
replays while the segment graphs are alive? This is the assumption behind
`_weak_ref_if_tensor` in breakable_cuda_graph.py.

Run on an NPU host:
    python v4_weak_ref_spike.py

Interpretation:
  - Section D PASS: replay writes through to the weak view, and the eager
    break computed from that view matches. Weak refs are usable while the
    segment graphs are alive.
  - Section E: negative control. After every segment graph is dropped, the
    pool may be released; the weak view may become garbage. Observing that
    confirms graph liveness is what pins the storage.
"""

import gc

import torch
import torch_npu  # noqa: F401

DEV = "npu:0"
N = 1024

pool = torch.npu.graph_pool_handle()
stream = torch.npu.Stream()
inp = torch.randn(N, device=DEV)


def capture_pair():
    """Capture seg1 (inp*2 -> mid) and seg2 (bridge+1); same pool/stream."""
    g1 = torch.npu.NPUGraph()
    g2 = torch.npu.NPUGraph()
    state = {}

    with torch.npu.stream(stream):
        with torch.npu.graph(g1, pool=pool, stream=stream, auto_dispatch_capture=True):
            mid = torch.mul(inp, 2.0)
        state["mid_ptr"] = mid.data_ptr()
        state["strong_mid"] = mid
        state["weak_mid"] = torch_npu._C._weak_ref_tensor(mid)
        bridge = torch.empty(N, device=DEV)
        torch.mul(mid, 3.0, out=bridge)  # eager break, like the BCG break fn
        with torch.npu.graph(g2, pool=pool, stream=stream, auto_dispatch_capture=True):
            torch.add(bridge, 1.0, out=bridge)
    torch.npu.synchronize()
    return g1, g2, bridge, state


print("== A. weak_ref_tensor basic ==")
t = torch.randn(N, device=DEV)
w = torch_npu._C._weak_ref_tensor(t)
print("weak view ok; first value:", w[0].item())

print("== B. capture two segments sharing the pool ==")
g1, g2, bridge, state = capture_pair()
weak_mid = state["weak_mid"]
mid_ptr = state["mid_ptr"]
print(f"mid ptr = 0x{mid_ptr:x}, weak ptr = 0x{weak_mid.data_ptr():x}")
assert weak_mid.data_ptr() == mid_ptr, "weak view must alias the original storage"
del state["strong_mid"]  # only the weak view remains
gc.collect()

print("== C. churn: same-pool capture (g3) + normal allocations ==")
g3 = torch.npu.NPUGraph()
with torch.npu.stream(stream):
    with torch.npu.graph(g3, pool=pool, stream=stream, auto_dispatch_capture=True):
        junk = torch.full((N,), 7.0, device=DEV)
        torch.add(junk, 1.0, out=junk)
torch.npu.synchronize()
g3.replay()
torch.npu.synchronize()
churn = [torch.empty(N, device=DEV) for _ in range(64)]
torch.npu.synchronize()

print("== D. replay with fresh input; break fn uses the weak view ==")
new_in = torch.randn(N, device=DEV)
inp.copy_(new_in)
g1.replay()
torch.mul(weak_mid, 3.0, out=bridge)  # mimics the eager break fn at replay
g2.replay()
torch.npu.synchronize()

expected = new_in * 2.0 * 3.0 + 1.0
err = (bridge - expected).abs().max().item()
weak_first = weak_mid[0].item()
print(f"max abs err = {err:.3e}")
print(f"weak_mid[0] = {weak_first:.6f} (expect {(new_in[0] * 2).item():.6f})")
print("RESULT:", "PASS" if err < 1e-3 else "FAIL")

print("== E. negative control: drop every segment graph, churn, read weak ==")
del g1, g2, g3
gc.collect()
churn2 = [torch.empty(N, device=DEV) for _ in range(256)]
torch.npu.synchronize()
try:
    print("weak_mid[0] after graphs dropped:", weak_mid[0].item())
except Exception as e:  # noqa: BLE001 - spike: report and keep going
    print("reading weak view raised:", type(e).__name__, e)

print("done.")
