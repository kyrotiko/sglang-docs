#!/usr/bin/env python3
"""Validate the experimental valid-row cast on an NPU; no model or HCCL needed.

Requires torch_npu and the SGLang checkout providing cast_valid_expert_rows.
Only the cast is captured; real routing and vendor Finalize are checked eagerly.
Routing uses unquantized BF16 inputs and GMM output is simulated; the A8W8 model
path still requires separate validation.
Host reads of R belong to verification/reference code, never the captured kernel.
"""

import argparse
import gc


CASES = ("zero", "partial", "skew", "full")
FLUSH_BYTES = 512 * 1024**2


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--experts", type=int, default=16, help="Local expert count E")
    parser.add_argument(
        "--ep-size",
        type=int,
        default=16,
        help="Partial case local fraction is about 1/EP",
    )
    parser.add_argument("--topk", type=int, default=6)
    parser.add_argument("--bench", action="store_true")
    parser.add_argument("--bench-tokens", type=int, default=32768)
    parser.add_argument(
        "--repeats", type=int, default=10, help="Passes over two large inputs"
    )
    args = parser.parse_args()
    if args.device < 0 or args.hidden < 1 or args.tokens < 2 or args.experts < 2:
        parser.error("need device >= 0, hidden >= 1, tokens >= 2, experts >= 2")
    if args.ep_size < 2:
        parser.error("need ep-size >= 2")
    if not 1 <= args.topk <= args.experts:
        parser.error("need 1 <= topk <= experts (distinct expert IDs per token)")
    if args.bench_tokens < 2 or args.repeats < 1:
        parser.error("need bench-tokens >= 2 and repeats >= 1")
    return args


def init_routing(x, ids, experts):
    return torch.ops.npu.npu_moe_init_routing_v2(
        x,
        ids,
        expert_num=experts,
        active_expert_range=[0, experts],
        active_num=ids.numel(),
        expert_tokens_num_type=1,
        expert_tokens_num_flag=True,
        row_idx_type=0,
        quant_mode=-1,
    )


def finalize(src, expanded, weights, ids):
    return torch.ops.npu.npu_moe_finalize_routing(
        src,
        skip1=None,
        skip2=None,
        bias=None,
        scales=weights,
        expanded_src_to_dst_row=expanded,
        export_for_source_row=ids,
        drop_pad_mode=2,
    )


def check_reverse_map(x, routed, raw, counts, ids, is_local, rows, experts, check_data):
    capacity = ids.numel()
    if not 0 <= rows <= capacity:
        raise AssertionError(f"invalid device count sum R={rows}, C={capacity}")
    if raw.shape != (capacity,):
        raise AssertionError(f"reverse map shape {tuple(raw.shape)} != ({capacity},)")
    if routed.shape != (capacity, x.shape[1]) or routed.dtype != torch.bfloat16:
        raise AssertionError(
            f"routing must return BF16 capacity [C,H], got {routed.shape}"
        )
    if counts.dtype not in (torch.int32, torch.int64) or counts.shape != (experts,):
        raise AssertionError(
            f"unexpected expert counts: {counts.shape}, {counts.dtype}"
        )
    local = is_local.cpu().reshape(-1)
    local_ids = ids.cpu().reshape(-1)[local].to(torch.int64)
    expected_counts = torch.bincount(local_ids, minlength=experts)
    if not torch.equal(counts.cpu().to(torch.int64), expected_counts):
        raise AssertionError(
            f"counts disagree with IDs: {counts.cpu()} vs {expected_counts}"
        )
    reverse = raw.cpu().to(torch.int64)
    valid = reverse[local]
    bad = (valid < 0) | (valid >= rows)
    if bad.any():
        route = local.nonzero().flatten()[bad.nonzero()[0, 0]].item()
        raise AssertionError(
            f"original is_local.reshape(-1) convention failed: C={capacity}, R={rows}, "
            f"flat route={route}, raw reverse={reverse[route].item()}; "
            "not trying a token/topk transpose or changing the production mask"
        )
    if not torch.equal(valid.sort().values, torch.arange(rows)):
        raise AssertionError("local reverse map is not a permutation of [0,R)")
    # Check expert grouping independently of a token-major interpretation.
    ends = expected_counts.cumsum(0)
    starts = ends - expected_counts
    if ((valid < starts[local_ids]) | (valid >= ends[local_ids])).any():
        raise AssertionError(
            "original flattened mask does not map IDs into their expert groups"
        )
    if check_data and rows:
        expected = x.cpu().repeat_interleave(ids.shape[1], dim=0)[local]
        actual = routed[:rows].cpu()[valid]
        if not torch.equal(actual, expected):
            raise AssertionError(
                "original flattened mask failed routed source-row check; "
                "reverse-map layout may differ from the production convention"
            )


def route(args, case, phase, device, tokens, check_data=True):
    token = torch.arange(tokens, dtype=torch.int64)[:, None]
    slot = torch.arange(args.topk, dtype=torch.int64)[None, :]
    local_ids = (token * 3 + slot) % args.experts
    if case == "zero":
        local = torch.zeros((tokens, args.topk), dtype=torch.bool)
    elif case == "full":
        local = torch.ones((tokens, args.topk), dtype=torch.bool)
    elif case == "partial":
        local = (token * 5 + slot * 3 + phase) % args.ep_size == 0
    elif case == "skew":
        local = slot == (token + phase) % args.topk
        local_ids = torch.where(token == tokens - 1, args.experts - 1, 0).expand_as(
            local
        )
    else:
        raise ValueError(case)
    # Global IDs span two expert partitions; this process owns [0,E).
    global_ids = torch.where(local, local_ids, args.experts + slot).to(torch.int32)
    global_ids = global_ids.to(device).contiguous()
    is_local = (global_ids >= 0) & (global_ids < args.experts)
    ids = global_ids.masked_fill(~is_local, args.experts).contiguous()
    weights = torch.rand((tokens, args.topk), device=device, dtype=torch.float32)
    weights.add_(0.125).masked_fill_(~is_local, 0)
    x = torch.rand((tokens, args.hidden), device=device, dtype=torch.bfloat16)
    x.add_(1 + phase / 8)
    routed, raw, counts, _ = init_routing(x, ids, args.experts)
    counts = counts.contiguous()
    rows = int(counts.sum().item())
    check_reverse_map(
        x, routed, raw, counts, ids, is_local, rows, args.experts, check_data
    )
    # Keep precisely the production mask; never silently repair its layout.
    expanded = raw.masked_fill(~is_local.reshape(-1), ids.numel() - 1)
    expanded = expanded.to(torch.int32).contiguous()
    return routed, counts, expanded, weights, ids, rows


def simulated_gmm(routed, rows, phase):
    src = torch.full_like(routed, -3 - phase)
    if rows:
        src[:rows].copy_(routed[:rows])
        src[:rows].mul_(0.5).add_(phase + 1)
    return src


def full_fp32(src, rows):
    ref = src.float()
    # When R=C, C-1 contains a real expert result.
    if rows < src.shape[0]:
        ref[-1].zero_()
    return ref


def assert_bits(got, expected, label):
    if (
        got.shape != expected.shape
        or got.dtype != torch.float32
        or expected.dtype != torch.float32
    ):
        raise AssertionError(f"{label}: expected matching FP32 outputs")
    # Bound host memory for the optional large-tensor Finalize check.
    for begin in range(0, got.shape[0], 256):
        a = got[begin : begin + 256].cpu().contiguous()
        b = expected[begin : begin + 256].cpu().contiguous()
        if not torch.isfinite(a).all() or not torch.isfinite(b).all():
            raise AssertionError(f"{label}: nonfinite output near row {begin}")
        if not torch.equal(a.view(torch.int32), b.view(torch.int32)):
            pos = (a.view(torch.int32) != b.view(torch.int32)).nonzero()[0].tolist()
            raise AssertionError(
                f"{label}: bit mismatch at ({begin + pos[0]}, {pos[1]}), "
                f"got={a[tuple(pos)].item()}, expected={b[tuple(pos)].item()}"
            )


def check_written(src, dst, rows):
    if rows:
        assert_bits(dst[:rows], src[:rows].float(), "valid cast prefix")
    if rows < src.shape[0]:
        sentinel = dst[-1].cpu()
        if not torch.equal(
            sentinel.view(torch.int32), torch.zeros_like(sentinel, dtype=torch.int32)
        ):
            raise AssertionError("R<C sentinel must contain positive FP32 zero")
        if not torch.isnan(dst[rows:-1].cpu()).all():
            raise AssertionError("kernel wrote outside [0,R) and the C-1 sentinel")


def check_finalize(src, dst, expanded, weights, ids, rows):
    ref = full_fp32(src, rows)
    expected = finalize(ref, expanded, weights, ids)
    actual = finalize(dst, expanded, weights, ids)
    assert_bits(actual, expected, "vendor Finalize vs full FP32")


def test_eager(args, cast, device):
    for phase, case in enumerate(CASES):
        routed, counts, expanded, weights, ids, rows = route(
            args, case, phase, device, args.tokens
        )
        src = simulated_gmm(routed, rows, phase)
        dst = torch.empty_like(src, dtype=torch.float32)
        for dtype in (torch.int64, torch.int32):
            dst.fill_(float("nan"))
            result = cast(src=src, dst=dst, expert_tokens=counts.to(dtype).contiguous())
            if result is not None:
                raise AssertionError("cast_valid_expert_rows must return None")
            torch.npu.synchronize()
            check_written(src, dst, rows)
            check_finalize(src, dst, expanded, weights, ids, rows)
        print(f"[PASS] eager {case}: R={rows}/{ids.numel()}, int64/int32", flush=True)


def test_graph(args, cast, device, dtype):
    routed, counts, _, _, _, rows = route(args, "partial", 0, device, args.tokens)
    src = simulated_gmm(routed, rows, 0)
    dst = torch.empty_like(src, dtype=torch.float32)
    device_counts = counts.to(dtype).contiguous()
    addresses = (src.data_ptr(), dst.data_ptr(), device_counts.data_ptr())

    def run():
        cast(src=src, dst=dst, expert_tokens=device_counts)

    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        for _ in range(3):
            run()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, stream=stream):
        run()
    torch.npu.synchronize()

    transitions = ("partial", "full", "skew", "zero", "full", "partial", "zero")
    observed = []
    for phase, case in enumerate(transitions, 1):
        routed, counts, expanded, weights, ids, rows = route(
            args, case, phase, device, args.tokens
        )
        src.copy_(simulated_gmm(routed, rows, phase))
        device_counts.copy_(counts)
        dst.fill_(float("nan"))
        torch.npu.synchronize()
        if addresses != (src.data_ptr(), dst.data_ptr(), device_counts.data_ptr()):
            raise AssertionError("captured buffer addresses changed")
        graph.replay()
        torch.npu.synchronize()
        check_written(src, dst, rows)
        check_finalize(src, dst, expanded, weights, ids, rows)
        observed.append(rows)
    print(
        f"[PASS] graph {dtype}: one capture, fixed addresses, R={observed}; "
        "changed inputs/counts; real routing/Finalize outside graph",
        flush=True,
    )


def measure_ms(fn, sources, repeats, flush):
    for src in sources:
        fn(src)
    torch.npu.synchronize()
    events = [
        (torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True))
        for _ in range(repeats * len(sources))
    ]
    for index, (start, end) in enumerate(events):
        # Flush on the same stream before the timed interval, even for a small valid prefix.
        flush.zero_()
        start.record()
        fn(sources[index % len(sources)])
        end.record()
    torch.npu.synchronize()
    return sum(start.elapsed_time(end) for start, end in events) / len(events)


def benchmark_case(args, cast, device, case):
    tokens, hidden = args.bench_tokens, args.hidden
    capacity = tokens * args.topk
    source_bytes = capacity * hidden * 2
    if source_bytes < 256 * 1024**2:
        print(
            "[SKIP] bench: each BF16 source must be >=256 MiB; increase bench-tokens/hidden"
        )
        return
    torch.npu.synchronize()
    gc.collect()
    torch.npu.empty_cache()
    free, _ = torch.npu.mem_get_info()
    # Two BF16 sources, one FP32 destination, two Finalize outputs, flush and workspace margin.
    budget = (
        8 * capacity * hidden
        + 12 * tokens * hidden
        + 32 * capacity
        + FLUSH_BYTES
        + 512 * 1024**2
    )
    print(
        f"[INFO] bench {case}: conservative budget={budget / 1024**3:.2f} GiB, "
        f"free={free / 1024**3:.2f} GiB (limit 70%); cases allocated sequentially",
        flush=True,
    )
    if budget > free * 0.7:
        print(
            "[SKIP] bench: insufficient memory budget; reduce --bench-tokens",
            flush=True,
        )
        return
    routed, counts, expanded, weights, ids, rows = route(
        args, case, 0, device, tokens, check_data=False
    )
    del routed
    torch.npu.synchronize()
    torch.npu.empty_cache()
    sources = tuple(
        torch.empty((capacity, hidden), device=device, dtype=torch.bfloat16).uniform_(
            -2, 2
        )
        for _ in range(2)
    )
    dst = torch.empty((capacity, hidden), device=device, dtype=torch.float32)
    flush = torch.empty(FLUSH_BYTES, device=device, dtype=torch.uint8)

    def run(src, optimized, with_finalize):
        if optimized:
            cast(src=src, dst=dst, expert_tokens=counts)
        else:
            dst.copy_(src)
            if rows < capacity:
                dst[-1].zero_()
        if with_finalize:
            return finalize(dst, expanded, weights, ids)

    # Check this large shape before timing; no correctness work enters the timed region.
    for src in sources:
        expected = run(src, False, True)
        dst.fill_(float("nan"))
        actual = run(src, True, True)
        assert_bits(actual, expected, "large vendor Finalize")
        del actual, expected
    print(
        f"[INFO] bench {case}: R={rows}/{capacity}, local fraction={rows / capacity:.6f}, "
        f"EP={args.ep_size}; two rotating {source_bytes / 1024**3:.2f} GiB BF16 inputs; "
        f"device event ms/call, independent {FLUSH_BYTES / 1024**2:.0f} MiB zero_ flush "
        "before every start event (flush excluded); full=copy_ + conditional sentinel",
        flush=True,
    )
    for with_finalize, label in ((False, "cast"), (True, "cast+Finalize")):
        full_ms = measure_ms(
            lambda src: run(src, False, with_finalize), sources, args.repeats, flush
        )
        valid_ms = measure_ms(
            lambda src: run(src, True, with_finalize), sources, args.repeats, flush
        )
        print(
            f"[MEASURED] {case} {label}: full={full_ms:.4f} ms, valid={valid_ms:.4f} ms",
            flush=True,
        )


def main():
    args = parse_args()
    # --help exits before importing torch, torch_npu, or SGLang.
    global torch
    try:
        import torch
        import torch_npu  # noqa: F401

        from sglang.kernels.ops.moe import cast_valid_expert_rows
    except ImportError as exc:
        raise SystemExit(
            f"NPU run requires torch_npu and the experimental SGLang cast module: {exc}"
        ) from exc
    if not torch.npu.is_available():
        raise SystemExit(
            "NPU unavailable; no device validation or timing was performed"
        )
    torch.npu.set_device(args.device)
    device = f"npu:{args.device}"
    torch.manual_seed(20261107)
    with torch.inference_mode():
        test_eager(args, cast_valid_expert_rows, device)
        for dtype in (torch.int64, torch.int32):
            test_graph(args, cast_valid_expert_rows, device, dtype)
        print(
            "[PASS] correctness suite complete; no performance claim from these small inputs",
            flush=True,
        )
        if args.bench:
            for case in ("partial", "full"):
                benchmark_case(args, cast_valid_expert_rows, device, case)
    torch.npu.synchronize()


if __name__ == "__main__":
    main()
