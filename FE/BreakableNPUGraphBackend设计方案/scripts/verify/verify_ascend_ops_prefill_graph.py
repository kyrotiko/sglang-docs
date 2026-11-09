#!/usr/bin/env python3
"""验证 ascend_ops `prefill=allgather` 能否进入 NPU prefill 图。

目标配置：

    --moe-a2a-backend ascend_ops \
    --moe-ascend-ops-config '{"prefill":"allgather","decode":"low_latency"}'

本脚本只做"事实核查"，不改 sglang、不依赖 sglang 运行时。它在真实 rank 数下
复刻 `AscendOpsDispatcher._dispatch_allgather` + `_pre_permute_allgather` +
`_allgather_weighted_sum` + `_combine_allgather` 的张量流，逐项回答四个问题：

1. account     这条路径在每个 bucket 下的峰值显存是多少（含 fp32 expert_output）；
2. collective  HCCL all_gather_into_tensor / reduce_scatter_tensor 能否被 NPUGraph 捕获；
3. routing     npu_moe_init_routing_v2 / npu_moe_finalize_routing 能否被捕获、形状是否静态；
4. e2e         整条链在图内 replay 与 eager 的数值是否一致，各段耗时多少。

用法（16 卡示例）：

    torchrun --nproc_per_node=16 verify_ascend_ops_prefill_graph.py \
        --mode all --bucket 1024 2048 4096 --topk 8 --hidden 7168 \
        --experts 256 --world 16

只算显存、不跑算子（不需要多卡，单进程即可）：

    python verify_ascend_ops_prefill_graph.py --mode account --world 16

所有判定写在结尾的汇总表里：PASS / FAIL / SKIP / INFO。
"""

from __future__ import annotations

import argparse
import os
import time
import traceback
from typing import Callable, List

import torch
import torch.distributed as dist

DTYPES = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}

# Results collected by every rank; only rank 0 prints the table.
RESULTS: List[tuple] = []


def record(name: str, status: str, detail: str = "") -> None:
    RESULTS.append((name, status, detail))
    if os.environ.get("LOCAL_RANK", "0") == "0":
        print(f"[{status:4}] {name}: {detail}", flush=True)


def gb(n: int) -> str:
    return f"{n / 1024 ** 3:.2f} GiB"


def bench(fn: Callable[[], None], warmup: int = 5, iters: int = 20) -> float:
    """Milliseconds per call, device-synchronized."""
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / iters * 1000.0


def capture_graph(run_fn: Callable[[], None], warmup: int = 3) -> "torch.npu.NPUGraph":
    """Warm up on the capture stream, then capture one instance."""
    for _ in range(warmup):
        run_fn()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        run_fn()
    torch.npu.synchronize()
    return graph


def free_mem(device) -> int:
    free, _total = torch.npu.mem_get_info(device)
    return free


# ---------------------------------------------------------------------------
# 1. 显存核算
# ---------------------------------------------------------------------------


def account(*, world: int, bucket: int, topk: int, hidden: int, dtype, note: str) -> None:
    """Peak bytes of the allgather path, computed from the code's tensor shapes."""
    esz = torch.tensor([], dtype=dtype).element_size()
    g_tokens = world * bucket  # gathered rows
    routes = g_tokens * topk  # [gathered, K].numel()

    gathered_x = g_tokens * hidden * esz
    gathered_ids = g_tokens * topk * 4
    gathered_w = g_tokens * topk * 4
    routed_x = routes * hidden * esz  # npu_moe_init_routing_v2 output
    # _allgather_weighted_sum: torch.zeros((num_routes, hidden), float32)
    expert_output_fp32 = routes * hidden * 4
    finalize_out = g_tokens * hidden * 4  # float32, reduce_scatter input
    local_out = bucket * hidden * 4

    peak = (
        gathered_x
        + gathered_ids
        + gathered_w
        + routed_x
        + expert_output_fp32
        + finalize_out
        + local_out
    )
    detail = (
        f"{note} bucket={bucket} gathered={gb(gathered_x)} routed={gb(routed_x)} "
        f"expert_output(fp32)={gb(expert_output_fp32)} peak~{gb(peak)}"
    )
    record(f"account[bucket={bucket}]", "INFO", detail)

    if torch.npu.is_available():
        free = free_mem(torch.npu.current_device())
        ok = peak < free * 0.5
        record(
            f"account-fit[bucket={bucket}]",
            "PASS" if ok else "FAIL",
            f"peak ~{gb(peak)} vs free {gb(free)} (要求 < 50%)",
        )


def try_alloc(*, world: int, bucket: int, topk: int, hidden: int) -> None:
    """真正分配 fp32 expert_output，确认是否 OOM。"""
    routes = world * bucket * topk
    device = torch.npu.current_device()
    before = free_mem(device)
    try:
        t = torch.zeros((routes, hidden), dtype=torch.float32, device=f"npu:{device}")
        torch.npu.synchronize()
        del t
        record(
            "alloc[expert_output fp32]",
            "PASS",
            f"bucket={bucket} routes={routes} 分配成功 (free {gb(before)})",
        )
    except Exception as exc:  # noqa: BLE001
        record(
            "alloc[expert_output fp32]",
            "FAIL",
            f"bucket={bucket} routes={routes} OOM (free {gb(before)}): {exc}",
        )
    torch.npu.empty_cache()


# ---------------------------------------------------------------------------
# 2. 集合通信能否入图
# ---------------------------------------------------------------------------


def test_collective(*, world: int, bucket: int, hidden: int, dtype) -> None:
    device = f"npu:{torch.npu.current_device()}"
    T, H = bucket, hidden

    x = torch.randn(T, H, dtype=dtype, device=device)
    gathered = torch.empty(world * T, H, dtype=dtype, device=device)
    scattered = torch.empty(T, H, dtype=dtype, device=device)

    # eager reference buffers (separate so they never alias graph storage)
    e_gathered = torch.empty(world * T, H, dtype=dtype, device=device)
    e_scattered = torch.empty(T, H, dtype=dtype, device=device)

    def eager(a, out_g, out_s):
        dist.all_gather_into_tensor(out_g, a)
        dist.reduce_scatter_tensor(out_s, out_g)

    def run():
        dist.all_gather_into_tensor(gathered, x)
        dist.reduce_scatter_tensor(scattered, gathered)

    xa = torch.randn_like(x)
    x.copy_(xa)
    eager(x, e_gathered, e_scattered)
    torch.npu.synchronize()

    try:
        graph = capture_graph(run)
        x.copy_(xa)
        scattered.zero_()
        graph.replay()
        torch.npu.synchronize()
        diff = (scattered.float() - e_scattered.float()).abs().max().item()
        ok = diff < (2e-2 if dtype is torch.bfloat16 else 1e-3)
        record(
            f"collective-capture[bucket={bucket}]",
            "PASS" if ok else "FAIL",
            f"all_gather+reduce_scatter 图内可捕获, max|diff|={diff:.3e}",
        )
        record(
            f"collective-time[bucket={bucket}]",
            "INFO",
            f"eager {bench(run):.3f} ms, replay {bench(graph.replay):.3f} ms",
        )
    except Exception as exc:  # noqa: BLE001
        record(
            f"collective-capture[bucket={bucket}]",
            "FAIL",
            f"{type(exc).__name__}: {exc}",
        )


# ---------------------------------------------------------------------------
# 3. routing 算子能否入图
# ---------------------------------------------------------------------------


def _init_routing(x, ids, num_experts):
    # Mirrors npu_moe_init_routing(..., active_num=ids.numel()) from
    # _pre_permute_allgather.
    return torch.ops.npu.npu_moe_init_routing_v2(
        x,
        ids,
        expert_num=num_experts,
        expert_tokens_num_type=1,
        expert_tokens_num_flag=True,
        row_idx_type=0,
        active_expert_range=[0, num_experts],
        quant_mode=-1,
        active_num=ids.numel(),
    )


def _finalize(x, expanded, weights, ids):
    return torch.ops.npu.npu_moe_finalize_routing(
        x.contiguous(),
        skip1=None,
        skip2=None,
        bias=None,
        scales=weights.contiguous(),
        expanded_src_to_dst_row=expanded.to(torch.int32).contiguous(),
        export_for_source_row=ids,
        drop_pad_mode=2,
    )


def test_routing(*, world: int, bucket: int, topk: int, hidden: int, num_experts: int, dtype) -> None:
    device = f"npu:{torch.npu.current_device()}"
    g_tokens = world * bucket
    H = hidden

    x = torch.randn(g_tokens, H, dtype=dtype, device=device)
    ids = torch.randint(0, num_experts, (g_tokens, topk), dtype=torch.int32, device=device)
    w = torch.rand(g_tokens, topk, dtype=torch.float32, device=device)

    # 固定输出缓冲：图只写这里，便于 replay 后比对。
    out = torch.empty(g_tokens, H, dtype=torch.float32, device=device)
    ref = torch.empty_like(out)

    def run(buf):
        r_x, r_idx, _cnt, _ = _init_routing(x, ids, num_experts)
        buf.copy_(_finalize(r_x.float(), r_idx, w, ids))

    try:
        r_x, r_idx, r_cnt, _ = _init_routing(x, ids, num_experts)
        torch.npu.synchronize()
        record(
            "routing-shape",
            "INFO",
            f"init_routing out={tuple(r_x.shape)} (容量形), expert_tokens={tuple(r_cnt.shape)}",
        )
    except Exception as exc:  # noqa: BLE001
        record("routing-eager", "FAIL", f"{type(exc).__name__}: {exc}")
        return

    try:
        graph = capture_graph(lambda: run(out))
        run(ref)  # eager reference on the same inputs
        out.zero_()
        graph.replay()
        torch.npu.synchronize()
        diff = (out - ref).abs().max().item()
        scale = ref.abs().max().item() + 1e-6
        rel = diff / scale
        ok = rel < 2e-2 if dtype is torch.bfloat16 else rel < 1e-3
        record(
            f"routing-capture[bucket={bucket}]",
            "PASS" if ok else "FAIL",
            f"init_routing + finalize 图内可捕获, replay vs eager rel={rel:.2e}",
        )
        record(
            f"routing-time[bucket={bucket}]",
            "INFO",
            f"eager {bench(lambda: run(ref)):.3f} ms, replay {bench(graph.replay):.3f} ms",
        )
    except Exception as exc:  # noqa: BLE001
        record(
            f"routing-capture[bucket={bucket}]",
            "FAIL",
            f"{type(exc).__name__}: {exc}",
        )


# ---------------------------------------------------------------------------
# 4. 端到端：allgather MoE 链
# ---------------------------------------------------------------------------


def _pseudo_experts(x):
    """Stand-in for the grouped matmul: we only need shape-stable compute."""
    return torch.nn.functional.gelu(x.float()).to(x.dtype)


def _chain(
    *,
    x_local,
    ids_local,
    w_local,
    out,
    bucket,
    topk,
    hidden,
    num_experts,
    num_local_experts,
    rank,
    world,
    dtype,
):
    device = x_local.device
    g_tokens = world * bucket
    H = hidden

    # _dispatch_allgather
    g_x = torch.empty(g_tokens, H, dtype=dtype, device=device)
    dist.all_gather_into_tensor(g_x, x_local)
    g_ids = torch.empty(g_tokens, topk, dtype=torch.int32, device=device)
    dist.all_gather_into_tensor(g_ids, ids_local)
    g_w = torch.empty(g_tokens, topk, dtype=torch.float32, device=device)
    dist.all_gather_into_tensor(g_w, w_local)

    # _pre_permute_allgather
    local_ids = g_ids - rank * num_local_experts
    is_local = (local_ids >= 0) & (local_ids < num_local_experts)
    weights = g_w.masked_fill(~is_local, 0)
    local_ids = local_ids.masked_fill(~is_local, num_local_experts)

    routed, expanded, expert_tokens, _ = _init_routing(g_x, local_ids, num_local_experts)
    experts = _pseudo_experts(routed)

    # _allgather_weighted_sum
    num_routes = local_ids.numel()
    expert_output = torch.zeros((num_routes, H), dtype=torch.float32, device=device)
    expert_output[: experts.shape[0]].copy_(experts.float())
    unused = torch.arange(num_routes, device=device) >= expert_tokens.sum()
    expert_output.masked_fill_(unused[:, None], 0)
    final = _finalize(expert_output, expanded, weights, local_ids)

    # _combine_allgather
    dist.reduce_scatter_tensor(out, final)


def test_e2e(*, world: int, bucket: int, topk: int, hidden: int, num_experts: int,
             dtype, max_bytes: int, force: bool) -> None:
    esz = torch.tensor([], dtype=dtype).element_size()
    routes = world * bucket * topk
    est = (world * bucket * hidden * esz  # gathered
           + routes * hidden * esz         # routed
           + routes * hidden * 4           # fp32 expert_output
           + world * bucket * hidden * 4)  # finalize / reduce_scatter
    if est > max_bytes and not force:
        record(
            f"e2e[bucket={bucket}]",
            "SKIP",
            f"预估 {gb(est)} > 预算 {gb(max_bytes)}，用 --e2e-max-bytes 或 --force 强制",
        )
        return

    device = f"npu:{torch.npu.current_device()}"
    rank = dist.get_rank()
    num_local_experts = max(1, num_experts // world)
    H = hidden

    x = torch.randn(bucket, H, dtype=dtype, device=device)
    ids = torch.randint(0, num_experts, (bucket, topk), dtype=torch.int32, device=device)
    w = torch.rand(bucket, topk, dtype=torch.float32, device=device)

    # 图只写 out；ref 只被 eager 写，两者不共享存储。
    out = torch.empty(bucket, H, dtype=torch.float32, device=device)
    ref = torch.empty_like(out)

    kwargs = dict(
        bucket=bucket,
        topk=topk,
        hidden=H,
        num_experts=num_experts,
        num_local_experts=num_local_experts,
        rank=rank,
        world=world,
        dtype=dtype,
    )

    def run(dst):
        _chain(x_local=x, ids_local=ids, w_local=w, out=dst, **kwargs)

    try:
        graph = capture_graph(lambda: run(out))
        torch.npu.synchronize()

        # Replay with different data, then compare against eager on the same data.
        x2 = torch.randn_like(x)
        ids2 = torch.randint(0, num_experts, (bucket, topk), dtype=torch.int32, device=device)
        w2 = torch.rand(bucket, topk, dtype=torch.float32, device=device)
        x.copy_(x2)
        ids.copy_(ids2)
        w.copy_(w2)

        out.zero_()
        graph.replay()
        torch.npu.synchronize()
        run(ref)  # eager on the same inputs
        torch.npu.synchronize()

        diff = (out - ref).abs().max().item()
        scale = ref.abs().max().item() + 1e-6
        rel = diff / scale
        ok = rel < 2e-2 if dtype is torch.bfloat16 else rel < 1e-3
        record(
            f"e2e[bucket={bucket}]",
            "PASS" if ok else "FAIL",
            f"图内 replay vs eager: max|diff|={diff:.3e} (rel {rel:.2e})",
        )
        record(
            f"e2e-time[bucket={bucket}]",
            "INFO",
            f"eager {bench(lambda: run(ref)):.3f} ms, replay {bench(graph.replay):.3f} ms",
        )
    except Exception as exc:  # noqa: BLE001
        record(f"e2e[bucket={bucket}]", "FAIL", f"{type(exc).__name__}: {exc}")
        traceback.print_exc()
    finally:
        torch.npu.empty_cache()


# ---------------------------------------------------------------------------
# entry
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", default="all",
                   choices=["all", "env", "account", "collective", "routing", "e2e"])
    p.add_argument("--bucket", type=int, nargs="+", default=[1024, 2048, 4096],
                   help="prefill 图 bucket（每 rank token 数）")
    p.add_argument("--topk", type=int, default=8)
    p.add_argument("--hidden", type=int, default=7168)
    p.add_argument("--experts", type=int, default=256)
    p.add_argument("--world", type=int, default=16, help="TP=EP 的 rank 数（单进程跑 account 时用）")
    p.add_argument("--dtype", default="bf16", choices=sorted(DTYPES))
    p.add_argument("--e2e-max-bytes", type=int, default=8 * 1024 ** 3)
    p.add_argument("--force", action="store_true", help="忽略显存预估，强制跑 e2e")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    dtype = DTYPES[args.dtype]
    modes = (
        ["env", "account", "collective", "routing", "e2e"]
        if args.mode == "all"
        else [args.mode]
    )

    import torch_npu  # noqa: F401  (must import before touching NPU tensors)

    distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    if distributed:
        torch.npu.set_device(int(os.environ.get("LOCAL_RANK", "0")))
        dist.init_process_group(backend="hccl")
        world = dist.get_world_size()
        rank = dist.get_rank()
    else:
        world = args.world
        rank = 0

    if rank == 0:
        print(f"== device: {torch.npu.get_device_name(0)} | ranks={world} | dtype={args.dtype}")
        print(f"== buckets={args.bucket} topk={args.topk} hidden={args.hidden} experts={args.experts}")

    if "env" in modes:
        record("env-npu", "INFO", f"{torch.npu.get_device_name(0)}, world={world}")
        # ascend_ops allgather 的前置条件（ascend_ops.py:159-168）
        record(
            "env-tp-eq-ep",
            "PASS" if world == args.world else "CHECK",
            f"distributed world={world}, 期望 TP=EP={args.world}",
        )
        if args.hidden % 64 != 0:
            record("env-mxfp8", "INFO", f"hidden={args.hidden} 不是 64 的倍数，MXFP8 路径不可用")

    for bucket in args.bucket:
        if "account" in modes:
            account(world=world, bucket=bucket, topk=args.topk, hidden=args.hidden,
                    dtype=dtype, note="allgather")
            if rank == 0 and torch.npu.is_available():
                try_alloc(world=world, bucket=bucket, topk=args.topk, hidden=args.hidden)

        if not distributed:
            continue

        if "collective" in modes:
            test_collective(world=world, bucket=bucket, hidden=args.hidden, dtype=dtype)
        if "routing" in modes:
            test_routing(world=world, bucket=bucket, topk=args.topk, hidden=args.hidden,
                         num_experts=args.experts, dtype=dtype)
        if "e2e" in modes:
            test_e2e(world=world, bucket=bucket, topk=args.topk, hidden=args.hidden,
                     num_experts=args.experts, dtype=dtype,
                     max_bytes=args.e2e_max_bytes, force=args.force)

    if distributed:
        dist.barrier()
        dist.destroy_process_group()

    if rank == 0:
        print("\n==== 汇总 ====")
        for name, status, detail in RESULTS:
            print(f"{status:4} | {name:32} | {detail}")
        fails = [r for r in RESULTS if r[1] == "FAIL"]
        print(f"\n结论: {'全部通过' if not fails else f'{len(fails)} 项失败'}")


if __name__ == "__main__":
    main()
