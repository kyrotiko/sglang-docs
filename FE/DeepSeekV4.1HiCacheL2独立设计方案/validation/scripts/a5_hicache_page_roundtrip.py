#!/usr/bin/env python3
"""Byte-level A5 validation of the V4.1 host-page round trip.

The model's greedy output is not repeatable on this A5 recipe (MoE/DeepEP
numerics), so text comparison cannot prove KV correctness.  This script
bypasses the model:

  - build the same full-page buffers that _dsv4_low_ratio_regions produces
    for SWA, C1/C2 KV, and indexer K/scale;
  - build the real DeepSeekV4PagedHostPool;
  - fill the device pages with deterministic bytes;
  - run the real AIV D2H backup and H2D load (kernel_ascend);
  - compare every byte before and after.

A PASS means the host pool geometry, page indexing, and AIV transfer preserve
the KV/indexer bytes for both ratio 1 and ratio 2.  It is independent of model
sampling and of whether the model output is repeatable.

Run on A5 after the HiCache code is checked out:

    SGLANG_HICACHE_HOST_MEM_BACKEND=acl \
    PYTHONPATH=$HOME/repos/sglang/python \
    python a5_hicache_page_roundtrip.py
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import torch
import torch_npu  # noqa: F401

from sglang.srt.mem_cache.deepseek_v4_buffer_info import DSV4PagedBufferInfo
from sglang.srt.mem_cache.device_pool_info import DevicePoolInfo
from sglang.srt.mem_cache.hicache_storage import PoolName
from sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler import (
    _dsv4_low_ratio_regions,
)
from sglang.srt.mem_cache.memory_pool_host import DeepSeekV4PagedHostPool


PAGE_SIZE = 128


def make_pattern(shape: tuple[int, ...]) -> torch.Tensor:
    total = 1
    for dim in shape:
        total *= dim
    return (
        torch.arange(total, dtype=torch.int64)
        .remainder(251)
        .to(torch.uint8)
        .reshape(shape)
        .npu()
    )


def make_info(name: PoolName, buffer: torch.Tensor, page_slots: int) -> DevicePoolInfo:
    return DevicePoolInfo(
        pool_name=name,
        indices_from_pool=PoolName.KV,
        layer_ids=(0,),
        buffer_info=DSV4PagedBufferInfo.from_buffers(
            [buffer], page_slots=page_slots, layout="pa_nd"
        ),
    )


def build_regions():
    # C1: 2 full pages, native K=64 -> 4 native index pages.
    # C2: 2 full pages, native K=64 -> 2 native index pages.
    buffers = {
        "c1_kv": make_pattern((2, 128, 1, 320)),
        "c1_indexer": make_pattern((4, 64, 1, 64)),
        "c1_scale": make_pattern((4, 64, 1, 4)),
        "c2_kv": make_pattern((2, 64, 1, 320)),
        "c2_indexer": make_pattern((2, 64, 1, 64)),
        "c2_scale": make_pattern((2, 64, 1, 4)),
    }
    infos = {
        PoolName.DEEPSEEK_V4_C1: make_info(
            PoolName.DEEPSEEK_V4_C1, buffers["c1_kv"], 128
        ),
        PoolName.DEEPSEEK_V4_C1_INDEXER: make_info(
            PoolName.DEEPSEEK_V4_C1_INDEXER, buffers["c1_indexer"], 64
        ),
        PoolName.DEEPSEEK_V4_C1_INDEXER_SCALE: make_info(
            PoolName.DEEPSEEK_V4_C1_INDEXER_SCALE, buffers["c1_scale"], 64
        ),
        PoolName.DEEPSEEK_V4_C2: make_info(
            PoolName.DEEPSEEK_V4_C2, buffers["c2_kv"], 64
        ),
        PoolName.DEEPSEEK_V4_C2_INDEXER: make_info(
            PoolName.DEEPSEEK_V4_C2_INDEXER, buffers["c2_indexer"], 64
        ),
        PoolName.DEEPSEEK_V4_C2_INDEXER_SCALE: make_info(
            PoolName.DEEPSEEK_V4_C2_INDEXER_SCALE, buffers["c2_scale"], 64
        ),
    }
    kvcache = SimpleNamespace(
        kv_pools={1: object(), 2: object()},
        index_pools={1: object(), 2: object()},
        start_layer=0,
    )
    return buffers, infos, kvcache


def roundtrip(
    *,
    name: str,
    buffers: list[torch.Tensor],
    slots_per_page: int,
    item_bytes: int,
) -> dict:
    device_buffer = buffers[0]
    expected = device_buffer[0].detach().cpu().clone()
    host = DeepSeekV4PagedHostPool(
        pool_name=name,
        device_buffers=buffers,
        item_bytes=item_bytes,
        num_host_pages=2,
        slot_page_size=PAGE_SIZE,
        layout="page_first",
        pin_memory=True,
        page_aligned_only=True,
    )
    # Indices stay in the logical full-page coordinate space (128); the host
    # pool expands C2/indexer to their native/full-page slots internally.
    device_indices = torch.arange(
        PAGE_SIZE, dtype=torch.int64, device=device_buffer.device
    )
    host_indices = device_indices.clone()
    dummy_pool = SimpleNamespace()

    host.backup_from_device_all_layer(
        dummy_pool, host_indices, device_indices, "kernel_ascend"
    )
    torch.npu.synchronize()
    host_page = host.kv_buffer[0, 0].detach().cpu().clone()
    d2h_ok = torch.equal(host_page.view_as(expected), expected)

    device_buffer[0].zero_()
    host.load_to_device_per_layer(
        dummy_pool,
        host_indices,
        device_indices,
        0,
        "kernel_ascend",
    )
    torch.npu.synchronize()
    h2d_ok = torch.equal(device_buffer[0].detach().cpu(), expected)

    # Keep a small artifact for post-mortem; never dump full model KV.
    mismatch = None
    if not d2h_ok:
        mismatch = int((host_page.view_as(expected) != expected).sum())
    elif not h2d_ok:
        mismatch = int((device_buffer[0].detach().cpu() != expected).sum())

    return {
        "name": name,
        "device_shape": list(device_buffer.shape),
        "slots_per_page": slots_per_page,
        "item_bytes": item_bytes,
        "d2h_ok": d2h_ok,
        "h2d_ok": h2d_ok,
        "mismatch_bytes": mismatch,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()
    out_dir = args.out_dir or (
        Path("probe_out")
        / f"page_roundtrip_{datetime.now():%Y%m%d_%H%M%S}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "argv": sys.argv,
        "cases": [],
    }
    try:
        buffers, infos, kvcache = build_regions()
        regions = {
            name: region_buffers
            for name, _pool, region_buffers, _mapping in _dsv4_low_ratio_regions(
                kvcache, infos, PAGE_SIZE
            )
        }
        # SWA is not part of the low-ratio region helper; build its full page
        # view directly, matching the server's SWA host pool.
        swa = make_pattern((2, 128, 1, 544))

        cases = [
            ("swa", [swa], 128, 128 * 544),
            (
                "c1_kv",
                regions[PoolName.DEEPSEEK_V4_C1],
                128,
                128 * 320,
            ),
            (
                "c1_indexer",
                regions[PoolName.DEEPSEEK_V4_C1_INDEXER],
                128,
                128 * 64,
            ),
            (
                "c1_scale",
                regions[PoolName.DEEPSEEK_V4_C1_INDEXER_SCALE],
                128,
                128 * 4,
            ),
            (
                "c2_kv",
                regions[PoolName.DEEPSEEK_V4_C2],
                64,
                64 * 320,
            ),
            (
                "c2_indexer",
                regions[PoolName.DEEPSEEK_V4_C2_INDEXER],
                64,
                64 * 64,
            ),
            (
                "c2_scale",
                regions[PoolName.DEEPSEEK_V4_C2_INDEXER_SCALE],
                64,
                64 * 4,
            ),
        ]
        for name, case_buffers, slots, item_bytes in cases:
            print(f"\n=== {name}: shape={tuple(case_buffers[0].shape)} ===")
            result = roundtrip(
                name=name,
                buffers=case_buffers,
                slots_per_page=slots,
                item_bytes=item_bytes,
            )
            report["cases"].append(result)
            print(
                f"{name}: d2h={result['d2h_ok']} h2d={result['h2d_ok']} "
                f"mismatch={result['mismatch_bytes']}"
            )
    except Exception:
        report["traceback"] = traceback.format_exc()
        print(report["traceback"])
    finally:
        out = out_dir / "report.json"
        out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
        print(f"\nreport: {out.resolve()}")

    return 0 if report["cases"] and all(
        case["d2h_ok"] and case["h2d_ok"] for case in report["cases"]
    ) else 1


if __name__ == "__main__":
    raise SystemExit(main())
