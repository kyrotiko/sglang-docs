#!/usr/bin/env python3
"""Probe the AIV kv_exchange full-page transfer on Ascend A5.

This validates the design assumption used by the revised HiCache L2 plan:

    device indexer: [layers, pages*I, K=64, 1, dim]
      --reshape-->   [layers, pages, S=128, 1, dim]
    host page_first: [pages, layers, S=128, 1, dim]
    transfer page_size = S = 128

It also runs the same payload with page_size=64 (one native page per logical
page) as a control.  The operator must produce byte-identical output; a failure
on page_size=128 means the full-page transfer design cannot be used as written.

Run on the A5 machine from the sglang-docs directory:

    python a5_kv_exchange_probe.py --impl ascendc_aiv --out probe_out

Use --impl torch_native for a reference run that does not use the AIV kernel.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from datetime import datetime
from pathlib import Path


def log(msg: str) -> None:
    print(msg, flush=True)


def env_info() -> dict:
    info = {}
    try:
        import torch

        info["torch"] = torch.__version__
        info["npu_available"] = bool(torch.npu.is_available())
        if info["npu_available"]:
            info["npu_device"] = torch.npu.get_device_name(0)
            info["npu_mem_info"] = list(torch.npu.mem_get_info(0))
    except Exception as exc:  # pragma: no cover - environment probe
        info["torch_error"] = repr(exc)
    try:
        import torch_npu

        info["torch_npu"] = torch_npu.__version__
    except Exception as exc:  # pragma: no cover
        info["torch_npu_error"] = repr(exc)
    try:
        import sglang

        info["sglang"] = getattr(sglang, "__version__", "unknown")
    except Exception as exc:  # pragma: no cover
        info["sglang_error"] = repr(exc)
    try:
        from sglang.srt.mem_cache.deepseek_v4_memory_pool import (
            DSV41_INDEX_PAGE_SIZE,
        )

        info["DSV41_INDEX_PAGE_SIZE"] = DSV41_INDEX_PAGE_SIZE
    except Exception as exc:  # pragma: no cover
        info["DSV41_INDEX_PAGE_SIZE_error"] = repr(exc)
    try:
        import sgl_kernel_npu

        info["sgl_kernel_npu"] = getattr(sgl_kernel_npu, "__version__", "unknown")
    except Exception as exc:  # pragma: no cover
        info["sgl_kernel_npu_error"] = repr(exc)
    return info


def register_host(host) -> str | None:
    """Register a CPU tensor for AIV.  Returns the backend name or None."""
    try:
        from sgl_kernel_npu.operators.mem_cache.memfabric.host_memory import (
            register_acl_host_memory,
        )

        register_acl_host_memory(host)
        return "acl"
    except Exception as exc:  # pragma: no cover - depends on A5 install
        log(f"register_acl_host_memory failed: {exc!r}")
        return None


def make_payload(*, layers: int, pages: int, page_size: int, native_page_size: int,
                 dim: int):
    import torch
    import torch_npu  # noqa: F401  (registers torch.npu)

    if page_size % native_page_size != 0:
        raise ValueError(
            f"page_size={page_size} is not tiled by native pages of "
            f"{native_page_size}"
        )
    native_pages = pages * (page_size // native_page_size)
    total = layers * native_pages * native_page_size * dim
    raw = (
        torch.arange(total, dtype=torch.int64)
        .remainder(251)
        .to(torch.uint8)
        .reshape(layers, native_pages, native_page_size, 1, dim)
    )
    device = raw.npu()
    device_view = device.reshape(layers, pages, page_size, 1, dim)
    expected_host = (
        raw.reshape(layers, pages, page_size, 1, dim)
        .permute(1, 0, 2, 3, 4)
        .contiguous()
    )
    return device, device_view, expected_host


def run_case(*, impl: str, pages: int, page_size: int, native_page_size: int,
             dim: int, layers: int) -> dict:
    import torch
    import torch_npu  # noqa: F401  (registers torch.npu)

    from sgl_kernel_npu.operators.mem_cache import (
        transfer_kv_all_layer_kv_split_lf_pf as backup,
        transfer_kv_per_layer_kv_split_pf_lf as load,
    )

    result = {
        "impl": impl,
        "pages": pages,
        "page_size": page_size,
        "native_page_size": native_page_size,
        "dim": dim,
        "layers": layers,
        "status": "FAIL",
    }
    host = torch.zeros((pages, layers, page_size, 1, dim), dtype=torch.uint8)
    if impl == "ascendc_aiv":
        backend = register_host(host)
        result["host_registration"] = backend
        if backend is None:
            result["error"] = "host registration failed"
            return result

    device, device_view, expected_host = make_payload(
        layers=layers,
        pages=pages,
        page_size=page_size,
        native_page_size=native_page_size,
        dim=dim,
    )
    # Both index tensors must live on the same NPU as device_k; host_k stays
    # on CPU.  This is an operator contract, not a host-buffer placement.
    indices = torch.arange(
        pages * page_size, dtype=torch.int64, device=device.device
    )
    device_indices = indices
    host_indices = indices.clone()
    empty_device = torch.empty(0, dtype=torch.uint8, device=device.device)
    empty_host = torch.empty(0, dtype=torch.uint8)

    # D2H: every layer's native pages for logical page p are contiguous in the
    # reshaped full-page view.
    backup(
        device_indices,
        host_indices,
        device_view,
        host,
        empty_device,
        empty_host,
        page_size=page_size,
        host_layer_start=0,
        impl=impl,
    )
    torch.npu.synchronize()
    d2h_ok = torch.equal(host, expected_host)
    result["d2h_ok"] = d2h_ok

    # H2D: load from the already-registered host buffer.  Do not pass a fresh
    # permuted copy here: AIV resolves host addresses from the registration
    # table, so every host operand must be the registered tensor itself.
    device_out = torch.zeros_like(device_view)
    load(
        device_indices,
        host_indices,
        device_out,
        host,
        empty_device,
        empty_host,
        page_size=page_size,
        device_layer_id=0,
        host_layer_id=0,
        impl=impl,
    )
    torch.npu.synchronize()
    h2d_ok = torch.equal(device_out.cpu(), device_view.cpu())
    result["h2d_ok"] = h2d_ok
    result["status"] = "PASS" if d2h_ok and h2d_ok else "FAIL"
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--impl", choices=("ascendc_aiv", "torch_native"),
                        default="ascendc_aiv")
    parser.add_argument("--pages", type=int, default=2)
    parser.add_argument("--dim", type=int, default=64,
                        help="indexer K row width in bytes (V4.1 = 64)")
    parser.add_argument("--native-page-size", type=int, default=64,
                        help="V4.1 DSV41_INDEX_PAGE_SIZE on A5")
    parser.add_argument("--page-sizes", type=int, nargs="+", default=[64, 128],
                        help="64 is the control; 128 is the full-page design")
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    report = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "argv": sys.argv,
        "env": env_info(),
        "cases": [],
    }
    try:
        for page_size in args.page_sizes:
            log(
                f"\n=== {args.impl}: page_size={page_size}, "
                f"native={args.native_page_size}, dim={args.dim} ==="
            )
            report["cases"].append(
                run_case(
                    impl=args.impl,
                    pages=args.pages,
                    page_size=page_size,
                    native_page_size=args.native_page_size,
                    dim=args.dim,
                    layers=args.layers,
                )
            )
    except Exception:
        report["traceback"] = traceback.format_exc()
        log(report["traceback"])
    finally:
        out = args.out
        if out is None:
            out = Path("probe_out") / (
                f"kv_exchange_{args.impl}_{datetime.now():%Y%m%d_%H%M%S}.json"
            )
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
        log(f"\nreport: {out.resolve()}")

    for case in report["cases"]:
        log(
            f"{case['impl']} page_size={case['page_size']}: "
            f"{case['status']} (d2h={case.get('d2h_ok')}, "
            f"h2d={case.get('h2d_ok')})"
        )
    return 0 if report["cases"] and all(
        case["status"] == "PASS" for case in report["cases"]
    ) else 1


if __name__ == "__main__":
    raise SystemExit(main())
