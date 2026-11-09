#!/usr/bin/env python3
"""Summarize per-op row counts from an Ascend ``kernel_details.csv``.

Compares the *real* prefill token count (SparseAttnSharedkv runs eagerly on the
live rows) with the MoE row count (the captured MoE always runs the full
bucket), which yields the prefill-graph padding factor directly.

Usage:
    python prof_op_rows.py <PROFILER_OUTPUT dir | kernel_details.csv> \
        --ops SparseAttnSharedkv MoeFinalizeRoutingV2 GroupedMatmul
"""

from __future__ import annotations

import argparse
import csv
import os
import re
from collections import Counter

_LEADING_INT = re.compile(r"\(\s*(\d+)")


def _first_dim(shape: str) -> int | None:
    m = _LEADING_INT.search(shape or "")
    return int(m.group(1)) if m else None


def _load(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        reader.fieldnames = [n.strip().lstrip("﻿") for n in reader.fieldnames]
        return list(reader)


def _column(rows: list[dict], *candidates: str) -> str:
    fields = rows[0].keys() if rows else ()
    for name in candidates:
        if name in fields:
            return name
    raise SystemExit(f"none of {candidates} in columns {list(fields)[:20]}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="ASCEND_PROFILER_OUTPUT dir or kernel_details.csv")
    ap.add_argument(
        "--ops",
        nargs="+",
        default=["SparseAttnSharedkv", "MoeFinalizeRoutingV2", "GroupedMatmul"],
    )
    ap.add_argument("--top", type=int, default=6)
    args = ap.parse_args()

    csv_path = args.path
    if os.path.isdir(csv_path):
        csv_path = os.path.join(csv_path, "kernel_details.csv")
    rows = _load(csv_path)
    if not rows:
        raise SystemExit("empty csv")

    op_col = _column(rows, "OP Type", "OP Name", "Op Name")
    shape_col = _column(rows, "Input Shapes", "Input Shape")

    summary = {}
    for op in args.ops:
        shapes = Counter()
        total = 0
        for row in rows:
            if row.get(op_col) != op:
                continue
            d = _first_dim(row.get(shape_col, ""))
            if d is None:
                continue
            shapes[d] += 1
            total += d
        if shapes:
            summary[op] = (sum(shapes.values()), total, shapes)

    print(f"{'op':26} {'calls':>6} {'sum(first_dim)':>15}  shapes")
    for op, (calls, total, shapes) in summary.items():
        top = ", ".join(f"{d}x{n}" for d, n in shapes.most_common(args.top))
        print(f"{op:26} {calls:6} {total:15}  {top}")

    attn = summary.get("SparseAttnSharedkv")
    moe = summary.get("MoeFinalizeRoutingV2")
    if attn and moe:
        # MoE rows are global tokens * top_k; the DP size folds into the shape,
        # so report the ratio of the two sums as "MoE rows per attention row".
        print(
            f"\nMoeFinalizeRoutingV2 rows / SparseAttnSharedkv rows = "
            f"{moe[1] / attn[1]:.2f}  (global_tokens * top_k per real token)"
        )


if __name__ == "__main__":
    main()
