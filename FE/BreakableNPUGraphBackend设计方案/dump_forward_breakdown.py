#!/usr/bin/env python3
"""Per-forward kernel breakdown for an Ascend msprof export.

The msprof aggregate reports mix every forward in the profiling window into
one row set.  That is fine for a single steady-state workload, but a profile
window that contains, say, eight target prefill forwards plus two forwards of
another phase looks identical to ten prefill forwards in the aggregate.

Every MoE layer calls ``MoeInitRoutingV3`` exactly once, so with ``--layers``
we can split the kernel timeline into forwards: every ``--layers`` init-routing
calls is one forward.  Boundaries between forwards are placed at the midpoint
between the last layer's init-routing call of one forward and the first one of
the next, which keeps each forward's attention prefix and LM-head tail inside
its own window.

Usage:
    python3 dump_forward_breakdown.py --profile /path/to/prof --layers 43
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

INIT_RE = re.compile(r"moe[\s_]*init[\s_]*routing", re.I)
ATTN_RE = re.compile(r"sparseattnsharedkv", re.I)
INDEXER_RE = re.compile(r"lightningindexer", re.I)


def find_kernel_csv(root: Path) -> Path | None:
    direct = sorted(root.rglob("ASCEND_PROFILER_OUTPUT/kernel_details*.csv"))
    if direct:
        return direct[0]
    hits = sorted(root.rglob("kernel_details*.csv"))
    return hits[0] if hits else None


def to_float(value: str) -> float | None:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def first_int(text: str) -> int | None:
    match = re.search(r"-?\d+", text or "")
    return int(match.group(0)) if match else None


def family(name: str, op_type: str) -> str:
    """Coarse bucket used by the per-forward table."""
    text = f"{name} {op_type}".lower()
    if "aicpukernel" in text:
        if "allgather" in text:
            return "aicpu_ag"
        if "reducescatter" in text or "reduce_scatter" in text:
            return "aicpu_rs"
        return "aicpu_other"
    if "hcom" in text or "allgather_" in text:
        if "reducescatter" in text or "reduce_scatter" in text:
            return "hcom_rs"
        if "allgather" in text:
            return "hcom_ag"
        return "hcom_other"
    return "compute"


class Row:
    __slots__ = ("model", "name", "op_type", "start", "dur", "wait", "shape", "kind")

    def __init__(self, model, name, op_type, start, dur, wait, shape):
        self.model = model
        self.name = name
        self.op_type = op_type
        self.start = start
        self.dur = dur
        self.wait = wait
        self.shape = shape
        self.kind = family(name, op_type)


def read_rows(path: Path, device: str) -> list[Row]:
    rows: list[Row] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        # Trim spaces that msprof sometimes leaves in header names.
        if reader.fieldnames is None:
            return rows
        header = {name.strip(): name for name in reader.fieldnames}
        columns = {
            "device": header.get("Device_id"),
            "model": header.get("Model ID"),
            "name": header.get("Name"),
            "type": header.get("Type"),
            "start": header.get("Start Time(us)"),
            "dur": header.get("Duration(us)"),
            "wait": header.get("Wait Time(us)"),
            "shape": header.get("Input Shapes"),
        }
        missing = [key for key, value in columns.items() if value is None]
        if missing:
            raise SystemExit(f"kernel_details is missing columns: {missing}")
        seen: set[tuple] = set()
        for raw in reader:
            if columns["device"] and raw[columns["device"]].strip() != device:
                continue
            start = to_float(raw[columns["start"]])
            dur = to_float(raw[columns["dur"]])
            if start is None or dur is None:
                continue
            name = (raw[columns["name"]] or "").strip()
            op_type = (raw[columns["type"]] or "").strip()
            row = Row(
                (raw[columns["model"]] or "").strip(),
                name,
                op_type,
                start,
                dur,
                to_float(raw[columns["wait"]]),
                raw[columns["shape"]] or "",
            )
            # msprof can emit the same HCOM/AICPU record twice (device +
            # analyze point of view).  Drop exact repeats before counting.
            if row.kind.startswith(("hcom", "aicpu")):
                key = (row.model, row.name, round(row.start, 1), round(row.dur, 1))
                if key in seen:
                    continue
                seen.add(key)
            rows.append(row)
    return rows


def split_forwards(rows: list[Row], layers: int, model: str):
    anchors = sorted((row for row in rows if INIT_RE.search(row.name)), key=lambda r: r.start)
    if not anchors:
        raise SystemExit(f"no MoeInitRouting rows for model {model}")
    if layers <= 0:
        raise SystemExit("--layers must be > 0")
    count = len(anchors) // layers
    remainder = len(anchors) % layers
    if count == 0:
        raise SystemExit(
            f"only {len(anchors)} init-routing calls; cannot form one {layers}-layer forward"
        )
    if remainder:
        print(
            f"[warn] model {model}: {len(anchors)} init-routing calls, "
            f"{remainder} left over",
            file=sys.stderr,
        )

    # Boundary between forward i and i+1: midpoint of the gap between the
    # last init-routing call of forward i and the first of forward i+1.
    boundaries = [min(row.start for row in rows)]
    for index in range(1, count + 1):
        if index == count:
            boundaries.append(max(row.start + row.dur for row in rows) + 1.0)
        else:
            left = anchors[index * layers - 1]
            right = anchors[index * layers]
            boundaries.append((left.start + right.start) / 2.0)
    return anchors, boundaries


def fmt_ms(value: float) -> str:
    return f"{value:.1f}"


def union_ms(intervals: list[tuple[float, float]]) -> float:
    """Wall-clock covered by at least one kernel (kernels overlap)."""
    if not intervals:
        return 0.0
    intervals.sort()
    total = 0.0
    current_start, current_end = intervals[0]
    for start, end in intervals[1:]:
        if start > current_end:
            total += current_end - current_start
            current_start, current_end = start, end
        else:
            current_end = max(current_end, end)
    return (total + current_end - current_start) / 1000.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True, help="msprof export root")
    parser.add_argument("--device", default="0", help="device id (default 0)")
    parser.add_argument("--layers", type=int, default=43, help="MoE layers per forward")
    parser.add_argument("--out", default=None, help="output text file")
    args = parser.parse_args()

    root = Path(args.profile).expanduser().resolve()
    kernel_csv = find_kernel_csv(root)
    if kernel_csv is None:
        raise SystemExit(f"kernel_details.csv not found under {root}")
    print(f"[info] reading {kernel_csv}", file=sys.stderr)
    rows = read_rows(kernel_csv, args.device)
    print(f"[info] {len(rows)} device rows", file=sys.stderr)

    lines: list[str] = []
    lines.append(f"profile: {root}")
    lines.append(f"kernel : {kernel_csv}")
    lines.append(f"device : {args.device}   layers/forward: {args.layers}")
    lines.append("")

    models = Counter(row.model for row in rows)
    lines.append("models: " + ", ".join(f"{m or '<empty>'}={n}" for m, n in models.most_common()))
    lines.append("")

    headers = [
        "fwd", "model", "tokens", "span_ms", "busy%", "compute", "aicpu_ag", "aicpu_rs",
        "hcom_ag", "hcom_rs", "attn", "init", "gmm", "swiglu_q", "clipped", "dynq", "hcpre",
        "scan_top_ops",
    ]
    lines.append("\t".join(headers))

    for model in models:
        model_rows = [row for row in rows if row.model == model]
        anchors, boundaries = split_forwards(model_rows, args.layers, model)
        forwards = len(boundaries) - 1
        for index in range(forwards):
            start, end = boundaries[index], boundaries[index + 1]
            window = [row for row in model_rows if start <= row.start < end]
            if not window:
                continue
            by_kind: dict[str, float] = defaultdict(float)
            names: Counter = Counter()
            name_ms: dict[str, float] = defaultdict(float)
            for row in window:
                by_kind[row.kind] += row.dur / 1000.0
                names[row.name] += 1
                name_ms[row.name] += row.dur / 1000.0

            init_rows = [row for row in window if INIT_RE.search(row.name)]
            tokens = first_int(init_rows[0].shape) if init_rows else None
            span = (max(row.start + row.dur for row in window) - start) / 1000.0
            busy = union_ms([(row.start, row.start + row.dur) for row in window])
            top = sorted(name_ms.items(), key=lambda item: -item[1])[:3]
            top_text = ";".join(f"{name}:{ms:.1f}ms" for name, ms in top)

            def count(pattern: re.Pattern) -> int:
                return sum(count for name, count in names.items() if pattern.search(name))

            lines.append(
                "\t".join(
                    [
                        str(index),
                        model or "<empty>",
                        str(tokens) if tokens is not None else "?",
                        fmt_ms(span),
                        f"{100.0 * busy / span:.1f}" if span > 0 else "0.0",
                        fmt_ms(by_kind["compute"]),
                        fmt_ms(by_kind["aicpu_ag"]),
                        fmt_ms(by_kind["aicpu_rs"]),
                        fmt_ms(by_kind["hcom_ag"]),
                        fmt_ms(by_kind["hcom_rs"]),
                        str(count(ATTN_RE)),
                        str(count(INIT_RE)),
                        str(sum(c for n, c in names.items() if "groupedmatmul" in n.lower())),
                        str(sum(c for n, c in names.items() if "swiglu_quant" in n.lower())),
                        str(sum(c for n, c in names.items() if "clippedswiglu" in n.lower())),
                        str(sum(c for n, c in names.items() if "dynamicquant" in n.lower())),
                        str(sum(c for n, c in names.items() if "hcpre" in n.lower())),
                        top_text,
                    ]
                )
            )
        lines.append("")

    text = "\n".join(lines) + "\n"
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"[info] wrote {args.out}", file=sys.stderr)
    else:
        sys.stdout.write(text)


if __name__ == "__main__":
    main()
