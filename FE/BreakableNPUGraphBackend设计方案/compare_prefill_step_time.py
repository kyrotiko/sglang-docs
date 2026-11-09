#!/usr/bin/env python3
"""Compare steady-state prefill step time between graph-on and graph-off logs.

End-to-end TTFT is dominated by queueing, so a 1-2% per-step change disappears
in run-to-run noise.  At a fixed chunk size every scheduler step logs
`... #new-token: N ... npu graph: ... input throughput (token/s): T`, where T
is N / step_time.  The median of T over the steady window is a low-noise
estimate of the per-step cost and is the right metric for a small graph effect.

Run the two server configs alternately (A B A B ...), with everything except
the prefill graph identical, and pass all logs of each side here:

    python3 compare_prefill_step_time.py \
        --graph-log g1.log g2.log g3.log \
        --eager-log e1.log e2.log e3.log \
        --tokens-per-step 2048

Only lines with the requested #new-token are used, and the first/last
--trim-fraction of each rank's steps are dropped to exclude ramp-up/drain.
"""

from __future__ import annotations

import argparse
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

LINE_RE = re.compile(
    r"^\[(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\s+DP(?P<dp>\d+)\b.*?"
    r"#new-token:\s*(?P<nt>\d+).*?"
    r"npu graph:\s*(?P<graph>True|False).*?"
    r"input throughput \(token/s\):\s*(?P<tps>[\d.]+)"
)


def load_side(
    paths: list[Path],
    want_graph: bool,
    tokens_per_step: int,
    trim_fraction: float,
) -> tuple[list[float], list[str], int, int]:
    """Return (throughput samples, per-file notes, total lines, matching lines)."""
    samples: list[float] = []
    notes: list[str] = []
    total_lines = 0
    kept = 0
    for path in paths:
        per_rank: dict[int, list[tuple[str, float]]] = defaultdict(list)
        file_total = 0
        file_flag = 0
        for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
            match = LINE_RE.search(raw)
            if match is None:
                continue
            file_total += 1
            if (match["graph"] == "True") != want_graph:
                continue
            file_flag += 1
            if int(match["nt"]) != tokens_per_step:
                continue
            per_rank[int(match["dp"])].append((match["ts"], float(match["tps"])))

        file_samples: list[float] = []
        for values in per_rank.values():
            values.sort()
            count = len(values)
            if count == 0:
                continue
            cut = int(count * trim_fraction)
            kept_values = values[cut : count - cut] if count - 2 * cut > 0 else values
            file_samples.extend(tps for _, tps in kept_values)

        samples.extend(file_samples)
        kept += len(file_samples)
        total_lines += file_total
        if file_samples:
            median = statistics.median(file_samples)
            notes.append(
                f"  {path.name}: lines={file_total} flag={file_flag} "
                f"steps={len(file_samples)} ranks={len(per_rank)} "
                f"median={median:.1f} tok/s ({tokens_per_step / median * 1000:.1f} ms)"
            )
        else:
            notes.append(f"  {path.name}: no usable steps (lines={file_total})")
    return samples, notes, total_lines, kept


def summarize(label: str, samples: list[float], tokens_per_step: int) -> float | None:
    if not samples:
        print(f"{label}: no samples")
        return None
    median = statistics.median(samples)
    mean = statistics.fmean(samples)
    step_ms = tokens_per_step / median * 1000.0
    if len(samples) >= 5:
        deciles = statistics.quantiles(samples, n=10)
        p10, p90 = deciles[0], deciles[-1]
        spread = f"  p10-p90={tokens_per_step / p10 * 1000:.1f}-{tokens_per_step / p90 * 1000:.1f} ms"
    else:
        spread = ""
    print(
        f"{label}: n={len(samples)}  median={median:.1f} tok/s "
        f"({step_ms:.1f} ms/step)  mean={mean:.1f} tok/s{spread}"
    )
    return step_ms


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph-log", nargs="+", required=True, help="graph-on logs")
    parser.add_argument("--eager-log", nargs="+", required=True, help="graph-off logs")
    parser.add_argument(
        "--tokens-per-step",
        type=int,
        default=2048,
        help="per-rank tokens of a full chunk (default 2048)",
    )
    parser.add_argument(
        "--trim-fraction",
        type=float,
        default=0.1,
        help="drop this fraction of each rank's steps at both ends (default 0.1)",
    )
    args = parser.parse_args()

    graph_paths = [Path(p).expanduser().resolve() for p in args.graph_log]
    eager_paths = [Path(p).expanduser().resolve() for p in args.eager_log]
    for path in graph_paths + eager_paths:
        if not path.exists():
            raise SystemExit(f"log does not exist: {path}")

    graph_samples, graph_notes, graph_total, graph_kept = load_side(
        graph_paths, True, args.tokens_per_step, args.trim_fraction
    )
    eager_samples, eager_notes, eager_total, eager_kept = load_side(
        eager_paths, False, args.tokens_per_step, args.trim_fraction
    )

    print(f"chunk tokens per step: {args.tokens_per_step}")
    print(f"trim fraction per rank : {args.trim_fraction:.0%}")
    print()
    print("graph-on files:")
    print("\n".join(graph_notes) if graph_notes else "  (none)")
    print("graph-off files:")
    print("\n".join(eager_notes) if eager_notes else "  (none)")
    print()

    graph_ms = summarize("GRAPH", graph_samples, args.tokens_per_step)
    eager_ms = summarize("EAGER", eager_samples, args.tokens_per_step)
    print()

    if graph_ms is not None and eager_ms is not None:
        delta = (graph_ms / eager_ms - 1.0) * 100.0
        sign = "faster" if delta < 0 else "slower"
        print(
            f"per-step delta: graph {abs(delta):.2f}% {sign} than eager "
            f"({graph_ms:.1f} vs {eager_ms:.1f} ms/step)"
        )
        print(
            "  report this number with the per-file medians above; if the "
            "per-file spread is comparable to the delta, add more runs."
        )

    print()
    print(
        f"lines scanned: graph={graph_total} (kept {graph_kept}), "
        f"eager={eager_total} (kept {eager_kept})"
    )
    if graph_total and graph_kept == 0:
        print("[warn] no graph=True full-chunk steps; is the graph enabled?", file=sys.stderr)
    if eager_total and eager_kept == 0:
        print("[warn] no graph=False full-chunk steps; is prefill graph disabled?", file=sys.stderr)


if __name__ == "__main__":
    main()
