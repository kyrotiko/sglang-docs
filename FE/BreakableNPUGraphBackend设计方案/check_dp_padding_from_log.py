#!/usr/bin/env python3
"""Verify DP MAX_LEN padding for the breakable prefill graph.

For ascend_ops (no SUM_LEN support) `ForwardBatch.prepare_mlp_sync_batch`
forces `DpPaddingMode.MAX_LEN` whenever the prefill graph may replay: every DP
rank is padded to the same captured token bucket.  A scheduling step in which
only a few DP ranks hold real requests therefore still runs the full-group
geometry, and the idle ranks do real work on padding rows.  This is invisible
in the aggregate profile: the graph replay records the padded shape by
construction.

This script finds the evidence in the place where the real token count still
exists -- the scheduler log -- and, optionally, prints the per-forward token
count from the profile so the two can be compared.

It groups "Prefill batch" log lines by wall-clock second, counts the distinct
DP ranks that logged in that second, and flags seconds where fewer than
--dp-size ranks were active while `npu graph` is True.  The padding ratio is
printed next to the group so it is clear how much of the step was wasted.

Controlled verification (the cleanest way to use this): run one benchmark with
fewer requests than DP ranks, e.g. 8 requests on DP=16.  Only 8 ranks will
ever log a prefill batch, so if the log still says `npu graph: True`, the other
8 ranks were padded.  Then:

    python3 check_dp_padding_from_log.py \
        --log server.log --profile /path/to/prof --dp-size 16 \
        --layers 43 --expected-active-ranks 8

The profile section prints `tokens=32768 expected=16384 padding=2.00x` for the
graph run, and `tokens=16384 padding=1.00x` for the eager run.

Usage:
    python3 check_dp_padding_from_log.py --log run.log [--profile DIR]
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

LINE_RE = re.compile(
    r"^\[(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\s+DP(?P<dp>\d+)\b.*?"
    r"#new-token:\s*(?P<nt>\d+).*?npu graph:\s*(?P<graph>True|False)"
)
INIT_RE = re.compile(r"moe[\s_]*init[\s_]*routing", re.I)


def parse_log(path: Path) -> list[tuple[float, int, int, bool]]:
    entries: list[tuple[float, int, int, bool]] = []
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = LINE_RE.search(raw)
        if match is None:
            continue
        ts = datetime.strptime(match["ts"], "%Y-%m-%d %H:%M:%S").replace(
            tzinfo=timezone.utc
        ).timestamp()
        entries.append(
            (ts, int(match["dp"]), int(match["nt"]), match["graph"] == "True")
        )
    return entries


def find_profile_info(root: Path) -> Path | None:
    candidates = [root / "profiler_info.json", root / "raw" / "profiler_info.json"]
    candidates.extend(sorted(root.glob("profiler_info_*.json")))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def profile_window(root: Path) -> tuple[float, float] | None:
    info_path = find_profile_info(root)
    if info_path is None:
        return None
    info = json.loads(info_path.read_text(encoding="utf-8"))
    try:
        end_s = info["end_info"]["collectionTimeEnd"] / 1e9
        duration_s = (
            info["end_info"]["MonotonicTimeEnd"] - info["start_info"]["start_monotonic"]
        ) / 1e9
    except (KeyError, TypeError):
        return None
    return end_s - duration_s, end_s


def pick_tz_offset(
    entries: list[tuple[float, int, int, bool]], window: tuple[float, float]
) -> tuple[float, int]:
    """Return (offset_hours, count_in_window) that best aligns log to profile."""
    best_offset = 0
    best_count = -1
    for offset in range(-12, 15):
        lo, hi = window
        count = sum(
            1
            for ts, _, _, _ in entries
            if lo <= ts + offset * 3600.0 <= hi
        )
        if count > best_count:
            best_offset = offset
            best_count = count
    return float(best_offset), best_count


def find_kernel_csv(root: Path) -> Path | None:
    direct = sorted(root.rglob("ASCEND_PROFILER_OUTPUT/kernel_details*.csv"))
    if direct:
        return direct[0]
    hits = sorted(root.rglob("kernel_details*.csv"))
    return hits[0] if hits else None


def first_int(text: str) -> int | None:
    match = re.search(r"-?\d+", text or "")
    return int(match.group(0)) if match else None


def forward_tokens(
    root: Path, device: str, layers: int
) -> dict[str, list[int]]:
    """Per-forward `MoeInitRoutingV3` first dimension, keyed by model id."""
    csv_path = find_kernel_csv(root)
    if csv_path is None:
        return {}
    anchors: dict[str, list[tuple[float, str]]] = defaultdict(list)
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            return {}
        header = {name.strip(): name for name in reader.fieldnames}
        cols = {
            key: header.get(name)
            for key, name in (
                ("device", "Device_id"),
                ("model", "Model ID"),
                ("name", "Name"),
                ("start", "Start Time(us)"),
                ("shape", "Input Shapes"),
            )
        }
        if any(value is None for value in cols.values()):
            return {}
        for raw in reader:
            if raw[cols["device"]].strip() != device:
                continue
            if not INIT_RE.search(raw[cols["name"]] or ""):
                continue
            try:
                start = float(raw[cols["start"]])
            except (TypeError, ValueError):
                continue
            anchors[(raw[cols["model"]] or "").strip()].append(
                (start, raw[cols["shape"]] or "")
            )
    out: dict[str, list[int]] = {}
    for model, items in anchors.items():
        items.sort()
        forwards = len(items) // layers
        if forwards == 0:
            continue
        tokens: list[int] = []
        for index in range(forwards):
            value = first_int(items[index * layers][1])
            if value is not None:
                tokens.append(value)
        out[model] = tokens
    return out


def fmt_time(epoch_s: float) -> str:
    return datetime.fromtimestamp(epoch_s, tz=timezone.utc).strftime(
        "%Y-%m-%d %H:%M:%S"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", required=True, help="scheduler log file")
    parser.add_argument(
        "--profile",
        default=None,
        help="optional msprof export root; prints its forward token counts",
    )
    parser.add_argument("--device", default="0", help="device id (default 0)")
    parser.add_argument("--layers", type=int, default=43, help="MoE layers per forward")
    parser.add_argument("--dp-size", type=int, default=16, help="DP group size")
    parser.add_argument(
        "--bucket", type=int, default=2048, help="captured prefill token bucket"
    )
    parser.add_argument(
        "--expected-active-ranks",
        type=int,
        default=None,
        help="controlled run: ranks that really hold tokens (e.g. 8 for 8 reqs)",
    )
    parser.add_argument(
        "--log-tz-offset-hours",
        type=float,
        default=None,
        help="force log timezone offset; default auto-match against the profile",
    )
    args = parser.parse_args()

    log_path = Path(args.log).expanduser().resolve()
    if not log_path.exists():
        raise SystemExit(f"log does not exist: {log_path}")
    entries = parse_log(log_path)
    if not entries:
        raise SystemExit(
            "no 'Prefill batch ... npu graph:' lines found; is this the scheduler log?"
        )

    print(f"log: {log_path}")
    print(f"prefill lines: {len(entries)}")
    print(f"ranks seen  : {sorted({dp for _, dp, _, _ in entries})}")

    # Clock alignment: the profile wall clock is UTC epoch, the log is host
    # local time.  Try every offset and keep the one with the most lines inside
    # the profiling window.
    offset_h = args.log_tz_offset_hours
    if args.profile is not None:
        profile_root = Path(args.profile).expanduser().resolve()
        window = profile_window(profile_root)
        if window is None:
            print("[warn] no profiler_info.json; cannot match clocks", file=sys.stderr)
        else:
            if offset_h is None:
                offset_h, count = pick_tz_offset(entries, window)
                print(
                    f"profile win : {fmt_time(window[0])} .. {fmt_time(window[1])} UTC"
                )
                print(
                    f"tz offset   : {offset_h:+.0f} h (auto, {count} lines in window)"
                )
            else:
                print(f"tz offset   : {offset_h:+.1f} h (forced)")
            lo, hi = window
            inside = [
                (ts + offset_h * 3600.0, dp, nt, graph)
                for ts, dp, nt, graph in entries
            ]
            in_window = [entry for entry in inside if lo <= entry[0] <= hi]
            if in_window:
                entries = in_window
                print(f"lines in profile window: {len(entries)}")
            else:
                print(
                    "[warn] no log lines overlap the profiling window; "
                    "analyzing the whole log",
                    file=sys.stderr,
                )
                entries = inside
    elif offset_h is not None:
        entries = [
            (ts + offset_h * 3600.0, dp, nt, graph)
            for ts, dp, nt, graph in entries
        ]

    entries.sort()

    # Per-rank line counts over the analyzed range.  A rank with zero lines the
    # whole time is conclusively idle, not a second-boundary artifact.
    rank_counts = Counter(dp for _, dp, _, _ in entries)
    missing = [rank for rank in range(args.dp_size) if rank_counts[rank] == 0]
    if missing:
        print(f"ranks with no prefill line at all: {missing}")

    buckets: dict[int, list[tuple[int, int, bool]]] = defaultdict(list)
    for ts, dp, nt, graph in entries:
        buckets[int(ts)].append((dp, nt, graph))

    header = (
        "time(UTC)   lines ranks  nt_min nt_max  graph_t graph_f   "
        "pad_ratio"
    )
    print()
    print(header)
    print("-" * len(header))
    sparse_graph: list[tuple[int, int, float]] = []
    for second in sorted(buckets):
        rows = buckets[second]
        ranks = {dp for dp, _, _ in rows}
        nts = [nt for _, nt, _ in rows]
        graph_true = sum(1 for _, _, graph in rows if graph)
        graph_false = len(rows) - graph_true
        ratio = float("nan")
        if graph_true and graph_false == 0 and sum(nts) > 0:
            steps = len(rows) / max(len(ranks), 1)
            geometry = steps * args.dp_size * args.bucket
            ratio = geometry / sum(nts)
            if len(ranks) < args.dp_size:
                sparse_graph.append((second, len(ranks), ratio))
        ratio_text = "-" if ratio != ratio else f"{ratio:.2f}x"
        print(
            f"{fmt_time(second)}  {len(rows):5d} {len(ranks):5d} "
            f"{min(nts):6d} {max(nts):6d} {graph_true:8d} {graph_false:7d}   "
            f"{ratio_text}"
        )

    print()
    if sparse_graph:
        print(
            f"VERDICT: {len(sparse_graph)} second(s) with < {args.dp_size} active "
            "DP ranks while the prefill graph was ON."
        )
        worst = max(sparse_graph, key=lambda item: item[2])
        print(
            f"  worst: {fmt_time(worst[0])} ranks={worst[1]} "
            f"padding={worst[2]:.2f}x "
            f"(geometry {args.dp_size}*{args.bucket} vs real rows)"
        )
        print(
            "  => MAX_LEN padding confirmed: idle ranks were padded to the bucket."
        )
    else:
        print(
            "VERDICT: no sparse seconds with graph=True found in the analyzed "
            "range."
        )
        print(
            "  A controlled run with fewer requests than DP ranks makes this "
            "unambiguous."
        )

    if args.profile is not None:
        profile_root = Path(args.profile).expanduser().resolve()
        tokens = forward_tokens(profile_root, args.device, args.layers)
        print()
        print("per-forward tokens (MoeInitRoutingV3 first dim):")
        if not tokens:
            print("  (kernel_details.csv not found; run on the raw profile root)")
        for model, values in sorted(tokens.items()):
            pretty = ", ".join(str(value) for value in values)
            print(f"  model {model}: {len(values)} forward(s): {pretty}")
            if args.expected_active_ranks is not None:
                expected = args.expected_active_ranks * args.bucket
                for index, value in enumerate(values):
                    ratio = value / expected if expected else float("nan")
                    print(
                        f"    fwd {index}: tokens={value} expected={expected} "
                        f"padding={ratio:.2f}x"
                    )


if __name__ == "__main__":
    main()
