#!/usr/bin/env python3
"""Extract an Ascend msprof report for graph-vs-eager prefill analysis.

Only the Python standard library is used so the script runs on an offline
intranet host.  ``--profile`` points at the directory exported by msprof; the
relevant CSV/JSON files are discovered recursively.

Outputs, under ``--out`` (default ``./npu_profile_report_<label>``):

  report.txt       readable summary
  aggregates.json  every aggregate, machine readable
  raw/             small source files and schema samples
  run_info.txt     template for the benchmark and server settings

Send report.txt, aggregates.json, raw/ and the filled run_info.txt back.

The kernel pass groups rows by Model ID (captured graphs carry a model ID;
4294967295 is the usual eager/no-model sentinel), then reports per-model
duration, busy/idle, compute/communication overlap, per-op costs, input-shape
signatures (padding), first-dim token volume, and host API time.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import statistics
import sys
from array import array
from collections import Counter
from functools import lru_cache
from pathlib import Path

DEFAULT_LABEL = "graph"
TOP_N = 25
MAX_INTERVAL_ROWS = 3_000_000
MAX_SHAPE_KEYS = 50_000
MAX_FIRST_DIM_VALUES = 30
MAX_REPORT_INVENTORY = 3000
MAX_RAW_LINE = 5000

DUR_BINS = (
    (0.0, 1.0, "<1us"),
    (1.0, 2.0, "1-2us"),
    (2.0, 5.0, "2-5us"),
    (5.0, 10.0, "5-10us"),
    (10.0, 20.0, "10-20us"),
    (20.0, 50.0, "20-50us"),
    (50.0, 100.0, "50-100us"),
    (100.0, 200.0, "100-200us"),
    (200.0, 500.0, "200-500us"),
    (500.0, 1000.0, "0.5-1ms"),
    (1000.0, 2000.0, "1-2ms"),
    (2000.0, 5000.0, "2-5ms"),
    (5000.0, 10000.0, "5-10ms"),
    (10000.0, 20000.0, "10-20ms"),
    (20000.0, 50000.0, "20-50ms"),
    (50000.0, 100000.0, "50-100ms"),
    (100000.0, float("inf"), ">=100ms"),
)

GAP_BINS = (
    (0.0, 0.05, "<0.05ms"),
    (0.05, 0.1, "0.05-0.1ms"),
    (0.1, 0.2, "0.1-0.2ms"),
    (0.2, 0.5, "0.2-0.5ms"),
    (0.5, 1.0, "0.5-1ms"),
    (1.0, 2.0, "1-2ms"),
    (2.0, 5.0, "2-5ms"),
    (5.0, 10.0, "5-10ms"),
    (10.0, 20.0, "10-20ms"),
    (20.0, 50.0, "20-50ms"),
    (50.0, 100.0, "50-100ms"),
    (100.0, float("inf"), ">=100ms"),
)

FAMILY_PATTERNS = (
    ("hcom_allgather", ("hcomallgather",)),
    ("hcom_reducescatter", ("hcomreducescatter",)),
    ("hcom_other", ("hcom",)),
    ("aicpu_allgather", ("allgatheraicpu",)),
    ("aicpu_reducescatter", ("reducescatteraicpu",)),
    ("aicpu_other", ("aicpu",)),
    ("moe_init_routing", ("moeinitrouting", "initrouting")),
    ("moe_finalize", ("moefinalizerouting",)),
    ("moe_token_unpermute", ("tokenunpermute",)),
    ("moe_gmm", ("groupedmatmul", "moegmm", "groupedgemm")),
    ("attention", ("sparseattn", "inferattention", "attention", "mla")),
    ("swiglu", ("swiglu", "silu")),
    ("quant", ("quant",)),
    ("gemm", ("gemm", "matmul")),
)
ANCHOR_FAMILIES = (
    "moe_init_routing",
    "moe_gmm",
    "attention",
    "hcom_allgather",
    "hcom_reducescatter",
)
KEY_FAMILIES = frozenset(
    (
        "moe_init_routing",
        "moe_finalize",
        "moe_token_unpermute",
        "moe_gmm",
        "attention",
        "hcom_allgather",
        "hcom_reducescatter",
        "aicpu_allgather",
        "aicpu_reducescatter",
        "cast",
    )
)
FIRST_DIM_FAMILIES = frozenset(
    (
        "moe_init_routing",
        "moe_gmm",
        "attention",
        "hcom_allgather",
        "hcom_reducescatter",
        "moe_token_unpermute",
        "cast",
    )
)
INTERVAL_CATEGORIES = ("compute", "hcom", "aicpu")


def norm_text(value):
    return re.sub(r"[^a-z0-9]+", "", (value or "").lower())


@lru_cache(maxsize=65536)
def family_of(name, op_type):
    key = norm_text(f"{name} {op_type}")
    if (
        key
        and "broadcast" not in key
        and (key.startswith("cast") or key.endswith("cast"))
    ):
        return "cast"
    for family, needles in FAMILY_PATTERNS:
        for needle in needles:
            if needle in key:
                return family
    return "other"


def interval_category(family):
    if family.startswith("hcom_"):
        return "hcom"
    if family.startswith("aicpu_"):
        return "aicpu"
    return "compute"


def collective_kind(name, op_type, family):
    key = norm_text(f"{name} {op_type}")
    if "allgather" in key:
        return "allgather"
    if "reducescatter" in key:
        return "reducescatter"
    return family


def norm_header(value):
    value = re.sub(r"\([^)]*\)", "", value or "")
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def is_ratio_header(value):
    tokens = re.split(r"[^a-z0-9]+", (value or "").lower())
    return any(token == "ratio" or token.endswith("ratio") for token in tokens)


def col_index(header, *candidates):
    normalized = [norm_header(item) for item in header]
    for candidate in candidates:
        for index, item in enumerate(normalized):
            if item == candidate:
                return index
    for candidate in candidates:
        for index, item in enumerate(normalized):
            if item.startswith(candidate):
                return index
    return None


def get_field(row, index):
    if index is None or index >= len(row):
        return ""
    return row[index]


def to_float(value):
    if value is None:
        return None
    text = str(value).strip().replace(",", "")
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def norm_device(value):
    text = (value or "").strip()
    try:
        return str(int(float(text)))
    except ValueError:
        return text or "unknown"


def norm_model(value):
    text = (value or "").strip()
    if not text:
        return "unknown"
    try:
        return str(int(float(text)))
    except ValueError:
        try:
            return str(int(text, 0))
        except ValueError:
            return text


SHAPE_GROUP = re.compile(r"[\[\(\{]([^\]\)\}]*)[\]\)\}]")


def parse_dims(raw):
    dims = []
    for token in re.split(r"[,\s]+", raw.strip()):
        if not token:
            continue
        try:
            dims.append(int(token))
        except ValueError:
            pass
    return dims


def shape_groups(value):
    text = (value or "").strip()
    groups = [parse_dims(raw) for raw in SHAPE_GROUP.findall(text)]
    groups = [group for group in groups if group]
    if groups:
        return groups
    # Ascend also exports shapes as "196608,4096;16,4096,4096", tensors
    # separated by semicolons with no brackets.
    if ";" in text:
        return [
            group for group in (parse_dims(raw) for raw in text.split(";")) if group
        ]
    return []


def first_dim(value):
    groups = shape_groups(value)
    if not groups or not groups[0]:
        return None
    return groups[0][0]


def hist_add(counter, value, bins):
    for low, high, label in bins:
        if value < high or high == float("inf"):
            counter[label] += 1
            return


def hist_json(counter, bins):
    return {label: counter.get(label, 0) for _, _, label in bins}


def bump(counter, key, duration, wait):
    entry = counter.get(key)
    if entry is None:
        counter[key] = [1, duration, wait, duration]
    else:
        entry[0] += 1
        entry[1] += duration
        entry[2] += wait
        if duration > entry[3]:
            entry[3] = duration


def merge_intervals(intervals):
    merged = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if merged and start <= merged[-1][1]:
            if end > merged[-1][1]:
                merged[-1][1] = end
        else:
            merged.append([start, end])
    return merged


def total_len(merged):
    return sum(end - start for start, end in merged)


def intersection_len(left, right):
    total = 0.0
    i = 0
    j = 0
    while i < len(left) and j < len(right):
        start = max(left[i][0], right[j][0])
        end = min(left[i][1], right[j][1])
        if end > start:
            total += end - start
        if left[i][1] < right[j][1]:
            i += 1
        else:
            j += 1
    return total


def percentile(values, fraction):
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(fraction * (len(ordered) - 1)))
    return ordered[index]


def new_model_state():
    return {
        "rows": 0,
        "sum_dur": 0.0,
        "sum_wait": 0.0,
        "min_start": None,
        "max_end": None,
        "ops": {},
        "families": {},
        "streams": Counter(),
        "op_states": Counter(),
        "accelerators": Counter(),
        "dur_hist": Counter(),
        "shapes": {},
        "shape_overflow": 0,
        "first_dims": {},
        "pmu": {},
        "anchors": {},
        "intervals": {
            category: (array("d"), array("d")) for category in INTERVAL_CATEGORIES
        },
        "interval_rows": 0,
        "interval_overflow": False,
    }


def discover_files(root):
    patterns = {
        "kernel_details": ("kernel_details*.csv",),
        "step_trace": ("step_trace_time*.csv",),
        "api_statistic": ("api_statistic*.csv",),
        "op_statistic": ("op_statistic*.csv",),
        "communication": ("communication.json",),
        "communication_matrix": ("communication_matrix.json",),
        "profiler_info": ("profiler_info*.json",),
        "info": ("info.json",),
        "op_summary": ("op_summary*.csv",),
        "trace_view": ("trace_view.json",),
        "memory_record": ("memory_record*.csv",),
        "operator_memory": ("operator_memory*.csv",),
    }
    found = {key: [] for key in patterns}
    inventory = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        try:
            size = path.stat().st_size
        except OSError:
            continue
        inventory.append((str(path.relative_to(root)), size))
        name = path.name
        for key, candidates in patterns.items():
            if any(Path(name).match(pattern) for pattern in candidates):
                found[key].append((size, path))
    inventory.sort(key=lambda item: (-item[1], item[0]))
    best = {}
    for key, items in found.items():
        best[key] = max(items, key=lambda item: item[0])[1] if items else None
    return best, inventory


def read_step_trace(path):
    if path is None:
        return None
    with open(path, newline="", encoding="utf-8-sig", errors="replace") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        rows = [row for row in reader if row]
    if header is None:
        return None
    sums = {}
    for index, title in enumerate(header):
        values = []
        for row in rows:
            value = to_float(get_field(row, index))
            if value is not None:
                values.append(value)
        if values:
            sums[f"{index}:{title}"] = sum(values)
    return {"path": str(path), "header": header, "rows": rows, "sums": sums}


def analyze_api(path):
    if path is None:
        return None
    with open(path, newline="", encoding="utf-8-sig", errors="replace") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        rows = [row for row in reader if row]
    if header is None:
        return None
    name_index = col_index(header, "apiname", "api", "name")
    time_index = col_index(header, "totaltime", "time", "duration")
    count_index = col_index(header, "count", "calls")
    average_index = col_index(header, "average", "avg", "meantime")
    level_index = col_index(header, "level")
    if name_index is None:
        return {
            "path": str(path),
            "header": header,
            "rows": rows[:20],
            "aggregated": False,
        }
    stats = {}
    for row in rows:
        name = get_field(row, name_index).strip() or "unknown"
        duration = (
            to_float(get_field(row, time_index)) if time_index is not None else None
        )
        count = (
            to_float(get_field(row, count_index)) if count_index is not None else None
        )
        average = (
            to_float(get_field(row, average_index))
            if average_index is not None
            else None
        )
        if duration is None and average is not None and count is not None:
            duration = average * count
        entry = stats.setdefault(name, [0, 0.0, 0.0, 0.0])
        entry[0] += int(count) if count is not None else 1
        if duration is not None:
            entry[1] += duration
        if average is not None:
            entry[2] = max(entry[2], average)
        if duration is not None:
            entry[3] = max(entry[3], duration)
    top = sorted(stats.items(), key=lambda item: item[1][1], reverse=True)[:40]
    total = sum(entry[1] for entry in stats.values())
    keywords = (
        "graph",
        "replay",
        "launch",
        "execute",
        "sync",
        "memcpy",
        "malloc",
        "free",
        "event",
        "stream",
    )
    interesting = [
        (name, entry)
        for name, entry in sorted(
            stats.items(), key=lambda item: item[1][1], reverse=True
        )
        if any(keyword in name.lower() for keyword in keywords)
    ][:40]
    levels = Counter()
    if level_index is not None:
        for row in rows:
            levels[get_field(row, level_index).strip() or "unknown"] += 1
    return {
        "path": str(path),
        "header": header,
        "aggregated": True,
        "total_us": total,
        "top": top,
        "interesting": interesting,
        "levels": dict(levels),
        "rows": len(rows),
    }


def analyze_op_statistic(path):
    if path is None:
        return None
    with open(path, newline="", encoding="utf-8-sig", errors="replace") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        rows = [row for row in reader if row][:40]
    if header is None:
        return None
    return {"path": str(path), "header": header, "rows": rows}


def analyze_simple_csv(path, max_rows=8):
    if path is None:
        return None
    with open(path, newline="", encoding="utf-8-sig", errors="replace") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        rows = []
        total = 0
        for row in reader:
            total += 1
            if len(rows) < max_rows:
                rows.append(row)
    return {"path": str(path), "header": header, "rows": rows, "total_rows": total}


def json_preview(value, depth=0, max_depth=3, max_items=30):
    if depth >= max_depth:
        if isinstance(value, dict):
            return f"<dict[{len(value)}]>"
        if isinstance(value, list):
            return f"<list[{len(value)}]>"
        return value
    if isinstance(value, dict):
        preview = {}
        for index, (key, item) in enumerate(value.items()):
            if index >= max_items:
                preview["..."] = f"{len(value) - max_items} more keys"
                break
            preview[str(key)] = json_preview(item, depth + 1, max_depth, max_items)
        return preview
    if isinstance(value, list):
        return [
            json_preview(item, depth + 1, max_depth, max_items)
            for item in value[:max_items]
        ]
    if isinstance(value, float):
        return round(value, 3)
    return value


def comm_kind(name):
    lower = name.lower()
    if "allgather" in lower:
        return "AllGather"
    if "reducescatter" in lower:
        return "ReduceScatter"
    if "allreduce" in lower:
        return "AllReduce"
    if "send" in lower or "recv" in lower:
        return "SendRecv"
    return "Other"


def time_in_ms(info, *needles):
    for key, value in info.items():
        lower = str(key).lower()
        if not any(needle in lower for needle in needles):
            continue
        number = to_float(value)
        if number is None:
            continue
        if "(ms)" in lower:
            return number
        if "(us)" in lower:
            return number / 1000.0
        return number
    return None


def parse_comm_info(info):
    if not isinstance(info, dict):
        return None
    elapse = time_in_ms(info, "elapse", "duration")
    if elapse is None:
        return None
    return {
        "elapse": elapse,
        "wait": time_in_ms(info, "wait", "sync") or 0.0,
        "transit": time_in_ms(info, "transit", "bandwidth") or 0.0,
        "idle": time_in_ms(info, "idle") or 0.0,
    }


def comm_instances(data):
    records = []
    total = None
    for name, value in data.items():
        if not isinstance(value, dict):
            continue
        parsed = parse_comm_info(value.get("Communication Time Info"))
        if parsed is None:
            continue
        entry = {"name": str(name), **parsed}
        if "total" in str(name).lower():
            total = entry
            continue
        entry["kind"] = comm_kind(str(name))
        records.append(entry)
    return records, total


def walk_comm_records(value, path, records):
    if isinstance(value, dict):
        lower = {str(key).lower(): key for key in value}

        def find(*needles):
            for needle in needles:
                for lower_key, original_key in lower.items():
                    if needle in lower_key:
                        return value[original_key]
            return None

        elapse = find("elapse", "duration")
        wait = find("wait", "sync")
        transit = find("transit", "bandwidth")
        idle = find("idle")
        label = find("name", "type", "collective", "opname", "op")
        if isinstance(elapse, (int, float)) and not isinstance(elapse, bool):
            if not isinstance(label, str) or not label:
                label = path[-1] if path else "record"
            records.append(
                {
                    "name": str(label),
                    "kind": comm_kind(str(label)),
                    "elapse": float(elapse),
                    "wait": float(wait) if isinstance(wait, (int, float)) else 0.0,
                    "transit": float(transit)
                    if isinstance(transit, (int, float))
                    else 0.0,
                    "idle": float(idle) if isinstance(idle, (int, float)) else 0.0,
                }
            )
            return
        for key, item in value.items():
            walk_comm_records(item, path + [str(key)], records)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            walk_comm_records(item, path + [str(index)], records)


def summarize_comm(records, total):
    groups = {}
    for record in records:
        group = groups.setdefault(
            record["kind"],
            {
                "count": 0,
                "elapse": 0.0,
                "wait": 0.0,
                "transit": 0.0,
                "idle": 0.0,
                "values": [],
            },
        )
        group["count"] += 1
        group["elapse"] += record["elapse"]
        group["wait"] += record["wait"]
        group["transit"] += record["transit"]
        group["idle"] += record["idle"]
        group["values"].append(record["elapse"])
    for group in groups.values():
        values = group.pop("values")
        group["avg_ms"] = group["elapse"] / group["count"] if group["count"] else 0.0
        group["p90_ms"] = percentile(values, 0.9) if values else 0.0
        group["max_ms"] = max(values) if values else 0.0
    top = sorted(records, key=lambda record: record["elapse"], reverse=True)[:12]
    return groups, top


def analyze_communication(path):
    if path is None:
        return None
    try:
        with open(path, encoding="utf-8-sig", errors="replace") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        return {"path": str(path), "error": str(error)}
    records, total = comm_instances(data)
    schema = "instances"
    if not records:
        records = []
        walk_comm_records(data, [], records)
        schema = "generic"
    groups, top = summarize_comm(records, total)
    return {
        "path": str(path),
        "preview": json_preview(data),
        "schema": schema,
        "instance_count": len(records),
        "groups": groups,
        "total": total,
        "top_instances": top,
    }


def analyze_kernels(path, device_filter):
    result = {
        "path": str(path),
        "header": [],
        "columns": {},
        "rows_total": 0,
        "rows_selected": 0,
        "rows_analyzed": 0,
        "devices": Counter(),
        "models": {},
        "duplicates": {"count": 0, "duration": 0.0, "by_family": Counter()},
        "skipped_rows": 0,
        "pmu_columns": [],
        "error": None,
    }
    with open(path, newline="", encoding="utf-8-sig", errors="replace") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if header is None:
            result["error"] = "empty kernel_details.csv"
            return result
        result["header"] = header
        columns = {
            "device": col_index(header, "deviceid", "device"),
            "model": col_index(header, "modelid", "model"),
            "task": col_index(header, "taskid"),
            "stream": col_index(header, "streamid", "stream"),
            "name": col_index(header, "name", "opname", "kernelname"),
            "type": col_index(header, "type", "optype"),
            "op_state": col_index(header, "opstate", "state"),
            "accelerator": col_index(
                header, "acceleratorcore", "coretype", "accelerator"
            ),
            "start": col_index(header, "starttime", "start"),
            "duration": col_index(header, "duration", "dur"),
            "wait": col_index(header, "waittime", "wait"),
            "input_shapes": col_index(header, "inputshapes", "inputshape"),
            "output_shapes": col_index(header, "outputshapes", "outputshape"),
            "input_dtypes": col_index(header, "inputdatatypes", "inputdtype"),
            "output_dtypes": col_index(header, "outputdatatypes", "outputdtype"),
        }
        if columns["name"] is None and columns["type"] is not None:
            columns["name"] = columns["type"]
        pmu_columns = [
            (index, title)
            for index, title in enumerate(header)
            if is_ratio_header(title)
        ]
        result["pmu_columns"] = [title for _, title in pmu_columns]
        result["columns"] = {key: value for key, value in columns.items()}
        if columns["start"] is None or columns["duration"] is None:
            result["error"] = "kernel_details.csv lacks Start Time or Duration"
            return result

        selected_device = norm_device(device_filter)
        duplicate_seen = set()
        models = result["models"]
        for row_number, row in enumerate(reader, 1):
            if not row:
                continue
            result["rows_total"] += 1
            device = (
                norm_device(get_field(row, columns["device"]))
                if columns["device"] is not None
                else "0"
            )
            result["devices"][device] += 1
            if selected_device != "all" and device != selected_device:
                continue
            start = to_float(get_field(row, columns["start"]))
            duration = to_float(get_field(row, columns["duration"]))
            if start is None or duration is None or duration <= 0:
                result["skipped_rows"] += 1
                continue
            name = get_field(row, columns["name"]).strip()
            op_type = get_field(row, columns["type"]).strip()
            wait = to_float(get_field(row, columns["wait"])) or 0.0
            model_id = norm_model(get_field(row, columns["model"]))
            stream = get_field(row, columns["stream"]).strip() or "unknown"
            op_state = get_field(row, columns["op_state"]).strip() or "unknown"
            accelerator = get_field(row, columns["accelerator"]).strip() or "unknown"
            input_shapes = get_field(row, columns["input_shapes"]).strip()
            output_shapes = get_field(row, columns["output_shapes"]).strip()
            result["rows_selected"] += 1
            family = family_of(name, op_type)
            # The same collective can appear more than once with different
            # names or stream placeholders; identical (kind, start, duration)
            # is one event, and keeping both would double every communication
            # aggregate.
            if family.startswith("hcom_") or family.startswith("aicpu_"):
                duplicate_key = (
                    model_id,
                    collective_kind(name, op_type, family),
                    round(start, 1),
                    round(duration, 1),
                )
                if duplicate_key in duplicate_seen:
                    result["duplicates"]["count"] += 1
                    result["duplicates"]["duration"] += duration
                    result["duplicates"]["by_family"][family] += 1
                    continue
                duplicate_seen.add(duplicate_key)

            model = models.get(model_id)
            if model is None:
                model = models[model_id] = new_model_state()
            result["rows_analyzed"] += 1
            model["rows"] += 1
            model["sum_dur"] += duration
            model["sum_wait"] += wait
            end = start + duration
            if model["min_start"] is None or start < model["min_start"]:
                model["min_start"] = start
            if model["max_end"] is None or end > model["max_end"]:
                model["max_end"] = end
            model["streams"][stream] += 1
            model["op_states"][op_state] += 1
            model["accelerators"][accelerator] += 1
            hist_add(model["dur_hist"], duration, DUR_BINS)

            bump(model["ops"], (name, op_type), duration, wait)
            bump(model["families"], family, duration, wait)
            for index, title in pmu_columns:
                value = to_float(get_field(row, index))
                if value is None:
                    continue
                pmu_entry = model["pmu"].get((family, title))
                if pmu_entry is None:
                    pmu_entry = model["pmu"][(family, title)] = [0.0, 0.0]
                pmu_entry[0] += value * duration
                pmu_entry[1] += duration

            if model["interval_rows"] < MAX_INTERVAL_ROWS:
                category = interval_category(family)
                model["intervals"][category][0].append(start)
                model["intervals"][category][1].append(end)
                model["interval_rows"] += 1
            else:
                model["interval_overflow"] = True

            if family in KEY_FAMILIES:
                key = (name, op_type, input_shapes, output_shapes)
                entry = model["shapes"].get(key)
                if entry is not None:
                    entry[0] += 1
                    entry[1] += duration
                elif len(model["shapes"]) < MAX_SHAPE_KEYS:
                    model["shapes"][key] = [1, duration]
                else:
                    model["shape_overflow"] += 1

            if family in FIRST_DIM_FAMILIES:
                dim = first_dim(input_shapes)
                if dim is not None and dim > 0:
                    stats = model["first_dims"].setdefault(
                        family,
                        {
                            "calls": 0,
                            "sum": 0,
                            "min": dim,
                            "max": dim,
                            "values": Counter(),
                        },
                    )
                    stats["calls"] += 1
                    stats["sum"] += dim
                    stats["min"] = min(stats["min"], dim)
                    stats["max"] = max(stats["max"], dim)
                    stats["values"][dim] += 1

            if family in ANCHOR_FAMILIES:
                model["anchors"].setdefault(family, []).append(start)

            if row_number % 500000 == 0:
                print(
                    f"[progress] kernel rows={row_number} elapsed-model={model_id}",
                    file=sys.stderr,
                    flush=True,
                )
    return result


def summarize_model(model, layers):
    compute = merge_intervals(zip(*model["intervals"]["compute"]))
    hcom = merge_intervals(zip(*model["intervals"]["hcom"]))
    aicpu = merge_intervals(zip(*model["intervals"]["aicpu"]))
    busy = merge_intervals(compute + hcom + aicpu)
    start = model["min_start"] or 0.0
    end = model["max_end"] or start
    span = end - start
    gaps = []
    if busy:
        previous = busy[0][1]
        for gap_start, gap_end in busy[1:]:
            if gap_start > previous:
                gaps.append(gap_start - previous)
            previous = max(previous, gap_end)
    gap_hist = Counter()
    for gap in gaps:
        hist_add(gap_hist, gap / 1000.0, GAP_BINS)
    overlap_compute_hcom = intersection_len(compute, hcom)
    overlap_compute_aicpu = intersection_len(compute, aicpu)
    overlap_hcom_aicpu = intersection_len(hcom, aicpu)
    anchors = {}
    for family, raw_starts in model["anchors"].items():
        if len(raw_starts) < 3:
            continue
        ordered = sorted(raw_starts)
        deltas = [right - left for left, right in zip(ordered, ordered[1:])]
        median = statistics.median(deltas)
        threshold = max(2.5 * median, 500.0)
        largest = sorted(
            zip(ordered[1:], deltas), key=lambda item: item[1], reverse=True
        )[:10]
        anchors[family] = {
            "calls": len(ordered),
            "median_gap_ms": median / 1000.0,
            "p90_gap_ms": percentile(deltas, 0.9) / 1000.0,
            "max_gap_ms": max(deltas) / 1000.0,
            "gaps_above_threshold": sum(1 for delta in deltas if delta > threshold),
            "largest_gaps_ms": [round(delta / 1000.0, 3) for _, delta in largest],
            "largest_gap_at_ms": [round((at - start) / 1000.0, 3) for at, _ in largest],
        }
    forwards = None
    if layers > 0 and "moe_init_routing" in model["families"]:
        forwards = model["families"]["moe_init_routing"][0] / layers
    return {
        "rows": model["rows"],
        "sum_dur_ms": model["sum_dur"] / 1000.0,
        "sum_wait_ms": model["sum_wait"] / 1000.0,
        "span_ms": span / 1000.0,
        "busy_ms": total_len(busy) / 1000.0,
        "busy_pct": 100.0 * total_len(busy) / span if span > 0 else 0.0,
        "compute_ms": total_len(compute) / 1000.0,
        "hcom_ms": total_len(hcom) / 1000.0,
        "aicpu_ms": total_len(aicpu) / 1000.0,
        "overlap_compute_hcom_ms": overlap_compute_hcom / 1000.0,
        "overlap_compute_aicpu_ms": overlap_compute_aicpu / 1000.0,
        "overlap_hcom_aicpu_ms": overlap_hcom_aicpu / 1000.0,
        "hcom_overlap_pct": 100.0 * overlap_compute_hcom / total_len(hcom)
        if total_len(hcom) > 0
        else 0.0,
        "hcom_not_overlapped_ms": (total_len(hcom) - overlap_compute_hcom) / 1000.0,
        "gap_count": len(gaps),
        "gap_sum_ms": sum(gaps) / 1000.0,
        "gap_max_ms": max(gaps) / 1000.0 if gaps else 0.0,
        "gap_hist": hist_json(gap_hist, GAP_BINS),
        "dur_hist": hist_json(model["dur_hist"], DUR_BINS),
        "anchors": anchors,
        "forwards": forwards,
        "interval_overflow": model["interval_overflow"],
    }


def top_ops(operations, limit):
    rows = [
        (name, op_type, value[0], value[1], value[2], value[3])
        for (name, op_type), value in operations.items()
    ]
    rows.sort(key=lambda item: item[3], reverse=True)
    return rows[:limit]


def top_families(families):
    rows = [
        (family, value[0], value[1], value[2], value[3])
        for family, value in families.items()
    ]
    rows.sort(key=lambda item: item[2], reverse=True)
    return rows


def top_shapes(shapes, limit):
    rows = [
        (name, op_type, input_shapes, output_shapes, value[0], value[1])
        for (name, op_type, input_shapes, output_shapes), value in shapes.items()
    ]
    rows.sort(key=lambda item: item[5], reverse=True)
    return rows[:limit]


def first_dim_json(stats):
    values = stats["values"]
    top = values.most_common(MAX_FIRST_DIM_VALUES)
    return {
        "calls": stats["calls"],
        "sum": stats["sum"],
        "avg": stats["sum"] / stats["calls"] if stats["calls"] else 0.0,
        "min": stats["min"],
        "max": stats["max"],
        "top_values": [[value, count] for value, count in top],
    }


def pmu_json(model):
    grouped = {}
    for (family, title), (weighted, duration) in model["pmu"].items():
        if duration <= 0:
            continue
        grouped.setdefault(family, {})[title] = weighted / duration
    return grouped


def model_json(model, summary):
    return {
        "summary": summary,
        "top_ops": [
            {
                "name": name,
                "type": op_type,
                "count": count,
                "sum_ms": duration / 1000.0,
                "wait_ms": wait / 1000.0,
                "max_ms": maximum / 1000.0,
            }
            for name, op_type, count, duration, wait, maximum in top_ops(
                model["ops"], TOP_N
            )
        ],
        "families": {
            family: {
                "count": count,
                "sum_ms": duration / 1000.0,
                "wait_ms": wait / 1000.0,
                "max_ms": maximum / 1000.0,
            }
            for family, count, duration, wait, maximum in top_families(
                model["families"]
            )
        },
        "streams": dict(model["streams"].most_common(20)),
        "op_states": dict(model["op_states"]),
        "accelerators": dict(model["accelerators"]),
        "shapes": [
            {
                "name": name,
                "type": op_type,
                "input": input_shapes,
                "output": output_shapes,
                "count": count,
                "sum_ms": duration / 1000.0,
            }
            for name, op_type, input_shapes, output_shapes, count, duration in top_shapes(
                model["shapes"], TOP_N
            )
        ],
        "shape_overflow": model["shape_overflow"],
        "pmu": pmu_json(model),
        "first_dims": {
            family: first_dim_json(stats)
            for family, stats in model["first_dims"].items()
        },
    }


def fmt_table(lines, headers, rows, widths):
    lines.append("  " + " ".join(str(h).ljust(w) for h, w in zip(headers, widths)))
    lines.append("  " + " ".join("-" * w for w in widths))
    for row in rows:
        lines.append(
            "  " + " ".join(str(value)[:w].ljust(w) for value, w in zip(row, widths))
        )


def build_report(
    label,
    root,
    files,
    inventory,
    step_trace,
    kernel,
    api,
    op_stat,
    comm,
    simple,
    layers,
):
    lines = []

    def add(text=""):
        lines.append(text)

    add("=" * 100)
    add(f"NPU profile report: {label}")
    add(f"profile root : {root}")
    add(f"script       : {Path(__file__).name}")
    add("=" * 100)
    add()
    add("[1] Files")
    for key in (
        "kernel_details",
        "step_trace",
        "api_statistic",
        "op_statistic",
        "communication",
        "communication_matrix",
        "profiler_info",
        "info",
        "op_summary",
        "trace_view",
        "memory_record",
        "operator_memory",
    ):
        path = files.get(key)
        if path is None:
            add(f"  {key:<22} MISSING")
        else:
            add(f"  {key:<22} {path.stat().st_size / 1024**2:10.2f} MB  {path}")
    add(
        f"  inventory entries: {len(inventory)} (top {MAX_REPORT_INVENTORY} in raw/file_inventory.txt)"
    )
    add()

    add("[2] Schemas")
    if kernel and kernel.get("header"):
        add(f"  kernel_details header ({len(kernel['header'])} columns):")
        add(f"    {str(kernel['header'])[:MAX_RAW_LINE]}")
        add(f"    column indices: {kernel['columns']}")
        if kernel.get("error"):
            add(f"    ERROR: {kernel['error']}")
    if step_trace:
        add(f"  step_trace_time header: {step_trace['header']}")
    if api:
        add(f"  api_statistic header: {api.get('header')}")
    if op_stat:
        add(f"  op_statistic header: {op_stat.get('header')}")
    if comm:
        if "error" in comm:
            add(f"  communication.json ERROR: {comm['error']}")
        else:
            add("  communication.json preview:")
            add(json.dumps(comm["preview"], indent=2)[:MAX_RAW_LINE])
    add()

    add("[3] step_trace_time")
    if step_trace is None:
        add("  MISSING")
    else:
        add(f"  rows: {len(step_trace['rows'])}")
        for key, value in step_trace["sums"].items():
            _, title = key.split(":", 1)
            add(f"  sum {title}: {value / 1000.0:.3f} ms")
        for row in step_trace["rows"][:100]:
            add(f"  {row}")
    add()

    add("[4] Kernel overview")
    if kernel is None:
        add("  MISSING")
        add()
        return lines
    add(
        f"  rows in file: {kernel['rows_total']}; selected device: {kernel['rows_selected']}; "
        f"analyzed after dedupe: {kernel['rows_analyzed']}"
    )
    add(f"  devices: {dict(kernel['devices'].most_common())}")
    add(f"  skipped rows (bad start/duration): {kernel['skipped_rows']}")
    duplicates = kernel["duplicates"]
    add(
        f"  duplicate HCOM/AICPU records: {duplicates['count']}, "
        f"{duplicates['duration'] / 1000.0:.3f} ms, "
        f"by family {dict(duplicates['by_family'])}"
    )
    add()
    add("  note: Model ID 4294967295 is the usual eager/no-captured-model sentinel.")
    model_ids = sorted(
        kernel["models"],
        key=lambda item: (0, int(item), "") if item.isdigit() else (1, 0, item),
    )
    model_summaries = {
        model_id: summarize_model(kernel["models"][model_id], layers)
        for model_id in model_ids
    }
    add("  key indicators:")
    indicator_rows = []
    for model_id in model_ids:
        summary = model_summaries[model_id]
        init_stats = kernel["models"][model_id]["first_dims"].get("moe_init_routing")
        indicator_rows.append(
            (
                model_id,
                summary["rows"],
                f"{summary['span_ms']:.3f}",
                f"{summary['busy_pct']:.2f}",
                init_stats["calls"] if init_stats else 0,
                f"{init_stats['sum'] / init_stats['calls']:.1f}" if init_stats else "-",
                init_stats["max"] if init_stats else "-",
                f"{summary['hcom_ms']:.3f}",
                f"{summary['hcom_overlap_pct']:.2f}",
                f"{summary['aicpu_ms']:.3f}",
                f"{summary['compute_ms']:.3f}",
                f"{summary['gap_sum_ms']:.3f}",
                f"{summary['forwards']:.2f}"
                if summary["forwards"] is not None
                else "-",
            )
        )
    fmt_table(
        lines,
        (
            "model",
            "rows",
            "span_ms",
            "busy%",
            "init_calls",
            "init_avg",
            "init_max",
            "hcom_ms",
            "hcom_ovl%",
            "aicpu_ms",
            "compute_ms",
            "gap_ms",
            "fwd",
        ),
        indicator_rows,
        (14, 8, 10, 7, 10, 9, 9, 10, 9, 10, 11, 10, 7),
    )
    add()

    for model_id in model_ids:
        model = kernel["models"][model_id]
        summary = model_summaries[model_id]
        add(f"[5] Model {model_id}")
        add(
            f"  rows={summary['rows']} span={summary['span_ms']:.3f} ms "
            f"kernel-sum={summary['sum_dur_ms']:.3f} ms wait-sum={summary['sum_wait_ms']:.3f} ms"
        )
        add(
            f"  busy={summary['busy_ms']:.3f} ms ({summary['busy_pct']:.2f}%) "
            f"compute={summary['compute_ms']:.3f} ms hcom={summary['hcom_ms']:.3f} ms "
            f"aicpu={summary['aicpu_ms']:.3f} ms"
        )
        add(
            f"  compute&hcom overlap={summary['overlap_compute_hcom_ms']:.3f} ms "
            f"({summary['hcom_overlap_pct']:.2f}% of hcom, "
            f"hcom not overlapped={summary['hcom_not_overlapped_ms']:.3f} ms); "
            f"compute&aicpu={summary['overlap_compute_aicpu_ms']:.3f} ms; "
            f"hcom&aicpu={summary['overlap_hcom_aicpu_ms']:.3f} ms"
        )
        if summary["interval_overflow"]:
            add(
                f"  WARNING: interval aggregation truncated at {MAX_INTERVAL_ROWS} rows"
            )
        add(
            f"  idle gaps: count={summary['gap_count']} sum={summary['gap_sum_ms']:.3f} ms "
            f"max={summary['gap_max_ms']:.3f} ms"
        )
        if summary["forwards"] is not None:
            add(f"  forwards (init routing calls / layers): {summary['forwards']:.2f}")
        add("  streams: " + str(dict(model["streams"].most_common(10))))
        add("  op states: " + str(dict(model["op_states"])))
        add("  duration histogram (us): " + str(summary["dur_hist"]))
        add("  gap histogram (ms): " + str(summary["gap_hist"]))
        pmu = pmu_json(model)
        if pmu:
            add("  PMU duration-weighted ratios:")
            for family in sorted(pmu):
                values = ", ".join(
                    f"{title}={value:.4f}"
                    for title, value in sorted(pmu[family].items())
                )
                add(f"    {family:<22} {values}")
        add("  anchors:")
        for family, stats in summary["anchors"].items():
            add(
                f"    {family:<22} calls={stats['calls']} median_gap={stats['median_gap_ms']:.3f} ms "
                f"p90={stats['p90_gap_ms']:.3f} ms max={stats['max_gap_ms']:.3f} ms "
                f"gaps_above_threshold={stats['gaps_above_threshold']} "
                f"largest={stats['largest_gaps_ms']} at={stats['largest_gap_at_ms']}"
            )
        add("  top ops by total duration:")
        fmt_table(
            lines,
            ("name", "type", "count", "sum_ms", "avg_ms", "max_ms"),
            [
                (
                    name,
                    op_type,
                    count,
                    f"{duration / 1000.0:.3f}",
                    f"{duration / count / 1000.0:.4f}",
                    f"{maximum / 1000.0:.3f}",
                )
                for name, op_type, count, duration, wait, maximum in top_ops(
                    model["ops"], TOP_N
                )
            ],
            (46, 26, 8, 12, 12, 12),
        )
        add("  families:")
        fmt_table(
            lines,
            ("family", "count", "sum_ms", "avg_ms", "max_ms", "wait_ms"),
            [
                (
                    family,
                    count,
                    f"{duration / 1000.0:.3f}",
                    f"{duration / count / 1000.0:.4f}",
                    f"{maximum / 1000.0:.3f}",
                    f"{wait / 1000.0:.3f}",
                )
                for family, count, duration, wait, maximum in top_families(
                    model["families"]
                )
            ],
            (24, 8, 12, 12, 12, 12),
        )
        add("  top input-shape signatures:")
        fmt_table(
            lines,
            ("name", "input", "output", "count", "sum_ms"),
            [
                (name, input_shapes, output_shapes, count, f"{duration / 1000.0:.3f}")
                for name, op_type, input_shapes, output_shapes, count, duration in top_shapes(
                    model["shapes"], TOP_N
                )
            ],
            (34, 30, 30, 8, 12),
        )
        add("  first-dim token volume:")
        for family, stats in sorted(model["first_dims"].items()):
            add(
                f"    {family:<22} calls={stats['calls']} sum={stats['sum']} "
                f"avg={stats['sum'] / stats['calls']:.2f} min={stats['min']} max={stats['max']} "
                f"top={stats['values'].most_common(8)}"
            )
        add()

    add("[6] Global top ops (all models, selected device)")
    global_ops = {}
    for model in kernel["models"].values():
        for key, value in model["ops"].items():
            entry = global_ops.setdefault(key, [0, 0.0, 0.0, 0.0])
            entry[0] += value[0]
            entry[1] += value[1]
            entry[2] += value[2]
            entry[3] = max(entry[3], value[3])
    fmt_table(
        lines,
        ("name", "type", "count", "sum_ms", "avg_ms", "max_ms"),
        [
            (
                name,
                op_type,
                count,
                f"{duration / 1000.0:.3f}",
                f"{duration / count / 1000.0:.4f}",
                f"{maximum / 1000.0:.3f}",
            )
            for name, op_type, count, duration, wait, maximum in top_ops(global_ops, 40)
        ],
        (46, 26, 8, 12, 12, 12),
    )
    add()

    add("[7] communication.json")
    if comm is None:
        add("  MISSING")
    elif "error" in comm:
        add(f"  ERROR: {comm['error']}")
    else:
        add(
            f"  schema={comm['schema']} instances={comm['instance_count']} "
            "(values normalized to ms from the key unit)"
        )
        fmt_table(
            lines,
            (
                "kind",
                "count",
                "elapse_ms",
                "wait_ms",
                "transit_ms",
                "idle_ms",
                "avg_ms",
                "p90_ms",
                "max_ms",
            ),
            [
                (
                    kind,
                    entry["count"],
                    f"{entry['elapse']:.3f}",
                    f"{entry['wait']:.3f}",
                    f"{entry['transit']:.3f}",
                    f"{entry['idle']:.3f}",
                    f"{entry['avg_ms']:.3f}",
                    f"{entry['p90_ms']:.3f}",
                    f"{entry['max_ms']:.3f}",
                )
                for kind, entry in sorted(
                    comm["groups"].items(),
                    key=lambda item: item[1]["elapse"],
                    reverse=True,
                )
            ],
            (14, 8, 12, 12, 12, 12, 10, 10, 10),
        )
        if comm["total"] is not None:
            total = comm["total"]
            add(
                f"  reported total: {total['name']} elapse={total['elapse']:.3f} ms "
                f"wait={total['wait']:.3f} ms transit={total['transit']:.3f} ms "
                f"idle={total['idle']:.3f} ms"
            )
        add("  top instances:")
        for record in comm["top_instances"]:
            add(
                f"    {record['elapse']:10.3f} ms  wait={record['wait']:8.3f} ms  "
                f"{record['name'][:90]}"
            )
        if not comm["groups"]:
            add("  no timing records recognized; see raw copy and schema preview")
    add()

    add("[8] api_statistic.csv (host API time)")
    if api is None:
        add("  MISSING")
    elif not api.get("aggregated"):
        add(f"  schema not recognized; rows: {api.get('rows')}")
    else:
        add(f"  rows={api['rows']} total_api_time={api['total_us'] / 1000.0:.3f} ms")
        add(f"  levels={api['levels']}")
        fmt_table(
            lines,
            ("api", "count", "time_ms", "max_avg_us"),
            [
                (name, entry[0], f"{entry[1] / 1000.0:.3f}", f"{entry[2]:.3f}")
                for name, entry in api["top"]
            ],
            (70, 10, 14, 14),
        )
        add("  graph/launch/sync-related:")
        for name, entry in api["interesting"]:
            add(f"    {name:<70} count={entry[0]} time_ms={entry[1] / 1000.0:.3f}")
    add()

    add("[9] op_statistic.csv")
    if op_stat is None:
        add("  MISSING")
    else:
        add(f"  header: {op_stat['header']}")
        for row in op_stat["rows"]:
            add(f"  {row}")
    add()

    add("[10] Memory / misc files")
    for key, value in simple.items():
        if value is None:
            add(f"  {key}: MISSING")
        else:
            add(f"  {key}: rows={value['total_rows']} header={value['header']}")
    add()
    add("[11] Send back")
    add(
        "  report.txt, aggregates.json, the raw/ directory, and the filled run_info.txt"
    )
    return lines


def build_aggregates(
    label,
    root,
    files,
    inventory,
    step_trace,
    kernel,
    api,
    op_stat,
    comm,
    simple,
    layers,
):
    aggregates = {
        "meta": {
            "label": label,
            "root": str(root),
            "script": Path(__file__).name,
            "layers": layers,
        },
        "files": {key: (str(path) if path else None) for key, path in files.items()},
        "inventory": [
            {"path": path, "bytes": size}
            for path, size in inventory[:MAX_REPORT_INVENTORY]
        ],
        "step_trace": step_trace,
        "kernel": None,
        "communication": comm,
        "api": None,
        "op_statistic": op_stat,
        "simple": simple,
    }
    if api is not None:
        aggregates["api"] = {
            "path": api.get("path"),
            "aggregated": api.get("aggregated"),
            "total_us": api.get("total_us"),
            "top": api.get("top"),
            "interesting": api.get("interesting"),
            "levels": api.get("levels"),
            "rows": api.get("rows"),
        }
    if kernel is not None:
        models = {}
        for model_id, model in kernel["models"].items():
            summary = summarize_model(model, layers)
            models[model_id] = model_json(model, summary)
        aggregates["kernel"] = {
            "path": kernel.get("path"),
            "header": kernel.get("header"),
            "columns": kernel.get("columns"),
            "pmu_columns": kernel.get("pmu_columns"),
            "rows_total": kernel.get("rows_total"),
            "rows_selected": kernel.get("rows_selected"),
            "rows_analyzed": kernel.get("rows_analyzed"),
            "skipped_rows": kernel.get("skipped_rows"),
            "devices": dict(kernel["devices"].most_common()),
            "duplicates": {
                "count": kernel["duplicates"]["count"],
                "duration_ms": kernel["duplicates"]["duration"] / 1000.0,
                "by_family": dict(kernel["duplicates"]["by_family"]),
            },
            "models": models,
            "error": kernel.get("error"),
        }
    return aggregates


def write_raw(out_dir, files, inventory, kernel):
    raw = out_dir / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    inventory_path = raw / "file_inventory.txt"
    with open(inventory_path, "w", encoding="utf-8") as handle:
        for path, size in inventory[:MAX_REPORT_INVENTORY]:
            handle.write(f"{size:>14}  {path}\n")
    copies = (
        ("step_trace", 8 * 1024**2),
        ("communication", 8 * 1024**2),
        ("communication_matrix", 16 * 1024**2),
        ("profiler_info", 512 * 1024),
        ("info", 512 * 1024),
    )
    for key, limit in copies:
        path = files.get(key)
        if path is None or path.stat().st_size > limit:
            continue
        target = raw / f"{key}{path.suffix}"
        try:
            shutil.copyfile(path, target)
        except OSError:
            pass
    for key in ("api_statistic", "op_statistic"):
        path = files.get(key)
        if path is None:
            continue
        try:
            with open(
                path, newline="", encoding="utf-8-sig", errors="replace"
            ) as source:
                reader = csv.reader(source)
                rows = []
                for index, row in enumerate(reader):
                    if index >= 6:
                        break
                    rows.append(row)
            sample = raw / f"{key}_head.txt"
            with open(sample, "w", encoding="utf-8") as target:
                for row in rows:
                    target.write(str(row)[:MAX_RAW_LINE] + "\n")
        except OSError:
            pass
    kernel_path = files.get("kernel_details")
    if kernel_path is not None:
        try:
            with open(
                kernel_path, newline="", encoding="utf-8-sig", errors="replace"
            ) as source:
                reader = csv.reader(source)
                rows = []
                for index, row in enumerate(reader):
                    if index >= 3:
                        break
                    rows.append(row)
            sample = raw / "kernel_details_head.txt"
            with open(sample, "w", encoding="utf-8") as target:
                for row in rows:
                    target.write(str(row)[:MAX_RAW_LINE] + "\n")
        except OSError:
            pass


def write_run_info_template(out_dir):
    path = out_dir / "run_info.txt"
    if path.exists():
        return
    path.write_text(
        """Fill this in and send it back with report.txt and aggregates.json.

mode (graph / eager):
server launch command:
parallelism (DP / TP / EP / CP):
model and quantization:
benchmark input_len / output_len / concurrency / num_prompts:
throughput (req/s or tok/s):
TTFT / TPOT:
profiling command and profiling window:
SGLANG_* env vars that affect prefill graph or MoE:
bucket sizes / max prefill tokens:
notes (warmup, mixed load, request length distribution):
""",
        encoding="utf-8",
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract an Ascend msprof report for graph-vs-eager analysis."
    )
    parser.add_argument("--profile", required=True, help="msprof export root")
    parser.add_argument(
        "--out",
        default=None,
        help="output directory (default ./npu_profile_report_<label>)",
    )
    parser.add_argument("--label", default=DEFAULT_LABEL)
    parser.add_argument(
        "--device",
        default="0",
        help="device id to analyze, or 'all' (default 0)",
    )
    parser.add_argument(
        "--layers",
        type=int,
        default=0,
        help="MoE layers per forward; >0 enables per-forward estimates",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    root = Path(args.profile).expanduser().resolve()
    if not root.exists():
        raise SystemExit(f"profile root does not exist: {root}")
    out_dir = (
        Path(args.out).expanduser().resolve()
        if args.out
        else Path.cwd() / f"npu_profile_report_{args.label}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[info] scanning {root}", file=sys.stderr)
    files, inventory = discover_files(root)
    step_trace = read_step_trace(files.get("step_trace"))
    api = analyze_api(files.get("api_statistic"))
    op_stat = analyze_op_statistic(files.get("op_statistic"))
    comm = analyze_communication(files.get("communication"))
    simple = {
        "memory_record": analyze_simple_csv(files.get("memory_record")),
        "operator_memory": analyze_simple_csv(files.get("operator_memory")),
    }
    kernel = None
    if files.get("kernel_details") is not None:
        print(f"[info] parsing {files['kernel_details']}", file=sys.stderr)
        kernel = analyze_kernels(files["kernel_details"], args.device)
        if kernel.get("rows_selected") == 0 and kernel.get("rows_total", 0) > 0:
            print(
                "[warn] no kernel rows matched --device; "
                f"devices in file: {dict(kernel['devices'].most_common())}",
                file=sys.stderr,
            )
    else:
        print("[warn] kernel_details.csv not found", file=sys.stderr)

    write_raw(out_dir, files, inventory, kernel)
    write_run_info_template(out_dir)

    report = build_report(
        args.label,
        root,
        files,
        inventory,
        step_trace,
        kernel,
        api,
        op_stat,
        comm,
        simple,
        args.layers,
    )
    (out_dir / "report.txt").write_text("\n".join(report) + "\n", encoding="utf-8")
    aggregates = build_aggregates(
        args.label,
        root,
        files,
        inventory,
        step_trace,
        kernel,
        api,
        op_stat,
        comm,
        simple,
        args.layers,
    )
    (out_dir / "aggregates.json").write_text(
        json.dumps(aggregates, indent=2, default=str), encoding="utf-8"
    )
    print(f"[done] report     : {out_dir / 'report.txt'}")
    print(f"[done] aggregates : {out_dir / 'aggregates.json'}")
    print(f"[done] raw files  : {out_dir / 'raw'}")
    print(f"[done] run info   : {out_dir / 'run_info.txt'}")


if __name__ == "__main__":
    main()
