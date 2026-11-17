#!/usr/bin/env python3
"""Collect single-card A5 evidence for DeepSeek V4.1 prefill graphs.

No model weights are loaded. Each probe runs in a fresh process. The parent
always writes a JSON report, Markdown summary, logs, and a tar.gz bundle.
Run --help or --list without importing torch. See the adjacent README.md.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import inspect
import json
import math
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import tarfile
import time
import traceback
from types import SimpleNamespace


BASELINE = "d3700bf0a9"
SCRIPT_REVISION = 7
PROBES = (
    "environment", "geometry", "metadata", "compressor", "epilog_fp8",
    "epilog_fp4", "prologue", "qli_c2", "candidate_chain",
    "attention_r0", "attention_r1", "attention_r2", "engram_break", "mhc_premix",
    "writer_attention_r0", "writer_attention_r1", "writer_attention_r2",
    "idle_chain_r0", "idle_chain_r1", "idle_chain_r2", "active_chain_r2",
    "compressor_boundary", "active_chain_r2_sync",
)
API_NAMES = {
    "attention": (
        "compressor_v2", "mixed_quant_sparse_flash_mla_dsl",
        "mixed_quant_sparse_flash_mla_metadata_dsl",
    ),
    "indexer": (
        "indexer_prologue_k_dsl", "quant_lightning_indexer_dsl",
        "quant_lightning_indexer_metadata_dsl",
        "quant_sparse_lightning_indexer_dsl",
        "quant_sparse_lightning_indexer_metadata_dsl",
    ),
    "mem_cache": ("kv_compress_epilog_v2",),
    "hyper_conn": ("mhc_pre_dpskv4",),
}
T = None
N = None
O = None
R = None


class Blocked(RuntimeError):
    pass


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def checkpoint(phase, **values):
    R["phase"] = phase
    R.update(values)
    print(f"[{R['probe']}] {phase}", flush=True)
    atomic_json(Path(O.output_dir) / (R["probe"] + ".json"), R)


def command(argv, cwd=None, timeout=15):
    try:
        result = subprocess.run(argv, cwd=cwd, capture_output=True, text=True,
                                errors="replace", timeout=timeout)
        return {"returncode": result.returncode,
                "stdout": result.stdout[-30000:], "stderr": result.stderr[-8000:]}
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def load_device():
    global T, N
    if O.sglang_root:
        source = Path(O.sglang_root).resolve() / "python"
        if not source.is_dir():
            raise Blocked(f"Missing SGLang Python source directory: {source}")
        sys.path.insert(0, str(source))
    try:
        T = importlib.import_module("torch")
        N = importlib.import_module("torch_npu")
    except ImportError as exc:
        raise Blocked(f"Use the A5 serving environment: {exc}") from exc
    if not T.npu.is_available():
        raise Blocked("torch.npu.is_available() is False")
    T.npu.set_device(O.device)
    T.set_grad_enabled(False)
    T.manual_seed(O.seed)
    R["registration_imports"] = []
    for name in O.import_module:
        module = importlib.import_module(name)
        R["registration_imports"].append({"module": name, "file": getattr(module, "__file__", None)})
    R["device"] = T.npu.get_device_name(O.device)
    R["torch"] = T.__version__
    R["torch_npu"] = N.__version__


def api(group, name):
    try:
        module = importlib.import_module("sgl_kernel_npu.operators." + group)
        function = getattr(module, name)
    except (ImportError, AttributeError) as exc:
        raise Blocked(f"Missing deployed API {group}.{name}: {exc}") from exc
    try:
        signature = str(inspect.signature(function))
    except (TypeError, ValueError):
        signature = "unavailable"
    try:
        source = inspect.getsourcefile(function)
    except (TypeError, OSError):
        source = None
    R.setdefault("apis", {})[name] = {
        "signature": signature,
        "module": getattr(function, "__module__", type(function).__module__),
        "file": source,
    }
    return function


def backend():
    return importlib.import_module("sglang.srt.hardware_backend.npu.attention.ascend_dsv4_backend")


def device_tensor(value, dtype=None):
    return T.as_tensor(value, dtype=dtype, device=f"npu:{O.device}")


def random_tensor(shape, seed, scale=0.1, dtype=None):
    generator = T.Generator(device="cpu").manual_seed(seed)
    value = T.randn(shape, generator=generator, dtype=T.float32) * scale
    return value.to(device=f"npu:{O.device}", dtype=dtype or T.bfloat16)


def snapshot(values):
    return {name: value.detach().cpu().clone() for name, value in values.items()}


def comparison_failure(label, name, reference, got, metrics):
    if R is None:
        return
    def sample(tensor):
        result = []
        for value in tensor.reshape(-1)[:12].tolist():
            result.append(str(value) if isinstance(value, float) and not math.isfinite(value) else value)
        return result
    R["comparison_failure"] = {"label": label, "tensor": name, "metrics": metrics,
                               "expected_sample": sample(reference), "actual_sample": sample(got)}
    if reference.shape == got.shape and reference.numel():
        left, right = reference.reshape(-1), got.reshape(-1)
        indices = T.nonzero(left != right, as_tuple=False).reshape(-1)[:32].tolist()
        points = []
        for index in indices:
            offset, coordinates = index, []
            for size in reversed(reference.shape):
                coordinates.append(offset % size)
                offset //= size
            points.append({"flat_index": index, "coordinates": list(reversed(coordinates)),
                           "expected": sample(left[index:index + 1])[0],
                           "actual": sample(right[index:index + 1])[0]})
        R["comparison_failure"]["mismatch_samples"] = points
        R["comparison_failure"]["mismatch_sample_rule"] = "First 32 exact inequalities; not a tolerance filter."


def compare(expected, actual, label, *, rtol=None, atol=None):
    rtol = O.rtol if rtol is None else rtol
    atol = O.atol if atol is None else atol
    metrics = {}
    if expected.keys() != actual.keys():
        raise AssertionError(f"{label}: output keys differ")
    for name, reference in expected.items():
        got = actual[name]
        if reference.shape != got.shape or reference.dtype != got.dtype:
            comparison_failure(label, name, reference, got, {"shape_or_dtype_mismatch": True})
            raise AssertionError(f"{label}/{name}: {reference.shape}/{reference.dtype} != {got.shape}/{got.dtype}")
        exact = bool(T.equal(reference, got))
        item = {"shape": list(got.shape), "dtype": str(got.dtype), "exact": exact}
        if reference.is_floating_point() and reference.numel():
            a, b = reference.double(), got.double()
            item["finite"] = bool(T.isfinite(a).all() and T.isfinite(b).all())
            if not item["finite"]:
                comparison_failure(label, name, reference, got, item)
                raise AssertionError(f"{label}/{name}: non-finite valid output")
            item["max_abs"] = float((a - b).abs().max())
            item["max_rel"] = float(((a - b).abs() / a.abs().clamp_min(1e-8)).max())
            try:
                T.testing.assert_close(got, reference, rtol=rtol, atol=atol,
                                       msg=lambda message: f"{label}/{name}: {message}")
            except AssertionError:
                comparison_failure(label, name, reference, got, item)
                raise
        elif not exact:
            count = int(T.count_nonzero(reference != got))
            item["mismatches"] = count
            comparison_failure(label, name, reference, got, item)
            raise AssertionError(f"{label}/{name}: {count} integer/byte values differ")
        metrics[name] = item
    return metrics


def capture(body):
    checkpoint("warmup")
    stream = T.npu.Stream()
    current = T.npu.current_stream()
    stream.wait_stream(current)
    with T.npu.stream(stream):
        for _ in range(2):
            body()
    T.npu.synchronize()
    graph = T.npu.NPUGraph()
    checkpoint("capture_begin")
    with T.npu.stream(stream):
        graph.capture_begin(capture_error_mode="global")
        outputs = body()
        graph.capture_end()
    current.wait_stream(stream)
    T.npu.synchronize()
    checkpoint("capture_end")
    return graph, outputs, stream


def matrix(cases, prepare, reset, body, observe, after=None, *, collect_failures=False,
           sync_before_replay=False):
    prepare(cases[0])
    reset(cases[0])
    graph, outputs, stream = capture(body)
    R["capture_supported"] = True
    R.setdefault("checks", {})["capture"] = "PASS"
    R["cases"] = []
    pointers = None
    observations = []
    for case in cases:
        checkpoint("prepare:" + case["name"])
        addresses = prepare(case) or {}
        if pointers is None:
            pointers = addresses
        if pointers != addresses:
            raise AssertionError("A graph-bound input was rebound between cases")
        reset(case)
        checkpoint("eager:" + case["name"])
        eager = body()
        T.npu.synchronize()
        expected = snapshot(observe(eager, case))
        reset(case)
        if sync_before_replay:
            T.npu.synchronize()
        checkpoint("replay:" + case["name"])
        graph.replay()
        T.npu.synchronize()
        actual = snapshot(observe(outputs, case))
        metrics, failures = {}, []
        if collect_failures:
            for name in expected:
                try:
                    metrics.update(compare({name: expected[name]}, {name: actual[name]}, case["name"]))
                except AssertionError as exc:
                    detail = dict(R["comparison_failure"])
                    metrics[name] = dict(detail["metrics"], status="FAIL", error=str(exc))
                    failures.append(detail)
            if failures:
                R.setdefault("case_failures", []).append({"case": case["name"], "failures": failures})
        else:
            metrics = compare(expected, actual, case["name"])
        R["cases"].append({"name": case["name"],
                           "parameters": {k: v for k, v in case.items() if not k.startswith("_")},
                           "status": "FAIL" if failures else "PASS", "comparisons": metrics})
        observations.append(expected)
        if after is not None:
            after(case, expected)
        checkpoint(("failed:" if failures else "passed:") + case["name"])
    R["bound_input_pointers"] = pointers
    R.setdefault("checks", {})["eager_replay"] = "FAIL" if R.get("case_failures") else "PASS"
    R["replay_count"] = len(cases)
    if len(observations) > 1:
        R["observed_output_changed"] = any(
            any(not T.equal(first[k], other[k]) for k in first)
            for first, other in zip(observations, observations[1:])
        )
    if R.get("case_failures"):
        checkpoint("comparison_failures_collected")
        raise AssertionError(f"{len(R['case_failures'])} case(s) failed comparison; all completed comparisons retained")
    return observations


def stage(buffers, key, value):
    if key not in buffers:
        buffers[key] = value.clone()
    elif buffers[key].shape != value.shape or buffers[key].dtype != value.dtype:
        raise AssertionError(f"Metadata shape/dtype changed: {key}: {buffers[key].shape} -> {value.shape}")
    else:
        buffers[key].copy_(value)
    return buffers[key]


def addresses(buffers):
    return {name: tensor.data_ptr() for name, tensor in buffers.items()}


def case_layouts():
    n = O.tokens
    return [
        {"name": "full_A", "live": n, "prefix": 6 * O.page_size, "page_shift": 0, "seed": O.seed},
        {"name": "short_B", "live": max(1, n // 2 - 1), "prefix": 1, "page_shift": 1, "seed": O.seed + 1},
        {"name": "page_boundary_C", "live": max(1, n // 4), "prefix": O.page_size - 1, "page_shift": 2, "seed": O.seed + 2},
        {"name": "different_prefix_D", "live": max(1, n - 3), "prefix": 4 * O.page_size + 1, "page_shift": 1, "seed": O.seed + 3},
        {"name": "repeat_A", "live": n, "prefix": 6 * O.page_size, "page_shift": 0, "seed": O.seed},
    ]


def compression_count(prefix, length, ratio=2):
    return (prefix + length) // ratio - prefix // ratio


def operator_inventory():
    inventory = {}
    for group in ("attention", "mem_cache", "hyper_conn"):
        key = "sgl_kernel_npu.operators." + group
        try:
            module = importlib.import_module(key)
            entries = {}
            for name in dir(module):
                if name.startswith("_") or not any(part in name.lower() for part in ("compress", "epilog", "mhc", "hc_pre")):
                    continue
                try:
                    function = getattr(module, name)
                    if not callable(function):
                        continue
                    entries[name] = {"signature": str(inspect.signature(function)),
                                     "file": inspect.getsourcefile(function),
                                     "doc": (inspect.getdoc(function) or "")[:2000]}
                except Exception as exc:
                    entries[name] = {"inspection_error": f"{type(exc).__name__}: {exc}"}
            inventory[key] = {"file": module.__file__, "exports": entries}
        except Exception as exc:
            inventory[key] = {"error": f"{type(exc).__name__}: {exc}"}
    schemas = []
    for schema in T._C._jit_get_all_schemas():
        text = str(schema)
        name = text.split("(", 1)[0]
        if name.startswith(("npu::", "custom::", "sgl_kernel_npu::", "cann_ops_transformer::")) and any(part in name.lower() for part in ("compress", "epilog", "mhc", "hc_pre")):
            schemas.append(text)
    inventory["registered_schemas"] = sorted(schemas)
    try:
        registry = importlib.import_module("sgl_kernel_npu.operators.utils.registry")
        resolved = {}
        selections = (("attention.compressor", "auto"), ("attention.compressor_v2", "auto"),
                      ("mem_cache.kv_compress_epilog", "auto"),
                      ("mem_cache.kv_compress_epilog_v2", "auto"),
                      ("hyper_conn.mhc_pre_dpskv4", "cann_recipes_infer_cann920"))
        for name, implementation in selections:
            try:
                function = registry.get_impl_by_name(name, implementation)
                resolved[name] = {"requested_impl": implementation, "module": function.__module__, "name": function.__name__,
                                  "signature": str(inspect.signature(function))}
            except Exception as exc:
                resolved[name] = {"requested_impl": implementation, "error": f"{type(exc).__name__}: {exc}"}
        inventory["auto_resolution_without_execution"] = resolved
    except Exception as exc:
        inventory["registry_inspection_error"] = f"{type(exc).__name__}: {exc}"
    return inventory


def probe_environment():
    R["python"] = sys.version
    R["platform"] = platform.platform()
    R["executable"] = sys.executable
    R["packages"] = {}
    for name in ("torch", "torch-npu", "sglang", "sgl-kernel-npu", "triton", "numpy"):
        try:
            R["packages"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            R["packages"][name] = None
    R["npu_smi"] = command(["npu-smi", "info"])
    R["cann_files"] = {}
    for name in ("/usr/local/Ascend/ascend-toolkit/latest/version.cfg",
                 "/usr/local/Ascend/ascend-toolkit/latest/version.info"):
        source = Path(name)
        if source.is_file():
            R["cann_files"][name] = source.read_text(errors="replace")[:12000]
    load_device()
    import sglang
    R["sglang_file"] = sglang.__file__
    root = Path(O.sglang_root).resolve() if O.sglang_root else Path(sglang.__file__).resolve().parents[2]
    R["source"] = {"root": str(root), "git": command(["git", "rev-parse", "HEAD"], cwd=root),
                   "status": command(["git", "status", "--short"], cwd=root)}
    for group, names in API_NAMES.items():
        for name in names:
            try:
                function = api(group, name)
                source = inspect.getsourcefile(function)
                if source and Path(source).is_file():
                    R["apis"][name]["source_sha256"] = hashlib.sha256(Path(source).read_bytes()).hexdigest()
            except Blocked as exc:
                R.setdefault("missing_apis", {})[name] = str(exc)
    R["custom_op_schemas"] = {}
    for namespace, name in (("custom", "kv_compress_epilog_v2"), ("custom", "npu_hc_pre_v2")):
        packet = getattr(getattr(T.ops, namespace), name, None)
        overload = getattr(packet, "default", None) if packet is not None else None
        R["custom_op_schemas"][namespace + "." + name] = str(getattr(overload, "_schema", "not registered"))
    R["legacy_custom_schema_note"] = "Historical diagnostics only; merged writer and mHC pre use public wrappers. Missing legacy schemas do not block the new probes."
    R["operator_inventory"] = operator_inventory()
    R["probe_index_page_size"] = O.index_page_size
    try:
        pool_module = importlib.import_module("sglang.srt.mem_cache.deepseek_v4_memory_pool")
        R["sglang_index_page_size"] = int(pool_module.DSV41_INDEX_PAGE_SIZE)
    except Exception as exc:
        R["index_page_inspection_error"] = f"{type(exc).__name__}: {exc}"
    try:
        layouts = importlib.import_module("sglang.srt.hardware_backend.npu.dsv4.dsv4_kv_layout").NPUDSV4KVLayout
        R["kv_layout_contracts"] = {}
        for name in ("V4_BF16", "V4_FP8", "V41_SWA_FP8", "V41_CMP_FP4"):
            layout = layouts[name]
            width, dtype = layout.kv_info(448, 64)
            R["kv_layout_contracts"][name] = {"width": width, "dtype": str(dtype),
                                                "quant_mode": layout.quant_mode,
                                                "quant_group": layout.quant_group}
        R["kv_layout_contract_note"] = "Allocation metadata only; not evidence that writer or draft attention executed."
    except Exception as exc:
        R["kv_layout_inspection_error"] = f"{type(exc).__name__}: {exc}"
    R["device_capabilities"] = str(T.npu.get_device_properties(O.device))
    R["runtime_environment"] = {
        name: os.environ.get(name) for name in (
            "ASCEND_HOME_PATH", "ASCEND_OPP_PATH", "ASCEND_RT_VISIBLE_DEVICES",
            "ASCEND_LAUNCH_BLOCKING", "PYTORCH_NPU_ALLOC_CONF",
        )
    }
    try:
        cls = backend().DeepseekV4AscendAttnBackend
        R["sglang_prefill_capability_source"] = inspect.getsource(cls._supports_dsv4_prefill_graph_capture)
        R["low_ratio_prefill_graph"] = cls.low_ratio_prefill_graph
    except Exception as exc:
        R["backend_inspection_error"] = f"{type(exc).__name__}: {exc}"
    R["scope"] = "Inspection only; API presence does not prove operator execution or graph support."


def probe_geometry():
    load_device()
    cls = backend().DeepseekV4AscendAttnBackend
    obj = cls.__new__(cls)
    n = O.tokens
    obj._dsv4_sliding_window_size = 128
    obj._dsv4_index_topk = 512
    fm = SimpleNamespace(
        dsv41_live_mask_buf=device_tensor([True] * n, T.bool),
        dsv41_win_sparse_indices=T.empty((n, 1, 128), dtype=T.int32, device=f"npu:{O.device}"),
        dsv41_win_topk_length=T.empty((n, 1), dtype=T.int32, device=f"npu:{O.device}"),
        dsv41_cmp_topk_length_c1=T.empty((n, 1), dtype=T.int32, device=f"npu:{O.device}"),
        dsv41_cmp_topk_length_c2=T.empty((n, 1), dtype=T.int32, device=f"npu:{O.device}"),
    )
    obj.forward_metadata = fm
    positions = T.zeros(n, dtype=T.int64, device=f"npu:{O.device}")
    slots = T.zeros_like(positions)
    batch = SimpleNamespace(positions=positions)
    c2 = min(n, n // 2 + 1)
    pos2, loc2 = T.empty(c2, dtype=T.int64, device=positions.device), T.empty(c2, dtype=T.int64, device=positions.device)

    def prepare(case):
        live = case["live"]
        positions.copy_(device_tensor([case["prefix"] + i if i < live else 0 for i in range(n)], T.int64))
        slots.copy_(device_tensor([O.page_size + case["prefix"] + i if i < live else 0 for i in range(n)], T.int64))
        fm.dsv41_live_mask_buf.copy_(device_tensor([i < live for i in range(n)], T.bool))
        return {"positions": positions.data_ptr(), "slots": slots.data_ptr(), "mask": fm.dsv41_live_mask_buf.data_ptr()}

    def body():
        win, length = obj._dsv41_window_indices(batch)
        l1, l2 = obj._dsv41_cmp_topk_length(batch, 1), obj._dsv41_cmp_topk_length(batch, 2)
        keep1, p1, s1 = obj._dsv41_ratio_row_values(positions, slots, 1, O.page_size, fm.dsv41_live_mask_buf)
        keep2, p2, s2 = obj._dsv41_ratio_row_values(positions, slots, 2, O.page_size, fm.dsv41_live_mask_buf)
        cls._stable_compact_1d(pos2, p2, keep2, fill=0)
        cls._stable_compact_1d(loc2, s2, keep2, fill=-1)
        return {"window": win, "window_length": length, "cmp1_length": l1, "cmp2_length": l2,
                "c1_positions": T.where(keep1, p1, 0), "c1_slots": T.where(keep1, s1, -1),
                "c2_positions": pos2, "c2_slots": loc2}

    def observe(out, case):
        expected = geometry_reference(case, n, c2)
        compare(expected, snapshot(out), "independent_cpu_geometry")
        return out

    matrix(case_layouts(), prepare, lambda case: None, body, observe)
    from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
    pool = ReqToTokenPool(size=2, max_context_len=128, device=f"npu:{O.device}", enable_memory_saver=False)
    if 0 in pool.free_slots or pool.req_to_token.shape[0] < 3:
        raise AssertionError("Request slot zero is not reserved")
    R.setdefault("checks", {})["cpu_reference"] = "PASS"
    R["request_slot_zero_reserved"] = True
    R["scope"] = "Actual SGLang geometry helpers and reserved request slot; explicit live mask supplied by this probe."


def geometry_reference(case, n, c2):
    window, lengths, c1len, c2len, p1, s1, p2, s2 = [], [], [], [], [], [], [], []
    for i in range(n):
        live = i < case["live"]
        position = case["prefix"] + i if live else 0
        width = min(position + 1, 128) if live else 0
        window.append([position - width + 1 + j if j < width else -1 for j in range(128)])
        lengths.append([width])
        c1len.append([min(position + 1, 512) if live else 0])
        c2len.append([min((position + 1) // 2, 512) if live else 0])
        p1.append(position if live else 0)
        s1.append(O.page_size + position if live else -1)
        if live and position % 2:
            p2.append(position - 1)
            s2.append((O.page_size + position) // 2)
    p2 += [0] * (c2 - len(p2))
    s2 += [-1] * (c2 - len(s2))
    return {"window": T.tensor(window, dtype=T.int32).unsqueeze(1),
            "window_length": T.tensor(lengths, dtype=T.int32),
            "cmp1_length": T.tensor(c1len, dtype=T.int32), "cmp2_length": T.tensor(c2len, dtype=T.int32),
            "c1_positions": T.tensor(p1), "c1_slots": T.tensor(s1),
            "c2_positions": T.tensor(p2), "c2_slots": T.tensor(s2)}


def probe_compressor():
    load_device()
    function = api("attention", "compressor_v2")
    n, hidden, dim = O.tokens, O.hidden_size, 512
    x = random_tensor((n, hidden), O.seed)
    wkv = random_tensor((dim, hidden), O.seed + 1, scale=0.01)
    wgate = random_tensor((dim, hidden), O.seed + 2, scale=0.01)
    ring = O.state_ring_size
    state = T.zeros((3, ring, 2 * dim), dtype=T.float32, device=x.device)
    cu, used, start, bank = device_tensor([0, n], T.int32), device_tensor([n], T.int32), device_tensor([0], T.int32), device_tensor([0], T.int32)
    carry = {}
    cases = [
        {"name": "capture_bank_zero", "live": n, "prefix": 0, "bank": 0, "seed": O.seed},
        {"name": "bank_one_short", "live": n // 2, "prefix": 0, "bank": 1, "seed": O.seed + 3},
        {"name": "pair_chunk_first", "live": 1, "prefix": 0, "bank": 1, "seed": O.seed + 4},
        {"name": "pair_chunk_second", "live": 1, "prefix": 1, "bank": 1, "seed": O.seed + 5, "carry": True},
        {"name": "ring_wrap", "live": min(7, n), "prefix": 2 * ring - 2, "bank": 2, "seed": O.seed + 6},
        {"name": "bank_reuse", "live": n // 2, "prefix": 0, "bank": 1, "seed": O.seed + 7},
        {"name": "repeat_capture_bank_zero", "live": n, "prefix": 0, "bank": 0, "seed": O.seed},
    ]

    def prepare(case):
        x.copy_(random_tensor(x.shape, case["seed"]))
        cu.copy_(device_tensor([0, case["live"]], T.int32))
        used.fill_(case["live"])
        start.fill_(case["prefix"])
        bank.fill_(case["bank"])
        return addresses({"x": x, "cu": cu, "used": used, "start": start, "bank": bank, "state": state})

    def reset(case):
        if case.get("carry"):
            state.copy_(carry["state"].to(state.device))
        else:
            state.zero_()

    def body():
        value = function(x=x, wkv=wkv, wgate=wgate, seqused=used, start_pos=start,
                         cmp_ratio=2, cu_seqlens=cu, state_cache=state, state_block_table=bank)
        return {"value": value, "state": state}

    def observe(out, case):
        count = compression_count(case["prefix"], case["live"])
        if out["value"].shape[0] < count:
            raise AssertionError("Compressor output capacity is smaller than valid pair count")
        R.setdefault("output_capacities", {})[case["name"]] = list(out["value"].shape)
        untouched = [i for i in range(3) if i != case["bank"]]
        if not case.get("carry") and bool(T.count_nonzero(state[untouched]).cpu()):
            raise AssertionError("Compressor wrote an unrelated synthetic request bank")
        return {"valid_value": out["value"][:count], "state": state}

    def after(case, expected):
        if case["name"] == "pair_chunk_first":
            carry["state"] = expected["state"]
        if case["name"] == "pair_chunk_second":
            carry["split_value"] = expected["valid_value"]

    matrix(cases, prepare, reset, body, observe, after)
    checkpoint("split_vs_concatenated_pair")
    state.zero_()
    a = random_tensor((n, hidden), O.seed + 4)
    b = random_tensor((n, hidden), O.seed + 5)
    x.copy_(a)
    x[1].copy_(b[0])
    cu.copy_(device_tensor([0, 2], T.int32))
    used.fill_(2)
    start.zero_()
    bank.fill_(1)
    joined = body()["value"]
    T.npu.synchronize()
    R["split_pair_reference"] = compare({"pair": joined[:1].cpu()}, {"pair": carry["split_value"]}, "split_pair")
    R["scope"] = "Actual compressor_v2; private banks only. Ratio 1 uses projection, not this fused compressor. Ring-wrap parity is not an independent compressor math oracle."


def epilog():
    R["writer_entrypoint"] = "sgl_kernel_npu.operators.mem_cache.kv_compress_epilog_v2"
    R["writer_profile"] = "public wrapper used by merged SGLang; no custom namespace fallback"
    return api("mem_cache", "kv_compress_epilog_v2")


def probe_epilog(kind):
    load_device()
    function = epilog()
    mode, group, width = ("mxfp8_bf16", 32, 544) if kind == "fp8" else ("mxfp4_bf16", 16, 320)
    n, rows = O.tokens, O.page_size + 2 * O.tokens + 16
    x = random_tensor((n, 512), O.seed)
    loc = T.zeros(n, dtype=T.int64, device=x.device)
    dtype = T.float8_e4m3fn if kind == "fp8" else T.uint8
    cache = T.zeros((rows, width), dtype=dtype, device=x.device)
    sentinel = T.full((rows, width), 23, dtype=T.uint8)
    cases = [
        {"name": "full_A", "live": n, "offset": O.page_size, "seed": O.seed},
        {"name": "short_with_nan_skip", "live": max(1, n // 2 - 1), "offset": O.page_size + n, "seed": O.seed + 1},
        {"name": "all_minus_one", "live": 0, "offset": O.page_size, "seed": O.seed + 2},
        {"name": "repeat_A", "live": n, "offset": O.page_size, "seed": O.seed},
    ]

    def prepare(case):
        x.copy_(random_tensor(x.shape, case["seed"]))
        x[case["live"]:].fill_(float("nan"))
        loc.copy_(device_tensor([case["offset"] + i if i < case["live"] else -1 for i in range(n)], T.int64))
        return addresses({"x": x, "loc": loc, "cache": cache})

    def reset(case):
        cache.view(T.uint8).copy_(sentinel.to(cache.device))

    def body():
        function(cache, x, loc, quant_group_size=group, quant_mode=mode,
                 round_scale=True, x_scale=1.0)
        return {"bytes": cache.view(T.uint8)}

    def observe(out, case):
        got = out["bytes"].cpu()
        mask = T.ones(rows, dtype=T.bool)
        mask[case["offset"]:case["offset"] + case["live"]] = False
        if not T.equal(got[mask], sentinel[mask]):
            raise AssertionError("-1/padding write changed a guard cache row")
        if case["live"] and T.equal(got[~mask], sentinel[~mask]):
            raise AssertionError("No active cache row was written")
        return out

    matrix(cases, prepare, reset, body, observe)
    R["packing"] = {"quant_mode": mode, "group": group, "row_bytes": width}
    R["scope"] = "Actual epilog byte parity and untouched guards, including NaN input on -1 rows."


def fp4_values():
    return T.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6,
                     -0.0, -0.5, -1, -1.5, -2, -3, -4, -6], dtype=T.float32)


def packed_cache(kind, rows, seed):
    generator = T.Generator(device="cpu").manual_seed(seed)
    if kind == "fp8":
        quantized = (T.randn((rows, 512), generator=generator) * 0.25).to(T.float8_e4m3fn)
        scale = T.ones((rows, 16), dtype=T.bfloat16)
        return T.cat((quantized.view(T.uint8), scale.view(T.uint8)), dim=-1), quantized.float()
    codes = T.randint(0, 16, (rows, 512), generator=generator, dtype=T.uint8)
    data = codes[:, 0::2] | (codes[:, 1::2] << 4)
    scale = T.full((rows, 32), 0.125, dtype=T.bfloat16)
    return T.cat((data, scale.view(T.uint8)), dim=-1), fp4_values()[codes.long()] * 0.125


def unpack_cache(kind, packed):
    packed = packed.detach().cpu().contiguous().view(T.uint8)
    if kind == "fp8":
        data = packed[:, :512].contiguous().view(T.float8_e4m3fn).float()
        scale = packed[:, 512:].contiguous().view(T.bfloat16).float()
        return data * scale.repeat_interleave(32, dim=-1)
    data = packed[:, :256]
    codes = T.stack((data & 15, data >> 4), dim=-1).reshape(-1, 512)
    scale = packed[:, 256:].contiguous().view(T.bfloat16).float()
    return fp4_values()[codes.long()] * scale.repeat_interleave(16, dim=-1)


def writer_cache_view(cache, kind):
    # Writer dtype and attention's byte view must alias the same storage.
    return cache.view(T.float8_e4m3fn) if kind == "fp8" else cache


def prepare_writer_cache(buffers, cpu, writer, case_name):
    for name, kind, mode, group in (
        ("original", "fp8", "mxfp8_bf16", 32),
        ("compressed", "fp4", "mxfp4_bf16", 16),
    ):
        if name not in buffers:
            continue
        cache = buffers[name]
        rows, width = cache.numel() // cache.shape[-1], cache.shape[-1]
        latent = cpu[name].to(device=cache.device, dtype=T.bfloat16).contiguous()
        locations = T.arange(rows, dtype=T.int64, device=cache.device)
        typed_cache = writer_cache_view(cache, kind)
        writer(typed_cache.view(rows, width), latent, locations,
               quant_group_size=group, quant_mode=mode, round_scale=True, x_scale=1.0)
        T.npu.synchronize()
        decoded = unpack_cache(kind, cache.view(rows, width))
        if not bool(T.isfinite(decoded).all()):
            raise AssertionError(f"Public writer produced non-finite {name} data under the advertised KV ABI")
        R.setdefault("writer_quantization_diagnostics", {}).setdefault(case_name, {})[name] = {
            "row_bytes": width, "quant_group": group,
            "writer_dtype": str(typed_cache.dtype), "attention_view_dtype": str(cache.dtype),
            "same_storage": typed_cache.data_ptr() == cache.data_ptr(),
            "max_abs_input_to_decoded": float((decoded - latent.cpu().float()).abs().max()),
            "note": "Recorded error, not a quantization-accuracy acceptance threshold.",
        }
        cpu[name] = decoded


def index_fixture(rows, seed):
    generator = T.Generator(device="cpu").manual_seed(seed)
    codes = T.randint(0, 16, (rows, 128), generator=generator, dtype=T.uint8)
    data = codes[:, 0::2] | (codes[:, 1::2] << 4)
    scales = T.full((rows, 4), 127, dtype=T.uint8)
    return data, scales


def index_cache_views(key, scale, index_page):
    rows = key.shape[0]
    if rows % index_page or index_page % 8:
        raise ValueError("Index pages must contain whole eight-token records")
    pages = rows // index_page
    record = T.cat((key.view(pages, index_page // 8, 8 * 64),
                    scale.view(pages, index_page // 8, 8 * 4)), dim=-1)
    return {"key": key.view(pages, index_page, 1, 64),
            "key_scale": scale.view(pages, index_page, 1, 2, 2),
            "combined_key": record.view(pages, index_page // 8, 544)}


def physical_tables(case, ratio):
    full_pages = math.ceil((6 * O.page_size + O.tokens) / O.page_size)
    table = [1 + (i + case["page_shift"]) % full_pages for i in range(full_pages)]
    full = T.tensor([table], dtype=T.int32)
    expansion = O.page_size // ratio // O.index_page_size
    index = (full.unsqueeze(-1) * expansion + T.arange(expansion, dtype=T.int32)).reshape(1, -1)
    return full, index, full_pages + 1


def query_geometry(case, ratio):
    n, live, prefix = O.tokens, case["live"], case["prefix"]
    positions = [prefix + i if i < live else 0 for i in range(n)]
    win_len = [min(prefix + i + 1, 128) if i < live else 0 for i in range(n)]
    window = [[positions[i] - win_len[i] + 1 + j if j < win_len[i] else -1 for j in range(128)] for i in range(n)]
    cmp_len = [min((prefix + i + 1) // ratio, 512) if i < live and ratio else 0 for i in range(n)]
    sparse = [[(prefix + i + 1) // ratio - 1 - j if ratio and j < cmp_len[i] else -1 for j in range(512)] for i in range(n)]
    return {
        "positions": device_tensor(positions, T.int64),
        "cu": device_tensor([0, live], T.int32),
        "kv_lengths": device_tensor([prefix + live], T.int32),
        "window": device_tensor(window, T.int32).view(n, 1, 128),
        "window_length": device_tensor(win_len, T.int32).view(n, 1),
        "cmp_length": device_tensor(cmp_len, T.int32).view(n, 1),
        "sparse": device_tensor(sparse, T.int32).view(n, 1, 512),
    }


def attention_metadata(function, buffers, ratio):
    return function(
        ori_topk_length=buffers["window_length"], cmp_topk_length=buffers["cmp_length"],
        cu_seqlens_q=buffers["cu"], num_heads_q=64, num_heads_kv=1,
        head_dim=512, quant_mode=1, layout_q="TND", layout_kv="PA_BBND",
        has_ori_kv=True, has_cmp_kv=ratio > 0,
    )


def index_metadata(function, qsli_function, buffers, case, ratio, candidate, *, zero_padding=False):
    common = dict(cu_seqlens_q=buffers["cu"], seqused_q=None,
                  seqused_k=buffers["index_kv_lengths"],
                  cmp_residual_k=None if ratio == 1 else buffers["residual"],
                  batch_size=1, num_heads_q=32, num_heads_k=1, head_dim=128,
                  topk=512, mask_mode=3, cmp_ratio=ratio,
                  layout_q="TND", layout_k="PA_BBND")
    kwargs = {"candidate_topk_blocks": O.candidate_blocks, "candidate_block_size": 8} if candidate else {}
    stage(buffers, "qli_metadata", function(**common, **kwargs))
    if candidate:
        visible = [min(O.candidate_blocks, (case["prefix"] + i + 8) // 8)
                   if i < case["live"] else (0 if zero_padding else 1) for i in range(O.tokens)]
        stage(buffers, "candidate_bound", device_tensor(visible, T.int32).view(O.tokens, 1))
        stage(buffers, "qsli_metadata", qsli_function(
            candidate_block_length=buffers["candidate_bound"], **common,
            quant_mode=1, candidate_block_size=8,
        ))


def probe_metadata():
    load_device()
    mq = api("attention", "mixed_quant_sparse_flash_mla_metadata_dsl")
    qli = api("indexer", "quant_lightning_indexer_metadata_dsl")
    qsli = api("indexer", "quant_sparse_lightning_indexer_metadata_dsl")
    stores = {ratio: {} for ratio in (0, 1, 2)}
    R["cases"] = []
    for case in case_layouts():
        checkpoint("metadata:" + case["name"])
        for ratio, buffers in stores.items():
            for key, value in query_geometry(case, ratio).items():
                stage(buffers, key, value)
            stage(buffers, "attention_metadata", attention_metadata(mq, buffers, ratio))
            if ratio:
                stage(buffers, "index_kv_lengths", buffers["kv_lengths"] // ratio)
                stage(buffers, "residual", buffers["kv_lengths"] % ratio)
                index_metadata(qli, qsli, buffers, case, ratio, candidate=ratio == 1)
            T.npu.synchronize()
            summary = {}
            for key, value in buffers.items():
                if "metadata" in key:
                    raw = value.detach().cpu().contiguous().view(T.uint8).numpy().tobytes()
                    summary[key] = {"shape": list(value.shape), "dtype": str(value.dtype),
                                    "data_ptr": value.data_ptr(), "sha256": hashlib.sha256(raw).hexdigest(),
                                    "first_32": value.detach().cpu().reshape(-1)[:32].tolist()}
            R["cases"].append({"name": case["name"], "ratio": ratio, "status": "PASS", "metadata": summary})
            checkpoint("metadata_passed:" + case["name"])
    R["scope"] = "Real metadata APIs execute eagerly, then copy to fixed buffers. This probe does not require metadata planning itself to be captured."


def probe_prologue():
    load_device()
    function = api("indexer", "indexer_prologue_k_dsl")
    n, rows = O.tokens, O.page_size + 2 * O.tokens
    index_page = O.index_page_size
    rows = math.ceil(rows / index_page) * index_page
    latent = random_tensor((n, 512), O.seed)
    weight = random_tensor((128, 512), O.seed + 1, scale=0.03)
    norm = T.ones(128, dtype=T.float32, device=latent.device)
    sine, cosine = T.zeros((n, 64), dtype=T.float32, device=latent.device), T.ones((n, 64), dtype=T.float32, device=latent.device)
    loc = T.zeros(n, dtype=T.int64, device=latent.device)
    key = T.zeros((rows // index_page, index_page, 1, 64), dtype=T.uint8, device=latent.device)
    scales = T.zeros((rows // index_page, index_page, 1, 4), dtype=T.uint8, device=latent.device)
    combined = T.zeros((rows // index_page, index_page // 8, 1, 544), dtype=T.uint8, device=latent.device)
    cases = [
        {"name": "full_A", "live": n, "offset": O.page_size, "seed": O.seed},
        {"name": "short_B", "live": max(1, n // 2 - 1), "offset": O.page_size + n, "seed": O.seed + 2},
        {"name": "all_skip", "live": 0, "offset": O.page_size, "seed": O.seed + 3},
        {"name": "repeat_A", "live": n, "offset": O.page_size, "seed": O.seed},
    ]

    def prepare(case):
        latent.copy_(random_tensor(latent.shape, case["seed"]))
        loc.copy_(device_tensor([case["offset"] + i if i < case["live"] else -1 for i in range(n)], T.int64))
        return addresses({"latent": latent, "loc": loc, "key": key, "scales": scales, "combined": combined})

    def reset(case):
        key.fill_(17)
        scales.fill_(127)
        combined[..., :512].fill_(17)
        combined[..., 512:].fill_(127)

    def body():
        # The NZ conversion follows the current serving call and is captured.
        common = dict(latent=latent, wk=N.npu_format_cast(weight, N.Format.FRACTAL_NZ),
                      norm_weight=norm, rope_sin=sine, rope_cos=cosine,
                      cache_index=loc, norm_eps=1e-6)
        function(**common, k_cache=key, k_scale_cache=scales, storage_mode=0)
        function(**common, k_cache=combined, k_scale_cache=None, storage_mode=1, combined_block_size=8)
        return {"key": key, "scales": scales, "combined": combined}

    def observe(out, case):
        cpu = snapshot(out)
        flat_key, flat_scale = cpu["key"].reshape(rows, 64), cpu["scales"].reshape(rows, 4)
        record = cpu["combined"].reshape(-1, 544)
        # Each record stores eight packed K rows, followed by their scales.
        combined_key = record[:, :512].reshape(rows, 64)
        combined_scale = record[:, 512:].reshape(rows, 4)
        compare({"key": flat_key, "scales": flat_scale},
                {"key": combined_key, "scales": combined_scale}, "storage_mode_equivalence")
        untouched = T.ones(rows, dtype=T.bool)
        untouched[case["offset"]:case["offset"] + case["live"]] = False
        if not bool((flat_key[untouched] == 17).all() and (flat_scale[untouched] == 127).all()):
            raise AssertionError("K prologue changed a guard row for a -1/unused slot")
        return out

    matrix(cases, prepare, reset, body, observe)
    R["scope"] = "Actual K prologue modes 0/1, fixed caches, -1 skip and packed-record equivalence; identity RoPE, synthetic BF16 weights."


def prepare_index(buffers, case, ratio, candidate, metadata_fn, qsli_metadata_fn):
    n = O.tokens
    full, table, physical_pages = physical_tables(case, ratio)
    for name, value in query_geometry(case, ratio).items():
        stage(buffers, name, value)
    stage(buffers, "block_table", table.to(f"npu:{O.device}"))
    stage(buffers, "index_kv_lengths", buffers["kv_lengths"] // ratio)
    stage(buffers, "residual", buffers["kv_lengths"] % ratio)
    rows = physical_pages * (O.page_size // ratio)
    key, scale = index_fixture(rows, case["seed"] + 10)
    for name, value in index_cache_views(key, scale, O.index_page_size).items():
        stage(buffers, name, value.to(f"npu:{O.device}"))
    q, qscale = index_fixture(n * 32, case["seed"] + 11)
    stage(buffers, "q", q.view(n, 32, 64).to(f"npu:{O.device}"))
    stage(buffers, "qscale", qscale.view(n, 32, 2, 2).to(f"npu:{O.device}"))
    weights = random_tensor((n, 32), case["seed"] + 12, scale=1.0, dtype=T.float32).abs() + 0.01
    stage(buffers, "weights", weights)
    index_metadata(metadata_fn, qsli_metadata_fn, buffers, case, ratio, candidate)
    return addresses(buffers)


def run_qli(function, buffers, ratio, candidate):
    return function(
        q=buffers["q"], k=buffers["key"], w=buffers["weights"],
        descale_q=buffers["qscale"], descale_k=buffers["key_scale"],
        cu_seqlens_q=buffers["cu"], cu_seqlens_k=None, seqused_q=None,
        seqused_k=buffers["index_kv_lengths"],
        cmp_residual_k=None if ratio == 1 else buffers["residual"],
        block_table=buffers["block_table"], output_idx_offset=None,
        metadata=buffers["qli_metadata"], topk=512, quant_mode=1,
        max_seqlen_q=-1, mask_mode=3, cmp_ratio=ratio,
        layout_q="TND", layout_k="PA_BBND", return_value=False,
        candidate_topk_blocks=O.candidate_blocks if candidate else -1,
        candidate_block_size=8 if candidate else -1,
    )


def run_qsli(function, buffers, indices, lengths):
    return function(
        q=buffers["q"], k=buffers["combined_key"], w=buffers["weights"],
        descale_q=buffers["qscale"], candidate_block_indices=indices,
        candidate_block_length=lengths, cu_seqlens_q=buffers["cu"],
        cu_seqlens_k=None, seqused_q=None, seqused_k=buffers["index_kv_lengths"],
        cmp_residual_k=None, block_table=buffers["block_table"],
        output_idx_offset=None, metadata=buffers["qsli_metadata"],
        topk=512, candidate_block_size=8, quant_mode=1, max_seqlen_q=-1,
        mask_mode=3, cmp_ratio=1, layout_q="TND", layout_k="PA_BBND", return_value=False,
    )


def checked_topk(value, case, ratio, label):
    value = value.detach().cpu().reshape(O.tokens, -1)[:case["live"]].to(T.int64)
    for row, indices in enumerate(value):
        active = indices[indices >= 0]
        limit = (case["prefix"] + row + 1) // ratio
        if active.numel() and int(active.max()) >= limit:
            raise AssertionError(f"{label}: row {row} selected a future/out-of-range key: {int(active.max())} >= {limit}")
    return value


def probe_index(candidate):
    load_device()
    qli = api("indexer", "quant_lightning_indexer_dsl")
    qli_metadata = api("indexer", "quant_lightning_indexer_metadata_dsl")
    qsli = api("indexer", "quant_sparse_lightning_indexer_dsl") if candidate else None
    qsli_metadata = api("indexer", "quant_sparse_lightning_indexer_metadata_dsl") if candidate else None
    ratio, buffers = (1 if candidate else 2), {}

    def prepare(case):
        return prepare_index(buffers, case, ratio, candidate, qli_metadata, qsli_metadata)

    def body():
        result = run_qli(qli, buffers, ratio, candidate)
        out = {"qli": result[0]}
        if candidate:
            out["candidate_indices"], out["candidate_lengths"] = result[2], result[3]
            out["qsli"] = run_qsli(qsli, buffers, result[2], result[3])[0]
        return out

    def observe(out, case):
        values = {"qli_ordered": checked_topk(out["qli"], case, ratio, "QLI")}
        if candidate:
            values["qsli_ordered"] = checked_topk(out["qsli"], case, 1, "QSLI")
            ids = out["candidate_indices"].cpu().reshape(O.tokens, -1)[:case["live"]]
            length = out["candidate_lengths"].cpu().reshape(O.tokens, -1)[:case["live"]]
            if bool((length < 0).any() or (length > ids.shape[-1]).any()):
                raise AssertionError("Invalid candidate length")
            valid = T.arange(ids.shape[-1]).view(1, -1) < length[:, :1]
            values["candidate_indices"] = T.where(valid, ids, -1)
            values["candidate_lengths"] = length
        return values

    matrix(case_layouts(), prepare, lambda case: None, body, observe)
    R["scope"] = "Real QLI -> QSLI graph dataflow" if candidate else "Real ratio-2 QLI graph"
    R["comparison_note"] = "Ordered indices are checked exactly; a failure with equal selected sets may be a tie and needs inspection. No independent score oracle is claimed."
    R["layout"] = {"query_heads": 32, "logical_head_dim": 128, "index_page_size": O.index_page_size,
                   "candidate_block_size": 8,
                   "topk": 512, "candidate_blocks": O.candidate_blocks if candidate else None}


def prepare_attention(buffers, cpu, case, ratio, metadata_fn):
    full, _, pages = physical_tables(case, max(ratio, 1))
    for name, value in query_geometry(case, ratio).items():
        stage(buffers, name, value)
    stage(buffers, "block_table", full.to(f"npu:{O.device}"))
    original, original_values = packed_cache("fp8", pages * O.page_size, case["seed"] + 100)
    stage(buffers, "original", original.view(pages, O.page_size, 1, 544).to(f"npu:{O.device}"))
    cpu["original"], cpu["table"] = original_values, full
    if ratio:
        compressed, compressed_values = packed_cache("fp4", pages * (O.page_size // ratio), case["seed"] + 101)
        stage(buffers, "compressed", compressed.view(pages, O.page_size // ratio, 1, 320).to(f"npu:{O.device}"))
        stage(buffers, "compressed_lengths", buffers["kv_lengths"] // ratio)
        cpu["compressed"] = compressed_values
    stage(buffers, "q", random_tensor((O.tokens, 64, 512), case["seed"] + 102))
    stage(buffers, "sinks", T.full((64,), -10000.0, dtype=T.float32, device=buffers["q"].device))
    stage(buffers, "metadata", attention_metadata(metadata_fn, buffers, ratio))
    cpu["q"] = buffers["q"].cpu().float()
    return addresses(buffers)


def attention_reference(cpu, case, ratio, row, head, *, round_probabilities=False):
    position = case["prefix"] + row
    logical = list(range(max(0, position - 127), position + 1))
    table = cpu["table"].reshape(-1)
    original_rows = [int(table[i // O.page_size]) * O.page_size + i % O.page_size for i in logical]
    values = cpu["original"][original_rows]
    if ratio:
        count = min((position + 1) // ratio, 512)
        logical_cmp = list(range((position + 1) // ratio - 1, (position + 1) // ratio - 1 - count, -1))
        page = O.page_size // ratio
        physical = [int(table[i // page]) * page + i % page for i in logical_cmp]
        values = T.cat((values, cpu["compressed"][physical]), dim=0)
    scores = T.mv(values, cpu["q"][row, head]) / math.sqrt(512)
    probabilities = T.softmax(scores, dim=0)
    if round_probabilities:
        probabilities = probabilities.to(T.bfloat16).float()
    return T.mv(values.T, probabilities)


def bf16_diagnostics(reference, actual):
    reference, actual = reference.double(), actual.double()
    if not bool(T.isfinite(reference).all() and T.isfinite(actual).all()):
        return {"finite": False}
    rounded = reference.to(T.bfloat16)
    ref_bits = rounded.contiguous().view(T.int16).to(T.int32)
    got_bits = actual.to(T.bfloat16).contiguous().view(T.int16).to(T.int32)
    def ordered(bits):
        magnitude = bits & 0x7fff
        return T.where(bits < 0, 0x8000 - magnitude, 0x8000 + magnitude)
    ulps = (ordered(ref_bits) - ordered(got_bits)).abs()
    error = (reference - actual).abs()
    index = int(error.argmax())
    return {
        "finite": True,
        "rmse": float(((reference - actual) ** 2).mean().sqrt()),
        "normalized_l2": float(T.linalg.vector_norm(reference - actual) / T.linalg.vector_norm(reference).clamp_min(1e-30)),
        "fp32_to_bf16_rounding_max_abs": float((reference - rounded.double()).abs().max()),
        "actual_to_rounded_reference_max_abs": float((actual - rounded.double()).abs().max()),
        "max_bf16_ulp_distance": int(ulps.max()),
        "within_one_bf16_ulp": int((ulps <= 1).sum()),
        "elements": reference.numel(),
        "largest_error": {"index": index, "reference_fp32": float(reference[index]),
                          "reference_bf16": float(rounded[index]), "actual": float(actual[index]),
                          "abs_error": float(error[index])},
        "interpretation": "Diagnostics only; neither ULP distance nor BF16 rounding changes the strict reference status.",
    }


def probe_attention(ratio, *, use_writer=False):
    load_device()
    writer = epilog() if use_writer else None
    function = api("attention", "mixed_quant_sparse_flash_mla_dsl")
    metadata_fn = api("attention", "mixed_quant_sparse_flash_mla_metadata_dsl")
    buffers, cpu = {}, {}

    def prepare(case):
        pointers = prepare_attention(buffers, cpu, case, ratio, metadata_fn)
        if writer is not None:
            checkpoint("public_writer_prepare:" + case["name"])
            prepare_writer_cache(buffers, cpu, writer, case["name"])
        return pointers

    def body():
        q = buffers["q"]
        output = T.full_like(q, float("nan"))
        lse = T.empty((0,), dtype=T.float32, device=q.device)
        kwargs = dict(
            ori_kv=buffers["original"], ori_sparse_indices=buffers["window"],
            ori_topk_length=buffers["window_length"], ori_block_table=buffers["block_table"],
            cu_seqlens_q=buffers["cu"], seqused_q=None, seqused_ori_kv=buffers["kv_lengths"],
            sinks=buffers["sinks"], metadata=buffers["metadata"], quant_mode=1,
            softmax_scale=1 / math.sqrt(512), layout_q="TND", layout_kv="PA_BBND",
            return_softmax_lse=False,
        )
        if ratio:
            kwargs.update(cmp_kv=buffers["compressed"], cmp_sparse_indices=buffers["sparse"],
                          cmp_block_table=buffers["block_table"],
                          seqused_cmp_kv=buffers["compressed_lengths"], cmp_topk_length=buffers["cmp_length"])
        function(q, out=output, lse=lse, **kwargs)
        return {"output": output}

    def observe(out, case):
        value = out["output"].cpu()
        metrics = {}
        for row in sorted({0, case["live"] - 1}):
            for head in (0, 31, 63):
                reference = attention_reference(cpu, case, ratio, row, head)
                key = f"row{row}_head{head}"
                actual = value[row, head].float()
                try:
                    metrics[key] = compare(
                        {"reference": reference}, {"reference": actual}, "CPU attention reference",
                        rtol=O.reference_rtol, atol=O.reference_atol)
                except AssertionError as exc:
                    metrics[key] = {"status": "FAIL", "error": str(exc),
                                    "details": dict(R["comparison_failure"])}
                    R.setdefault("cpu_reference_failures", {})[case["name"]] = True
                    hypothesis = attention_reference(cpu, case, ratio, row, head, round_probabilities=True).to(T.bfloat16).float()
                    difference = (hypothesis - actual).abs()
                    metrics[key]["rounded_softmax_hypothesis_max_abs"] = float(difference.max()) if bool(T.isfinite(difference).all()) else None
                    metrics[key]["hypothesis_note"] = "Diagnostic alternate arithmetic; not substituted for the FP32 oracle."
                metrics[key]["bf16_diagnostics"] = bf16_diagnostics(reference, actual)
        R.setdefault("cpu_reference", {})[case["name"]] = metrics
        padding = value[case["live"]:].float()
        R.setdefault("padding_observations", {})[case["name"]] = {
            "rows": int(padding.shape[0]),
            "finite_values": int(T.isfinite(padding).sum()), "elements": padding.numel(),
            "zero_values": int((padding == 0).sum()),
            "max_abs_finite": float(padding[T.isfinite(padding)].abs().max()) if bool(T.isfinite(padding).any()) else None,
            "note": "Output seeded with NaN; inactive rows are observed, not required to be written by the kernel.",
        }
        return {"valid_output": out["output"][:case["live"]]}

    matrix(case_layouts(), prepare, lambda case: None, body, observe)
    R["scope"] = "Real MQSMLA eager/replay, varying lengths/page IDs/Q/KV, with CPU float32 reference on selected rows/heads."
    R["packing"] = (
        "Public V4.1 writer produced the cache before replay; CPU oracle decodes its 544/320B bytes. Writer executes outside this consumer graph."
        if use_writer else
        "CPU-built valid fp8+BF16-scale ori rows and fp4+BF16-scale cmp rows; independent of epilog registration."
    )
    R["cpu_reference_scope"] = "Six or fewer row/head samples per case, not all output elements; sinks fixed at -10000."
    R.setdefault("checks", {})["cpu_reference"] = "FAIL" if R.get("cpu_reference_failures") else "PASS"
    if R.get("cpu_reference_failures"):
        raise AssertionError("Graph parity completed, but the sampled CPU attention reference mismatched; inspect cpu_reference separately.")


def tiny_hasher(device):
    from sglang.srt.layers.engram import EngramHasher
    hasher = EngramHasher.__new__(EngramHasher)
    T.nn.Module.__init__(hasher)
    hasher.max_ngram_size = 3
    hasher.pad_id = 0
    hasher.image_token_id = None
    hasher._cpu_hash_params = None
    hasher.token_map = T.arange(128, device=device)
    hasher.multipliers = T.tensor([[3, 5, 7]], device=device)
    hasher.primes = T.tensor([[[11, 13], [17, 19]]], device=device)
    hasher.offsets = T.tensor([[0, 11, 24, 41]], device=device)
    hasher.init_history(num_req_slots=3, device=device)
    return hasher


def hasher_batch(device):
    from sglang.srt.model_executor.forward_mode import ForwardMode
    return SimpleNamespace(
        forward_mode=ForwardMode.EXTEND,
        positions=T.zeros(O.tokens, dtype=T.int64, device=device),
        req_pool_indices=T.tensor([1], dtype=T.int64, device=device),
        extend_seq_lens=T.tensor([O.tokens], dtype=T.int32, device=device),
        extend_start_loc=T.zeros(1, dtype=T.int32, device=device),
        extend_seq_lens_cpu=[O.tokens],
        out_cache_loc=T.arange(O.tokens, device=device) + O.page_size,
        engram_history=None, ragged_verify_layout=None, spec_info=None,
    )


def prepare_hasher_batch(batch, case):
    live = case["live"]
    batch.positions.copy_(T.tensor([case["prefix"] + i if i < live else 0 for i in range(O.tokens)], device=batch.positions.device))
    batch.req_pool_indices.fill_(1 + case["page_shift"] % 2)
    batch.extend_seq_lens.fill_(live)
    batch.extend_seq_lens_cpu = [live]
    batch.out_cache_loc.copy_(T.tensor([O.page_size + i if i < live else 0 for i in range(O.tokens)], device=batch.positions.device))


def probe_engram():
    load_device()
    from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
        BreakableCUDAGraph, BreakableCUDAGraphCapture, eager_on_graph,
    )
    from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.context import enable_breakable_cuda_graph
    hasher, reference = tiny_hasher(f"npu:{O.device}"), tiny_hasher("cpu")
    batch, cpu_batch = hasher_batch(f"npu:{O.device}"), hasher_batch("cpu")
    tokens = device_tensor([(i % 100) + 1 for i in range(O.tokens)], T.int64)
    output = hasher.new_hash_ids(tokens)
    calls = [0]

    @eager_on_graph(True)
    def hash_break():
        calls[0] += 1
        output.copy_(hasher(tokens, batch))

    checkpoint("engram_warmup")
    for _ in range(2):
        hasher(tokens, batch)
    hasher.history.zero_()
    T.npu.synchronize()
    graph, stream = BreakableCUDAGraph(), T.npu.Stream()
    stream.wait_stream(T.npu.current_stream())
    checkpoint("engram_capture")
    with enable_breakable_cuda_graph(), BreakableCUDAGraphCapture(
        cuda_graph=graph, pool=T.npu.graph_pool_handle(), stream=stream,
    ):
        hash_break()
        consumer = output.to(T.float32) * 2
    T.npu.current_stream().wait_stream(stream)
    T.npu.synchronize()
    if calls[0] != 0:
        raise AssertionError("Engram eager break executed during capture")
    R["capture_supported"] = True
    R.setdefault("checks", {})["capture"] = "PASS"
    R["cases"] = []
    for case in case_layouts():
        checkpoint("engram_replay:" + case["name"])
        prepare_hasher_batch(batch, case)
        prepare_hasher_batch(cpu_batch, case)
        ids = T.tensor([((case["seed"] + i) % 100) + 1 for i in range(O.tokens)])
        tokens.copy_(ids.to(tokens.device))
        history = T.tensor([[0, 0], [3, 4], [5, 6], [0, 0]], dtype=T.int32)
        hasher.history.copy_(history.to(hasher.history.device))
        reference.history.copy_(history)
        expected_hash = reference(ids, cpu_batch)
        before_calls = calls[0]
        with enable_breakable_cuda_graph():
            graph.replay()
        T.npu.synchronize()
        if calls[0] != before_calls + 1:
            raise AssertionError("History/hash break did not execute exactly once per replay")
        metrics = compare(
            {"hash": expected_hash, "history": reference.history, "consumer": expected_hash.float() * 2},
            {"hash": output.cpu(), "history": hasher.history.cpu(), "consumer": consumer.cpu()},
            "Engram CPU fallback and graph break")
        R["cases"].append({"name": case["name"], "status": "PASS", "comparisons": metrics})
        checkpoint("engram_passed:" + case["name"])
    R["break_calls"] = calls[0]
    check_engram_verify(hasher, reference)
    R.setdefault("checks", {}).update(eager_replay="PASS", cpu_reference="PASS", verify_commit="PASS")
    R["scope"] = "Actual EngramHasher fallback and shared breakable graph, synthetic identity tokenizer and tiny hash parameters. No lookup tables, lookup communication, or model weights."


def check_engram_verify(hasher, reference):
    from sglang.srt.model_executor.forward_mode import ForwardMode
    checkpoint("engram_verify_deferred_commit")
    batch = SimpleNamespace(
        forward_mode=ForwardMode.TARGET_VERIFY, ragged_verify_layout=None,
        spec_info=SimpleNamespace(draft_token_num=5),
        req_pool_indices=device_tensor([1], T.int64),
        positions=device_tensor([10, 11, 12, 13, 14], T.int64),
        out_cache_loc=device_tensor([128, 129, 130, 131, 132], T.int64),
    )
    ids = device_tensor([7, 8, 9, 10, 11], T.int64)
    before = hasher.history.cpu().clone()
    hasher(ids, batch)
    T.npu.synchronize()
    compare({"history": before}, {"history": hasher.history.cpu()}, "verify must not commit")
    reference.history.copy_(before)
    reference.commit_after_verify(T.tensor([[7, 8, 9, 10, 11]]), T.tensor([1]), T.tensor([3]))
    hasher.commit_after_verify(ids.view(1, 5), batch.req_pool_indices, device_tensor([3], T.int64))
    T.npu.synchronize()
    R["verify_commit"] = compare({"history": reference.history}, {"history": hasher.history.cpu()}, "accepted verify commit")


def probe_mhc_premix():
    load_device()
    function = api("hyper_conn", "mhc_pre_dpskv4")
    rows, mult, hidden = O.tokens, 4, O.hidden_size
    x = random_tensor((rows, mult, hidden), O.seed)
    fn = random_tensor((24, mult * hidden), O.seed + 1, scale=0.005, dtype=T.float32)
    scale = device_tensor([0.1, 0.1, 0.1], T.float32)
    base = T.zeros(24, dtype=T.float32, device=x.device)
    previous_pre = T.zeros((rows, mult), dtype=T.float32, device=x.device)
    cases = [
        {"name": "identity_A", "weights": [1.0, 0.0, 0.0, 0.0], "seed": O.seed},
        {"name": "uniform_same_x", "weights": [0.25] * 4, "seed": O.seed},
        {"name": "nonuniform_same_x", "weights": [0.1, 0.2, 0.3, 0.4], "seed": O.seed},
        {"name": "different_x", "weights": [0.25] * 4, "seed": O.seed + 2},
        {"name": "repeat_A", "weights": [1.0, 0.0, 0.0, 0.0], "seed": O.seed},
    ]
    R["entrypoint"] = "hyper_conn.mhc_pre_dpskv4"
    R["implementation"] = "cann_recipes_infer_cann920"
    R["parameter_shapes"] = {"x": list(x.shape), "fn": list(fn.shape),
                             "scale": list(scale.shape), "base": list(base.shape)}

    def prepare(case):
        x.copy_(random_tensor(x.shape, case["seed"]))
        previous_pre.copy_(device_tensor(case["weights"], T.float32).expand(rows, mult))
        return addresses({"x": x, "fn": fn, "scale": scale, "base": base, "previous_pre": previous_pre})

    def body():
        y, post_mix, comb, pre = function(
            x, fn, scale, base, mult, 20, 1e-6, 1e-6,
            pre_mix=previous_pre, return_pre=True,
            impl="cann_recipes_infer_cann920",
        )
        return {"mixed": y, "pre": pre, "post": post_mix, "comb": comb}

    def observe(out, case):
        expected_shapes = {"mixed": (rows, hidden), "pre": (rows, mult),
                           "post": (rows, mult), "comb": (rows, mult, mult)}
        for name, shape in expected_shapes.items():
            if tuple(out[name].shape) != shape:
                raise AssertionError(f"mHC {name} shape: {tuple(out[name].shape)} != {shape}")
        reference = (x.cpu().float() * previous_pre.cpu().unsqueeze(-1)).sum(1)
        R.setdefault("fold_reference", {})[case["name"]] = compare(
            {"mixed": reference.to(out["mixed"].dtype)}, {"mixed": out["mixed"].cpu()},
            "previous-sublayer pre-mix fold", rtol=O.reference_rtol, atol=O.reference_atol)
        return out

    observations = matrix(cases, prepare, lambda case: None, body, observe)
    reference = {name: observations[0][name] for name in ("pre", "post", "comb")}
    R["same_x_triplet_invariance"] = [
        compare(reference, {name: observations[i][name] for name in reference}, "own triplet does not depend on previous pre")
        for i in (1, 2)
    ]
    R.setdefault("checks", {})["premix_fold"] = "PASS"
    R["scope"] = "Merged mHC public premix call, fixed-shape eager/replay and previous-pre folding. No external RMSNorm, post kernel, full HcState chain or multi-stream test."


def idle_chain_layout(case, ratio, tokens, page_size):
    live, prefix = case["live"], case["prefix"]
    if not 0 <= live <= tokens:
        raise ValueError("Idle probe live rows exceed the static bucket")
    positions = [prefix + i if i < live else 0 for i in range(tokens)]
    swa_locs = [page_size + prefix + i if i < live else 0 for i in range(tokens)]
    capacity = min(tokens, tokens // ratio + 1) if ratio else 0
    if ratio == 1:
        cmp_locs = [loc if i < live else -1 for i, loc in enumerate(swa_locs)]
    elif ratio == 2:
        cmp_locs = [swa_locs[i] // ratio for i in range(live) if positions[i] % ratio == ratio - 1]
        cmp_locs += [-1] * (capacity - len(cmp_locs))
    else:
        cmp_locs = []
    return {"positions": positions, "swa_locs": swa_locs, "cmp_locs": cmp_locs,
            "used": live, "query_rows": live if live else tokens,
            "kv_length": prefix + live if live else tokens,
            "bank": case["bank"] if live else 0}


def idle_chain_protected(persistent):
    protected = {name: value for name, value in persistent.items() if name != "original"}
    protected["original"] = persistent["original"][1:]
    protected["state"] = persistent["state"][1:].contiguous().view(T.uint8)
    return protected


def prepare_idle_chain(buffers, case, ratio, functions):
    layout = idle_chain_layout(case, ratio, O.tokens, O.page_size)
    for name, value in query_geometry(case, ratio).items():
        stage(buffers, name, value)
    stage(buffers, "cu", device_tensor([0, layout["query_rows"]], T.int32))
    stage(buffers, "kv_lengths", device_tensor([layout["kv_length"]], T.int32))
    if not case["live"]:
        buffers["window"][:, :, 0].zero_()
    stage(buffers, "used", device_tensor([layout["used"]], T.int32))
    stage(buffers, "start", device_tensor([case["prefix"] if case["live"] else 0], T.int32))
    stage(buffers, "bank", device_tensor([layout["bank"]], T.int32))
    stage(buffers, "swa_locs", device_tensor(layout["swa_locs"], T.int64))
    cols = math.ceil(O.tokens / O.page_size)
    full = T.tensor([list(range(1, cols + 1)) if case["live"] else [0] * cols], dtype=T.int32)
    stage(buffers, "block_table", full.to(f"npu:{O.device}"))
    stage(buffers, "x", random_tensor((O.tokens, O.hidden_size), case["seed"]))
    stage(buffers, "swa_input", random_tensor((O.tokens, 512), case["seed"] + 1))
    stage(buffers, "q", random_tensor((O.tokens, 64, 512), case["seed"] + 2))
    stage(buffers, "metadata", attention_metadata(functions.attention_metadata, buffers, ratio))
    if ratio:
        stage(buffers, "cmp_locs", device_tensor(layout["cmp_locs"], T.int64))
        stage(buffers, "compressed_lengths", buffers["kv_lengths"] // ratio)
        stage(buffers, "index_kv_lengths", buffers["kv_lengths"] // ratio)
        stage(buffers, "residual", buffers["kv_lengths"] % ratio)
        factor = (O.page_size // ratio) // O.index_page_size
        table = (full.unsqueeze(-1) * factor + T.arange(factor, dtype=T.int32)).reshape(1, -1)
        stage(buffers, "index_table", table.to(f"npu:{O.device}"))
        q, scale = index_fixture(O.tokens * 32, case["seed"] + 3)
        stage(buffers, "index_q", q.view(O.tokens, 32, 64).to(f"npu:{O.device}"))
        stage(buffers, "index_qscale", scale.view(O.tokens, 32, 2, 2).to(f"npu:{O.device}"))
        index = dict(buffers, q=buffers["index_q"], qscale=buffers["index_qscale"],
                     block_table=buffers["index_table"])
        index_metadata(functions.qli_metadata, functions.qsli_metadata, index,
                       case, ratio, candidate=ratio == 1, zero_padding=True)
        for key in ("qli_metadata", "candidate_bound", "qsli_metadata"):
            if key in index:
                stage(buffers, key, index[key])
    R.setdefault("idle_layouts", {})[case["name"]] = {
        "physical_rows": O.tokens, "effective_rows": case["live"],
        "cu": [0, layout["query_rows"]], "kv_length": layout["kv_length"],
        "seqused": layout["used"], "bank": layout["bank"],
        "window_length_zero": not bool(case["live"]),
        "cmp_skip_rows": layout["cmp_locs"].count(-1),
    }
    return addresses(buffers)


def run_idle_chain(buffers, ratio, functions):
    writer = functions.writer
    original = writer_cache_view(buffers["original"], "fp8")
    writer(original.view(-1, 544), buffers["swa_input"], buffers["swa_locs"],
           quant_group_size=32, quant_mode="mxfp8_bf16", round_scale=True, x_scale=1.0)
    if ratio:
        if ratio == 2:
            latent = functions.compressor(
                x=buffers["x"], wkv=buffers["wkv"], wgate=buffers["wgate"],
                seqused=buffers["used"], start_pos=buffers["start"], cmp_ratio=2,
                cu_seqlens=buffers["cu"], state_cache=buffers["state"], state_block_table=buffers["bank"])
        else:
            latent = T.nn.functional.linear(buffers["x"], buffers["wkv"])
        raw_latent = latent
        latent = N.npu_rms_norm(latent, buffers["kv_norm"], 1e-6)[0]
        if latent.shape[0] != buffers["cmp_locs"].numel():
            raise AssertionError("Compressor output rows do not match the fixed writer capacity")
        writer(buffers["compressed"].view(-1, 320), latent, buffers["cmp_locs"],
               quant_group_size=16, quant_mode="mxfp4_bf16", round_scale=True, x_scale=1.0)
        common = dict(latent=latent, wk=N.npu_format_cast(buffers["wk"], N.Format.FRACTAL_NZ),
                      norm_weight=buffers["index_norm"], rope_sin=buffers["sin"], rope_cos=buffers["cos"],
                      cache_index=buffers["cmp_locs"], norm_eps=1e-6)
        functions.prologue(**common, k_cache=buffers["key"],
                           k_scale_cache=buffers["key_scale"].view(*buffers["key_scale"].shape[:3], 4), storage_mode=0)
        if ratio == 1:
            functions.prologue(**common, k_cache=buffers["combined_key"].unsqueeze(-2),
                               k_scale_cache=None, storage_mode=1, combined_block_size=8)
        index = dict(buffers, q=buffers["index_q"], qscale=buffers["index_qscale"],
                     block_table=buffers["index_table"])
        result = run_qli(functions.qli, index, ratio, candidate=ratio == 1)
        topk = run_qsli(functions.qsli, index, result[2], result[3])[0] if ratio == 1 else result[0]
    output = T.full_like(buffers["q"], float("nan"))
    kwargs = dict(ori_kv=buffers["original"], ori_sparse_indices=buffers["window"],
                  ori_topk_length=buffers["window_length"], ori_block_table=buffers["block_table"],
                  cu_seqlens_q=buffers["cu"], seqused_q=None, seqused_ori_kv=buffers["kv_lengths"],
                  sinks=buffers["sinks"], metadata=buffers["metadata"], quant_mode=1,
                  softmax_scale=1 / math.sqrt(512), layout_q="TND", layout_kv="PA_BBND", return_softmax_lse=False)
    if ratio:
        kwargs.update(cmp_kv=buffers["compressed"], cmp_sparse_indices=topk.to(T.int32).view(O.tokens, 1, 512),
                      cmp_block_table=buffers["block_table"], seqused_cmp_kv=buffers["compressed_lengths"],
                      cmp_topk_length=buffers["cmp_length"])
    functions.attention(buffers["q"], out=output,
                        lse=T.empty((0,), dtype=T.float32, device=output.device), **kwargs)
    result = {"output": output}
    if ratio:
        result["topk"] = topk
        result["source_raw"] = raw_latent
        result["source_norm"] = latent
    return result


def prime_idle_chain_pair(buffers, compressor):
    checkpoint("prime_legal_pending_pair")
    before = snapshot({"other_banks": buffers["state"][:2]})
    buffers["state"][2].zero_()
    compressor(
        x=random_tensor((O.tokens, O.hidden_size), O.seed + 20),
        wkv=buffers["wkv"], wgate=buffers["wgate"], state_cache=buffers["state"],
        cmp_ratio=2, state_block_table=device_tensor([2], T.int32),
        cu_seqlens=device_tensor([0, 1], T.int32), seqused=device_tensor([1], T.int32),
        start_pos=device_tensor([0], T.int32))
    T.npu.synchronize()
    if not T.equal(before["other_banks"], buffers["state"][:2].cpu()):
        raise AssertionError("Pending-pair setup changed an unrelated bank")
    raw = buffers["state"][2].cpu().contiguous().view(T.uint8).numpy().tobytes()
    R["pending_pair_setup"] = {
        "bank": 2, "prefix_before": 0, "live": 1, "prefix_after": 1,
        "seed": O.seed + 20, "state_sha256": hashlib.sha256(raw).hexdigest(),
        "method": "Zero private bank, then call actual compressor_v2 with one valid token before capture.",
    }


def probe_idle_chain(ratio, *, include_idle=True, sync_before_replay=False):
    load_device()
    functions = SimpleNamespace(
        writer=epilog(), attention=api("attention", "mixed_quant_sparse_flash_mla_dsl"),
        attention_metadata=api("attention", "mixed_quant_sparse_flash_mla_metadata_dsl"))
    if ratio:
        functions.compressor = api("attention", "compressor_v2") if ratio == 2 else None
        functions.prologue = api("indexer", "indexer_prologue_k_dsl")
        functions.qli = api("indexer", "quant_lightning_indexer_dsl")
        functions.qli_metadata = api("indexer", "quant_lightning_indexer_metadata_dsl")
        functions.qsli = api("indexer", "quant_sparse_lightning_indexer_dsl") if ratio == 1 else None
        functions.qsli_metadata = api("indexer", "quant_sparse_lightning_indexer_metadata_dsl") if ratio == 1 else None
    n, pages = O.tokens, math.ceil(O.tokens / O.page_size) + 2
    original, _ = packed_cache("fp8", pages * O.page_size, O.seed)
    buffers = {"original": original.view(pages, O.page_size, 1, 544).to(f"npu:{O.device}"),
               "state": random_tensor((3, O.state_ring_size, 1024), O.seed + 1, dtype=T.float32),
               "sinks": device_tensor([-10000.0] * 64, T.float32)}
    persistent = {key: buffers[key] for key in ("original", "state")}
    if ratio:
        rows, capacity = pages * (O.page_size // ratio), min(n, n // ratio + 1)
        compressed, _ = packed_cache("fp4", rows, O.seed + 2)
        buffers["compressed"] = compressed.view(pages, O.page_size // ratio, 1, 320).to(f"npu:{O.device}")
        key, scale = index_fixture(rows, O.seed + 3)
        for name, value in index_cache_views(key, scale, O.index_page_size).items():
            buffers[name] = value.to(f"npu:{O.device}")
        buffers.update(wkv=random_tensor((512, O.hidden_size), O.seed + 4, scale=0.01),
                       wgate=random_tensor((512, O.hidden_size), O.seed + 5, scale=0.01),
                       wk=random_tensor((128, 512), O.seed + 6, scale=0.03),
                       kv_norm=device_tensor([1.0] * 512, T.bfloat16),
                       index_norm=device_tensor([1.0] * 128, T.float32),
                       sin=device_tensor([[0.0] * 64] * capacity, T.float32),
                       cos=device_tensor([[1.0] * 64] * capacity, T.float32),
                       weights=random_tensor((n, 32), O.seed + 7, dtype=T.float32).abs() + 0.01)
        persistent.update({key: buffers[key] for key in ("compressed", "key", "key_scale", "combined_key")})
    if ratio == 2:
        prime_idle_chain_pair(buffers, functions.compressor)
    cases = [
        {"name": "live_full", "live": n, "prefix": 0, "bank": 1, "seed": O.seed},
        {"name": "idle_after_full", "live": 0, "prefix": 0, "bank": 0, "seed": O.seed + 10},
        {"name": "live_odd_prefix", "live": n // 2 - 1, "prefix": 1, "bank": 2, "seed": O.seed + 11},
        {"name": "idle_after_short", "live": 0, "prefix": 0, "bank": 0, "seed": O.seed + 12},
        {"name": "live_again", "live": n, "prefix": 0, "bank": 1, "seed": O.seed},
    ]
    if not include_idle:
        cases = [case for case in cases if case["live"]]
    R["insert_idle_steps"] = include_idle
    R["sync_before_replay"] = sync_before_replay
    saved = {}

    def prepare(case):
        return prepare_idle_chain(buffers, case, ratio, functions)

    def reset(case):
        if case["name"] not in saved:
            saved[case["name"]] = snapshot(persistent)
        for name, value in saved[case["name"]].items():
            persistent[name].copy_(value.to(persistent[name].device))

    def observe(out, case):
        values = snapshot(idle_chain_protected(persistent))
        if not case["live"]:
            checks = R.setdefault("idle_state_checks", {}).setdefault(case["name"], {})
            mode = R["phase"].split(":", 1)[0]
            checks[mode] = "FAIL"
            reference = idle_chain_protected(saved[case["name"]])
            for name, value in values.items():
                if not T.equal(value, reference[name].cpu()):
                    raise AssertionError("Idle replay changed protected real state/cache: " + name)
            checks[mode] = "PASS"
            R.setdefault("ignored_idle_output", {}).setdefault(case["name"], []).append({
                "phase": R["phase"], "finite": int(T.isfinite(out["output"]).sum().cpu()),
                "elements": out["output"].numel(), "comparison": "ignored; all effective lengths are zero"})
        values["attention_live"] = out["output"][:case["live"]]
        if ratio:
            values["topk_live"] = checked_topk(out["topk"], case, ratio, "idle-chain indexer")
            count = compression_count(case["prefix"], case["live"], ratio)
            values["source_raw_valid"] = out["source_raw"][:count]
            values["source_norm_valid"] = out["source_norm"][:count]
        return values

    R["scope"] = "Single-NPU synthetic live/idle/live chain, identity RoPE, physical bucket queries on idle. Not real metadata-hook admission, multi-stream, DeepEP, or full-model validation."
    R["idle_protocol"] = "cu=[0,T_cap], KV=T_cap, seqused=0, state bank=0, c-loc=-1, SWA loc=0, sparse lengths=0; QLI/QSLI still run over dummy rows."
    R["capture_writer_indexer_attention_together"] = True
    idle_names = [case["name"] for case in cases if not case["live"]]
    try:
        matrix(cases, prepare, reset, lambda: run_idle_chain(buffers, ratio, functions),
               observe, collect_failures=True, sync_before_replay=sync_before_replay)
    finally:
        checks = R.setdefault("checks", {})
        statuses = [R.get("idle_state_checks", {}).get(name, {}).get(mode, "INCOMPLETE")
                    for name in idle_names for mode in ("eager", "replay")]
        checks["idle_real_state_unchanged"] = (
            "NOT_RUN" if not statuses else "PASS" if all(value == "PASS" for value in statuses)
            else "FAIL" if "FAIL" in statuses else "INCOMPLETE")


def tensor_digest(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().view(T.uint8).numpy().tobytes()).hexdigest()


def compressor_boundary_history(buffers, body):
    history = {}
    R["history_setup"] = {}
    for prefix in (0, 1, 2, 3):
        checkpoint(f"history_prefix:{prefix}")
        buffers["state"].zero_()
        buffers["x"].copy_(random_tensor(buffers["x"].shape, O.seed + 20))
        buffers["cu"].copy_(device_tensor([0, prefix], T.int32))
        buffers["used"].fill_(prefix)
        buffers["start"].zero_()
        buffers["bank"].fill_(2)
        if prefix:
            body()
        T.npu.synchronize()
        history[prefix] = buffers["state"].cpu().clone()
        R["history_setup"][str(prefix)] = {"bank": 2, "valid_prefix": prefix,
                                          "state_sha256": tensor_digest(history[prefix])}
    return history


def boundary_compare(reference, actual, label):
    metrics, failures = {}, []
    for name in reference:
        try:
            metrics.update(compare({name: reference[name]}, {name: actual[name]}, label))
        except AssertionError as exc:
            detail = dict(R["comparison_failure"])
            failures.append(detail)
            metrics[name] = dict(detail["metrics"], status="FAIL", error=str(exc))
    raw, got = reference["raw"].float(), actual["raw"].float()
    row_errors = (raw - got).abs().amax(dim=-1)
    return {"status": "FAIL" if failures else "PASS", "comparisons": metrics,
            "max_abs_by_row": [float(x) if bool(T.isfinite(x)) else None for x in row_errors],
            "exact_by_row": [bool(T.equal(a, b)) for a, b in zip(reference["raw"], actual["raw"])],
            "failures": failures}


def probe_compressor_boundary():
    load_device()
    function = api("attention", "compressor_v2")
    n = O.tokens
    buffers = {
        "x": random_tensor((n, O.hidden_size), O.seed),
        "wkv": random_tensor((512, O.hidden_size), O.seed + 4, scale=0.01),
        "wgate": random_tensor((512, O.hidden_size), O.seed + 5, scale=0.01),
        "state": device_tensor([[[0.0] * 1024] * O.state_ring_size] * 3, T.float32),
        "cu": device_tensor([0, n], T.int32), "used": device_tensor([n], T.int32),
        "start": device_tensor([0], T.int32), "bank": device_tensor([0], T.int32),
    }

    def body():
        return function(x=buffers["x"], wkv=buffers["wkv"], wgate=buffers["wgate"],
                        state_cache=buffers["state"], cmp_ratio=2,
                        state_block_table=buffers["bank"], cu_seqlens=buffers["cu"],
                        seqused=buffers["used"], start_pos=buffers["start"])

    history = compressor_boundary_history(buffers, body)
    buffers["state"].zero_()
    buffers["x"].copy_(random_tensor(buffers["x"].shape, O.seed))
    buffers["cu"].copy_(device_tensor([0, n], T.int32))
    buffers["used"].fill_(n)
    buffers["start"].zero_()
    buffers["bank"].zero_()
    graph, output, stream = capture(body)
    R["checks"] = {"capture": "PASS"}
    R["capture_supported"] = True
    R["scope"] = "Isolated actual compressor_v2; eager repeats and graph repeats share a fully synchronized capture stream. No norm/writer/indexer/attention/collective."
    R["cases"], observations = [], {}
    pointers = addresses(buffers)
    cases = [("even_full", 0, n, O.seed), ("odd_one", 1, 1, O.seed + 11),
             ("odd_long", 1, n // 2 - 1, O.seed + 11),
             ("even_long", 2, n // 2 - 1, O.seed + 11),
             ("shifted_odd_long", 3, n // 2 - 1, O.seed + 11),
             ("odd_long_repeat", 1, n // 2 - 1, O.seed + 11)]
    for name, prefix, live, seed in cases:
        values, inputs = {}, {}
        for mode in ("eager1", "eager2", "replay1", "replay2"):
            checkpoint(f"{mode}:{name}")
            with T.npu.stream(stream):
                buffers["x"].copy_(random_tensor(buffers["x"].shape, seed))
                buffers["state"].copy_(history[prefix].to(buffers["state"].device))
                buffers["cu"].copy_(device_tensor([0, live], T.int32))
                buffers["used"].fill_(live)
                buffers["start"].fill_(prefix)
                buffers["bank"].fill_(2)
                T.npu.synchronize()
                inputs[mode] = {"state_sha256": tensor_digest(buffers["state"]),
                                "x_sha256": tensor_digest(buffers["x"])}
                if inputs[mode]["state_sha256"] != tensor_digest(history[prefix]):
                    raise AssertionError("GPU state differs from the intended initial history")
                if mode.startswith("replay"):
                    graph.replay()
                    result = output
                else:
                    result = body()
                T.npu.synchronize()
                count = compression_count(prefix, live)
                values[mode] = snapshot({"raw": result[:count],
                                         "state": buffers["state"][1:].contiguous().view(T.uint8)})
            if addresses(buffers) != pointers:
                raise AssertionError("Compressor-boundary input storage was rebound")
        if len({tuple(sorted(x.items())) for x in inputs.values()}) != 1:
            raise AssertionError("Repeated calls did not receive identical x/state")
        comparisons = {mode: boundary_compare(values["eager1"], values[mode], name + "/" + mode)
                       for mode in ("eager2", "replay1", "replay2")}
        observations[name] = values
        R["cases"].append({"name": name, "parameters": {"prefix": prefix, "live": live, "bank": 2, "seed": seed},
                           "status": "PASS" if all(v["status"] == "PASS" for v in comparisons.values()) else "FAIL",
                           "input_hashes": inputs, "comparisons": comparisons})
        checkpoint("collected:" + name)
    invariance = {}
    for mode in ("eager1", "eager2", "replay1", "replay2"):
        a = observations["odd_one"][mode]["raw"][:1]
        b = observations["odd_long"][mode]["raw"][:1]
        invariance[mode] = boundary_compare({"raw": a}, {"raw": b}, "same first pair/" + mode)
    R["first_pair_invariance"] = invariance
    R["bound_input_pointers"] = pointers
    R["checks"]["eager_repeat"] = "PASS" if all(c["comparisons"]["eager2"]["status"] == "PASS" for c in R["cases"]) else "FAIL"
    R["checks"]["eager_replay"] = "PASS" if all(c["comparisons"][m]["status"] == "PASS" for c in R["cases"] for m in ("replay1", "replay2")) else "FAIL"
    R["checks"]["first_pair_invariance"] = "PASS" if all(v["status"] == "PASS" for v in invariance.values()) else "FAIL"
    if "FAIL" in R["checks"].values():
        raise AssertionError("Compressor boundary comparison failed; all repeats and row diagnostics retained")


def dispatch_probe(name):
    if name == "environment":
        probe_environment()
    elif name == "geometry":
        probe_geometry()
    elif name == "metadata":
        probe_metadata()
    elif name == "compressor":
        probe_compressor()
    elif name == "compressor_boundary":
        probe_compressor_boundary()
    elif name.startswith("epilog_"):
        probe_epilog(name.split("_", 1)[1])
    elif name == "prologue":
        probe_prologue()
    elif name == "qli_c2":
        probe_index(False)
    elif name == "candidate_chain":
        probe_index(True)
    elif name.startswith("attention_r"):
        probe_attention(int(name[-1]))
    elif name.startswith("writer_attention_r"):
        probe_attention(int(name[-1]), use_writer=True)
    elif name == "active_chain_r2_sync":
        probe_idle_chain(2, include_idle=False, sync_before_replay=True)
    elif name == "active_chain_r2":
        probe_idle_chain(2, include_idle=False)
    elif name.startswith("idle_chain_r"):
        probe_idle_chain(int(name[-1]))
    elif name == "engram_break":
        probe_engram()
    elif name == "mhc_premix":
        probe_mhc_premix()
    else:
        raise ValueError(name)


def worker(options):
    global O, R
    O = options
    R = {"probe": options.worker, "status": "RUNNING", "phase": "start",
         "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
         "script_revision": SCRIPT_REVISION,
         "rtol": options.rtol, "atol": options.atol,
         "reference_rtol": options.reference_rtol, "reference_atol": options.reference_atol}
    started = time.monotonic()
    checkpoint("start")
    try:
        dispatch_probe(options.worker)
        R["status"] = "PASS"
        checkpoint("complete")
    except (Blocked, ImportError) as exc:
        R["status"], R["error"] = "BLOCKED", str(exc)
        R["traceback"] = traceback.format_exc()
        print(R["traceback"], flush=True)
    except Exception as exc:
        R["status"], R["error"] = "FAIL", f"{type(exc).__name__}: {exc}"
        R["traceback"] = traceback.format_exc()
        print(R["traceback"], flush=True)
    finally:
        R["elapsed_seconds"] = round(time.monotonic() - started, 3)
        atomic_json(Path(options.output_dir) / (options.worker + ".json"), R)
    return 0 if R["status"] == "PASS" else 2 if R["status"] == "BLOCKED" else 1


def stop_child(process):
    if process.poll() is not None:
        return
    if os.name == "posix":
        os.killpg(process.pid, signal.SIGKILL)
    else:
        process.kill()
    process.wait(timeout=15)


def run_child(name, options):
    output = Path(options.output_dir)
    argv = [sys.executable, "-u", str(Path(__file__).resolve()), "--worker", name,
            "--output-dir", str(output), "--device", str(options.device),
            "--tokens", str(options.tokens), "--hidden-size", str(options.hidden_size),
            "--page-size", str(options.page_size), "--candidate-blocks", str(options.candidate_blocks),
            "--index-page-size", str(options.index_page_size),
            "--reference-rtol", str(options.reference_rtol), "--reference-atol", str(options.reference_atol),
            "--state-ring-size", str(options.state_ring_size),
            "--seed", str(options.seed), "--rtol", str(options.rtol), "--atol", str(options.atol)]
    if options.sglang_root:
        argv += ["--sglang-root", options.sglang_root]
    for registration_module in options.import_module:
        argv += ["--import-module", registration_module]
    started = time.monotonic()
    timeout = False
    with (output / (name + ".log")).open("w", encoding="utf-8") as log:
        process = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=os.name == "posix")
        try:
            process.wait(timeout=options.timeout)
        except subprocess.TimeoutExpired:
            timeout = True
            stop_child(process)
        except BaseException:
            stop_child(process)
            raise
    result_file = output / (name + ".json")
    try:
        result = json.loads(result_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        result = {"probe": name, "phase": "before_first_checkpoint"}
    result["returncode"] = process.returncode
    result["wall_seconds"] = round(time.monotonic() - started, 3)
    if timeout:
        result["status"] = "TIMEOUT"
        result["error"] = f"Worker exceeded {options.timeout}s; private process group terminated"
    elif result.get("status") not in ("PASS", "FAIL", "BLOCKED"):
        result["status"] = "CRASH"
        result["error"] = f"Worker exited without completing its report (code {process.returncode})"
    elif process.returncode and result["status"] == "PASS":
        result["status"] = "CRASH"
        result["error"] = f"Worker reported PASS but exited with code {process.returncode}"
    atomic_json(result_file, result)
    return result


def finish_report(options, results, interrupted=False):
    output = Path(options.output_dir)
    summary = {
        "schema_version": 2, "script_revision": SCRIPT_REVISION, "expected_source_baseline": BASELINE,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "options": {k: v for k, v in vars(options).items() if k != "worker"},
        "interrupted": interrupted, "results": results,
        "not_covered": [
            "Full model, real checkpoint/tokenizer/Engram lookup tables",
            "Full mHC state/post/external norm chain and multi-stream joins",
            "Real serving allocator/source-group lifetime and cache sharing",
            "Real model compressor/RoPE/cache/indexer/attention integration",
            "DSpark packed aux hidden and draft KV injection",
            "DP/TP/EP collectives, multi-rank idle replay and admission consensus",
            "End-to-end accuracy, throughput, latency and capture memory budget",
        ],
        "interpretation": (
            "PASS applies only to a probe's recorded scope and cases. Metadata planning is eager. "
            "Environment/API presence is inspection, not device execution evidence. "
            "An operator FAIL before capture is not proof of graph incompatibility. "
            "No source files or serving configuration are modified."
        ),
    }
    atomic_json(output / "report.json", summary)
    lines = ["# DeepSeek V4.1 A5 prefill graph probe", "",
             f"Expected source baseline: `{BASELINE}`", "",
             "| Probe | Result | Capture | Eager/replay | CPU reference | Last phase | Seconds |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    for result in results:
        checks = result.get("checks", {})
        lines.append(f"| {result['probe']} | {result['status']} | {checks.get('capture', '-')} | {checks.get('eager_replay', '-')} | {checks.get('cpu_reference', '-')} | {result.get('phase', '').replace('|', '/')} | {result.get('wall_seconds', 0)} |")
    lines += ["", "## Failures / blocked probes", ""]
    for result in results:
        if result["status"] != "PASS":
            error = result.get("error", "See worker log").replace("\n", " ")
            lines.append(f"- **{result['probe']} / {result['status']}**: {error}")
    lines += ["", "## Interpretation", "", summary["interpretation"], "",
              "## Not covered", ""] + ["- " + value for value in summary["not_covered"]]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (output / "verify_dsv41_prefill_graph.py").write_bytes(Path(__file__).read_bytes())
    bundle = output / "dsv41-prefill-evidence.tar.gz"
    members = sorted(p for p in output.iterdir() if p.suffix in (".json", ".log", ".md", ".py"))
    with tarfile.open(bundle, "w:gz") as archive:
        for member in members:
            archive.add(member, arcname=member.name)
    counts = {status: sum(r["status"] == status for r in results)
              for status in ("PASS", "FAIL", "BLOCKED", "TIMEOUT", "CRASH")}
    print("Results: " + ", ".join(f"{key}={value}" for key, value in counts.items()), flush=True)
    print(f"Report: {output / 'report.md'}", flush=True)
    print(f"Send back: {bundle}", flush=True)
    if interrupted:
        return 130
    if any(counts[k] for k in ("FAIL", "TIMEOUT", "CRASH")):
        return 1
    return 2 if counts["BLOCKED"] else 0


def self_test(options):
    global T, O
    T, O = importlib.import_module("torch"), options
    for ratio in (1, 2):
        for prefix in range(9):
            for length in range(17):
                closing = sum((prefix + i + 1) % ratio == 0 for i in range(length))
                if closing != compression_count(prefix, length, ratio):
                    raise AssertionError("Compressed-count boundary formula failed")
    for kind, width, group in (("fp8", 544, 32), ("fp4", 320, 16)):
        packed, reference = packed_cache(kind, 17, options.seed)
        if packed.shape != (17, width):
            raise AssertionError("Invalid packed KV row width")
        if kind == "fp8":
            decoded = packed[:, :512].contiguous().view(T.float8_e4m3fn).float()
            scale = packed[:, 512:].contiguous().view(T.bfloat16).float()
        else:
            data = packed[:, :256]
            codes = T.stack((data & 15, data >> 4), dim=-1).reshape(17, 512)
            decoded = fp4_values()[codes.long()]
            scale = packed[:, 256:].contiguous().view(T.bfloat16).float()
        decoded *= scale.repeat_interleave(group, dim=-1)
        T.testing.assert_close(decoded, reference, rtol=0, atol=0)
        T.testing.assert_close(unpack_cache(kind, packed), reference, rtol=0, atol=0)
    for case in case_layouts():
        reference = geometry_reference(case, options.tokens, min(options.tokens, options.tokens // 2 + 1))
        if not bool((reference["c1_slots"][case["live"]:] == -1).all()):
            raise AssertionError("Ratio-1 padding sentinel failed")
        for ratio in (1, 2):
            full, index, physical_pages = physical_tables(case, ratio)
            if int(index.max()) >= physical_pages * options.page_size // ratio // options.index_page_size:
                raise AssertionError("Expanded index page table overflows")
    for index_page in (16, 32, 64, 128):
        key, scale = index_fixture(3 * index_page, options.seed)
        views = index_cache_views(key, scale, index_page)
        record = views["combined_key"].reshape(-1, 544)
        assert views["key"].shape == (3, index_page, 1, 64)
        assert views["combined_key"].shape == (3, index_page // 8, 544)
        T.testing.assert_close(record[:, :512].reshape(-1, 64), key, rtol=0, atol=0)
        T.testing.assert_close(record[:, 512:].reshape(-1, 4), scale, rtol=0, atol=0)
    diagnostics = bf16_diagnostics(T.tensor([1.0, -1.0, 0.0]), T.tensor([1.0, -1.0, 0.0]))
    assert diagnostics["max_bf16_ulp_distance"] == 0
    diagnostics = bf16_diagnostics(T.tensor([1.0]), T.tensor([1.0 + 1.0 / 128]))
    assert diagnostics["max_bf16_ulp_distance"] == 1
    raw = (T.arange(4 * 544) % 256).to(T.uint8).reshape(4, 544)
    typed = writer_cache_view(raw, "fp8")
    assert typed.dtype == T.float8_e4m3fn and typed.data_ptr() == raw.data_ptr()
    assert T.equal(typed.view(T.uint8), raw)
    typed.view(T.uint8)[0, 0] = 17
    assert raw[0, 0].item() == 17
    fp4 = T.zeros((4, 320), dtype=T.uint8)
    assert writer_cache_view(fp4, "fp4").dtype == T.uint8
    assert writer_cache_view(fp4, "fp4").data_ptr() == fp4.data_ptr()
    for ratio in (0, 1, 2):
        case = {"live": 0, "prefix": 0, "bank": 2}
        idle = idle_chain_layout(case, ratio, 64, options.page_size)
        assert idle["query_rows"] == 64 and idle["kv_length"] == 64
        assert idle["used"] == 0 and idle["bank"] == 0
        assert idle["swa_locs"] == [0] * 64
        assert all(loc == -1 for loc in idle["cmp_locs"])
        if ratio:
            case = {"live": 31, "prefix": 1, "bank": 2}
            layout = idle_chain_layout(case, ratio, 64, options.page_size)
            expected = [options.page_size + 1 + i for i in range(31)
                        if (1 + i + 1) % ratio == 0]
            expected = [loc // ratio for loc in expected]
            assert [loc for loc in layout["cmp_locs"] if loc >= 0] == expected
    print("CPU self-test passed: count boundaries, KV packing, index PA/record separation, BF16 diagnostics, dtype aliasing and idle protocol geometry.")
    print("This is a script/oracle check, not NPU graph evidence.")
    return 0


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", default=None, help="New directory for logs/results; must not exist")
    parser.add_argument("--sglang-root", help="Optional checkout root; its python/ takes priority over installed SGLang")
    parser.add_argument("--device", type=int, default=0, help="Logical NPU device (default: 0)")
    parser.add_argument("--tokens", type=int, default=64, help="Static token capacity, multiple of 16 (default: 64)")
    parser.add_argument("--hidden-size", type=int, default=4096, help="Synthetic compressor input width, >=1024 and divisible by 64")
    parser.add_argument("--page-size", type=int, choices=(128, 256), default=128)
    parser.add_argument("--index-page-size", type=int, choices=(16, 32, 64, 128), default=64,
                        help="Index K PA page size, distinct from the fixed candidate block size 8 (default: 64)")
    parser.add_argument("--import-module", action="append", default=[],
                        help="Explicit serving registration module to import in each worker; repeatable")
    parser.add_argument("--candidate-blocks", type=int, default=2048, help="QLI candidate output capacity (default: 2048)")
    parser.add_argument("--state-ring-size", type=int, choices=(2, 4, 8, 16, 32, 64, 128), default=2,
                        help="Ratio-2 private ring size; default 2 matches non-speculative serving")
    parser.add_argument("--timeout", type=int, default=600, help="Seconds per independent probe, including JIT compilation (default: 600)")
    parser.add_argument("--seed", type=int, default=20261116)
    parser.add_argument("--rtol", type=float, default=1e-3, help="Recorded float parity/reference tolerance")
    parser.add_argument("--atol", type=float, default=1e-3, help="Recorded eager/replay float parity tolerance")
    parser.add_argument("--reference-rtol", type=float, default=1e-3, help="Independent CPU FP32 oracle tolerance")
    parser.add_argument("--reference-atol", type=float, default=1e-3, help="Independent CPU FP32 oracle tolerance")
    parser.add_argument("--probes", default="all", help="Comma-separated probe names, or all")
    parser.add_argument("--list", action="store_true", help="List probes without importing torch")
    parser.add_argument("--self-test", action="store_true", help="CPU checks of script oracles; no torch_npu needed")
    parser.add_argument("--worker", choices=PROBES, help=argparse.SUPPRESS)
    options = parser.parse_args(argv)
    if options.tokens < 16 or options.tokens % 16:
        parser.error("--tokens must be a positive multiple of 16")
    if options.hidden_size < 1024 or options.hidden_size % 64:
        parser.error("--hidden-size must be >=1024 and divisible by 64")
    if options.timeout <= 0 or options.candidate_blocks <= 0:
        parser.error("timeout/candidate-blocks must be positive")
    tolerances = (options.rtol, options.atol, options.reference_rtol, options.reference_atol)
    if any(value < 0 or not math.isfinite(value) for value in tolerances):
        parser.error("tolerances must be finite and nonnegative")
    if (options.page_size // 2) % options.index_page_size:
        parser.error("--index-page-size must divide --page-size/2 for the ratio-2 cases")
    if options.sglang_root:
        options.sglang_root = str(Path(options.sglang_root).expanduser().resolve())
    return options


def main():
    options = parse_args()
    if options.list:
        print("\n".join(PROBES))
        return 0
    if options.self_test:
        return self_test(options)
    if options.worker:
        return worker(options)
    selected = list(PROBES) if options.probes == "all" else options.probes.split(",")
    if not selected or any(name not in PROBES for name in selected):
        raise SystemExit("Unknown --probes value; use --list")
    if len(set(selected)) != len(selected):
        raise SystemExit("Duplicate probe names are not allowed")
    output = Path(options.output_dir or time.strftime("dsv41-probe-%Y%m%d-%H%M%S")).expanduser().resolve()
    if output.exists():
        raise SystemExit(f"Output directory already exists; choose a new --output-dir: {output}")
    output.mkdir(parents=True, exist_ok=False)
    options.output_dir = str(output)
    results, interrupted = [], False
    print(f"Evidence directory: {output}", flush=True)
    print("Single NPU, synthetic tensors, no model weights. Run in the A5 serving environment.", flush=True)
    try:
        for index, name in enumerate(selected, 1):
            print(f"[{index}/{len(selected)}] {name} ...", flush=True)
            result = run_child(name, options)
            results.append(result)
            print(f"  {result['status']} ({result['wall_seconds']}s), phase={result.get('phase')}", flush=True)
    except KeyboardInterrupt:
        interrupted = True
        print("Interrupted; preserving completed and partial evidence.", flush=True)
        for name in selected:
            path = output / (name + ".json")
            if path.exists() and not any(r["probe"] == name for r in results):
                partial = json.loads(path.read_text(encoding="utf-8"))
                partial["status"] = "INTERRUPTED"
                results.append(partial)
    except Exception as exc:
        results.append({"probe": "parent", "status": "FAIL", "phase": "orchestration",
                        "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()})
    return finish_report(options, results, interrupted)


if __name__ == "__main__":
    raise SystemExit(main())
