"""Check dynamic compressor replay against eager, including state writes.

Run in the NPU serving environment, using its actual operator registration:

    python compressor_graph_replay_spike.py --kernel-repo /path/to/sgl-kernel-npu \
        --variant c4li --op npu.compressor --report c4li.json

For an operator registered by an additional module, pass --import-module MODULE.
Select the same op namespace and registration modules as the serving process.
Only two CPU fixture builders are extracted from the kernel checkout; its test
module is not imported and its operator/device initialization does not run.

Each case resets state before BOTH eager and replay. Only valid output rows are
compared; the entire state cache is compared. No conclusion is drawn from two
different cases producing equal/different outputs. This is a compressor gate,
not evidence that indexer, epilog, attention, or a full model already works.
"""

from __future__ import annotations

import argparse
import ast
import importlib
import json
import subprocess
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path


VARIANTS = {"c4li": (4, 2, 128), "c4a": (4, 2, 512), "c128a": (128, 1, 512)}


@dataclass(frozen=True)
class Case:
    name: str
    capacities: tuple[int, int]
    used: tuple[int, int]
    starts: tuple[int, int]
    seed: int

    def validate(self, tokens: int) -> None:
        assert 0 <= sum(self.capacities) <= tokens
        assert all(0 <= n <= c for n, c in zip(self.used, self.capacities))
        assert all(p >= 0 for p in self.starts)

    def compressed_rows(self, ratio: int) -> int:
        return sum((p + n) // ratio - p // ratio for p, n in zip(self.starts, self.used))


def make_cases(tokens: int, ratio: int) -> list[Case]:
    half, quarter = tokens // 2, tokens // 4
    cases = [
        Case("baseline", (half, half), (half, half), (0, 0), 101),
        Case("less_used", (half, half), (quarter, quarter // 2), (1, ratio - 1), 102),
        Case("cu_repartition", (3 * quarter, quarter), (3 * quarter, quarter), (3, ratio - 1), 103),
        Case("cu_short_total", (quarter, quarter), (quarter, quarter), (0, ratio - 1), 104),
        Case("one_request", (tokens, 0), (half, 0), (ratio - 1, 0), 105),
        Case("cross_boundary", (half, half), (1, 1), (ratio - 1, 2 * ratio - 1), 106),
        Case("no_new_blocks", (half, half), (1, 1), (0, 0), 107),
        Case("idle", (0, 0), (0, 0), (0, 0), 108),
    ]
    cases.append(cases[0])  # A -> other layouts -> A, with the same initial state.
    for case in cases:
        case.validate(tokens)
        assert case.compressed_rows(ratio) <= min(tokens, tokens // ratio + 2)
    return cases


def load_fixture_builders(repo: Path, torch, arch35: bool):
    source = repo / "tests/python/sgl_kernel_npu/test_compressor.py"
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    names = {"_make_inputs", "_build_explicit_state_loc_table"}
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    if {n.name for n in nodes} != names:
        raise RuntimeError(f"Fixture helpers changed; review {source} before running")
    namespace = {"torch": torch, "_is_arch35": lambda: arch35}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), namespace)
    return namespace["_make_inputs"], namespace["_build_explicit_state_loc_table"]


def make_payload(torch, case, base, table_builder, tokens, ratio, coff, arch35):
    capacity = min(tokens, tokens // ratio + 2)
    payload = {
        "x": (torch.randn(tokens, base["x"].shape[1], generator=torch.Generator().manual_seed(case.seed)) * 0.02).to(base["x"].dtype),
        "cu_seqlens": torch.tensor([0, case.capacities[0], sum(case.capacities)], dtype=torch.int32),
        "seqused": torch.tensor(case.used, dtype=torch.int32),
        "start_pos": torch.tensor(case.starts, dtype=torch.int32),
    }
    if arch35:
        payload["state_block_table"] = torch.tensor([0, 1], dtype=torch.int32)
    else:
        table, banks, dummy = table_builder(case.starts, case.capacities, base["state_cache"].shape[1], coff, ratio, banks_per_batch=1)
        assert banks == base["state_cache"].shape[0]
        padded = torch.full((2, coff * ratio + tokens), dummy, dtype=torch.int32)
        padded[:, :table.shape[1]].copy_(table)
        payload["state_block_table"] = padded
    positions = [
        block * ratio
        for prefix, used in zip(case.starts, case.used)
        for block in range(prefix // ratio, (prefix + used) // ratio)
    ]
    assert len(positions) == case.compressed_rows(ratio)
    positions += [0] * (capacity - len(positions))
    frequencies = 10000.0 ** (-torch.arange(0, 64, 2).float() / 64)
    angles = (torch.tensor(positions).float()[:, None] * frequencies).repeat_interleave(2, dim=1)
    payload["rope_sin"], payload["rope_cos"] = angles.sin(), angles.cos()
    return payload


def compare(torch, actual, expected, *, atol, rtol):
    actual, expected = actual.float().cpu(), expected.float().cpu()
    error = float((actual - expected).abs().max()) if actual.numel() else 0.0
    try:
        torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol, equal_nan=False)
        return True, error
    except AssertionError:
        return False, error


def write_report(path, report):
    if path:
        path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def run(options, report):
    import torch
    import torch_npu

    torch.npu.set_device(options.device)
    for name in options.import_module:
        importlib.import_module(name)
    device = f"npu:{options.device}"
    device_name = torch.npu.get_device_name(options.device)
    arch35 = "950" in device_name.lower()
    namespace, operator = options.op.split(".")
    op = getattr(getattr(torch.ops, namespace), operator).default
    report["environment"] = {
        "torch": torch.__version__, "torch_npu": getattr(torch_npu, "__version__", "unknown"),
        "device": device_name, "op": options.op, "schema": str(op._schema),
        "imports": options.import_module, "capture_api": options.capture_api,
        "kernel_repo": str(options.kernel_repo.resolve()),
    }
    try:
        revision = subprocess.run(["git", "-C", str(options.kernel_repo), "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
        report["environment"]["fixture_revision"] = revision.stdout.strip() or "unknown"
    except OSError:
        report["environment"]["fixture_revision"] = "unknown"
    print(json.dumps(report["environment"]), flush=True)
    make_inputs, table_builder = load_fixture_builders(options.kernel_repo, torch, arch35)
    ratio, coff, head_dim = VARIANTS[options.variant]
    cases = make_cases(options.tokens, ratio)
    base = make_inputs([0, 0], options.tokens, coff, ratio, head_dim, options.hidden, 2, "TH", torch.bfloat16, 2, 16, ring_size=coff * ratio if arch35 else None)
    payloads = [make_payload(torch, case, base, table_builder, options.tokens, ratio, coff, arch35) for case in cases]
    buffers = {name: tensor.to(device) for name, tensor in payloads[0].items()}
    weights = {name: base[name].to(device) for name in ("wkv", "wgate", "ape", "norm_weight")}
    initial_state = base["state_cache"].to(device)
    state = initial_state.clone()
    addresses = {name: (tensor.data_ptr(), tuple(tensor.shape), tensor.dtype) for name, tensor in buffers.items()}
    state_address = state.data_ptr()

    def load(payload):
        for name, value in payload.items():
            assert tuple(value.shape) == tuple(buffers[name].shape), name
            buffers[name].copy_(value)
        state.copy_(initial_state)
        assert state.data_ptr() == state_address
        assert addresses == {name: (t.data_ptr(), tuple(t.shape), t.dtype) for name, t in buffers.items()}

    def call():
        return op(buffers["x"], weights["wkv"], weights["wgate"], state, weights["ape"], weights["norm_weight"],
                  rope_sin=buffers["rope_sin"], rope_cos=buffers["rope_cos"],
                  state_block_table=buffers["state_block_table"], cu_seqlens=buffers["cu_seqlens"],
                  seqused=buffers["seqused"], start_pos=buffers["start_pos"], rope_head_dim=64,
                  cmp_ratio=ratio, coff=coff, norm_eps=1e-6, rotary_mode=2, cache_mode=2)

    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    graph = torch.npu.NPUGraph()
    pool = torch.npu.graph_pool_handle()
    with torch.npu.stream(stream):
        for _ in range(2):
            load(payloads[0])
            call()
        load(payloads[0])
        torch.npu.synchronize()
        report["stage"] = "capture"
        write_report(options.report, report)
        if options.capture_api == "raw":
            graph.capture_begin(pool=pool, capture_error_mode="global")
            captured = call()
            graph.capture_end()
        else:
            with torch.npu.graph(graph, pool=pool, stream=stream, auto_dispatch_capture=False):
                captured = call()
        torch.npu.synchronize()
        expected_shape = (min(options.tokens, options.tokens // ratio + 2), head_dim)
        assert tuple(captured.shape) == expected_shape, (tuple(captured.shape), expected_shape)
        for number, (case, payload) in enumerate(zip(cases, payloads)):
            report["stage"] = f"eager:{number}:{case.name}"
            write_report(options.report, report)
            load(payload)
            expected = call()
            torch.npu.synchronize()
            valid = case.compressed_rows(ratio)
            expected_output = expected[:valid].float().cpu()
            expected_state = state.float().cpu()
            report["stage"] = f"replay:{number}:{case.name}"
            write_report(options.report, report)
            load(payload)
            graph.replay()
            torch.npu.synchronize()
            output_ok, output_error = compare(torch, captured[:valid], expected_output, atol=options.atol, rtol=options.rtol)
            state_ok, state_error = compare(torch, state, expected_state, atol=options.state_atol, rtol=options.rtol)
            result = {**asdict(case), "valid_output_rows": valid, "output_ok": output_ok, "state_ok": state_ok,
                      "output_max_abs_error": output_error, "state_max_abs_error": state_error}
            # Store non-finite error diagnostics without writing invalid JSON.
            import math
            for name in ("output_max_abs_error", "state_max_abs_error"):
                if not math.isfinite(result[name]):
                    result[name] = str(result[name])
            report["cases"].append(result)
            print(json.dumps(result), flush=True)
            write_report(options.report, report)
    report["passed"] = all(case["output_ok"] and case["state_ok"] for case in report["cases"])
    report["stage"] = "complete"
    return 0 if report["passed"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--kernel-repo", type=Path, help="Fixture source checkout; not a replacement for installed operator code")
    parser.add_argument("--variant", choices=VARIANTS, default="c4li")
    parser.add_argument("--op", choices=("npu.compressor", "custom.compressor"), help="Exact namespace used by serving")
    parser.add_argument("--import-module", action="append", default=[], help="Additional serving registration module; repeatable")
    parser.add_argument("--tokens", type=int, default=256)
    parser.add_argument("--hidden", type=int, default=1024)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--capture-api", choices=("raw", "context"), default="raw")
    parser.add_argument("--atol", type=float, default=0.02)
    parser.add_argument("--state-atol", type=float, default=0.001)
    parser.add_argument("--rtol", type=float, default=0.01)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--list-cases", action="store_true", help="Print the case matrix without importing torch")
    options = parser.parse_args()
    if options.tokens < 256 or options.tokens % 128:
        parser.error("--tokens must be >= 256 and a multiple of 128")
    if options.hidden < 1024 or options.hidden % 512:
        parser.error("--hidden must be >= 1024 and a multiple of 512")
    if options.list_cases:
        ratio = VARIANTS[options.variant][0]
        print(json.dumps([{**asdict(case), "valid_output_rows": case.compressed_rows(ratio)} for case in make_cases(options.tokens, ratio)], indent=2))
        return 0
    if options.kernel_repo is None or options.op is None:
        parser.error("NPU runs require --kernel-repo and --op")
    report = {"variant": options.variant, "tokens": options.tokens, "hidden": options.hidden,
              "tolerances": {"atol": options.atol, "state_atol": options.state_atol, "rtol": options.rtol},
              "passed": False, "stage": "initialization", "cases": []}
    try:
        result = run(options, report)
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        traceback.print_exc()
        result = 2
    write_report(options.report, report)
    print(json.dumps({"passed": report["passed"], "stage": report["stage"], "cases": len(report["cases"])}), flush=True)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
