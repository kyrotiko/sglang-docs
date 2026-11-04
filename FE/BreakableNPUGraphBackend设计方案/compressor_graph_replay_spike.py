"""Check dynamic compressor replay against eager, including state writes.

Run in the NPU serving environment, using its actual operator registration:

    python compressor_graph_replay_spike.py \
        --variant c4li --op npu.compressor --report c4li.json

Missing npu.compressor is registered by importing sgl_kernel_npu; missing
custom.compressor is registered by importing custom_ops. Override these imports
with --import-module MODULE when serving uses another registration module.
Inputs are built in. Optionally pass --kernel-repo REAL_CHECKOUT to use its two
CPU fixture builders instead; its test module is not imported.

Each case resets state before BOTH eager and replay. Only valid output rows are
compared; the entire state cache is compared. No conclusion is drawn from two
different cases producing equal/different outputs. Default mode checks only the
compressor; it does not validate indexer, epilog, attention, or a full model.

On A3, add --with-epilog to capture Hadamard/int8 quantization/K+scale scatter
(c4li), or BF16 cache writes (c4a/c128a), together with the compressor. Inactive
rows are poisoned with NaN and cleared by a device mask. They write to distinct
private scratch slots, never shared real cache slots. The eager reference writes
only valid rows; all real cache rows, including untouched guards, are compared.
This scratch reservation is experimental; a serving allocator must explicitly
provide equivalent isolation or use a masked write kernel.
"""

from __future__ import annotations

import argparse
import ast
import importlib
import json
import math
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


def explicit_state_layout(start_pos, capacities, block_size, coff, ratio):
    """A3 cache_mode=2: one physical bank per request and one dummy bank."""
    batch = len(start_pos)
    history = coff * ratio
    width = history + max(max(capacities, default=0), 1)
    dummy = batch * block_size
    table = [[dummy] * width for _ in range(batch)]
    for request, (start, capacity) in enumerate(zip(start_pos, capacities)):
        for column in range(history + capacity):
            position = start - history + column
            if position >= 0:
                table[request][column] = request * block_size + position % block_size
    return table, batch + 1, dummy


def builtin_fixture_builders(torch, arch35):
    """Synthetic eager-vs-replay inputs, using the serving state-table ABIs."""
    def build_table(starts, capacities, block_size, coff, ratio, banks_per_batch=1):
        if banks_per_batch != 1:
            raise ValueError("Built-in fixtures use one bank per request")
        table, banks, dummy = explicit_state_layout(starts, capacities, block_size, coff, ratio)
        return torch.tensor(table, dtype=torch.int32), banks, dummy

    def make_inputs(starts, seq_len, coff, ratio, head_dim, hidden, cache_mode,
                    layout, dtype, batch, block_size, ring_size=None):
        if cache_mode != 2 or layout != "TH" or len(starts) != batch:
            raise ValueError("Built-in fixtures require TH, cache_mode=2, and one start per request")
        generator = torch.Generator().manual_seed(20260813)
        width = coff * head_dim

        def normal(*shape, scale=0.02, target_dtype=dtype):
            return (torch.randn(*shape, generator=generator) * scale).to(target_dtype)

        state_block_size = ring_size or max(block_size, coff * ratio + seq_len - 1, 1)
        banks = batch if arch35 else batch + 1
        return {
            "x": normal(batch * seq_len, hidden),
            "wkv": normal(width, hidden),
            "wgate": normal(width, hidden),
            "ape": normal(ratio, width, scale=0.01, target_dtype=torch.float32),
            "norm_weight": normal(head_dim, target_dtype=torch.float32) + 1.0,
            "state_cache": normal(banks, state_block_size, 2 * width, scale=0.01, target_dtype=torch.float32),
        }

    return make_inputs, build_table


def resolve_operator(torch, requested, modules, report):
    namespace, operator = requested.split(".")

    def lookup():
        packet = getattr(getattr(torch.ops, namespace), operator, None)
        return getattr(packet, "default", None) if packet is not None else None

    loaded = []
    report["registration_modules"] = loaded
    if not modules and lookup() is None:
        modules = ["sgl_kernel_npu" if namespace == "npu" else "custom_ops"]
    for name in modules:
        try:
            module = importlib.import_module(name)
        except Exception as error:
            raise RuntimeError(
                f"Cannot import {name!r}, required to register {requested}: {error}. "
                "Run in the same environment as serving. Use --import-module MODULE "
                "if serving registers this operator through another module."
            ) from error
        loaded.append({"name": name, "file": str(getattr(module, "__file__", "unknown"))})
    op = lookup()
    if op is None:
        available = [f"{ns}.compressor" for ns in ("npu", "custom")
                     if getattr(getattr(torch.ops, ns), "compressor", None) is not None]
        raise RuntimeError(
            f"{requested} is not registered after imports {[m['name'] for m in loaded]}. "
            f"Registered compressor namespaces: {available or 'none'}. "
            "Check the installed kernel build and the operator used by serving. "
            "No capture or replay has run."
        )
    return op


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


def epilog_slot_plan(capacity, valid, seed):
    """Disjoint real locations and one unique scratch slot per padding row."""
    if not 0 <= valid <= capacity:
        raise ValueError("Invalid epilog row count")
    real_slots = 3 * capacity + 1
    slots = list(range(real_slots, real_slots + capacity))
    first_real = 1 + (seed % 3) * capacity
    slots[:valid] = range(first_real, first_real + valid)
    mask = [True] * valid + [False] * (capacity - valid)
    return slots, mask, real_slots


class A3EpilogProbe:
    def __init__(self, torch, torch_npu, capacity, head_dim, indexer, device):
        self.torch, self.torch_npu = torch, torch_npu
        self.capacity, self.head_dim, self.indexer = capacity, head_dim, indexer
        self.real_slots = 3 * capacity + 1
        slots = self.real_slots + capacity
        self.loc = torch.empty(capacity, dtype=torch.int64, device=device)
        self.valid = torch.empty(capacity, dtype=torch.bool, device=device)
        dtype = torch.int8 if indexer else torch.bfloat16
        self.initial_cache = torch.full((slots, 1, head_dim), 17, dtype=dtype, device=device)
        self.cache = self.initial_cache.clone()
        self.scale = self.initial_scale = self.hadamard = None
        if indexer:
            self.initial_scale = torch.full((slots, 1, 1), 1.25, dtype=torch.float16, device=device)
            self.scale = self.initial_scale.clone()
            hadamard = torch.ones((1, 1), dtype=torch.float32, device=device)
            while hadamard.shape[0] < head_dim:
                hadamard = torch.cat((torch.cat((hadamard, hadamard), dim=1),
                                      torch.cat((hadamard, -hadamard), dim=1)), dim=0) / math.sqrt(2)
            self.hadamard = hadamard.to(torch.bfloat16)
        tensors = [self.loc, self.valid, self.cache]
        if self.scale is not None:
            tensors.append(self.scale)
        self.tensors = tensors
        self.addresses = [(t.data_ptr(), tuple(t.shape), t.dtype) for t in tensors]

    def reset(self, case, valid):
        torch = self.torch
        slots, mask, _ = epilog_slot_plan(self.capacity, valid, case.seed)
        self.loc.copy_(torch.tensor(slots, dtype=torch.int64))
        self.valid.copy_(torch.tensor(mask, dtype=torch.bool))
        self.cache.copy_(self.initial_cache)
        if self.scale is not None:
            self.scale.copy_(self.initial_scale)
        assert self.addresses == [(t.data_ptr(), tuple(t.shape), t.dtype) for t in self.tensors]

    def _write(self, values, locations):
        if self.indexer:
            values = values.matmul(self.hadamard).to(self.torch.bfloat16)
            quantized, scales = self.torch_npu.npu_dynamic_quant(values)
            self.torch_npu.npu_scatter_nd_update_(
                self.cache, locations.view(-1, 1), quantized.view(-1, 1, self.head_dim))
            self.torch_npu.npu_scatter_nd_update_(
                self.scale, locations.view(-1, 1), scales.to(self.torch.float16).view(-1, 1, 1))
        else:
            self.cache[locations] = values.to(self.cache.dtype).view(-1, 1, self.head_dim)

    def apply_in_graph(self, compressed):
        torch = self.torch
        mask = self.valid[:, None]
        # Stress the case where the compressor never wrote its padded rows.
        poisoned = torch.where(mask, compressed, torch.full_like(compressed, float("nan")))
        safe = torch.where(mask, poisoned, torch.zeros_like(compressed))
        self._write(safe, self.loc)

    def apply_eager_reference(self, compressed, valid):
        # CPU control flow and slicing are intentional in the eager oracle only.
        if valid:
            self._write(compressed[:valid], self.loc[:valid])

    def snapshot_real(self):
        result = {"cache": self.cache[:self.real_slots].cpu()}
        if self.scale is not None:
            result["scale"] = self.scale[:self.real_slots].cpu()
        return result

    def check(self, expected, atol, rtol):
        cache_ok, cache_error = compare(self.torch, self.cache[:self.real_slots], expected["cache"], atol=0, rtol=0)
        scale_ok, scale_error = True, 0.0
        if self.scale is not None:
            scale_ok, scale_error = compare(self.torch, self.scale[:self.real_slots], expected["scale"], atol=atol, rtol=rtol)
        scratch_finite = bool(self.torch.isfinite(self.cache[self.real_slots:].float()).all().item())
        if self.scale is not None:
            scratch_finite = scratch_finite and bool(self.torch.isfinite(self.scale[self.real_slots:]).all().item())
        return {"cache_ok": cache_ok, "cache_max_abs_error": cache_error,
                "scale_ok": scale_ok, "scale_max_abs_error": scale_error,
                "scratch_finite": scratch_finite}


def write_report(path, report):
    if path:
        path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def run(options, report):
    import torch

    report["stage"] = "registration"
    op = resolve_operator(torch, options.op, options.import_module, report)
    import torch_npu

    torch.npu.set_device(options.device)
    torch_npu.npu.set_compile_mode(jit_compile=False)
    device = f"npu:{options.device}"
    device_name = torch.npu.get_device_name(options.device)
    arch35 = "950" in device_name.lower()
    if options.with_epilog and arch35:
        raise ValueError("--with-epilog currently models A3 writes; A5 FP8/packed epilogs require a separate probe")
    report["environment"] = {
        "torch": torch.__version__, "torch_npu": getattr(torch_npu, "__version__", "unknown"),
        "device": device_name, "op": options.op, "schema": str(op._schema),
        "imports": report["registration_modules"], "capture_api": options.capture_api,
        "kernel_repo": str(options.kernel_repo.resolve()) if options.kernel_repo else None,
        "fixtures": "kernel_checkout" if options.kernel_repo else "builtin",
        "pipeline": "compressor_with_a3_epilog" if options.with_epilog else "compressor_only",
    }
    if options.kernel_repo:
        try:
            revision = subprocess.run(["git", "-C", str(options.kernel_repo), "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
            report["environment"]["fixture_revision"] = revision.stdout.strip() or "unknown"
        except OSError:
            report["environment"]["fixture_revision"] = "unknown"
    print(json.dumps(report["environment"]), flush=True)
    if options.kernel_repo:
        make_inputs, table_builder = load_fixture_builders(options.kernel_repo, torch, arch35)
    else:
        make_inputs, table_builder = builtin_fixture_builders(torch, arch35)
    ratio, coff, head_dim = VARIANTS[options.variant]
    cases = make_cases(options.tokens, ratio)
    base = make_inputs([0, 0], options.tokens, coff, ratio, head_dim, options.hidden, 2, "TH", torch.bfloat16, 2, 16, ring_size=coff * ratio if arch35 else None)
    payloads = [make_payload(torch, case, base, table_builder, options.tokens, ratio, coff, arch35) for case in cases]
    buffers = {name: tensor.to(device) for name, tensor in payloads[0].items()}
    weights = {name: base[name].to(device) for name in ("wkv", "wgate", "ape", "norm_weight")}
    initial_state = base["state_cache"].to(device)
    state = initial_state.clone()
    epilog = A3EpilogProbe(torch, torch_npu, min(options.tokens, options.tokens // ratio + 2),
                          head_dim, options.variant == "c4li", device) if options.with_epilog else None
    if epilog is not None:
        report["epilog"] = {"real_slots": epilog.real_slots, "private_scratch_slots": epilog.capacity,
                            "poison_padding": True, "cache_comparison": "exact"}
    addresses = {name: (tensor.data_ptr(), tuple(tensor.shape), tensor.dtype) for name, tensor in buffers.items()}
    state_address = state.data_ptr()

    def load(payload, case):
        for name, value in payload.items():
            assert tuple(value.shape) == tuple(buffers[name].shape), name
            buffers[name].copy_(value)
        state.copy_(initial_state)
        assert state.data_ptr() == state_address
        assert addresses == {name: (t.data_ptr(), tuple(t.shape), t.dtype) for name, t in buffers.items()}
        if epilog is not None:
            epilog.reset(case, case.compressed_rows(ratio))

    def call():
        return op(buffers["x"], weights["wkv"], weights["wgate"], state, weights["ape"], weights["norm_weight"],
                  rope_sin=buffers["rope_sin"], rope_cos=buffers["rope_cos"],
                  state_block_table=buffers["state_block_table"], cu_seqlens=buffers["cu_seqlens"],
                  seqused=buffers["seqused"], start_pos=buffers["start_pos"], rope_head_dim=64,
                  cmp_ratio=ratio, coff=coff, norm_eps=1e-6, rotary_mode=2, cache_mode=2)

    def graph_step():
        compressed = call()
        if epilog is not None:
            epilog.apply_in_graph(compressed)
        return compressed

    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    graph = torch.npu.NPUGraph()
    pool = torch.npu.graph_pool_handle()
    with torch.npu.stream(stream):
        for _ in range(2):
            load(payloads[0], cases[0])
            graph_step()
        load(payloads[0], cases[0])
        torch.npu.synchronize()
        report["stage"] = "capture"
        write_report(options.report, report)
        if options.capture_api == "raw":
            graph.capture_begin(pool=pool, capture_error_mode="global")
            captured = graph_step()
            graph.capture_end()
        else:
            with torch.npu.graph(graph, pool=pool, stream=stream, auto_dispatch_capture=False):
                captured = graph_step()
        torch.npu.synchronize()
        expected_shape = (min(options.tokens, options.tokens // ratio + 2), head_dim)
        assert tuple(captured.shape) == expected_shape, (tuple(captured.shape), expected_shape)
        for number, (case, payload) in enumerate(zip(cases, payloads)):
            report["stage"] = f"eager:{number}:{case.name}"
            write_report(options.report, report)
            load(payload, case)
            expected = call()
            valid = case.compressed_rows(ratio)
            if epilog is not None:
                epilog.apply_eager_reference(expected, valid)
            torch.npu.synchronize()
            expected_output = expected[:valid].float().cpu()
            expected_state = state.float().cpu()
            expected_cache = epilog.snapshot_real() if epilog is not None else None
            report["stage"] = f"replay:{number}:{case.name}"
            write_report(options.report, report)
            load(payload, case)
            graph.replay()
            torch.npu.synchronize()
            output_ok, output_error = compare(torch, captured[:valid], expected_output, atol=options.atol, rtol=options.rtol)
            state_ok, state_error = compare(torch, state, expected_state, atol=options.state_atol, rtol=options.rtol)
            result = {**asdict(case), "valid_output_rows": valid, "output_ok": output_ok, "state_ok": state_ok,
                      "output_max_abs_error": output_error, "state_max_abs_error": state_error}
            if epilog is not None:
                result.update(epilog.check(expected_cache, options.state_atol, options.rtol))
            # Store non-finite error diagnostics without writing invalid JSON.
            for name in list(result):
                if name.endswith("_max_abs_error") and not math.isfinite(result[name]):
                    result[name] = str(result[name])
            report["cases"].append(result)
            print(json.dumps(result), flush=True)
            write_report(options.report, report)
    report["passed"] = all(case["output_ok"] and case["state_ok"] and
                           all(case.get(name, True) for name in ("cache_ok", "scale_ok", "scratch_finite"))
                           for case in report["cases"])
    report["stage"] = "complete"
    return 0 if report["passed"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--kernel-repo", type=Path, help="Optional real fixture checkout; omit to use built-in inputs")
    parser.add_argument("--variant", choices=VARIANTS, default="c4li")
    parser.add_argument("--op", choices=("npu.compressor", "custom.compressor"), help="Exact namespace used by serving")
    parser.add_argument("--import-module", action="append", default=[], help="Override registration imports; repeatable. Defaults: npu -> sgl_kernel_npu, custom -> custom_ops")
    parser.add_argument("--tokens", type=int, default=256)
    parser.add_argument("--hidden", type=int, default=1024)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--capture-api", choices=("raw", "context"), default="raw")
    parser.add_argument("--with-epilog", action="store_true", help="A3 only: capture Hadamard/int8/K+scale scatter for c4li, or BF16 writes for c4a/c128a, and validate all real cache rows")
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
    if options.op is None:
        parser.error("NPU runs require --op")
    if options.kernel_repo is not None:
        options.kernel_repo = options.kernel_repo.expanduser().resolve()
        fixture = options.kernel_repo / "tests/python/sgl_kernel_npu/test_compressor.py"
        if not fixture.is_file():
            parser.error(f"Fixture source does not exist: {fixture}. Omit --kernel-repo to use built-in inputs.")
    report = {"variant": options.variant, "tokens": options.tokens, "hidden": options.hidden,
              "with_epilog": options.with_epilog,
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
