"""A3 C4 attention operator-chain replay check, using synthetic weights.

Copy this file AND compressor_graph_replay_spike.py to the NPU host, then run:

    python dsv4_c4_attention_graph_spike.py --report c4_chain.json

One raw NPUGraph contains main/indexer compressors, their epilogs, synthetic
Q/KV/weights projections and RoPE, query quantization, fused lightning indexer,
top-k handoff, and sparse attention. Only metadata planning and fixed-buffer
updates run before replay. No graph.update() is used.

The oracle runs the same operators eagerly, writing only valid cache rows.
State and real cache regions are reset identically before oracle and replay.
All oracle results are saved BEFORE capture. After capture, only metadata
planning/buffer updates and replay run, so an eager core-op call cannot refresh
hidden launch/tiling caches immediately before the measured replay.
This checks capture/replay agreement, not checkpoint/model correctness or
continuous-chunk state evolution. Private scratch slots are experimental and
must be reserved explicitly when integrating a serving allocator.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path

import compressor_graph_replay_spike as common


@dataclass
class Layout:
    case: common.Case
    q_cu: list[int]
    kv_lens: list[int]
    q_gather: list[int]
    q_positions: list[int]
    q_valid: list[bool]
    valid_q_rows: list[int]
    restore: list[int]
    input_valid: list[bool]
    swa_loc: list[int]
    cmp_loc: list[int]
    comp_valid: list[bool]
    swa_table: list[list[int]]
    cmp_table: list[list[int]]


def build_layout(case, tokens, page_size, max_pages):
    """No torch dependency: explicit mapping of capacity, query and cache rows."""
    case.validate(tokens)
    ratio, batch = 4, 2
    query_capacity = tokens + batch
    comp_capacity = min(tokens, tokens // ratio + batch)
    cmp_page = page_size // ratio
    real_swa = (1 + batch * max_pages) * page_size
    real_cmp = (1 + batch * max_pages) * cmp_page
    gather, positions, valid, swa_loc = [], [], [], []
    restore, input_valid = [0] * tokens, [False] * tokens
    q_cu, kv_lens, swa_table, cmp_table, comp_loc = [0], [], [], [], []
    capacity_start = 0
    for request, (capacity, used, prefix) in enumerate(zip(case.capacities, case.used, case.starts)):
        # Each request owns disjoint pages; changing seed changes page bindings.
        pages = [1 + request * max_pages + (p + case.seed) % max_pages for p in range(max_pages)]
        kv_len = prefix + used if used else ratio
        if kv_len > max_pages * page_size:
            raise ValueError("Context exceeds the fixed page-table width")
        swa_count = (kv_len + page_size - 1) // page_size
        cmp_count = ((kv_len // ratio) + cmp_page - 1) // cmp_page
        swa_table.append((pages[:swa_count] if used else [0]) + [-1] * (max_pages - (swa_count if used else 1)))
        cmp_table.append((pages[:cmp_count] if used else [0]) + [-1] * (max_pages - (cmp_count if used else 1)))
        kv_lens.append(kv_len)
        if used:
            for token in range(used):
                source = capacity_start + token
                query_row = len(gather)
                position = prefix + token
                restore[source], input_valid[source] = query_row, True
                gather.append(source)
                positions.append(position)
                valid.append(True)
                swa_loc.append(pages[position // page_size] * page_size + position % page_size)
        else:
            # A legal dummy query avoids relying on zero-query metadata support.
            gather.append(0)
            positions.append(0)
            valid.append(False)
            swa_loc.append(real_swa + len(swa_loc))
        q_cu.append(len(gather))
        for block in range(prefix // ratio, (prefix + used) // ratio):
            comp_loc.append(pages[block // cmp_page] * cmp_page + block % cmp_page)
        capacity_start += capacity
    valid_q_rows = [i for i, keep in enumerate(valid) if keep]
    while len(gather) < query_capacity:
        gather.append(0)
        positions.append(0)
        valid.append(False)
        swa_loc.append(real_swa + len(swa_loc))
    if len(gather) != query_capacity or len(comp_loc) > comp_capacity:
        raise ValueError("Query/compressed capacity overflow")
    comp_valid = [True] * len(comp_loc) + [False] * (comp_capacity - len(comp_loc))
    comp_loc += [real_cmp + row for row in range(len(comp_loc), comp_capacity)]
    assert sum(input_valid) == sum(case.used) == len(valid_q_rows)
    assert sum(comp_valid) == case.compressed_rows(ratio)
    assert len(set(swa_loc)) == len(swa_loc)
    assert len(set(comp_loc)) == len(comp_loc)
    return Layout(case, q_cu, kv_lens, gather, positions, valid, valid_q_rows,
                  restore, input_valid, swa_loc, comp_loc, comp_valid, swa_table, cmp_table)


class PagedEpilog(common.A3EpilogProbe):
    def __init__(self, torch, torch_npu, capacity, head_dim, indexer, device, real_pages, page_size, seed):
        super().__init__(torch, torch_npu, capacity, head_dim, indexer, device)
        self.page_size = page_size
        self.real_slots = real_pages * page_size
        self.pages = real_pages + (capacity + page_size - 1) // page_size
        slots = self.pages * page_size
        generator = torch.Generator().manual_seed(seed)
        if indexer:
            initial = torch.randint(-16, 17, (slots, 1, head_dim), generator=generator, dtype=torch.int8)
        else:
            initial = (torch.randn(slots, 1, head_dim, generator=generator) * 0.1).to(torch.bfloat16)
            # Different page contents make a stale page-table binding observable.
            bias = ((torch.arange(slots) // page_size) % 7 - 3).float() * 0.2
            initial = (initial.float() + bias[:, None, None]).to(torch.bfloat16)
        initial[:page_size].zero_()  # page 0 belongs only to dummy requests
        self.initial_cache = initial.to(device)
        self.cache = self.initial_cache.clone()
        if indexer:
            initial_scale = torch.full((slots, 1, 1), 0.02, dtype=torch.float16)
            initial_scale[:page_size].fill_(1.0)
            self.initial_scale = initial_scale.to(device)
            self.scale = self.initial_scale.clone()
        self.tensors = [self.loc, self.valid, self.cache] + ([self.scale] if indexer else [])
        self.addresses = [(t.data_ptr(), tuple(t.shape), t.dtype) for t in self.tensors]

    def reset_layout(self, layout):
        self.loc.copy_(self.torch.tensor(layout.cmp_loc, dtype=self.torch.int64))
        self.valid.copy_(self.torch.tensor(layout.comp_valid, dtype=self.torch.bool))
        self.cache.copy_(self.initial_cache)
        if self.scale is not None:
            self.scale.copy_(self.initial_scale)
        assert self.addresses == [(t.data_ptr(), tuple(t.shape), t.dtype) for t in self.tensors]

    def paged_cache(self):
        return self.cache.view(self.pages, self.page_size, 1, self.head_dim)

    def paged_scale(self):
        return self.scale.view(self.pages, self.page_size, 1, 1).squeeze(-2)


class CompressorArm:
    def __init__(self, torch, torch_npu, op, x, tokens, hidden, head_dim, device, real_pages, page_size, seed):
        self.torch, self.op, self.x = torch, op, x
        self.tokens, self.head_dim, self.device = tokens, head_dim, device
        self.make_inputs, self.table_builder = common.builtin_fixture_builders(torch, False)
        self.base = self.make_inputs([0, 0], tokens, 2, 4, head_dim, hidden, 2, "TH", torch.bfloat16, 2, 16)
        self.initial_state = self.base["state_cache"].to(device)
        self.state = self.initial_state.clone()
        self.weights = {name: self.base[name].to(device) for name in ("wkv", "wgate", "ape", "norm_weight")}
        self.buffers = None
        self.epilog = PagedEpilog(torch, torch_npu, tokens // 4 + 2, head_dim,
                                 head_dim == 128, device, real_pages, page_size, seed)

    def load(self, layout):
        payload = common.make_payload(self.torch, layout.case, self.base, self.table_builder,
                                      self.tokens, 4, 2, False)
        payload.pop("x")
        if self.buffers is None:
            self.buffers = {name: value.to(self.device) for name, value in payload.items()}
        else:
            for name, value in payload.items():
                assert tuple(value.shape) == tuple(self.buffers[name].shape), name
                self.buffers[name].copy_(value)
        self.state.copy_(self.initial_state)
        self.epilog.reset_layout(layout)

    def compute(self, eager_valid=None):
        w, b = self.weights, self.buffers
        out = self.op(self.x, w["wkv"], w["wgate"], self.state, w["ape"], w["norm_weight"],
                      rope_sin=b["rope_sin"], rope_cos=b["rope_cos"], state_block_table=b["state_block_table"],
                      cu_seqlens=b["cu_seqlens"], seqused=b["seqused"], start_pos=b["start_pos"],
                      rope_head_dim=64, cmp_ratio=4, coff=2, norm_eps=1e-6, rotary_mode=2, cache_mode=2)
        if eager_valid is None:
            self.epilog.apply_in_graph(out)
        else:
            self.epilog.apply_eager_reference(out, eager_valid)


class C4Chain:
    def __init__(self, torch, torch_npu, options, device, layouts):
        self.torch, self.torch_npu, self.options = torch, torch_npu, options
        self.device, self.tokens, self.q_capacity = device, options.tokens, options.tokens + 2
        self.max_pages = math.ceil((options.prefix + options.tokens + 256) / options.page_size)
        self.real_pages = 1 + 2 * self.max_pages
        self.real_swa_slots = self.real_pages * options.page_size
        self.swa_pages = self.real_pages + math.ceil(self.q_capacity / options.page_size)
        self.x = torch.empty(options.tokens, options.hidden, dtype=torch.bfloat16, device=device)
        compressor = torch.ops.npu.compressor.default
        self.main = CompressorArm(torch, torch_npu, compressor, self.x, options.tokens, options.hidden,
                                  512, device, self.real_pages, options.page_size // 4, 201)
        self.index = CompressorArm(torch, torch_npu, compressor, self.x, options.tokens, options.hidden,
                                   128, device, self.real_pages, options.page_size // 4, 202)
        generator = torch.Generator().manual_seed(203)
        swa_initial = (torch.randn(self.swa_pages * options.page_size, 1, 512, generator=generator) * 0.1).to(torch.bfloat16)
        bias = ((torch.arange(self.swa_pages * options.page_size) // options.page_size) % 7 - 3).float() * 0.2
        swa_initial = (swa_initial.float() + bias[:, None, None]).to(torch.bfloat16)
        swa_initial[:options.page_size].zero_()
        self.swa_initial = swa_initial.to(device)
        self.swa = self.swa_initial.clone()
        # Synthetic rank-128 projections keep the probe small; no checkpoint is loaded.
        self.wq = (torch.randn(128, options.heads * 512, generator=generator) * 0.02).to(device=device, dtype=torch.bfloat16)
        self.wkv = (torch.randn(128, 512, generator=generator) * 0.02).to(device=device, dtype=torch.bfloat16)
        self.wiq = (torch.randn(128, 64 * 128, generator=generator) * 0.02).to(device=device, dtype=torch.bfloat16)
        self.ww = (torch.randn(128, 64, generator=generator) * 0.02).to(device=device, dtype=torch.bfloat16)
        self.norm = torch.ones(128, dtype=torch.bfloat16, device=device)
        self.kv_norm = torch.ones(512, dtype=torch.bfloat16, device=device)
        self.sinks = torch.full((options.heads,), -1e4, dtype=torch.float32, device=device)
        self.topk = torch.empty(self.q_capacity, 1, options.topk, dtype=torch.int32, device=device)
        self.li_metadata = torch.empty(1024, dtype=torch.int32, device=device)
        self.attn_metadata = torch.empty(1024, dtype=torch.int32, device=device)
        self.dynamic = None
        self.layouts = layouts
        self.load(layouts[0])
        self.addresses = self.signature()

    def signature(self):
        named = {**self.dynamic, "x": self.x, "swa": self.swa, "topk": self.topk,
                 "li_metadata": self.li_metadata, "attn_metadata": self.attn_metadata}
        for name, arm in (("main", self.main), ("index", self.index)):
            named[name + ".state"] = arm.state
            named[name + ".cache"] = arm.epilog.cache
            named[name + ".loc"] = arm.epilog.loc
            named[name + ".valid"] = arm.epilog.valid
            named.update({name + "." + key: value for key, value in arm.buffers.items()})
        named["index.scale"] = self.index.epilog.scale
        return {name: (value.data_ptr(), tuple(value.shape), value.dtype) for name, value in named.items()}

    def load(self, layout):
        torch = self.torch
        o = self.options
        self.x.copy_((torch.randn(o.tokens, o.hidden, generator=torch.Generator().manual_seed(layout.case.seed)) * 0.02).to(torch.bfloat16))
        self.main.load(layout)
        self.index.load(layout)
        self.swa.copy_(self.swa_initial)
        self.topk.fill_(-1)
        frequencies = 10000.0 ** (-torch.arange(0, 64, 2).float() / 64)
        angles = (torch.tensor(layout.q_positions).float()[:, None] * frequencies).repeat_interleave(2, dim=1)
        integers = {"q_cu": layout.q_cu, "q_ends": layout.q_cu[1:], "kv_lens": layout.kv_lens,
                    "swa_table": layout.swa_table, "cmp_table": layout.cmp_table}
        indices = {"q_gather": layout.q_gather, "restore": layout.restore, "swa_loc": layout.swa_loc}
        values = {**{key: torch.tensor(value, dtype=torch.int32) for key, value in integers.items()},
                  **{key: torch.tensor(value, dtype=torch.int64) for key, value in indices.items()},
                  "q_valid": torch.tensor(layout.q_valid, dtype=torch.bool),
                  "input_valid": torch.tensor(layout.input_valid, dtype=torch.bool),
                  "cos": angles.cos().to(torch.bfloat16), "sin": angles.sin().to(torch.bfloat16)}
        if self.dynamic is None:
            self.dynamic = {key: value.to(self.device) for key, value in values.items()}
        else:
            for key, value in values.items():
                assert tuple(value.shape) == tuple(self.dynamic[key].shape), key
                self.dynamic[key].copy_(value)
        d = self.dynamic
        new_li = torch.ops.custom.npu_quant_lightning_indexer_metadata(
            num_heads_q=64, num_heads_k=1, head_dim=128, query_quant_mode=0, key_quant_mode=0,
            actual_seq_lengths_query=d["q_ends"], actual_seq_lengths_key=d["kv_lens"],
            layout_query="TND", layout_key="PA_BSND", sparse_count=o.topk, sparse_mode=3,
            cmp_ratio=4, device=self.device)
        new_attn = torch.ops.custom.npu_sparse_attn_sharedkv_metadata(
            num_heads_q=o.heads, num_heads_kv=1, head_dim=512, batch_size=2,
            cu_seqlens_q=d["q_cu"], seqused_kv=d["kv_lens"], cmp_topk=o.topk, cmp_ratio=4,
            ori_mask_mode=4, cmp_mask_mode=3, ori_win_left=127, ori_win_right=0,
            layout_q="TND", layout_kv="PA_ND", has_ori_kv=True, has_cmp_kv=True, device=self.device)
        assert new_li.numel() == new_attn.numel() == 1024
        self.li_metadata.copy_(new_li.reshape(-1))
        self.attn_metadata.copy_(new_attn.reshape(-1))
        if hasattr(self, "addresses"):
            assert self.addresses == self.signature()

    def rope(self, value):
        torch, d = self.torch, self.dynamic
        rope = value[..., -64:]
        pairs = rope.reshape(*rope.shape[:-1], 32, 2)
        rotated = torch.stack((-pairs[..., 1], pairs[..., 0]), dim=-1).flatten(-2)
        cosine, sine = d["cos"], d["sin"]
        if value.dim() == 3:
            cosine, sine = cosine[:, None, :], sine[:, None, :]
        return torch.cat((value[..., :-64], rope * cosine + rotated * sine), dim=-1)

    def step(self, eager_layout=None):
        torch, npu, d, o = self.torch, self.torch_npu, self.dynamic, self.options
        valid_compressed = eager_layout.case.compressed_rows(4) if eager_layout is not None else None
        self.main.compute(valid_compressed)
        self.index.compute(valid_compressed)
        gathered = self.x.index_select(0, d["q_gather"])[:, :128].contiguous()
        source = torch.where(d["q_valid"][:, None], gathered, torch.zeros_like(gathered))
        normalized = npu.npu_rms_norm(source, self.norm)[0]
        q = self.rope(normalized.matmul(self.wq).view(self.q_capacity, o.heads, 512))
        kv = self.rope(npu.npu_rms_norm(source.matmul(self.wkv), self.kv_norm)[0])
        if eager_layout is None:
            self.swa[d["swa_loc"]] = kv[:, None, :]
        elif eager_layout.valid_q_rows:
            rows = torch.tensor(eager_layout.valid_q_rows, dtype=torch.int64, device=self.device)
            self.swa[d["swa_loc"].index_select(0, rows)] = kv.index_select(0, rows)[:, None, :]
        iq = self.rope(normalized.matmul(self.wiq).view(self.q_capacity, 64, 128))
        iq = iq.reshape(-1, 128).matmul(self.index.epilog.hadamard).view(self.q_capacity, 64, 128)
        quantized_q, q_scale = npu.npu_dynamic_quant(iq)
        q_scale = q_scale.to(torch.float16)
        weights = (source.matmul(self.ww) * (128 ** -0.5 * 64 ** -0.5)).to(torch.float16)
        topk, _ = torch.ops.custom.npu_quant_lightning_indexer(
            query=quantized_q, key=self.index.epilog.paged_cache(), weights=weights,
            query_dequant_scale=q_scale, key_dequant_scale=self.index.epilog.paged_scale(),
            query_quant_mode=0, key_quant_mode=0, actual_seq_lengths_query=d["q_ends"],
            actual_seq_lengths_key=d["kv_lens"], block_table=d["cmp_table"], metadata=self.li_metadata,
            layout_query="TND", layout_key="PA_BSND", sparse_count=o.topk, sparse_mode=3, cmp_ratio=4)
        topk = topk.view(self.q_capacity, 1, o.topk)
        self.topk.copy_(torch.where(d["q_valid"][:, None, None], topk, torch.full_like(topk, -1)))
        attention, _ = torch.ops.custom.npu_sparse_attn_sharedkv(
            q, ori_kv=self.swa.view(self.swa_pages, o.page_size, 1, 512),
            cmp_kv=self.main.epilog.paged_cache(), cmp_sparse_indices=self.topk,
            ori_block_table=d["swa_table"], cmp_block_table=d["cmp_table"],
            cu_seqlens_q=d["q_cu"], seqused_kv=d["kv_lens"], sinks=self.sinks,
            metadata=self.attn_metadata, softmax_scale=512 ** -0.5, cmp_ratio=4,
            ori_mask_mode=4, cmp_mask_mode=3, ori_win_left=127, ori_win_right=0,
            layout_q="TND", layout_kv="PA_ND")
        restored = attention.index_select(0, d["restore"])
        return torch.where(d["input_valid"][:, None, None], restored, torch.zeros_like(restored))

    def snapshot(self, output):
        return {"attention": output.cpu(), "topk": self.topk.cpu(),
                "main_state": self.main.state.cpu(), "index_state": self.index.state.cpu(),
                "main_cache": self.main.epilog.cache[:self.main.epilog.real_slots].cpu(),
                "index_cache": self.index.epilog.cache[:self.index.epilog.real_slots].cpu(),
                "index_scale": self.index.epilog.scale[:self.index.epilog.real_slots].cpu(),
                "swa_cache": self.swa[:self.real_swa_slots].cpu()}

    def check(self, output, expected):
        torch, o = self.torch, self.options
        actual = self.snapshot(output)
        checks, errors = {}, {}
        for key in actual:
            if key == "topk":
                continue
            exact = key in ("main_cache", "index_cache", "swa_cache")
            atol = 0 if exact else o.atol if key == "attention" else o.state_atol
            ok, error = common.compare(torch, actual[key], expected[key], atol=atol, rtol=0 if exact else o.rtol)
            checks[key + "_ok"], errors[key + "_max_abs_error"] = ok, error if math.isfinite(error) else str(error)
        checks["topk_set_ok"] = torch.equal(actual["topk"].sort(dim=-1).values, expected["topk"].sort(dim=-1).values)
        exact_topk = torch.equal(actual["topk"], expected["topk"])
        invalid = ~self.dynamic["input_valid"].cpu()
        checks["padding_output_zero"] = not bool(actual["attention"][invalid].count_nonzero().item())
        scratch = (self.swa[self.real_swa_slots:], self.main.epilog.cache[self.main.epilog.real_slots:],
                   self.index.epilog.cache[self.index.epilog.real_slots:], self.index.epilog.scale[self.index.epilog.real_slots:])
        checks["scratch_finite"] = all(bool(torch.isfinite(t.float()).all().item()) for t in scratch)
        return {**checks, **errors, "topk_exact_ok": exact_topk, "passed": all(checks.values())}


def run(options, report):
    import torch

    report["stage"] = "registration"
    imports = []
    for name in ("custom_ops", "sgl_kernel_npu"):
        module = importlib.import_module(name)
        imports.append({"name": name, "file": str(getattr(module, "__file__", "unknown"))})
    import torch_npu

    torch.npu.set_device(options.device)
    torch_npu.npu.set_compile_mode(jit_compile=False)
    device = f"npu:{options.device}"
    device_name = torch.npu.get_device_name(options.device)
    if "950" in device_name.lower():
        raise ValueError("This probe uses the A3 int8/BF16 cache ABI")
    schemas = common.registered_attention_schemas(torch)
    missing = [name for name in (*common.A3_ATTENTION_OPS, "npu.compressor") if schemas[name] is None]
    if missing:
        raise RuntimeError(f"Required operators are missing: {missing}")
    report["environment"] = {"torch": torch.__version__, "torch_npu": torch_npu.__version__,
                             "device": device_name, "imports": imports, "schemas": schemas,
                             "capture_api": options.capture_api, "pipeline": "a3_c4_attention_chain",
                             "x_capacity": options.tokens, "query_capacity": options.tokens + 2,
                             "request_slots": 2, "prefix": options.prefix, "topk": options.topk,
                             "hidden": options.hidden, "heads": options.heads,
                             "metadata_planning": "outside_graph_in_place_refresh",
                             "oracle_schedule": "all_references_before_capture",
                             "dummy_queries": "one_per_inactive_request"}
    print(json.dumps({key: value for key, value in report["environment"].items() if key != "schemas"}), flush=True)
    max_pages = math.ceil((options.prefix + options.tokens + 256) / options.page_size)
    cases = [common.Case(c.name, c.capacities, c.used, tuple(options.prefix + p for p in c.starts), c.seed)
             for c in common.make_cases(options.tokens, 4)]
    layouts = [build_layout(case, options.tokens, options.page_size, max_pages) for case in cases]
    chain = C4Chain(torch, torch_npu, options, device, layouts)
    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    graph, pool = torch.npu.NPUGraph(), torch.npu.graph_pool_handle()
    with torch.npu.stream(stream):
        references = []
        for layout in layouts:
            report["stage"] = "eager_reference:" + layout.case.name
            common.write_report(options.report, report)
            chain.load(layout)
            eager = chain.step(eager_layout=layout)
            torch.npu.synchronize()
            references.append(chain.snapshot(eager))
            report["reference_cases"] = len(references)
            print(json.dumps({"reference": layout.case.name, "computed": True}), flush=True)
        report["stage"] = "warmup"
        common.write_report(options.report, report)
        for _ in range(2):
            chain.load(layouts[0])
            chain.step()
        chain.load(layouts[0])
        torch.npu.synchronize()
        report["stage"] = "capture"
        common.write_report(options.report, report)
        if options.capture_api == "raw":
            graph.capture_begin(pool=pool, capture_error_mode="global")
            output = chain.step()
            graph.capture_end()
        else:
            with torch.npu.graph(graph, pool=pool, stream=stream, auto_dispatch_capture=False):
                output = chain.step()
        torch.npu.synchronize()
        for layout, expected in zip(layouts, references):
            report["stage"] = "replay:" + layout.case.name
            common.write_report(options.report, report)
            chain.load(layout)
            graph.replay()
            torch.npu.synchronize()
            result = {**asdict(layout.case), "q_cu": layout.q_cu, "kv_lens": layout.kv_lens,
                      "real_queries": sum(layout.case.used), "compressed_rows": layout.case.compressed_rows(4),
                      **chain.check(output, expected)}
            report["cases"].append(result)
            print(json.dumps(result), flush=True)
            common.write_report(options.report, report)
    report["stage"] = "complete"
    report["passed"] = all(case["passed"] for case in report["cases"])
    return 0 if report["passed"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tokens", type=int, default=256)
    parser.add_argument("--hidden", type=int, default=1024)
    parser.add_argument("--prefix", type=int, default=2048)
    parser.add_argument("--page-size", type=int, choices=(64, 128), default=128)
    parser.add_argument("--heads", type=int, choices=(64, 128), default=64)
    parser.add_argument("--topk", type=int, choices=(512, 1024), default=512)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--capture-api", choices=("raw", "context"), default="raw")
    parser.add_argument("--atol", type=float, default=0.001)
    parser.add_argument("--state-atol", type=float, default=0.001)
    parser.add_argument("--rtol", type=float, default=0.01)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--list-cases", action="store_true")
    options = parser.parse_args()
    if options.tokens < 256 or options.tokens % 128 or options.hidden < 1024 or options.hidden % 512:
        parser.error("--tokens must be >=256 and 128-aligned; --hidden must be >=1024 and 512-aligned")
    if options.prefix < 128 or options.prefix % 128:
        parser.error("--prefix must be >=128 and 128-aligned (this probe uses existing history)")
    if options.list_cases:
        pages = math.ceil((options.prefix + options.tokens + 256) / options.page_size)
        layouts = [build_layout(common.Case(c.name, c.capacities, c.used, tuple(options.prefix + p for p in c.starts), c.seed),
                                options.tokens, options.page_size, pages) for c in common.make_cases(options.tokens, 4)]
        print(json.dumps([{**asdict(p.case), "q_cu": p.q_cu, "kv_lens": p.kv_lens,
                           "valid_queries": sum(p.q_valid), "valid_comp_rows": sum(p.comp_valid)} for p in layouts], indent=2))
        return 0
    report = {"passed": False, "stage": "initialization", "cases": [],
              "tolerances": {"attention_atol": options.atol, "state_scale_atol": options.state_atol,
                             "rtol": options.rtol, "cache_exact": True, "topk_set_exact": True}}
    try:
        result = run(options, report)
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        traceback.print_exc()
        result = 2
    common.write_report(options.report, report)
    print(json.dumps({"passed": report["passed"], "stage": report["stage"], "cases": len(report["cases"])}), flush=True)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
