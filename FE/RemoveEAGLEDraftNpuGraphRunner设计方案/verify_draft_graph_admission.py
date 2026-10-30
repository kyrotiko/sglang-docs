"""Exercise graph admission fragments from the adjacent SGLang checkout.

Only the Python standard library is required. Functions are extracted with AST
so that importing SGLang, torch, accelerator libraries, and model weights is not
necessary. The candidate removes only the final seed/MIN block from the real
can_run_graph implementation. The worker's subsequent seed guard and the
scheduler's compatibility predicate are also extracted from production source.

The in-process MIN simulator tests Boolean decisions and participation only.
It does not test torch.distributed, graph execution, or accelerator deadlocks.

Run:
    python verify_draft_graph_admission.py --sglang-dir /path/to/sglang -v
"""

from __future__ import annotations

import argparse
import ast
import copy
import itertools
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace as NS


RUNNER = "python/sglang/srt/speculative/eagle_draft_cuda_graph_runner.py"
WORKER = "python/sglang/srt/speculative/eagle_worker_v2.py"
DP_ATTN = "python/sglang/srt/managers/scheduler_components/dp_attn.py"
INFO = "python/sglang/srt/speculative/eagle_info.py"


def extract(repo, filename, function, class_name=None, *, variant="original"):
    tree = ast.parse((repo / filename).read_text(encoding="utf-8"))
    scope = tree
    if class_name is not None:
        scope = next(
            node for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == class_name
        )
    node = copy.deepcopy(next(
        node for node in scope.body
        if isinstance(node, ast.FunctionDef) and node.name == function
    ))
    if variant == "without_final_vote":
        cuts = [
            i for i, stmt in enumerate(node.body)
            if isinstance(stmt, ast.If)
            and "self.eagle_worker.seed_dsa_topk_from_draft_extend"
            in ast.unparse(stmt.test)
        ]
        assert len(cuts) == 1, "Review the changed production function before using this spike"
        assert any(
            isinstance(n, ast.Call) and ast.unparse(n.func) == "torch.distributed.all_reduce"
            for stmt in node.body[cuts[0]:] for n in ast.walk(stmt)
        )
        node.body = node.body[:cuts[0]] + [
            ast.Return(value=ast.Name(id="is_bs_supported", ctx=ast.Load()))
        ]
    elif variant == "worker_seed_guard":
        cuts = [
            i for i, stmt in enumerate(node.body)
            if isinstance(stmt, ast.If)
            and "self.seed_dsa_topk_from_draft_extend" in ast.unparse(stmt.test)
            and "draft_input.dsa_topk_indices is None" in ast.unparse(stmt.test)
        ]
        assert len(cuts) == 1
        node.body = node.body[:cuts[0] + 1] + [
            ast.Return(value=ast.Name(id="can_run_decode_cuda_graph", ctx=ast.Load()))
        ]
    module = ast.Module(body=[
        ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
        node,
    ], type_ignores=[])
    return compile(ast.fix_missing_locations(module), str(repo / filename), "exec")


class Scalar:
    def __init__(self, value):
        self.value = value

    def item(self):
        return self.value


class MinGroup:
    def __init__(self, participants=2):
        self.votes = []
        self.lock = threading.Lock()
        self.barrier = threading.Barrier(participants)

    def reduce(self, tensor, *, op, group):
        assert group is self and op == "MIN"
        with self.lock:
            self.votes.append(tensor.value)
        # A short timeout converts missing participation into a test result.
        self.barrier.wait(timeout=2)
        tensor.value = min(self.votes)


class AdmissionAnalysis(unittest.TestCase):
    repo: Path

    @classmethod
    def setUpClass(cls):
        cls.original_code = extract(cls.repo, RUNNER, "can_run_graph", "EAGLEDraftCudaGraphRunner")
        cls.candidate_code = extract(
            cls.repo, RUNNER, "can_run_graph", "EAGLEDraftCudaGraphRunner",
            variant="without_final_vote",
        )
        cls.worker_code = extract(
            cls.repo, WORKER, "draft", "EagleDraftWorker", variant="worker_seed_guard"
        )
        cls.scheduler_code = extract(cls.repo, DP_ATTN, "_spec_input_cuda_graph_compatible")
        cls.merge_code = extract(cls.repo, INFO, "merge_batch", "EagleDraftInput")

    def load(self, code, function, env=None):
        namespace = {} if env is None else dict(env)
        exec(code, namespace)
        return namespace[function]

    @staticmethod
    def input(state, bs=None, *, compatible=None, width=1):
        idle = state == "idle"
        has_seed = state == "seeded"
        mode = NS(is_idle=lambda: idle, is_prebuilt=lambda: False)
        return NS(
            forward_mode=mode,
            batch_size=(0 if idle else 1) if bs is None else bs,
            spec_info=NS(
                num_tokens_per_req=width,
                dsa_topk_indices=object() if has_seed else None,
                cuda_graph_compatible=(idle or has_seed) if compatible is None else compatible,
            ),
            can_run_decode_cuda_graph=True,
        )

    def run_pair(
        self, inputs, *, candidate=False, gathered=True, no_padding=False,
        seed_enabled=True, dp_size=2,
    ):
        scheduler_vote = self.load(self.scheduler_code, "_spec_input_cuda_graph_compatible")
        # MLPSyncBatchInfo uses min on compatibility column 8 after exchange.
        # Here only this reduction is simulated; the predicate is real source.
        global_compatible = all(scheduler_vote(batch) for batch in inputs)
        for batch in inputs:
            batch.can_run_dp_draft_cuda_graph = global_compatible
            batch.original_global_num_tokens_cpu = [b.batch_size for b in inputs]
        group = MinGroup()
        torch_stub = NS(
            tensor=lambda value, **kwargs: Scalar(value),
            int32="int32",
            distributed=NS(all_reduce=group.reduce, ReduceOp=NS(MIN="MIN")),
        )
        function = self.load(
            self.candidate_code if candidate else self.original_code,
            "can_run_graph",
            {"torch": torch_stub, "get_parallel": lambda: NS(tp_group=NS(cpu_group=group))},
        )

        def execute(batch):
            runner = NS(
                captured_req_width=1,
                require_mlp_tp_gather=gathered,
                require_mlp_sync=True,
                disable_padding=no_padding,
                max_bs=8,
                eagle_worker=NS(seed_dsa_topk_from_draft_extend=seed_enabled),
                attn_dp_size=dp_size,
                backend=NS(can_run=lambda _batch, key: key in range(1, 9)),
                _make_graph_key=lambda bs, **kwargs: bs,
                _sampling_variant=lambda _batch: None,
                _resolve_attention_variant=lambda _batch, bs: None,
            )
            try:
                admitted = function(runner, batch)
                # Execute the real draft() prefix through the worker seed guard.
                worker = NS(
                    req_to_token_pool=None, cuda_graph_runner=runner, draft_runner=None,
                    topk=1, speculative_num_steps=3, speculative_num_draft_tokens=4,
                    seed_dsa_topk_from_draft_extend=seed_enabled,
                    target_worker=NS(model_runner=NS(kv_index_translator=None)),
                )
                worker_choice = self.load(
                    self.worker_code, "draft",
                    {"prepare_for_draft": lambda *args, **kwargs: (batch, admitted, None)},
                )
                return bool(worker_choice(worker, NS(spec_info=batch.spec_info)))
            except threading.BrokenBarrierError:
                return "missing_vote_participant"

        with ThreadPoolExecutor(max_workers=2) as executor:
            outputs = list(executor.map(execute, inputs))
        return outputs, group.votes

    def test_01_current_seed_protocol_with_global_batch_size(self):
        # Covers active missing/present seeds and idle ranks, both below and
        # above the graph limit. Ordinary producers keep compatible in sync.
        for states in itertools.product(("idle", "seeded", "seedless"), repeat=2):
            for active_bs in (1, 9):
                with self.subTest(states=states, active_bs=active_bs):
                    outputs = []
                    for candidate in (False, True):
                        inputs = [self.input(s, 0 if s == "idle" else active_bs) for s in states]
                        outputs.append(self.run_pair(inputs, candidate=candidate)[0])
                    self.assertEqual(outputs[0], outputs[1])
                    self.assertEqual(outputs[0][0], outputs[0][1])

    def test_02_historical_missing_seed_without_scheduler_protocol(self):
        for candidate, expected in ((False, [False, False]), (True, [False, True])):
            inputs = [self.input("seedless", compatible=True), self.input("idle")]
            self.assertEqual(self.run_pair(inputs, candidate=candidate)[0], expected)

    def test_03_local_batch_limit_is_not_in_seed_vote(self):
        for candidate, expected in ((False, [False, False]), (True, [False, True])):
            inputs = [self.input("seeded", 9), self.input("seeded", 1)]
            self.assertEqual(self.run_pair(inputs, candidate=candidate, gathered=False)[0], expected)

    def test_04_no_padding_idle_has_no_zero_size_graph(self):
        for candidate, expected in ((False, [False, False]), (True, [True, False])):
            inputs = [self.input("seeded", 1), self.input("idle")]
            self.assertEqual(
                self.run_pair(inputs, candidate=candidate, gathered=False, no_padding=True)[0],
                expected,
            )

    def test_05_global_batch_limit_is_uniform(self):
        for candidate in (False, True):
            inputs = [self.input("seeded", 9), self.input("seeded", 1)]
            self.assertEqual(self.run_pair(inputs, candidate=candidate)[0], [False, False])

    def test_06_future_merge_retains_compatibility_even_with_stale_tensor(self):
        merge = self.load(self.merge_code, "merge_batch", {"torch": NS(cat=lambda parts: sum(parts, []))})
        seeded = NS(
            future_indices=[0], future_dsa_topk_indices_available=True,
            cuda_graph_compatible=True, dsa_topk_indices=object(),
        )
        seedless = NS(
            future_indices=[1], future_dsa_topk_indices_available=False,
            cuda_graph_compatible=False, dsa_topk_indices=None,
        )
        merge(seeded, seedless)
        self.assertIsNotNone(seeded.dsa_topk_indices)
        self.assertFalse(seeded.future_dsa_topk_indices_available)
        self.assertFalse(seeded.cuda_graph_compatible)
        vote = self.load(self.scheduler_code, "_spec_input_cuda_graph_compatible")
        self.assertFalse(vote(NS(
            spec_info=seeded,
            forward_mode=NS(is_idle=lambda: False, is_prebuilt=lambda: False),
        )))

    def test_07_config_gates_skip_the_extra_collective(self):
        for enabled, dp_size in ((False, 2), (True, 1)):
            outputs, votes = self.run_pair(
                [self.input("seeded"), self.input("seeded")],
                seed_enabled=enabled, dp_size=dp_size,
            )
            self.assertEqual(outputs, [True, True])
            self.assertEqual(votes, [])

    def test_08_nonuniform_width_can_skip_collective_participation(self):
        outputs, votes = self.run_pair([
            self.input("seeded", width=2), self.input("seeded", width=1)
        ])
        self.assertEqual(outputs, [False, "missing_vote_participant"])
        self.assertEqual(len(votes), 1)

    def test_09_new_vote_reads_worker_even_when_scheduler_already_vetoed(self):
        # Matches the minimum fields in the existing __new__ unit fixture.
        runner = NS(
            require_mlp_tp_gather=False, require_mlp_sync=True,
            disable_padding=False, captured_req_width=1, max_bs=8,
        )
        batch = NS(
            spec_info=NS(num_tokens_per_req=1), batch_size=1,
            can_run_decode_cuda_graph=True, can_run_dp_draft_cuda_graph=False,
        )
        original = self.load(self.original_code, "can_run_graph")
        with self.assertRaisesRegex(AttributeError, "eagle_worker"):
            original(runner, batch)
        candidate = self.load(self.candidate_code, "can_run_graph")
        self.assertFalse(candidate(runner, batch))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sglang-dir", type=Path,
        default=Path(__file__).resolve().parents[3] / "sglang",
    )
    args, unittest_args = parser.parse_known_args()
    AdmissionAnalysis.repo = args.sglang_dir.resolve()
    if not (AdmissionAnalysis.repo / RUNNER).is_file():
        parser.error("--sglang-dir must point to the SGLang repository")
    unittest.main(argv=[__file__, *unittest_args])
