#!/usr/bin/env python3
"""A5 end-to-end probe for DeepSeek-V4.1 HiCache L2.

What it does:
  1. Starts a baseline server with --disable-radix-cache and records a cold
     greedy output for one long prompt A.
  2. Starts a candidate server with HiCache L2 (explicit --hicache-ratio),
     sends A, fills device with unique prefixes, then sends A again.
  3. Records startup pool geometry, host-hit metrics, cached_tokens, output
     equality, and server logs.
  4. Optionally starts a server with only --enable-hierarchical-cache to
     test the default-ratio auto-size path (G10).

It does not modify the repository.  All output goes to --out-dir.

Example:
    python a5_hicache_e2e_probe.py \
        --model-path /models/DeepSeek-V4.1-Flash \
        --tp-size 8 \
        --out-dir probe_out

Run with --g10 to also test the default-ratio startup path.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import time
import traceback
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path


def log(msg: str) -> None:
    print(msg, flush=True)


def http_text(url: str, timeout: float = 10.0) -> str:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="replace")


def http_json(url: str, payload: dict | None = None, timeout: float = 60.0):
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8", errors="replace"))


def find_key(obj, key: str):
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for value in obj.values():
            found = find_key(value, key)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for value in obj:
            found = find_key(value, key)
            if found is not None:
                return found
    return None


class Server:
    def __init__(self, *, name: str, port: int, log_path: Path):
        self.name = name
        self.port = port
        self.log_path = log_path
        self.proc: subprocess.Popen | None = None

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def start(self, cmd: list[str], env_updates: dict | None = None) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env.update(env_updates or {})
        env["PYTHONUNBUFFERED"] = "1"
        log(f"[{self.name}] starting: {' '.join(shlex.quote(x) for x in cmd)}")
        log(f"[{self.name}] PYTHONPATH={env.get('PYTHONPATH', '')}")
        self._log_file = self.log_path.open("w", encoding="utf-8", errors="replace")
        self.proc = subprocess.Popen(
            cmd,
            stdout=self._log_file,
            stderr=subprocess.STDOUT,
            env=env,
            text=True,
        )

    def wait_ready(self, timeout_s: float) -> bool:
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if self.proc is not None and self.proc.poll() is not None:
                log(f"[{self.name}] exited with code {self.proc.returncode}")
                return False
            try:
                # /health is the cheapest endpoint; fall back to model info.
                try:
                    text = http_text(f"{self.base_url}/health", timeout=2.0)
                    if text:
                        return True
                except urllib.error.HTTPError as exc:
                    if exc.code != 404:
                        raise
                info = http_json(f"{self.base_url}/get_model_info", timeout=3.0)
                if info:
                    return True
            except Exception:
                time.sleep(2.0)
        log(f"[{self.name}] startup timed out after {timeout_s:.0f}s")
        return False

    def server_info(self) -> dict:
        for path in ("/get_server_info", "/get_model_info"):
            try:
                value = http_json(f"{self.base_url}{path}", timeout=10.0)
                if isinstance(value, dict):
                    return value
            except Exception:
                continue
        return {}

    def generate(self, prompt: str, max_new_tokens: int, timeout_s: float) -> dict:
        payload = {
            "text": prompt,
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": max_new_tokens,
                "ignore_eos": True,
            },
        }
        value = http_json(
            f"{self.base_url}/generate", payload=payload, timeout=timeout_s
        )
        if isinstance(value, list):
            value = value[0] if value else {}
        return value if isinstance(value, dict) else {"raw": value}

    def metrics_text(self) -> str:
        try:
            return http_text(f"{self.base_url}/metrics", timeout=10.0)
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                log(
                    f"[{self.name}] /metrics is disabled; launch with "
                    "--enable-metrics to check host_hit"
                )
                return ""
            raise

    def tail_log(self, n: int = 80) -> list[str]:
        try:
            lines = self.log_path.read_text(
                encoding="utf-8", errors="replace"
            ).splitlines()
            return lines[-n:]
        except OSError:
            return []

    def stop(self) -> None:
        if self.proc is None:
            return
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=10)
        try:
            self._log_file.close()
        except Exception:
            pass


METRIC_RE = re.compile(
    r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})?\s+([0-9.eE+\-]+)$"
)


def metric_values(text: str, name: str) -> list[tuple[str, float]]:
    values = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        match = METRIC_RE.match(line)
        if not match:
            continue
        metric, labels, raw = match.groups()
        if metric != name:
            continue
        try:
            values.append((labels or "", float(raw)))
        except ValueError:
            continue
    return values


def metric_sum(text: str, name: str, label_substr: str | None = None) -> float:
    values = metric_values(text, name)
    if label_substr is not None:
        values = [(labels, value) for labels, value in values
                  if label_substr in labels]
    return sum(value for _, value in values)


def metric_last(text: str, name: str) -> float | None:
    values = metric_values(text, name)
    return values[-1][1] if values else None


POOL_RE = re.compile(
    r"Allocating .* host memory for V4 paged pool '([^']+)' "
    r"\(layers=(\d+), pages=(\d+), item_bytes=(\d+), layout=([^)]+)\)"
)


def parse_pools(log_path: Path) -> dict:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    pools = {}
    for match in POOL_RE.finditer(text):
        name, layers, pages, item_bytes, layout = match.groups()
        pools[name] = {
            "layers": int(layers),
            "pages": int(pages),
            "item_bytes": int(item_bytes),
            "layout": layout,
        }
    return pools


def expected_v41_item_bytes(page_size: int) -> dict:
    # V4.1 row widths: SWA 544 B/token; C1/C2 compressed 320 B/slot;
    # indexer K 64 B/slot; scale 4 B/slot.
    return {
        "swa": page_size * 544,
        "deepseek_v4_c1": page_size * 320,
        "deepseek_v4_c2": (page_size // 2) * 320,
        "deepseek_v4_c1_indexer": page_size * 64,
        "deepseek_v4_c1_indexer_scale": page_size * 4,
        "deepseek_v4_c2_indexer": (page_size // 2) * 64,
        "deepseek_v4_c2_indexer_scale": (page_size // 2) * 4,
    }


def geometry_checks(pools: dict, page_size: int) -> list[dict]:
    expected = expected_v41_item_bytes(page_size)
    checks = []
    for name, want in expected.items():
        actual = pools.get(name)
        checks.append(
            {
                "pool": name,
                "expected_item_bytes": want,
                "actual_item_bytes": None if actual is None else actual["item_bytes"],
                "ok": actual is not None and actual["item_bytes"] == want,
            }
        )
    return checks


def build_prompts(page_size: int, a_pages: int, filler_pages: int):
    base = (
        "This is a long technical document used to exercise the KV prefix "
        "cache. The same paragraph is repeated to reach a predictable length. "
        "It contains no instructions other than the final question. "
    )
    # Character counts are approximate; the server performs the tokenization.
    a_chars = page_size * a_pages * 4
    filler_chars = page_size * filler_pages * 4

    def body(tag: str, chars: int) -> str:
        head = f"[{tag}] "
        repeats = max(1, (chars - len(head)) // len(base) + 1)
        return head + (base * repeats)[: max(0, chars - len(head))]

    prompt_a = body("DOC-A", a_chars) + "\n\nReply with the single word OK."
    return prompt_a, filler_chars, body


def server_cmd(
    *, args, port: int, extra: list[str], log_name: str
) -> tuple[list[str], Path]:
    log_path = args.out_dir / f"{log_name}.log"
    cmd = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        args.model_path,
        "--device",
        "npu",
        "--page-size",
        str(args.page_size),
        "--tp-size",
        str(args.tp_size),
        "--trust-remote-code",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
    ]
    if args.mem_fraction_static is not None:
        cmd += ["--mem-fraction-static", str(args.mem_fraction_static)]
    if args.json_model_override_args:
        cmd += ["--json-model-override-args", args.json_model_override_args]
    cmd += extra
    if args.extra_args:
        cmd += shlex.split(args.extra_args)
    return cmd, log_path


def start_server(
    name: str,
    cmd: list[str],
    log_path: Path,
    timeout_s: float,
    env_updates: dict | None = None,
):
    server = Server(name=name, port=int(cmd[cmd.index("--port") + 1]),
                    log_path=log_path)
    server.start(cmd, env_updates=env_updates)
    if not server.wait_ready(timeout_s):
        tail = server.tail_log(80)
        server.stop()
        return server, False, tail
    return server, True, []


def probe_env(args) -> dict:
    env = {}
    if args.sglang_path:
        existing = os.environ.get("PYTHONPATH")
        env["PYTHONPATH"] = (
            f"{args.sglang_path}{os.pathsep}{existing}"
            if existing
            else args.sglang_path
        )
    for item in args.env:
        key, sep, value = item.partition("=")
        if not sep or not key:
            raise ValueError(f"--env expects KEY=VALUE, got {item!r}")
        env[key] = value
    return env


def run_e2e(args) -> dict:
    report = {
        "mode": "e2e",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "argv": sys.argv,
        "baseline": {},
        "candidate": {},
        "checks": {},
    }
    prompt_a, filler_chars, body = build_prompts(
        args.page_size, args.a_pages, args.filler_pages
    )
    env_updates = probe_env(args)
    report["probe_env"] = {
        "sglang_path": args.sglang_path,
        "env_keys": sorted(env_updates),
    }

    # Baseline: cold output with radix cache disabled.
    baseline_port = args.port
    cmd, log_path = server_cmd(
        args=args,
        port=baseline_port,
        extra=["--disable-radix-cache"],
        log_name="baseline",
    )
    baseline, ok, tail = start_server(
        "baseline",
        cmd,
        log_path,
        args.startup_timeout,
        env_updates=env_updates,
    )
    report["baseline"]["started"] = ok
    report["baseline"]["log"] = str(log_path)
    if ok:
        try:
            result = baseline.generate(
                prompt_a, args.output_tokens, args.request_timeout
            )
            report["baseline"]["prompt_chars"] = len(prompt_a)
            report["baseline"]["text"] = result.get("text")
            report["baseline"]["meta_info"] = result.get("meta_info", {})
            report["baseline"]["ok"] = True
        except Exception:
            report["baseline"]["ok"] = False
            report["baseline"]["traceback"] = traceback.format_exc()
            report["baseline"]["log_tail"] = baseline.tail_log(80)
    else:
        report["baseline"]["log_tail"] = tail
    baseline.stop()

    # Candidate: HiCache L2 with an explicit ratio.
    candidate_port = args.port + 1
    extra = [
        "--enable-hierarchical-cache",
        "--hicache-ratio",
        str(args.hicache_ratio),
        "--hicache-write-policy",
        "write_through",
    ]
    cmd, log_path = server_cmd(
        args=args,
        port=candidate_port,
        extra=extra,
        log_name="candidate",
    )
    candidate, ok, tail = start_server(
        "candidate",
        cmd,
        log_path,
        args.startup_timeout,
        env_updates=env_updates,
    )
    report["candidate"]["started"] = ok
    report["candidate"]["log"] = str(log_path)
    report["candidate"]["pools"] = parse_pools(log_path)
    report["candidate"]["geometry_checks"] = geometry_checks(
        report["candidate"]["pools"], args.page_size
    )
    if not ok:
        report["candidate"]["log_tail"] = tail
        return report

    try:
        info = candidate.server_info()
        report["candidate"]["server_info"] = info
        max_tokens = find_key(info, "max_total_num_tokens")
        if not isinstance(max_tokens, int) or max_tokens <= 0:
            max_tokens = args.device_tokens
        report["candidate"]["max_total_num_tokens"] = max_tokens

        before = candidate.metrics_text()
        report["candidate"]["host_used_before"] = metric_last(
            before, "sglang:hicache_host_used_tokens"
        )
        host_hit_before = metric_sum(
            before,
            "sglang:prefill_effective_tokens_total",
            'mode="host_hit"',
        )

        first = candidate.generate(
            prompt_a, args.output_tokens, args.request_timeout
        )
        report["candidate"]["first_text"] = first.get("text")
        report["candidate"]["first_meta_info"] = first.get("meta_info", {})
        report["candidate"]["first_ok"] = True

        # Fill device with unique prefixes.  Total unique tokens stay below
        # hicache_ratio * device capacity so A is not evicted from host too.
        a_tokens = args.page_size * args.a_pages
        target_tokens = int(max_tokens * args.fill_factor)
        filler_tokens = args.page_size * args.filler_pages
        filler_count = max(
            1, min(args.max_fillers, (target_tokens - a_tokens) // filler_tokens)
        )
        report["candidate"]["filler_count"] = filler_count
        report["candidate"]["target_unique_tokens"] = target_tokens
        errors = 0
        for i in range(filler_count):
            filler = body(f"FILLER-{i:04d}", filler_chars)
            try:
                candidate.generate(filler, 1, args.request_timeout)
            except Exception:
                errors += 1
                if candidate.proc is not None and candidate.proc.poll() is not None:
                    break
                if errors <= 3:
                    report.setdefault("candidate", {}).setdefault(
                        "filler_errors", []
                    ).append(traceback.format_exc())
            if (i + 1) % 20 == 0:
                log(f"[candidate] fillers {i + 1}/{filler_count}")
        report["candidate"]["filler_errors_count"] = errors

        after_fill = candidate.metrics_text()
        report["candidate"]["host_used_after_fill"] = metric_last(
            after_fill, "sglang:hicache_host_used_tokens"
        )
        report["candidate"]["host_hit_after_fill"] = metric_sum(
            after_fill,
            "sglang:prefill_effective_tokens_total",
            'mode="host_hit"',
        )

        hit = candidate.generate(
            prompt_a, args.output_tokens, args.request_timeout
        )
        report["candidate"]["hit_text"] = hit.get("text")
        report["candidate"]["hit_meta_info"] = hit.get("meta_info", {})
        after = candidate.metrics_text()
        host_hit_after = metric_sum(
            after,
            "sglang:prefill_effective_tokens_total",
            'mode="host_hit"',
        )
        report["candidate"]["host_hit_after_hit"] = host_hit_after
        report["candidate"]["host_hit_delta"] = host_hit_after - host_hit_before
        report["candidate"]["hit_cached_tokens"] = find_key(
            hit.get("meta_info", {}), "cached_tokens"
        )
    except Exception:
        report["candidate"]["traceback"] = traceback.format_exc()
        report["candidate"]["log_tail"] = candidate.tail_log(120)
    finally:
        candidate.stop()

    baseline_text = report["baseline"].get("text")
    hit_text = report["candidate"].get("hit_text")
    checks = report["checks"]
    checks["baseline_started"] = bool(report["baseline"].get("started"))
    checks["candidate_started"] = bool(report["candidate"].get("started"))
    checks["first_request_ok"] = bool(report["candidate"].get("first_ok"))
    checks["host_hit_delta_positive"] = (
        report["candidate"].get("host_hit_delta", 0) > 0
    )
    checks["host_used_increased"] = (
        report["candidate"].get("host_used_after_fill") or 0
    ) > (report["candidate"].get("host_used_before") or 0)
    checks["hit_cached_tokens_positive"] = (
        report["candidate"].get("hit_cached_tokens") or 0
    ) > 0
    checks["output_matches_baseline"] = (
        baseline_text is not None
        and hit_text is not None
        and baseline_text == hit_text
    )
    kv_checks = [
        check
        for check in report["candidate"].get("geometry_checks", [])
        if check["pool"] in ("deepseek_v4_c1", "deepseek_v4_c2")
    ]
    checks["g1_kv_item_bytes_ok"] = (
        len(kv_checks) == 2 and all(check["ok"] for check in kv_checks)
    )
    return report


def run_g10(args) -> dict:
    report = {
        "mode": "g10",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "argv": sys.argv,
    }
    cmd, log_path = server_cmd(
        args=args,
        port=args.port + 2,
        extra=["--enable-hierarchical-cache"],
        log_name="g10_default_ratio",
    )
    server, ok, tail = start_server(
        "g10",
        cmd,
        log_path,
        min(args.startup_timeout, 180),
        env_updates=probe_env(args),
    )
    report["started"] = ok
    report["log"] = str(log_path)
    report["log_tail"] = tail if not ok else server.tail_log(40)
    if ok:
        server.stop()
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--tp-size", type=int, default=8)
    parser.add_argument("--mem-fraction-static", type=float, default=None)
    parser.add_argument("--page-size", type=int, default=128)
    parser.add_argument("--hicache-ratio", type=float, default=2.0)
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--startup-timeout", type=float, default=900.0)
    parser.add_argument("--request-timeout", type=float, default=600.0)
    parser.add_argument("--output-tokens", type=int, default=16)
    parser.add_argument("--a-pages", type=int, default=8)
    parser.add_argument("--filler-pages", type=int, default=2)
    parser.add_argument("--fill-factor", type=float, default=1.3,
                        help="unique device tokens / device capacity")
    parser.add_argument("--max-fillers", type=int, default=300)
    parser.add_argument("--device-tokens", type=int, default=32768,
                        help="fallback when /get_server_info has no max_total_num_tokens")
    parser.add_argument(
        "--sglang-path",
        default=None,
        help="prepend this python/ directory to PYTHONPATH for the server subprocess",
    )
    parser.add_argument(
        "--env",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="extra environment variable for the server subprocess; repeatable",
    )
    parser.add_argument(
        "--json-model-override-args",
        default=None,
        help="passed unchanged to --json-model-override-args",
    )
    parser.add_argument("--extra-args", default="")
    parser.add_argument("--g10", action="store_true",
                        help="also test --enable-hierarchical-cache without ratio")
    args = parser.parse_args()

    if args.out_dir is None:
        args.out_dir = Path("probe_out") / f"hicache_e2e_{datetime.now():%Y%m%d_%H%M%S}"
    args.out_dir = args.out_dir.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    report = {"created_at": datetime.now().isoformat(timespec="seconds"),
              "argv": sys.argv}
    try:
        report["e2e"] = run_e2e(args)
        if args.g10:
            report["g10"] = run_g10(args)
    except Exception:
        report["traceback"] = traceback.format_exc()
        log(report["traceback"])

    out = args.out_dir / "report.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    log(f"\nreport: {out}")
    log(f"logs:   {args.out_dir}")

    if "e2e" in report:
        e2e = report["e2e"]
        log("\n=== summary ===")
        for key, value in e2e.get("checks", {}).items():
            log(f"{key}: {value}")
        for check in e2e.get("candidate", {}).get("geometry_checks", []):
            log(
                f"pool {check['pool']}: item_bytes "
                f"actual={check['actual_item_bytes']} expected="
                f"{check['expected_item_bytes']} ok={check['ok']}"
            )
        if e2e.get("baseline", {}).get("log_tail"):
            log(f"baseline log tail:\n" + "\n".join(
                e2e["baseline"]["log_tail"][-40:]
            ))
        if e2e.get("candidate", {}).get("log_tail"):
            log(f"candidate log tail:\n" + "\n".join(
                e2e["candidate"]["log_tail"][-40:]
            ))
    if "g10" in report:
        g10 = report["g10"]
        log(f"\nG10 started={g10['started']} log={g10['log']}")
        log("\n".join(g10.get("log_tail", [])[-40:]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
