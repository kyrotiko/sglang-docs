#!/usr/bin/env python3
"""Compare per-token prompt logprobs between two server runs.

Used to check a numerically sensitive kernel change (for example the
ascend_ops expert-output dtype switch): launch the server twice with only the
environment variable differing, dump logprobs from each, then diff them.

    # run A: new path (default)
    python compare_logprobs.py dump --url http://127.0.0.1:30000 --out a.json
    # run B: SGLANG_ASCEND_OPS_FP32_EXPERT_OUT=1
    python compare_logprobs.py dump --url http://127.0.0.1:30000 --out b.json

    python compare_logprobs.py diff a.json b.json

Greedy decoding plus ``echo`` makes the prompt token logprobs the probe: any
divergence inside prefill shows up there before it reaches the sampled text.
"""

from __future__ import annotations

import argparse
import json

import requests

PROMPTS = [
    "The capital of France is",
    "2 + 2 = 4, 4 + 4 = 8, 8 + 8 =",
    "def fibonacci(n):\n    if n <= 1:",
    "Explain in one sentence what a mixture-of-experts layer does:",
    "请用一句话解释什么是专家混合模型：",
    "下面这段代码的输出是什么？\nprint(sum(range(10)))",
    "Q: What is the boiling point of water at sea level?\nA:",
    "List three prime numbers greater than 100:",
]


def dump(url: str, model: str, out: str) -> None:
    rows = []
    for prompt in PROMPTS:
        resp = requests.post(
            url.rstrip("/") + "/v1/completions",
            json={
                "model": model,
                "prompt": prompt,
                "max_tokens": 4,
                "temperature": 0.0,
                "echo": True,
                "logprobs": 1,
            },
            timeout=300,
        )
        resp.raise_for_status()
        choice = resp.json()["choices"][0]
        rows.append(
            {
                "prompt": prompt,
                "logprobs": (choice.get("logprobs") or {}).get("token_logprobs"),
                "text": choice.get("text"),
            }
        )
    with open(out, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=1)
    print(f"wrote {out}: {len(rows)} prompts")


def diff(a_path: str, b_path: str) -> None:
    a = json.load(open(a_path, encoding="utf-8"))
    b = json.load(open(b_path, encoding="utf-8"))

    worst = 0.0
    worst_prompt = ""
    compared = 0
    text_mismatch = []

    for ra, rb in zip(a, b):
        la = ra.get("logprobs") or []
        lb = rb.get("logprobs") or []
        for x, y in zip(la, lb):
            if x is None or y is None:
                continue
            delta = abs(x - y)
            compared += 1
            if delta > worst:
                worst, worst_prompt = delta, ra["prompt"][:40]
        if ra.get("text") != rb.get("text"):
            text_mismatch.append(ra["prompt"][:40])

    print(f"tokens compared     : {compared}")
    print(f"max |logprob diff|  : {worst:.3e}   ({worst_prompt!r})")
    print(f"generation mismatch : {len(text_mismatch)}/{len(a)}")
    for p in text_mismatch:
        print(f"  - {p!r}")

    if compared == 0:
        print("\nno logprobs in the response; check that the server returns them")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("dump", help="collect logprobs from a running server")
    d.add_argument("--url", default="http://127.0.0.1:30000")
    d.add_argument("--model", default="default")
    d.add_argument("--out", required=True)

    c = sub.add_parser("diff", help="compare two dump files")
    c.add_argument("a")
    c.add_argument("b")

    args = ap.parse_args()
    if args.cmd == "dump":
        dump(args.url, args.model, args.out)
    else:
        diff(args.a, args.b)


if __name__ == "__main__":
    main()
