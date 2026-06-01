"""
eval/run_eval.py — measure quality instead of eyeballing it. This is the part
most portfolios skip, and the reason GroundTruth is trustworthy.

On the gold set (eval/gold.jsonl) we score three things:

  retrieval_recall     (in-scope)     did the expected source doc get retrieved?
  faithfulness         (in-scope)     an LLM judge checks the answer is supported
                                      by the retrieved context (not hallucinated)
  abstention_accuracy  (out-of-scope) did the system correctly refuse to answer?

It writes eval/report.json, prints deltas vs. the previous run, and exits non-zero
if any metric falls below its threshold — so it can act as a CI regression gate.

Run:  python -m eval.run_eval
"""

import json
import sys
import time
from pathlib import Path

import cohere

from app.config import settings
from app.cohere_utils import call_with_retry
from app.rag import answer

co = cohere.ClientV2(api_key=settings.cohere_api_key)

# CI fails the build if a metric drops below these.
THRESHOLDS = {
    "retrieval_recall": 0.75,
    "faithfulness": 0.75,
    "abstention_accuracy": 0.80,
}

REPORT_PATH = Path("eval/report.json")
HISTORY_PATH = Path("eval/history.jsonl")

JUDGE_PROMPT = """You are grading an ANSWER for FAITHFULNESS to its CONTEXT.
Reply with only one word: "yes" if the answer's claims are supported by the context,
or "no" only if the answer states something the context does not support. Ignore
formatting, citation markers like [1], and minor rephrasing — judge the facts only.

CONTEXT:
{context}

ANSWER:
{answer}

Supported (yes/no):"""


def judge_faithfulness(context: str, ans: str) -> bool:
    resp = call_with_retry(
        co.chat,
        model=settings.chat_model,
        messages=[{"role": "user", "content": JUDGE_PROMPT.format(context=context, answer=ans)}],
    )
    return resp.message.content[0].text.strip().lower().startswith("yes")


def load_gold() -> list[dict]:
    lines = Path("eval/gold.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def main() -> None:
    gold = load_gold()
    in_scope = [c for c in gold if not c.get("out_of_scope")]
    out_scope = [c for c in gold if c.get("out_of_scope")]

    recall_hits = faithful_hits = abstain_hits = 0
    latencies: list[float] = []
    costs: list[float] = []

    for case in gold:
        result = answer(case["question"])
        latencies.append(result["latency_ms"])
        costs.append(result["est_cost_usd"])

        if case.get("out_of_scope"):
            ok = result["abstained"]
            abstain_hits += int(ok)
            print(f"[out] {case['question']}\n      abstained={result['abstained']}  -> {'OK' if ok else 'MISS'}")
        else:
            retrieved = {s["doc"] for s in result["sources"]}
            recalled = case["expected_doc"] in retrieved
            recall_hits += int(recalled)

            context = "\n\n".join(s["snippet"] for s in result["sources"])
            faithful = bool(result["sources"]) and judge_faithfulness(context, result["answer"])
            faithful_hits += int(faithful)
            print(f"[in]  {case['question']}\n      recall={recalled} faithful={faithful}  ans={result['answer'][:80]!r}")

    n_in = max(len(in_scope), 1)
    n_out = max(len(out_scope), 1)
    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model": settings.chat_model,
        "use_langgraph": settings.use_langgraph,
        "n_total": len(gold),
        "n_in_scope": len(in_scope),
        "n_out_of_scope": len(out_scope),
        "retrieval_recall": round(recall_hits / n_in, 3),
        "faithfulness": round(faithful_hits / n_in, 3),
        "abstention_accuracy": round(abstain_hits / n_out, 3),
        "avg_latency_ms": round(sum(latencies) / len(latencies), 1),
        "p95_latency_ms": round(sorted(latencies)[int(0.95 * (len(latencies) - 1))], 1),
        "avg_cost_usd": round(sum(costs) / len(costs), 6),
        "thresholds": THRESHOLDS,
    }

    prev = json.loads(REPORT_PATH.read_text()) if REPORT_PATH.exists() else None
    REPORT_PATH.write_text(json.dumps(report, indent=2))
    with HISTORY_PATH.open("a") as f:
        f.write(json.dumps(report) + "\n")

    print("\n" + "=" * 60)
    for key in ("retrieval_recall", "faithfulness", "abstention_accuracy",
                "p95_latency_ms", "avg_cost_usd"):
        delta = ""
        if prev and key in prev:
            d = report[key] - prev[key]
            delta = f"  (prev {prev[key]}, {'+' if d >= 0 else ''}{round(d, 3)})"
        print(f"{key:22}: {report[key]}{delta}")

    failed = [
        f"{k} {report[k]} < {v}"
        for k, v in THRESHOLDS.items()
        if report[k] < v
    ]
    print("=" * 60)
    if failed:
        print("FAIL (regression gate):")
        for f in failed:
            print("  -", f)
        sys.exit(1)
    print("PASS: all metrics above thresholds.")


if __name__ == "__main__":
    main()
