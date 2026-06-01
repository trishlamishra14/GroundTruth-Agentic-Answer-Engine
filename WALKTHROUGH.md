# GroundTruth — Walkthrough & Interview Notes

This is the "explain it like you built it" guide. Read it before an interview; it maps each
design choice to the reason behind it, which is exactly what interviewers probe for.

## The one-sentence pitch

> GroundTruth is an agentic RAG service that answers documentation questions with citations,
> retries weak queries, refuses out-of-scope ones, and is gated by an automated evaluation
> suite that runs in CI.

## Why each piece exists

**Hybrid retrieval (vector + keyword + RRF).** Pure vector search misses exact tokens
(error codes, flag names); pure keyword search misses paraphrases. Reciprocal Rank Fusion
combines both ranked lists so a passage strong in either surfaces, and one strong in *both*
wins. *Talking point: "I didn't assume embeddings were enough — I fused lexical and semantic."*

**Cohere rerank.** Embedding distance is a rough relevance proxy. A cross-encoder reranker
reads the query and each candidate together and scores true relevance, so the 3–4 passages we
feed the model are the right ones. The top rerank score also drives the abstain decision.

**The agent loop (`agent.py` / `graph.py`).** A one-shot RAG fails silently on hard queries.
GroundTruth grades the best passage; if it's weak and retries remain, it asks the model to
rewrite the question into a better search query and tries again; if still weak, it abstains.
*Talking point: "It's a controller with conditional branching, not a straight pipeline —
that's why I expressed it as a LangGraph graph."*

**Abstention.** Hallucination is the #1 trust killer. The prompt forces a fixed refusal
sentence when context is insufficient, and a score threshold abstains before generation. We
treat refusal as a feature and **measure** it.

**Evaluation harness (`eval/run_eval.py`).** The differentiator. On a gold set we score:
- *retrieval recall* — did the expected doc get retrieved? (bad retrieval caps everything)
- *faithfulness* — an LLM judge checks the answer is supported by context (no hallucination)
- *abstention accuracy* — did we correctly refuse the out-of-scope questions?
It writes `report.json`, prints deltas vs. the previous run, and exits non-zero below
threshold so it acts as a **regression gate** in CI.

**Observability + cost.** Every answer records latency and an estimated token cost; `/metrics`
exposes rolling p50/p95 latency and average cost. *Talking point: "I can tell you the p95
latency and the cost per query, and how the reranker changed both."*

## A demo script (90 seconds)

1. Ask an in-scope question → show the cited answer + the `trace` (retrieve → grade → generate).
2. Ask an out-of-scope question → show it abstain.
3. Ask a deliberately vague question → show the agent **rewrite and retry** in the trace.
4. Run `python -m eval.run_eval` → show recall / faithfulness / abstention and the pass/fail gate.
5. Hit `/metrics` → show p95 latency and cost/query.

## Honest limitations (say these — it signals maturity)

- The gold set is small; numbers are directional until it grows to ~150 questions.
- The faithfulness judge is an LLM, so it has its own error rate; a human spot-check matters.
- Cost is an estimate from token counts, not a billing readout.
- Sentence-based chunking is simple; structure-aware chunking would help on big docs.

## What I'd do next

Public demo deploy, a small UI that shows citations inline, structure-aware chunking, caching
of embeddings/answers, and a larger gold set with adversarial out-of-scope questions.
