"""
eval/calibrate.py — choose the abstention threshold from data, not by guessing.

For each gold question we run retrieval + rerank (no generation, so it's cheap) and
record the top rerank score. Then we sweep candidate thresholds and, at each one, measure:

  in-scope answered   — in-scope questions whose top score clears the threshold
                        (want high: don't refuse answerable questions)
  abstention accuracy — out-of-scope questions correctly left below the threshold
  balanced            — the average of the two

If the in-scope and out-of-scope score ranges separate cleanly, the recommended
threshold is the midpoint of the gap; otherwise it's the threshold that maximizes the
balanced score. Put the result in .env as ABSTAIN_THRESHOLD and re-run the eval.

Run:  python -m eval.calibrate
"""

import json
from pathlib import Path

from app.retrieve import retrieve


def top_score(question: str) -> float:
    hits = retrieve(question)
    return hits[0]["score"] if hits else 0.0


def main() -> None:
    gold = [json.loads(l) for l in Path("eval/gold.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]

    in_scores: list[float] = []
    oos_scores: list[float] = []
    for c in gold:
        s = top_score(c["question"])
        (oos_scores if c.get("out_of_scope") else in_scores).append(s)
        tag = "OOS" if c.get("out_of_scope") else "in "
        print(f"  [{tag}] score={s:.3f}  {c['question'][:58]}")

    n_in, n_oos = max(len(in_scores), 1), max(len(oos_scores), 1)

    print("\n thresh | in-scope answered | abstention acc | balanced")
    print(" -------+-------------------+----------------+----------")
    best = (0.0, -1.0)
    t = 0.02
    while t <= 0.5001:
        ans = sum(1 for s in in_scores if s >= t) / n_in
        abst = sum(1 for s in oos_scores if s < t) / n_oos
        bal = (ans + abst) / 2
        if bal > best[1] + 1e-9:
            best = (round(t, 2), bal)
        print(f"  {t:0.2f}  |      {ans * 100:5.1f}%       |     {abst * 100:5.1f}%     |  {bal * 100:5.1f}%")
        t += 0.02

    print("\nScore separation:")
    if in_scores:
        srt = sorted(in_scores)
        print(f"  in-scope     min/median/max: {min(in_scores):.3f} / {srt[len(srt) // 2]:.3f} / {max(in_scores):.3f}")
    if oos_scores:
        srt = sorted(oos_scores)
        print(f"  out-of-scope min/median/max: {min(oos_scores):.3f} / {srt[len(srt) // 2]:.3f} / {max(oos_scores):.3f}")

    if in_scores and oos_scores and max(oos_scores) < min(in_scores):
        rec = round((max(oos_scores) + min(in_scores)) / 2, 3)
        print(f"\nScores separate cleanly. Recommended ABSTAIN_THRESHOLD = {rec}")
        print("(any value in the gap perfectly splits in-scope from out-of-scope on this set)")
    else:
        print(f"\nRecommended ABSTAIN_THRESHOLD = {best[0]:.2f}  (balanced {best[1] * 100:.1f}%)")

    print("Set it in .env, then re-run:  python -m eval.run_eval")


if __name__ == "__main__":
    main()
