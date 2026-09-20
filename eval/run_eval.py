"""Compare chunking strategies on one fixed question set.

Retrieval metrics (answerable questions only, gold_pages required):
    Hit@k  fraction of questions where at least one of the top-k chunks
           overlaps a gold page. "Did the evidence make it into the context?"
    MRR    mean of 1/rank of the first correct chunk. Rewards ranking it high.
    Recall fraction of gold pages covered anywhere in the top-k.

Refusal metrics (needs no gold pages):
    Correct answers    answerable questions where something passed the gate
    Correct refusals   unanswerable questions where nothing passed the gate

Usage:
    python eval/run_eval.py                  # compare strategies
    python eval/run_eval.py --calibrate      # pick a distance threshold
    python eval/run_eval.py --k 8
"""

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import DISTANCE_THRESHOLD, STRATEGIES, TOP_K  # noqa: E402
from src.retriever import search  # noqa: E402

QUESTIONS_PATH = ROOT / "eval" / "questions.json"
RESULTS_PATH = ROOT / "eval" / "results.md"


def load_questions():
    questions = json.loads(QUESTIONS_PATH.read_text())
    answerable = [q for q in questions if q["answerable"]]
    labelled = [q for q in answerable if q.get("gold_pages")]
    if not labelled:
        print("No gold_pages filled in yet — retrieval metrics will be skipped.\n"
              "Open eval/questions.json and add the page numbers that answer each\n"
              "question. That labelling is what turns an opinion into evidence.\n")
    return questions, answerable, labelled


def evaluate(strategy: str, questions, k: int, threshold: float) -> dict:
    hits_at_k, reciprocal_ranks, recalls = [], [], []
    correct_answers = correct_refusals = 0
    answerable_total = unanswerable_total = 0
    best_answerable, best_unanswerable = [], []
    per_question = []

    for item in questions:
        results = search(item["question"], strategy=strategy, k=k)
        best = results[0].distance if results else float("inf")
        passed = any(r.distance <= threshold for r in results)

        if item["answerable"]:
            answerable_total += 1
            best_answerable.append(best)
            correct_answers += int(passed)
        else:
            unanswerable_total += 1
            best_unanswerable.append(best)
            correct_refusals += int(not passed)

        gold = item.get("gold_pages") or []
        if item["answerable"] and gold:
            ranks = [i for i, r in enumerate(results, start=1) if r.covers(gold)]
            hits_at_k.append(1 if ranks else 0)
            reciprocal_ranks.append(1 / ranks[0] if ranks else 0.0)
            covered = {p for p in gold if any(r.covers([p]) for r in results)}
            recalls.append(len(covered) / len(gold))

        per_question.append({
            "id": item["id"],
            "best_distance": round(best, 4),
            "passed_gate": passed,
            "answerable": item["answerable"],
            "top_source": f"{results[0].source} p.{results[0].pages}" if results else "-",
        })

    mean = lambda xs: round(statistics.mean(xs), 3) if xs else None  # noqa: E731

    return {
        "strategy": strategy,
        "labelled": len(hits_at_k),
        "hit_at_k": mean(hits_at_k),
        "mrr": mean(reciprocal_ranks),
        "page_recall": mean(recalls),
        "answer_rate": round(correct_answers / answerable_total, 3) if answerable_total else None,
        "refusal_rate": round(correct_refusals / unanswerable_total, 3) if unanswerable_total else None,
        "mean_best_answerable": mean(best_answerable),
        "mean_best_unanswerable": mean(best_unanswerable),
        "per_question": per_question,
    }


def comparison_table(rows) -> str:
    head = ("| Strategy | Hit@k | MRR | Page recall | Answered (should) | "
            "Refused (should) | Mean best dist. answerable | unanswerable |")
    sep = "|---|---|---|---|---|---|---|---|"
    lines = [head, sep]
    for r in rows:
        fmt = lambda v: "n/a" if v is None else v  # noqa: E731
        lines.append(
            f"| {r['strategy']} | {fmt(r['hit_at_k'])} | {fmt(r['mrr'])} | "
            f"{fmt(r['page_recall'])} | {fmt(r['answer_rate'])} | "
            f"{fmt(r['refusal_rate'])} | {fmt(r['mean_best_answerable'])} | "
            f"{fmt(r['mean_best_unanswerable'])} |"
        )
    return "\n".join(lines)


def calibrate(questions, k: int) -> str:
    """Sweep thresholds and show where answerable and unanswerable separate."""
    out = []
    for strategy in STRATEGIES:
        answerable, unanswerable = [], []
        for item in questions:
            results = search(item["question"], strategy=strategy, k=k)
            best = results[0].distance if results else float("inf")
            (answerable if item["answerable"] else unanswerable).append(best)

        out.append(f"\n### {strategy}")
        out.append(f"answerable   best distance: min {min(answerable):.3f} "
                   f"median {statistics.median(answerable):.3f} max {max(answerable):.3f}")
        out.append(f"unanswerable best distance: min {min(unanswerable):.3f} "
                   f"median {statistics.median(unanswerable):.3f} max {max(unanswerable):.3f}")
        out.append("")
        out.append("| threshold | answered correctly | refused correctly | total correct |")
        out.append("|---|---|---|---|")

        best_t, best_score = None, -1
        t = 0.30
        while t <= 1.30001:
            ok_ans = sum(1 for d in answerable if d <= t) / len(answerable)
            ok_ref = sum(1 for d in unanswerable if d > t) / len(unanswerable)
            score = (ok_ans * len(answerable) + ok_ref * len(unanswerable))
            if score > best_score:
                best_score, best_t = score, t
            out.append(f"| {t:.2f} | {ok_ans:.2f} | {ok_ref:.2f} | {int(round(score))} |")
            t += 0.05
        out.append(f"\nBest separation for '{strategy}' at DISTANCE_THRESHOLD={best_t:.2f}")
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=TOP_K)
    parser.add_argument("--threshold", type=float, default=DISTANCE_THRESHOLD)
    parser.add_argument("--calibrate", action="store_true")
    args = parser.parse_args()

    questions, _, _ = load_questions()

    if args.calibrate:
        report = calibrate(questions, args.k)
        print(report)
        (ROOT / "eval" / "calibration.md").write_text(
            f"# Threshold calibration (k={args.k})\n{report}\n"
        )
        return

    rows = [evaluate(s, questions, args.k, args.threshold) for s in STRATEGIES]
    table = comparison_table(rows)
    print(f"\nk={args.k}, threshold={args.threshold}, "
          f"{len(questions)} questions ({rows[0]['labelled']} with gold pages)\n")
    print(table)

    RESULTS_PATH.write_text(
        f"# Chunking comparison\n\n"
        f"k={args.k}, distance threshold={args.threshold}, "
        f"{len(questions)} questions, {rows[0]['labelled']} with gold pages.\n\n"
        f"{table}\n\n## Per question\n\n"
        + "\n".join(
            f"### {r['strategy']}\n\n"
            + "| id | answerable | best distance | passed gate | top hit |\n|---|---|---|---|---|\n"
            + "\n".join(
                f"| {p['id']} | {p['answerable']} | {p['best_distance']} | "
                f"{p['passed_gate']} | {p['top_source']} |"
                for p in r["per_question"]
            )
            for r in rows
        )
        + "\n"
    )
    print(f"\nWritten to {RESULTS_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()