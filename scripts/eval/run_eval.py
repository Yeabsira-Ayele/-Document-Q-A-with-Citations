"""Compare chunking strategies on one fixed, labelled question set.

Every answerable question in questions.json carries:
    gold_pages   pages that contain the answer (1-based, as in a PDF viewer)
    alt_pages    other pages that also answer it (e.g. a quick-reference table);
                 they count as correct for Hit@k and MRR but not for recall
    answer_keys  short phrases that must ALL appear for the answer to be complete

Retrieval metrics (answerable questions, top-k chunks):
    Fact@k       some retrieved chunk contains every answer key. This is the
                 headline metric: "did the model get the whole answer?"
    Fact@1       the same, but only for the single best chunk
    Hit@k        some retrieved chunk overlaps a gold/alt page
    MRR          mean of 1/rank of the first chunk on a correct page
    Page recall  share of gold pages covered anywhere in the top-k
    Intact       (needs no embeddings) the answer sits whole inside ONE chunk of
                 the index, i.e. the chunker did not cut it in half

Refusal metrics, gate only (whether anything passes DISTANCE_THRESHOLD):
    Answered     answerable questions where something passed the gate
    Refused      unanswerable questions where nothing passed the gate, split into
                 off_topic (nothing in the corpus is related) and
                 in_domain_missing (related topic, but the answer is not there;
                 a distance gate cannot catch these, the LLM prompt must)

Usage:
    python eval/run_eval.py                 # compare strategies -> eval/results.md
    python eval/run_eval.py --calibrate     # choose DISTANCE_THRESHOLD -> eval/calibration.md
    python eval/run_eval.py --e2e           # full pipeline through Gemini -> eval/e2e.md
    python eval/run_eval.py --chunks-only   # chunk-integrity only, no embeddings needed
    python eval/run_eval.py --k 8
"""

import argparse
import json
import re
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import DEFAULT_STRATEGY, DISTANCE_THRESHOLD, STRATEGIES, TOP_K  # noqa: E402

EVAL_DIR = ROOT / "eval"
RESULTS_PATH = EVAL_DIR / "results.md"
CALIBRATION_PATH = EVAL_DIR / "calibration.md"
E2E_PATH = EVAL_DIR / "e2e.md"


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


def _questions_path() -> Path:
    # Accept questions.json or questions.JSON (matters on case-sensitive systems).
    for name in ("questions.json", "questions.JSON"):
        if (EVAL_DIR / name).exists():
            return EVAL_DIR / name
    raise FileNotFoundError(f"No questions.json found in {EVAL_DIR}")


def load_questions() -> list:
    questions = json.loads(_questions_path().read_text(encoding="utf-8"))
    for q in questions:
        q.setdefault("kind", "answerable" if q["answerable"] else "off_topic")
        q.setdefault("gold_pages", [])
        q.setdefault("alt_pages", [])
        q.setdefault("answer_keys", [])
    unlabelled = [q["id"] for q in questions if q["answerable"] and not q["gold_pages"]]
    if unlabelled:
        print(f"Warning: no gold_pages for {', '.join(unlabelled)}; "
              f"page metrics skip them.")
    return questions


def _norm(text: str) -> str:
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    return re.sub(r"\s+", " ", text).lower()


def has_all_keys(text: str, keys: list) -> bool:
    """True if every key phrase appears in text (whitespace/case-insensitive)."""
    if not keys:
        return False
    haystack = _norm(text)
    return all(_norm(k) in haystack for k in keys)


def fmt_frac(hits: int, total: int) -> str:
    return "n/a" if total == 0 else f"{hits / total:.3f} ({hits}/{total})"


# --------------------------------------------------------------------------
# Embedding-free: does the chunker keep each answer whole?
# --------------------------------------------------------------------------


def chunk_report(strategy: str, questions: list, pages) -> dict:
    from src.chunking import chunk, describe

    chunks = chunk(strategy, pages)
    answerable = [q for q in questions if q["answerable"] and q["answer_keys"]]
    intact = {
        q["id"]: any(has_all_keys(c.page_content, q["answer_keys"]) for c in chunks)
        for q in answerable
    }
    return {
        "strategy": strategy,
        "summary": describe(chunks),
        "intact": intact,
        "intact_n": sum(intact.values()),
        "intact_total": len(intact),
        "broken": [qid for qid, ok in intact.items() if not ok],
    }


# --------------------------------------------------------------------------
# Retrieval evaluation
# --------------------------------------------------------------------------


def evaluate(strategy: str, questions: list, k: int, threshold: float, chunk_info: dict) -> dict:
    from src import retriever

    rows = []
    for q in questions:
        results = retriever.search(q["question"], strategy=strategy, k=k)
        best = results[0].distance if results else float("inf")
        row = {
            "id": q["id"],
            "kind": q["kind"],
            "answerable": q["answerable"],
            "best_distance": round(best, 4),
            "passed_gate": any(r.distance <= threshold for r in results),
            "top_source": f"p.{results[0].pages}" if results else "-",
        }
        gold, alt, keys = q["gold_pages"], q["alt_pages"], q["answer_keys"]
        if q["answerable"] and gold:
            ranks = [i for i, r in enumerate(results, 1) if r.covers(gold + alt)]
            row["hit"] = bool(ranks)
            row["rr"] = 1 / ranks[0] if ranks else 0.0
            row["recall"] = (
                sum(1 for p in gold if any(r.covers([p]) for r in results)) / len(gold)
            )
        if q["answerable"] and keys:
            fact_ranks = [i for i, r in enumerate(results, 1)
                          if has_all_keys(r.document.page_content, keys)]
            row["fact"] = bool(fact_ranks)
            row["fact1"] = fact_ranks[:1] == [1]
        rows.append(row)

    ans = [r for r in rows if r["answerable"]]
    with_pages = [r for r in ans if "hit" in r]
    with_keys = [r for r in ans if "fact" in r]
    off = [r for r in rows if r["kind"] == "off_topic"]
    dom = [r for r in rows if r["kind"] == "in_domain_missing"]
    mean = lambda xs: round(statistics.mean(xs), 3) if xs else None  # noqa: E731

    return {
        "strategy": strategy,
        "chunks": chunk_info["summary"],
        "n_answerable": len(ans),
        "fact_n": sum(r["fact"] for r in with_keys), "fact_total": len(with_keys),
        "fact1_n": sum(r["fact1"] for r in with_keys),
        "hit_n": sum(r["hit"] for r in with_pages), "hit_total": len(with_pages),
        "mrr": mean([r["rr"] for r in with_pages]),
        "recall": mean([r["recall"] for r in with_pages]),
        "intact_n": chunk_info["intact_n"], "intact_total": chunk_info["intact_total"],
        "answered_n": sum(r["passed_gate"] for r in ans), "answered_total": len(ans),
        "off_n": sum(not r["passed_gate"] for r in off), "off_total": len(off),
        "dom_n": sum(not r["passed_gate"] for r in dom), "dom_total": len(dom),
        "mean_best_ans": mean([r["best_distance"] for r in ans]),
        "mean_best_off": mean([r["best_distance"] for r in off]),
        "rows": rows,
    }


def sort_key(r: dict) -> tuple:
    """Ranking: whole answer retrieved > ranked high > right page > chunk intact."""
    frac = lambda n, t: (n / t) if t else 0.0  # noqa: E731
    return (
        frac(r["fact_n"], r["fact_total"]),
        r["mrr"] or 0.0,
        frac(r["hit_n"], r["hit_total"]),
        r["recall"] or 0.0,
        frac(r["intact_n"], r["intact_total"]),
    )


def tables(results: list) -> str:
    head1 = ("| Strategy | Fact@k | Fact@1 | Hit@k | MRR | Page recall | Intact |\n"
             "|---|---|---|---|---|---|---|")
    body1 = [
        f"| {r['strategy']} | {fmt_frac(r['fact_n'], r['fact_total'])} | "
        f"{fmt_frac(r['fact1_n'], r['fact_total'])} | "
        f"{fmt_frac(r['hit_n'], r['hit_total'])} | {r['mrr']} | {r['recall']} | "
        f"{fmt_frac(r['intact_n'], r['intact_total'])} |"
        for r in results
    ]
    head2 = ("| Strategy | Chunks | Answered (gate) | Refused off-topic (gate) | "
             "Refused in-domain-missing (gate) | Mean best dist. answerable | off-topic |\n"
             "|---|---|---|---|---|---|---|")
    body2 = [
        f"| {r['strategy']} | {r['chunks'].replace(' | ', '; ')} | "
        f"{fmt_frac(r['answered_n'], r['answered_total'])} | "
        f"{fmt_frac(r['off_n'], r['off_total'])} | "
        f"{fmt_frac(r['dom_n'], r['dom_total'])} | "
        f"{r['mean_best_ans']} | {r['mean_best_off']} |"
        for r in results
    ]
    return ("### Retrieval quality\n\n" + "\n".join([head1] + body1) +
            "\n\n### Chunks and refusal gate\n\n" + "\n".join([head2] + body2))


def verdict(results: list) -> str:
    ranked = sorted(results, key=sort_key, reverse=True)
    best, other = ranked[0], ranked[1]
    if sort_key(best) == sort_key(other):
        headline = ("**No clear winner:** the strategies tie on every metric. "
                    "Prefer the simpler one (recursive).")
    else:
        headline = (f"**Winner: `{best['strategy']}`.** Strategies are ranked by Fact@k "
                    f"(does the retrieved context contain the whole answer?), then MRR, "
                    f"Hit@k, page recall and chunk integrity.")

    lines = [headline, ""]
    for r in ranked:
        lines.append(
            f"- `{r['strategy']}`: Fact@k {fmt_frac(r['fact_n'], r['fact_total'])}, "
            f"Fact@1 {fmt_frac(r['fact1_n'], r['fact_total'])}, MRR {r['mrr']}, "
            f"Hit@k {fmt_frac(r['hit_n'], r['hit_total'])}, "
            f"answers left intact in one chunk {fmt_frac(r['intact_n'], r['intact_total'])}."
        )

    fact = {r["strategy"]: {x["id"]: x.get("fact") for x in r["rows"] if "fact" in x}
            for r in results}
    a, b = ranked[0]["strategy"], ranked[1]["strategy"]
    only_a = [qid for qid, ok in fact[a].items() if ok and not fact[b].get(qid)]
    only_b = [qid for qid, ok in fact[b].items() if ok and not fact[a].get(qid)]
    lines += ["",
              f"Answered completely by `{a}` but not `{b}`: {', '.join(only_a) or 'none'}.",
              f"Answered completely by `{b}` but not `{a}`: {', '.join(only_b) or 'none'}."]
    return "\n".join(lines)


def per_question(results: list) -> str:
    out = []
    for r in results:
        out.append(f"### {r['strategy']}\n")
        out.append("| id | kind | best distance | gate | top hit | correct page | full answer |")
        out.append("|---|---|---|---|---|---|---|")
        mark = lambda v: "-" if v is None else ("yes" if v else "NO")  # noqa: E731
        for x in r["rows"]:
            out.append(f"| {x['id']} | {x['kind']} | {x['best_distance']} | "
                       f"{'pass' if x['passed_gate'] else 'refuse'} | {x['top_source']} | "
                       f"{mark(x.get('hit'))} | {mark(x.get('fact'))} |")
        out.append("")
    return "\n".join(out)


# --------------------------------------------------------------------------
# Threshold calibration
# --------------------------------------------------------------------------


def calibrate(questions: list, k: int) -> str:
    """Sweep the gate. Only off-topic questions are used as negatives: a distance
    gate cannot separate in-domain-but-missing questions, and tuning to them would
    only wreck recall on real questions."""
    from src import retriever

    out = []
    for strategy in STRATEGIES:
        ans, off = [], []
        for q in questions:
            if not q["answerable"] and q["kind"] != "off_topic":
                continue
            res = retriever.search(q["question"], strategy=strategy, k=k)
            best = res[0].distance if res else float("inf")
            (ans if q["answerable"] else off).append(best)

        out.append(f"\n### {strategy}")
        out.append(f"answerable best distance:  min {min(ans):.3f}  "
                   f"median {statistics.median(ans):.3f}  max {max(ans):.3f}")
        out.append(f"off-topic  best distance:  min {min(off):.3f}  "
                   f"median {statistics.median(off):.3f}  max {max(off):.3f}")

        scores = {}
        for i in range(30, 101):
            t = i / 100
            scores[t] = (sum(d <= t for d in ans), sum(d > t for d in off))
        top = max(a + r for a, r in scores.values())
        plateau = sorted(t for t, (a, r) in scores.items() if a + r == top)
        runs, run = [], [plateau[0]]
        for t in plateau[1:]:
            if round(t - run[-1], 2) > 0.011:
                runs.append(run)
                run = []
            run.append(t)
        runs.append(run)
        longest = max(runs, key=len)
        rec = round((longest[0] + longest[-1]) / 2, 2)
        a_ok, r_ok = scores[rec]

        out.append("")
        out.append("| threshold | answered correctly | refused correctly |")
        out.append("|---|---|---|")
        for i in range(30, 101, 5):
            t = i / 100
            a, r = scores[t]
            out.append(f"| {t:.2f} | {a}/{len(ans)} | {r}/{len(off)} |")
        gap = "clean gap" if max(ans) < min(off) else "the two groups overlap"
        out.append(f"\n{gap}. Best setting for '{strategy}': DISTANCE_THRESHOLD={rec:.2f} "
                   f"({a_ok}/{len(ans)} answered, {r_ok}/{len(off)} refused).")
    return "\n".join(out)


# --------------------------------------------------------------------------
# End to end (needs GOOGLE_API_KEY)
# --------------------------------------------------------------------------


def run_e2e(questions: list, strategy: str, k: int, threshold: float, delay: float) -> str:
    from src.qa import answer

    rows = []
    for q in questions:
        res = None
        for attempt in range(3):
            try:
                res = answer(q["question"], strategy=strategy, k=k, threshold=threshold)
                break
            except Exception as exc:  # network / quota
                print(f"  {q['id']}: {type(exc).__name__}; retry {attempt + 1}/3")
                time.sleep(15 * (attempt + 1))
        if res is None:
            rows.append({"id": q["id"], "q": q, "error": True})
            continue
        cited = bool(re.search(r"\[\d+\]", res["answer"])) and bool(res["sources"])
        rows.append({"id": q["id"], "q": q, "refused": res["refused"], "cited": cited,
                     "n_sources": len(res["sources"]), "answer": res["answer"]})
        print(f"  {q['id']}: {'REFUSED' if res['refused'] else 'answered'}")
        if res["sources"]:
            time.sleep(delay)  # stay under free-tier rate limits

    ok = [r for r in rows if not r.get("error")]
    ans = [r for r in ok if r["q"]["answerable"]]
    unans = [r for r in ok if not r["q"]["answerable"]]
    answered = [r for r in ans if not r["refused"]]
    lines = [
        f"# End-to-end test ({strategy}, k={k}, threshold={threshold})\n",
        "Runs the full pipeline (retrieval, gate, Gemini) on every question.\n",
        f"- Answerable questions answered: {fmt_frac(len(answered), len(ans))}",
        f"- Answers that cite passages `[n]` and show their sources: "
        f"{fmt_frac(sum(r['cited'] for r in answered), len(answered))}",
        f"- Unanswerable questions correctly refused: "
        f"{fmt_frac(sum(r['refused'] for r in unans), len(unans))}",
    ]
    for kind in ("off_topic", "in_domain_missing"):
        sub = [r for r in unans if r["q"]["kind"] == kind]
        lines.append(f"  - {kind}: {fmt_frac(sum(r['refused'] for r in sub), len(sub))}")
    failed = [r["id"] for r in rows if r.get("error")]
    if failed:
        lines.append(f"- Not evaluated (API errors): {', '.join(failed)}")
    lines += ["", "| id | kind | refused | cited | sources | answer (first 110 chars) |",
              "|---|---|---|---|---|---|"]
    for r in ok:
        snippet = r["answer"].replace("\n", " ").replace("|", "/")[:110]
        lines.append(f"| {r['id']} | {r['q']['kind']} | {'yes' if r['refused'] else 'no'} | "
                     f"{'yes' if r['cited'] else '-'} | {r['n_sources']} | {snippet} |")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--k", type=int, default=TOP_K)
    parser.add_argument("--threshold", type=float, default=DISTANCE_THRESHOLD)
    parser.add_argument("--calibrate", action="store_true", help="choose DISTANCE_THRESHOLD")
    parser.add_argument("--e2e", action="store_true", help="test the full pipeline with Gemini")
    parser.add_argument("--chunks-only", action="store_true", help="chunk integrity, no embeddings")
    parser.add_argument("--strategy", default=DEFAULT_STRATEGY, choices=STRATEGIES,
                        help="strategy for --e2e")
    parser.add_argument("--delay", type=float, default=6.5,
                        help="seconds between Gemini calls in --e2e")
    args = parser.parse_args()

    questions = load_questions()

    if args.calibrate:
        report = calibrate(questions, args.k)
        print(report)
        CALIBRATION_PATH.write_text(
            f"# Threshold calibration (k={args.k})\n{report}\n", encoding="utf-8")
        print(f"\nWritten to {CALIBRATION_PATH.relative_to(ROOT)}")
        return

    if args.e2e:
        report = run_e2e(questions, args.strategy, args.k, args.threshold, args.delay)
        E2E_PATH.write_text(report, encoding="utf-8")
        print("\n" + report)
        print(f"Written to {E2E_PATH.relative_to(ROOT)}")
        return

    from src.chunking import load_documents

    pages = load_documents()
    chunk_info = {s: chunk_report(s, questions, pages) for s in STRATEGIES}

    if args.chunks_only:
        for s, info in chunk_info.items():
            print(f"{s:>10}: {info['summary']} | answers intact "
                  f"{info['intact_n']}/{info['intact_total']}"
                  + (f" | cut in half: {', '.join(info['broken'])}" if info["broken"] else ""))
        return

    results = [evaluate(s, questions, args.k, args.threshold, chunk_info[s])
               for s in STRATEGIES]
    n_ans = results[0]["n_answerable"]
    header = (f"k={args.k}, distance threshold={args.threshold}, {len(questions)} questions "
              f"({n_ans} answerable, {len(questions) - n_ans} unanswerable).")
    body = (f"# Chunking comparison\n\n{header}\n\n## Verdict\n\n{verdict(results)}\n\n"
            f"## Metrics\n\n{tables(results)}\n\n## Per question\n\n{per_question(results)}\n")
    RESULTS_PATH.write_text(body, encoding="utf-8")

    print("\n" + header + "\n")
    print(verdict(results).replace("**", "").replace("`", ""))
    print("\n" + tables(results))
    print(f"\nWritten to {RESULTS_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()