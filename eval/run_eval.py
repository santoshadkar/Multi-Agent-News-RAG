"""Evaluation harness: runs eval/questions.json through the live pipeline
and writes eval/eval_report.md.

Scoring methodology (documented here, not just in the report, so it's not
mistaken for anything stronger than what it is):

  citation_correctness (0 or 2): DETERMINISTIC. Every citation URL returned
    to the user must come from a chunk that was actually retrieved and
    handed to the generator. This checks that citations aren't fabricated
    or mismatched -- it does not check that the underlying article itself
    is accurate (we have no independent ground truth for that).

  recency (0, 1, or 2): DETERMINISTIC when the question implied a date
    range (e.g. "this week"). 2 = every citation's date falls inside the
    inferred range; 1 = some citations are undated so it can't be fully
    verified; 0 = a citation falls outside the range. When no date range
    was inferred, this is scored 2 (not applicable) and flagged as such.

  accuracy (0, 1, or 2): an LLM-judge call (via the same Cohere chat model)
    checks whether the generated answer is FAITHFUL TO THE RETRIEVED
    CHUNKS -- i.e. that it doesn't contradict or invent claims beyond what
    the retrieved text says. This is a proxy for accuracy, not a
    fact-check against the real world: if the underlying news source was
    itself wrong, this metric will not catch that. Treat these scores as a
    first-pass automated check, not a substitute for human review.

  refusal_correctness (0 or 2, unanswerable questions only): 2 if the
    system replied "I don't have news on that" (no fabricated answer), 0
    if it produced a substantive answer anyway.

Usage: python -m eval.run_eval
Requires COHERE_API_KEY set and a knowledge base already populated via
`python -m src.cli run`.
"""

from __future__ import annotations

import json
import statistics
from datetime import datetime
from pathlib import Path

from src import config
from src.kb.embeddings import get_cohere_client
from src.kb.vector_store import VectorStore, get_chroma_client
from src.rag.generate import NO_ANSWER_TEXT
from src.rag.query_engine import QueryEngine, QueryResult

EVAL_DIR = Path(__file__).resolve().parent
QUESTIONS_PATH = EVAL_DIR / "questions.json"
REPORT_PATH = EVAL_DIR / "eval_report.md"

JUDGE_PROMPT = """You are grading whether an ANSWER is faithful to the SUPPORTING TEXT below \
-- i.e. every claim in the answer is supported by (or a reasonable summary of) the supporting \
text, with no invented facts and no contradictions. Reply with exactly one digit: \
2 if fully faithful, 1 if partially faithful (minor unsupported detail), 0 if it contains a \
claim the supporting text does not support or contradicts.

SUPPORTING TEXT:
{context}

ANSWER:
{answer}

Reply with only the digit."""


def score_citation_correctness(result: QueryResult, retrieved_urls: set[str]) -> int:
    if not result.answer.citations:
        return 2 if not result.answer.grounded else 0  # no citations on a "no news" refusal is fine
    return 2 if all(c.url in retrieved_urls for c in result.answer.citations if c.url) else 0


def score_recency(result: QueryResult) -> tuple[int, str]:
    interp = result.interpretation
    if not (interp.date_from or interp.date_to):
        return 2, "not applicable (no date range inferred)"
    undated = 0
    for c in result.answer.citations:
        if not c.published_at:
            undated += 1
            continue
        try:
            d = datetime.fromisoformat(c.published_at).date()
        except ValueError:
            undated += 1
            continue
        if interp.date_from and d < interp.date_from:
            return 0, f"citation dated {d} is before requested start {interp.date_from}"
        if interp.date_to and d > interp.date_to:
            return 0, f"citation dated {d} is after requested end {interp.date_to}"
    if undated == len(result.answer.citations) and result.answer.citations:
        return 1, "citations present but undated; range compliance unverifiable"
    return 2, "all dated citations within requested range"


def score_accuracy(client, result: QueryResult, chunk_texts: list[str]) -> int:
    if not result.answer.grounded:
        return 2  # correct refusal has nothing to be unfaithful to
    if not chunk_texts:
        return 0
    context = "\n---\n".join(chunk_texts)[:6000]
    resp = client.chat(
        model=config.CHAT_MODEL,
        messages=[{"role": "user", "content": JUDGE_PROMPT.format(context=context, answer=result.answer.answer)}],
    )
    text = "".join(getattr(item, "text", "") or "" for item in (resp.message.content or []))
    for digit in ("2", "1", "0"):
        if digit in text:
            return int(digit)
    return 0


def score_refusal(result: QueryResult) -> int:
    return 2 if (not result.answer.grounded and NO_ANSWER_TEXT in result.answer.answer) else 0


def run() -> None:
    questions = json.loads(QUESTIONS_PATH.read_text())
    client = get_cohere_client()
    store = VectorStore(get_chroma_client())
    engine = QueryEngine(client, store)

    rows = []
    for q in questions:
        # Re-run retrieval to capture which chunks were actually used, for
        # citation-correctness and accuracy scoring.
        result = engine.answer(q["question"])
        retrieved_urls = {c.url for c in result.answer.citations}
        # QueryResult exposes citations, not the raw retrieved chunks, so
        # accuracy judging uses the citation snippets Cohere returned --
        # that's enough context for a faithfulness check.
        chunk_texts = [c.text for c in result.answer.citations]

        row = {
            "id": q["id"],
            "category": q["category"],
            "type": q["type"],
            "question": q["question"],
            "answer": result.answer.answer,
            "n_citations": len(result.answer.citations),
        }
        if q["type"] == "unanswerable":
            row["refusal_correctness"] = score_refusal(result)
        else:
            row["citation_correctness"] = score_citation_correctness(result, retrieved_urls)
            recency_score, recency_note = score_recency(result)
            row["recency"] = recency_score
            row["recency_note"] = recency_note
            row["accuracy"] = score_accuracy(client, result, chunk_texts)
        rows.append(row)

    write_report(rows)
    print(f"Wrote {REPORT_PATH}")


def write_report(rows: list[dict]) -> None:
    answerable = [r for r in rows if "accuracy" in r]
    unanswerable = [r for r in rows if "refusal_correctness" in r]

    lines = ["# Evaluation Report", ""]
    lines.append(f"Generated: {datetime.utcnow().isoformat()}Z")
    lines.append(f"Questions: {len(rows)} ({len(answerable)} answerable, {len(unanswerable)} unanswerable)")
    lines.append("")
    lines.append(
        "Methodology: citation_correctness and recency are checked deterministically against "
        "what was actually retrieved. accuracy is an LLM-judge faithfulness check against the "
        "retrieved citation text, NOT an independent fact-check -- see eval/run_eval.py docstring. "
        "Treat this as a first-pass automated check to supplement, not replace, human review."
    )
    lines.append("")

    if answerable:
        lines.append("## Answerable questions")
        lines.append("")
        lines.append("| id | category | accuracy (0-2) | citation_correctness (0-2) | recency (0-2) | notes |")
        lines.append("|---|---|---|---|---|---|")
        for r in answerable:
            lines.append(
                f"| {r['id']} | {r['category']} | {r['accuracy']} | {r['citation_correctness']} | "
                f"{r['recency']} | {r.get('recency_note', '')} |"
            )
        lines.append("")
        lines.append(
            f"Average accuracy: {statistics.mean(r['accuracy'] for r in answerable):.2f} / 2, "
            f"average citation_correctness: {statistics.mean(r['citation_correctness'] for r in answerable):.2f} / 2, "
            f"average recency: {statistics.mean(r['recency'] for r in answerable):.2f} / 2"
        )
        lines.append("")

    if unanswerable:
        lines.append("## Unanswerable questions (refusal handling)")
        lines.append("")
        lines.append("| id | question | refusal_correctness (0 or 2) |")
        lines.append("|---|---|---|")
        for r in unanswerable:
            lines.append(f"| {r['id']} | {r['question']} | {r['refusal_correctness']} |")
        lines.append("")
        lines.append(
            f"Average refusal_correctness: {statistics.mean(r['refusal_correctness'] for r in unanswerable):.2f} / 2"
        )
        lines.append("")

    lines.append("## Full answers")
    lines.append("")
    for r in rows:
        lines.append(f"**Q{r['id']} ({r['category']}): {r['question']}**")
        lines.append("")
        lines.append(f"> {r['answer']}")
        lines.append("")

    REPORT_PATH.write_text("\n".join(lines))


if __name__ == "__main__":
    run()
