"""Step 1 of the RAG pipeline: infer a time range and/or category filter
from the raw question, using simple rules rather than an LLM call.

Rule-based interpretation is deliberately used here instead of asking the
chat model to classify the query: it's deterministic, free, fast, and easy
to unit-test. It will miss creative phrasing an LLM classifier would catch
-- see README "Extending" for how to swap in an LLM-based classifier.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Optional

import yaml

from src import config

with open(config.ROOT_DIR / "sources.yaml") as f:
    _SOURCES_CFG = yaml.safe_load(f)

_CATEGORY_KEYWORDS = {
    cat: [k.lower() for k in _SOURCES_CFG.get(cat, {}).get("relevance_keywords", [])]
    for cat in config.CATEGORIES
}
# A few extra everyday synonyms that help category inference without
# touching the fetcher's own relevance keyword list.
_CATEGORY_KEYWORDS["technology"] += ["tech", "technology"]
_CATEGORY_KEYWORDS["finance"] += ["finance", "financial", "money"]
_CATEGORY_KEYWORDS["politics"] += ["politics", "political"]


@dataclass
class QueryInterpretation:
    category: Optional[str]
    date_from: Optional[date]
    date_to: Optional[date]
    cleaned_question: str


def _infer_category(question: str) -> Optional[str]:
    q = question.lower()
    scores = {cat: sum(1 for kw in kws if kw in q) for cat, kws in _CATEGORY_KEYWORDS.items()}
    best_cat = max(scores, key=scores.get)
    if scores[best_cat] == 0:
        return None
    # Require a clear winner; a tie or near-tie means "don't filter".
    runner_up = max((v for k, v in scores.items() if k != best_cat), default=0)
    if scores[best_cat] <= runner_up:
        return None
    return best_cat


def _infer_date_range(question: str, today: date) -> tuple[Optional[date], Optional[date]]:
    q = question.lower()
    if "today" in q:
        return today, today
    if "yesterday" in q:
        y = today - timedelta(days=1)
        return y, y
    if "this week" in q:
        return today - timedelta(days=today.weekday()), today
    if "last week" in q:
        start_this_week = today - timedelta(days=today.weekday())
        return start_this_week - timedelta(days=7), start_this_week - timedelta(days=1)
    if "this month" in q:
        return today.replace(day=1), today
    if "last month" in q:
        first_of_this_month = today.replace(day=1)
        last_month_end = first_of_this_month - timedelta(days=1)
        return last_month_end.replace(day=1), last_month_end
    match = re.search(r"last (\d+) days?", q)
    if match:
        n = int(match.group(1))
        return today - timedelta(days=n), today
    return None, None


def interpret_query(question: str, today: Optional[date] = None) -> QueryInterpretation:
    today = today or date.today()
    category = _infer_category(question)
    date_from, date_to = _infer_date_range(question, today)
    return QueryInterpretation(
        category=category, date_from=date_from, date_to=date_to, cleaned_question=question.strip()
    )
