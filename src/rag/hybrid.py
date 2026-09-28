"""Steps 2-3 of the RAG pipeline: hybrid (semantic + keyword) retrieval with
a recency boost, followed by a cross-encoder rerank.
"""

from __future__ import annotations

import math
import re
from datetime import date, datetime
from typing import Optional

from rank_bm25 import BM25Okapi

from src import config

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _minmax_normalize(values: list[float]) -> list[float]:
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi - lo < 1e-9:
        return [1.0 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]


def _recency_weight(published_at: Optional[str], as_of: date) -> float:
    """Exponential decay: weight halves every RECENCY_HALF_LIFE_DAYS days.
    Undated chunks (some official-page listings lack a parseable date) get
    a neutral mid-range weight rather than being penalized to zero."""
    if not published_at:
        return 0.5
    try:
        pub_date = datetime.fromisoformat(published_at).date()
    except ValueError:
        return 0.5
    age_days = max((as_of - pub_date).days, 0)
    half_life = config.RECENCY_HALF_LIFE_DAYS
    return math.pow(0.5, age_days / half_life)


def hybrid_rank(
    query: str,
    semantic_results: list[dict],
    lexical_pool: list[dict],
    top_k: int,
    as_of: Optional[date] = None,
    semantic_weight: float = 0.55,
    lexical_weight: float = 0.25,
    recency_weight: float = 0.20,
) -> list[dict]:
    """Combine Chroma's semantic scores with BM25 lexical scores over the
    same (metadata-filtered) candidate pool, then blend in a recency boost.
    `lexical_pool` should be a superset of `semantic_results` (typically:
    every chunk that passed the metadata filter) so BM25 has real corpus
    statistics to work with.
    """
    as_of = as_of or date.today()
    if not semantic_results:
        return []

    corpus = [c["text"] for c in lexical_pool] or [c["text"] for c in semantic_results]
    bm25 = BM25Okapi([_tokenize(t) for t in corpus])
    id_to_lexical_idx = {c["id"]: i for i, c in enumerate(lexical_pool)}

    query_tokens = _tokenize(query)
    bm25_scores_full = bm25.get_scores(query_tokens)

    semantic_scores = [r["semantic_score"] for r in semantic_results]
    lexical_scores = []
    for r in semantic_results:
        idx = id_to_lexical_idx.get(r["id"])
        lexical_scores.append(bm25_scores_full[idx] if idx is not None else 0.0)
    recency_scores = [_recency_weight(r["metadata"].get("published_at"), as_of) for r in semantic_results]

    norm_sem = _minmax_normalize(semantic_scores)
    norm_lex = _minmax_normalize(lexical_scores)
    # Recency is already in [0, 1] by construction; no need to re-normalize.

    combined = []
    for r, s, l, rec in zip(semantic_results, norm_sem, norm_lex, recency_scores):
        score = semantic_weight * s + lexical_weight * l + recency_weight * rec
        combined.append({**r, "hybrid_score": score})

    combined.sort(key=lambda c: c["hybrid_score"], reverse=True)
    return combined[:top_k]
