"""Near-duplicate detection for stories covered by many outlets.

URL-exact dedupe already happens in RawStore (UNIQUE constraint). This
module catches the "same story, 20 outlets, 20 different URLs" case using
shingled Jaccard similarity over the headline + first part of the body --
cheap, deterministic, no embedding calls needed at this stage.
"""

from __future__ import annotations

import re
from src.models import RawArticle

_WORD_RE = re.compile(r"[a-z0-9]+")


def _shingles(text: str, k: int = 3) -> set[str]:
    words = _WORD_RE.findall(text.lower())
    if len(words) < k:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i : i + k]) for i in range(len(words) - k + 1)}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def dedupe_near_duplicates(
    articles: list[RawArticle], threshold: float = 0.6
) -> list[RawArticle]:
    """Greedy near-dup filter: keeps the earliest-published copy of a story
    (or first-seen, if dates are missing) and drops later copies whose
    headline+snippet shingles overlap the kept copy above `threshold`."""

    def sort_key(a: RawArticle):
        # Earliest published_at first; articles with no date sort last so
        # they don't wrongly "win" as the canonical copy.
        return (a.published_at is None, a.published_at or a.fetched_at)

    ordered = sorted(articles, key=sort_key)
    kept: list[RawArticle] = []
    kept_shingles: list[set[str]] = []

    for art in ordered:
        text = f"{art.headline} {art.body[:400]}"
        sh = _shingles(text)
        is_dup = any(jaccard(sh, existing) >= threshold for existing in kept_shingles)
        if not is_dup:
            kept.append(art)
            kept_shingles.append(sh)
    return kept
