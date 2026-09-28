"""Lightweight entity extraction and tagging.

This uses a regex heuristic (runs of capitalized words) rather than a full
NER model (spaCy, etc.) so the pipeline has no extra model-download
dependency. It's deliberately simple -- see README "Extending" for how to
swap in spaCy's `en_core_web_sm` NER for higher recall/precision.
"""

from __future__ import annotations

import re

_CAP_RUN_RE = re.compile(r"\b([A-Z][a-zA-Z&.'-]*(?:\s+[A-Z][a-zA-Z&.'-]*){0,3})\b")

_STOPWORD_STARTS = {
    "The", "A", "An", "This", "That", "It", "In", "On", "At", "For", "With",
    "And", "But", "Or", "As", "Is", "Are", "Was", "Were", "Be", "By", "To",
    "From", "Of", "New", "How", "Why", "What", "When", "Where",
}


def extract_entities(text: str, max_entities: int = 12) -> list[str]:
    candidates = _CAP_RUN_RE.findall(text)
    seen: dict[str, int] = {}
    for c in candidates:
        c = c.strip()
        first_word = c.split(" ")[0]
        if len(c) < 3 or first_word in _STOPWORD_STARTS:
            continue
        seen[c] = seen.get(c, 0) + 1
    ranked = sorted(seen.items(), key=lambda kv: (-kv[1], kv[0]))
    return [name for name, _ in ranked[:max_entities]]
