"""Boilerplate stripping and text normalization."""

from __future__ import annotations

import re

_BOILERPLATE_PATTERNS = [
    r"read more.*$",
    r"click here to.*$",
    r"subscribe to our newsletter.*$",
    r"follow us on (twitter|facebook|instagram).*$",
    r"advertisement\s*",
    r"^\(reuters\)\s*-?\s*",
    r"share this article.*$",
    r"all rights reserved\.?",
]
_COMPILED = [re.compile(p, re.IGNORECASE) for p in _BOILERPLATE_PATTERNS]


def clean_body(raw_body: str) -> str:
    text = raw_body or ""
    text = re.sub(r"\s+", " ", text).strip()
    for pattern in _COMPILED:
        text = pattern.sub("", text).strip()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extractive_summary(clean_text: str, max_sentences: int = 3) -> str:
    """Cheap, deterministic, offline summary: first N sentences.

    This is intentionally simple so the pipeline works with zero extra API
    calls. Swap in an LLM-based abstractive summary (e.g. a short Cohere
    chat call) if you want tighter summaries -- see README "Extending".
    """
    sentences = re.split(r"(?<=[.!?])\s+", clean_text)
    return " ".join(sentences[:max_sentences]).strip()
