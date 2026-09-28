"""Steps 4-5 of the RAG pipeline: grounded generation with citations.

Uses Cohere chat v2's native `documents` + `citation_options` support
rather than stuffing sources into the prompt as plain text: the model
returns structured citations (start/end offsets into its own answer, plus
which document ids support each span), which we map back to our chunk
metadata for display -- far more reliable than asking the model to type
out "[Source: ...]" itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cohere

from src import config
from src.rag.prompts import SYSTEM_PROMPT, build_user_turn


@dataclass
class Citation:
    text: str
    headline: str
    url: str
    source: str
    published_at: str | None


@dataclass
class RagAnswer:
    answer: str
    citations: list[Citation] = field(default_factory=list)
    grounded: bool = True
    chunks_used: int = 0


NO_ANSWER_TEXT = "I don't have news on that."


def generate_answer(
    client: cohere.ClientV2,
    question: str,
    chunks: list[dict],
    has_category_filter: bool = False,
    has_date_filter: bool = False,
) -> RagAnswer:
    if not chunks:
        return RagAnswer(answer=NO_ANSWER_TEXT, citations=[], grounded=False, chunks_used=0)

    documents = [
        cohere.Document(
            id=chunk["id"],
            data={
                "text": chunk["text"],
                "headline": chunk["metadata"].get("headline", ""),
                "url": chunk["metadata"].get("url", ""),
                "source": chunk["metadata"].get("source", ""),
                "published_at": chunk["metadata"].get("published_at", ""),
            },
        )
        for chunk in chunks
    ]
    chunk_by_id = {c["id"]: c for c in chunks}

    resp = client.chat(
        model=config.CHAT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_turn(question, has_category_filter, has_date_filter)},
        ],
        documents=documents,
        citation_options={"mode": "ACCURATE"},
    )

    answer_text = _extract_text(resp)
    citations = _extract_citations(resp, chunk_by_id)
    grounded = answer_text.strip().rstrip(".") != NO_ANSWER_TEXT.rstrip(".")

    return RagAnswer(answer=answer_text, citations=citations, grounded=grounded, chunks_used=len(chunks))


def _extract_text(resp) -> str:
    parts = []
    for item in getattr(resp.message, "content", None) or []:
        text = getattr(item, "text", None)
        if text:
            parts.append(text)
    return "".join(parts).strip() or NO_ANSWER_TEXT


def _extract_citations(resp, chunk_by_id: dict) -> list[Citation]:
    citations: list[Citation] = []
    seen_urls: set[str] = set()
    raw_citations = getattr(resp.message, "citations", None) or []
    for c in raw_citations:
        for source in getattr(c, "sources", None) or []:
            doc = getattr(source, "document", None) or {}
            url = doc.get("url", "")
            if url and url in seen_urls:
                continue
            if url:
                seen_urls.add(url)
            citations.append(
                Citation(
                    text=c.text,
                    headline=doc.get("headline", ""),
                    url=url,
                    source=doc.get("source", ""),
                    published_at=doc.get("published_at") or None,
                )
            )
    return citations
