"""Chroma-backed knowledge base: chunk text + embedding + rich metadata.

Metadata (category, source, url, date, headline) is stored alongside each
chunk so queries can be filtered, e.g. "only finance, last 7 days" -- this
is the `where` filter Chroma supports natively on metadata fields.
"""

from __future__ import annotations

from datetime import date
from typing import Optional

import chromadb

from src import config
from src.models import Chunk

COLLECTION_NAME = "news_chunks"


def get_chroma_client() -> chromadb.ClientAPI:
    return chromadb.PersistentClient(path=config.CHROMA_DIR)


class VectorStore:
    def __init__(self, client: chromadb.ClientAPI):
        self.client = client
        self.collection = client.get_or_create_collection(
            name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
        )

    def add_chunks(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if not chunks:
            return
        self.collection.upsert(
            ids=[c.id for c in chunks],
            embeddings=embeddings,
            documents=[c.text for c in chunks],
            metadatas=[
                {
                    "article_id": c.article_id,
                    "headline": c.headline,
                    "url": c.url,
                    "source": c.source,
                    "category": c.category,
                    "published_at": c.published_at or "",
                    # Chroma's $gte/$lte only accept numeric operands, not
                    # ISO date strings, so a sortable integer (date
                    # ordinal) is stored alongside the human-readable
                    # string for range filtering.
                    "published_at_ord": date.fromisoformat(c.published_at).toordinal()
                    if c.published_at
                    else 0,
                    "chunk_index": c.chunk_index,
                }
                for c in chunks
            ],
        )

    def count(self) -> int:
        return self.collection.count()

    @staticmethod
    def build_where(
        category: Optional[str] = None,
        date_from: Optional[date] = None,
        date_to: Optional[date] = None,
    ) -> Optional[dict]:
        clauses = []
        if category:
            clauses.append({"category": category})
        if date_from:
            clauses.append({"published_at_ord": {"$gte": date_from.toordinal()}})
        if date_to:
            clauses.append({"published_at_ord": {"$lte": date_to.toordinal()}})
        if not clauses:
            return None
        if len(clauses) == 1:
            return clauses[0]
        return {"$and": clauses}

    def semantic_search(
        self,
        query_embedding: list[float],
        n_results: int,
        where: Optional[dict] = None,
    ) -> list[dict]:
        res = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        out = []
        ids = res.get("ids", [[]])[0]
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0]
        for i, doc, meta, dist in zip(ids, docs, metas, dists):
            out.append(
                {
                    "id": i,
                    "text": doc,
                    "metadata": meta,
                    # Chroma cosine "distance" -> similarity in [0, 1]-ish.
                    "semantic_score": max(0.0, 1.0 - dist / 2.0),
                }
            )
        return out

    def all_chunks_in(self, where: Optional[dict] = None, limit: int = 5000) -> list[dict]:
        """Used by the BM25 side of hybrid search: pull the metadata-filtered
        candidate pool's raw text so it can be scored lexically."""
        res = self.collection.get(where=where, limit=limit, include=["documents", "metadatas"])
        out = []
        for i, doc, meta in zip(res.get("ids", []), res.get("documents", []), res.get("metadatas", [])):
            out.append({"id": i, "text": doc, "metadata": meta})
        return out
