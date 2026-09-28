"""Orchestrates the full 5-step RAG pipeline described in the brief:

  1. interpret the question (time range / category)
  2. hybrid retrieve (semantic + keyword)
  3. rerank, weighted toward recent news
  4. generate an answer using only the retrieved content
  5. cite sources with links and dates; refuse if nothing relevant was found
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Optional

import cohere

from src import config
from src.kb.embeddings import EmbeddingClient
from src.kb.vector_store import VectorStore
from src.rag.generate import RagAnswer, generate_answer
from src.rag.hybrid import hybrid_rank
from src.rag.interpret import QueryInterpretation, interpret_query
from src.rag.rerank import rerank_chunks


@dataclass
class QueryResult:
    interpretation: QueryInterpretation
    answer: RagAnswer


class QueryEngine:
    def __init__(
        self,
        cohere_client: cohere.ClientV2,
        vector_store: VectorStore,
        embedding_client: Optional[EmbeddingClient] = None,
    ):
        self.client = cohere_client
        self.store = vector_store
        self.embedder = embedding_client or EmbeddingClient(cohere_client)

    def answer(
        self,
        question: str,
        category_override: Optional[str] = None,
        date_from_override: Optional[date] = None,
        date_to_override: Optional[date] = None,
        as_of: Optional[date] = None,
    ) -> QueryResult:
        interp = interpret_query(question, today=as_of)
        category = category_override or interp.category
        date_from = date_from_override or interp.date_from
        date_to = date_to_override or interp.date_to

        where = VectorStore.build_where(category=category, date_from=date_from, date_to=date_to)

        query_embedding = self.embedder.embed_query(question)
        semantic_results = self.store.semantic_search(
            query_embedding, n_results=config.SEMANTIC_CANDIDATES, where=where
        )

        if not semantic_results:
            # Metadata filter may have been too narrow (e.g. a category
            # guessed wrong, or no articles yet for that date). Retry once
            # with no filter before giving up, so a bad filter guess
            # doesn't manufacture a false "no news" refusal.
            semantic_results = self.store.semantic_search(
                query_embedding, n_results=config.SEMANTIC_CANDIDATES, where=None
            )
            where = None

        lexical_pool = self.store.all_chunks_in(where=where) if where else semantic_results

        hybrid = hybrid_rank(
            question, semantic_results, lexical_pool, top_k=config.HYBRID_TOP_K, as_of=as_of
        )
        reranked = rerank_chunks(self.client, question, hybrid, top_n=config.RERANK_TOP_N)
        relevant = [c for c in reranked if c.get("rerank_score", 0.0) >= config.MIN_RELEVANCE_SCORE]

        answer = generate_answer(
            self.client,
            question,
            relevant,
            has_category_filter=category is not None,
            has_date_filter=date_from is not None or date_to is not None,
        )
        return QueryResult(interpretation=interp, answer=answer)
