from src.kb.embeddings import EmbeddingClient
from src.models import Chunk
from src.rag.generate import NO_ANSWER_TEXT
from src.rag.query_engine import QueryEngine


def _index_sample_chunk(fake_client, vector_store):
    embedder = EmbeddingClient(fake_client)
    chunk = Chunk(
        id="c1_0",
        article_id="c1",
        text="The Reserve Bank of India cut the repo rate by 25 basis points on Wednesday, "
        "and markets reacted positively with the Sensex rising 200 points.",
        headline="RBI cuts repo rate, markets rally",
        url="https://example.com/rbi-rate-cut",
        source="Example Finance Daily",
        category="finance",
        published_at="2026-03-01",
        chunk_index=0,
    )
    embeddings = embedder.embed_documents([chunk.text])
    vector_store.add_chunks([chunk], embeddings)
    return chunk


def test_query_engine_returns_grounded_answer_with_citation(fake_client, vector_store):
    _index_sample_chunk(fake_client, vector_store)
    engine = QueryEngine(fake_client, vector_store)

    result = engine.answer("What did the RBI announce this week and how did markets react?")

    assert result.answer.grounded is True
    assert len(result.answer.citations) >= 1
    assert result.answer.citations[0].url == "https://example.com/rbi-rate-cut"


def test_query_engine_refuses_on_empty_knowledge_base(fake_client, vector_store):
    engine = QueryEngine(fake_client, vector_store)
    result = engine.answer("What did the RBI announce this week?")
    assert result.answer.grounded is False
    assert result.answer.answer == NO_ANSWER_TEXT


def test_category_filter_is_applied_when_data_exists_for_it(fake_client, vector_store):
    chunk = _index_sample_chunk(fake_client, vector_store)
    engine = QueryEngine(fake_client, vector_store)

    result = engine.answer("Tell me about recent news.", category_override="finance")
    assert result.answer.grounded is True
    assert result.answer.citations[0].url == chunk.url
