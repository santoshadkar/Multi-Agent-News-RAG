from datetime import date, timedelta

from src.rag.hybrid import hybrid_rank


def make_chunk(id_, text, published_at, semantic_score):
    return {
        "id": id_,
        "text": text,
        "metadata": {"published_at": published_at},
        "semantic_score": semantic_score,
    }


def test_recent_chunk_outranks_older_chunk_with_similar_semantic_score():
    today = date(2026, 6, 15)
    old_chunk = make_chunk("old", "central bank interest rate decision", (today - timedelta(days=30)).isoformat(), 0.80)
    new_chunk = make_chunk("new", "central bank interest rate decision", (today - timedelta(days=1)).isoformat(), 0.80)

    ranked = hybrid_rank(
        "central bank interest rate decision",
        semantic_results=[old_chunk, new_chunk],
        lexical_pool=[old_chunk, new_chunk],
        top_k=2,
        as_of=today,
    )
    assert ranked[0]["id"] == "new"


def test_top_k_is_respected():
    today = date(2026, 6, 15)
    chunks = [make_chunk(str(i), f"story number {i}", today.isoformat(), 0.5) for i in range(10)]
    ranked = hybrid_rank("story", chunks, chunks, top_k=3, as_of=today)
    assert len(ranked) == 3


def test_empty_semantic_results_returns_empty():
    assert hybrid_rank("anything", [], [], top_k=5) == []
