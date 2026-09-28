from src.rag.generate import NO_ANSWER_TEXT, generate_answer


def test_no_chunks_returns_refusal_without_calling_client(fake_client):
    result = generate_answer(fake_client, "What is the weather?", chunks=[])
    assert result.answer == NO_ANSWER_TEXT
    assert result.grounded is False
    assert result.citations == []
    assert fake_client.chat_calls == []  # short-circuited before any API call


def test_chunks_present_produces_grounded_answer_with_citation(fake_client):
    chunks = [
        {
            "id": "chunk_1",
            "text": "The RBI cut the repo rate by 25 basis points on Wednesday.",
            "metadata": {
                "headline": "RBI cuts repo rate",
                "url": "https://example.com/rbi-cut",
                "source": "Example News",
                "published_at": "2026-03-01",
            },
        }
    ]
    result = generate_answer(fake_client, "What did the RBI announce?", chunks)
    assert result.grounded is True
    assert result.chunks_used == 1
    assert len(result.citations) == 1
    citation = result.citations[0]
    assert citation.url == "https://example.com/rbi-cut"
    assert citation.headline == "RBI cuts repo rate"
    assert citation.published_at == "2026-03-01"


def test_client_receives_documents_with_expected_ids(fake_client):
    chunks = [
        {
            "id": "chunk_42",
            "text": "Some article text.",
            "metadata": {"headline": "H", "url": "https://x.com", "source": "S", "published_at": "2026-01-01"},
        }
    ]
    generate_answer(fake_client, "question", chunks)
    sent_docs = fake_client.chat_calls[0]["documents"]
    assert sent_docs[0].id == "chunk_42"
