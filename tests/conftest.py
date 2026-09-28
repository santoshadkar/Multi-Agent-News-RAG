from __future__ import annotations

import sys
from pathlib import Path

import chromadb
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.kb.vector_store import VectorStore
from tests.fakes import FakeCohereClient


@pytest.fixture()
def fake_client() -> FakeCohereClient:
    return FakeCohereClient()


@pytest.fixture()
def vector_store(tmp_path) -> VectorStore:
    client = chromadb.PersistentClient(path=str(tmp_path / "chroma"))
    return VectorStore(client)
