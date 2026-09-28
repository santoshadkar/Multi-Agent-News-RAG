"""Central configuration. Reads environment variables (loaded from .env if
python-dotenv style loading is desired -- kept dependency-free here, so
export vars in your shell or use `set -a; source .env; set +a`).

Also checks Streamlit's secrets store (st.secrets) as a fallback, so the
same code works unchanged when deployed to Streamlit Community Cloud,
where secrets are entered in that platform's web UI rather than a .env
file. See README "Deploying to Streamlit Community Cloud".
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path


def _secret(name: str, default: str = "") -> str:
    """env var first, then st.secrets if running under Streamlit -- checked
    in that order so a local .env always wins over a stale cloud secret."""
    value = os.environ.get(name)
    if value:
        return value
    try:
        import streamlit as st  # only importable/meaningful inside a Streamlit process

        return str(st.secrets.get(name, default))
    except Exception:
        # Not running under Streamlit, or no secrets.toml configured -- fine,
        # just fall through to the default.
        return default


def _ensure_writable_data_dir(path: Path) -> Path:
    """Chroma's SQLite backend needs to write WAL/lock files next to
    chroma.sqlite3 even to *read* it -- so if `path` isn't writable at
    runtime, opening a PersistentClient there fails (surfaces as a
    confusing "Could not connect to tenant default_tenant" error, not an
    obvious permissions error).

    This matters for the Streamlit Community Cloud deployment: the git
    checkout that ships cloud_data/ may not be writable from the running
    app process. I have not confirmed this against Streamlit's current
    docs -- it's a plausible, testable cause for exactly the symptom seen,
    not a confirmed platform fact -- so this is a defensive fallback: if
    `path` isn't writable, copy it once into a writable temp directory and
    use that copy instead. Safe to always run: on a normal writable
    checkout (local dev, GitHub Actions) the write-test below succeeds and
    this is a no-op.
    """
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".write_test"
        probe.write_text("ok")
        probe.unlink()
        return path
    except OSError:
        writable = Path(tempfile.gettempdir()) / "news_rag_data_writable"
        if not writable.exists():
            if path.exists():
                shutil.copytree(path, writable)
            else:
                writable.mkdir(parents=True, exist_ok=True)
        return writable


ROOT_DIR = Path(__file__).resolve().parent.parent
# Overridable so a cloud deployment can point this at a git-tracked
# directory (e.g. a GitHub Action commits fetched data there) instead of
# the default, gitignored local dev directory. Checked via _secret() too,
# in case your Streamlit Cloud setup only exposes it through st.secrets
# rather than as a real env var -- see README.
DATA_DIR = _ensure_writable_data_dir(Path(_secret("DATA_DIR", str(ROOT_DIR / "data"))))

RAW_DB_PATH = str(DATA_DIR / "raw_articles.db")
CHROMA_DIR = str(DATA_DIR / "chroma")
RUN_LOG_PATH = str(DATA_DIR / "run_log.jsonl")
BRIEFING_PATH = str(DATA_DIR / "todays_briefing.json")

COHERE_API_KEY = _secret("COHERE_API_KEY")
NEWSAPI_KEY = _secret("NEWSAPI_KEY")
GNEWS_KEY = _secret("GNEWS_KEY")

# Cohere models. Override via env if your account has different access.
EMBED_MODEL = os.environ.get("COHERE_EMBED_MODEL", "embed-english-v3.0")
RERANK_MODEL = os.environ.get("COHERE_RERANK_MODEL", "rerank-english-v3.0")
CHAT_MODEL = os.environ.get("COHERE_CHAT_MODEL", "command-r-plus-08-2024")

CATEGORIES = ["technology", "finance", "politics"]

# Chunking
CHUNK_MIN_WORDS = 220
CHUNK_MAX_WORDS = 380
CHUNK_OVERLAP_WORDS = 40

# Retrieval
SEMANTIC_CANDIDATES = 40  # candidates pulled from Chroma before hybrid fusion
HYBRID_TOP_K = 12  # survivors of hybrid fusion, sent to the reranker
RERANK_TOP_N = 6  # final chunks handed to the generator
RECENCY_HALF_LIFE_DAYS = 3.0  # recency weight halves every N days
MIN_RELEVANCE_SCORE = 0.15  # below this (post-rerank), treat as "no result"

# Scheduling (used by src/scheduler.py)
FETCH_CRON = os.environ.get("FETCH_CRON", "0 * * * *")  # hourly by default
BRIEFING_CRON = os.environ.get("BRIEFING_CRON", "0 7 * * *")  # 7am daily

REQUEST_TIMEOUT_SECS = 15
FETCH_MAX_RETRIES = 3
USER_AGENT = "news-rag-agent/1.0 (+educational project)"
