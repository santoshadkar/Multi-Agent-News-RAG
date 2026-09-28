"""Streamlit chat interface for the news RAG system.

Run with: streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config
from src.kb.embeddings import get_cohere_client
from src.kb.vector_store import VectorStore, get_chroma_client
from src.rag.briefing import load_briefing
from src.rag.query_engine import QueryEngine

st.set_page_config(page_title="News Analyst", page_icon="📰", layout="wide")


@st.cache_resource
def get_engine() -> QueryEngine:
    client = get_cohere_client()
    store = VectorStore(get_chroma_client())
    return QueryEngine(client, store)


def render_sidebar():
    st.sidebar.title("Filters")
    category = st.sidebar.selectbox(
        "Category", options=["All", "technology", "finance", "politics"], index=0
    )
    st.sidebar.caption("Leave on 'All' to let the assistant infer the category from your question.")

    use_date_range = st.sidebar.checkbox("Filter by date range", value=False)
    date_from, date_to = None, None
    if use_date_range:
        default_from = date.today() - timedelta(days=7)
        picked = st.sidebar.date_input("Date range", value=(default_from, date.today()))
        if isinstance(picked, tuple) and len(picked) == 2:
            date_from, date_to = picked

    st.sidebar.divider()
    with st.sidebar.expander("🔧 Diagnostics", expanded=False):
        st.caption(f"DATA_DIR: `{config.DATA_DIR}`")
        st.caption(f"CHROMA_DIR: `{config.CHROMA_DIR}`")
        st.caption(f"CHROMA_DIR exists: {Path(config.CHROMA_DIR).exists()}")
        try:
            n = VectorStore(get_chroma_client()).count()
            st.caption(f"Chunks in collection: {n}")
        except Exception as e:  # noqa: BLE001 -- diagnostics must never crash the app
            st.caption(f"Count check failed: {e}")

    st.sidebar.divider()
    briefing = load_briefing()
    if briefing:
        st.sidebar.subheader("📋 Today's Briefing")
        for cat, section in briefing.get("sections", {}).items():
            with st.sidebar.expander(cat.capitalize()):
                st.write(section["summary"])
                for c in section.get("citations", []):
                    st.caption(f"[{c['headline']}]({c['url']})")
    else:
        st.sidebar.caption("No briefing generated yet. Run: python -m src.cli briefing")

    return (None if category == "All" else category), date_from, date_to


def main():
    st.title("📰 News Analyst")
    st.caption(
        "Ask about today's technology, finance, or politics news. Answers are grounded in "
        "the collected articles and cite their sources."
    )

    category, date_from, date_to = render_sidebar()

    if "history" not in st.session_state:
        st.session_state.history = []

    for turn in st.session_state.history:
        with st.chat_message(turn["role"]):
            st.markdown(turn["content"])

    question = st.chat_input("e.g. What did the RBI announce this week and how did markets react?")
    if not question:
        return

    st.session_state.history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    try:
        engine = get_engine()
    except RuntimeError as e:
        with st.chat_message("assistant"):
            st.error(str(e))
        return

    with st.chat_message("assistant"):
        with st.spinner("Searching the knowledge base..."):
            result = engine.answer(
                question,
                category_override=category,
                date_from_override=date_from,
                date_to_override=date_to,
            )
        st.markdown(result.answer.answer)
        if result.answer.citations:
            st.markdown("**Sources:**")
            for c in result.answer.citations:
                date_str = f" ({c.published_at})" if c.published_at else ""
                st.markdown(f"- [{c.headline}]({c.url}){date_str} — {c.source}")

    st.session_state.history.append({"role": "assistant", "content": result.answer.answer})


if __name__ == "__main__":
    main()
