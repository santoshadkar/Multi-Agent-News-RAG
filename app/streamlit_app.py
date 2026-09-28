"""Streamlit chat interface for the news RAG system.

Run with: streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.kb.embeddings import get_cohere_client
from src.kb.vector_store import VectorStore, get_chroma_client
from src.rag.briefing import load_briefing
from src.rag.query_engine import QueryEngine

st.set_page_config(page_title="News Analyst", page_icon="📰", layout="wide")

USER_AVATAR = "🧑‍💼"
ASSISTANT_AVATAR = "📰"

CATEGORY_META = {
    "technology": {"icon": "💻", "color": "#2563eb"},
    "finance": {"icon": "📈", "color": "#059669"},
    "politics": {"icon": "🏛️", "color": "#7c3aed"},
}

EXAMPLE_QUESTIONS = [
    "What did the RBI announce this week and how did markets react?",
    "What's the latest in technology today?",
    "Any major political developments this week?",
]

CUSTOM_CSS = """
<style>
.block-container { padding-top: 2rem; max-width: 1000px; }

.app-header {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    margin-bottom: 0.1rem;
}
.app-header h1 { margin: 0; font-size: 2rem; }
.app-subtitle { color: var(--text-color, #6b7280); font-size: 0.95rem; margin-bottom: 1.3rem; }

.citation-card {
    border: 1px solid rgba(128,128,128,0.25);
    border-radius: 10px;
    padding: 0.55rem 0.8rem;
    margin-bottom: 0.45rem;
    font-size: 0.86rem;
    background: rgba(128,128,128,0.05);
}
.citation-card a { text-decoration: none; font-weight: 600; }
.citation-meta { color: #6b7280; font-size: 0.8rem; margin-top: 0.15rem; }

.cat-badge {
    display: inline-block;
    padding: 0.1rem 0.55rem;
    border-radius: 999px;
    font-size: 0.72rem;
    font-weight: 600;
    color: white;
    margin-bottom: 0.4rem;
}

.briefing-time {
    color: #9ca3af;
    font-size: 0.75rem;
    margin-bottom: 0.6rem;
}

div[data-testid="stChatInput"] textarea { border-radius: 12px; }
</style>
"""


@st.cache_resource
def get_engine() -> QueryEngine:
    client = get_cohere_client()
    store = VectorStore(get_chroma_client())
    return QueryEngine(client, store)


def _format_briefing_time(raw: str) -> str:
    try:
        dt = datetime.fromisoformat(raw)
        return dt.strftime("%b %d, %Y · %H:%M UTC")
    except (TypeError, ValueError):
        return ""


def render_sidebar():
    st.sidebar.markdown("### 🔍 Filters")
    category = st.sidebar.selectbox(
        "Category", options=["All", "technology", "finance", "politics"], index=0,
        format_func=lambda c: c if c == "All" else f"{CATEGORY_META[c]['icon']} {c.capitalize()}",
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
    st.sidebar.markdown("### 📋 Today's Briefing")
    briefing = load_briefing()
    if briefing:
        generated = _format_briefing_time(briefing.get("generated_at", ""))
        if generated:
            st.sidebar.markdown(f'<div class="briefing-time">Generated {generated}</div>', unsafe_allow_html=True)
        for cat, section in briefing.get("sections", {}).items():
            meta = CATEGORY_META.get(cat, {"icon": "📰", "color": "#6b7280"})
            with st.sidebar.expander(f"{meta['icon']} {cat.capitalize()}"):
                st.write(section["summary"])
                for c in section.get("citations", []):
                    st.caption(f"[{c['headline']}]({c['url']})")
    else:
        st.sidebar.info("No briefing generated yet.\n\nRun: `python -m src.cli briefing`", icon="🕐")

    return (None if category == "All" else category), date_from, date_to


def render_citations(citations):
    if not citations:
        return
    st.markdown("**Sources**")
    for c in citations:
        date_str = f" · {c.published_at}" if c.published_at else ""
        st.markdown(
            f'<div class="citation-card">'
            f'<a href="{c.url}" target="_blank">{c.headline}</a>'
            f'<div class="citation-meta">{c.source}{date_str}</div>'
            f"</div>",
            unsafe_allow_html=True,
        )


def render_empty_state():
    st.markdown(
        "Ask a question below, or try one of these:",
    )
    cols = st.columns(len(EXAMPLE_QUESTIONS))
    picked = None
    for col, q in zip(cols, EXAMPLE_QUESTIONS):
        if col.button(q, use_container_width=True):
            picked = q
    return picked


def ask(question: str, category, date_from, date_to):
    st.session_state.history.append({"role": "user", "content": question})
    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(question)

    try:
        engine = get_engine()
    except RuntimeError as e:
        with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
            st.error(str(e))
        return

    with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
        with st.spinner("Searching the knowledge base..."):
            result = engine.answer(
                question,
                category_override=category,
                date_from_override=date_from,
                date_to_override=date_to,
            )
        st.markdown(result.answer.answer)
        render_citations(result.answer.citations)

    st.session_state.history.append({"role": "assistant", "content": result.answer.answer})


def main():
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    st.markdown('<div class="app-header"><h1>📰 News Analyst</h1></div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="app-subtitle">Ask about today\'s technology, finance, or politics news. '
        "Answers are grounded in the collected articles and cite their sources.</div>",
        unsafe_allow_html=True,
    )

    category, date_from, date_to = render_sidebar()

    if "history" not in st.session_state:
        st.session_state.history = []

    for turn in st.session_state.history:
        avatar = USER_AVATAR if turn["role"] == "user" else ASSISTANT_AVATAR
        with st.chat_message(turn["role"], avatar=avatar):
            st.markdown(turn["content"])

    picked_example = None
    if not st.session_state.history:
        picked_example = render_empty_state()

    question = st.chat_input("e.g. What did the RBI announce this week and how did markets react?")
    question = question or picked_example
    if not question:
        return

    ask(question, category, date_from, date_to)
    st.rerun()


if __name__ == "__main__":
    main()
