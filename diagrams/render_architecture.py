"""Renders diagrams/architecture.png -- a static picture of the same
architecture described in diagrams/architecture.mmd, using matplotlib so
no external Mermaid renderer or network call is required.

Run: python diagrams/render_architecture.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

OUT_PATH = Path(__file__).resolve().parent / "architecture.png"


def box(ax, xy, w, h, text, fc="#eef3fb", ec="#2c3e50", fontsize=9):
    rect = mpatches.FancyBboxPatch(
        xy, w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
        linewidth=1.3, edgecolor=ec, facecolor=fc,
    )
    ax.add_patch(rect)
    cx, cy = xy[0] + w / 2, xy[1] + h / 2
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fontsize, wrap=True)
    return (cx, xy[1] + h), (cx, xy[1])  # (top-mid, bottom-mid)


def arrow(ax, p1, p2, color="#555555"):
    ax.annotate(
        "", xy=p2, xytext=p1,
        arrowprops=dict(arrowstyle="-|>", color=color, lw=1.4, shrinkA=2, shrinkB=2),
    )


def main() -> None:
    fig, ax = plt.subplots(figsize=(11, 13))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 15)
    ax.axis("off")
    fig.suptitle("News RAG System — Architecture", fontsize=14, fontweight="bold")

    sched_top, sched_bot = box(ax, (3.5, 13.6), 3.0, 0.7, "Scheduler\n(APScheduler cron)")
    orch_top, orch_bot = box(ax, (3.5, 12.4), 3.0, 0.7, "Orchestrator")
    arrow(ax, sched_bot, orch_top)

    fetcher_y = 11.0
    tech_top, tech_bot = box(ax, (0.5, fetcher_y), 2.2, 0.9, "Technology\nAgent", fc="#eafaf1")
    fin_top, fin_bot = box(ax, (3.4, fetcher_y), 2.2, 0.9, "Finance\nAgent", fc="#eafaf1")
    pol_top, pol_bot = box(ax, (6.3, fetcher_y), 2.2, 0.9, "Politics\nAgent", fc="#eafaf1")
    for b in (tech_top, fin_top, pol_top):
        arrow(ax, orch_bot, b)

    raw_top, raw_bot = box(ax, (3.0, 9.6), 4.0, 0.8, "Raw article store\n(SQLite, URL-deduped)", fc="#fdf3e7")
    for b in (tech_bot, fin_bot, pol_bot):
        arrow(ax, b, raw_top)

    proc_top, proc_bot = box(
        ax, (2.3, 8.2), 5.4, 0.9,
        "Processing Agent\ndedupe → clean → summarize → tag → chunk",
        fc="#fdf3e7",
    )
    arrow(ax, raw_bot, proc_top)

    emb_top, emb_bot = box(ax, (3.0, 7.0), 4.0, 0.7, "Cohere Embeddings\n(embed-english-v3.0)", fc="#f4ecfb")
    arrow(ax, proc_bot, emb_top)

    kb_top, kb_bot = box(
        ax, (2.6, 5.7), 4.8, 0.9,
        "Knowledge Base\nChromaDB + metadata (date / category / source / url)",
        fc="#f4ecfb",
    )
    arrow(ax, emb_bot, kb_top)

    # RAG query engine steps
    rag_x, rag_w = 1.0, 8.0
    ax.add_patch(
        mpatches.FancyBboxPatch(
            (rag_x, 1.5), rag_w, 3.7, boxstyle="round,pad=0.05,rounding_size=0.08",
            linewidth=1.0, edgecolor="#999999", facecolor="none", linestyle="dashed",
        )
    )
    ax.text(rag_x + rag_w / 2, 5.35, "RAG Query Engine", ha="center", fontsize=10, fontweight="bold")

    steps = [
        "1. Interpret question\n(date range + category)",
        "2. Hybrid retrieve\n(semantic + BM25, filtered)",
        "3. Rerank\n(recency-weighted)",
        "4. Generate\n(grounded on retrieved chunks)",
        "5. Cite sources or refuse\n“I don't have news on that.”",
    ]
    step_w = 1.5
    xs = [1.3 + i * 1.6 for i in range(5)]
    prev_bot = None
    tops_bots = []
    for x, text in zip(xs, steps):
        top, bot = box(ax, (x, 2.3), step_w, 1.7, text, fc="#eef3fb", fontsize=7.5)
        tops_bots.append((top, bot))
    for i in range(len(tops_bots) - 1):
        arrow(ax, (xs[i] + step_w, 3.15), (xs[i + 1], 3.15))

    arrow(ax, kb_bot, ((xs[1] + step_w / 2), 4.0))

    ui_top, ui_bot = box(
        ax, (2.8, 0.2), 4.4, 0.9,
        "Chat Interface (Streamlit)\ncategory filter · date picker · Today's Briefing",
        fc="#eafaf1",
    )
    arrow(ax, (xs[4] + step_w / 2, 2.3), ui_top)

    plt.tight_layout()
    fig.savefig(OUT_PATH, dpi=160)
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
