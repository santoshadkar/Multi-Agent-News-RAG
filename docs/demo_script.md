# 10-minute demo script

A suggested walkthrough for presenting this project. Timings are
approximate; adjust to your audience (technical vs. business).

## 1. Problem framing (1 min)

"Business owners can't read hundreds of news sources a day. General
chatbots either don't know today's news, or can't show where their
information came from. This is a personal news analyst that's always
current, organized by domain, and answers only from verified, cited
sources."

## 2. Architecture, at a glance (2 min)

Show `diagrams/architecture.png`. Walk top to bottom:
- Scheduler triggers three fetcher agents (Tech / Finance / Politics),
  each with its own sources and relevance rules.
- A processing agent dedupes the same story from 20 outlets down to one,
  cleans it, tags it, and chunks it.
- Chunks are embedded (Cohere) and stored in ChromaDB with metadata —
  category, date, source, URL — so retrieval can be filtered.
- A 5-step RAG query engine answers questions: interpret -> hybrid
  retrieve -> rerank -> generate -> cite (or refuse).

## 3. Live demo (5 min)

Run `streamlit run app/streamlit_app.py` ahead of time so the knowledge
base is already populated (`python -m src.cli run`, ideally executed a few
times across a day or two beforehand so there's real content).

Suggested live questions, in order:
1. **A grounded, cited answer**: "What did the RBI announce this week and
   how did markets react?" — point out the citations with dates and links
   in the sidebar/answer.
2. **Category + date filtering**: toggle the category filter to
   "technology" and ask "What happened in AI this week?" — show that the
   filter and the inferred date range both narrow retrieval.
3. **Honest refusal**: ask something outside the knowledge base entirely,
   e.g. "What's the score of tonight's cricket match?" — the system should
   reply "I don't have news on that." rather than guessing. This is the
   moment to emphasize: *grounded means it also knows what it doesn't
   know.*
4. **Today's Briefing**: open the sidebar panel to show the auto-generated
   morning digest per category.

## 4. Evaluation (1 min)

Show `eval/eval_report.md` (after running `python -m eval.run_eval`).
Explain the three checks: citation correctness is verified
deterministically (every citation URL must trace back to a chunk that was
actually retrieved), recency compliance is checked against the inferred
date range, and accuracy is an LLM-judge faithfulness check against the
retrieved text — call out explicitly that this isn't an independent
fact-check of the news itself, just a check that the model didn't invent
or contradict what it retrieved.

## 5. Wrap-up (1 min)

What this covers from the brief: multi-agent orchestration (3 fetcher
agents + processing agent + orchestrator), a real data pipeline
(fetch -> dedupe -> clean -> chunk), embeddings + vector search (Cohere +
ChromaDB, metadata-filtered hybrid retrieval), prompt design (grounded
generation with structural refusal, not just a prompt instruction), and a
usable product (Streamlit chat with filters and a daily briefing).
