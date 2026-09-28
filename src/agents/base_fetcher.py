"""Domain fetcher agent.

One instance per category (technology / finance / politics). Each agent:
  - owns its own list of sources (RSS feeds, an optional official press-
    release page, and optionally NewsAPI/GNews) as loaded from sources.yaml
  - owns its own relevance rule: keep an article only if it matches at
    least one keyword and none of the drop keywords
  - fetches with retries, and stores raw articles via RawStore (which
    enforces URL-level dedupe)

This is intentionally a plain Python class rather than a framework agent
(LangGraph/CrewAI) -- see README "Design decisions" for why, and how to
swap one in.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Optional

import feedparser
import requests
from bs4 import BeautifulSoup
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from src import config
from src.models import RawArticle
from src.storage.raw_store import RawStore

logger = logging.getLogger(__name__)

_HEADERS = {"User-Agent": config.USER_AGENT}


class FetchError(Exception):
    pass


@retry(
    stop=stop_after_attempt(config.FETCH_MAX_RETRIES),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type((requests.RequestException, FetchError)),
    reraise=True,
)
def _http_get(url: str) -> requests.Response:
    resp = requests.get(url, headers=_HEADERS, timeout=config.REQUEST_TIMEOUT_SECS)
    if resp.status_code >= 500:
        # Retry server errors; don't retry 4xx (won't fix itself).
        raise FetchError(f"{url} returned {resp.status_code}")
    resp.raise_for_status()
    return resp


def _strip_html(html: str) -> str:
    return BeautifulSoup(html or "", "html.parser").get_text(" ", strip=True)


def _parse_feed_date(entry) -> Optional[datetime]:
    for field in ("published_parsed", "updated_parsed"):
        val = getattr(entry, field, None)
        if val:
            return datetime(*val[:6], tzinfo=timezone.utc)
    return None


class FetcherAgent:
    def __init__(self, category: str, source_config: dict, store: RawStore):
        self.category = category
        self.cfg = source_config
        self.store = store
        self.keep_kw = [k.lower() for k in source_config.get("relevance_keywords", [])]
        self.drop_kw = [k.lower() for k in source_config.get("drop_keywords", [])]

    # ---- relevance rule -------------------------------------------------
    def is_relevant(self, headline: str, body_snippet: str) -> bool:
        text = f"{headline} {body_snippet}".lower()
        if any(bad in text for bad in self.drop_kw):
            return False
        if not self.keep_kw:
            return True
        return any(good in text for good in self.keep_kw)

    # ---- RSS --------------------------------------------------------------
    def fetch_rss(self, feed_name: str, feed_url: str) -> list[RawArticle]:
        try:
            resp = _http_get(feed_url)
        except Exception as e:
            logger.warning("RSS fetch failed for %s (%s): %s", feed_name, feed_url, e)
            return []

        parsed = feedparser.parse(resp.content)
        articles: list[RawArticle] = []
        for entry in parsed.entries:
            headline = getattr(entry, "title", "").strip()
            summary_html = getattr(entry, "summary", "") or getattr(entry, "description", "")
            body = _strip_html(summary_html)
            url = getattr(entry, "link", "").strip()
            if not url or not headline:
                continue
            if not self.is_relevant(headline, body):
                continue
            articles.append(
                RawArticle(
                    url=url,
                    headline=headline,
                    body=body or headline,
                    source=feed_name,
                    category=self.category,
                    published_at=_parse_feed_date(entry),
                )
            )
        return articles

    # ---- official press-release listing pages -----------------------------
    def fetch_official_page(self, page_name: str, page_cfg: dict) -> list[RawArticle]:
        url = page_cfg["url"]
        try:
            resp = _http_get(url)
        except Exception as e:
            logger.warning("Official page fetch failed for %s (%s): %s", page_name, url, e)
            return []

        soup = BeautifulSoup(resp.content, "html.parser")
        must_contain = page_cfg.get("link_must_contain", "")
        links = soup.select(page_cfg.get("link_selector", "a"))

        articles: list[RawArticle] = []
        seen_urls: set[str] = set()
        for a in links:
            href = a.get("href", "")
            if not href or (must_contain and must_contain.lower() not in href.lower()):
                continue
            full_url = href if href.startswith("http") else requests.compat.urljoin(url, href)
            if full_url in seen_urls:
                continue
            seen_urls.add(full_url)
            headline = a.get_text(" ", strip=True)
            if not headline or len(headline) < 8:
                continue
            if not self.is_relevant(headline, ""):
                continue
            articles.append(
                RawArticle(
                    url=full_url,
                    headline=headline,
                    body=headline,  # listing pages rarely give body text; the
                    # processing agent can be pointed at full_url for a deeper
                    # fetch later. Kept simple here to avoid hammering .gov
                    # sites with per-article requests during a demo/course
                    # project.
                    source=page_name,
                    category=self.category,
                    published_at=None,
                )
            )
        return articles

    # ---- optional NewsAPI --------------------------------------------------
    def fetch_newsapi(self, query_terms: list[str]) -> list[RawArticle]:
        if not config.NEWSAPI_KEY:
            return []
        try:
            resp = requests.get(
                "https://newsapi.org/v2/everything",
                params={
                    "q": " OR ".join(query_terms[:10]) or self.category,
                    "language": "en",
                    "sortBy": "publishedAt",
                    "pageSize": 25,
                    "apiKey": config.NEWSAPI_KEY,
                },
                timeout=config.REQUEST_TIMEOUT_SECS,
            )
            resp.raise_for_status()
        except requests.RequestException as e:
            logger.warning("NewsAPI fetch failed: %s", e)
            return []

        articles = []
        for item in resp.json().get("articles", []):
            headline = (item.get("title") or "").strip()
            body = (item.get("description") or item.get("content") or "").strip()
            url = (item.get("url") or "").strip()
            if not headline or not url or not self.is_relevant(headline, body):
                continue
            published_at = None
            if item.get("publishedAt"):
                try:
                    published_at = datetime.fromisoformat(item["publishedAt"].replace("Z", "+00:00"))
                except ValueError:
                    pass
            articles.append(
                RawArticle(
                    url=url,
                    headline=headline,
                    body=body,
                    source=(item.get("source") or {}).get("name", "NewsAPI"),
                    category=self.category,
                    published_at=published_at,
                )
            )
        return articles

    # ---- run everything for this agent ------------------------------------
    def run(self) -> dict:
        run_id = str(uuid.uuid4())
        self.store.log_run_start(run_id, self.category)
        fetched, kept, error = 0, 0, None
        try:
            all_articles: list[RawArticle] = []
            for feed in self.cfg.get("feeds", []):
                all_articles.extend(self.fetch_rss(feed["name"], feed["url"]))
            for page in self.cfg.get("official_pages", []):
                all_articles.extend(self.fetch_official_page(page["name"], page))
            all_articles.extend(self.fetch_newsapi(self.keep_kw))

            fetched = len(all_articles)
            for art in all_articles:
                if self.store.insert_if_new(art):
                    kept += 1
            self.store.log_run_end(run_id, self.category, "success", fetched, kept)
        except Exception as e:  # noqa: BLE001 - orchestrator decides retry policy
            error = str(e)
            self.store.log_run_end(run_id, self.category, "failed", fetched, kept, error)
            raise
        return {"run_id": run_id, "category": self.category, "fetched": fetched, "kept": kept, "error": error}
