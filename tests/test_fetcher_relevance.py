from src.agents.base_fetcher import FetcherAgent
from src.storage.raw_store import RawStore


def make_agent(tmp_path, category, keep, drop):
    store = RawStore(str(tmp_path / "raw.db"))
    cfg = {"relevance_keywords": keep, "drop_keywords": drop}
    return FetcherAgent(category, cfg, store)


def test_finance_agent_keeps_market_news_and_drops_lifestyle(tmp_path):
    agent = make_agent(
        tmp_path,
        "finance",
        keep=["market", "rbi", "earnings"],
        drop=["horoscope", "celebrity"],
    )
    assert agent.is_relevant("RBI holds rates steady", "The central bank left the policy rate unchanged.")
    assert not agent.is_relevant("Today's horoscope for markets fans", "Your daily celebrity horoscope.")
    assert not agent.is_relevant("A completely unrelated lifestyle piece", "Nothing relevant in this text at all.")


def test_no_keep_keywords_means_everything_not_dropped_is_kept(tmp_path):
    agent = make_agent(tmp_path, "politics", keep=[], drop=["horoscope"])
    assert agent.is_relevant("Any headline", "any body")
    assert not agent.is_relevant("Weekly horoscope", "stars and signs")
