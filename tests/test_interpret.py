from datetime import date

from src.rag.interpret import interpret_query


def test_infers_finance_category():
    result = interpret_query("What did the RBI announce about interest rates?")
    assert result.category == "finance"


def test_infers_technology_category():
    result = interpret_query("What's new with AI startups and chip makers?")
    assert result.category == "technology"


def test_no_clear_category_returns_none():
    result = interpret_query("Tell me something interesting.")
    assert result.category is None


def test_infers_today_date_range():
    today = date(2026, 6, 15)
    result = interpret_query("What happened today?", today=today)
    assert result.date_from == today
    assert result.date_to == today


def test_infers_this_week_date_range():
    today = date(2026, 6, 18)  # a Thursday
    result = interpret_query("What happened this week in finance?", today=today)
    assert result.date_from <= today <= result.date_to
    assert (result.date_to - result.date_from).days <= 6


def test_no_date_phrase_leaves_range_unset():
    result = interpret_query("What did the RBI say?")
    assert result.date_from is None
    assert result.date_to is None
