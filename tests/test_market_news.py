from datetime import datetime, timedelta, timezone
from unittest.mock import Mock, patch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ai_agent.market_news import NewsProvider


def article(title="Bitcoin outlook", source="Reuters", hours=1, url="https://example.com/story"):
    return {"title": title, "url": url, "source": source, "published_at": (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()}


def test_date_filter_and_required_citations():
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(hours=48)
    assert NewsProvider._normalize(article(hours=49), cutoff, now) is None
    assert NewsProvider._normalize(article(hours=-1), cutoff, now) is None
    assert NewsProvider._normalize({**article(), "published_at": "2026-11-16T12:00:00"}, cutoff, now) is None
    assert NewsProvider._normalize(article(source="Unverified influencer"), cutoff, now) is None
    assert NewsProvider._normalize(article(url="javascript:alert(1)"), cutoff, now) is None
    result = NewsProvider._normalize(article(title="<b>Bitcoin</b> &amp; gold - Reuters"), cutoff, now)
    assert result["title"] == "Bitcoin & gold"
    assert result["published_at"].endswith("Z")


def test_shared_queries_deduplicate_and_merge_asset_coverage():
    assets = [{"id": "XSOXL", "news_query": "semiconductor Nvidia"}, {"id": "XSOXS", "news_query": "semiconductor Nvidia"}]
    provider = NewsProvider(api_key="")
    def fetch(query, *args):
        return ([article(title="Nvidia outlook", url="https://example.com/semis")] if query == "semiconductor Nvidia" else []), []
    with patch.object(provider, "_fetch_query", side_effect=fetch) as call:
        data = provider.collect(assets)
    assert call.call_count == 2  # shared asset query plus macro
    assert len(data["articles"]) == 1
    assert data["articles"][0]["asset_ids"] == ["XSOXL", "XSOXS"]
    assert data["per_asset"]["XSOXL"]["article_count"] == 1
    assert data["per_asset"]["XSOXS"]["article_count"] == 1
    assert all(e["error"] == "no_recent_macro_articles" for e in data["errors"])


def test_api_auth_failure_falls_back_without_leaking_key():
    provider = NewsProvider(api_key="secret-test-value")
    failed = Mock(status_code=401)
    feed = Mock(status_code=200)
    date = (datetime.now(timezone.utc) - timedelta(hours=1)).strftime("%a, %d %b %Y %H:%M:%S GMT")
    feed.content = f"<rss><channel><item><title>Gold market - Reuters</title><link>https://example.com/gold</link><source>Reuters</source><pubDate>{date}</pubDate></item></channel></rss>".encode()
    with patch("ai_agent.market_news.requests.get", side_effect=[failed, feed]) as get:
        rows, errors = provider._fetch_query("gold", datetime.now(timezone.utc) - timedelta(hours=48), datetime.now(timezone.utc), 48)
    assert rows[0]["source"] == "Reuters"
    assert errors == [{"source": "newsapi", "error": "http_401"}]
    assert "secret-test-value" not in str(errors)
    assert "apiKey" not in get.call_args_list[0].kwargs["params"]


def test_total_outage_is_missing_evidence_not_neutral_sentiment():
    provider = NewsProvider(api_key="")
    with patch("ai_agent.market_news.requests.get", side_effect=RuntimeError("request with sensitive credentials")):
        data = provider.collect([{"id": "BTC", "news_query": "bitcoin"}])
    assert data["articles"] == []
    assert data["per_asset"]["BTC"]["article_count"] == 0
    assert any(e["error"] == "no_recent_asset_articles" for e in data["errors"])
    assert "sensitive credentials" not in str(data)
    assert "sentiment" not in data


def test_article_limits_and_title_deduplication():
    rows = []
    for i in range(12):
        rows.append({**article(title=f"Gold report {i}", url=f"https://example.com/{i}", hours=i + 1), "asset_ids": ["XAUT"], "scope": "asset"})
    rows.append({**rows[0], "url": "https://other.example/story", "asset_ids": ["BTC"]})
    selected = NewsProvider._deduplicate_and_limit(rows, ["XAUT", "BTC"])
    assert len(selected) == 6
    assert len({a["id"] for a in selected}) == 6
    assert set(selected[0]["asset_ids"]) == {"XAUT", "BTC"}


def test_macro_duplicate_does_not_inflate_direct_asset_coverage():
    assets = [{"id": "BTC", "news_query": "bitcoin"}, {"id": "XAUT", "news_query": "gold"}, {"id": "CL", "news_query": "oil"}]
    provider = NewsProvider(api_key="")
    def fetch(query, *args):
        return ([] if query in ("gold", "oil") else [article(title="Bitcoin and interest rates")]), []
    with patch.object(provider, "_fetch_query", side_effect=fetch):
        data = provider.collect(assets)
    assert len(data["articles"]) == 1
    story = data["articles"][0]
    assert set(story["asset_ids"]) == {"BTC", "XAUT", "CL"}
    assert story["direct_asset_ids"] == ["BTC"]
    assert story["is_macro"] is True
    assert data["per_asset"]["BTC"] == {"article_count": 1, "macro_article_count": 1}
    assert data["per_asset"]["XAUT"] == {"article_count": 0, "macro_article_count": 1}
    assert data["per_asset"]["CL"] == {"article_count": 0, "macro_article_count": 1}
    missing = [e["asset_ids"] for e in data["errors"] if e["error"] == "no_recent_asset_articles"]
    assert ["XAUT"] in missing and ["CL"] in missing


def test_global_limit_preserves_fair_coverage_and_latest_per_category():
    asset_ids = ["BTC", "XAUT", "XSOXL", "XSOXS", "CL"]
    rows = []
    for asset_index, asset_id in enumerate(asset_ids):
        for i in range(8):
            rows.append({**article(title=f"{asset_id} report {i}", url=f"https://example.com/{asset_id}/{i}", hours=asset_index * 10 + i), "asset_ids": [asset_id], "scope": "asset"})
    for i in range(8):
        rows.append({**article(title=f"Macro report {i}", url=f"https://example.com/macro/{i}", hours=i), "asset_ids": asset_ids, "scope": "macro"})
    selected = NewsProvider._deduplicate_and_limit(rows, asset_ids)
    assert len(selected) == 30
    assert all(sum(a in row["direct_asset_ids"] for row in selected) == 5 for a in asset_ids)
    assert sum(row["is_macro"] for row in selected) == 5
    assert not any(row["title"].endswith("report 7") for row in selected)


def test_search_body_match_is_rejected_without_asset_headline_keywords():
    assets = [
        {"id": "BTC", "news_query": "bitcoin", "news_keywords": ["bitcoin", "btc", "crypto", "cryptocurrency"]},
        {"id": "XSOXL", "news_query": "nvidia", "news_keywords": ["nvidia", "semiconductor", "amd", "chip", "chips"]},
    ]
    provider = NewsProvider(api_key="")
    story = article(title="Jensen Huang weighs Nvidia comeback days after shock exit: FT")
    def fetch(query, *args):
        return ([story] if query in ("bitcoin", "nvidia") else []), []
    with patch.object(provider, "_fetch_query", side_effect=fetch):
        data = provider.collect(assets)
    assert len(data["articles"]) == 1
    assert data["articles"][0]["direct_asset_ids"] == ["XSOXL"]
    assert data["articles"][0]["asset_ids"] == ["XSOXL"]
    assert data["per_asset"]["BTC"]["article_count"] == 0
    assert data["per_asset"]["XSOXL"]["article_count"] == 1


def test_keyword_word_boundaries_do_not_match_other_words():
    from ai_agent.market_news import _matches_title
    assert _matches_title("NVIDIA chips drive semiconductor rally", ["nvidia"])
    assert _matches_title("BTC-USDT tumbles", ["btc"])
    assert not _matches_title("Oilseed and spoil prices rise", ["oil"])
    assert not _matches_title("Chipotle demand increases", ["chip"])


def test_irrelevant_newsapi_headlines_trigger_rss_fallback():
    provider = NewsProvider(api_key="secret-test-value")
    now = datetime.now(timezone.utc)
    wrong = Mock(status_code=200)
    wrong.json.return_value = {"articles": [{"title": "Nvidia rally continues", "source": {"name": "Reuters"}, "url": "https://example.com/nvidia", "publishedAt": (now - timedelta(hours=1)).isoformat()}]}
    feed = Mock(status_code=200)
    date = (now - timedelta(hours=1)).strftime("%a, %d %b %Y %H:%M:%S GMT")
    feed.content = f"<rss><channel><item><title>Bitcoin outlook - Reuters</title><link>https://example.com/btc</link><source>Reuters</source><pubDate>{date}</pubDate></item></channel></rss>".encode()
    with patch("ai_agent.market_news.requests.get", side_effect=[wrong, feed]) as get:
        rows, errors = provider._fetch_query("bitcoin", now - timedelta(hours=48), now, 48, ["bitcoin", "btc"])
    assert get.call_count == 2
    assert rows[0]["title"] == "Bitcoin outlook - Reuters"
    assert errors == [{"source": "newsapi", "error": "no_recent_relevant_articles"}]
