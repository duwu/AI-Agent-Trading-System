"""Timestamped public news for cross-asset decisions; no synthetic sentiment.

NewsAPI is optional. Google News RSS is the key-free fallback. Articles are
headlines with citations, not verified facts or instructions for the model.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from hashlib import sha256
from html import unescape
import math
import os
import re
import time
from urllib.parse import urlparse
import xml.etree.ElementTree as ET

import requests


UTC = timezone.utc
KNOWN_SOURCES = (
    "reuters", "associated press", "ap news", "bloomberg", "financial times",
    "wall street journal", "wsj", "cnbc", "bbc", "yahoo finance", "marketwatch",
    "investing.com", "barron's", "morningstar", "seeking alpha", "the economist",
    "nikkei", "the guardian", "cnn", "npr", "cbs news", "abc news", "coindesk",
    "cointelegraph", "the block", "decrypt", "bitcoin magazine", "kitco",
    "oilprice.com", "etf.com", "zacks", "nasdaq", "the motley fool",
    "investopedia", "fortune", "business insider", "axios", "south china morning post",
    "semiconductor engineering", "s&p global", "eia.gov", "federal reserve",
)
MACRO_QUERY = '"Federal Reserve" OR inflation OR "interest rates"'


def _utc_iso(value):
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _clean_text(value):
    return re.sub(r"\s+", " ", unescape(re.sub(r"<[^>]*>", "", str(value or "")))).strip()


def _parse_date(value):
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        try:
            parsed = parsedate_to_datetime(value)
        except (TypeError, ValueError, OverflowError):
            return None
    # A date without timezone cannot establish whether an article is current.
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(UTC)


def _matches_title(title, keywords):
    """Search results can match article bodies; require explicit headline evidence.

    Missing keywords preserve the public provider's optional-filter interface.
    Word boundaries prevent e.g. the ticker AMD matching a longer word.
    """
    if not keywords:
        return True
    return any(
        re.search(r"(?<!\w)" + re.escape(keyword) + r"(?!\w)", title, re.IGNORECASE)
        for keyword in keywords
    )


class NewsProvider:
    """Collect recent finance headlines with dates, publishers and clickable URLs.

    ``assets`` contain ``id`` and ``news_query``. Shared queries are fetched once
    (SOXL and SOXS typically share the same semiconductor headlines).
    """

    def __init__(self, api_key=None, request_timeout=12, max_workers=4):
        self.api_key = os.getenv("NEWSAPI_KEY", "").strip() if api_key is None else api_key.strip()
        self.request_timeout = request_timeout
        self.max_workers = max_workers

    def collect(self, assets, lookback_hours=48):
        if not isinstance(lookback_hours, (int, float)) or isinstance(lookback_hours, bool) or not math.isfinite(lookback_hours) or lookback_hours <= 0:
            raise ValueError("lookback_hours must be a positive finite number")
        now = datetime.now(UTC)
        cutoff = now - timedelta(hours=lookback_hours)
        queries = {}
        asset_keywords = {}
        errors = []
        for asset in assets:
            asset_id = str(asset["id"])
            raw_keywords = asset.get("news_keywords") or []
            if isinstance(raw_keywords, str):
                raw_keywords = [raw_keywords]
            asset_keywords[asset_id] = [str(k).strip() for k in raw_keywords if str(k).strip()]
            query = str(asset.get("news_query") or "").strip()
            if not query:
                errors.append({"asset_ids": [asset_id], "source": "configuration", "error": "missing_news_query"})
                continue
            queries.setdefault(query, {"asset_ids": [], "scope": "asset"})["asset_ids"].append(asset_id)
        queries[MACRO_QUERY] = {"asset_ids": [str(a["id"]) for a in assets], "scope": "macro"}
        candidates = []
        with ThreadPoolExecutor(max_workers=max(1, min(self.max_workers, 4))) as pool:
            tasks = {}
            for query, details in queries.items():
                # A shared query can cover several assets. Accept a NewsAPI batch
                # if at least one configured asset has headline evidence; apply
                # each asset's own keyword filter below before assigning coverage.
                keywords = None
                if details["scope"] == "asset" and all(asset_keywords[a] for a in details["asset_ids"]):
                    keywords = sorted({k for a in details["asset_ids"] for k in asset_keywords[a]})
                tasks[pool.submit(self._fetch_query, query, cutoff, now, lookback_hours, keywords)] = details
            for task in as_completed(tasks):
                details = tasks[task]
                try:
                    articles, fetch_errors = task.result()
                except Exception as exc:
                    # Never log exception strings: upstream errors may contain credentials.
                    articles, fetch_errors = [], [{"source": "news", "error": type(exc).__name__}]
                for error in fetch_errors:
                    errors.append({**error, "asset_ids": details["asset_ids"], "scope": details["scope"]})
                for raw in articles:
                    article = self._normalize(raw, cutoff, now)
                    if article:
                        direct_ids = [
                            a for a in details["asset_ids"]
                            if details["scope"] == "asset" and _matches_title(article["title"], asset_keywords[a])
                        ]
                        if details["scope"] == "asset" and not direct_ids:
                            continue
                        article.update(
                            asset_ids=direct_ids[:] if details["scope"] == "asset" else details["asset_ids"][:], scope=details["scope"],
                            direct_asset_ids=direct_ids,
                            is_macro=details["scope"] == "macro",
                        )
                        candidates.append(article)
        articles = self._deduplicate_and_limit(candidates, [str(a["id"]) for a in assets])
        per_asset = {
            str(asset["id"]): {
                "article_count": sum(str(asset["id"]) in a["direct_asset_ids"] for a in articles),
                "macro_article_count": sum(str(asset["id"]) in a["asset_ids"] and a["is_macro"] for a in articles),
            }
            for asset in assets
        }
        for asset_id, coverage in per_asset.items():
            if not coverage["article_count"]:
                errors.append({"asset_ids": [asset_id], "source": "news", "error": "no_recent_asset_articles"})
        if not any(article["is_macro"] for article in articles):
            errors.append({"asset_ids": [str(a["id"]) for a in assets], "source": "news", "error": "no_recent_macro_articles", "scope": "macro"})
        return {
            "collected_at": _utc_iso(now), "lookback_hours": lookback_hours,
            "articles": articles, "per_asset": per_asset, "errors": errors,
            "selection_policy": "Recent timestamped headlines from recognized financial/general publishers; publication does not verify the claim.",
        }

    def _fetch_query(self, query, cutoff, now, lookback_hours, keywords=None):
        errors = []
        if self.api_key:
            try:
                response = self._get(
                    "https://newsapi.org/v2/everything",
                    headers={"X-Api-Key": self.api_key},
                    params={"q": query, "from": _utc_iso(cutoff), "to": _utc_iso(now), "language": "en", "sortBy": "publishedAt", "pageSize": 30},
                    timeout=self.request_timeout,
                )
                if response.status_code == 200:
                    payload = response.json()
                    rows = [
                        {"title": a.get("title"), "url": a.get("url"), "source": (a.get("source") or {}).get("name"), "published_at": a.get("publishedAt"), "origin": "newsapi"}
                        for a in payload.get("articles", [])
                    ]
                    relevant = [self._normalize(a, cutoff, now) for a in rows]
                    if any(a and _matches_title(a["title"], keywords) for a in relevant):
                        return rows, errors
                    errors.append({"source": "newsapi", "error": "no_recent_relevant_articles"})
                else:
                    errors.append({"source": "newsapi", "error": f"http_{response.status_code}"})
            except Exception as exc:
                errors.append({"source": "newsapi", "error": type(exc).__name__})
        try:
            response = self._get(
                "https://news.google.com/rss/search",
                params={"q": f"({query}) when:{max(1, math.ceil(lookback_hours))}h", "hl": "en-US", "gl": "US", "ceid": "US:en"},
                headers={"User-Agent": "AI-Agent-Market-Research/1.0"}, timeout=self.request_timeout,
            )
            response.raise_for_status()
            root = ET.fromstring(response.content)
            rows = []
            for item in root.findall("./channel/item")[:100]:
                rows.append({"title": item.findtext("title"), "url": item.findtext("link"), "source": item.findtext("source"), "published_at": item.findtext("pubDate"), "origin": "google_news_rss"})
            return rows, errors
        except Exception as exc:
            errors.append({"source": "google_news_rss", "error": type(exc).__name__})
            return [], errors

    @staticmethod
    def _get(url, **kwargs):
        # One bounded retry for transient outages/rate limiting. Authentication
        # failures are returned immediately so the public fallback can run.
        for attempt in range(2):
            try:
                response = requests.get(url, **kwargs)
                if response.status_code not in (429, 500, 502, 503, 504) or attempt:
                    return response
            except (requests.Timeout, requests.ConnectionError):
                if attempt:
                    raise
            time.sleep(0.5)

    @staticmethod
    def _normalize(raw, cutoff, now):
        title, source = _clean_text(raw.get("title")), _clean_text(raw.get("source"))
        url = str(raw.get("url") or "").strip()
        published = _parse_date(raw.get("published_at"))
        parsed_url = urlparse(url)
        if not title or title == "[Removed]" or parsed_url.scheme not in ("https", "http") or not parsed_url.netloc:
            return None
        if published is None or not cutoff <= published <= now:
            return None
        if not any(s in source.casefold() for s in KNOWN_SOURCES):
            return None
        # Google appends a publisher already present in the separate source field.
        if title.casefold().endswith(" - " + source.casefold()):
            title = title[:-(len(source) + 3)].strip()
        return {"title": title[:500], "url": url, "source": source[:100], "published_at": _utc_iso(published), "origin": raw.get("origin", "news")}

    @staticmethod
    def _deduplicate_and_limit(candidates, asset_ids):
        # Same story can belong to both semiconductor products. Merge citations
        # instead of using duplicate headlines to inflate the apparent evidence.
        by_title, by_url, merged = {}, {}, []
        for article in sorted(candidates, key=lambda a: (a["published_at"], a["url"]), reverse=True):
            title_key = re.sub(r"\W+", "", article["title"].casefold())
            direct_ids = article.get("direct_asset_ids", article["asset_ids"] if article["scope"] == "asset" else [])
            is_macro = article.get("is_macro", article["scope"] == "macro")
            existing = by_title.get(title_key) or by_url.get(article["url"])
            if existing is not None:
                existing["asset_ids"] = sorted(set(existing["asset_ids"]) | set(article["asset_ids"]))
                existing["direct_asset_ids"] = sorted(set(existing["direct_asset_ids"]) | set(direct_ids))
                existing["is_macro"] = existing["is_macro"] or is_macro
                existing["scope"] = "asset" if existing["direct_asset_ids"] else "macro"
                by_title[title_key] = by_url[article["url"]] = existing
                continue
            entry = {**article, "asset_ids": sorted(set(article["asset_ids"])), "direct_asset_ids": sorted(set(direct_ids)), "is_macro": is_macro}
            by_title[title_key] = by_url[entry["url"]] = entry
            merged.append(entry)
        # Use each category's newest six stories, then round-robin the categories
        # at the global cap. An asset whose news is older must still get coverage.
        buckets = {a: [i for i, row in enumerate(merged) if a in row["direct_asset_ids"]][:6] for a in asset_ids}
        buckets["__macro__"] = [i for i, row in enumerate(merged) if row["is_macro"]][:6]
        selected_indexes = set()
        for rank in range(6):
            for indexes in buckets.values():
                if rank < len(indexes):
                    selected_indexes.add(indexes[rank])
                if len(selected_indexes) >= 30:
                    break
            if len(selected_indexes) >= 30:
                break
        selected = []
        for index in sorted(selected_indexes):
            article = merged[index]
            article["direct_asset_ids"] = [a for a in asset_ids if index in buckets[a]]
            article["is_macro"] = index in buckets["__macro__"]
            article["scope"] = "asset" if article["direct_asset_ids"] else "macro"
            if not article["is_macro"]:
                article["asset_ids"] = article["direct_asset_ids"][:]
            key = article["title"].casefold() + "|" + article["published_at"]
            article["id"] = "n_" + sha256(key.encode()).hexdigest()[:12]
            selected.append(article)
        return selected
