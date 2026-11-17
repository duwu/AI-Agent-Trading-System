"""Read-only OKX market discovery and completed-candle analysis.

Instrument names are resolved against the exchange's live catalogue. An unavailable
market, stale quote or missing indicator remains unavailable rather than receiving
a synthetic price or a neutral indicator value.
"""

from datetime import datetime, timezone
import math
import os
import statistics
import threading
import time
from typing import Any, Optional

import requests


BAR_SECONDS = {"5m": 300, "15m": 900, "1H": 3600, "4H": 14400, "1D": 86400}
_REQUEST_SLOTS = threading.BoundedSemaphore(4)


def _utc(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, timezone.utc).isoformat()


def _positive(value: Any) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError("price must be finite and positive")
    return number


def _ema(values: list[float], period: int) -> list[Optional[float]]:
    """EMA seeded with its first period's simple average."""
    result: list[Optional[float]] = [None] * len(values)
    if len(values) < period:
        return result
    value = sum(values[:period]) / period
    result[period - 1] = value
    alpha = 2 / (period + 1)
    for index in range(period, len(values)):
        value += alpha * (values[index] - value)
        result[index] = value
    return result


def _wilder(values: list[float], period: int) -> float:
    value = sum(values[:period]) / period
    for current in values[period:]:
        value = ((period - 1) * value + current) / period
    return value


class OKXMarketProvider:
    """Public endpoints only; no account credentials or order methods."""

    def __init__(self, base_url: Optional[str] = None):
        self.base_url = (base_url or os.getenv("OKX_BASE_URL", "https://www.okx.com")).rstrip("/")
        self._catalog: Optional[dict[str, dict]] = None
        self._catalog_errors: list[str] = []
        self._catalog_lock = threading.Lock()

    def _get(self, path: str, params: dict) -> list:
        """One bounded retry for transport, 429 and upstream 5xx failures."""
        for attempt in range(2):
            try:
                with _REQUEST_SLOTS:
                    response = requests.get(
                        self.base_url + path, params=params, timeout=(8, 20),
                        headers={"Accept": "application/json", "User-Agent": "ai-agent-okx-analysis/1.0"},
                    )
                if (response.status_code == 429 or response.status_code >= 500) and attempt == 0:
                    time.sleep(0.6)
                    continue
                response.raise_for_status()
                payload = response.json()
                if str(payload.get("code")) != "0":
                    raise ValueError(f"OKX {payload.get('code')}: {str(payload.get('msg', 'request failed'))[:180]}")
                if not isinstance(payload.get("data"), list):
                    raise ValueError("OKX response data is not a list")
                return payload["data"]
            except (requests.ConnectionError, requests.Timeout):
                if attempt == 1:
                    raise
                time.sleep(0.6)
        raise RuntimeError("OKX request exhausted")

    def resolve_assets(self, asset_configs: list[dict]) -> list[dict]:
        """Select the first exact live candidate; never guess another product."""
        with self._catalog_lock:
            if self._catalog is None:
                self._catalog = {}
                for instrument_type in ("SPOT", "SWAP"):
                    try:
                        rows = self._get("/api/v5/public/instruments", {"instType": instrument_type})
                        self._catalog.update({row["instId"]: row for row in rows})
                    except Exception as exc:
                        self._catalog_errors.append(f"{instrument_type} catalogue: {type(exc).__name__}: {exc}")
        result = []
        for config in asset_configs:
            asset = dict(config)
            asset.update({"instrument_available": False, "inst_id": None, "inst_type": None,
                          "state": None, "allowed_directions": ["wait"]})
            rejected = []
            for candidate in config.get("candidates", []):
                instrument = self._catalog.get(candidate)
                if instrument and instrument.get("state") == "live":
                    kind = instrument["instType"]
                    asset.update({"instrument_available": True, "inst_id": candidate,
                                  "inst_type": kind, "state": "live",
                                  "allowed_directions": ["long", "short", "wait"] if kind == "SWAP" else ["long", "wait"],
                                  "instrument": {key: instrument.get(key) for key in
                                      ("instId", "instType", "baseCcy", "quoteCcy", "settleCcy", "ctType", "ctVal", "ctValCcy", "tickSz", "lotSz", "minSz")}})
                    break
                rejected.append(f"{candidate}: {instrument.get('state') if instrument else 'not listed'}")
            if not asset["instrument_available"]:
                asset["unavailable_reason"] = "; ".join(rejected + self._catalog_errors) or "no instrument candidates configured"
            result.append(asset)
        return result

    def _ticker(self, asset: dict) -> dict:
        rows = self._get("/api/v5/market/ticker", {"instId": asset["inst_id"]})
        if not rows:
            raise ValueError("empty ticker")
        row = rows[0]
        if row.get("instId") != asset["inst_id"]:
            raise ValueError("ticker instrument does not match requested market")
        stamp = float(row["ts"]) / 1000
        age = time.time() - stamp
        if age > 120 or age < -30:
            raise ValueError(f"stale or future ticker (age {age:.1f}s)")
        last, bid, ask, opening = (_positive(row[field]) for field in ("last", "bidPx", "askPx", "open24h"))
        if ask < bid:
            raise ValueError("crossed order book")
        volume = float(row["volCcy24h"])
        if not math.isfinite(volume) or volume < 0:
            raise ValueError("invalid ticker volume")
        # OKX spot volCcy24h is quote volume; derivative volCcy24h is base volume.
        derivative = asset["inst_type"] == "SWAP"
        quote_volume = volume * last if derivative else volume
        return {"last": last, "bid": bid, "ask": ask,
                "spread_bps": round((ask - bid) / ((ask + bid) / 2) * 10000, 5),
                "open_24h": opening, "change_24h_pct": round((last / opening - 1) * 100, 5),
                "volume_quote_24h": quote_volume, "volume_quote_24h_estimated": derivative,
                "volume_quote_24h_method": "base volume times latest price" if derivative else "exchange quote volume",
                "observed_at": _utc(stamp), "age_seconds": round(age, 2),
                "source": self.base_url + "/api/v5/market/ticker"}

    def _timeframe(self, inst_id: str, bar: str, limit: int) -> dict:
        if bar not in BAR_SECONDS:
            raise ValueError(f"unsupported timeframe: {bar}")
        rows = self._get("/api/v5/market/candles", {"instId": inst_id, "bar": bar, "limit": str(limit)})
        # An incomplete candle (confirm=0) must never influence the indicators.
        candles = {}
        for row in rows:
            if len(row) < 9 or str(row[8]) != "1":
                continue
            timestamp = int(row[0]) / 1000
            opening, high, low, close = (_positive(row[index]) for index in (1, 2, 3, 4))
            if high < max(opening, low, close) or low > min(opening, high, close):
                raise ValueError("invalid OHLC candle")
            candles[timestamp] = (opening, high, low, close)
        ordered = sorted(candles.items())
        if len(ordered) < 60:
            raise ValueError(f"insufficient completed candles: {len(ordered)} (need 60)")
        latest_open = ordered[-1][0]
        latest_close = latest_open + BAR_SECONDS[bar]
        age = time.time() - latest_close
        if age < -30 or age > 2 * BAR_SECONDS[bar]:
            raise ValueError(f"stale or future {bar} completed candles (age {age:.1f}s)")
        if any(right[0] - left[0] != BAR_SECONDS[bar] for left, right in zip(ordered, ordered[1:])):
            raise ValueError("gaps in completed candle sequence")
        highs = [row[1][1] for row in ordered]
        lows = [row[1][2] for row in ordered]
        closes = [row[1][3] for row in ordered]
        changes = [right - left for left, right in zip(closes, closes[1:])]
        average_gain = _wilder([max(change, 0) for change in changes], 14)
        average_loss = _wilder([max(-change, 0) for change in changes], 14)
        rsi = 50.0 if average_gain == average_loss == 0 else (100.0 if average_loss == 0 else 100 - 100 / (1 + average_gain / average_loss))
        fast, slow = _ema(closes, 12), _ema(closes, 26)
        macd_values = [left - right for left, right in zip(fast, slow) if left is not None and right is not None]
        macd, signal = macd_values[-1], _ema(macd_values, 9)[-1]
        true_ranges = [max(highs[index] - lows[index], abs(highs[index] - closes[index - 1]),
                           abs(lows[index] - closes[index - 1])) for index in range(1, len(closes))]
        returns = [(right / left - 1) * 100 for left, right in zip(closes[-21:], closes[-20:])]
        return {"available": True, "timestamp": _utc(latest_close), "open_time": _utc(latest_open),
                "age_seconds": round(age, 2), "close": closes[-1], "rsi14": round(rsi, 5),
                "ema20": _ema(closes, 20)[-1], "ema50": _ema(closes, 50)[-1],
                "macd": macd, "macd_signal": signal, "macd_histogram": macd - signal,
                "atr14": _wilder(true_ranges, 14), "volatility_pct": statistics.stdev(returns),
                "support": min(lows[-20:]), "resistance": max(highs[-20:]),
                "candle_count": len(ordered), "source": self.base_url + "/api/v5/market/candles"}

    def _market_session(self, inst_id: str) -> dict:
        """Best-effort context from OKX's public website, not an execution gate.

        A token quote may update around the clock while its reference ETF has
        different liquidity through the US trading day. This endpoint describes
        that session; it does not provide a live ETF NAV or underlying price.
        """
        path = "/priapi/v5/rubik/stock/market-status-cex-region"
        rows = self._get(path, {"instId": inst_id})
        if not rows:
            raise ValueError("empty website market-session response")
        current, market = rows[0].get("current") or {}, rows[0].get("market") or {}
        if not all(isinstance(value, str) and value for value in
                   (current.get("sessionCode"), current.get("liquidity"), market.get("timezone"))):
            raise ValueError("incomplete website market-session response")
        stamp = float(current["serverTs"]) / 1000
        age = time.time() - stamp
        if age > 900 or age < -30:
            raise ValueError(f"stale or future website market-session timestamp (age {age:.1f}s)")
        return {"session_code": current["sessionCode"], "liquidity": current["liquidity"],
                "timezone": market["timezone"], "market_description": market.get("marketSubText", ""),
                "observed_at": _utc(stamp), "source": self.base_url + path + "?instId=" + inst_id,
                "source_type": "public_website_api", "underlying_price_available": False}

    def collect_asset(self, asset: dict, timeframes: list[str], limit: int = 120) -> dict:
        started = time.monotonic()
        result = dict(asset)
        result.update({"market_available": False, "data_complete": False, "ticker": None,
                       "timeframes": {}, "funding": None, "market_session": None, "errors": []})
        if not asset.get("instrument_available"):
            result["errors"].append(asset.get("unavailable_reason", "instrument unavailable"))
            result["fetch_duration_seconds"] = round(time.monotonic() - started, 3)
            return result
        if not 61 <= limit <= 300:
            raise ValueError("candle limit must be between 61 and 300")
        try:
            result["ticker"] = self._ticker(asset)
        except Exception as exc:
            result["errors"].append(f"ticker: {type(exc).__name__}: {exc}")
        for bar in dict.fromkeys(timeframes):
            try:
                result["timeframes"][bar] = self._timeframe(asset["inst_id"], bar, limit)
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                result["timeframes"][bar] = {"available": False, "error": error}
                result["errors"].append(f"{bar}: {error}")
        if asset["inst_type"] == "SWAP":
            try:
                rows = self._get("/api/v5/public/funding-rate", {"instId": asset["inst_id"]})
                if rows:
                    rate = float(rows[0]["fundingRate"])
                    if not math.isfinite(rate):
                        raise ValueError("invalid funding rate")
                    result["funding"] = {"rate": rate, "funding_time": _utc(float(rows[0]["fundingTime"]) / 1000),
                                         "source": self.base_url + "/api/v5/public/funding-rate"}
            except Exception as exc:
                result["errors"].append(f"funding: {type(exc).__name__}: {exc}")
        if str(asset.get("id", "")).upper() in ("XSOXS", "XSOXL"):
            try:
                result["market_session"] = self._market_session(asset["inst_id"])
            except Exception as exc:
                result["errors"].append(f"market_session (optional): {type(exc).__name__}: {exc}")
        valid = [row.get("available", False) for row in result["timeframes"].values()]
        result["market_available"] = result["ticker"] is not None and any(valid)
        result["data_complete"] = result["ticker"] is not None and bool(valid) and all(valid)
        result["collected_at"] = _utc(time.time())
        result["fetch_duration_seconds"] = round(time.monotonic() - started, 3)
        return result
