"""Checks for exact product discovery and market data quality gates."""

import pytest

from ai_agent.okx_market import OKXMarketProvider


NOW = 1_800_000_000.0


def candle_rows(count=100, bar_seconds=900):
    latest_open = (int(NOW) // bar_seconds - 1) * bar_seconds
    rows = []
    for offset in range(count):
        close = 100 + (count - offset) / 10
        rows.append([str((latest_open - offset * bar_seconds) * 1000),
                     str(close - 0.2), str(close + 1), str(close - 1), str(close),
                     "1", "1", "100", "1"])
    # Deliberately extreme unfinished price must not affect the indicators.
    rows.insert(0, [str((latest_open + bar_seconds) * 1000), "1", "1000000", "1", "1000000", "1", "1", "1", "0"])
    return rows


def resolved_asset(kind="SWAP"):
    return {"id": "btc", "inst_id": "BTC-USDT-SWAP", "inst_type": kind,
            "instrument_available": True, "allowed_directions": ["long", "short", "wait"]}


def test_exact_live_products_only_and_spot_cannot_be_shorted(monkeypatch):
    provider = OKXMarketProvider()
    catalog = {
        "SWAP": [{"instId": "BTC-USDT-SWAP", "instType": "SWAP", "state": "live"},
                 {"instId": "XAUT-USDT-SWAP", "instType": "SWAP", "state": "suspend"}],
        "SPOT": [{"instId": "XAUT-USDT", "instType": "SPOT", "state": "live"}],
    }
    calls = []
    def get(path, params):
        calls.append(params["instType"])
        return catalog[params["instType"]]
    monkeypatch.setattr(provider, "_get", get)
    configs = [{"id": "btc", "candidates": ["BTC-USDT-SWAP", "BTC-USDT"]},
               {"id": "xaut", "candidates": ["XAUT-USDT-SWAP", "XAUT-USDT"]},
               {"id": "bad", "candidates": ["CLUSDT"]}]
    btc, gold, missing = provider.resolve_assets(configs)
    assert btc["inst_id"] == "BTC-USDT-SWAP"
    assert "short" in btc["allowed_directions"]
    assert gold["inst_id"] == "XAUT-USDT"
    assert gold["allowed_directions"] == ["long", "wait"]
    assert not missing["instrument_available"]
    assert missing["inst_id"] is None
    provider.resolve_assets(configs)
    assert calls == ["SPOT", "SWAP"]  # catalogue reused for this provider


def test_catalog_failure_does_not_hide_other_instrument_type(monkeypatch):
    provider = OKXMarketProvider()
    def get(path, params):
        if params["instType"] == "SPOT":
            raise ValueError("unavailable")
        return [{"instId": "BTC-USDT-SWAP", "instType": "SWAP", "state": "live"}]
    monkeypatch.setattr(provider, "_get", get)
    btc, missing = provider.resolve_assets([{"id": "btc", "candidates": ["BTC-USDT-SWAP"]},
                                           {"id": "xaut", "candidates": ["XAUT-USDT"]}])
    assert btc["instrument_available"]
    assert not missing["instrument_available"]
    assert "SPOT catalogue" in missing["unavailable_reason"]


def test_indicators_use_completed_candles_and_real_extremes(monkeypatch):
    provider = OKXMarketProvider()
    monkeypatch.setattr("ai_agent.okx_market.time.time", lambda: NOW)
    monkeypatch.setattr(provider, "_get", lambda *args: candle_rows())
    frame = provider._timeframe("BTC-USDT-SWAP", "15m", 120)
    assert frame["candle_count"] == 100
    assert frame["close"] == 110
    assert frame["rsi14"] == 100
    assert 100 < frame["ema50"] < frame["ema20"] < 110
    assert frame["atr14"] == pytest.approx(2)
    assert frame["resistance"] == 111
    assert frame["support"] == pytest.approx(107.1)
    assert frame["available"]


@pytest.mark.parametrize("fault", ["stale", "short", "gap", "invalid"])
def test_bad_candles_are_unavailable(monkeypatch, fault):
    provider = OKXMarketProvider()
    monkeypatch.setattr("ai_agent.okx_market.time.time", lambda: NOW)
    rows = candle_rows()
    if fault == "stale":
        for row in rows:
            row[0] = str(int(row[0]) - 10 * 900 * 1000)
    elif fault == "short":
        rows = rows[:20]
    elif fault == "gap":
        rows.pop(20)
    else:
        rows[1][2] = "1"  # high below the close
    monkeypatch.setattr(provider, "_get", lambda *args: rows)
    with pytest.raises(ValueError):
        provider._timeframe("BTC-USDT-SWAP", "15m", 120)


def test_stale_ticker_cannot_create_available_market(monkeypatch):
    provider = OKXMarketProvider()
    monkeypatch.setattr("ai_agent.okx_market.time.time", lambda: NOW)
    ticker = {"instId": "BTC-USDT-SWAP", "ts": str(int((NOW - 180) * 1000)),
              "last": "100", "bidPx": "99", "askPx": "101", "open24h": "100", "volCcy24h": "10"}
    def get(path, params):
        return candle_rows() if path.endswith("candles") else ([ticker] if path.endswith("ticker") else [])
    monkeypatch.setattr(provider, "_get", get)
    result = provider.collect_asset(resolved_asset(), ["15m"])
    assert not result["market_available"]
    assert not result["data_complete"]
    assert result["ticker"] is None
    assert result["timeframes"]["15m"]["available"]
    assert "stale" in result["errors"][0]


def test_quote_volume_conversion_is_explicit(monkeypatch):
    provider = OKXMarketProvider()
    monkeypatch.setattr("ai_agent.okx_market.time.time", lambda: NOW)
    ticker = {"instId": "BTC-USDT-SWAP", "ts": str(int(NOW * 1000)),
              "last": "100", "bidPx": "99", "askPx": "101", "open24h": "80", "volCcy24h": "10"}
    monkeypatch.setattr(provider, "_get", lambda *args: [ticker])
    swap = provider._ticker(resolved_asset())
    spot = provider._ticker(resolved_asset("SPOT"))
    assert swap["volume_quote_24h"] == 1000
    assert swap["volume_quote_24h_estimated"]
    assert spot["volume_quote_24h"] == 10
    assert not spot["volume_quote_24h_estimated"]
    assert swap["change_24h_pct"] == 25


def test_partial_timeframe_failure_is_visible(monkeypatch):
    provider = OKXMarketProvider()
    monkeypatch.setattr(provider, "_ticker", lambda asset: {"last": 100})
    def frame(inst_id, bar, limit):
        if bar == "4H":
            raise ValueError("missing")
        return {"available": True}
    monkeypatch.setattr(provider, "_timeframe", frame)
    monkeypatch.setattr(provider, "_get", lambda *args: [])
    result = provider.collect_asset(resolved_asset(), ["15m", "4H"])
    assert result["market_available"]
    assert not result["data_complete"]
    assert not result["timeframes"]["4H"]["available"]
    assert "missing" in result["errors"][0]


def test_unavailable_instrument_never_requests_market_data(monkeypatch):
    provider = OKXMarketProvider()
    monkeypatch.setattr(provider, "_get", lambda *args: pytest.fail("unexpected network request"))
    result = provider.collect_asset({"id": "missing", "instrument_available": False,
                                     "unavailable_reason": "not listed"}, ["15m"])
    assert not result["market_available"]
    assert result["errors"] == ["not listed"]


def test_website_session_exposes_liquidity_without_underlying_price(monkeypatch):
    provider = OKXMarketProvider()
    monkeypatch.setattr("ai_agent.okx_market.time.time", lambda: NOW)
    response = [{"current": {"sessionCode": "overnight", "liquidity": "low", "serverTs": NOW * 1000},
                 "market": {"timezone": "America/New_York", "marketSubText": "Token trades 24/7; reference liquidity varies."}}]
    monkeypatch.setattr(provider, "_get", lambda *args: response)
    session = provider._market_session("XSOXL-USDT")
    assert session["session_code"] == "overnight"
    assert session["liquidity"] == "low"
    assert session["timezone"] == "America/New_York"
    assert session["source_type"] == "public_website_api"
    assert not session["underlying_price_available"]


def test_optional_session_failure_does_not_invalidate_fresh_market(monkeypatch):
    provider = OKXMarketProvider()
    monkeypatch.setattr(provider, "_ticker", lambda asset: {"last": 100})
    monkeypatch.setattr(provider, "_timeframe", lambda *args: {"available": True})
    def fail_session(inst_id):
        raise ValueError("website API unavailable")
    monkeypatch.setattr(provider, "_market_session", fail_session)
    asset = {**resolved_asset("SPOT"), "id": "XSOXL", "inst_id": "XSOXL-USDT"}
    result = provider.collect_asset(asset, ["15m"])
    assert result["market_available"]
    assert result["data_complete"]
    assert result["market_session"] is None
    assert result["errors"][0].startswith("market_session (optional)")
