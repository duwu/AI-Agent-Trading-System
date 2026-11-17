"""Regression tests for cross-asset decisions and evidence validation."""

from datetime import datetime, timedelta, timezone
import json

import pytest

from ai_agent.portfolio_decision import number, parse_decision, render_report, validate_decision


NOW = datetime(2026, 11, 17, 3, 0, tzinfo=timezone.utc)


@pytest.fixture
def config():
    return {"horizon": "未来1–3天", "min_reward_risk": 1.5, "round_trip_cost_bps": 20,
            "min_confidence": 0.6, "max_spread_bps": 50, "max_snapshot_age_seconds": 900,
            "news_lookback_hours": 48}


@pytest.fixture
def snapshot():
    def market(asset_id, kind):
        return {"id": asset_id, "name": asset_id, "exposure": "test exposure",
                "inst_id": f"{asset_id}-USDT" + ("-SWAP" if kind == "SWAP" else ""),
                "inst_type": kind, "instrument_available": True,
                "allowed_directions": ["long", "short", "wait"] if kind == "SWAP" else ["long", "wait"],
                "market_available": True, "data_complete": True,
                "ticker": {"last": 100.5, "spread_bps": 2, "observed_at": NOW.isoformat()},
                "timeframes": {"15m": {"available": True, "timestamp": NOW.isoformat()}},
                "errors": []}
    def article(asset_id):
        return {"id": f"n_{asset_id}", "title": f"Recent {asset_id} headline", "source": "Reuters",
                "url": f"https://example.com/{asset_id}", "published_at": (NOW - timedelta(hours=1)).isoformat(),
                "asset_ids": [asset_id], "scope": "asset"}
    return {"started_at": NOW.isoformat(), "markets": [market("BTC", "SWAP"), market("XAUT", "SPOT")],
            "news": {"articles": [article("BTC"), article("XAUT")], "errors": [],
                     "lookback_hours": 48, "collected_at": NOW.isoformat()}}


@pytest.fixture
def raw():
    def row(asset_id, rank):
        return {"asset_id": asset_id, "rank": rank, "direction": "long", "confidence": 0.8,
                "rationale": "行情与近期报道支持条件交易。", "catalysts": ["近期报道"],
                "risks": ["走势可能反转"], "invalidation": "跌破支撑后失效",
                "evidence_ids": [f"n_{asset_id}"], "entry_low": 100.0, "entry_high": 101.0,
                "entry_trigger": "价格进入区间且支撑有效", "stop_loss": 98.0, "take_profit": 108.0}
    return {"market_summary": "比较两个资产", "assessments": [row("BTC", 1), row("XAUT", 2)]}


def by_id(decision):
    return {row["asset_id"]: row for row in decision["assessments"]}


def test_parse_nested_json_and_markdown_without_regex_damage(raw):
    raw["assessments"][0]["extra"] = {"nested": [1, {"quote": 'a "quoted" {value}'}]}
    text = 'Explanation {"unrelated": true}\n```json\n' + json.dumps(raw, ensure_ascii=False) + "\n```"
    assert parse_decision(text) == raw


@pytest.mark.parametrize("literal", ["NaN", "Infinity", "-Infinity"])
def test_parser_rejects_nonfinite_json_even_in_unused_field(literal):
    with pytest.raises(ValueError):
        parse_decision('{"assessments": [], "unrecognized": ' + literal + "}")


@pytest.mark.parametrize("mutate", ["missing", "duplicate", "unknown", "rank_duplicate", "rank_gap", "rank_bool"])
def test_requires_complete_unique_assets_and_contiguous_integer_ranks(raw, snapshot, config, mutate):
    if mutate == "missing":
        raw["assessments"].pop()
    elif mutate == "duplicate":
        raw["assessments"][1]["asset_id"] = "BTC"
    elif mutate == "unknown":
        raw["assessments"][1]["asset_id"] = "UNKNOWN"
    elif mutate == "rank_duplicate":
        raw["assessments"][1]["rank"] = 1
    elif mutate == "rank_gap":
        raw["assessments"][1]["rank"] = 3
    else:
        raw["assessments"][0]["rank"] = True
    with pytest.raises(ValueError):
        validate_decision(raw, snapshot, config, now=NOW)


def test_valid_long_has_cost_adjusted_worst_entry_ratio(raw, snapshot, config):
    decision = validate_decision(raw, snapshot, config, now=NOW)
    btc = by_id(decision)["BTC"]
    assert btc["direction"] == "long"
    assert btc["reward_risk_after_cost"] == round((108 - 101 - 0.202) / (101 - 98 + 0.202), 3)
    assert decision["preferred_asset"] == "BTC"


def test_valid_swap_short_uses_lowest_entry_price(raw, snapshot, config):
    raw["assessments"][0].update(direction="short", stop_loss=104.0, take_profit=91.0)
    btc = by_id(validate_decision(raw, snapshot, config, now=NOW))["BTC"]
    assert btc["direction"] == "short"
    assert btc["reward_risk_after_cost"] == round((100 - 91 - 0.2) / (104 - 100 + 0.2), 3)


def test_spot_short_is_rejected_and_original_opinion_preserved(raw, snapshot, config):
    raw["assessments"][1].update(direction="short", stop_loss=104.0, take_profit=91.0)
    gold = by_id(validate_decision(raw, snapshot, config, now=NOW))["XAUT"]
    assert gold["direction"] == "wait"
    assert gold["ai_direction"] == "short"
    assert gold["validation_notes"]
    assert all(gold[field] is None for field in ("entry_low", "entry_high", "stop_loss", "take_profit"))


@pytest.mark.parametrize("direction,target,stop", [("long", 105.6, 98), ("short", 93.9, 104)])
def test_costs_can_invalidate_otherwise_adequate_gross_reward_risk(raw, snapshot, config, direction, target, stop):
    raw["assessments"][0].update(direction=direction, take_profit=target, stop_loss=stop)
    btc = by_id(validate_decision(raw, snapshot, config, now=NOW))["BTC"]
    assert btc["direction"] == "wait"
    assert any("盈亏比" in note for note in btc["validation_notes"])


@pytest.mark.parametrize("direction,low,high,stop,target", [
    ("long", 100, 101, 102, 108), ("long", 100, 101, 98, 99),
    ("short", 100, 101, 99, 91), ("short", 100, 101, 104, 102),
    ("long", 101, 100, 98, 108), ("short", 101, 100, 104, 91),
])
def test_inverted_or_wrong_side_prices_never_become_actionable(raw, snapshot, config, direction, low, high, stop, target):
    raw["assessments"][0].update(direction=direction, entry_low=low, entry_high=high, stop_loss=stop, take_profit=target)
    assert by_id(validate_decision(raw, snapshot, config, now=NOW))["BTC"]["direction"] == "wait"


@pytest.mark.parametrize("value", [True, False, float("nan"), float("inf"), float("-inf"), "100", None])
def test_nonfinite_bool_and_non_numeric_values_are_not_numbers(value):
    with pytest.raises(ValueError):
        number(value)


@pytest.mark.parametrize("value", [True, float("nan"), float("inf"), "100", -1, 0])
def test_invalid_prices_are_wait_not_trade(raw, snapshot, config, value):
    raw["assessments"][0]["entry_low"] = value
    assert by_id(validate_decision(raw, snapshot, config, now=NOW))["BTC"]["direction"] == "wait"


@pytest.mark.parametrize("value", [True, float("nan"), float("inf"), -0.1, 1.1])
def test_invalid_confidence_is_wait(raw, snapshot, config, value):
    raw["assessments"][0]["confidence"] = value
    assert by_id(validate_decision(raw, snapshot, config, now=NOW))["BTC"]["direction"] == "wait"


def test_missing_ticker_is_wait_and_does_not_break_other_assets(raw, snapshot, config):
    snapshot["markets"][0].update(ticker=None, market_available=False, data_complete=False)
    decision = validate_decision(raw, snapshot, config, now=NOW)
    rows = by_id(decision)
    assert rows["BTC"]["direction"] == "wait"
    assert rows["XAUT"]["direction"] == "long"
    assert decision["preferred_asset"] == "XAUT"
    assert rows["BTC"]["ai_rank"] == 1
    assert rows["BTC"]["rank"] == 2


def test_partial_candles_force_only_that_asset_to_wait(raw, snapshot, config):
    snapshot["markets"][0]["data_complete"] = False
    rows = by_id(validate_decision(raw, snapshot, config, now=NOW))
    assert rows["BTC"]["direction"] == "wait"
    assert rows["XAUT"]["direction"] == "long"


def test_expired_snapshot_has_no_preferred_trade(raw, snapshot, config):
    decision = validate_decision(raw, snapshot, config, now=NOW + timedelta(seconds=901))
    assert decision["preferred_asset"] is None
    assert all(row["direction"] == "wait" for row in decision["assessments"])


@pytest.mark.parametrize("quoted_at", [(NOW - timedelta(seconds=901)).isoformat(), None, "bad-date"])
def test_expired_or_unverifiable_quote_is_wait_despite_recent_snapshot(raw, snapshot, config, quoted_at):
    snapshot["markets"][0]["ticker"]["observed_at"] = quoted_at
    rows = by_id(validate_decision(raw, snapshot, config, now=NOW))
    assert rows["BTC"]["direction"] == "wait"
    assert rows["XAUT"]["direction"] == "long"


def test_missing_asset_news_forces_wait(raw, snapshot, config):
    snapshot["news"]["articles"] = [snapshot["news"]["articles"][1]]
    rows = by_id(validate_decision(raw, snapshot, config, now=NOW))
    assert rows["BTC"]["direction"] == "wait"
    assert rows["XAUT"]["direction"] == "long"


@pytest.mark.parametrize("offset", [timedelta(hours=-49), timedelta(hours=1)])
def test_old_or_future_news_cannot_support_trade(raw, snapshot, config, offset):
    snapshot["news"]["articles"][0]["published_at"] = (NOW + offset).isoformat()
    rows = by_id(validate_decision(raw, snapshot, config, now=NOW))
    assert rows["BTC"]["direction"] == "wait"


def test_macro_only_citations_do_not_hide_missing_asset_coverage(raw, snapshot, config):
    snapshot["news"]["articles"][0]["scope"] = "macro"
    rows = by_id(validate_decision(raw, snapshot, config, now=NOW))
    assert rows["BTC"]["direction"] == "wait"


@pytest.mark.parametrize("refs", [["hallucinated"], ["n_XAUT"], ["n_BTC", "hallucinated"], []])
def test_hallucinated_wrong_asset_and_missing_citations_force_wait(raw, snapshot, config, refs):
    raw["assessments"][0]["evidence_ids"] = refs
    btc = by_id(validate_decision(raw, snapshot, config, now=NOW))["BTC"]
    assert btc["direction"] == "wait"
    assert all(ref == "n_BTC" for ref in btc["evidence_ids"])


def test_render_collect_only_report_handles_unavailable_market(snapshot):
    snapshot["markets"][0].update(ticker=None, market_available=False, data_complete=False)
    report = render_report({"snapshot": snapshot, "horizon": "未来1–3天", "status": "collected", "decision": None})
    assert "BTC" in report
    assert "缺失" in report


def test_render_decision_handles_unavailable_market(snapshot, raw, config):
    decision = validate_decision(raw, snapshot, config, now=NOW)
    # The reporter must tolerate the exact shape produced for a failed quote.
    snapshot["markets"][0].update(ticker=None, market_available=False, data_complete=False)
    report = render_report({"snapshot": snapshot, "horizon": "未来1–3天", "status": "completed",
                            "decision": decision, "model": "test", "ai_elapsed_seconds": 1})
    assert "BTC" in report
    assert "缺失" in report
