from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from io import StringIO

import pytest

from lumiere.attribution import AttributionLedger
from lumiere.engine import EngineConfig, TradingEngine
from lumiere.logging_config import configure_logging
from lumiere.models import DecisionAction, MarketCandle, OrderRequest
from lumiere.risk import RiskConfig, RiskManager
from lumiere.strategy import MovingAverageCrossoverConfig, MovingAverageCrossoverStrategy
from tests.fakes import DeterministicFakeExchange, FakeExchangeError


def candles(closes: list[str]) -> list[MarketCandle]:
    start = datetime(2026, 6, 24, tzinfo=UTC)
    return [
        MarketCandle(
            ts=start + timedelta(minutes=index),
            open=Decimal(close),
            high=Decimal(close),
            low=Decimal(close),
            close=Decimal(close),
        )
        for index, close in enumerate(closes)
    ]


class CollectingNotifier:
    def __init__(self) -> None:
        self.messages: list[str] = []

    async def send(self, text: str) -> None:
        self.messages.append(text)


@pytest.mark.asyncio
async def test_fake_exchange_engine_tick_reaches_fill_attribution_and_telegram(tmp_path) -> None:
    exchange = DeterministicFakeExchange(
        {"BTC-USDT": candles(["100", "101", "110"])},
        orderbooks_by_inst={
            "BTC-USDT": {
                "bids": [["109.90", "2"]],
                "asks": [["110.10", "2"]],
            }
        },
    )
    notifier = CollectingNotifier()
    attribution = AttributionLedger(tmp_path / "attribution.jsonl")
    engine = TradingEngine(
        client=exchange,
        strategy=MovingAverageCrossoverStrategy(
            MovingAverageCrossoverConfig(
                fast_window=2,
                slow_window=3,
                trade_size_btc=Decimal("0.5"),
            )
        ),
        risk_manager=RiskManager(RiskConfig(cooldown_seconds=0, max_position_btc=Decimal("1"))),
        notifier=notifier,
        attribution_ledger=attribution,
    )

    await engine.tick()

    assert len(exchange.orders) == 1
    assert exchange.orders[0].side is DecisionAction.BUY
    assert "<b>BUY BTC-USDT</b>" in notifier.messages[-1]
    event_types = [event["type"] for event in attribution.events]
    assert event_types == ["account", "candle", "decision", "risk", "order", "fill", "account"]
    fill_event = next(event for event in attribution.events if event["type"] == "fill")
    assert fill_event["order_id"] == "ord-1"
    assert fill_event["trade_id"] == "trade-1-1"
    assert Decimal(fill_event["price"]) > Decimal(fill_event["decision_price"])

    report = attribution.report().to_dict()
    assert report["metrics"]["trade_count"] == 1
    assert Decimal(str(report["metrics"]["fees_usdt"])) > 0
    assert Decimal(str(report["metrics"]["average_slippage_bps"])) > 0
    performance_text = await engine.performance_text()
    assert "<b>Attribution ledger</b>" in performance_text
    assert "Net PnL:" in performance_text


@pytest.mark.asyncio
async def test_fake_exchange_scenario_catches_cost_gate_rejections(tmp_path) -> None:
    exchange = DeterministicFakeExchange(
        {"BTC-USDT": candles(["100", "101", "102"])},
        orderbooks_by_inst={
            "BTC-USDT": {
                "bids": [["90", "1"]],
                "asks": [["130", "1"]],
            }
        },
    )
    notifier = CollectingNotifier()
    attribution = AttributionLedger(tmp_path / "attribution.jsonl")
    engine = TradingEngine(
        client=exchange,
        strategy=MovingAverageCrossoverStrategy(
            MovingAverageCrossoverConfig(
                fast_window=2,
                slow_window=3,
                trade_size_btc=Decimal("0.5"),
            )
        ),
        risk_manager=RiskManager(RiskConfig(cooldown_seconds=0, max_position_btc=Decimal("1"))),
        notifier=notifier,
        attribution_ledger=attribution,
    )

    await engine.tick()

    assert exchange.orders == []
    assert any("Risk blocked trade" in message for message in notifier.messages)
    blocked_risk_events = [
        event for event in attribution.events if event["type"] == "risk" and not event["allowed"]
    ]
    assert blocked_risk_events[0]["reason"] == "expected_edge_below_cost"
    assert attribution.report().rejection_counts == {"expected_edge_below_cost": 1}


@pytest.mark.asyncio
async def test_fake_exchange_models_partials_rejects_balances_and_api_failures() -> None:
    exchange = DeterministicFakeExchange(
        {"BTC-USDT": candles(["100", "101", "110"])},
        orderbooks_by_inst={"BTC-USDT": {"bids": [["109", "1"]], "asks": [["111", "1"]]}},
        fill_splits_by_order_number={1: (Decimal("0.25"), Decimal("0.75"))},
        reject_order_numbers={2},
    )

    first = await exchange.place_market_order(
        OrderRequest("BTC-USDT", DecisionAction.BUY, Decimal("1"))
    )
    fills = await exchange.fetch_order_fills(first, decision_price=Decimal("110"))
    snapshot = await exchange.fetch_account_snapshot()
    second = await exchange.place_market_order(
        OrderRequest("BTC-USDT", DecisionAction.BUY, Decimal("1"))
    )

    assert first.status == "filled"
    assert [fill.size_base for fill in fills] == [Decimal("0.25"), Decimal("0.75")]
    assert snapshot.position_size("BTC-USDT") == Decimal("1.00")
    assert snapshot.available_usdt < Decimal("1000")
    assert second.status == "rejected"
    failing_exchange = DeterministicFakeExchange(
        {"BTC-USDT": candles(["100"])},
        fail_on={"account"},
    )
    with pytest.raises(FakeExchangeError, match="fake account failure"):
        await failing_exchange.fetch_account_snapshot()


@pytest.mark.asyncio
async def test_run_forever_sends_lifecycle_messages_and_stops_cleanly() -> None:
    exchange = DeterministicFakeExchange({"BTC-USDT": candles(["100"])})
    notifier = CollectingNotifier()
    engine = TradingEngine(
        client=exchange,
        strategy=MovingAverageCrossoverStrategy(
            MovingAverageCrossoverConfig(fast_window=1, slow_window=2)
        ),
        risk_manager=RiskManager(RiskConfig(cooldown_seconds=0)),
        notifier=notifier,
        config=EngineConfig(poll_interval_seconds=0.01),
    )

    async def stop_soon() -> None:
        await asyncio.sleep(0.03)
        await engine.stop()

    await asyncio.wait_for(asyncio.gather(engine.run_forever(), stop_soon()), timeout=1)

    assert "Lumiere started" in notifier.messages[0]
    assert "Lumiere stopped" in notifier.messages[-1]
    assert engine.status().running is False


@pytest.mark.asyncio
async def test_structured_exception_logging_path_does_not_crash() -> None:
    stream = StringIO()
    configure_logging("INFO", stream=stream, force_colors=False)
    exchange = DeterministicFakeExchange(
        {"BTC-USDT": candles(["100", "101", "110"])},
        fail_on={"account"},
    )
    notifier = CollectingNotifier()
    engine = TradingEngine(
        client=exchange,
        strategy=MovingAverageCrossoverStrategy(
            MovingAverageCrossoverConfig(fast_window=2, slow_window=3)
        ),
        risk_manager=RiskManager(RiskConfig(cooldown_seconds=0)),
        notifier=notifier,
    )

    await engine.tick()

    assert "Trading error" in notifier.messages[-1]
    log_output = stream.getvalue()
    assert "engine_tick_failed" in log_output
    assert "fake account failure" in log_output
