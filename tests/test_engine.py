from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from lumiere.engine import EngineConfig, TradingEngine
from lumiere.models import (
    AccountSnapshot,
    DecisionAction,
    MarketCandle,
    OrderRequest,
    OrderResult,
    Position,
)
from lumiere.paper_trading import PaperTradingConfig, PaperTradingLedger
from lumiere.risk import RiskConfig, RiskManager
from lumiere.strategy import MovingAverageCrossoverConfig, MovingAverageCrossoverStrategy


def candles(closes: list[str]) -> list[MarketCandle]:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    return [
        MarketCandle(
            ts=start + timedelta(minutes=i),
            open=Decimal(close),
            high=Decimal(close),
            low=Decimal(close),
            close=Decimal(close),
        )
        for i, close in enumerate(closes)
    ]


class FakeClient:
    def __init__(self, market: list[MarketCandle] | dict[str, list[MarketCandle]]) -> None:
        self.market = market
        self.account = AccountSnapshot(equity_usdt=Decimal("1000"), available_usdt=Decimal("1000"))
        self.orders: list[OrderRequest] = []
        self.cancelled = False

    async def fetch_candles(self, inst_id: str | None = None) -> list[MarketCandle]:
        if isinstance(self.market, dict):
            assert inst_id is not None
            return self.market[inst_id]
        return self.market

    async def fetch_account_snapshot(self) -> AccountSnapshot:
        return self.account

    async def place_market_order(self, request: OrderRequest) -> OrderResult:
        self.orders.append(request)
        return OrderResult(
            order_id="ord-1",
            client_order_id="client-1",
            inst_id=request.inst_id,
            side=request.side,
            size_btc=request.size_btc,
            status="submitted",
        )

    async def cancel_open_orders(self) -> list[dict]:
        self.cancelled = True
        return [{"ordId": "ord-1"}]


class CollectingNotifier:
    def __init__(self) -> None:
        self.messages: list[str] = []

    async def send(self, text: str) -> None:
        self.messages.append(text)


@pytest.mark.asyncio
async def test_engine_ignores_unconfirmed_latest_candle_by_default() -> None:
    market = candles(["100", "101", "110"])
    market[-1] = MarketCandle(
        ts=market[-1].ts,
        open=market[-1].open,
        high=market[-1].high,
        low=market[-1].low,
        close=market[-1].close,
        confirmed=False,
    )
    client = FakeClient(market)
    engine = TradingEngine(
        client=client,
        strategy=MovingAverageCrossoverStrategy(
            MovingAverageCrossoverConfig(
                fast_window=2,
                slow_window=3,
                trade_size_btc=Decimal("0.001"),
            )
        ),
        risk_manager=RiskManager(RiskConfig(cooldown_seconds=0)),
    )

    await engine.tick()

    assert client.orders == []
    assert engine.status().last_decision == "BTC-USDT:hold"


@pytest.mark.asyncio
async def test_engine_tick_places_order_from_strategy_signal() -> None:
    client = FakeClient(candles(["100", "101", "110"]))
    notifier = CollectingNotifier()
    engine = TradingEngine(
        client=client,
        strategy=MovingAverageCrossoverStrategy(
            MovingAverageCrossoverConfig(
                fast_window=2, slow_window=3, trade_size_btc=Decimal("0.001")
            )
        ),
        risk_manager=RiskManager(RiskConfig(cooldown_seconds=0)),
        notifier=notifier,
        config=EngineConfig(td_mode="cash"),
    )

    await engine.tick()

    assert len(client.orders) == 1
    assert client.orders[0].side is DecisionAction.BUY
    assert "<b>BUY BTC-USDT</b>" in notifier.messages[-1]


@pytest.mark.asyncio
async def test_engine_records_paper_decision_against_shadow_portfolio(tmp_path) -> None:
    client = FakeClient(candles(["110", "101", "100"]))
    # The live/demo account holds BTC, which should make the live strategy sell on a down-cross.
    # Paper is flat and must hold instead of fabricating a sell fill from live inventory.
    client.account = AccountSnapshot(
        equity_usdt=Decimal("1000"),
        available_usdt=Decimal("1000"),
        btc_position=Position("BTC-USDT", Decimal("1"), Decimal("105")),
    )
    paper = PaperTradingLedger(PaperTradingConfig(path=tmp_path / "paper.jsonl"))
    engine = TradingEngine(
        client=client,
        strategy=MovingAverageCrossoverStrategy(
            MovingAverageCrossoverConfig(
                fast_window=2, slow_window=3, trade_size_btc=Decimal("0.001")
            )
        ),
        risk_manager=RiskManager(RiskConfig(cooldown_seconds=0)),
        paper_ledger=paper,
    )

    await engine.tick()

    assert [order.side for order in client.orders] == [DecisionAction.SELL]
    assert [event["type"] for event in paper.events] == ["decision", "portfolio_state"]
    assert paper.events[0]["action"] == "hold"
    assert paper.events[0]["inputs"]["position_base"] == "0"


@pytest.mark.asyncio
async def test_engine_tick_runs_btc_and_eth_strategies() -> None:
    client = FakeClient(
        {
            "BTC-USDT": candles(["100", "101", "110"]),
            "ETH-USDT": candles(["100", "101", "110"]),
        }
    )
    notifier = CollectingNotifier()
    engine = TradingEngine(
        client=client,
        strategy=(
            MovingAverageCrossoverStrategy(
                MovingAverageCrossoverConfig(
                    inst_id="BTC-USDT",
                    fast_window=2,
                    slow_window=3,
                    trade_size_btc=Decimal("0.001"),
                )
            ),
            MovingAverageCrossoverStrategy(
                MovingAverageCrossoverConfig(
                    inst_id="ETH-USDT", fast_window=2, slow_window=3, trade_size_btc=Decimal("0.01")
                )
            ),
        ),
        risk_manager=RiskManager(
            RiskConfig(
                allowed_inst_ids=("BTC-USDT", "ETH-USDT"),
                cooldown_seconds=0,
                max_position_by_inst_id={
                    "BTC-USDT": Decimal("0.005"),
                    "ETH-USDT": Decimal("0.05"),
                },
            )
        ),
        notifier=notifier,
        config=EngineConfig(td_mode="cash"),
    )

    await engine.tick()

    assert [order.inst_id for order in client.orders] == ["BTC-USDT", "ETH-USDT"]


@pytest.mark.asyncio
async def test_engine_pause_resume_and_panic_controls() -> None:
    client = FakeClient(candles(["100", "101", "110"]))
    notifier = CollectingNotifier()
    engine = TradingEngine(
        client=client,
        strategy=MovingAverageCrossoverStrategy(
            MovingAverageCrossoverConfig(fast_window=2, slow_window=3)
        ),
        risk_manager=RiskManager(RiskConfig(cooldown_seconds=0)),
        notifier=notifier,
    )

    await engine.pause()
    await engine.tick()
    await engine.resume()
    await engine.panic()
    await engine.resume()

    assert client.orders == []
    assert client.cancelled is True
    assert engine.paused is True
    assert engine.panic_stopped is True
    assert "Cannot resume" in notifier.messages[-1]


@pytest.mark.asyncio
async def test_engine_suppresses_repeated_cooldown_push_notifications() -> None:
    client = FakeClient(candles(["100", "101", "110"]))
    notifier = CollectingNotifier()
    engine = TradingEngine(
        client=client,
        strategy=MovingAverageCrossoverStrategy(
            MovingAverageCrossoverConfig(fast_window=2, slow_window=3)
        ),
        risk_manager=RiskManager(RiskConfig(cooldown_seconds=300)),
        notifier=notifier,
    )

    await engine.tick()
    await engine.tick()

    assert len(client.orders) == 1
    assert not any("cooldown active" in message for message in notifier.messages)


@pytest.mark.asyncio
async def test_engine_pauses_after_repeated_client_failures() -> None:
    class FailingClient(FakeClient):
        async def fetch_account_snapshot(self) -> AccountSnapshot:
            raise RuntimeError("boom")

    notifier = CollectingNotifier()
    engine = TradingEngine(
        client=FailingClient(candles(["100", "101", "110"])),
        strategy=MovingAverageCrossoverStrategy(
            MovingAverageCrossoverConfig(fast_window=2, slow_window=3)
        ),
        risk_manager=RiskManager(RiskConfig(max_consecutive_failures=2)),
        notifier=notifier,
    )

    await engine.tick()
    await engine.tick()

    assert engine.paused is True
    assert "Max consecutive failures" in notifier.messages[-1]
