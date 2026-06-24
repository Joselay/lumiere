from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from lumiere.backtest import BacktestConfig, Backtester, CostModel
from lumiere.engine import EngineConfig, TradingEngine
from lumiere.models import (
    AccountSnapshot,
    DecisionAction,
    MarketCandle,
    OrderRequest,
    OrderResult,
    StrategyDecision,
)
from lumiere.paper_trading import PaperTradingConfig, PaperTradingLedger
from lumiere.risk import RiskConfig, RiskManager


class BuyOnceStrategy:
    name = "buy_once"

    class Config:
        inst_id = "BTC-USDT"

    config = Config()

    def describe(self) -> dict[str, str]:
        return {"name": self.name, "inst_id": self.config.inst_id}

    def decide(self, candles: list[MarketCandle], account: AccountSnapshot) -> StrategyDecision:
        if account.position_size(self.config.inst_id) <= 0:
            return StrategyDecision(
                DecisionAction.BUY,
                self.config.inst_id,
                Decimal("1"),
                "buy_once",
                {"decision_price": str(candles[-1].close), "expected_edge_bps": "100"},
            )
        return StrategyDecision.hold(self.config.inst_id, "already_long")


def market() -> list[MarketCandle]:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    return [
        MarketCandle(start, Decimal("100"), Decimal("100"), Decimal("100"), Decimal("100")),
        MarketCandle(
            start + timedelta(minutes=1),
            Decimal("100"),
            Decimal("101"),
            Decimal("99.80"),
            Decimal("99.90"),
        ),
        MarketCandle(
            start + timedelta(minutes=2),
            Decimal("103"),
            Decimal("104"),
            Decimal("102"),
            Decimal("103"),
        ),
    ]


def test_backtest_post_only_maker_reports_fill_quality_and_adverse_selection() -> None:
    report = Backtester(
        BuyOnceStrategy(),
        BacktestConfig(
            starting_equity_usdt=Decimal("1000"),
            cost_model=CostModel(
                taker_fee_bps=Decimal("10"),
                maker_fee_bps=Decimal("2"),
                spread_bps=Decimal("10"),
                slippage_bps=Decimal("0"),
                execution_policy="post_only_maker",
                maker_fill_fraction=Decimal("0.5"),
            ),
            include_same_close_comparison=False,
        ),
    ).run(market())

    assert report.metrics.trade_count == 1
    assert report.execution_quality is not None
    quality = report.execution_quality
    assert quality["policy"] == "post_only_maker"
    assert quality["attempted_order_count"] == 1
    assert quality["filled_order_count"] == 1
    assert quality["partial_fill_count"] == 1
    assert quality["non_fill_rate"] == "0"
    assert Decimal(quality["realized_spread_capture_bps"]) > 0
    assert Decimal(quality["adverse_selection_bps"]) > 0


def test_backtest_post_only_maker_can_wait_until_cancel_replace_timeout() -> None:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    candles = [
        MarketCandle(start, Decimal("100"), Decimal("100"), Decimal("100"), Decimal("100")),
        MarketCandle(
            start + timedelta(minutes=1),
            Decimal("103"),
            Decimal("105"),
            Decimal("102"),
            Decimal("103"),
        ),
        MarketCandle(
            start + timedelta(minutes=2),
            Decimal("100"),
            Decimal("101"),
            Decimal("99.80"),
            Decimal("100"),
        ),
    ]

    report = Backtester(
        BuyOnceStrategy(),
        BacktestConfig(
            starting_equity_usdt=Decimal("1000"),
            cost_model=CostModel(
                spread_bps=Decimal("10"),
                slippage_bps=Decimal("0"),
                execution_policy="post_only_maker",
                maker_timeout_bars=2,
            ),
            include_same_close_comparison=False,
        ),
    ).run(candles)

    assert report.metrics.trade_count == 1
    assert report.execution_quality is not None
    assert report.execution_quality["average_fill_delay_bars"] == "2"


def test_backtest_post_only_maker_reports_non_fill_opportunity_cost() -> None:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    candles = [
        MarketCandle(start, Decimal("100"), Decimal("100"), Decimal("100"), Decimal("100")),
        MarketCandle(
            start + timedelta(minutes=1),
            Decimal("103"),
            Decimal("105"),
            Decimal("102"),
            Decimal("105"),
        ),
    ]

    report = Backtester(
        BuyOnceStrategy(),
        BacktestConfig(
            starting_equity_usdt=Decimal("1000"),
            cost_model=CostModel(
                taker_fee_bps=Decimal("10"),
                maker_fee_bps=Decimal("2"),
                spread_bps=Decimal("10"),
                slippage_bps=Decimal("0"),
                execution_policy="post_only_maker",
                maker_timeout_bars=1,
            ),
            include_same_close_comparison=False,
        ),
    ).run(candles)

    assert report.metrics.trade_count == 0
    assert report.execution_quality is not None
    quality = report.execution_quality
    assert quality["non_fill_count"] == 1
    assert quality["non_fill_rate"] == "1"
    assert Decimal(quality["missed_trade_opportunity_cost_usdt"]) > 0


def test_paper_trading_records_maker_execution_quality(tmp_path) -> None:
    ledger = PaperTradingLedger(
        PaperTradingConfig(
            path=tmp_path / "paper.jsonl",
            cost_model=CostModel(
                maker_fee_bps=Decimal("2"),
                spread_bps=Decimal("10"),
                slippage_bps=Decimal("0"),
                execution_policy="post_only_maker",
                maker_fill_fraction=Decimal("0.5"),
            ),
        )
    )
    candle = market()[1]

    decision = BuyOnceStrategy().decide([market()[0]], ledger.account_snapshot())
    ledger.record_decision(decision, candle, strategy_name="buy_once")

    fill = next(event for event in ledger.events if event["type"] == "simulated_fill")
    assert fill["execution_policy"] == "post_only_maker"
    assert fill["size_base"] == "0.5"
    assert Decimal(fill["realized_spread_capture_bps"]) > 0
    assert Decimal(fill["adverse_selection_bps"]) > 0


def test_risk_guard_blocks_maker_mode_when_quality_is_too_poor() -> None:
    risk = RiskManager(
        RiskConfig(
            max_maker_non_fill_rate=Decimal("0.25"),
            max_maker_adverse_selection_bps=Decimal("5"),
        )
    )
    decision = StrategyDecision(
        DecisionAction.BUY,
        "BTC-USDT",
        Decimal("0.001"),
        "maker_signal",
        {"execution_policy": "post_only_maker"},
    )

    blocked = risk.assess(
        decision,
        AccountSnapshot(
            equity_usdt=Decimal("1000"),
            available_usdt=Decimal("1000"),
            maker_non_fill_rate=Decimal("0.5"),
            maker_adverse_selection_bps=Decimal("2"),
        ),
    )
    allowed = risk.assess(
        decision,
        AccountSnapshot(
            equity_usdt=Decimal("1000"),
            available_usdt=Decimal("1000"),
            maker_non_fill_rate=Decimal("0.1"),
            maker_adverse_selection_bps=Decimal("2"),
        ),
    )

    assert blocked.reason == "maker_non_fill_rate_too_high"
    assert allowed.allowed


class OrderBookClient:
    def __init__(self) -> None:
        self.orders: list[OrderRequest] = []

    async def fetch_candles(self, inst_id: str | None = None):
        _ = inst_id
        return market()[:1]

    async def fetch_account_snapshot(self) -> AccountSnapshot:
        return AccountSnapshot(equity_usdt=Decimal("1000"), available_usdt=Decimal("1000"))

    async def fetch_orderbook_top(self, inst_id: str):
        assert inst_id == "BTC-USDT"
        return {"bid": Decimal("99.90"), "ask": Decimal("100.10")}

    async def place_market_order(self, request: OrderRequest) -> OrderResult:
        self.orders.append(request)
        return OrderResult(
            "ord-1",
            "client-1",
            request.inst_id,
            request.side,
            request.size_btc,
            "ok",
        )

    async def cancel_open_orders(self) -> list[dict]:
        return []


@pytest.mark.asyncio
async def test_live_engine_builds_safe_post_only_limit_from_order_book() -> None:
    client = OrderBookClient()
    engine = TradingEngine(
        client,
        BuyOnceStrategy(),
        RiskManager(RiskConfig(cooldown_seconds=0, max_position_btc=Decimal("1"))),
        config=EngineConfig(execution_policy="post_only_maker"),
    )

    await engine.tick()

    assert len(client.orders) == 1
    order = client.orders[0]
    assert order.order_type == "post_only"
    assert order.limit_price == Decimal("99.90")
    assert order.cancel_replace_timeout_seconds == 30
