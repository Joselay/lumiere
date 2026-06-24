from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from lumiere.backtest import BacktestConfig, Backtester, CostModel
from lumiere.engine import TradingEngine
from lumiere.models import (
    AccountSnapshot,
    DecisionAction,
    MarketCandle,
    OrderRequest,
    OrderResult,
    StrategyDecision,
)
from lumiere.paper_gate import PerformanceGateConfig
from lumiere.paper_trading import PaperTradingConfig, PaperTradingLedger
from lumiere.risk import RiskConfig, RiskManager


class BuyFirstCandleStrategy:
    name = "buy_first_candle"

    class Config:
        inst_id = "BTC-USDT"

    config = Config()

    def describe(self) -> dict[str, str]:
        return {"name": self.name, "inst_id": self.config.inst_id}

    def decide(self, candles: list[MarketCandle], account: AccountSnapshot) -> StrategyDecision:
        if len(candles) == 1 and account.position_size(self.config.inst_id) <= 0:
            return StrategyDecision(
                DecisionAction.BUY,
                self.config.inst_id,
                Decimal("0.001"),
                "first_candle",
                {"decision_price": str(candles[-1].close), "expected_edge_bps": "100"},
            )
        return StrategyDecision.hold(self.config.inst_id, "no_signal")


class FakeClient:
    def __init__(self, candles: list[MarketCandle]) -> None:
        self.candles = candles
        self.account = AccountSnapshot(equity_usdt=Decimal("1000"), available_usdt=Decimal("1000"))
        self.orders: list[OrderRequest] = []

    async def fetch_candles(self, inst_id: str | None = None) -> list[MarketCandle]:
        _ = inst_id
        return self.candles[:1]

    async def fetch_account_snapshot(self) -> AccountSnapshot:
        return self.account

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


def market() -> list[MarketCandle]:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    return [
        MarketCandle(start, Decimal("100"), Decimal("100"), Decimal("100"), Decimal("100")),
        MarketCandle(
            start + timedelta(minutes=1),
            Decimal("100"),
            Decimal("100"),
            Decimal("100"),
            Decimal("100"),
        ),
    ]


def zero_cost_model() -> CostModel:
    return CostModel(taker_fee_bps=0, spread_bps=0, slippage_bps=0)


def parity_risk() -> RiskConfig:
    return RiskConfig(cooldown_seconds=0, max_position_btc=Decimal("0.005"))


@pytest.mark.asyncio
async def test_fake_exchange_parity_across_backtest_paper_and_live(tmp_path) -> None:
    candles = market()
    strategy = BuyFirstCandleStrategy()
    risk_config = parity_risk()

    backtest = Backtester(
        strategy,
        BacktestConfig(
            cost_model=zero_cost_model(),
            risk_config=risk_config,
            include_same_close_comparison=False,
        ),
    ).run(candles)

    paper = PaperTradingLedger(
        PaperTradingConfig(
            path=tmp_path / "paper.jsonl",
            cost_model=zero_cost_model(),
            gate=PerformanceGateConfig(min_trades=1, min_profit_factor=None),
            risk_config=risk_config,
        )
    )
    paper.record_decision(
        strategy.decide(candles[:1], paper.account_snapshot()),
        candles[0],
        strategy_name=strategy.name,
    )

    client = FakeClient(candles)
    engine = TradingEngine(client, strategy, RiskManager(risk_config))
    await engine.tick()

    paper_fill = next(event for event in paper.events if event["type"] == "simulated_fill")
    assert backtest.metrics.trade_count == 1
    assert backtest.metrics.exposure_curve[-1].exposure_usdt == Decimal("0.100")
    assert paper_fill["size_base"] == "0.001"
    assert [order.size_btc for order in client.orders] == [Decimal("0.001")]


@pytest.mark.asyncio
async def test_live_engine_blocks_post_clamp_below_minimum_without_order_failure() -> None:
    candles = market()
    client = FakeClient(candles)
    tiny_exposure_risk = RiskConfig(
        cooldown_seconds=0,
        min_order_btc=Decimal("0.00001"),
        max_portfolio_exposure_pct=Decimal("0.0000001"),
    )
    engine = TradingEngine(client, BuyFirstCandleStrategy(), RiskManager(tiny_exposure_risk))

    await engine.tick()

    assert client.orders == []
    assert engine.status().last_risk_reason == "clamped_order_size_below_minimum"
    assert engine.status().consecutive_failures == 0


def test_backtest_and_paper_record_risk_rejections_and_blocked_notional(tmp_path) -> None:
    candles = market()
    strategy = BuyFirstCandleStrategy()
    blocking_risk = RiskConfig(cooldown_seconds=0, max_position_btc=Decimal("0.0005"))

    backtest = Backtester(
        strategy,
        BacktestConfig(cost_model=zero_cost_model(), risk_config=blocking_risk),
    ).run(candles)
    paper = PaperTradingLedger(
        PaperTradingConfig(
            path=tmp_path / "paper.jsonl",
            cost_model=zero_cost_model(),
            risk_config=blocking_risk,
        )
    )
    paper.record_decision(
        strategy.decide(candles[:1], paper.account_snapshot()),
        candles[0],
        strategy_name=strategy.name,
    )

    assert backtest.risk_rejection_count == 1
    assert backtest.risk_rejections == {"max_position_size_exceeded": 1}
    assert backtest.blocked_signal_opportunity_cost_usdt == Decimal("0.100")
    rejection = next(event for event in paper.events if event["type"] == "simulated_rejection")
    assert rejection["reason"] == "max_position_size_exceeded"
    assert rejection["blocked_signal_notional_usdt"] == "0.100"
