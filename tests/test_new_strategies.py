from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

from lumiere.backtest import BacktestConfig, Backtester, CostModel
from lumiere.models import AccountSnapshot, DecisionAction, MarketCandle, Position
from lumiere.strategies import (
    RsiMeanReversionConfig,
    RsiMeanReversionStrategy,
    VolatilityBreakoutConfig,
    VolatilityBreakoutStrategy,
)


def candles(closes: list[str]) -> list[MarketCandle]:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    rows = []
    for index, close in enumerate(closes):
        price = Decimal(close)
        rows.append(
            MarketCandle(
                ts=start + timedelta(minutes=index),
                open=price,
                high=price + Decimal("1"),
                low=price - Decimal("1"),
                close=price,
            )
        )
    return rows


def account(position: str = "0") -> AccountSnapshot:
    positions = ()
    if Decimal(position) > 0:
        positions = (Position("BTC-USDT", Decimal(position)),)
    return AccountSnapshot(
        equity_usdt=Decimal("1000"),
        available_usdt=Decimal("1000"),
        positions=positions,
        spread_bps=Decimal("1"),
    )


def test_rsi_mean_reversion_buys_oversold_ranging_regime() -> None:
    strategy = RsiMeanReversionStrategy(
        RsiMeanReversionConfig(rsi_period=3, oversold_rsi=Decimal("40"))
    )

    decision = strategy.decide(candles(["100", "95", "90", "89"]), account())

    assert decision.action is DecisionAction.BUY
    assert decision.reason == "rsi_oversold_in_ranging_regime"
    assert "ranging" in decision.inputs["allowed_regimes"]
    assert "allowed_regimes" in strategy.describe()


def test_rsi_mean_reversion_sells_overbought_exit() -> None:
    strategy = RsiMeanReversionStrategy(
        RsiMeanReversionConfig(rsi_period=3, overbought_rsi=Decimal("60"))
    )

    decision = strategy.decide(candles(["90", "95", "100", "105"]), account("0.001"))

    assert decision.action is DecisionAction.SELL
    assert decision.reason == "rsi_overbought_exit"


def test_volatility_breakout_buys_trending_high_volatility_regime() -> None:
    strategy = VolatilityBreakoutStrategy(
        VolatilityBreakoutConfig(
            lookback=3,
            atr_period=3,
            atr_multiplier=Decimal("0"),
            min_atr_pct=Decimal("0"),
        )
    )

    decision = strategy.decide(candles(["100", "101", "102", "110"]), account())

    assert decision.action is DecisionAction.BUY
    assert decision.reason == "atr_breakout_above_prior_high"
    assert "trending" in decision.inputs["allowed_regimes"]
    assert "allowed_regimes" in strategy.describe()


def test_new_strategy_backtest_reports_allowed_regimes() -> None:
    strategy = VolatilityBreakoutStrategy(
        VolatilityBreakoutConfig(
            lookback=3,
            atr_period=3,
            atr_multiplier=Decimal("0"),
            min_atr_pct=Decimal("0"),
            trade_size_btc=Decimal("1"),
        )
    )

    report = Backtester(
        strategy,
        BacktestConfig(cost_model=CostModel(taker_fee_bps=Decimal("0"))),
    ).run(candles(["100", "101", "102", "110", "111"]))

    assert report.parameters["name"] == "volatility_breakout"
    assert "trending" in report.parameters["allowed_regimes"]
