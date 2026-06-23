from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

from lumiere.backtest import BacktestConfig, Backtester, CostModel
from lumiere.models import MarketCandle
from lumiere.strategy import MovingAverageCrossoverConfig, MovingAverageCrossoverStrategy


def candles(closes: list[str]) -> list[MarketCandle]:
    start = datetime(2026, 1, 1, tzinfo=UTC)
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


def test_backtest_report_includes_net_pnl_after_fee_spread_and_slippage() -> None:
    strategy = MovingAverageCrossoverStrategy(
        MovingAverageCrossoverConfig(fast_window=2, slow_window=3, trade_size_btc=Decimal("1"))
    )
    report = Backtester(
        strategy,
        BacktestConfig(
            starting_equity_usdt=Decimal("1000"),
            cost_model=CostModel(
                taker_fee_bps=Decimal("10"),
                spread_bps=Decimal("2"),
                slippage_bps=Decimal("5"),
            ),
        ),
    ).run(candles(["100", "101", "110", "100", "90", "120", "130"]))

    assert report.metrics.trade_count >= 2
    assert report.metrics.fees_usdt > 0
    assert report.metrics.net_pnl_usdt == report.metrics.ending_equity_usdt - Decimal("1000")
    assert report.buy_and_hold_pnl_usdt != report.no_trade_pnl_usdt
    assert report.assumptions["taker_fee_bps"] == "10"


def test_backtest_models_rejected_orders() -> None:
    strategy = MovingAverageCrossoverStrategy(
        MovingAverageCrossoverConfig(fast_window=2, slow_window=3, trade_size_btc=Decimal("1"))
    )
    report = Backtester(
        strategy,
        BacktestConfig(
            starting_equity_usdt=Decimal("1000"),
            cost_model=CostModel(reject_every_n_orders=1),
        ),
    ).run(candles(["100", "101", "110", "100", "90"]))

    assert report.metrics.trade_count == 0
    assert report.rejected_order_count > 0
