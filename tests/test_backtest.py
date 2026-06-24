from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

from lumiere.backtest import BacktestConfig, Backtester, CostModel
from lumiere.models import AccountSnapshot, DecisionAction, MarketCandle, StrategyDecision
from lumiere.strategy import MovingAverageCrossoverConfig, MovingAverageCrossoverStrategy


class BuyFirstClosedCandleStrategy:
    name = "buy_first_closed_candle"

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
                Decimal("1"),
                "first_closed_candle",
            )
        return StrategyDecision.hold(self.config.inst_id, "no_signal")


class BuyThenHoldStrategy(BuyFirstClosedCandleStrategy):
    def decide(self, candles: list[MarketCandle], account: AccountSnapshot) -> StrategyDecision:
        if account.position_size(self.config.inst_id) <= 0:
            return StrategyDecision(
                DecisionAction.BUY,
                self.config.inst_id,
                Decimal("1"),
                "enter",
            )
        return StrategyDecision.hold(self.config.inst_id, "already_long")


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


def test_backtest_reports_risk_of_ruin_metrics_and_exposure_curve() -> None:
    strategy = MovingAverageCrossoverStrategy(
        MovingAverageCrossoverConfig(fast_window=1, slow_window=2, trade_size_btc=Decimal("1"))
    )
    report = Backtester(
        strategy,
        BacktestConfig(
            starting_equity_usdt=Decimal("1000"),
            cost_model=CostModel(
                taker_fee_bps=Decimal("0"),
                spread_bps=Decimal("0"),
                slippage_bps=Decimal("0"),
            ),
            max_bars_in_trade=1,
        ),
    ).run(candles(["100", "101", "90", "91", "80"]))

    assert report.metrics.largest_loss_usdt >= 0
    assert report.metrics.losing_streak >= 0
    assert report.metrics.max_drawdown_duration_bars >= 0
    assert len(report.metrics.exposure_curve) == 5


def test_backtest_default_uses_next_bar_open_and_reports_same_close_comparison() -> None:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    market = [
        MarketCandle(start, Decimal("100"), Decimal("100"), Decimal("100"), Decimal("100")),
        MarketCandle(
            start + timedelta(minutes=1),
            Decimal("90"),
            Decimal("200"),
            Decimal("90"),
            Decimal("200"),
        ),
    ]

    report = Backtester(
        BuyFirstClosedCandleStrategy(),
        BacktestConfig(
            starting_equity_usdt=Decimal("1000"),
            cost_model=CostModel(taker_fee_bps=0, spread_bps=0, slippage_bps=0),
        ),
    ).run(market)

    assert report.execution_timing == "next_open"
    assert report.metrics.trade_count == 1
    assert report.metrics.ending_equity_usdt == Decimal("1110")
    assert report.assumptions["fill_price"] == "next_bar_open"
    assert report.same_close_comparison is not None
    assert report.same_close_comparison["ending_equity_usdt"] == "1100"


def test_backtest_ignores_unconfirmed_candles_and_does_not_fill_final_signal() -> None:
    market = candles(["100", "110"])
    market[1] = MarketCandle(
        market[1].ts,
        market[1].open,
        market[1].high,
        market[1].low,
        market[1].close,
        confirmed=False,
    )

    report = Backtester(
        BuyThenHoldStrategy(),
        BacktestConfig(
            starting_equity_usdt=Decimal("1000"),
            cost_model=CostModel(taker_fee_bps=0, spread_bps=0, slippage_bps=0),
        ),
    ).run(market)

    assert report.period_end == market[0].ts
    assert report.metrics.trade_count == 0
    assert report.rejected_order_count == 1


def test_backtest_stop_loss_uses_intrabar_low_not_close() -> None:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    market = [
        MarketCandle(start, Decimal("100"), Decimal("100"), Decimal("100"), Decimal("100")),
        MarketCandle(
            start + timedelta(minutes=1),
            Decimal("100"),
            Decimal("110"),
            Decimal("80"),
            Decimal("105"),
        ),
    ]

    report = Backtester(
        BuyFirstClosedCandleStrategy(),
        BacktestConfig(
            starting_equity_usdt=Decimal("1000"),
            cost_model=CostModel(taker_fee_bps=0, spread_bps=0, slippage_bps=0),
            stop_loss_bps=Decimal("1000"),
            take_profit_bps=Decimal("500"),
        ),
    ).run(market)

    assert report.metrics.trade_count == 2
    assert report.metrics.ending_equity_usdt == Decimal("990")
    assert report.metrics.closed_trade_count == 1


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
