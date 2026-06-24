from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

from lumiere.backtest import CostModel
from lumiere.models import MarketCandle
from lumiere.paper_gate import PerformanceGateConfig
from lumiere.strategy_evaluation import (
    EvaluationConfig,
    MovingAverageCandidate,
    StrategyCandidate,
    evaluate_parameter_grid,
    split_backtest_reports,
    train_test_split,
    train_validation_test_split,
    walk_forward_backtest_reports,
    walk_forward_splits,
)


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


def test_train_test_split_preserves_chronological_order() -> None:
    train, test = train_test_split(candles(["1", "2", "3", "4"]), train_fraction=Decimal("0.5"))

    assert [candle.close for candle in train] == [1, 2]
    assert [candle.close for candle in test] == [3, 4]


def test_train_validation_test_split_boundaries_are_chronological_and_disjoint() -> None:
    train, validation, test = train_validation_test_split(
        candles(["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]),
        train_fraction=Decimal("0.5"),
        validation_fraction=Decimal("0.2"),
    )

    assert [candle.close for candle in train.candles] == [1, 2, 3, 4, 5]
    assert [candle.close for candle in validation.candles] == [6, 7]
    assert [candle.close for candle in test.candles] == [8, 9, 10]
    assert train.candles[-1].ts < validation.candles[0].ts < test.candles[0].ts


def test_split_and_walk_forward_reports_use_no_lookahead_timing() -> None:
    market = candles(["100", "101", "102", "103", "104", "105", "106", "107"])
    candidate = MovingAverageCandidate(1, 2, Decimal("1"))
    config = EvaluationConfig(
        cost_model=CostModel(
            taker_fee_bps=Decimal("0"),
            spread_bps=Decimal("0"),
            slippage_bps=Decimal("0"),
        )
    )

    split_reports = split_backtest_reports(
        "BTC-USDT",
        market,
        candidate,
        config,
        train_fraction=Decimal("0.5"),
        validation_fraction=Decimal("0.25"),
    )
    walk_reports = walk_forward_backtest_reports(
        "BTC-USDT",
        market,
        candidate,
        config,
        train_size=4,
        test_size=2,
        step_size=2,
    )

    assert all(report.report.execution_timing == "next_open" for report in split_reports)
    assert all(
        report.report.assumptions["signal_candle"] == "closed_confirmed"
        for report in split_reports
    )
    assert split_reports[0].report.period_end == market[3].ts
    assert split_reports[1].report.period_start == market[4].ts
    assert walk_reports[0]["train"]["period_end"] == market[3].ts.isoformat()
    assert walk_reports[0]["test"]["period_start"] == market[4].ts.isoformat()
    assert walk_reports[0]["test"]["execution_timing"] == "next_open"


def test_walk_forward_splits_roll_without_lookahead() -> None:
    windows = walk_forward_splits(
        candles(["1", "2", "3", "4", "5", "6"]),
        train_size=3,
        test_size=2,
        step_size=1,
    )

    assert len(windows) == 2
    assert [candle.close for candle in windows[0].train] == [1, 2, 3]
    assert [candle.close for candle in windows[0].test] == [4, 5]
    assert [candle.close for candle in windows[1].train] == [2, 3, 4]
    assert [candle.close for candle in windows[1].test] == [5, 6]
    assert all(window.train[-1].ts < window.test[0].ts for window in windows)


def test_parameter_grid_marks_candidates_not_passing_test_gate_as_rejected() -> None:
    evaluations = evaluate_parameter_grid(
        "BTC-USDT",
        candles(["100", "101", "110", "100", "90", "91", "92", "80"]),
        (MovingAverageCandidate(2, 3, Decimal("1")),),
        EvaluationConfig(
            cost_model=CostModel(taker_fee_bps=Decimal("10")),
            performance_gate=PerformanceGateConfig(min_trades=10),
        ),
    )

    assert evaluations[0].accepted is False
    assert evaluations[0].rejection_reason == "not_enough_trades"


def test_parameter_grid_sorts_accepted_candidates_before_rejected_candidates() -> None:
    evaluations = evaluate_parameter_grid(
        "BTC-USDT",
        candles(["100", "90", "91", "150", "100", "90", "91", "150"]),
        (
            MovingAverageCandidate(2, 3, Decimal("10")),
            MovingAverageCandidate(1, 2, Decimal("10")),
        ),
        EvaluationConfig(
            train_fraction=Decimal("0.5"),
            cost_model=CostModel(
                taker_fee_bps=Decimal("0"),
                spread_bps=Decimal("0"),
                slippage_bps=Decimal("0"),
            ),
            performance_gate=PerformanceGateConfig(min_trades=1, min_profit_factor=None),
        ),
    )

    assert [evaluation.accepted for evaluation in evaluations] == [False, False]
    assert {evaluation.rejection_reason for evaluation in evaluations} == {"not_enough_trades"}
    assert evaluations[0].test_report.execution_timing == "next_open"
    assert {evaluation.candidate for evaluation in evaluations} == {
        MovingAverageCandidate(2, 3, Decimal("10")),
        MovingAverageCandidate(1, 2, Decimal("10")),
    }


def test_parameter_grid_evaluates_non_ma_candidates_with_empirical_expectancy() -> None:
    evaluations = evaluate_parameter_grid(
        "BTC-USDT",
        candles(["100", "98", "96", "99", "101", "99", "97", "100", "102", "104"]),
        (
            StrategyCandidate(
                "rsi_mean_reversion",
                Decimal("1"),
                rsi_period=2,
                oversold_rsi=Decimal("40"),
                overbought_rsi=Decimal("70"),
                stop_loss_bps=Decimal("100"),
                max_bars_in_trade=2,
            ),
        ),
        EvaluationConfig(
            train_fraction=Decimal("0.6"),
            cost_model=CostModel(
                taker_fee_bps=Decimal("0"),
                spread_bps=Decimal("0"),
                slippage_bps=Decimal("0"),
            ),
            performance_gate=PerformanceGateConfig(
                min_trades=0,
                min_net_pnl_usdt=Decimal("-999"),
                min_profit_factor=None,
            ),
            require_baseline_outperformance=False,
        ),
    )

    payload = evaluations[0].to_dict()
    assert payload["candidate"]["strategy"] == "rsi_mean_reversion"
    assert payload["candidate"]["stop_loss_bps"] == "100"
    assert payload["expectancy_calibration"]["source"] == "historical_forward_return_after_costs"
    assert payload["expectancy_calibration"]["signal_count"] > 0


def test_overfit_gate_rejects_train_test_divergence() -> None:
    evaluations = evaluate_parameter_grid(
        "BTC-USDT",
        candles(["100", "90", "91", "200", "100", "90", "91", "100"]),
        (MovingAverageCandidate(1, 2, Decimal("10")),),
        EvaluationConfig(
            train_fraction=Decimal("0.5"),
            cost_model=CostModel(
                taker_fee_bps=Decimal("0"),
                spread_bps=Decimal("0"),
                slippage_bps=Decimal("0"),
            ),
            performance_gate=PerformanceGateConfig(min_trades=1, min_profit_factor=None),
            max_train_test_pnl_ratio=Decimal("2"),
            require_baseline_outperformance=False,
        ),
    )

    assert evaluations[0].accepted is False
    assert evaluations[0].rejection_reason == "net_pnl_not_positive_after_costs"
    assert evaluations[0].test_report.same_close_comparison is not None
