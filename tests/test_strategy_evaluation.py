from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

from lumiere.backtest import CostModel
from lumiere.models import MarketCandle
from lumiere.paper_gate import PerformanceGateConfig
from lumiere.strategy_evaluation import (
    EvaluationConfig,
    MovingAverageCandidate,
    evaluate_parameter_grid,
    train_test_split,
    train_validation_test_split,
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

    assert [evaluation.accepted for evaluation in evaluations] == [True, False]
    assert evaluations[0].candidate == MovingAverageCandidate(1, 2, Decimal("10"))
    assert (
        evaluations[0].test_report.metrics.net_pnl_usdt
        > evaluations[0].test_report.buy_and_hold_pnl_usdt
    )


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
    assert evaluations[0].rejection_reason == "train_test_pnl_divergence"
