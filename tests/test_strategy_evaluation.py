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
