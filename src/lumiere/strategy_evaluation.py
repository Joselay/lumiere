from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from math import ceil
from typing import Any

from lumiere.backtest import BacktestConfig, Backtester, BacktestReport, CostModel
from lumiere.models import MarketCandle
from lumiere.paper_gate import (
    PerformanceGateConfig,
    PerformanceGateDecision,
    assess_report,
)
from lumiere.strategy import MovingAverageCrossoverConfig, MovingAverageCrossoverStrategy


@dataclass(frozen=True, slots=True)
class MovingAverageCandidate:
    fast_window: int
    slow_window: int
    trade_size_base: Decimal

    def __post_init__(self) -> None:
        if self.fast_window <= 0:
            raise ValueError("fast_window must be positive")
        if self.slow_window <= self.fast_window:
            raise ValueError("slow_window must be greater than fast_window")
        if self.trade_size_base <= 0:
            raise ValueError("trade_size_base must be positive")


@dataclass(frozen=True, slots=True)
class EvaluationConfig:
    train_fraction: Decimal = Decimal("0.6")
    starting_equity_usdt: Decimal = Decimal("1000")
    cost_model: CostModel = CostModel()
    performance_gate: PerformanceGateConfig = PerformanceGateConfig(min_trades=1)
    max_train_test_pnl_ratio: Decimal = Decimal("4")

    def __post_init__(self) -> None:
        if self.train_fraction <= 0 or self.train_fraction >= 1:
            raise ValueError("train_fraction must be between 0 and 1")
        if self.max_train_test_pnl_ratio <= 0:
            raise ValueError("max_train_test_pnl_ratio must be positive")


@dataclass(frozen=True, slots=True)
class CandidateEvaluation:
    candidate: MovingAverageCandidate
    train_report: BacktestReport
    test_report: BacktestReport
    gate: PerformanceGateDecision
    accepted: bool
    rejection_reason: str | None


@dataclass(frozen=True, slots=True)
class SplitWindow:
    name: str
    candles: tuple[MarketCandle, ...]

    @property
    def start_index(self) -> int:
        return 0


@dataclass(frozen=True, slots=True)
class WalkForwardWindow:
    window: int
    train: tuple[MarketCandle, ...]
    test: tuple[MarketCandle, ...]


@dataclass(frozen=True, slots=True)
class SplitBacktestReport:
    split_name: str
    report: BacktestReport
    role: str

    def to_dict(self) -> dict[str, Any]:
        payload = self.report.to_dict()
        payload["split_name"] = self.split_name
        payload["role"] = self.role
        payload["baseline_comparison"] = baseline_comparison(self.report)
        return payload


def evaluate_candidate(
    inst_id: str,
    candles: list[MarketCandle] | tuple[MarketCandle, ...],
    candidate: MovingAverageCandidate,
    config: EvaluationConfig | None = None,
) -> CandidateEvaluation:
    config = config or EvaluationConfig()
    train, test = train_test_split(candles, train_fraction=config.train_fraction)
    train_report = _run_candidate(inst_id, train, candidate, config)
    test_report = _run_candidate(inst_id, test, candidate, config)
    gate = assess_report(test_report, config.performance_gate)
    rejection_reason = _overfit_rejection_reason(train_report, test_report, gate, config)
    return CandidateEvaluation(
        candidate=candidate,
        train_report=train_report,
        test_report=test_report,
        gate=gate,
        accepted=rejection_reason is None,
        rejection_reason=rejection_reason,
    )


def evaluate_parameter_grid(
    inst_id: str,
    candles: list[MarketCandle] | tuple[MarketCandle, ...],
    candidates: list[MovingAverageCandidate] | tuple[MovingAverageCandidate, ...],
    config: EvaluationConfig | None = None,
) -> tuple[CandidateEvaluation, ...]:
    if not candidates:
        raise ValueError("at least one candidate is required")
    evaluations = tuple(
        evaluate_candidate(inst_id, candles, candidate, config) for candidate in candidates
    )
    return tuple(
        sorted(
            evaluations,
            key=lambda evaluation: (
                evaluation.accepted,
                evaluation.test_report.metrics.net_pnl_usdt,
                -evaluation.test_report.metrics.max_drawdown_usdt,
            ),
            reverse=True,
        )
    )


def train_test_split(
    candles: list[MarketCandle] | tuple[MarketCandle, ...],
    *,
    train_fraction: Decimal,
) -> tuple[tuple[MarketCandle, ...], tuple[MarketCandle, ...]]:
    ordered = tuple(sorted(candles, key=lambda candle: candle.ts))
    if len(ordered) < 2:
        raise ValueError("at least two candles are required for train/test split")
    train_size = int(Decimal(len(ordered)) * train_fraction)
    train_size = min(max(train_size, 1), len(ordered) - 1)
    return ordered[:train_size], ordered[train_size:]


def train_validation_test_split(
    candles: list[MarketCandle] | tuple[MarketCandle, ...],
    *,
    train_fraction: Decimal = Decimal("0.6"),
    validation_fraction: Decimal = Decimal("0.2"),
) -> tuple[SplitWindow, SplitWindow, SplitWindow]:
    """Return chronological train/validation/test windows without overlap or lookahead."""

    ordered = tuple(sorted(candles, key=lambda candle: candle.ts))
    if len(ordered) < 3:
        raise ValueError("at least three candles are required for train/validation/test split")
    if train_fraction <= 0 or validation_fraction <= 0:
        raise ValueError("split fractions must be positive")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("train + validation fractions must leave room for test data")

    train_size = _bounded_split_size(len(ordered), train_fraction, minimum=1)
    validation_size = _bounded_split_size(len(ordered), validation_fraction, minimum=1)
    if train_size + validation_size >= len(ordered):
        validation_size = max(1, len(ordered) - train_size - 1)
    test_size = len(ordered) - train_size - validation_size
    if test_size <= 0:
        train_size = max(1, len(ordered) - 2)
        validation_size = 1

    train_end = train_size
    validation_end = train_end + validation_size
    return (
        SplitWindow("train", ordered[:train_end]),
        SplitWindow("validation", ordered[train_end:validation_end]),
        SplitWindow("test", ordered[validation_end:]),
    )


def walk_forward_splits(
    candles: list[MarketCandle] | tuple[MarketCandle, ...],
    *,
    train_size: int,
    test_size: int,
    step_size: int | None = None,
) -> tuple[WalkForwardWindow, ...]:
    ordered = tuple(sorted(candles, key=lambda candle: candle.ts))
    if train_size <= 0 or test_size <= 0:
        raise ValueError("train_size and test_size must be positive")
    step = step_size or test_size
    if step <= 0:
        raise ValueError("step_size must be positive")
    windows: list[WalkForwardWindow] = []
    start = 0
    window_number = 1
    while start + train_size + test_size <= len(ordered):
        train = ordered[start : start + train_size]
        test = ordered[start + train_size : start + train_size + test_size]
        windows.append(WalkForwardWindow(window_number, train, test))
        window_number += 1
        start += step
    return tuple(windows)


def split_backtest_reports(
    inst_id: str,
    candles: list[MarketCandle] | tuple[MarketCandle, ...],
    candidate: MovingAverageCandidate,
    config: EvaluationConfig,
    *,
    train_fraction: Decimal = Decimal("0.6"),
    validation_fraction: Decimal = Decimal("0.2"),
) -> tuple[SplitBacktestReport, ...]:
    reports: list[SplitBacktestReport] = []
    for split in train_validation_test_split(
        candles,
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
    ):
        reports.append(
            SplitBacktestReport(
                split_name=split.name,
                role="in_sample" if split.name == "train" else "out_of_sample",
                report=_run_candidate(inst_id, split.candles, candidate, config),
            )
        )
    return tuple(reports)


def walk_forward_backtest_reports(
    inst_id: str,
    candles: list[MarketCandle] | tuple[MarketCandle, ...],
    candidate: MovingAverageCandidate,
    config: EvaluationConfig,
    *,
    train_size: int,
    test_size: int,
    step_size: int | None = None,
) -> tuple[dict[str, Any], ...]:
    windows = walk_forward_splits(
        candles,
        train_size=train_size,
        test_size=test_size,
        step_size=step_size,
    )
    payloads: list[dict[str, Any]] = []
    for window in windows:
        train_report = _run_candidate(inst_id, window.train, candidate, config)
        test_report = _run_candidate(inst_id, window.test, candidate, config)
        payloads.append(
            {
                "window": window.window,
                "train": SplitBacktestReport(
                    split_name=f"walk_forward_{window.window}_train",
                    role="in_sample",
                    report=train_report,
                ).to_dict(),
                "test": SplitBacktestReport(
                    split_name=f"walk_forward_{window.window}_test",
                    role="out_of_sample",
                    report=test_report,
                ).to_dict(),
            }
        )
    return tuple(payloads)


def baseline_comparison(report: BacktestReport) -> dict[str, str]:
    net = report.metrics.net_pnl_usdt
    return {
        "net_pnl_minus_buy_and_hold_usdt": str(net - report.buy_and_hold_pnl_usdt),
        "net_pnl_minus_no_trade_usdt": str(net - report.no_trade_pnl_usdt),
    }


def _run_candidate(
    inst_id: str,
    candles: tuple[MarketCandle, ...],
    candidate: MovingAverageCandidate,
    config: EvaluationConfig,
) -> BacktestReport:
    strategy = MovingAverageCrossoverStrategy(
        MovingAverageCrossoverConfig(
            inst_id=inst_id,
            fast_window=candidate.fast_window,
            slow_window=candidate.slow_window,
            trade_size_btc=candidate.trade_size_base,
        )
    )
    return Backtester(
        strategy,
        BacktestConfig(
            starting_equity_usdt=config.starting_equity_usdt,
            cost_model=config.cost_model,
        ),
    ).run(candles)


def _overfit_rejection_reason(
    train_report: BacktestReport,
    test_report: BacktestReport,
    gate: PerformanceGateDecision,
    config: EvaluationConfig,
) -> str | None:
    if not gate.allowed:
        return gate.reason
    train_pnl = train_report.metrics.net_pnl_usdt
    test_pnl = test_report.metrics.net_pnl_usdt
    if test_pnl <= 0:
        return "test_net_pnl_not_positive"
    if train_pnl > 0 and train_pnl > test_pnl * config.max_train_test_pnl_ratio:
        return "train_test_pnl_divergence"
    return None


def _bounded_split_size(total: int, fraction: Decimal, *, minimum: int) -> int:
    return min(max(int(ceil(total * float(fraction))), minimum), total - 1)
