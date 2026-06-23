from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

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
