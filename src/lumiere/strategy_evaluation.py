from __future__ import annotations

from dataclasses import dataclass, replace
from decimal import Decimal
from math import ceil
from typing import Any

from lumiere.backtest import BacktestConfig, Backtester, BacktestReport, CostModel
from lumiere.models import AccountSnapshot, DecisionAction, MarketCandle
from lumiere.paper_gate import (
    PerformanceGateConfig,
    PerformanceGateDecision,
    assess_report,
)
from lumiere.risk import RiskConfig
from lumiere.strategy import MovingAverageCrossoverConfig, MovingAverageCrossoverStrategy
from lumiere.strategy_factory import build_strategy


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
class StrategyCandidate:
    """Optimizer candidate spanning strategy parameters and execution exits."""

    strategy_name: str
    trade_size_base: Decimal
    fast_window: int | None = None
    slow_window: int | None = None
    rsi_period: int | None = None
    oversold_rsi: Decimal | None = None
    overbought_rsi: Decimal | None = None
    breakout_lookback: int | None = None
    breakout_atr_period: int | None = None
    breakout_atr_multiplier: Decimal | None = None
    breakout_min_atr_pct: Decimal | None = None
    stop_loss_bps: Decimal | None = None
    take_profit_bps: Decimal | None = None
    trailing_stop_bps: Decimal | None = None
    max_bars_in_trade: int | None = None

    def __post_init__(self) -> None:
        if self.trade_size_base <= 0:
            raise ValueError("trade_size_base must be positive")
        if self.strategy_name == "moving_average_crossover":
            if self.fast_window is None or self.slow_window is None:
                raise ValueError("MA candidates require fast_window and slow_window")
            if self.fast_window <= 0:
                raise ValueError("fast_window must be positive")
            if self.slow_window <= self.fast_window:
                raise ValueError("slow_window must be greater than fast_window")
        if self.strategy_name == "rsi_mean_reversion":
            if self.rsi_period is None or self.rsi_period <= 1:
                raise ValueError("rsi_period must be greater than 1")
            if self.oversold_rsi is None or self.overbought_rsi is None:
                raise ValueError("RSI candidates require thresholds")
            if self.oversold_rsi <= 0 or self.overbought_rsi >= 100:
                raise ValueError("RSI thresholds must be inside 0..100")
            if self.oversold_rsi >= self.overbought_rsi:
                raise ValueError("oversold_rsi must be below overbought_rsi")
        if self.strategy_name == "volatility_breakout":
            if self.breakout_lookback is None or self.breakout_lookback <= 1:
                raise ValueError("breakout_lookback must be greater than 1")
            if self.breakout_atr_period is None or self.breakout_atr_period <= 0:
                raise ValueError("breakout_atr_period must be positive")
            if self.breakout_atr_multiplier is None or self.breakout_atr_multiplier < 0:
                raise ValueError("breakout_atr_multiplier cannot be negative")
            if self.breakout_min_atr_pct is None or self.breakout_min_atr_pct < 0:
                raise ValueError("breakout_min_atr_pct cannot be negative")
        for value in (self.stop_loss_bps, self.take_profit_bps, self.trailing_stop_bps):
            if value is not None and value <= 0:
                raise ValueError("exit bps values must be positive when configured")
        if self.max_bars_in_trade is not None and self.max_bars_in_trade <= 0:
            raise ValueError("max_bars_in_trade must be positive when configured")


OptimizerCandidate = MovingAverageCandidate | StrategyCandidate


@dataclass(frozen=True, slots=True)
class ExpectancyCalibration:
    signal_count: int
    average_forward_return_bps: Decimal | None
    average_forward_return_after_cost_bps: Decimal | None
    horizon_bars: int
    cost_bps: Decimal

    @property
    def calibrated(self) -> bool:
        return self.signal_count > 0 and self.average_forward_return_after_cost_bps is not None

    def to_dict(self) -> dict[str, str | int | bool | None]:
        return {
            "source": "historical_forward_return_after_costs",
            "calibrated": self.calibrated,
            "signal_count": self.signal_count,
            "horizon_bars": self.horizon_bars,
            "cost_bps": str(self.cost_bps),
            "average_forward_return_bps": None
            if self.average_forward_return_bps is None
            else str(self.average_forward_return_bps),
            "average_forward_return_after_cost_bps": None
            if self.average_forward_return_after_cost_bps is None
            else str(self.average_forward_return_after_cost_bps),
        }


@dataclass(frozen=True, slots=True)
class EvaluationConfig:
    train_fraction: Decimal = Decimal("0.6")
    starting_equity_usdt: Decimal = Decimal("1000")
    cost_model: CostModel = CostModel()
    stop_loss_bps: Decimal | None = None
    take_profit_bps: Decimal | None = None
    trailing_stop_bps: Decimal | None = None
    max_bars_in_trade: int | None = None
    performance_gate: PerformanceGateConfig = PerformanceGateConfig(min_trades=1)
    max_train_test_pnl_ratio: Decimal = Decimal("4")
    require_baseline_outperformance: bool = True
    min_walk_forward_windows: int = 0
    walk_forward_train_size: int = 0
    walk_forward_test_size: int = 0
    min_walk_forward_pass_rate: Decimal = Decimal("0.5")
    min_stable_neighbors: int = 0
    parameter_stability_radius: int = 1
    min_calibration_signals: int = 1
    min_expected_edge_bps: Decimal = Decimal("0")
    expectancy_horizon_bars: int = 1
    risk_config: RiskConfig | None = None

    def __post_init__(self) -> None:
        if self.train_fraction <= 0 or self.train_fraction >= 1:
            raise ValueError("train_fraction must be between 0 and 1")
        if self.max_train_test_pnl_ratio <= 0:
            raise ValueError("max_train_test_pnl_ratio must be positive")
        if self.min_walk_forward_windows < 0:
            raise ValueError("min_walk_forward_windows cannot be negative")
        if self.walk_forward_train_size < 0 or self.walk_forward_test_size < 0:
            raise ValueError("walk-forward sizes cannot be negative")
        if self.min_walk_forward_pass_rate < 0 or self.min_walk_forward_pass_rate > 1:
            raise ValueError("min_walk_forward_pass_rate must be between 0 and 1")
        if self.min_stable_neighbors < 0:
            raise ValueError("min_stable_neighbors cannot be negative")
        if self.parameter_stability_radius < 0:
            raise ValueError("parameter_stability_radius cannot be negative")
        if self.min_calibration_signals < 0:
            raise ValueError("min_calibration_signals cannot be negative")
        if self.expectancy_horizon_bars <= 0:
            raise ValueError("expectancy_horizon_bars must be positive")


@dataclass(frozen=True, slots=True)
class CandidateEvaluation:
    candidate: OptimizerCandidate
    train_report: BacktestReport
    test_report: BacktestReport
    gate: PerformanceGateDecision
    expectancy: ExpectancyCalibration
    accepted: bool
    rejection_reason: str | None
    walk_forward_gates: tuple[PerformanceGateDecision, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate": candidate_to_dict(self.candidate),
            "accepted": self.accepted,
            "rejection_reason": self.rejection_reason,
            "gate": {"allowed": self.gate.allowed, "reason": self.gate.reason},
            "expectancy_calibration": self.expectancy.to_dict(),
            "walk_forward_gates": [
                {"allowed": gate.allowed, "reason": gate.reason}
                for gate in self.walk_forward_gates
            ],
            "rank_metrics": rank_metrics(self.test_report),
            "train_report": SplitBacktestReport(
                split_name="train",
                role="in_sample",
                report=self.train_report,
            ).to_dict(),
            "test_report": SplitBacktestReport(
                split_name="test",
                role="out_of_sample",
                report=self.test_report,
            ).to_dict(),
        }


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
    candidate: OptimizerCandidate,
    config: EvaluationConfig | None = None,
) -> CandidateEvaluation:
    config = config or EvaluationConfig()
    train, test = train_test_split(candles, train_fraction=config.train_fraction)
    train_report = _run_candidate(inst_id, train, candidate, config)
    test_report = _run_candidate(inst_id, test, candidate, config)
    gate = assess_report(test_report, config.performance_gate)
    expectancy = calibrate_expectancy(inst_id, train, candidate, config)
    walk_forward_gates = _walk_forward_gate_decisions(inst_id, candles, candidate, config)
    rejection_reason = _overfit_rejection_reason(
        train_report,
        test_report,
        gate,
        config,
        expectancy=expectancy,
        walk_forward_gates=walk_forward_gates,
    )
    return CandidateEvaluation(
        candidate=candidate,
        train_report=train_report,
        test_report=test_report,
        gate=gate,
        expectancy=expectancy,
        accepted=rejection_reason is None,
        rejection_reason=rejection_reason,
        walk_forward_gates=walk_forward_gates,
    )


def evaluate_parameter_grid(
    inst_id: str,
    candles: list[MarketCandle] | tuple[MarketCandle, ...],
    candidates: list[OptimizerCandidate] | tuple[OptimizerCandidate, ...],
    config: EvaluationConfig | None = None,
) -> tuple[CandidateEvaluation, ...]:
    if not candidates:
        raise ValueError("at least one candidate is required")
    config = config or EvaluationConfig()
    evaluations = tuple(
        evaluate_candidate(inst_id, candles, candidate, config) for candidate in candidates
    )
    evaluations = _apply_parameter_stability(evaluations, config)
    return tuple(sorted(evaluations, key=evaluation_sort_key, reverse=True))


def calibrate_expectancy(
    inst_id: str,
    candles: list[MarketCandle] | tuple[MarketCandle, ...],
    candidate: OptimizerCandidate,
    config: EvaluationConfig,
) -> ExpectancyCalibration:
    ordered = tuple(sorted(candles, key=lambda candle: candle.ts))
    horizon = config.expectancy_horizon_bars
    if len(ordered) <= horizon:
        return ExpectancyCalibration(0, None, None, horizon, _round_trip_cost_bps(config))
    strategy = _build_candidate_strategy(inst_id, candidate)
    cost_bps = _round_trip_cost_bps(config)
    forward_returns: list[Decimal] = []
    account = AccountSnapshot(
        equity_usdt=config.starting_equity_usdt,
        available_usdt=config.starting_equity_usdt,
        positions=(),
        spread_bps=config.cost_model.spread_bps,
        estimated_slippage_bps=config.cost_model.slippage_bps + config.cost_model.market_impact_bps,
        estimated_total_cost_bps=config.cost_model.order_cost_bps + config.cost_model.taker_fee_bps,
        performance_gate_passed=True,
    )
    for index in range(0, len(ordered) - horizon):
        decision = strategy.decide(list(ordered[: index + 1]), account)
        if decision.action is not DecisionAction.BUY:
            continue
        entry = ordered[index].close
        exit_price = ordered[index + horizon].open
        if entry <= 0:
            continue
        forward_returns.append((exit_price - entry) / entry * Decimal("10000"))
    if not forward_returns:
        return ExpectancyCalibration(0, None, None, horizon, cost_bps)
    average = sum(forward_returns, Decimal("0")) / Decimal(len(forward_returns))
    return ExpectancyCalibration(
        signal_count=len(forward_returns),
        average_forward_return_bps=average,
        average_forward_return_after_cost_bps=average - cost_bps,
        horizon_bars=horizon,
        cost_bps=cost_bps,
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
    candidate: OptimizerCandidate,
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
    candidate: OptimizerCandidate,
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


def rank_metrics(report: BacktestReport) -> dict[str, str | int | float | None]:
    metrics = report.metrics
    return {
        "net_pnl_usdt": str(metrics.net_pnl_usdt),
        "max_drawdown_usdt": str(metrics.max_drawdown_usdt),
        "max_drawdown_duration_bars": metrics.max_drawdown_duration_bars,
        "largest_loss_usdt": str(metrics.largest_loss_usdt),
        "losing_streak": metrics.losing_streak,
        "turnover_trade_count": metrics.trade_count,
        "profit_factor": None if metrics.profit_factor is None else str(metrics.profit_factor),
        "sharpe": metrics.sharpe,
        "sortino": metrics.sortino,
        "trade_count": metrics.trade_count,
        "closed_trade_count": metrics.closed_trade_count,
        "win_rate": str(metrics.win_rate),
        "buy_and_hold_pnl_usdt": str(report.buy_and_hold_pnl_usdt),
        "no_trade_pnl_usdt": str(report.no_trade_pnl_usdt),
        "risk_rejection_count": report.risk_rejection_count,
        "blocked_signal_opportunity_cost_usdt": str(
            report.blocked_signal_opportunity_cost_usdt
        ),
        **baseline_comparison(report),
    }


def evaluation_sort_key(evaluation: CandidateEvaluation) -> tuple:
    metrics = evaluation.test_report.metrics
    profit_factor = metrics.profit_factor
    if profit_factor is None:
        profit_factor_score = Decimal("-1")
    elif profit_factor.is_infinite():
        profit_factor_score = Decimal("999999999")
    else:
        profit_factor_score = profit_factor
    expected_edge = (
        evaluation.expectancy.average_forward_return_after_cost_bps or Decimal("-999999")
    )
    return (
        evaluation.accepted,
        expected_edge,
        metrics.net_pnl_usdt,
        -metrics.max_drawdown_usdt,
        -metrics.max_drawdown_duration_bars,
        -abs(metrics.largest_loss_usdt),
        -metrics.trade_count,
        profit_factor_score,
        metrics.sharpe or float("-inf"),
        metrics.sortino or float("-inf"),
        metrics.win_rate,
    )


def candidate_to_dict(candidate: OptimizerCandidate) -> dict[str, Any]:
    if isinstance(candidate, MovingAverageCandidate):
        return {
            "strategy": "moving_average_crossover",
            "fast_window": candidate.fast_window,
            "slow_window": candidate.slow_window,
            "trade_size_base": str(candidate.trade_size_base),
        }
    payload: dict[str, Any] = {
        "strategy": candidate.strategy_name,
        "trade_size_base": str(candidate.trade_size_base),
    }
    optional_values = {
        "fast_window": candidate.fast_window,
        "slow_window": candidate.slow_window,
        "rsi_period": candidate.rsi_period,
        "oversold_rsi": candidate.oversold_rsi,
        "overbought_rsi": candidate.overbought_rsi,
        "breakout_lookback": candidate.breakout_lookback,
        "breakout_atr_period": candidate.breakout_atr_period,
        "breakout_atr_multiplier": candidate.breakout_atr_multiplier,
        "breakout_min_atr_pct": candidate.breakout_min_atr_pct,
        "stop_loss_bps": candidate.stop_loss_bps,
        "take_profit_bps": candidate.take_profit_bps,
        "trailing_stop_bps": candidate.trailing_stop_bps,
        "max_bars_in_trade": candidate.max_bars_in_trade,
    }
    for key, value in optional_values.items():
        if value is not None:
            payload[key] = str(value) if isinstance(value, Decimal) else value
    return payload


def _run_candidate(
    inst_id: str,
    candles: tuple[MarketCandle, ...],
    candidate: OptimizerCandidate,
    config: EvaluationConfig,
) -> BacktestReport:
    strategy = _build_candidate_strategy(inst_id, candidate)
    return Backtester(
        strategy,
        BacktestConfig(
            starting_equity_usdt=config.starting_equity_usdt,
            cost_model=config.cost_model,
            stop_loss_bps=_candidate_value(candidate, "stop_loss_bps", config.stop_loss_bps),
            take_profit_bps=_candidate_value(candidate, "take_profit_bps", config.take_profit_bps),
            trailing_stop_bps=_candidate_value(
                candidate,
                "trailing_stop_bps",
                config.trailing_stop_bps,
            ),
            max_bars_in_trade=_candidate_value(
                candidate,
                "max_bars_in_trade",
                config.max_bars_in_trade,
            ),
            risk_config=config.risk_config,
        ),
    ).run(candles)


def _build_candidate_strategy(inst_id: str, candidate: OptimizerCandidate):
    if isinstance(candidate, MovingAverageCandidate):
        return MovingAverageCrossoverStrategy(
            MovingAverageCrossoverConfig(
                inst_id=inst_id,
                fast_window=candidate.fast_window,
                slow_window=candidate.slow_window,
                trade_size_btc=candidate.trade_size_base,
            )
        )
    return build_strategy(
        candidate.strategy_name,
        inst_id=inst_id,
        trade_size_btc=candidate.trade_size_base,
        dust_threshold_btc=Decimal("0.00000001"),
        fast_window=candidate.fast_window or 5,
        slow_window=candidate.slow_window or 20,
        rsi_period=candidate.rsi_period or 14,
        oversold_rsi=candidate.oversold_rsi or Decimal("30"),
        overbought_rsi=candidate.overbought_rsi or Decimal("70"),
        breakout_lookback=candidate.breakout_lookback or 20,
        breakout_atr_period=candidate.breakout_atr_period or 14,
        breakout_atr_multiplier=candidate.breakout_atr_multiplier or Decimal("0.5"),
        breakout_min_atr_pct=candidate.breakout_min_atr_pct or Decimal("0.001"),
    )


def _candidate_value(candidate: OptimizerCandidate, field: str, default):
    if not isinstance(candidate, StrategyCandidate):
        return default
    value = getattr(candidate, field)
    return default if value is None else value


def _overfit_rejection_reason(
    train_report: BacktestReport,
    test_report: BacktestReport,
    gate: PerformanceGateDecision,
    config: EvaluationConfig,
    *,
    expectancy: ExpectancyCalibration,
    walk_forward_gates: tuple[PerformanceGateDecision, ...],
) -> str | None:
    if not gate.allowed:
        return gate.reason
    train_pnl = train_report.metrics.net_pnl_usdt
    test_pnl = test_report.metrics.net_pnl_usdt
    if test_pnl <= 0:
        return "test_net_pnl_not_positive"
    if config.require_baseline_outperformance:
        if test_pnl <= test_report.no_trade_pnl_usdt:
            return "no_trade_baseline_not_beaten"
        if test_pnl <= test_report.buy_and_hold_pnl_usdt:
            return "buy_and_hold_baseline_not_beaten"
    if train_pnl > 0 and train_pnl > test_pnl * config.max_train_test_pnl_ratio:
        return "train_test_pnl_divergence"
    if config.min_walk_forward_windows > 0:
        if len(walk_forward_gates) < config.min_walk_forward_windows:
            return "not_enough_walk_forward_windows"
        passing = sum(1 for decision in walk_forward_gates if decision.allowed)
        pass_rate = Decimal(passing) / Decimal(len(walk_forward_gates))
        if pass_rate < config.min_walk_forward_pass_rate:
            return "walk_forward_consistency_failed"
    if expectancy.signal_count < config.min_calibration_signals:
        return "expectancy_not_calibrated"
    if (
        expectancy.average_forward_return_after_cost_bps is None
        or expectancy.average_forward_return_after_cost_bps <= config.min_expected_edge_bps
    ):
        return "expected_edge_not_positive_after_costs"
    return None


def _walk_forward_gate_decisions(
    inst_id: str,
    candles: list[MarketCandle] | tuple[MarketCandle, ...],
    candidate: OptimizerCandidate,
    config: EvaluationConfig,
) -> tuple[PerformanceGateDecision, ...]:
    if config.min_walk_forward_windows <= 0:
        return ()
    train_size = config.walk_forward_train_size
    test_size = config.walk_forward_test_size
    if train_size <= 0 or test_size <= 0:
        total = len(candles)
        train_size = max(1, int(total * 0.5))
        test_size = max(1, int(total * 0.2))
    windows = walk_forward_splits(candles, train_size=train_size, test_size=test_size)
    decisions: list[PerformanceGateDecision] = []
    for window in windows:
        report = _run_candidate(inst_id, window.test, candidate, config)
        decisions.append(assess_report(report, config.performance_gate))
    return tuple(decisions)


def _apply_parameter_stability(
    evaluations: tuple[CandidateEvaluation, ...],
    config: EvaluationConfig,
) -> tuple[CandidateEvaluation, ...]:
    if config.min_stable_neighbors <= 0:
        return evaluations
    stable_evaluations: list[CandidateEvaluation] = []
    for evaluation in evaluations:
        if not evaluation.accepted:
            stable_evaluations.append(evaluation)
            continue
        stable_neighbors = sum(
            1
            for neighbor in evaluations
            if neighbor is not evaluation
            and neighbor.accepted
            and _are_nearby_parameters(
                evaluation.candidate,
                neighbor.candidate,
                radius=config.parameter_stability_radius,
            )
        )
        if stable_neighbors < config.min_stable_neighbors:
            stable_evaluations.append(
                replace(
                    evaluation,
                    accepted=False,
                    rejection_reason="parameter_stability_failed",
                )
            )
        else:
            stable_evaluations.append(evaluation)
    return tuple(stable_evaluations)


def _are_nearby_parameters(
    left: OptimizerCandidate,
    right: OptimizerCandidate,
    *,
    radius: int,
) -> bool:
    left_values = _stability_parameters(left)
    right_values = _stability_parameters(right)
    if left_values.get("strategy") != right_values.get("strategy"):
        return False
    common = set(left_values) & set(right_values) - {"strategy"}
    if not common:
        return False
    return all(
        abs(Decimal(str(left_values[key])) - Decimal(str(right_values[key]))) <= Decimal(radius)
        for key in common
    )


def _stability_parameters(candidate: OptimizerCandidate) -> dict[str, str | int | Decimal]:
    payload = candidate_to_dict(candidate)
    values: dict[str, str | int | Decimal] = {"strategy": payload["strategy"]}
    for key, value in payload.items():
        if key in {"strategy", "trade_size_base"}:
            continue
        values[key] = value
    return values


def _round_trip_cost_bps(config: EvaluationConfig) -> Decimal:
    one_way = config.cost_model.order_cost_bps + config.cost_model.taker_fee_bps
    return one_way * Decimal("2")


def _bounded_split_size(total: int, fraction: Decimal, *, minimum: int) -> int:
    return min(max(int(ceil(total * float(fraction))), minimum), total - 1)
