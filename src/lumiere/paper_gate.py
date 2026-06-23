from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from lumiere.backtest import BacktestReport
from lumiere.ledger import PnlMetrics


@dataclass(frozen=True, slots=True)
class PerformanceGateConfig:
    min_trades: int = 20
    min_net_pnl_usdt: Decimal = Decimal("0")
    max_drawdown_usdt: Decimal | None = None
    min_profit_factor: Decimal | None = Decimal("1")

    def __post_init__(self) -> None:
        if self.min_trades < 0:
            raise ValueError("min_trades cannot be negative")
        if self.max_drawdown_usdt is not None and self.max_drawdown_usdt <= 0:
            raise ValueError("max_drawdown_usdt must be positive when configured")
        if self.min_profit_factor is not None and self.min_profit_factor < 0:
            raise ValueError("min_profit_factor cannot be negative")


@dataclass(frozen=True, slots=True)
class PerformanceGateDecision:
    allowed: bool
    reason: str


def assess_metrics(
    metrics: PnlMetrics,
    config: PerformanceGateConfig | None = None,
) -> PerformanceGateDecision:
    config = config or PerformanceGateConfig()
    if metrics.trade_count < config.min_trades:
        return PerformanceGateDecision(False, "not_enough_trades")
    if metrics.net_pnl_usdt <= config.min_net_pnl_usdt:
        return PerformanceGateDecision(False, "net_pnl_not_positive_after_costs")
    if (
        config.max_drawdown_usdt is not None
        and metrics.max_drawdown_usdt > config.max_drawdown_usdt
    ):
        return PerformanceGateDecision(False, "max_drawdown_exceeded")
    if config.min_profit_factor is not None:
        if metrics.profit_factor is None:
            return PerformanceGateDecision(False, "profit_factor_unavailable")
        if metrics.profit_factor < config.min_profit_factor:
            return PerformanceGateDecision(False, "profit_factor_too_low")
    return PerformanceGateDecision(True, "performance_gate_passed")


def assess_report(
    report: BacktestReport,
    config: PerformanceGateConfig | None = None,
) -> PerformanceGateDecision:
    return assess_metrics(report.metrics, config)
