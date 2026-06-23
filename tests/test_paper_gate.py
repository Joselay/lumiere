from __future__ import annotations

from decimal import Decimal

from lumiere.ledger import PnlMetrics
from lumiere.paper_gate import PerformanceGateConfig, assess_metrics


def metrics(**overrides) -> PnlMetrics:
    values = {
        "starting_equity_usdt": Decimal("1000"),
        "ending_equity_usdt": Decimal("1010"),
        "net_pnl_usdt": Decimal("10"),
        "realized_pnl_usdt": Decimal("10"),
        "unrealized_pnl_usdt": Decimal("0"),
        "fees_usdt": Decimal("1"),
        "trade_count": 20,
        "closed_trade_count": 10,
        "win_rate": Decimal("0.6"),
        "profit_factor": Decimal("1.5"),
        "max_drawdown_usdt": Decimal("5"),
        "sharpe": None,
        "sortino": None,
        "equity_curve": (),
    }
    values.update(overrides)
    return PnlMetrics(**values)


def test_performance_gate_requires_sample_size_and_positive_net_pnl_after_costs() -> None:
    too_few = assess_metrics(metrics(trade_count=3), PerformanceGateConfig(min_trades=10))
    losing = assess_metrics(metrics(net_pnl_usdt=Decimal("-1")), PerformanceGateConfig())
    passing = assess_metrics(metrics(), PerformanceGateConfig())

    assert too_few.allowed is False
    assert too_few.reason == "not_enough_trades"
    assert losing.allowed is False
    assert losing.reason == "net_pnl_not_positive_after_costs"
    assert passing.allowed is True
