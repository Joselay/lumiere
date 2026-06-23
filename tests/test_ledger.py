from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

from lumiere.ledger import TradeFill, build_pnl_metrics, realized_pnl_for_period
from lumiere.models import DecisionAction


def fill(
    side: DecisionAction,
    size: str,
    price: str,
    *,
    minutes: int,
    fee: str = "0",
) -> TradeFill:
    return TradeFill(
        inst_id="BTC-USDT",
        side=side,
        size_base=Decimal(size),
        price=Decimal(price),
        fee=Decimal(fee),
        fee_ccy="USDT",
        ts=datetime(2026, 1, 1, tzinfo=UTC) + timedelta(minutes=minutes),
    )


def test_ledger_reports_realized_pnl_after_fees() -> None:
    metrics = build_pnl_metrics(
        [
            fill(DecisionAction.BUY, "1", "100", minutes=0, fee="1"),
            fill(DecisionAction.SELL, "1", "120", minutes=1, fee="1"),
        ],
        starting_equity_usdt=Decimal("1000"),
    )

    assert metrics.realized_pnl_usdt == Decimal("18")
    assert metrics.net_pnl_usdt == Decimal("18")
    assert metrics.fees_usdt == Decimal("2")
    assert metrics.win_rate == Decimal("1")
    assert metrics.profit_factor == Decimal("Infinity")


def test_realized_pnl_for_period_uses_older_fills_as_cost_basis() -> None:
    period_start = datetime(2026, 1, 2, tzinfo=UTC)
    fills = [
        fill(DecisionAction.BUY, "1", "100", minutes=0, fee="1"),
        TradeFill(
            inst_id="BTC-USDT",
            side=DecisionAction.SELL,
            size_base=Decimal("1"),
            price=Decimal("120"),
            fee=Decimal("1"),
            fee_ccy="USDT",
            ts=period_start,
        ),
    ]

    assert realized_pnl_for_period(fills, period_start=period_start) == Decimal("18")
