from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

from lumiere.attribution import AttributionLedger
from lumiere.models import AccountSnapshot, DecisionAction, MarketCandle, OrderRequest, OrderResult


def test_attribution_ledger_persists_across_restarts(tmp_path) -> None:
    path = tmp_path / "attribution.jsonl"
    ledger = AttributionLedger(path)
    ts = datetime(2026, 1, 1, tzinfo=UTC)

    ledger.record_account(
        AccountSnapshot(
            equity_usdt=Decimal("1000"),
            available_usdt=Decimal("900"),
            spread_bps=Decimal("2"),
        ),
        ts=ts,
    )
    ledger.record_fill(
        inst_id="BTC-USDT",
        side=DecisionAction.BUY,
        size_base=Decimal("1"),
        price=Decimal("100"),
        fee=Decimal("1"),
        ts=ts,
        decision_price=Decimal("99"),
    )

    restarted = AttributionLedger(path)

    assert len(restarted.events) == 2
    assert restarted.events[0]["type"] == "account"


def test_attribution_report_dedupes_fills_and_flags_missing_order_attribution(tmp_path) -> None:
    path = tmp_path / "attribution.jsonl"
    ledger = AttributionLedger(path)
    ts = datetime(2026, 1, 1, tzinfo=UTC)
    order = OrderRequest("BTC-USDT", DecisionAction.BUY, Decimal("1"))
    result = OrderResult("ord-1", "client-1", "BTC-USDT", DecisionAction.BUY, Decimal("1"), "OK")
    ledger.record_order(order, result, ts=ts)
    for _ in range(2):
        ledger.record_fill(
            inst_id="BTC-USDT",
            side=DecisionAction.BUY,
            size_base=Decimal("1"),
            price=Decimal("101"),
            fee=Decimal("1"),
            ts=ts + timedelta(seconds=1),
            decision_price=Decimal("100"),
            order_id="ord-1",
            trade_id="trade-1",
            client_order_id="client-1",
        )
    ledger.record_order(
        OrderRequest("BTC-USDT", DecisionAction.SELL, Decimal("1")),
        OrderResult("ord-2", "client-2", "BTC-USDT", DecisionAction.SELL, Decimal("1"), "OK"),
        ts=ts + timedelta(minutes=1),
    )

    report = ledger.report(window=timedelta(days=1), now=ts + timedelta(hours=1))

    assert report.metrics["trade_count"] == 1
    assert report.metrics["fees_usdt"] == "1"
    assert report.metrics["average_slippage_bps"] == "100.00"
    assert report.metrics["fill_completeness"] == {
        "orders": 2,
        "orders_without_final_attribution": 1,
        "filled_orders_without_fills": 0,
    }
    assert "orders_missing_final_attribution" in report.alerts


def test_attribution_report_flags_fills_without_cost_basis(tmp_path) -> None:
    path = tmp_path / "attribution.jsonl"
    ledger = AttributionLedger(path)
    ts = datetime(2026, 1, 1, tzinfo=UTC)
    ledger.record_fill(
        inst_id="BTC-USDT",
        side=DecisionAction.SELL,
        size_base=Decimal("1"),
        price=Decimal("100"),
        fee=Decimal("0"),
        ts=ts,
    )

    report = ledger.report(window=timedelta(days=1), now=ts + timedelta(hours=1))

    assert report.metrics["inventory_completeness"] == {
        "sell_fills_without_cost_basis": 1,
        "uncosted_sell_size_by_inst": {"BTC-USDT": "1"},
    }
    assert "fills_without_cost_basis" in report.alerts


def test_attribution_report_explains_pnl_and_rejections(tmp_path) -> None:
    path = tmp_path / "attribution.jsonl"
    ledger = AttributionLedger(path)
    ts = datetime(2026, 1, 1, tzinfo=UTC)
    ledger.record_candle(
        "BTC-USDT",
        MarketCandle(
            ts=ts,
            open=Decimal("100"),
            high=Decimal("100"),
            low=Decimal("100"),
            close=Decimal("100"),
        ),
    )
    ledger.record_fill(
        inst_id="BTC-USDT",
        side=DecisionAction.BUY,
        size_base=Decimal("1"),
        price=Decimal("100"),
        fee=Decimal("0"),
        ts=ts,
        decision_price=Decimal("100"),
    )
    ledger.record_fill(
        inst_id="BTC-USDT",
        side=DecisionAction.SELL,
        size_base=Decimal("1"),
        price=Decimal("90"),
        fee=Decimal("0"),
        ts=ts + timedelta(minutes=1),
        decision_price=Decimal("100"),
    )
    ledger.record_risk(
        "BTC-USDT",
        "buy",
        False,
        "expected_edge_below_cost",
        ts=ts + timedelta(minutes=2),
    )
    ledger.record_order(
        OrderRequest("BTC-USDT", DecisionAction.BUY, Decimal("1")),
        OrderResult("ord-1", "client-1", "BTC-USDT", DecisionAction.BUY, Decimal("1"), "filled"),
        ts=ts,
    )

    report = ledger.report(window=timedelta(days=1), now=ts + timedelta(hours=1))

    assert report.metrics["net_pnl_usdt"] == "-10"
    assert report.metrics["fees_usdt"] == "0"
    assert report.rejection_counts == {"expected_edge_below_cost": 1}
    assert "negative_rolling_expectancy" in report.alerts
    assert "cost_gate_rejections" in report.alerts
