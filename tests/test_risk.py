from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from lumiere.models import AccountSnapshot, DecisionAction, OrderRequest, Position, StrategyDecision
from lumiere.risk import RiskConfig, RiskManager


def account(position: str = "0", pnl: str = "0") -> AccountSnapshot:
    btc_position = None
    if Decimal(position) != 0:
        btc_position = Position(inst_id="BTC-USDT", size_btc=Decimal(position))
    return AccountSnapshot(
        equity_usdt=Decimal("1000"),
        available_usdt=Decimal("1000"),
        btc_position=btc_position,
        daily_realized_pnl_usdt=Decimal(pnl),
    )


def buy(size: str = "0.001", inst_id: str = "BTC-USDT") -> StrategyDecision:
    return StrategyDecision(
        action=DecisionAction.BUY,
        inst_id=inst_id,
        size_btc=Decimal(size),
        reason="test",
    )


def test_risk_config_refuses_live_okx_flag() -> None:
    with pytest.raises(ValueError, match="OKX_FLAG"):
        RiskConfig(demo_flag="0")


def test_risk_allows_configured_eth_instrument() -> None:
    risk = RiskManager(RiskConfig(allowed_inst_ids=("BTC-USDT", "ETH-USDT")))

    decision = risk.assess(buy(inst_id="ETH-USDT"), account())

    assert decision.allowed


def test_risk_blocks_unconfigured_instrument() -> None:
    risk = RiskManager(RiskConfig(allowed_inst_ids=("BTC-USDT", "ETH-USDT")))

    decision = risk.assess(buy(inst_id="SOL-USDT"), account())

    assert not decision.allowed
    assert decision.reason == "instrument_not_allowed"


def test_risk_blocks_max_position_size() -> None:
    risk = RiskManager(RiskConfig(max_position_btc=Decimal("0.005")))

    decision = risk.assess(buy(size="0.002"), account(position="0.004"))

    assert not decision.allowed
    assert decision.reason == "max_position_size_exceeded"


def test_risk_blocks_cooldown_between_trades() -> None:
    risk = RiskManager(RiskConfig(cooldown_seconds=60))
    now = datetime(2026, 1, 1, tzinfo=UTC)
    risk.record_trade(now, inst_id="BTC-USDT")

    decision = risk.assess(buy(), account(), now=now + timedelta(seconds=10))

    assert not decision.allowed
    assert decision.reason == "cooldown_active"


def test_risk_cooldown_is_per_instrument() -> None:
    risk = RiskManager(RiskConfig(allowed_inst_ids=("BTC-USDT", "ETH-USDT"), cooldown_seconds=60))
    now = datetime(2026, 1, 1, tzinfo=UTC)
    risk.record_trade(now, inst_id="BTC-USDT")

    decision = risk.assess(buy(inst_id="ETH-USDT"), account(), now=now + timedelta(seconds=10))

    assert decision.allowed


def test_risk_stops_after_repeated_failures() -> None:
    risk = RiskManager(RiskConfig(max_consecutive_failures=2))
    risk.record_failure()
    risk.record_failure()

    decision = risk.assess(buy(), account())

    assert not decision.allowed
    assert decision.reason == "max_consecutive_failures_reached"


def test_risk_blocks_order_below_minimum_size() -> None:
    risk = RiskManager(RiskConfig(min_order_btc=Decimal("0.00001")))

    decision = risk.assess(buy(size="0.00000000268"), account())

    assert not decision.allowed
    assert decision.reason == "order_size_below_minimum"


def test_validate_order_accepts_demo_btc_and_eth_orders() -> None:
    risk = RiskManager(RiskConfig(allowed_inst_ids=("BTC-USDT", "ETH-USDT")))

    risk.validate_order(OrderRequest("BTC-USDT", DecisionAction.BUY, Decimal("0.001")))
    risk.validate_order(OrderRequest("ETH-USDT", DecisionAction.BUY, Decimal("0.01")))


def test_risk_blocks_real_daily_loss_from_account_snapshot() -> None:
    risk = RiskManager(RiskConfig(max_daily_loss_usdt=Decimal("25")))

    decision = risk.assess(buy(), account(pnl="-25"))

    assert not decision.allowed
    assert decision.reason == "max_daily_loss_reached"


def test_risk_blocks_drawdown_daily_trade_limit_and_spread_guard() -> None:
    assert (
        RiskManager(RiskConfig(max_drawdown_usdt=Decimal("10")))
        .assess(
            buy(),
            AccountSnapshot(
                equity_usdt=Decimal("1000"),
                available_usdt=Decimal("1000"),
                max_drawdown_usdt=Decimal("10"),
            ),
        )
        .reason
        == "max_drawdown_reached"
    )
    assert (
        RiskManager(RiskConfig(max_daily_trades=2))
        .assess(
            buy(),
            AccountSnapshot(
                equity_usdt=Decimal("1000"),
                available_usdt=Decimal("1000"),
                daily_trade_count=2,
            ),
        )
        .reason
        == "daily_trade_limit_reached"
    )
    assert (
        RiskManager(RiskConfig(max_spread_bps=Decimal("5")))
        .assess(
            buy(),
            AccountSnapshot(
                equity_usdt=Decimal("1000"),
                available_usdt=Decimal("1000"),
                spread_bps=Decimal("6"),
            ),
        )
        .reason
        == "spread_too_wide"
    )


def test_risk_requires_performance_gate_when_configured() -> None:
    risk = RiskManager(RiskConfig(performance_gate_required=True))

    blocked = risk.assess(buy(), account())
    allowed = risk.assess(
        buy(),
        AccountSnapshot(
            equity_usdt=Decimal("1000"),
            available_usdt=Decimal("1000"),
            performance_gate_passed=True,
        ),
    )

    assert blocked.reason == "performance_gate_not_passed"
    assert allowed.allowed
