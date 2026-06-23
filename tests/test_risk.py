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


def test_risk_blocks_non_btc_instrument() -> None:
    risk = RiskManager(RiskConfig())

    decision = risk.assess(buy(inst_id="ETH-USDT"), account())

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
    risk.record_trade(now)

    decision = risk.assess(buy(), account(), now=now + timedelta(seconds=10))

    assert not decision.allowed
    assert decision.reason == "cooldown_active"


def test_risk_stops_after_repeated_failures() -> None:
    risk = RiskManager(RiskConfig(max_consecutive_failures=2))
    risk.record_failure()
    risk.record_failure()

    decision = risk.assess(buy(), account())

    assert not decision.allowed
    assert decision.reason == "max_consecutive_failures_reached"


def test_validate_order_accepts_demo_btc_order() -> None:
    risk = RiskManager(RiskConfig())

    risk.validate_order(OrderRequest("BTC-USDT", DecisionAction.BUY, Decimal("0.001")))
