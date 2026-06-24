from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
from decimal import Decimal

from lumiere.models import AccountSnapshot, DecisionAction, StrategyDecision
from lumiere.risk import RiskDecision, RiskManager


@dataclass(frozen=True, slots=True)
class ExecutionPlan:
    """Shared risk/clamp decision used before simulated or live order placement."""

    decision: StrategyDecision
    risk_decision: RiskDecision
    requested_size_btc: Decimal
    clamped_size_btc: Decimal
    blocked_signal_notional_usdt: Decimal = Decimal("0")

    @property
    def allowed(self) -> bool:
        return self.risk_decision.allowed

    @property
    def reason(self) -> str:
        return self.risk_decision.reason


def prepare_execution_plan(
    decision: StrategyDecision,
    account: AccountSnapshot,
    risk_manager: RiskManager | None,
    *,
    now: datetime | None = None,
    mark_price: Decimal | None = None,
) -> ExecutionPlan:
    """Apply live RiskManager assessment plus post-clamp minimum-size checks.

    The same pre-order decision is reusable by live trading, paper trading, and
    backtests so a research fill cannot assume a size or risk state that live
    order validation would reject.
    """

    if risk_manager is None:
        return ExecutionPlan(
            decision=decision,
            risk_decision=RiskDecision(True, "risk_not_configured"),
            requested_size_btc=decision.size_btc,
            clamped_size_btc=decision.size_btc,
        )

    risk_decision = risk_manager.assess(decision, account, now=now)
    if not risk_decision.allowed or decision.action is DecisionAction.HOLD:
        clamped_size = Decimal("0") if decision.action is DecisionAction.HOLD else decision.size_btc
        return ExecutionPlan(
            decision=decision,
            risk_decision=risk_decision,
            requested_size_btc=decision.size_btc,
            clamped_size_btc=clamped_size,
            blocked_signal_notional_usdt=_blocked_notional(decision, mark_price),
        )

    clamped_size = risk_manager.clamp_order_size(decision, account)
    minimum = risk_manager.config.min_order_for(decision.inst_id)
    if clamped_size < minimum:
        return ExecutionPlan(
            decision=decision,
            risk_decision=RiskDecision(False, "clamped_order_size_below_minimum"),
            requested_size_btc=decision.size_btc,
            clamped_size_btc=clamped_size,
            blocked_signal_notional_usdt=_blocked_notional(decision, mark_price),
        )

    clamped_decision = replace(
        decision,
        size_btc=clamped_size,
        inputs=decision.inputs
        | {
            "requested_size_btc": str(decision.size_btc),
            "clamped_size_btc": str(clamped_size),
        },
    )
    return ExecutionPlan(
        decision=clamped_decision,
        risk_decision=risk_decision,
        requested_size_btc=decision.size_btc,
        clamped_size_btc=clamped_size,
    )


def _blocked_notional(decision: StrategyDecision, mark_price: Decimal | None) -> Decimal:
    if decision.action not in {DecisionAction.BUY, DecisionAction.SELL} or mark_price is None:
        return Decimal("0")
    return abs(decision.size_btc * mark_price)
