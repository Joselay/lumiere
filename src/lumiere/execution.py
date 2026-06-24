from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
from decimal import Decimal
from typing import Literal

from lumiere.models import AccountSnapshot, DecisionAction, OrderRequest, StrategyDecision
from lumiere.risk import RiskDecision, RiskManager

ExecutionPolicyName = Literal["market", "marketable_limit", "post_only_maker"]


@dataclass(frozen=True, slots=True)
class OrderBookTop:
    bid: Decimal
    ask: Decimal

    def __post_init__(self) -> None:
        if self.bid <= 0 or self.ask <= 0:
            raise ValueError("order book bid/ask must be positive")
        if self.bid > self.ask:
            raise ValueError("order book bid cannot exceed ask")

    @property
    def mid(self) -> Decimal:
        return (self.bid + self.ask) / Decimal("2")


@dataclass(frozen=True, slots=True)
class ExecutionPolicy:
    name: ExecutionPolicyName = "market"
    marketable_limit_buffer_bps: Decimal = Decimal("1")
    post_only_offset_bps: Decimal = Decimal("0")
    cancel_replace_timeout_seconds: int = 30

    def __post_init__(self) -> None:
        if self.name not in {"market", "marketable_limit", "post_only_maker"}:
            raise ValueError(
                "execution policy must be market, marketable_limit, or post_only_maker"
            )
        if self.marketable_limit_buffer_bps < 0:
            raise ValueError("marketable_limit_buffer_bps cannot be negative")
        if self.post_only_offset_bps < 0:
            raise ValueError("post_only_offset_bps cannot be negative")
        if self.cancel_replace_timeout_seconds <= 0:
            raise ValueError("cancel_replace_timeout_seconds must be positive")


def order_request_for_execution(
    decision: StrategyDecision,
    *,
    size_btc: Decimal,
    td_mode: str,
    policy: ExecutionPolicy,
    order_book: OrderBookTop | None = None,
) -> OrderRequest:
    """Build the live order request for the selected execution policy.

    Market orders do not need book data. Marketable limit and post-only maker policies require
    top-of-book data so limit prices are anchored to executable bid/ask quotes rather than to a
    strategy's stale candle close.
    """

    if policy.name == "market":
        return OrderRequest(decision.inst_id, decision.action, size_btc, td_mode=td_mode)
    if order_book is None:
        raise ValueError("limit execution policies require order book top of book")
    return OrderRequest(
        decision.inst_id,
        decision.action,
        size_btc,
        td_mode=td_mode,
        order_type="limit" if policy.name == "marketable_limit" else "post_only",
        limit_price=limit_price_for_policy(decision.action, policy, order_book),
        cancel_replace_timeout_seconds=policy.cancel_replace_timeout_seconds,
    )


def limit_price_for_policy(
    side: DecisionAction,
    policy: ExecutionPolicy,
    order_book: OrderBookTop,
) -> Decimal:
    if side is DecisionAction.BUY:
        if policy.name == "marketable_limit":
            return order_book.ask * (
                Decimal("1") + policy.marketable_limit_buffer_bps / Decimal("10000")
            )
        if policy.name == "post_only_maker":
            return order_book.bid * (
                Decimal("1") - policy.post_only_offset_bps / Decimal("10000")
            )
    if side is DecisionAction.SELL:
        if policy.name == "marketable_limit":
            return order_book.bid * (
                Decimal("1") - policy.marketable_limit_buffer_bps / Decimal("10000")
            )
        if policy.name == "post_only_maker":
            return order_book.ask * (
                Decimal("1") + policy.post_only_offset_bps / Decimal("10000")
            )
    raise ValueError("limit_price_for_policy requires buy/sell and a limit policy")


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
