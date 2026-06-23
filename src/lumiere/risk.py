from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal

from lumiere.models import AccountSnapshot, DecisionAction, OrderRequest, StrategyDecision, utc_now


@dataclass(frozen=True, slots=True)
class RiskConfig:
    demo_flag: str = "1"
    allowed_inst_ids: tuple[str, ...] = ("BTC-USDT",)
    max_position_btc: Decimal = Decimal("0.005")
    min_order_btc: Decimal = Decimal("0.00001")
    max_daily_loss_usdt: Decimal = Decimal("25")
    cooldown_seconds: int = 300
    max_consecutive_failures: int = 3

    def __post_init__(self) -> None:
        if self.demo_flag != "1":
            raise ValueError("OKX demo guard failed: OKX_FLAG must be '1'")
        if not self.allowed_inst_ids:
            raise ValueError("allowed_inst_ids must not be empty")
        if any(not inst_id.startswith("BTC-") for inst_id in self.allowed_inst_ids):
            raise ValueError("BTC-only guard failed: all allowed instruments must start with BTC-")
        if self.max_position_btc <= 0:
            raise ValueError("max_position_btc must be positive")
        if self.min_order_btc <= 0:
            raise ValueError("min_order_btc must be positive")
        if self.max_daily_loss_usdt <= 0:
            raise ValueError("max_daily_loss_usdt must be positive")
        if self.cooldown_seconds < 0:
            raise ValueError("cooldown_seconds cannot be negative")
        if self.max_consecutive_failures <= 0:
            raise ValueError("max_consecutive_failures must be positive")


@dataclass(frozen=True, slots=True)
class RiskDecision:
    allowed: bool
    reason: str


class RiskManager:
    def __init__(self, config: RiskConfig) -> None:
        self.config = config
        self._last_trade_at: datetime | None = None
        self._consecutive_failures = 0

    @property
    def consecutive_failures(self) -> int:
        return self._consecutive_failures

    @property
    def stopped_by_failures(self) -> bool:
        return self._consecutive_failures >= self.config.max_consecutive_failures

    def record_success(self) -> None:
        self._consecutive_failures = 0

    def record_failure(self) -> None:
        self._consecutive_failures += 1

    def record_trade(self, at: datetime | None = None) -> None:
        self._last_trade_at = at or utc_now()

    def assess(
        self,
        decision: StrategyDecision,
        account: AccountSnapshot,
        now: datetime | None = None,
    ) -> RiskDecision:
        now = now or utc_now()
        if self.config.demo_flag != "1":
            return RiskDecision(False, "demo_guard_failed")
        if decision.inst_id not in self.config.allowed_inst_ids:
            return RiskDecision(False, "instrument_not_allowed")
        if not decision.inst_id.startswith("BTC-"):
            return RiskDecision(False, "btc_only_guard_failed")
        if self.stopped_by_failures:
            return RiskDecision(False, "max_consecutive_failures_reached")
        if account.daily_realized_pnl_usdt <= -self.config.max_daily_loss_usdt:
            return RiskDecision(False, "max_daily_loss_reached")
        if decision.action is DecisionAction.HOLD:
            return RiskDecision(True, "hold_allowed")
        if decision.size_btc < self.config.min_order_btc:
            return RiskDecision(False, "order_size_below_minimum")
        if self._last_trade_at is not None:
            cooldown_until = self._last_trade_at + timedelta(seconds=self.config.cooldown_seconds)
            if now < cooldown_until:
                return RiskDecision(False, "cooldown_active")
        if decision.action is DecisionAction.BUY:
            projected_position = account.btc_position_size + decision.size_btc
            if projected_position > self.config.max_position_btc:
                return RiskDecision(False, "max_position_size_exceeded")
        if decision.size_btc <= 0:
            return RiskDecision(False, "non_positive_order_size")
        return RiskDecision(True, "risk_checks_passed")

    def validate_order(self, order: OrderRequest) -> None:
        if self.config.demo_flag != "1":
            raise ValueError("OKX demo guard failed: refusing order when OKX_FLAG != '1'")
        if order.inst_id not in self.config.allowed_inst_ids or not order.inst_id.startswith(
            "BTC-"
        ):
            raise ValueError(f"instrument not allowed: {order.inst_id}")
        if order.side not in {DecisionAction.BUY, DecisionAction.SELL}:
            raise ValueError(f"invalid order side: {order.side}")
        if order.size_btc <= 0:
            raise ValueError("order size must be positive")
        if order.size_btc < self.config.min_order_btc:
            raise ValueError("order size below minimum")
