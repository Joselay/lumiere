from __future__ import annotations

from collections.abc import Mapping
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
    max_position_by_inst_id: Mapping[str, Decimal] | None = None
    min_order_by_inst_id: Mapping[str, Decimal] | None = None
    max_drawdown_usdt: Decimal | None = None
    max_daily_trades: int | None = None
    max_spread_bps: Decimal | None = None
    min_expected_edge_buffer_bps: Decimal = Decimal("0")
    performance_gate_required: bool = False
    max_risk_per_trade_pct: Decimal = Decimal("0")
    max_portfolio_exposure_pct: Decimal = Decimal("1")
    drawdown_derisk_threshold_usdt: Decimal = Decimal("0")
    drawdown_derisk_multiplier: Decimal = Decimal("0.5")

    def __post_init__(self) -> None:
        if self.demo_flag != "1":
            raise ValueError("OKX demo guard failed: OKX_FLAG must be '1'")
        if not self.allowed_inst_ids:
            raise ValueError("allowed_inst_ids must not be empty")
        if any("-" not in inst_id for inst_id in self.allowed_inst_ids):
            raise ValueError("allowed instruments must be OKX instrument ids like BTC-USDT")
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
        if self.max_drawdown_usdt is not None and self.max_drawdown_usdt <= 0:
            raise ValueError("max_drawdown_usdt must be positive when configured")
        if self.max_daily_trades is not None and self.max_daily_trades <= 0:
            raise ValueError("max_daily_trades must be positive when configured")
        if self.max_spread_bps is not None and self.max_spread_bps <= 0:
            raise ValueError("max_spread_bps must be positive when configured")
        if self.min_expected_edge_buffer_bps < 0:
            raise ValueError("min_expected_edge_buffer_bps cannot be negative")
        if self.max_risk_per_trade_pct < 0 or self.max_risk_per_trade_pct > 1:
            raise ValueError("max_risk_per_trade_pct must be between 0 and 1")
        if self.max_portfolio_exposure_pct <= 0 or self.max_portfolio_exposure_pct > 1:
            raise ValueError("max_portfolio_exposure_pct must be between 0 and 1")
        if self.drawdown_derisk_threshold_usdt < 0:
            raise ValueError("drawdown_derisk_threshold_usdt cannot be negative")
        if self.drawdown_derisk_multiplier <= 0 or self.drawdown_derisk_multiplier > 1:
            raise ValueError("drawdown_derisk_multiplier must be between 0 and 1")
        for inst_id, value in (self.max_position_by_inst_id or {}).items():
            if inst_id not in self.allowed_inst_ids:
                raise ValueError(f"max position configured for disallowed instrument: {inst_id}")
            if value <= 0:
                raise ValueError("per-instrument max positions must be positive")
        for inst_id, value in (self.min_order_by_inst_id or {}).items():
            if inst_id not in self.allowed_inst_ids:
                raise ValueError(f"minimum order configured for disallowed instrument: {inst_id}")
            if value <= 0:
                raise ValueError("per-instrument minimum orders must be positive")

    def max_position_for(self, inst_id: str) -> Decimal:
        if self.max_position_by_inst_id is None:
            return self.max_position_btc
        return self.max_position_by_inst_id.get(inst_id, self.max_position_btc)

    def min_order_for(self, inst_id: str) -> Decimal:
        if self.min_order_by_inst_id is None:
            return self.min_order_btc
        return self.min_order_by_inst_id.get(inst_id, self.min_order_btc)

    @property
    def requires_execution_quality(self) -> bool:
        return self.max_spread_bps is not None or self.min_expected_edge_buffer_bps > 0


@dataclass(frozen=True, slots=True)
class RiskDecision:
    allowed: bool
    reason: str


class RiskManager:
    def __init__(self, config: RiskConfig) -> None:
        self.config = config
        self._last_trade_at_by_inst_id: dict[str, datetime] = {}
        self._consecutive_failures = 0
        self._rejected_by_cost_count = 0

    @property
    def consecutive_failures(self) -> int:
        return self._consecutive_failures

    @property
    def rejected_by_cost_count(self) -> int:
        return self._rejected_by_cost_count

    @property
    def stopped_by_failures(self) -> bool:
        return self._consecutive_failures >= self.config.max_consecutive_failures

    def record_success(self) -> None:
        self._consecutive_failures = 0

    def record_failure(self) -> None:
        self._consecutive_failures += 1

    def record_trade(self, at: datetime | None = None, inst_id: str | None = None) -> None:
        self._last_trade_at_by_inst_id[inst_id or "*"] = at or utc_now()

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
        if self.stopped_by_failures:
            return RiskDecision(False, "max_consecutive_failures_reached")
        if decision.action is DecisionAction.HOLD:
            return RiskDecision(True, "hold_allowed")
        if _protective_exit(decision):
            if decision.action is not DecisionAction.SELL:
                return RiskDecision(False, "protective_exit_must_sell")
            if decision.size_btc <= 0:
                return RiskDecision(False, "non_positive_order_size")
            if decision.size_btc < self.config.min_order_for(decision.inst_id):
                return RiskDecision(False, "order_size_below_minimum")
            return RiskDecision(True, "protective_exit_allowed")
        if account.daily_realized_pnl_usdt <= -self.config.max_daily_loss_usdt:
            return RiskDecision(False, "max_daily_loss_reached")
        if self.config.performance_gate_required and not account.performance_gate_passed:
            return RiskDecision(False, "performance_gate_not_passed")
        if (
            self.config.max_drawdown_usdt is not None
            and account.max_drawdown_usdt >= self.config.max_drawdown_usdt
        ):
            return RiskDecision(False, "max_drawdown_reached")
        if (
            self.config.max_daily_trades is not None
            and account.daily_trade_count >= self.config.max_daily_trades
        ):
            return RiskDecision(False, "daily_trade_limit_reached")
        if self.config.max_spread_bps is not None:
            if account.spread_bps is None:
                return RiskDecision(False, "spread_unavailable")
            if account.spread_bps > self.config.max_spread_bps:
                return RiskDecision(False, "spread_too_wide")
        edge_decision = _expected_edge_bps(decision.inputs)
        expected_edge_bps = (
            edge_decision if edge_decision is not None else account.expected_edge_bps
        )
        if expected_edge_bps is not None and account.estimated_total_cost_bps is not None:
            required_edge = (
                account.estimated_total_cost_bps + self.config.min_expected_edge_buffer_bps
            )
            if expected_edge_bps <= required_edge:
                self._rejected_by_cost_count += 1
                return RiskDecision(False, "expected_edge_below_cost")
        if decision.size_btc < self.config.min_order_for(decision.inst_id):
            return RiskDecision(False, "order_size_below_minimum")
        last_trade_at = self._last_trade_at_by_inst_id.get(
            decision.inst_id,
            self._last_trade_at_by_inst_id.get("*"),
        )
        if last_trade_at is not None:
            cooldown_until = last_trade_at + timedelta(seconds=self.config.cooldown_seconds)
            if now < cooldown_until:
                return RiskDecision(False, "cooldown_active")
        if decision.action is DecisionAction.BUY:
            projected_position = account.position_size(decision.inst_id) + decision.size_btc
            if projected_position > self.config.max_position_for(decision.inst_id):
                return RiskDecision(False, "max_position_size_exceeded")
        if decision.size_btc <= 0:
            return RiskDecision(False, "non_positive_order_size")
        return RiskDecision(True, "risk_checks_passed")

    def clamp_order_size(self, decision: StrategyDecision, account: AccountSnapshot) -> Decimal:
        if decision.action is DecisionAction.HOLD:
            return Decimal("0")
        if _protective_exit(decision) and decision.action is DecisionAction.SELL:
            return decision.size_btc
        size = decision.size_btc
        price = _decision_price(decision)
        if price is not None:
            max_exposure = account.equity_usdt * self.config.max_portfolio_exposure_pct
            max_size_by_exposure = max_exposure / price
            size = min(size, max_size_by_exposure)
            stop_loss_bps = _input_decimal(decision.inputs, "stop_loss_bps")
            volatility_bps = _input_decimal(decision.inputs, "volatility_bps")
            risk_bps = stop_loss_bps or volatility_bps
            if self.config.max_risk_per_trade_pct > 0 and risk_bps is not None and risk_bps > 0:
                risk_budget = account.equity_usdt * self.config.max_risk_per_trade_pct
                risk_per_unit = price * risk_bps / Decimal("10000")
                if risk_per_unit > 0:
                    size = min(size, risk_budget / risk_per_unit)
        if (
            self.config.drawdown_derisk_threshold_usdt > 0
            and account.max_drawdown_usdt >= self.config.drawdown_derisk_threshold_usdt
        ):
            size *= self.config.drawdown_derisk_multiplier
        return max(size, Decimal("0"))

    def validate_order(self, order: OrderRequest) -> None:
        if self.config.demo_flag != "1":
            raise ValueError("OKX demo guard failed: refusing order when OKX_FLAG != '1'")
        if order.inst_id not in self.config.allowed_inst_ids:
            raise ValueError(f"instrument not allowed: {order.inst_id}")
        if order.side not in {DecisionAction.BUY, DecisionAction.SELL}:
            raise ValueError(f"invalid order side: {order.side}")
        if order.size_btc <= 0:
            raise ValueError("order size must be positive")
        if order.size_btc < self.config.min_order_for(order.inst_id):
            raise ValueError("order size below minimum")


def _expected_edge_bps(inputs: Mapping[str, object]) -> Decimal | None:
    return _input_decimal(inputs, "expected_edge_bps")


def _protective_exit(decision: StrategyDecision) -> bool:
    return str(decision.inputs.get("protective_exit", "")).lower() in {"1", "true", "yes"}


def _decision_price(decision: StrategyDecision) -> Decimal | None:
    return _input_decimal(decision.inputs, "decision_price") or _input_decimal(
        decision.inputs,
        "price",
    )


def _input_decimal(inputs: Mapping[str, object], key: str) -> Decimal | None:
    raw = inputs.get(key)
    if raw in {None, ""}:
        return None
    return Decimal(str(raw))
