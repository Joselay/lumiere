from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from enum import StrEnum
from typing import Any


class DecisionAction(StrEnum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


@dataclass(frozen=True, slots=True)
class MarketCandle:
    ts: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal = Decimal("0")


@dataclass(frozen=True, slots=True)
class Position:
    inst_id: str
    size_btc: Decimal
    avg_px: Decimal = Decimal("0")
    unrealized_pnl_usdt: Decimal = Decimal("0")

    @property
    def size_base(self) -> Decimal:
        return self.size_btc


@dataclass(frozen=True, slots=True)
class AccountSnapshot:
    equity_usdt: Decimal
    available_usdt: Decimal
    btc_position: Position | None = None
    positions: tuple[Position, ...] = ()
    daily_realized_pnl_usdt: Decimal = Decimal("0")
    daily_trade_count: int = 0
    max_drawdown_usdt: Decimal = Decimal("0")
    spread_bps: Decimal | None = None
    performance_gate_passed: bool = False

    def __post_init__(self) -> None:
        positions = tuple(self.positions)
        if self.btc_position is not None and all(
            position.inst_id != self.btc_position.inst_id for position in positions
        ):
            positions = (*positions, self.btc_position)
        btc_position = self.btc_position or next(
            (position for position in positions if position.inst_id.startswith("BTC-")),
            None,
        )
        object.__setattr__(self, "positions", positions)
        object.__setattr__(self, "btc_position", btc_position)

    def position_for(self, inst_id: str) -> Position | None:
        for position in self.positions:
            if position.inst_id == inst_id:
                return position
        return None

    def position_size(self, inst_id: str) -> Decimal:
        position = self.position_for(inst_id)
        if position is None:
            return Decimal("0")
        return position.size_btc

    @property
    def btc_position_size(self) -> Decimal:
        if self.btc_position is None:
            return Decimal("0")
        return self.btc_position.size_btc


@dataclass(frozen=True, slots=True)
class StrategyDecision:
    action: DecisionAction
    inst_id: str
    size_btc: Decimal
    reason: str
    inputs: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def hold(
        cls,
        inst_id: str,
        reason: str,
        inputs: dict[str, Any] | None = None,
    ) -> StrategyDecision:
        return cls(
            action=DecisionAction.HOLD,
            inst_id=inst_id,
            size_btc=Decimal("0"),
            reason=reason,
            inputs=inputs or {},
        )


@dataclass(frozen=True, slots=True)
class OrderRequest:
    inst_id: str
    side: DecisionAction
    size_btc: Decimal
    td_mode: str = "cash"
    order_type: str = "market"


@dataclass(frozen=True, slots=True)
class OrderResult:
    order_id: str
    client_order_id: str | None
    inst_id: str
    side: DecisionAction
    size_btc: Decimal
    status: str
    raw: dict[str, Any] = field(default_factory=dict)


def utc_now() -> datetime:
    return datetime.now(tz=UTC)
