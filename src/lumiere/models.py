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


@dataclass(frozen=True, slots=True)
class AccountSnapshot:
    equity_usdt: Decimal
    available_usdt: Decimal
    btc_position: Position | None = None
    daily_realized_pnl_usdt: Decimal = Decimal("0")

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
