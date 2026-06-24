from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from lumiere.ledger import TradeFill
from lumiere.models import (
    AccountSnapshot,
    DecisionAction,
    MarketCandle,
    OrderResult,
    Position,
    StrategyDecision,
)


@dataclass(frozen=True, slots=True)
class LivePositionState:
    """Persisted explanation and exit state for one live/demo position."""

    inst_id: str
    size_base: Decimal
    entry_price: Decimal
    opened_at: datetime
    updated_at: datetime
    highest_price: Decimal
    last_bar_ts: datetime | None = None
    bars_open: int = 0
    strategy_name: str = "unknown"
    entry_reason: str = "unknown"
    source: str = "live_order"

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        for key, value in list(data.items()):
            if isinstance(value, Decimal):
                data[key] = str(value)
            elif isinstance(value, datetime):
                data[key] = value.isoformat()
            elif value is None:
                data[key] = None
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LivePositionState:
        return cls(
            inst_id=str(data["inst_id"]),
            size_base=Decimal(str(data["size_base"])),
            entry_price=Decimal(str(data["entry_price"])),
            opened_at=_parse_datetime(data["opened_at"]),
            updated_at=_parse_datetime(data["updated_at"]),
            highest_price=Decimal(str(data["highest_price"])),
            last_bar_ts=_parse_datetime(data["last_bar_ts"]) if data.get("last_bar_ts") else None,
            bars_open=int(data.get("bars_open", 0)),
            strategy_name=str(data.get("strategy_name") or "unknown"),
            entry_reason=str(data.get("entry_reason") or "unknown"),
            source=str(data.get("source") or "live_order"),
        )


class LivePositionStore:
    """Small JSON store for position provenance and deterministic live exits."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path else None
        self._positions: dict[str, LivePositionState] = {}
        self._load()

    def get(self, inst_id: str) -> LivePositionState | None:
        return self._positions.get(inst_id)

    def managed_inst_ids(self) -> set[str]:
        return set(self._positions)

    def all(self) -> tuple[LivePositionState, ...]:
        return tuple(self._positions[inst_id] for inst_id in sorted(self._positions))

    def describe(self, account: AccountSnapshot | None = None) -> tuple[dict[str, Any], ...]:
        rows: list[dict[str, Any]] = []
        for state in self.all():
            position = account.position_for(state.inst_id) if account is not None else None
            rows.append(
                state.to_dict()
                | {
                    "managed": True,
                    "account_size_base": str(position.size_base) if position else "0",
                    "unrealized_pnl_usdt": str(position.unrealized_pnl_usdt) if position else "0",
                }
            )
        if account is not None:
            for position in account.positions:
                if position.size_base > 0 and position.inst_id not in self._positions:
                    rows.append(
                        {
                            "inst_id": position.inst_id,
                            "managed": False,
                            "account_size_base": str(position.size_base),
                            "entry_price": str(position.avg_px),
                            "unrealized_pnl_usdt": str(position.unrealized_pnl_usdt),
                            "source": "unexpected_account_inventory",
                            "entry_reason": "requires_flatten_ignore_or_adopt_policy",
                        }
                    )
        return tuple(rows)

    def adopt_unexpected_position(
        self,
        position: Position,
        *,
        current_price: Decimal,
        now: datetime,
        strategy_name: str,
        reason: str = "adopted_unexpected_position",
    ) -> LivePositionState:
        entry_price = position.avg_px if position.avg_px > 0 else current_price
        state = LivePositionState(
            inst_id=position.inst_id,
            size_base=position.size_base,
            entry_price=entry_price,
            opened_at=now,
            updated_at=now,
            highest_price=max(entry_price, current_price),
            last_bar_ts=now,
            bars_open=0,
            strategy_name=strategy_name,
            entry_reason=reason,
            source="adopted_account_inventory",
        )
        self._positions[position.inst_id] = state
        self._save()
        return state

    def observe_bar(self, inst_id: str, candle: MarketCandle) -> LivePositionState | None:
        state = self._positions.get(inst_id)
        if state is None:
            return None
        bars_open = state.bars_open
        if state.last_bar_ts is None or candle.ts > state.last_bar_ts:
            bars_open += 1
        updated = replace(
            state,
            highest_price=max(state.highest_price, candle.high, candle.close),
            last_bar_ts=candle.ts,
            bars_open=bars_open,
            updated_at=candle.ts,
        )
        self._positions[inst_id] = updated
        self._save()
        return updated

    def record_order_execution(
        self,
        decision: StrategyDecision,
        result: OrderResult,
        *,
        fills: tuple[TradeFill, ...] = (),
        candle: MarketCandle,
        strategy_name: str,
    ) -> None:
        filled_size = _filled_size(result, fills)
        if filled_size <= 0:
            return
        fill_price = _fill_vwap(fills) or _decision_price(decision) or candle.close
        now = candle.ts
        current = self._positions.get(decision.inst_id)
        if decision.action is DecisionAction.BUY:
            if current is None:
                state = LivePositionState(
                    inst_id=decision.inst_id,
                    size_base=filled_size,
                    entry_price=fill_price,
                    opened_at=now,
                    updated_at=now,
                    highest_price=max(fill_price, candle.high, candle.close),
                    last_bar_ts=candle.ts,
                    bars_open=0,
                    strategy_name=strategy_name,
                    entry_reason=decision.reason,
                    source="live_order",
                )
            else:
                new_size = current.size_base + filled_size
                entry_price = (
                    (current.entry_price * current.size_base) + (fill_price * filled_size)
                ) / new_size
                state = replace(
                    current,
                    size_base=new_size,
                    entry_price=entry_price,
                    updated_at=now,
                    highest_price=max(current.highest_price, fill_price, candle.high, candle.close),
                    last_bar_ts=candle.ts,
                    strategy_name=strategy_name,
                    entry_reason=decision.reason,
                )
            self._positions[decision.inst_id] = state
            self._save()
            return
        if decision.action is DecisionAction.SELL and current is not None:
            remaining = max(current.size_base - filled_size, Decimal("0"))
            if remaining <= 0:
                self._positions.pop(decision.inst_id, None)
            else:
                self._positions[decision.inst_id] = replace(
                    current,
                    size_base=remaining,
                    updated_at=now,
                    last_bar_ts=candle.ts,
                )
            self._save()

    def reconcile_account(self, account: AccountSnapshot) -> None:
        changed = False
        for inst_id in list(self._positions):
            state = self._positions[inst_id]
            position = account.position_for(inst_id)
            if position is None or position.size_base <= 0:
                del self._positions[inst_id]
                changed = True
                continue
            entry_price = position.avg_px if position.avg_px > 0 else state.entry_price
            updated = replace(
                state,
                size_base=position.size_base,
                entry_price=entry_price,
                updated_at=state.updated_at,
            )
            if updated != state:
                self._positions[inst_id] = updated
                changed = True
        if changed:
            self._save()

    def _load(self) -> None:
        if self.path is None or not self.path.exists():
            return
        data = json.loads(self.path.read_text())
        if isinstance(data, dict):
            positions = data.get("positions", [])
        elif isinstance(data, list):
            positions = data
        else:
            positions = []
        self._positions = {
            state.inst_id: state
            for state in (LivePositionState.from_dict(row) for row in positions)
            if state.size_base > 0
        }

    def _save(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"positions": [state.to_dict() for state in self.all()]}
        self.path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _filled_size(result: OrderResult, fills: tuple[TradeFill, ...]) -> Decimal:
    filled = sum((fill.size_base for fill in fills), Decimal("0"))
    return filled if filled > 0 else result.size_btc


def _fill_vwap(fills: tuple[TradeFill, ...]) -> Decimal | None:
    size = sum((fill.size_base for fill in fills), Decimal("0"))
    if size <= 0:
        return None
    notional = sum((fill.size_base * fill.price for fill in fills), Decimal("0"))
    return notional / size


def _decision_price(decision: StrategyDecision) -> Decimal | None:
    raw = decision.inputs.get("decision_price") or decision.inputs.get("price")
    if raw in {None, ""}:
        return None
    return Decimal(str(raw))


def _parse_datetime(value: Any) -> datetime:
    parsed = datetime.fromisoformat(str(value))
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed
