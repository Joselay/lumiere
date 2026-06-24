from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

from lumiere.ledger import TradeFill, build_pnl_metrics
from lumiere.models import AccountSnapshot, DecisionAction, MarketCandle, OrderRequest, OrderResult


@dataclass(frozen=True, slots=True)
class AttributionReport:
    window_start: datetime
    window_end: datetime
    metrics: dict[str, Any]
    rejection_counts: dict[str, int]
    alerts: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "window_start": self.window_start.isoformat(),
            "window_end": self.window_end.isoformat(),
            "metrics": self.metrics,
            "rejection_counts": self.rejection_counts,
            "alerts": list(self.alerts),
        }


class AttributionLedger:
    """Persistent JSONL signal -> risk -> order/fill -> PnL attribution store."""

    def __init__(self, path: Path, *, starting_equity_usdt: Decimal = Decimal("1000")) -> None:
        self.path = path
        self.starting_equity_usdt = starting_equity_usdt
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._events = _load_events(path)

    @property
    def events(self) -> tuple[dict[str, Any], ...]:
        return tuple(self._events)

    def record_candle(self, inst_id: str, candle: MarketCandle) -> None:
        self.record_event(
            "candle",
            candle.ts,
            inst_id=inst_id,
            open=str(candle.open),
            high=str(candle.high),
            low=str(candle.low),
            close=str(candle.close),
            volume=str(candle.volume),
        )

    def record_account(self, account: AccountSnapshot, *, ts: datetime | None = None) -> None:
        self.record_event(
            "account",
            ts or datetime.now(tz=UTC),
            equity_usdt=str(account.equity_usdt),
            available_usdt=str(account.available_usdt),
            spread_bps=None if account.spread_bps is None else str(account.spread_bps),
            estimated_total_cost_bps=None
            if account.estimated_total_cost_bps is None
            else str(account.estimated_total_cost_bps),
            performance_gate_passed=account.performance_gate_passed,
            performance_gate_reason=account.performance_gate_reason,
        )

    def record_decision(self, strategy_name: str, decision, *, ts: datetime) -> None:
        self.record_event(
            "decision",
            ts,
            strategy_name=strategy_name,
            inst_id=decision.inst_id,
            action=decision.action.value,
            size_base=str(decision.size_btc),
            reason=decision.reason,
            inputs=_json_safe(decision.inputs),
        )

    def record_risk(
        self,
        inst_id: str,
        action: str,
        allowed: bool,
        reason: str,
        *,
        ts: datetime,
    ) -> None:
        self.record_event(
            "risk",
            ts,
            inst_id=inst_id,
            action=action,
            allowed=allowed,
            reason=reason,
        )

    def record_order(self, order: OrderRequest, result: OrderResult, *, ts: datetime) -> None:
        self.record_event(
            "order",
            ts,
            inst_id=order.inst_id,
            side=order.side.value,
            size_base=str(order.size_btc),
            order_type=order.order_type,
            order_id=result.order_id,
            client_order_id=result.client_order_id,
            status=result.status,
            raw=_json_safe(result.raw),
        )

    def record_fill(
        self,
        *,
        inst_id: str,
        side: DecisionAction,
        size_base: Decimal,
        price: Decimal,
        fee: Decimal,
        ts: datetime,
        decision_price: Decimal | None = None,
    ) -> None:
        self.record_event(
            "fill",
            ts,
            inst_id=inst_id,
            side=side.value,
            size_base=str(size_base),
            price=str(price),
            fee=str(fee),
            decision_price=None if decision_price is None else str(decision_price),
        )

    def record_event(self, event_type: str, ts: datetime, **payload: Any) -> None:
        event = {"type": event_type, "ts": _normalise(ts).isoformat(), **payload}
        self._events.append(event)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, sort_keys=True) + "\n")

    def report(
        self,
        *,
        window: timedelta = timedelta(days=1),
        now: datetime | None = None,
    ) -> AttributionReport:
        now = now or datetime.now(tz=UTC)
        start = now - window
        events = [event for event in self._events if _parse_datetime(event["ts"]) >= start]
        fills = [_fill_from_event(event) for event in events if event.get("type") == "fill"]
        marks = _latest_mark_prices(events)
        metrics = build_pnl_metrics(
            fills,
            starting_equity_usdt=self.starting_equity_usdt,
            mark_prices=marks,
        )
        rejection_counts: dict[str, int] = {}
        for event in events:
            if event.get("type") == "risk" and not event.get("allowed"):
                reason = str(event.get("reason") or "unknown")
                rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
        slippages = _slippage_bps(events)
        alerts = _alerts(metrics.to_dict(), rejection_counts, slippages, events)
        payload = metrics.to_dict()
        payload["average_slippage_bps"] = (
            None if not slippages else str(sum(slippages) / len(slippages))
        )
        payload["risk_rejections"] = rejection_counts
        payload["baseline_comparison"] = {"no_trade_pnl_usdt": "0"}
        return AttributionReport(start, now, payload, rejection_counts, tuple(alerts))


def _load_events(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _fill_from_event(event: dict[str, Any]) -> TradeFill:
    return TradeFill(
        inst_id=str(event["inst_id"]),
        side=DecisionAction(str(event["side"])),
        size_base=Decimal(str(event["size_base"])),
        price=Decimal(str(event["price"])),
        fee=Decimal(str(event.get("fee") or "0")),
        ts=_parse_datetime(str(event["ts"])),
    )


def _latest_mark_prices(events: list[dict[str, Any]]) -> dict[str, Decimal]:
    prices: dict[str, Decimal] = {}
    for event in events:
        if event.get("type") == "candle":
            prices[str(event["inst_id"])] = Decimal(str(event["close"]))
        if event.get("type") == "fill":
            prices[str(event["inst_id"])] = Decimal(str(event["price"]))
    return prices


def _slippage_bps(events: list[dict[str, Any]]) -> list[Decimal]:
    values: list[Decimal] = []
    for event in events:
        if event.get("type") != "fill" or event.get("decision_price") in {None, ""}:
            continue
        decision_price = Decimal(str(event["decision_price"]))
        fill_price = Decimal(str(event["price"]))
        if decision_price > 0:
            values.append(abs(fill_price - decision_price) / decision_price * Decimal("10000"))
    return values


def _alerts(
    metrics: dict[str, Any],
    rejection_counts: dict[str, int],
    slippages: list[Decimal],
    events: list[dict[str, Any]],
) -> list[str]:
    alerts: list[str] = []
    if Decimal(str(metrics["net_pnl_usdt"])) < 0:
        alerts.append("negative_rolling_expectancy")
    if slippages and max(slippages) > Decimal("25"):
        alerts.append("abnormal_slippage")
    if rejection_counts.get("expected_edge_below_cost", 0) > 0:
        alerts.append("cost_gate_rejections")
    if any(event.get("performance_gate_passed") is False for event in events):
        alerts.append("performance_gate_failure")
    return alerts


def _parse_datetime(raw: str) -> datetime:
    parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    return _normalise(parsed)


def _normalise(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _json_safe(value: Any) -> Any:
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    return value
