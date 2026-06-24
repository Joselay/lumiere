from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

from lumiere.backtest import CostModel
from lumiere.ledger import TradeFill, build_pnl_metrics
from lumiere.models import DecisionAction, MarketCandle, StrategyDecision
from lumiere.paper_gate import PerformanceGateConfig, PerformanceGateDecision, assess_metrics


@dataclass(frozen=True, slots=True)
class PaperTradingConfig:
    path: Path
    starting_equity_usdt: Decimal = Decimal("1000")
    cost_model: CostModel = CostModel()
    gate: PerformanceGateConfig = PerformanceGateConfig()
    max_evidence_age: timedelta = timedelta(days=7)


class PaperTradingLedger:
    """Persistent JSONL shadow ledger for forward paper-trading evidence."""

    def __init__(self, config: PaperTradingConfig) -> None:
        self.config = config
        self.config.path.parent.mkdir(parents=True, exist_ok=True)
        self._events = _load_events(self.config.path)

    @property
    def events(self) -> tuple[dict[str, Any], ...]:
        return tuple(self._events)

    def record_decision(
        self,
        decision: StrategyDecision,
        candle: MarketCandle,
        *,
        strategy_name: str,
    ) -> None:
        event = {
            "type": "decision",
            "ts": candle.ts.isoformat(),
            "strategy_name": strategy_name,
            "inst_id": decision.inst_id,
            "action": decision.action.value,
            "size_base": str(decision.size_btc),
            "reason": decision.reason,
            "decision_price": str(candle.close),
            "inputs": _json_safe(decision.inputs),
        }
        self._append(event)
        if decision.action in {DecisionAction.BUY, DecisionAction.SELL} and decision.size_btc > 0:
            price = self.config.cost_model.execution_price(decision.action, candle.close)
            fee = self.config.cost_model.fee_usdt(decision.size_btc * price)
            self._append(
                {
                    "type": "simulated_fill",
                    "ts": candle.ts.isoformat(),
                    "strategy_name": strategy_name,
                    "inst_id": decision.inst_id,
                    "side": decision.action.value,
                    "size_base": str(decision.size_btc),
                    "decision_price": str(candle.close),
                    "fill_price": str(price),
                    "fee_usdt": str(fee),
                    "slippage_bps": str(self.config.cost_model.order_cost_bps),
                    "spread_bps": str(self.config.cost_model.spread_bps),
                }
            )

    def gate_decision(self, *, now: datetime | None = None) -> PerformanceGateDecision:
        now = now or datetime.now(tz=UTC)
        latest = self._latest_event_time()
        if latest is None:
            return PerformanceGateDecision(False, "paper_gate_no_evidence")
        if now - latest > self.config.max_evidence_age:
            return PerformanceGateDecision(False, "paper_gate_decayed")
        fills = self._fills_since(now - self.config.max_evidence_age)
        if not fills:
            return PerformanceGateDecision(False, "paper_gate_no_fills")
        marks = self._latest_prices()
        metrics = build_pnl_metrics(
            fills,
            starting_equity_usdt=self.config.starting_equity_usdt,
            mark_prices=marks,
        )
        return assess_metrics(metrics, self.config.gate)

    def _fills_since(self, since: datetime) -> tuple[TradeFill, ...]:
        fills: list[TradeFill] = []
        for event in self._events:
            if event.get("type") != "simulated_fill":
                continue
            ts = _parse_datetime(str(event["ts"]))
            if ts < since:
                continue
            fills.append(
                TradeFill(
                    inst_id=str(event["inst_id"]),
                    side=DecisionAction(str(event["side"])),
                    size_base=Decimal(str(event["size_base"])),
                    price=Decimal(str(event["fill_price"])),
                    fee=Decimal(str(event["fee_usdt"])),
                    fee_ccy="USDT",
                    ts=ts,
                )
            )
        return tuple(fills)

    def _latest_prices(self) -> dict[str, Decimal]:
        prices: dict[str, Decimal] = {}
        for event in self._events:
            inst_id = event.get("inst_id")
            decision_price = event.get("decision_price")
            if inst_id and decision_price:
                prices[str(inst_id)] = Decimal(str(decision_price))
        return prices

    def _latest_event_time(self) -> datetime | None:
        if not self._events:
            return None
        return max(_parse_datetime(str(event["ts"])) for event in self._events)

    def _append(self, event: dict[str, Any]) -> None:
        self._events.append(event)
        with self.config.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, sort_keys=True) + "\n")


def _load_events(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _parse_datetime(raw: str) -> datetime:
    parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _json_safe(value: Any) -> Any:
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    return value
