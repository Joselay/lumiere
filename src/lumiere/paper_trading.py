from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

from lumiere.backtest import CostModel
from lumiere.ledger import EquityPoint, TradeFill, build_pnl_metrics, max_drawdown_usdt
from lumiere.models import AccountSnapshot, DecisionAction, MarketCandle, Position, StrategyDecision
from lumiere.paper_gate import PerformanceGateConfig, PerformanceGateDecision, assess_metrics


@dataclass(frozen=True, slots=True)
class PaperTradingConfig:
    path: Path
    starting_equity_usdt: Decimal = Decimal("1000")
    cost_model: CostModel = CostModel()
    gate: PerformanceGateConfig = PerformanceGateConfig()
    max_evidence_age: timedelta = timedelta(days=7)


@dataclass(slots=True)
class _PaperPosition:
    inst_id: str
    size_base: Decimal = Decimal("0")
    cost_basis_usdt: Decimal = Decimal("0")
    realized_pnl_usdt: Decimal = Decimal("0")

    @property
    def avg_px(self) -> Decimal:
        if self.size_base <= 0:
            return Decimal("0")
        return self.cost_basis_usdt / self.size_base


@dataclass(slots=True)
class _PaperPortfolio:
    starting_equity_usdt: Decimal
    cash_usdt: Decimal
    positions: dict[str, _PaperPosition] = field(default_factory=dict)
    latest_prices: dict[str, Decimal] = field(default_factory=dict)
    realized_pnl_usdt: Decimal = Decimal("0")
    fees_usdt: Decimal = Decimal("0")
    trade_count: int = 0
    fills: list[TradeFill] = field(default_factory=list)
    equity_curve: list[EquityPoint] = field(default_factory=list)
    rejected_fills: int = 0

    def mark(self, inst_id: str, price: Decimal, ts: datetime) -> None:
        self.latest_prices[inst_id] = price
        self.record_equity(ts)

    def record_equity(self, ts: datetime) -> None:
        self.equity_curve.append(EquityPoint(ts=ts, equity_usdt=self.equity_usdt()))

    def equity_usdt(self, mark_prices: dict[str, Decimal] | None = None) -> Decimal:
        prices = {**self.latest_prices, **(mark_prices or {})}
        equity = self.cash_usdt
        for inst_id, position in self.positions.items():
            if position.size_base > 0:
                equity += position.size_base * prices.get(inst_id, Decimal("0"))
        return equity

    def unrealized_pnl_usdt(self, mark_prices: dict[str, Decimal] | None = None) -> Decimal:
        prices = {**self.latest_prices, **(mark_prices or {})}
        unrealized = Decimal("0")
        for inst_id, position in self.positions.items():
            if position.size_base > 0:
                unrealized += (
                    position.size_base * prices.get(inst_id, Decimal("0"))
                    - position.cost_basis_usdt
                )
        return unrealized


class PaperTradingLedger:
    """Persistent JSONL shadow ledger for self-contained paper-trading evidence.

    The ledger owns a shadow cash account and per-instrument cost basis. It never borrows
    inventory from the live/demo account, so paper sells without paper inventory are rejected
    instead of becoming fills with fabricated cost basis.
    """

    def __init__(self, config: PaperTradingConfig) -> None:
        self.config = config
        self.config.path.parent.mkdir(parents=True, exist_ok=True)
        self._events = _load_events(self.config.path)
        self._portfolio = self._replay(self._events)

    @property
    def events(self) -> tuple[dict[str, Any], ...]:
        return tuple(self._events)

    def account_snapshot(
        self,
        *,
        mark_prices: dict[str, Decimal] | None = None,
        now: datetime | None = None,
    ) -> AccountSnapshot:
        """Return the current shadow account state for strategy and gate decisions."""

        prices = {**self._portfolio.latest_prices, **(mark_prices or {})}
        positions = tuple(
            Position(
                inst_id=inst_id,
                size_btc=position.size_base,
                avg_px=position.avg_px,
                unrealized_pnl_usdt=(
                    position.size_base * prices.get(inst_id, Decimal("0"))
                    - position.cost_basis_usdt
                ),
            )
            for inst_id, position in sorted(self._portfolio.positions.items())
            if position.size_base > 0
        )
        gate = self.gate_decision(now=now)
        return AccountSnapshot(
            equity_usdt=self._portfolio.equity_usdt(mark_prices),
            available_usdt=max(self._portfolio.cash_usdt, Decimal("0")),
            positions=positions,
            daily_realized_pnl_usdt=self._portfolio.realized_pnl_usdt,
            daily_trade_count=self._portfolio.trade_count,
            max_drawdown_usdt=max_drawdown_usdt(tuple(self._portfolio.equity_curve)),
            performance_gate_passed=gate.allowed,
            performance_gate_reason=gate.reason,
        )

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
        if (
            decision.action not in {DecisionAction.BUY, DecisionAction.SELL}
            or decision.size_btc <= 0
        ):
            self._append_portfolio_state(
                candle.ts,
                strategy_name=strategy_name,
                inst_id=decision.inst_id,
            )
            return

        fill_size = self._paper_fill_size(decision)
        if fill_size <= 0:
            self._append(
                {
                    "type": "simulated_rejection",
                    "ts": candle.ts.isoformat(),
                    "strategy_name": strategy_name,
                    "inst_id": decision.inst_id,
                    "side": decision.action.value,
                    "size_base": str(decision.size_btc),
                    "decision_price": str(candle.close),
                    "reason": "paper_no_inventory"
                    if decision.action is DecisionAction.SELL
                    else "insufficient_paper_cash",
                }
            )
            self._append_portfolio_state(
                candle.ts,
                strategy_name=strategy_name,
                inst_id=decision.inst_id,
            )
            return

        price = self.config.cost_model.execution_price(decision.action, candle.close)
        fee = self.config.cost_model.fee_usdt(fill_size * price)
        if (
            decision.action is DecisionAction.BUY
            and fill_size * price + fee > self._portfolio.cash_usdt
        ):
            self._append(
                {
                    "type": "simulated_rejection",
                    "ts": candle.ts.isoformat(),
                    "strategy_name": strategy_name,
                    "inst_id": decision.inst_id,
                    "side": decision.action.value,
                    "size_base": str(decision.size_btc),
                    "decision_price": str(candle.close),
                    "reason": "insufficient_paper_cash",
                }
            )
            self._append_portfolio_state(
                candle.ts,
                strategy_name=strategy_name,
                inst_id=decision.inst_id,
            )
            return

        self._append(
            {
                "type": "simulated_fill",
                "ts": candle.ts.isoformat(),
                "strategy_name": strategy_name,
                "inst_id": decision.inst_id,
                "side": decision.action.value,
                "size_base": str(fill_size),
                "requested_size_base": str(decision.size_btc),
                "decision_price": str(candle.close),
                "fill_price": str(price),
                "fee_usdt": str(fee),
                "slippage_bps": str(self.config.cost_model.order_cost_bps),
                "spread_bps": str(self.config.cost_model.spread_bps),
            }
        )
        self._append_portfolio_state(
            candle.ts,
            strategy_name=strategy_name,
            inst_id=decision.inst_id,
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
        metrics = build_pnl_metrics(
            fills,
            starting_equity_usdt=self.config.starting_equity_usdt,
            mark_prices=self._portfolio.latest_prices,
        )
        return assess_metrics(metrics, self.config.gate)

    def _paper_fill_size(self, decision: StrategyDecision) -> Decimal:
        if decision.action is DecisionAction.BUY:
            return decision.size_btc
        position = self._portfolio.positions.get(decision.inst_id)
        if position is None or position.size_base <= 0:
            return Decimal("0")
        return min(decision.size_btc, position.size_base)

    def _fills_since(self, since: datetime) -> tuple[TradeFill, ...]:
        return tuple(fill for fill in self._portfolio.fills if fill.ts >= since)

    def _latest_event_time(self) -> datetime | None:
        if not self._events:
            return None
        return max(_parse_datetime(str(event["ts"])) for event in self._events if "ts" in event)

    def _append(self, event: dict[str, Any]) -> None:
        self._events.append(event)
        self._apply_event(self._portfolio, event)
        with self.config.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, sort_keys=True) + "\n")

    def _append_portfolio_state(self, ts: datetime, *, strategy_name: str, inst_id: str) -> None:
        self._append(
            {
                "type": "portfolio_state",
                "ts": ts.isoformat(),
                "strategy_name": strategy_name,
                "inst_id": inst_id,
                "cash_usdt": str(self._portfolio.cash_usdt),
                "equity_usdt": str(self._portfolio.equity_usdt()),
                "realized_pnl_usdt": str(self._portfolio.realized_pnl_usdt),
                "unrealized_pnl_usdt": str(self._portfolio.unrealized_pnl_usdt()),
                "fees_usdt": str(self._portfolio.fees_usdt),
                "trade_count": self._portfolio.trade_count,
                "positions": [
                    {
                        "inst_id": position.inst_id,
                        "size_base": str(position.size_base),
                        "avg_px": str(position.avg_px),
                        "cost_basis_usdt": str(position.cost_basis_usdt),
                        "realized_pnl_usdt": str(position.realized_pnl_usdt),
                    }
                    for position in sorted(
                        self._portfolio.positions.values(), key=lambda item: item.inst_id
                    )
                    if position.size_base > 0
                ],
            }
        )

    def _replay(self, events: list[dict[str, Any]]) -> _PaperPortfolio:
        portfolio = _PaperPortfolio(
            starting_equity_usdt=self.config.starting_equity_usdt,
            cash_usdt=self.config.starting_equity_usdt,
        )
        for event in events:
            self._apply_event(portfolio, event)
        return portfolio

    def _apply_event(self, portfolio: _PaperPortfolio, event: dict[str, Any]) -> None:
        event_type = event.get("type")
        if event_type == "decision":
            inst_id = str(event["inst_id"])
            price = Decimal(str(event["decision_price"]))
            portfolio.mark(inst_id, price, _parse_datetime(str(event["ts"])))
            return
        if event_type == "simulated_fill":
            fill = TradeFill(
                inst_id=str(event["inst_id"]),
                side=DecisionAction(str(event["side"])),
                size_base=Decimal(str(event["size_base"])),
                price=Decimal(str(event["fill_price"])),
                fee=Decimal(str(event["fee_usdt"])),
                fee_ccy="USDT",
                ts=_parse_datetime(str(event["ts"])),
            )
            self._apply_fill(portfolio, fill)
            portfolio.record_equity(fill.ts)
            return
        if event_type == "simulated_rejection":
            portfolio.rejected_fills += 1

    def _apply_fill(self, portfolio: _PaperPortfolio, fill: TradeFill) -> None:
        portfolio.latest_prices[fill.inst_id] = fill.price
        fee_cost = fill.fee_cost_usdt()
        position = portfolio.positions.setdefault(fill.inst_id, _PaperPosition(fill.inst_id))
        if fill.side is DecisionAction.BUY:
            total_cost = fill.notional_usdt + fee_cost
            if total_cost > portfolio.cash_usdt:
                portfolio.rejected_fills += 1
                return
            portfolio.cash_usdt -= total_cost
            position.size_base += fill.size_base
            position.cost_basis_usdt += total_cost
        else:
            if position.size_base <= 0 or fill.size_base > position.size_base:
                portfolio.rejected_fills += 1
                return
            removed_cost_basis = position.avg_px * fill.size_base
            proceeds = fill.notional_usdt
            close_pnl = proceeds - removed_cost_basis - fee_cost
            portfolio.cash_usdt += proceeds - fee_cost
            position.size_base -= fill.size_base
            position.cost_basis_usdt -= removed_cost_basis
            position.realized_pnl_usdt += close_pnl
            portfolio.realized_pnl_usdt += close_pnl
            if position.size_base <= 0:
                position.size_base = Decimal("0")
                position.cost_basis_usdt = Decimal("0")
        portfolio.fees_usdt += fee_cost
        portfolio.trade_count += 1
        portfolio.fills.append(fill)


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
