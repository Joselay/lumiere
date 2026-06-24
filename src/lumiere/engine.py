from __future__ import annotations

import asyncio
from collections.abc import Sequence
from contextlib import suppress
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from decimal import Decimal, InvalidOperation
from typing import Protocol

import structlog

from lumiere.attribution import AttributionLedger
from lumiere.ledger import TradeFill
from lumiere.models import (
    AccountSnapshot,
    DecisionAction,
    OrderRequest,
    OrderResult,
    StrategyDecision,
    utc_now,
)
from lumiere.paper_trading import PaperTradingLedger
from lumiere.risk import RiskDecision, RiskManager
from lumiere.strategy import TradingStrategy
from lumiere.telegram_ui import (
    format_error,
    format_lifecycle,
    format_order_submitted,
    format_panic,
    format_pause,
    format_performance,
    format_resume,
    format_risk,
    format_risk_blocked,
    format_status,
)

log = structlog.get_logger(__name__)


class TradingClient(Protocol):
    async def fetch_candles(self, inst_id: str | None = None): ...

    async def fetch_account_snapshot(self) -> AccountSnapshot: ...

    async def place_market_order(self, request: OrderRequest) -> OrderResult: ...

    async def cancel_open_orders(self) -> list[dict]: ...


class FillAwareTradingClient(Protocol):
    async def fetch_order_fills(
        self,
        result: OrderResult,
        *,
        decision_price: Decimal | None = None,
        submitted_at: datetime | None = None,
    ) -> Sequence[TradeFill]: ...


class Notifier(Protocol):
    async def send(self, text: str) -> None: ...


class NullNotifier:
    async def send(self, text: str) -> None:
        _ = text


@dataclass(frozen=True, slots=True)
class EngineConfig:
    poll_interval_seconds: float = 30.0
    td_mode: str = "cash"
    order_type: str = "market"


@dataclass(frozen=True, slots=True)
class EngineStatus:
    running: bool
    paused: bool
    panic_stopped: bool
    consecutive_failures: int
    last_decision: str | None
    last_risk_reason: str | None
    last_error: str | None


class TradingEngine:
    def __init__(
        self,
        client: TradingClient,
        strategy: TradingStrategy | Sequence[TradingStrategy],
        risk_manager: RiskManager,
        notifier: Notifier | None = None,
        config: EngineConfig | None = None,
        paper_ledger: PaperTradingLedger | None = None,
        attribution_ledger: AttributionLedger | None = None,
    ) -> None:
        self.client = client
        self.strategies = _normalise_strategies(strategy)
        self.strategy = self.strategies[0]
        self.risk_manager = risk_manager
        self.notifier = notifier or NullNotifier()
        self.config = config or EngineConfig()
        self.paper_ledger = paper_ledger
        self.attribution_ledger = attribution_ledger
        self._paused = False
        self._panic_stopped = False
        self._running = False
        self._stop_event = asyncio.Event()
        self._last_decision: str | None = None
        self._last_risk_reason: str | None = None
        self._last_error: str | None = None
        self._last_account: AccountSnapshot | None = None
        self._risk_notification_sent_at: dict[tuple[str, str, str], datetime] = {}
        self._risk_notification_suppressed: dict[tuple[str, str, str], int] = {}

    @property
    def paused(self) -> bool:
        return self._paused

    @property
    def panic_stopped(self) -> bool:
        return self._panic_stopped

    def status(self) -> EngineStatus:
        return EngineStatus(
            running=self._running,
            paused=self._paused,
            panic_stopped=self._panic_stopped,
            consecutive_failures=self.risk_manager.consecutive_failures,
            last_decision=self._last_decision,
            last_risk_reason=self._last_risk_reason,
            last_error=self._last_error,
        )

    def describe_strategies(self) -> tuple[dict, ...]:
        return tuple(strategy.describe() for strategy in self.strategies)

    async def status_text(self) -> str:
        account = self._last_account
        if account is None:
            try:
                account = await self.client.fetch_account_snapshot()
                self._last_account = account
            except Exception as exc:  # noqa: BLE001 - status should report failures to operators
                self._last_error = str(exc)
                return f"status=error error={exc}"
        return format_status(self.status(), account)

    async def performance_text(self) -> str:
        account = await self._account_for_report()
        return format_performance(
            account,
            rejected_by_cost_count=(
                account.rejected_by_cost_count + self.risk_manager.rejected_by_cost_count
            ),
            attribution_report=self._attribution_report(),
        )

    async def risk_text(self) -> str:
        account = await self._account_for_report()
        return format_risk(
            self.risk_manager.config,
            account,
            failures=self.risk_manager.consecutive_failures,
        )

    async def _account_for_report(self) -> AccountSnapshot:
        account = self._last_account
        if account is None:
            account = await self.client.fetch_account_snapshot()
            account = self._account_with_paper_gate(account)
            self._last_account = account
        return account

    async def pause(self) -> None:
        self._paused = True
        log.warning("trading_paused")
        await self.notifier.send(format_pause())

    async def resume(self) -> None:
        if self._panic_stopped:
            log.warning("trading_resume_rejected", reason="panic_stop_active")
            await self.notifier.send("🚫 <b>Cannot resume</b>\nPanic stop active; restart required")
            return
        self._paused = False
        log.info("trading_resumed")
        await self.notifier.send(format_resume())

    async def panic(self) -> None:
        self._paused = True
        self._panic_stopped = True
        self._stop_event.set()
        cancelled = await self.client.cancel_open_orders()
        log.critical("panic_stop", cancelled_orders=len(cancelled))
        await self.notifier.send(format_panic(len(cancelled)))

    async def stop(self) -> None:
        self._stop_event.set()

    async def run_forever(self) -> None:
        self._running = True
        log.info(
            "engine_started",
            strategies=len(self.strategies),
            poll_interval_seconds=self.config.poll_interval_seconds,
            td_mode=self.config.td_mode,
        )
        await self.notifier.send(format_lifecycle("started"))
        try:
            while not self._stop_event.is_set() and not self._panic_stopped:
                await self.tick()
                with suppress(TimeoutError):
                    await asyncio.wait_for(
                        self._stop_event.wait(),
                        timeout=self.config.poll_interval_seconds,
                    )
        finally:
            self._running = False
            log.info("engine_stopped")
            await self.notifier.send(format_lifecycle("stopped"))

    async def tick(self) -> None:
        if self._paused or self._panic_stopped:
            log.info("engine_tick_skipped", paused=self._paused, panic_stopped=self._panic_stopped)
            return
        try:
            account = await self.client.fetch_account_snapshot()
            account = self._account_with_paper_gate(account)
            self._last_account = account
            self._record_attribution_account(account)
            for strategy in self.strategies:
                candles = await self.client.fetch_candles(strategy.config.inst_id)
                self._record_attribution_candles(strategy.config.inst_id, candles)
                decision = strategy.decide(candles, account)
                self._record_paper_decision(strategy.name, decision, candles)
                self._record_attribution_decision(strategy.name, decision, candles)
                self._last_decision = f"{decision.inst_id}:{decision.action.value}"
                risk_decision = self.risk_manager.assess(decision, account)
                self._record_attribution_risk(decision, risk_decision, candles)
                self._last_risk_reason = risk_decision.reason
                decision_log = log.debug if decision.action is DecisionAction.HOLD else log.info
                decision_log(
                    "strategy_decision",
                    inst_id=decision.inst_id,
                    action=decision.action.value,
                    reason=decision.reason,
                    fast_ma=decision.inputs.get("fast_ma"),
                    slow_ma=decision.inputs.get("slow_ma"),
                    position_base=decision.inputs.get("position_base"),
                    risk_allowed=risk_decision.allowed,
                    risk_reason=risk_decision.reason,
                    expected_edge_bps=decision.inputs.get("expected_edge_bps"),
                    estimated_total_cost_bps=account.estimated_total_cost_bps,
                )

                if not risk_decision.allowed:
                    log.warning(
                        "risk_blocked",
                        inst_id=decision.inst_id,
                        action=decision.action.value,
                        reason=decision.reason,
                        risk_reason=risk_decision.reason,
                        expected_edge_bps=decision.inputs.get("expected_edge_bps"),
                        estimated_total_cost_bps=account.estimated_total_cost_bps,
                    )
                    if self._should_notify_risk_block(decision, risk_decision):
                        await self.notifier.send(
                            format_risk_blocked(
                                decision.inst_id,
                                decision.action.value,
                                risk_decision.reason,
                                decision.reason,
                            )
                        )
                    continue
                if decision.action is DecisionAction.HOLD:
                    self.risk_manager.record_success()
                    continue

                order_size = self.risk_manager.clamp_order_size(decision, account)
                order = OrderRequest(
                    inst_id=decision.inst_id,
                    side=decision.action,
                    size_btc=order_size,
                    td_mode=self.config.td_mode,
                    order_type=self.config.order_type,
                )
                submitted_at = utc_now()
                result = await self.client.place_market_order(order)
                self._record_attribution_order(order, result, candles)
                await self._record_attribution_fills(result, decision, candles, submitted_at)
                self.risk_manager.record_trade(inst_id=decision.inst_id)
                self.risk_manager.record_success()
                log.info(
                    "order_submitted",
                    inst_id=result.inst_id,
                    side=result.side.value,
                    size_btc=str(result.size_btc),
                    order_id=result.order_id,
                    reason=decision.reason,
                )
                await self.notifier.send(format_order_submitted(result, decision.reason))
        except Exception as exc:  # noqa: BLE001 - engine must convert all failures into risk state
            self.risk_manager.record_failure()
            self._last_error = str(exc)
            log.exception("engine_tick_failed", error=str(exc))
            await self.notifier.send(format_error(exc))
            if self.risk_manager.stopped_by_failures:
                self._paused = True
                await self.notifier.send(
                    "⏸️ <b>Trading paused</b>\nMax consecutive failures reached"
                )

    def _account_with_paper_gate(self, account: AccountSnapshot) -> AccountSnapshot:
        if self.paper_ledger is None:
            return account
        gate = self.paper_ledger.gate_decision()
        return replace(
            account,
            performance_gate_passed=gate.allowed,
            performance_gate_reason=gate.reason,
        )

    def _record_paper_decision(self, strategy_name: str, decision, candles) -> None:
        if self.paper_ledger is None or not candles:
            return
        self.paper_ledger.record_decision(decision, candles[-1], strategy_name=strategy_name)

    def _record_attribution_account(self, account: AccountSnapshot) -> None:
        if self.attribution_ledger is not None:
            self.attribution_ledger.record_account(account)

    def _record_attribution_candles(self, inst_id: str, candles) -> None:
        if self.attribution_ledger is None:
            return
        for candle in candles[-1:]:
            self.attribution_ledger.record_candle(inst_id, candle)

    def _record_attribution_decision(self, strategy_name: str, decision, candles) -> None:
        if self.attribution_ledger is None or not candles:
            return
        self.attribution_ledger.record_decision(strategy_name, decision, ts=candles[-1].ts)

    def _record_attribution_risk(self, decision, risk_decision, candles) -> None:
        if self.attribution_ledger is None or not candles:
            return
        self.attribution_ledger.record_risk(
            decision.inst_id,
            decision.action.value,
            risk_decision.allowed,
            risk_decision.reason,
            ts=candles[-1].ts,
        )

    def _record_attribution_order(self, order: OrderRequest, result: OrderResult, candles) -> None:
        if self.attribution_ledger is None or not candles:
            return
        self.attribution_ledger.record_order(order, result, ts=candles[-1].ts)

    async def _record_attribution_fills(
        self,
        result: OrderResult,
        decision: StrategyDecision,
        candles,
        submitted_at: datetime,
    ) -> None:
        if self.attribution_ledger is None or not candles:
            return
        fetch_order_fills = getattr(self.client, "fetch_order_fills", None)
        if fetch_order_fills is None:
            return
        decision_price = (
            _decimal_or_none(decision.inputs.get("decision_price")) or candles[-1].close
        )
        fills = await fetch_order_fills(
            result,
            decision_price=decision_price,
            submitted_at=submitted_at,
        )
        for fill in fills:
            self.attribution_ledger.record_fill(
                inst_id=fill.inst_id,
                side=fill.side,
                size_base=fill.size_base,
                price=fill.price,
                fee=fill.fee,
                fee_ccy=fill.fee_ccy,
                ts=fill.ts,
                decision_price=fill.decision_price or decision_price,
                order_id=fill.order_id or result.order_id,
                trade_id=fill.trade_id,
                client_order_id=fill.client_order_id or result.client_order_id or "",
                latency_ms=fill.latency_ms,
                raw=fill.raw,
            )

    def _attribution_report(self) -> dict | None:
        if self.attribution_ledger is None:
            return None
        return self.attribution_ledger.report().to_dict()

    def _should_notify_risk_block(
        self,
        decision: StrategyDecision,
        risk_decision: RiskDecision,
    ) -> bool:
        key = (decision.inst_id, decision.action.value, risk_decision.reason)
        if risk_decision.reason == "cooldown_active":
            self._risk_notification_suppressed[key] = (
                self._risk_notification_suppressed.get(key, 0) + 1
            )
            return False
        now = utc_now()
        last_sent_at = self._risk_notification_sent_at.get(key)
        if last_sent_at is not None and now - last_sent_at < timedelta(minutes=5):
            self._risk_notification_suppressed[key] = (
                self._risk_notification_suppressed.get(key, 0) + 1
            )
            return False
        self._risk_notification_sent_at[key] = now
        return True


def _decimal_or_none(value: object) -> Decimal | None:
    if value in {None, "", "None"}:
        return None
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None


def _normalise_strategies(
    strategy: TradingStrategy | Sequence[TradingStrategy],
) -> tuple[TradingStrategy, ...]:
    if hasattr(strategy, "decide"):
        return (strategy,)  # type: ignore[return-value]
    strategies = tuple(strategy)
    if not strategies:
        raise ValueError("at least one strategy is required")
    return strategies
