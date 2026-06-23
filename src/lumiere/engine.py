from __future__ import annotations

import asyncio
from collections.abc import Sequence
from contextlib import suppress
from dataclasses import dataclass
from typing import Protocol

import structlog

from lumiere.models import AccountSnapshot, DecisionAction, OrderRequest, OrderResult
from lumiere.risk import RiskManager
from lumiere.strategy import MovingAverageCrossoverStrategy

log = structlog.get_logger(__name__)


class TradingClient(Protocol):
    async def fetch_candles(self, inst_id: str | None = None): ...

    async def fetch_account_snapshot(self) -> AccountSnapshot: ...

    async def place_market_order(self, request: OrderRequest) -> OrderResult: ...

    async def cancel_open_orders(self) -> list[dict]: ...


class Notifier(Protocol):
    async def send(self, text: str) -> None: ...


class NullNotifier:
    async def send(self, text: str) -> None:
        _ = text


@dataclass(frozen=True, slots=True)
class EngineConfig:
    poll_interval_seconds: float = 30.0
    td_mode: str = "cash"


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
        strategy: MovingAverageCrossoverStrategy | Sequence[MovingAverageCrossoverStrategy],
        risk_manager: RiskManager,
        notifier: Notifier | None = None,
        config: EngineConfig | None = None,
    ) -> None:
        self.client = client
        self.strategies = _normalise_strategies(strategy)
        self.strategy = self.strategies[0]
        self.risk_manager = risk_manager
        self.notifier = notifier or NullNotifier()
        self.config = config or EngineConfig()
        self._paused = False
        self._panic_stopped = False
        self._running = False
        self._stop_event = asyncio.Event()
        self._last_decision: str | None = None
        self._last_risk_reason: str | None = None
        self._last_error: str | None = None
        self._last_account: AccountSnapshot | None = None

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

    def describe_strategies(self) -> tuple[dict[str, str | int], ...]:
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
        status = self.status()
        return (
            f"running={status.running} paused={status.paused} panic={status.panic_stopped} "
            f"equity_usdt={account.equity_usdt} available_usdt={account.available_usdt} "
            f"positions={_positions_text(account)} failures={status.consecutive_failures} "
            f"last_decision={status.last_decision} last_risk={status.last_risk_reason}"
        )

    async def pause(self) -> None:
        self._paused = True
        await self.notifier.send("Trading paused")

    async def resume(self) -> None:
        if self._panic_stopped:
            await self.notifier.send("Cannot resume after panic stop; restart the bot")
            return
        self._paused = False
        await self.notifier.send("Trading resumed")

    async def panic(self) -> None:
        self._paused = True
        self._panic_stopped = True
        self._stop_event.set()
        cancelled = await self.client.cancel_open_orders()
        await self.notifier.send(f"PANIC stop active. Cancelled open orders: {len(cancelled)}")

    async def stop(self) -> None:
        self._stop_event.set()

    async def run_forever(self) -> None:
        self._running = True
        await self.notifier.send("Trading engine started")
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
            await self.notifier.send("Trading engine stopped")

    async def tick(self) -> None:
        if self._paused or self._panic_stopped:
            log.info("engine_tick_skipped", paused=self._paused, panic_stopped=self._panic_stopped)
            return
        try:
            account = await self.client.fetch_account_snapshot()
            self._last_account = account
            for strategy in self.strategies:
                candles = await self.client.fetch_candles(strategy.config.inst_id)
                decision = strategy.decide(candles, account)
                self._last_decision = f"{decision.inst_id}:{decision.action.value}"
                risk_decision = self.risk_manager.assess(decision, account)
                self._last_risk_reason = risk_decision.reason
                log.info(
                    "strategy_decision",
                    inst_id=decision.inst_id,
                    action=decision.action.value,
                    reason=decision.reason,
                    inputs=decision.inputs,
                    risk_allowed=risk_decision.allowed,
                    risk_reason=risk_decision.reason,
                )

                if not risk_decision.allowed:
                    await self.notifier.send(
                        f"Risk blocked {decision.inst_id} {decision.action.value}: "
                        f"{risk_decision.reason} ({decision.reason})"
                    )
                    continue
                if decision.action is DecisionAction.HOLD:
                    self.risk_manager.record_success()
                    continue

                order = OrderRequest(
                    inst_id=decision.inst_id,
                    side=decision.action,
                    size_btc=decision.size_btc,
                    td_mode=self.config.td_mode,
                )
                result = await self.client.place_market_order(order)
                self.risk_manager.record_trade(inst_id=decision.inst_id)
                self.risk_manager.record_success()
                await self.notifier.send(
                    f"Order submitted: {result.side.value} {result.size_btc} {result.inst_id} "
                    f"order_id={result.order_id} reason={decision.reason}"
                )
        except Exception as exc:  # noqa: BLE001 - engine must convert all failures into risk state
            self.risk_manager.record_failure()
            self._last_error = str(exc)
            log.exception("engine_tick_failed", error=str(exc))
            await self.notifier.send(f"Trading error: {exc}")
            if self.risk_manager.stopped_by_failures:
                self._paused = True
                await self.notifier.send("Trading paused: max consecutive failures reached")


def _normalise_strategies(
    strategy: MovingAverageCrossoverStrategy | Sequence[MovingAverageCrossoverStrategy],
) -> tuple[MovingAverageCrossoverStrategy, ...]:
    if isinstance(strategy, MovingAverageCrossoverStrategy):
        return (strategy,)
    strategies = tuple(strategy)
    if not strategies:
        raise ValueError("at least one strategy is required")
    return strategies


def _positions_text(account: AccountSnapshot) -> str:
    if not account.positions:
        return "none"
    return ",".join(f"{position.inst_id}:{position.size_btc}" for position in account.positions)
