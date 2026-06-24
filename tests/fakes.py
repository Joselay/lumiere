from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal

from lumiere.ledger import BASIS_POINTS, TradeFill
from lumiere.models import (
    AccountSnapshot,
    DecisionAction,
    MarketCandle,
    OrderRequest,
    OrderResult,
    Position,
)


class FakeExchangeError(RuntimeError):
    pass


class DeterministicFakeExchange:
    """Deterministic in-process exchange adapter for money-path tests.

    It implements the TradingEngine client protocol plus fill reconciliation hooks. The fake keeps
    cash, average-cost positions, order books, accepted/rejected orders, partial fills, and injected
    API failures fully in memory so tests can exercise live paths without OKX credentials.
    """

    def __init__(
        self,
        candles_by_inst: Mapping[str, Sequence[MarketCandle]],
        *,
        orderbooks_by_inst: Mapping[str, dict[str, list[list[str]]]] | None = None,
        starting_cash_usdt: Decimal = Decimal("1000"),
        taker_fee_bps: Decimal = Decimal("10"),
        reject_order_numbers: set[int] | None = None,
        fill_splits_by_order_number: Mapping[int, Sequence[Decimal]] | None = None,
        fail_on: set[str] | None = None,
    ) -> None:
        self.candles_by_inst = {
            inst_id: list(candles) for inst_id, candles in candles_by_inst.items()
        }
        self.orderbooks_by_inst = dict(orderbooks_by_inst or {})
        self.cash_usdt = starting_cash_usdt
        self.taker_fee_bps = taker_fee_bps
        self.reject_order_numbers = set(reject_order_numbers or set())
        self.fill_splits_by_order_number = {
            order_number: tuple(splits)
            for order_number, splits in (fill_splits_by_order_number or {}).items()
        }
        self.fail_on = set(fail_on or set())
        self.orders: list[OrderRequest] = []
        self.order_results: list[OrderResult] = []
        self.cancelled = False
        self._position_size_by_inst: dict[str, Decimal] = {}
        self._avg_cost_by_inst: dict[str, Decimal] = {}
        self._fills_by_order_id: dict[str, list[TradeFill]] = {}

    async def fetch_candles(self, inst_id: str | None = None) -> list[MarketCandle]:
        self._maybe_fail("candles")
        resolved = inst_id or next(iter(self.candles_by_inst))
        return list(self.candles_by_inst[resolved])

    async def fetch_account_snapshot(self) -> AccountSnapshot:
        self._maybe_fail("account")
        positions = []
        for inst_id, size in sorted(self._position_size_by_inst.items()):
            if size <= 0:
                continue
            mark = self._last_price(inst_id)
            avg_cost = self._avg_cost_by_inst.get(inst_id, Decimal("0"))
            positions.append(
                Position(
                    inst_id=inst_id,
                    size_btc=size,
                    avg_px=avg_cost,
                    unrealized_pnl_usdt=(mark - avg_cost) * size,
                )
            )
        mark_value = sum(
            size * self._last_price(inst_id)
            for inst_id, size in self._position_size_by_inst.items()
            if size > 0
        )
        spread, depth_slippage = self._worst_execution_quality()
        return AccountSnapshot(
            equity_usdt=self.cash_usdt + mark_value,
            available_usdt=self.cash_usdt,
            positions=tuple(positions),
            daily_trade_count=sum(len(fills) for fills in self._fills_by_order_id.values()),
            spread_bps=spread,
            estimated_slippage_bps=depth_slippage,
            estimated_total_cost_bps=spread / Decimal("2") + depth_slippage + self.taker_fee_bps,
        )

    async def place_market_order(self, request: OrderRequest) -> OrderResult:
        self._maybe_fail("orders")
        self.orders.append(request)
        order_number = len(self.orders)
        order_id = f"ord-{order_number}"
        client_order_id = f"client-{order_number}"
        if order_number in self.reject_order_numbers:
            result = OrderResult(
                order_id=order_id,
                client_order_id=client_order_id,
                inst_id=request.inst_id,
                side=request.side,
                size_btc=request.size_btc,
                status="rejected",
                raw={"sCode": "51000", "sMsg": "fake reject"},
            )
            self.order_results.append(result)
            return result
        current_position = self._position_size_by_inst.get(request.inst_id, Decimal("0"))
        if request.side is DecisionAction.SELL and current_position <= 0:
            result = OrderResult(
                order_id=order_id,
                client_order_id=client_order_id,
                inst_id=request.inst_id,
                side=request.side,
                size_btc=request.size_btc,
                status="rejected_no_inventory",
                raw={"sCode": "51008", "sMsg": "insufficient fake inventory"},
            )
            self.order_results.append(result)
            return result

        splits = self.fill_splits_by_order_number.get(order_number, (Decimal("1"),))
        fills: list[TradeFill] = []
        filled_size = Decimal("0")
        for index, split in enumerate(splits, start=1):
            if split <= 0:
                continue
            size = request.size_btc * split
            price = self._execution_price(request.inst_id, request.side, size)
            fee = size * price * self.taker_fee_bps / BASIS_POINTS
            trade_id = f"trade-{order_number}-{index}"
            fill = TradeFill(
                inst_id=request.inst_id,
                side=request.side,
                size_base=size,
                price=price,
                fee=fee,
                fee_ccy="USDT",
                ts=self.candles_by_inst[request.inst_id][-1].ts + timedelta(milliseconds=index),
                order_id=order_id,
                trade_id=trade_id,
                client_order_id=client_order_id,
                raw={
                    "instId": request.inst_id,
                    "side": request.side.value,
                    "fillSz": str(size),
                    "fillPx": str(price),
                    "fee": str(fee),
                    "feeCcy": "USDT",
                    "ordId": order_id,
                    "tradeId": trade_id,
                },
            )
            fills.append(fill)
            filled_size += size
            self._apply_fill(fill)
        self._fills_by_order_id[order_id] = fills
        status = "filled" if filled_size >= request.size_btc else "partially_filled"
        result = OrderResult(
            order_id=order_id,
            client_order_id=client_order_id,
            inst_id=request.inst_id,
            side=request.side,
            size_btc=filled_size,
            status=status,
            raw={
                "ordId": order_id,
                "clOrdId": client_order_id,
                "fills": [fill.raw for fill in fills],
            },
        )
        self.order_results.append(result)
        return result

    async def fetch_order_fills(
        self,
        result: OrderResult,
        *,
        decision_price: Decimal | None = None,
        submitted_at: datetime | None = None,
    ) -> tuple[TradeFill, ...]:
        self._maybe_fail("fills")
        reconciled: list[TradeFill] = []
        base_ts = submitted_at or datetime.now(tz=UTC)
        for index, fill in enumerate(self._fills_by_order_id.get(result.order_id, []), start=1):
            ts = base_ts + timedelta(milliseconds=index)
            raw = dict(fill.raw)
            raw["ts"] = str(int(ts.timestamp() * 1000))
            reconciled.append(
                replace(
                    fill,
                    ts=ts,
                    decision_price=decision_price,
                    latency_ms=index,
                    raw=raw,
                )
            )
        return tuple(reconciled)

    async def cancel_open_orders(self) -> list[dict[str, str]]:
        self._maybe_fail("cancel")
        self.cancelled = True
        return [
            {"ordId": result.order_id}
            for result in self.order_results
            if result.status != "filled"
        ]

    def _maybe_fail(self, operation: str) -> None:
        if operation in self.fail_on:
            raise FakeExchangeError(f"fake {operation} failure")

    def _last_price(self, inst_id: str) -> Decimal:
        return self.candles_by_inst[inst_id][-1].close

    def _worst_execution_quality(self) -> tuple[Decimal, Decimal]:
        spreads: list[Decimal] = []
        slippages: list[Decimal] = []
        for inst_id in self.candles_by_inst:
            orderbook = self.orderbooks_by_inst.get(inst_id)
            if orderbook is None:
                spreads.append(Decimal("0"))
                slippages.append(Decimal("0"))
                continue
            spreads.append(_spread_bps(orderbook))
            slippages.append(_depth_slippage_bps(orderbook, Decimal("1"), side="buy"))
            slippages.append(_depth_slippage_bps(orderbook, Decimal("1"), side="sell"))
        return max(spreads, default=Decimal("0")), max(slippages, default=Decimal("0"))

    def _execution_price(self, inst_id: str, side: DecisionAction, size: Decimal) -> Decimal:
        orderbook = self.orderbooks_by_inst.get(inst_id)
        if orderbook is None:
            return self._last_price(inst_id)
        levels = orderbook["asks"] if side is DecisionAction.BUY else orderbook["bids"]
        return _weighted_price(levels, size)

    def _apply_fill(self, fill: TradeFill) -> None:
        size = self._position_size_by_inst.get(fill.inst_id, Decimal("0"))
        avg_cost = self._avg_cost_by_inst.get(fill.inst_id, Decimal("0"))
        if fill.side is DecisionAction.BUY:
            old_cost = size * avg_cost
            new_size = size + fill.size_base
            new_cost = old_cost + fill.notional_usdt + fill.fee_cost_usdt()
            self._position_size_by_inst[fill.inst_id] = new_size
            self._avg_cost_by_inst[fill.inst_id] = new_cost / new_size
            self.cash_usdt -= fill.notional_usdt + fill.fee_cost_usdt()
            return

        close_size = min(size, fill.size_base)
        remaining_size = size - close_size
        self.cash_usdt += close_size * fill.price - fill.fee_cost_usdt()
        self._position_size_by_inst[fill.inst_id] = remaining_size
        self._avg_cost_by_inst[fill.inst_id] = Decimal("0") if remaining_size <= 0 else avg_cost


def _spread_bps(orderbook: dict[str, list[list[str]]]) -> Decimal:
    best_bid = Decimal(str(orderbook["bids"][0][0]))
    best_ask = Decimal(str(orderbook["asks"][0][0]))
    midpoint = (best_bid + best_ask) / Decimal("2")
    return (best_ask - best_bid) / midpoint * BASIS_POINTS


def _depth_slippage_bps(
    orderbook: dict[str, list[list[str]]],
    size: Decimal,
    *,
    side: str,
) -> Decimal:
    best_bid = Decimal(str(orderbook["bids"][0][0]))
    best_ask = Decimal(str(orderbook["asks"][0][0]))
    midpoint = (best_bid + best_ask) / Decimal("2")
    levels = orderbook["asks"] if side == "buy" else orderbook["bids"]
    vwap = _weighted_price(levels, size)
    if side == "buy":
        return max((vwap - midpoint) / midpoint * BASIS_POINTS, Decimal("0"))
    return max((midpoint - vwap) / midpoint * BASIS_POINTS, Decimal("0"))


def _weighted_price(levels: Sequence[Sequence[str]], size: Decimal) -> Decimal:
    remaining = size
    notional = Decimal("0")
    filled = Decimal("0")
    last_price = Decimal(str(levels[-1][0]))
    for level in levels:
        price = Decimal(str(level[0]))
        available = Decimal(str(level[1]))
        take = min(remaining, available)
        notional += take * price
        filled += take
        remaining -= take
        last_price = price
        if remaining <= 0:
            break
    if remaining > 0:
        notional += remaining * last_price
        filled += remaining
    return notional / filled
