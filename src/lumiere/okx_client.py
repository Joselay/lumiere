from __future__ import annotations

import asyncio
import re
from collections.abc import Callable, Iterable
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any
from uuid import uuid4

from lumiere.config import Settings
from lumiere.ledger import TradeFill, realized_pnl_for_period, trade_fill_from_okx_row
from lumiere.models import (
    AccountSnapshot,
    MarketCandle,
    OrderRequest,
    OrderResult,
    Position,
)
from lumiere.risk import RiskManager


class OKXAPIError(RuntimeError):
    pass


class OKXDemoClient:
    """Async facade over the synchronous `python-okx` REST clients."""

    def __init__(self, settings: Settings, risk_manager: RiskManager) -> None:
        if settings.okx_flag != "1":
            raise ValueError(
                "OKX demo guard failed: refusing to create client when OKX_FLAG != '1'"
            )
        self.settings = settings
        self.risk_manager = risk_manager

        # Import here so pure strategy/risk tests do not need the SDK installed.
        from okx import Account, MarketData, Trade  # type: ignore[import-not-found]

        common = {
            "api_key": settings.okx_api_key,
            "api_secret_key": settings.okx_api_secret,
            "passphrase": settings.okx_passphrase,
            "flag": settings.okx_flag,
            "debug": False,
        }
        self._account = Account.AccountAPI(**common)
        self._market = MarketData.MarketAPI(flag=settings.okx_flag, debug=False)
        self._trade = Trade.TradeAPI(**common)

    async def fetch_candles(self, inst_id: str | None = None) -> list[MarketCandle]:
        inst_id = inst_id or self.settings.enabled_inst_ids[0]
        response = await asyncio.to_thread(
            self._market.get_candlesticks,
            instId=inst_id,
            bar=self.settings.engine_candle_bar,
            limit=str(self.settings.engine_candle_limit),
        )
        data = _require_ok_response(response)
        candles = [_parse_candle(row) for row in data]
        return sorted(candles, key=lambda candle: candle.ts)

    async def fetch_account_snapshot(self) -> AccountSnapshot:
        inst_ids = self.settings.enabled_inst_ids
        responses = await asyncio.gather(
            asyncio.to_thread(self._account.get_account_balance),
            *(
                asyncio.to_thread(self._account.get_positions, instId=inst_id)
                for inst_id in inst_ids
            ),
        )
        balance_response = responses[0]
        position_responses = responses[1:]
        balance_data = _require_ok_response(balance_response)
        equity, available, spot_positions = _parse_account_balances(balance_data, inst_ids)

        positions_by_inst_id = {position.inst_id: position for position in spot_positions}
        for inst_id, response in zip(inst_ids, position_responses, strict=True):
            derivatives_position = _parse_position(inst_id, _require_ok_response(response))
            if derivatives_position is not None:
                positions_by_inst_id[inst_id] = derivatives_position

        positions = tuple(
            positions_by_inst_id[inst_id] for inst_id in inst_ids if inst_id in positions_by_inst_id
        )
        daily_realized_pnl, daily_trade_count = await self.fetch_daily_realized_pnl()
        spread_bps = None
        estimated_slippage_bps = None
        estimated_total_cost_bps = None
        if self.risk_manager.config.max_spread_bps is not None:
            spread_bps, estimated_slippage_bps = await self.fetch_execution_quality_bps()
            estimated_total_cost_bps = spread_bps + estimated_slippage_bps + Decimal("10")
        return AccountSnapshot(
            equity_usdt=equity,
            available_usdt=available,
            positions=positions,
            daily_realized_pnl_usdt=daily_realized_pnl,
            daily_trade_count=daily_trade_count,
            spread_bps=spread_bps,
            estimated_slippage_bps=estimated_slippage_bps,
            estimated_total_cost_bps=estimated_total_cost_bps,
        )

    async def fetch_daily_realized_pnl(
        self,
        now: datetime | None = None,
    ) -> tuple[Decimal, int]:
        """Derive today's realized PnL from OKX fills/fill history instead of placeholders."""

        now = now or datetime.now(tz=UTC)
        period_start = datetime(now.year, now.month, now.day, tzinfo=UTC)
        rows = await self._fetch_fill_rows(period_start=period_start, period_end=now)
        return (
            _daily_realized_pnl_from_fill_rows(rows, period_start),
            _daily_trade_count(rows, period_start),
        )

    async def fetch_max_spread_bps(self) -> Decimal:
        spread, _ = await self.fetch_execution_quality_bps()
        return spread

    async def fetch_execution_quality_bps(self) -> tuple[Decimal, Decimal]:
        spreads: list[Decimal] = []
        slippages: list[Decimal] = []
        for inst_id in self.settings.enabled_inst_ids:
            response = await asyncio.to_thread(
                self._market.get_orderbook,
                instId=inst_id,
                sz="50",
            )
            data = _require_ok_response(response)
            trade_size = self.settings._trade_size_for(inst_id)  # noqa: SLF001 - config-owned sizing
            spreads.append(_spread_bps_from_orderbook(data))
            buy_slippage = _depth_slippage_bps_from_orderbook(data, trade_size, side="buy")
            sell_slippage = _depth_slippage_bps_from_orderbook(data, trade_size, side="sell")
            slippages.append(max(buy_slippage, sell_slippage))
        return max(spreads, default=Decimal("0")), max(slippages, default=Decimal("0"))

    async def fetch_order_fills(
        self,
        result: OrderResult,
        *,
        decision_price: Decimal | None = None,
        submitted_at: datetime | None = None,
    ) -> tuple[TradeFill, ...]:
        """Fetch and reconcile OKX fills for one submitted order."""

        now = datetime.now(tz=UTC)
        period_start = submitted_at or datetime(now.year, now.month, now.day, tzinfo=UTC)
        rows = await self._fetch_fill_rows(
            period_start=period_start - timedelta(minutes=5),
            period_end=now,
            order_id=result.order_id,
        )
        matched_rows = [row for row in rows if _row_matches_order(row, result)]
        fills: list[TradeFill] = []
        for row in matched_rows:
            fill = trade_fill_from_okx_row(row)
            latency_ms = None
            if submitted_at is not None:
                latency_ms = max(int((fill.ts - submitted_at).total_seconds() * 1000), 0)
            fills.append(
                replace(
                    fill,
                    decision_price=decision_price,
                    latency_ms=latency_ms,
                    client_order_id=str(row.get("clOrdId") or result.client_order_id or ""),
                    raw=dict(row),
                )
            )
        return tuple(fills)

    async def _fetch_fill_rows(
        self,
        *,
        period_start: datetime,
        period_end: datetime,
        order_id: str | None = None,
    ) -> list[dict[str, Any]]:
        begin = _to_okx_millis(period_start)
        end = _to_okx_millis(period_end)
        rows: list[dict[str, Any]] = []
        for inst_id in self.settings.enabled_inst_ids:
            rows.extend(
                await self._fetch_paginated_fill_rows(
                    self._trade.get_fills,
                    {
                        "instType": "SPOT",
                        "instId": inst_id,
                        "begin": begin,
                        "end": end,
                        "ordId": order_id or "",
                    },
                )
            )
            if hasattr(self._trade, "get_fills_history"):
                rows.extend(
                    await self._fetch_paginated_fill_rows(
                        lambda **kwargs: self._trade.get_fills_history("SPOT", **kwargs),
                        {
                            "instId": inst_id,
                            "ordId": order_id or "",
                        },
                    )
                )
        return _dedupe_fill_rows(rows)

    async def _fetch_paginated_fill_rows(
        self,
        fetch_page: Callable[..., dict[str, Any]],
        base_kwargs: dict[str, str],
        *,
        max_pages: int = 20,
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        before = ""
        seen_cursors: set[str] = set()
        for _ in range(max_pages):
            kwargs = {**base_kwargs, "limit": "100"}
            if before:
                kwargs["before"] = before
            response = await asyncio.to_thread(fetch_page, **kwargs)
            page = _require_ok_response(response)
            if not page:
                break
            rows.extend(page)
            if len(page) < 100:
                break
            cursor = _fill_cursor(page[-1])
            if not cursor or cursor in seen_cursors:
                break
            seen_cursors.add(cursor)
            before = cursor
        return rows

    async def place_market_order(self, request: OrderRequest) -> OrderResult:
        self.risk_manager.validate_order(request)
        client_order_id = f"lum{uuid4().hex[:24]}"
        order_kwargs = {
            "instId": request.inst_id,
            "tdMode": request.td_mode,
            "side": request.side.value,
            "ordType": request.order_type,
            "sz": str(request.size_btc),
            "clOrdId": client_order_id,
            "tag": self.settings.okx_order_tag,
        }
        if request.limit_price is not None:
            order_kwargs["px"] = str(request.limit_price)
        if request.side.value == "buy" and request.order_type == "market":
            # OKX interprets spot market-buy sz as quote currency by default. Lumiere's
            # strategy/risk sizes are base units, so make that explicit for buys.
            order_kwargs["tgtCcy"] = "base_ccy"
        response = await asyncio.to_thread(self._trade.place_order, **order_kwargs)
        data = _require_ok_response(response)
        if not data:
            raise OKXAPIError(f"OKX returned no order data: {response!r}")
        first = data[0]
        s_code = str(first.get("sCode", "0"))
        if s_code != "0":
            raise OKXAPIError(f"OKX order rejected: {first!r}")
        return OrderResult(
            order_id=str(first.get("ordId", "")),
            client_order_id=str(first.get("clOrdId") or client_order_id),
            inst_id=request.inst_id,
            side=request.side,
            size_btc=request.size_btc,
            status=str(first.get("sMsg") or "submitted"),
            raw=first,
        )

    async def cancel_open_orders(self) -> list[dict[str, Any]]:
        cancelled: list[dict[str, Any]] = []
        for inst_id in self.settings.enabled_inst_ids:
            response = await asyncio.to_thread(
                self._trade.get_order_list,
                instId=inst_id,
            )
            open_orders = _require_ok_response(response)
            for order in open_orders:
                cancel_response = await asyncio.to_thread(
                    self._trade.cancel_order,
                    instId=inst_id,
                    ordId=str(order.get("ordId", "")),
                    clOrdId=str(order.get("clOrdId", "")),
                )
                cancelled.extend(_require_ok_response(cancel_response))
        return cancelled


def _spread_bps_from_orderbook(data: list[Any]) -> Decimal:
    if not data:
        raise OKXAPIError("OKX returned no orderbook data")
    orderbook = data[0]
    bids = orderbook.get("bids") or []
    asks = orderbook.get("asks") or []
    if not bids or not asks:
        raise OKXAPIError(f"OKX orderbook missing best bid/ask: {_safe_response_repr(orderbook)}")
    best_bid = Decimal(str(bids[0][0]))
    best_ask = Decimal(str(asks[0][0]))
    midpoint = (best_bid + best_ask) / Decimal("2")
    if midpoint <= 0:
        raise OKXAPIError(f"OKX orderbook midpoint is invalid: {_safe_response_repr(orderbook)}")
    return (best_ask - best_bid) / midpoint * Decimal("10000")


def _depth_slippage_bps_from_orderbook(
    data: list[Any],
    size_base: Decimal,
    *,
    side: str,
) -> Decimal:
    if not data or size_base <= 0:
        return Decimal("0")
    orderbook = data[0]
    bids = orderbook.get("bids") or []
    asks = orderbook.get("asks") or []
    if not bids or not asks:
        raise OKXAPIError(f"OKX orderbook missing depth: {_safe_response_repr(orderbook)}")
    best_bid = Decimal(str(bids[0][0]))
    best_ask = Decimal(str(asks[0][0]))
    midpoint = (best_bid + best_ask) / Decimal("2")
    if midpoint <= 0:
        raise OKXAPIError(f"OKX orderbook midpoint is invalid: {_safe_response_repr(orderbook)}")
    levels = asks if side == "buy" else bids
    remaining = size_base
    notional = Decimal("0")
    filled = Decimal("0")
    for level in levels:
        price = Decimal(str(level[0]))
        available = Decimal(str(level[1]))
        take = min(remaining, available)
        notional += take * price
        filled += take
        remaining -= take
        if remaining <= 0:
            break
    if filled <= 0:
        return Decimal("Infinity")
    vwap = notional / filled
    if side == "buy":
        return max((vwap - midpoint) / midpoint * Decimal("10000"), Decimal("0"))
    return max((midpoint - vwap) / midpoint * Decimal("10000"), Decimal("0"))


def _daily_realized_pnl_from_fill_rows(
    rows: list[dict[str, Any]],
    period_start: datetime,
) -> Decimal:
    period_rows = [row for row in rows if _fill_ts(row) >= period_start]
    exchange_pnl = Decimal("0")
    exchange_fee_cost = Decimal("0")
    has_exchange_pnl = False
    for row in period_rows:
        raw_pnl = row.get("fillPnl")
        if raw_pnl not in {None, ""}:
            pnl_value = Decimal(str(raw_pnl or "0"))
            has_exchange_pnl = has_exchange_pnl or pnl_value != 0
            exchange_pnl += pnl_value
            exchange_fee_cost += trade_fill_from_okx_row(row).fee_cost_usdt()
    if has_exchange_pnl:
        return exchange_pnl - exchange_fee_cost

    fills = [trade_fill_from_okx_row(row) for row in rows]
    return realized_pnl_for_period(fills, period_start=period_start)


def _daily_trade_count(rows: list[dict[str, Any]], period_start: datetime) -> int:
    return sum(1 for row in rows if _fill_ts(row) >= period_start)


def _row_matches_order(row: dict[str, Any], result: OrderResult) -> bool:
    row_order_id = str(row.get("ordId") or "")
    row_client_id = str(row.get("clOrdId") or "")
    return bool(
        (result.order_id and row_order_id == result.order_id)
        or (result.client_order_id and row_client_id == result.client_order_id)
    )


def _fill_cursor(row: dict[str, Any]) -> str:
    return str(
        row.get("tradeId")
        or row.get("execId")
        or row.get("billId")
        or row.get("ts")
        or ""
    )


def _dedupe_fill_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str, str, str]] = set()
    for row in rows:
        key = (
            str(row.get("tradeId") or row.get("execId") or ""),
            str(row.get("ordId") or ""),
            str(row.get("instId") or ""),
            str(row.get("ts") or ""),
            str(row.get("side") or ""),
            str(row.get("fillSz") or row.get("sz") or ""),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return deduped


def _fill_ts(row: dict[str, Any]) -> datetime:
    return datetime.fromtimestamp(int(str(row.get("ts") or "0")) / 1000, tz=UTC)


def _to_okx_millis(value: datetime) -> str:
    return str(int(value.timestamp() * 1000))


def _require_ok_response(response: dict[str, Any]) -> list[Any]:
    code = str(response.get("code", ""))
    if code != "0":
        raise OKXAPIError(f"OKX API error: {_safe_response_repr(response)}")
    data = response.get("data", [])
    if not isinstance(data, list):
        raise OKXAPIError(f"OKX response data is not a list: {_safe_response_repr(response)}")
    return data


_SENSITIVE_RESPONSE_KEYS = {
    "apiKey",
    "api_key",
    "apiSecret",
    "api_secret",
    "api_secret_key",
    "passphrase",
}
_UUID_RE = re.compile(
    r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"
)
_IPV4_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")


def _safe_response_repr(value: Any) -> str:
    return repr(_redact_sensitive(value))


def _redact_sensitive(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: "<redacted>" if key in _SENSITIVE_RESPONSE_KEYS else _redact_sensitive(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact_sensitive(item) for item in value]
    if isinstance(value, str):
        return _IPV4_RE.sub("<redacted-ip>", _UUID_RE.sub("<redacted-id>", value))
    return value


def _parse_candle(row: list[str]) -> MarketCandle:
    # OKX candle: [ts, open, high, low, close, volume, volCcy, volCcyQuote, confirm]
    return MarketCandle(
        ts=datetime.fromtimestamp(int(row[0]) / 1000, tz=UTC),
        open=Decimal(str(row[1])),
        high=Decimal(str(row[2])),
        low=Decimal(str(row[3])),
        close=Decimal(str(row[4])),
        volume=Decimal(str(row[5])) if len(row) > 5 else Decimal("0"),
    )


def _parse_account_balances(
    data: list[dict[str, Any]],
    inst_ids: Iterable[str],
) -> tuple[Decimal, Decimal, tuple[Position, ...]]:
    if not data:
        return Decimal("0"), Decimal("0"), ()

    inst_ids = tuple(inst_ids)
    account = data[0]
    equity = Decimal(str(account.get("totalEq") or "0"))
    available_usdt = Decimal("0")
    base_ccy_by_inst_id = {inst_id: inst_id.split("-")[0] for inst_id in inst_ids}
    size_by_base_ccy = dict.fromkeys(base_ccy_by_inst_id.values(), Decimal("0"))

    for detail in account.get("details", []):
        ccy = detail.get("ccy")
        if ccy == "USDT":
            available_usdt = Decimal(str(detail.get("availBal") or detail.get("cashBal") or "0"))
        if ccy in size_by_base_ccy:
            size_by_base_ccy[ccy] = Decimal(
                str(detail.get("cashBal") or detail.get("availBal") or "0")
            )

    positions = tuple(
        Position(inst_id=inst_id, size_btc=size_by_base_ccy[base_ccy])
        for inst_id, base_ccy in base_ccy_by_inst_id.items()
        if size_by_base_ccy[base_ccy] != 0
    )
    return equity, available_usdt, positions


def _parse_position(inst_id: str, data: list[dict[str, Any]]) -> Position | None:
    for row in data:
        if row.get("instId") != inst_id:
            continue
        size = Decimal(str(row.get("pos") or row.get("availPos") or "0"))
        if size == 0:
            return None
        return Position(
            inst_id=inst_id,
            size_btc=size,
            avg_px=Decimal(str(row.get("avgPx") or "0")),
            unrealized_pnl_usdt=Decimal(str(row.get("upl") or "0")),
        )
    return None


def _parse_btc_position(inst_id: str, data: list[dict[str, Any]]) -> Position | None:
    return _parse_position(inst_id, data)
