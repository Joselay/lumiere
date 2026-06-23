from __future__ import annotations

import asyncio
import re
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any
from uuid import uuid4

from lumiere.config import Settings
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

    async def fetch_candles(self) -> list[MarketCandle]:
        response = await asyncio.to_thread(
            self._market.get_candlesticks,
            instId=self.settings.okx_inst_id,
            bar=self.settings.engine_candle_bar,
            limit=str(self.settings.engine_candle_limit),
        )
        data = _require_ok_response(response)
        candles = [_parse_candle(row) for row in data]
        return sorted(candles, key=lambda candle: candle.ts)

    async def fetch_account_snapshot(self) -> AccountSnapshot:
        balance_response, positions_response = await asyncio.gather(
            asyncio.to_thread(self._account.get_account_balance),
            asyncio.to_thread(self._account.get_positions, instId=self.settings.okx_inst_id),
        )
        balance_data = _require_ok_response(balance_response)
        positions_data = _require_ok_response(positions_response)
        equity, available, spot_btc_position = _parse_account_balances(
            balance_data,
            self.settings.okx_inst_id,
        )
        derivatives_position = _parse_btc_position(self.settings.okx_inst_id, positions_data)
        return AccountSnapshot(
            equity_usdt=equity,
            available_usdt=available,
            btc_position=derivatives_position or spot_btc_position,
            daily_realized_pnl_usdt=Decimal("0"),
        )

    async def place_market_order(self, request: OrderRequest) -> OrderResult:
        self.risk_manager.validate_order(request)
        client_order_id = f"lum{uuid4().hex[:24]}"
        response = await asyncio.to_thread(
            self._trade.place_order,
            instId=request.inst_id,
            tdMode=request.td_mode,
            side=request.side.value,
            ordType=request.order_type,
            sz=str(request.size_btc),
            clOrdId=client_order_id,
            tag=self.settings.okx_order_tag,
        )
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
        response = await asyncio.to_thread(
            self._trade.get_order_list,
            instId=self.settings.okx_inst_id,
        )
        open_orders = _require_ok_response(response)
        cancelled: list[dict[str, Any]] = []
        for order in open_orders:
            cancel_response = await asyncio.to_thread(
                self._trade.cancel_order,
                instId=self.settings.okx_inst_id,
                ordId=str(order.get("ordId", "")),
                clOrdId=str(order.get("clOrdId", "")),
            )
            cancelled.extend(_require_ok_response(cancel_response))
        return cancelled


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
    inst_id: str,
) -> tuple[Decimal, Decimal, Position | None]:
    if not data:
        return Decimal("0"), Decimal("0"), None

    account = data[0]
    equity = Decimal(str(account.get("totalEq") or "0"))
    available_usdt = Decimal("0")
    btc_size = Decimal("0")
    base_ccy = inst_id.split("-")[0]

    for detail in account.get("details", []):
        ccy = detail.get("ccy")
        if ccy == "USDT":
            available_usdt = Decimal(str(detail.get("availBal") or detail.get("cashBal") or "0"))
        if ccy == base_ccy:
            btc_size = Decimal(str(detail.get("cashBal") or detail.get("availBal") or "0"))

    position = None
    if btc_size != 0:
        position = Position(inst_id=inst_id, size_btc=btc_size)
    return equity, available_usdt, position


def _parse_btc_position(inst_id: str, data: list[dict[str, Any]]) -> Position | None:
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
