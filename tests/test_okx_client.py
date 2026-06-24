from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

import pytest

from lumiere.config import Settings
from lumiere.models import DecisionAction, OrderRequest
from lumiere.okx_client import (
    OKXDemoClient,
    _daily_realized_pnl_from_fill_rows,
    _depth_slippage_bps_from_orderbook,
    _parse_account_balances,
    _parse_btc_position,
    _spread_bps_from_orderbook,
)
from lumiere.risk import RiskConfig, RiskManager


def test_parse_account_balances_includes_spot_btc_and_eth_as_positions() -> None:
    equity, available, positions = _parse_account_balances(
        [
            {
                "totalEq": "1000",
                "details": [
                    {"ccy": "USDT", "availBal": "900"},
                    {"ccy": "BTC", "cashBal": "0.002"},
                    {"ccy": "ETH", "cashBal": "0.05"},
                ],
            }
        ],
        ("BTC-USDT", "ETH-USDT"),
    )

    assert equity == Decimal("1000")
    assert available == Decimal("900")
    assert {position.inst_id: position.size_btc for position in positions} == {
        "BTC-USDT": Decimal("0.002"),
        "ETH-USDT": Decimal("0.05"),
    }


def test_parse_derivatives_position_uses_okx_positions_payload() -> None:
    position = _parse_btc_position(
        "BTC-USDT-SWAP",
        [{"instId": "BTC-USDT-SWAP", "pos": "2", "avgPx": "100000", "upl": "12.5"}],
    )

    assert position is not None
    assert position.size_btc == Decimal("2")
    assert position.avg_px == Decimal("100000")
    assert position.unrealized_pnl_usdt == Decimal("12.5")


def test_spread_bps_from_orderbook_uses_best_bid_and_ask() -> None:
    spread = _spread_bps_from_orderbook(
        [{"bids": [["99", "1", "0", "1"]], "asks": [["101", "1", "0", "1"]]}]
    )

    assert spread == Decimal("200")


def test_depth_slippage_uses_orderbook_levels_for_executable_size() -> None:
    data = [
        {
            "bids": [["99", "1"], ["98", "2"]],
            "asks": [["101", "1"], ["103", "2"]],
        }
    ]

    slippage = _depth_slippage_bps_from_orderbook(data, Decimal("2"), side="buy")

    assert slippage > Decimal("100")


def test_daily_realized_pnl_is_derived_from_okx_fill_rows_after_fees() -> None:
    rows = [
        {
            "instId": "BTC-USDT",
            "side": "buy",
            "fillSz": "1",
            "fillPx": "100",
            "fee": "-1",
            "feeCcy": "USDT",
            "ts": "1767225600000",
            "tradeId": "1",
        },
        {
            "instId": "BTC-USDT",
            "side": "sell",
            "fillSz": "1",
            "fillPx": "120",
            "fee": "-1",
            "feeCcy": "USDT",
            "ts": "1767225660000",
            "tradeId": "2",
        },
    ]

    pnl = _daily_realized_pnl_from_fill_rows(rows, datetime(2026, 1, 1, tzinfo=UTC))

    assert pnl == Decimal("18")


class FakeTradeAPI:
    def __init__(self) -> None:
        self.kwargs: dict[str, str] | None = None

    def place_order(self, **kwargs):
        self.kwargs = kwargs
        return {"code": "0", "data": [{"ordId": "ord-1", "sCode": "0", "sMsg": "OK"}]}


def make_demo_client(fake_trade: FakeTradeAPI) -> OKXDemoClient:
    settings = Settings(
        _env_file=None,
        okx_api_key="key",
        okx_api_secret="secret",
        okx_passphrase="passphrase",
        telegram_bot_token="token",
    )
    client = OKXDemoClient.__new__(OKXDemoClient)
    client.settings = settings
    client.risk_manager = RiskManager(RiskConfig(cooldown_seconds=0))
    client._trade = fake_trade
    return client


@pytest.mark.asyncio
async def test_place_market_buy_order_requests_base_currency_size() -> None:
    fake_trade = FakeTradeAPI()
    client = make_demo_client(fake_trade)

    await client.place_market_order(OrderRequest("BTC-USDT", DecisionAction.BUY, Decimal("0.001")))

    assert fake_trade.kwargs is not None
    assert fake_trade.kwargs["sz"] == "0.001"
    assert fake_trade.kwargs["tgtCcy"] == "base_ccy"


@pytest.mark.asyncio
async def test_place_market_sell_order_leaves_default_size_currency() -> None:
    fake_trade = FakeTradeAPI()
    client = make_demo_client(fake_trade)

    await client.place_market_order(OrderRequest("BTC-USDT", DecisionAction.SELL, Decimal("0.001")))

    assert fake_trade.kwargs is not None
    assert fake_trade.kwargs["sz"] == "0.001"
    assert "tgtCcy" not in fake_trade.kwargs
