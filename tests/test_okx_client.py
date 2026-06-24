from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from lumiere.config import Settings
from lumiere.models import DecisionAction, OrderRequest
from lumiere.okx_client import (
    OKXAPIError,
    OKXDemoClient,
    _daily_realized_pnl_from_fill_rows,
    _depth_slippage_bps_from_orderbook,
    _parse_account_balances,
    _parse_btc_position,
    _parse_candle,
    _spread_bps_from_orderbook,
)
from lumiere.risk import RiskConfig, RiskManager
from lumiere.risk_state import RiskStateStore


def test_parse_candle_preserves_okx_confirm_flag() -> None:
    confirmed = _parse_candle(["1767225600000", "100", "101", "99", "100", "1", "", "", "1"])
    unconfirmed = _parse_candle(
        ["1767225660000", "100", "101", "99", "100", "1", "", "", "0"]
    )

    assert confirmed.confirmed is True
    assert unconfirmed.confirmed is False


@pytest.mark.asyncio
async def test_fetch_candles_filters_unconfirmed_okx_candle() -> None:
    class CandleMarketAPI:
        def get_candlesticks(self, **kwargs):
            _ = kwargs
            return {
                "code": "0",
                "data": [
                    ["1767225660000", "100", "101", "99", "100", "1", "", "", "0"],
                    ["1767225600000", "90", "91", "89", "90", "1", "", "", "1"],
                ],
            }

    client = make_demo_client(FakeTradeAPI())
    client._market = CandleMarketAPI()

    candles = await client.fetch_candles("BTC-USDT")

    assert len(candles) == 1
    assert candles[0].close == Decimal("90")
    assert candles[0].confirmed is True


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
        self.cancelled: list[dict[str, str]] = []
        self.now = datetime.now(tz=UTC)

    def place_order(self, **kwargs):
        self.kwargs = kwargs
        return {"code": "0", "data": [{"ordId": "ord-1", "sCode": "0", "sMsg": "OK"}]}

    def get_fills(self, **kwargs):
        _ = kwargs
        return {"code": "0", "data": self._fill_rows()}

    def get_fills_history(self, *args, **kwargs):
        _ = args, kwargs
        return {"code": "0", "data": self._fill_rows()[:1]}

    def get_order_list(self, **kwargs):
        _ = kwargs
        return {"code": "0", "data": [{"ordId": "open-1"}, {"ordId": "open-2"}]}

    def cancel_order(self, **kwargs):
        self.cancelled.append(kwargs)
        return {"code": "0", "data": [{"ordId": kwargs["ordId"], "sCode": "0"}]}

    def _fill_rows(self) -> list[dict[str, str]]:
        start = datetime(self.now.year, self.now.month, self.now.day, tzinfo=UTC)
        buy_ts = int((start + timedelta(minutes=1)).timestamp() * 1000)
        sell_ts = int((start + timedelta(minutes=2)).timestamp() * 1000)
        return [
            {
                "instId": "BTC-USDT",
                "side": "buy",
                "fillSz": "1",
                "fillPx": "100",
                "fee": "-1",
                "feeCcy": "USDT",
                "ts": str(buy_ts),
                "tradeId": "fill-1",
                "ordId": "ord-1",
            },
            {
                "instId": "BTC-USDT",
                "side": "sell",
                "fillSz": "1",
                "fillPx": "120",
                "fee": "-1",
                "feeCcy": "USDT",
                "ts": str(sell_ts),
                "tradeId": "fill-2",
                "ordId": "ord-2",
            },
        ]


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
async def test_fetch_order_fills_matches_order_and_dedupes_history_rows() -> None:
    fake_trade = FakeTradeAPI()
    client = make_demo_client(fake_trade)
    submitted_at = datetime.now(tz=UTC)
    result = await client.place_market_order(
        OrderRequest("BTC-USDT", DecisionAction.BUY, Decimal("0.001"))
    )

    fills = await client.fetch_order_fills(
        result,
        decision_price=Decimal("99"),
        submitted_at=submitted_at,
    )

    assert len(fills) == 1
    assert fills[0].order_id == "ord-1"
    assert fills[0].trade_id == "fill-1"
    assert fills[0].decision_price == Decimal("99")
    assert fills[0].latency_ms is not None
    assert fills[0].raw["ordId"] == "ord-1"


@pytest.mark.asyncio
async def test_fill_history_paginates_beyond_single_hundred_row_window() -> None:
    class PaginatedTradeAPI(FakeTradeAPI):
        def __init__(self) -> None:
            super().__init__()
            start = datetime(2026, 1, 1, tzinfo=UTC)
            self.rows = [
                {
                    "instId": "BTC-USDT",
                    "side": "buy",
                    "fillSz": "0.001",
                    "fillPx": "100",
                    "fee": "0",
                    "feeCcy": "USDT",
                    "ts": str(int((start + timedelta(seconds=i)).timestamp() * 1000)),
                    "tradeId": f"fill-{i:03d}",
                    "ordId": f"ord-{i:03d}",
                }
                for i in range(150)
            ]
            self.calls: list[dict[str, str]] = []

        def get_fills(self, **kwargs):
            self.calls.append(kwargs)
            before = kwargs.get("before")
            if not before:
                page = self.rows[:100]
            else:
                index = next(i for i, row in enumerate(self.rows) if row["tradeId"] == before)
                page = self.rows[index + 1 : index + 101]
            return {"code": "0", "data": page}

        def get_fills_history(self, *args, **kwargs):
            _ = args, kwargs
            return {"code": "0", "data": []}

    fake_trade = PaginatedTradeAPI()
    client = make_demo_client(fake_trade)

    rows = await client._fetch_fill_rows(  # noqa: SLF001 - verifies pagination helper behavior
        period_start=datetime(2026, 1, 1, tzinfo=UTC),
        period_end=datetime(2026, 1, 2, tzinfo=UTC),
    )

    assert len(rows) == 150
    assert len(fake_trade.calls) == 2
    assert fake_trade.calls[1]["before"] == "fill-099"


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


class FakeAccountAPI:
    def get_account_balance(self):
        return {
            "code": "0",
            "data": [
                {
                    "totalEq": "1000",
                    "details": [
                        {"ccy": "USDT", "availBal": "800"},
                        {"ccy": "BTC", "cashBal": "0.5"},
                        {"ccy": "ETH", "cashBal": "2"},
                    ],
                }
            ],
        }

    def get_positions(self, **kwargs):
        inst_id = kwargs["instId"]
        if inst_id == "ETH-USDT":
            return {
                "code": "0",
                "data": [{"instId": "ETH-USDT", "pos": "2", "avgPx": "100", "upl": "3"}],
            }
        return {"code": "0", "data": []}


class FakeMarketAPI:
    def __init__(self) -> None:
        self.orderbook_calls = 0

    def get_orderbook(self, **kwargs):
        _ = kwargs
        self.orderbook_calls += 1
        return {
            "code": "0",
            "data": [
                {
                    "bids": [["99", "3"], ["98", "3"]],
                    "asks": [["101", "3"], ["102", "3"]],
                }
            ],
        }


@pytest.mark.asyncio
async def test_fetch_account_snapshot_combines_balances_positions_fills_and_orderbooks() -> None:
    fake_trade = FakeTradeAPI()
    settings = Settings(
        _env_file=None,
        okx_api_key="key",
        okx_api_secret="secret",
        okx_passphrase="passphrase",
        telegram_bot_token="token",
        okx_inst_ids="BTC-USDT,ETH-USDT",
    )
    client = OKXDemoClient.__new__(OKXDemoClient)
    client.settings = settings
    client.risk_manager = RiskManager(
        RiskConfig(
            allowed_inst_ids=("BTC-USDT", "ETH-USDT"),
            cooldown_seconds=0,
            max_spread_bps=Decimal("1000"),
        )
    )
    client._account = FakeAccountAPI()
    client._market = FakeMarketAPI()
    client._trade = fake_trade

    snapshot = await client.fetch_account_snapshot()

    assert snapshot.equity_usdt == Decimal("1000")
    assert snapshot.available_usdt == Decimal("800")
    assert snapshot.position_size("BTC-USDT") == Decimal("0.5")
    assert snapshot.position_size("ETH-USDT") == Decimal("2")
    assert snapshot.daily_realized_pnl_usdt == Decimal("18")
    assert snapshot.daily_trade_count == 2
    assert snapshot.spread_bps == Decimal("200")
    assert snapshot.estimated_total_cost_bps is not None


@pytest.mark.asyncio
async def test_fetch_account_snapshot_updates_persistent_drawdown_and_fetches_edge_costs(
    tmp_path,
) -> None:
    class MutableAccountAPI(FakeAccountAPI):
        def __init__(self) -> None:
            self.equity = "1000"

        def get_account_balance(self):
            payload = super().get_account_balance()
            payload["data"][0]["totalEq"] = self.equity
            return payload

    fake_trade = FakeTradeAPI()
    account = MutableAccountAPI()
    market = FakeMarketAPI()
    settings = Settings(
        _env_file=None,
        okx_api_key="key",
        okx_api_secret="secret",
        okx_passphrase="passphrase",
        telegram_bot_token="token",
        risk_state_path=str(tmp_path / "risk_state.json"),
    )
    client = OKXDemoClient.__new__(OKXDemoClient)
    client.settings = settings
    client.risk_manager = RiskManager(
        RiskConfig(cooldown_seconds=0, min_expected_edge_buffer_bps=Decimal("1"))
    )
    client._account = account
    client._market = market
    client._trade = fake_trade
    client._risk_state_store = RiskStateStore(settings.risk_state_path)

    first = await client.fetch_account_snapshot()
    account.equity = "950"
    second = await client.fetch_account_snapshot()

    assert first.max_drawdown_usdt == Decimal("0")
    assert second.max_drawdown_usdt == Decimal("50")
    assert second.estimated_total_cost_bps is not None
    assert market.orderbook_calls == 2


@pytest.mark.asyncio
async def test_cancel_open_orders_cancels_each_okx_open_order() -> None:
    fake_trade = FakeTradeAPI()
    client = make_demo_client(fake_trade)

    cancelled = await client.cancel_open_orders()

    assert cancelled == [{"ordId": "open-1", "sCode": "0"}, {"ordId": "open-2", "sCode": "0"}]
    assert fake_trade.cancelled == [
        {"instId": "BTC-USDT", "ordId": "open-1", "clOrdId": ""},
        {"instId": "BTC-USDT", "ordId": "open-2", "clOrdId": ""},
    ]


@pytest.mark.asyncio
async def test_place_order_raises_when_okx_rejects_or_returns_no_order_data() -> None:
    class RejectingTradeAPI(FakeTradeAPI):
        def place_order(self, **kwargs):
            self.kwargs = kwargs
            return {"code": "0", "data": [{"ordId": "bad", "sCode": "51000", "sMsg": "no"}]}

    class EmptyTradeAPI(FakeTradeAPI):
        def place_order(self, **kwargs):
            self.kwargs = kwargs
            return {"code": "0", "data": []}

    with pytest.raises(OKXAPIError, match="OKX order rejected"):
        await make_demo_client(RejectingTradeAPI()).place_market_order(
            OrderRequest("BTC-USDT", DecisionAction.BUY, Decimal("0.001"))
        )
    with pytest.raises(OKXAPIError, match="no order data"):
        await make_demo_client(EmptyTradeAPI()).place_market_order(
            OrderRequest("BTC-USDT", DecisionAction.BUY, Decimal("0.001"))
        )
