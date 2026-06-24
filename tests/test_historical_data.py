from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

import pytest

from lumiere.historical_data import (
    HistoricalCandleRequest,
    OKXHistoricalDataClient,
    load_dataset,
    save_dataset,
)
from lumiere.models import MarketCandle


class FakeMarketAPI:
    def __init__(self) -> None:
        self.called = False
        self.kwargs = None

    def get_history_candlesticks(self, **kwargs):
        self.called = True
        self.kwargs = kwargs
        return {
            "code": "0",
            "data": [
                ["2000", "101", "101", "101", "101", "2"],
                ["1000", "100", "100", "100", "100", "1"],
            ],
        }


@pytest.mark.asyncio
async def test_historical_client_uses_okx_sdk_history_candles_and_sorts() -> None:
    fake = FakeMarketAPI()
    client = OKXHistoricalDataClient(market_api=fake)

    candles = await client.fetch_candles(HistoricalCandleRequest("BTC-USDT", bar="1m", limit=2))

    assert fake.called is True
    assert fake.kwargs == {
        "instId": "BTC-USDT",
        "after": "",
        "before": "",
        "bar": "1m",
        "limit": "2",
    }
    assert [candle.close for candle in candles] == [100, 101]


class PaginatedFakeMarketAPI:
    def __init__(self) -> None:
        self.kwargs = []

    def get_history_candlesticks(self, **kwargs):
        self.kwargs.append(kwargs)
        after = kwargs.get("after") or ""
        if after == "":
            return {
                "code": "0",
                "data": [
                    ["4000", "103", "103", "103", "103", "1"],
                    ["3000", "102", "102", "102", "102", "1"],
                    ["2000", "101", "101", "101", "101", "1"],
                ],
            }
        if after == "2000":
            return {
                "code": "0",
                "data": [
                    ["2000", "101", "101", "101", "101", "1"],
                    ["1000", "100", "100", "100", "100", "1"],
                ],
            }
        return {"code": "0", "data": []}


@pytest.mark.asyncio
async def test_paginated_history_orders_deduplicates_and_filters_range() -> None:
    fake = PaginatedFakeMarketAPI()
    client = OKXHistoricalDataClient(market_api=fake)

    candles = await client.fetch_candles_paginated(
        HistoricalCandleRequest("BTC-USDT", bar="1m", limit=3),
        start=datetime.fromtimestamp(1, tz=UTC),
        end=datetime.fromtimestamp(3, tz=UTC),
    )

    assert [candle.ts.timestamp() for candle in candles] == [1, 2, 3]
    assert [candle.close for candle in candles] == [100, 101, 102]
    assert fake.kwargs[0]["before"] == "3000"
    assert fake.kwargs[1]["after"] == "2000"


def test_historical_dataset_round_trips_with_checksum_metadata(tmp_path) -> None:
    rows = (
        MarketCandle(
            ts=datetime(2026, 1, 1, tzinfo=UTC),
            open=Decimal("1"),
            high=Decimal("2"),
            low=Decimal("1"),
            close=Decimal("2"),
        ),
    )

    saved = save_dataset(
        tmp_path,
        inst_id="BTC-USDT",
        bar="1m",
        candles=rows,
        start=datetime(2026, 1, 1, tzinfo=UTC),
        end=datetime(2026, 1, 2, tzinfo=UTC),
    )
    loaded = load_dataset(
        tmp_path,
        inst_id="BTC-USDT",
        bar="1m",
        start=datetime(2026, 1, 1, tzinfo=UTC),
        end=datetime(2026, 1, 2, tzinfo=UTC),
    )

    assert loaded.metadata.checksum_sha256 == saved.metadata.checksum_sha256
    assert loaded.candles == rows
