from __future__ import annotations

import pytest

from lumiere.historical_data import HistoricalCandleRequest, OKXHistoricalDataClient


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
