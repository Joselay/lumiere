from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime

from lumiere.models import MarketCandle
from lumiere.okx_client import _parse_candle, _require_ok_response


@dataclass(frozen=True, slots=True)
class HistoricalCandleRequest:
    inst_id: str
    bar: str = "1m"
    limit: int = 100
    before: str = ""
    after: str = ""

    def __post_init__(self) -> None:
        if "-" not in self.inst_id:
            raise ValueError("inst_id must be an OKX instrument id like BTC-USDT")
        if self.limit <= 0:
            raise ValueError("limit must be positive")


class OKXHistoricalDataClient:
    """Historical candle access through the python-okx MarketData SDK only."""

    def __init__(self, *, flag: str = "1", market_api=None) -> None:
        if flag != "1":
            raise ValueError("Lumiere historical data uses OKX demo flag only; set flag='1'")
        if market_api is None:
            from okx import MarketData  # type: ignore[import-not-found]

            market_api = MarketData.MarketAPI(flag=flag, debug=False)
        self._market_api = market_api

    async def fetch_candles(self, request: HistoricalCandleRequest) -> list[MarketCandle]:
        response = await asyncio.to_thread(
            self._market_api.get_history_candlesticks,
            instId=request.inst_id,
            after=request.after,
            before=request.before,
            bar=request.bar,
            limit=str(request.limit),
        )
        candles = [_parse_candle(row) for row in _require_ok_response(response)]
        return sorted(candles, key=lambda candle: candle.ts)

    async def fetch_many(
        self,
        inst_ids: tuple[str, ...],
        *,
        bar: str = "1m",
        limit: int = 100,
    ) -> dict[str, list[MarketCandle]]:
        requests = tuple(
            HistoricalCandleRequest(inst_id=inst_id, bar=bar, limit=limit) for inst_id in inst_ids
        )
        results = await asyncio.gather(*(self.fetch_candles(request) for request in requests))
        return dict(zip(inst_ids, results, strict=True))


def candles_between(
    candles: list[MarketCandle] | tuple[MarketCandle, ...],
    *,
    start: datetime | None = None,
    end: datetime | None = None,
) -> list[MarketCandle]:
    filtered = []
    for candle in candles:
        if start is not None and candle.ts < start:
            continue
        if end is not None and candle.ts > end:
            continue
        filtered.append(candle)
    return filtered
