from __future__ import annotations

import asyncio
import csv
import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from tempfile import NamedTemporaryFile

from lumiere.models import MarketCandle
from lumiere.okx_client import _parse_candle, _require_ok_response

DATASET_VERSION = 1


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


@dataclass(frozen=True, slots=True)
class HistoricalDatasetMetadata:
    version: int
    inst_id: str
    bar: str
    fetched_at: datetime
    start: datetime | None
    end: datetime | None
    row_count: int
    checksum_sha256: str
    source: str = "okx_history_candlesticks"

    def to_json_dict(self) -> dict[str, str | int | None]:
        payload = asdict(self)
        payload["fetched_at"] = self.fetched_at.isoformat()
        payload["start"] = None if self.start is None else self.start.isoformat()
        payload["end"] = None if self.end is None else self.end.isoformat()
        return payload

    @classmethod
    def from_json_dict(cls, payload: dict) -> HistoricalDatasetMetadata:
        return cls(
            version=int(payload["version"]),
            inst_id=str(payload["inst_id"]),
            bar=str(payload["bar"]),
            fetched_at=_parse_datetime(str(payload["fetched_at"])),
            start=None if payload.get("start") is None else _parse_datetime(str(payload["start"])),
            end=None if payload.get("end") is None else _parse_datetime(str(payload["end"])),
            row_count=int(payload["row_count"]),
            checksum_sha256=str(payload["checksum_sha256"]),
            source=str(payload.get("source") or "okx_history_candlesticks"),
        )


@dataclass(frozen=True, slots=True)
class HistoricalDataset:
    metadata: HistoricalDatasetMetadata
    candles: tuple[MarketCandle, ...]


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

    async def fetch_candles_paginated(
        self,
        request: HistoricalCandleRequest,
        *,
        start: datetime | None = None,
        end: datetime | None = None,
        max_pages: int | None = None,
    ) -> list[MarketCandle]:
        """Fetch historical candles over a date range using OKX cursor pagination.

        OKX history endpoints return bounded pages. Lumiere walks backwards from ``end`` (or the
        latest available candle) by sending the oldest timestamp from the previous page as the
        next ``after`` cursor. Returned candles are deduplicated by timestamp, sorted
        chronologically, and filtered to the requested inclusive ``start``/``end`` window so
        downstream split and walk-forward code cannot look ahead.
        """

        start = _normalise_datetime(start)
        end = _normalise_datetime(end)
        if start is not None and end is not None and start > end:
            raise ValueError("start must be before end")

        by_timestamp: dict[datetime, MarketCandle] = {}
        after = request.after
        before = request.before or (str(_timestamp_ms(end)) if end is not None else "")
        previous_oldest: datetime | None = None
        page_count = 0

        while True:
            page_count += 1
            if max_pages is not None and page_count > max_pages:
                break
            page = await self.fetch_candles(
                HistoricalCandleRequest(
                    inst_id=request.inst_id,
                    bar=request.bar,
                    limit=request.limit,
                    before=before,
                    after=after,
                )
            )
            if not page:
                break

            for candle in page:
                if start is not None and candle.ts < start:
                    continue
                if end is not None and candle.ts > end:
                    continue
                by_timestamp[candle.ts] = candle

            oldest = page[0].ts
            if start is not None and oldest <= start:
                break
            if previous_oldest is not None and oldest >= previous_oldest:
                break
            previous_oldest = oldest
            after = str(_timestamp_ms(oldest))
            before = ""

        return sorted(by_timestamp.values(), key=lambda candle: candle.ts)

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

    async def fetch_many_paginated(
        self,
        inst_ids: tuple[str, ...],
        *,
        bar: str = "1m",
        limit: int = 100,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> dict[str, list[MarketCandle]]:
        requests = tuple(
            HistoricalCandleRequest(inst_id=inst_id, bar=bar, limit=limit) for inst_id in inst_ids
        )
        results = await asyncio.gather(
            *(
                self.fetch_candles_paginated(request, start=start, end=end)
                for request in requests
            )
        )
        return dict(zip(inst_ids, results, strict=True))


def candles_between(
    candles: list[MarketCandle] | tuple[MarketCandle, ...],
    *,
    start: datetime | None = None,
    end: datetime | None = None,
) -> list[MarketCandle]:
    start = _normalise_datetime(start)
    end = _normalise_datetime(end)
    filtered = []
    for candle in candles:
        if start is not None and candle.ts < start:
            continue
        if end is not None and candle.ts > end:
            continue
        filtered.append(candle)
    return filtered


def dataset_paths(
    cache_dir: Path | str,
    *,
    inst_id: str,
    bar: str,
    start: datetime | None,
    end: datetime | None,
) -> tuple[Path, Path]:
    cache_path = Path(cache_dir)
    stem = "_".join(
        [
            _safe_token(inst_id),
            _safe_token(bar),
            _range_token(start),
            _range_token(end),
            f"v{DATASET_VERSION}",
        ]
    )
    return cache_path / f"{stem}.csv", cache_path / f"{stem}.metadata.json"


def save_dataset(
    cache_dir: Path | str,
    *,
    inst_id: str,
    bar: str,
    candles: list[MarketCandle] | tuple[MarketCandle, ...],
    start: datetime | None = None,
    end: datetime | None = None,
    fetched_at: datetime | None = None,
) -> HistoricalDataset:
    ordered = tuple(sorted(candles, key=lambda candle: candle.ts))
    csv_path, metadata_path = dataset_paths(
        cache_dir,
        inst_id=inst_id,
        bar=bar,
        start=start,
        end=end,
    )
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_text = candles_to_csv_text(ordered)
    checksum = hashlib.sha256(csv_text.encode("utf-8")).hexdigest()
    metadata = HistoricalDatasetMetadata(
        version=DATASET_VERSION,
        inst_id=inst_id,
        bar=bar,
        fetched_at=_normalise_datetime(fetched_at) or datetime.now(tz=UTC),
        start=_normalise_datetime(start),
        end=_normalise_datetime(end),
        row_count=len(ordered),
        checksum_sha256=checksum,
    )
    _atomic_write_text(csv_path, csv_text)
    _atomic_write_text(metadata_path, json.dumps(metadata.to_json_dict(), indent=2, sort_keys=True))
    return HistoricalDataset(metadata=metadata, candles=ordered)


def load_dataset(
    cache_dir: Path | str,
    *,
    inst_id: str,
    bar: str,
    start: datetime | None = None,
    end: datetime | None = None,
) -> HistoricalDataset:
    csv_path, metadata_path = dataset_paths(
        cache_dir,
        inst_id=inst_id,
        bar=bar,
        start=start,
        end=end,
    )
    csv_text = csv_path.read_text(encoding="utf-8")
    checksum = hashlib.sha256(csv_text.encode("utf-8")).hexdigest()
    metadata = HistoricalDatasetMetadata.from_json_dict(
        json.loads(metadata_path.read_text(encoding="utf-8"))
    )
    if metadata.version != DATASET_VERSION:
        raise ValueError(f"unsupported dataset version: {metadata.version}")
    if metadata.inst_id != inst_id or metadata.bar != bar:
        raise ValueError("dataset metadata does not match requested instrument/bar")
    if metadata.checksum_sha256 != checksum:
        raise ValueError("historical dataset checksum mismatch")
    candles = csv_text_to_candles(csv_text)
    if len(candles) != metadata.row_count:
        raise ValueError("historical dataset row count mismatch")
    return HistoricalDataset(
        metadata=metadata,
        candles=tuple(candles_between(candles, start=start, end=end)),
    )


def dataset_exists(
    cache_dir: Path | str,
    *,
    inst_id: str,
    bar: str,
    start: datetime | None = None,
    end: datetime | None = None,
) -> bool:
    csv_path, metadata_path = dataset_paths(
        cache_dir,
        inst_id=inst_id,
        bar=bar,
        start=start,
        end=end,
    )
    return csv_path.exists() and metadata_path.exists()


def candles_to_csv_text(candles: tuple[MarketCandle, ...]) -> str:
    lines = ["ts,open,high,low,close,volume,confirmed\n"]
    for candle in candles:
        lines.append(
            ",".join(
                [
                    candle.ts.isoformat(),
                    str(candle.open),
                    str(candle.high),
                    str(candle.low),
                    str(candle.close),
                    str(candle.volume),
                    "1" if candle.confirmed else "0",
                ]
            )
            + "\n"
        )
    return "".join(lines)


def csv_text_to_candles(csv_text: str) -> tuple[MarketCandle, ...]:
    rows = csv.DictReader(csv_text.splitlines())
    candles = [
        MarketCandle(
            ts=_parse_datetime(row["ts"]),
            open=Decimal(row["open"]),
            high=Decimal(row["high"]),
            low=Decimal(row["low"]),
            close=Decimal(row["close"]),
            volume=Decimal(row.get("volume") or "0"),
            confirmed=(row.get("confirmed") or "1") in {"1", "true", "True"},
        )
        for row in rows
    ]
    return tuple(sorted(candles, key=lambda candle: candle.ts))


def parse_cli_datetime(raw: str | None) -> datetime | None:
    if raw is None or not raw.strip():
        return None
    return _parse_datetime(raw.strip())


def _parse_datetime(raw: str) -> datetime:
    parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    return _normalise_datetime(parsed) or parsed.replace(tzinfo=UTC)


def _normalise_datetime(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _timestamp_ms(value: datetime) -> int:
    return int(value.timestamp() * 1000)


def _safe_token(raw: str) -> str:
    return raw.replace("/", "-").replace(":", "").replace(" ", "_")


def _range_token(value: datetime | None) -> str:
    if value is None:
        return "open"
    return value.strftime("%Y%m%dT%H%M%SZ")


def _atomic_write_text(path: Path, text: str) -> None:
    with NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        handle.write(text)
        temporary_path = Path(handle.name)
    temporary_path.replace(path)
