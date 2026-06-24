from __future__ import annotations

import json
import sys
from datetime import UTC, datetime, timedelta
from decimal import Decimal

from lumiere import backtest_cli, evidence_cli, optimizer_cli
from lumiere.historical_data import save_dataset
from lumiere.models import MarketCandle


def candles(closes: list[str]) -> tuple[MarketCandle, ...]:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    return tuple(
        MarketCandle(
            ts=start + timedelta(minutes=index),
            open=Decimal(close),
            high=Decimal(close),
            low=Decimal(close),
            close=Decimal(close),
        )
        for index, close in enumerate(closes)
    )


def cache_dataset(tmp_path, closes: list[str]) -> tuple[datetime, datetime]:
    rows = candles(closes)
    start = rows[0].ts
    end = rows[-1].ts
    save_dataset(
        tmp_path / "cache",
        inst_id="BTC-USDT",
        bar="1m",
        candles=rows,
        start=start,
        end=end,
    )
    return start, end


def test_backtest_cli_smoke_uses_offline_cached_fixture(tmp_path, monkeypatch, capsys) -> None:
    start, end = cache_dataset(tmp_path, ["100", "101", "110", "90", "120"])
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "lumiere-backtest",
            "--offline",
            "--cache-dir",
            str(tmp_path / "cache"),
            "--inst-id",
            "BTC-USDT",
            "--bar",
            "1m",
            "--start",
            start.isoformat(),
            "--end",
            end.isoformat(),
            "--fast-window",
            "1",
            "--slow-window",
            "2",
            "--trade-size-btc",
            "1",
            "--taker-fee-bps",
            "0",
            "--spread-bps",
            "0",
            "--slippage-bps",
            "0",
        ],
    )

    backtest_cli.main()

    payload = json.loads(capsys.readouterr().out)
    assert payload["reports"][0]["inst_id"] == "BTC-USDT"
    assert payload["reports"][0]["dataset"]["source"] == "okx_history_candlesticks"


def test_optimizer_cli_smoke_writes_offline_artifacts(tmp_path, monkeypatch, capsys) -> None:
    start, end = cache_dataset(tmp_path, ["100", "90", "91", "150", "100", "90", "91", "150"])
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "lumiere-optimize",
            "--offline",
            "--cache-dir",
            str(tmp_path / "cache"),
            "--output-dir",
            str(tmp_path / "reports"),
            "--inst-id",
            "BTC-USDT",
            "--bar",
            "1m",
            "--strategy",
            "moving_average_crossover",
            "--start",
            start.isoformat(),
            "--end",
            end.isoformat(),
            "--fast-window",
            "1,2",
            "--slow-window",
            "2,3",
            "--trade-size-btc",
            "10",
            "--taker-fee-bps",
            "0",
            "--spread-bps",
            "0",
            "--slippage-bps",
            "0",
            "--min-trades",
            "1",
            "--min-profit-factor",
            "none",
        ],
    )

    optimizer_cli.main()

    payload = json.loads(capsys.readouterr().out)
    assert payload["reports"][0]["candidate_count"] == 3
    assert (tmp_path / "reports" / "optimizer_report.json").exists()
    assert (tmp_path / "reports" / "accepted_candidates.json").exists()


def test_evidence_cli_smoke_writes_promotion_packet(tmp_path, monkeypatch, capsys) -> None:
    optimizer_report = tmp_path / "optimizer.json"
    optimizer_report.write_text(
        json.dumps({"accepted_configs": [{"inst_id": "BTC-USDT", "strategy": "fixture"}]}),
        encoding="utf-8",
    )
    output = tmp_path / "promotion.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "lumiere-evidence",
            "--stage",
            "backtest",
            "--optimizer-report",
            str(optimizer_report),
            "--attribution-ledger",
            str(tmp_path / "absent-attribution.jsonl"),
            "--output",
            str(output),
        ],
    )

    evidence_cli.main()

    payload = json.loads(capsys.readouterr().out)
    assert payload["go"] is True
    assert output.exists()
