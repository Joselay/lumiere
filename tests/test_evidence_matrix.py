from __future__ import annotations

from argparse import Namespace
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from lumiere.evidence_matrix import run_evidence_matrix
from lumiere.historical_data import save_dataset
from lumiere.models import MarketCandle

MATRIX_CLOSES = [
    "100",
    "101",
    "102",
    "103",
    "103",
    "101",
    "99",
    "97",
    "97",
    "99",
    "101",
    "103",
]


def make_candles(closes: list[str]) -> tuple[MarketCandle, ...]:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    candles = []
    for index, close in enumerate(closes):
        price = Decimal(close)
        candles.append(
            MarketCandle(
                ts=start + timedelta(minutes=index),
                open=price,
                high=price * Decimal("1.01"),
                low=price * Decimal("0.99"),
                close=price,
                volume=Decimal("10"),
            )
        )
    return tuple(candles)


def matrix_args(tmp_path, *, min_span_days: int = 0, min_regime_passes: int = 0) -> Namespace:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    end = datetime(2026, 1, 1, 0, 11, tzinfo=UTC)
    return Namespace(
        inst_ids=["BTC-USDT"],
        bar=["1m"],
        strategies=["moving_average_crossover"],
        limit=300,
        start=start.isoformat(),
        end=end.isoformat(),
        cache_dir=str(tmp_path / "cache"),
        offline=True,
        refresh_cache=False,
        output_dir=str(tmp_path / "reports"),
        min_span_days=min_span_days,
        max_span_days=730,
        min_regime_passes=min_regime_passes,
        regime_window_size=4,
        min_regime_candles=4,
        trend_threshold_bps="50",
        high_volatility_bps="50",
        drawdown_threshold_bps="100",
        event_range_bps="150",
        starting_equity_usdt="1000",
        fast_window=1,
        slow_window=2,
        trade_size_btc="0.001",
        trade_size_eth="0.01",
        rsi_period=2,
        oversold_rsi="30",
        overbought_rsi="70",
        breakout_lookback=2,
        breakout_atr_period=2,
        breakout_atr_multiplier="0.5",
        breakout_min_atr_pct="0.001",
        taker_fee_bps="0",
        spread_bps="0",
        slippage_bps="0",
        market_impact_bps="0",
        reject_every_n_orders=0,
        stop_loss_bps="0",
        take_profit_bps="0",
        trailing_stop_bps="0",
        max_bars_in_trade=0,
        train_fraction="0.5",
        validation_fraction="0.25",
        walk_forward_train_size=4,
        walk_forward_test_size=4,
        walk_forward_step_size=4,
        no_walk_forward=False,
        min_trades=0,
        min_net_pnl_usdt="-100000",
        max_drawdown_usdt="0",
        min_profit_factor="none",
        allow_baseline_underperformance=True,
    )


def save_matrix_dataset(tmp_path, args: Namespace) -> None:
    save_dataset(
        tmp_path / "cache",
        inst_id="BTC-USDT",
        bar="1m",
        candles=make_candles(MATRIX_CLOSES),
        start=datetime.fromisoformat(args.start),
        end=datetime.fromisoformat(args.end),
    )


@pytest.mark.asyncio
async def test_evidence_matrix_builds_offline_artifact_with_checksums_and_regime_labels(
    tmp_path,
) -> None:
    args = matrix_args(tmp_path)
    save_matrix_dataset(tmp_path, args)

    payload = await run_evidence_matrix(args)

    row = payload["matrix"][0]
    assert row["dataset"]["checksum_sha256"] == row["dataset_horizon"]["checksum_sha256"]
    assert row["market_regimes"]["window_count"] == 3
    assert {report["split_name"] for report in row["split_reports"]} == {
        "train",
        "validation",
        "test",
    }
    assert row["walk_forward_reports"][0]["test"]["role"] == "out_of_sample"
    assert row["accepted"] is True
    assert payload["accepted_configs"][0]["source_report"] == "evidence_matrix"
    assert (tmp_path / "reports" / "evidence_matrix.json").exists()
    assert (tmp_path / "reports" / "accepted_configs.json").exists()


@pytest.mark.asyncio
async def test_evidence_matrix_requires_multiple_regime_passes_before_accepting_config(
    tmp_path,
) -> None:
    args = matrix_args(tmp_path, min_regime_passes=4)
    save_matrix_dataset(tmp_path, args)

    payload = await run_evidence_matrix(args)

    row = payload["matrix"][0]
    assert row["accepted"] is False
    assert row["rejection_reason"] == "insufficient_regime_passes"
    assert payload["accepted_configs"] == []


@pytest.mark.asyncio
async def test_evidence_matrix_rejects_short_horizon_dataset(tmp_path) -> None:
    args = matrix_args(tmp_path, min_span_days=180)
    save_matrix_dataset(tmp_path, args)

    payload = await run_evidence_matrix(args)

    row = payload["matrix"][0]
    assert row["dataset_horizon"]["too_short"] is True
    assert row["accepted"] is False
    assert row["rejection_reason"] == "dataset_horizon_short"
