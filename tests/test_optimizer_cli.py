from __future__ import annotations

from argparse import Namespace
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from lumiere.historical_data import save_dataset
from lumiere.models import MarketCandle
from lumiere.optimizer_cli import run_optimizer


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


@pytest.mark.asyncio
async def test_optimizer_cli_produces_sorted_report_and_artifacts(tmp_path) -> None:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    end = datetime(2026, 1, 1, 0, 7, tzinfo=UTC)
    save_dataset(
        tmp_path / "cache",
        inst_id="BTC-USDT",
        bar="1m",
        candles=candles(["100", "90", "91", "150", "100", "90", "91", "150"]),
        start=start,
        end=end,
    )

    payload = await run_optimizer(
        Namespace(
            inst_ids=["BTC-USDT"],
            bar=["1m"],
            limit=300,
            start=start.isoformat(),
            end=end.isoformat(),
            cache_dir=str(tmp_path / "cache"),
            offline=True,
            refresh_cache=False,
            output_dir=str(tmp_path / "reports"),
            fast_window="1,2",
            slow_window="2,3",
            trade_size_btc="10",
            trade_size_eth="10",
            cooldown_seconds=30,
            starting_equity_usdt="1000",
            taker_fee_bps="0",
            spread_bps="0",
            slippage_bps="0",
            market_impact_bps="0",
            reject_every_n_orders=0,
            train_fraction="0.5",
            min_trades=1,
            min_net_pnl_usdt="0",
            max_drawdown_usdt="0",
            min_profit_factor="none",
            max_train_test_pnl_ratio="4",
            min_walk_forward_windows=0,
            walk_forward_train_size=0,
            walk_forward_test_size=0,
            min_walk_forward_pass_rate="0.5",
            min_stable_neighbors=0,
            parameter_stability_radius=1,
        )
    )

    candidates = payload["reports"][0]["candidates"]
    assert candidates[0]["accepted"] is True
    assert candidates[-1]["accepted"] is False
    assert payload["accepted_configs"][0]["cooldown_seconds"] == 30
    assert (tmp_path / "reports" / "optimizer_report.json").exists()
    assert (tmp_path / "reports" / "accepted_candidates.json").exists()
