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
            maker_fee_bps="2",
            execution_policy="market",
            marketable_limit_buffer_bps="1",
            post_only_offset_bps="0",
            maker_timeout_bars=1,
            maker_fill_fraction="1",
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
    assert payload["reports"][0]["strategies"] == [
        "moving_average_crossover",
        "rsi_mean_reversion",
        "volatility_breakout",
    ]
    assert {candidate["candidate"]["strategy"] for candidate in candidates} == {
        "moving_average_crossover",
        "rsi_mean_reversion",
        "volatility_breakout",
    }
    assert [candidate["accepted"] for candidate in candidates] == [False] * 5
    assert {candidate["rejection_reason"] for candidate in candidates} == {"not_enough_trades"}
    assert all("expectancy_calibration" in candidate for candidate in candidates)
    assert candidates[0]["test_report"]["execution_timing"] == "next_open"
    assert payload["accepted_configs"] == []
    assert (tmp_path / "reports" / "optimizer_report.json").exists()
    assert (tmp_path / "reports" / "accepted_candidates.json").exists()


@pytest.mark.asyncio
async def test_optimizer_cli_supports_post_only_maker_cost_assumptions(tmp_path) -> None:
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
            strategies=["moving_average_crossover"],
            limit=300,
            start=start.isoformat(),
            end=end.isoformat(),
            cache_dir=str(tmp_path / "cache"),
            offline=True,
            refresh_cache=False,
            output_dir=str(tmp_path / "reports"),
            fast_window="1",
            slow_window="2",
            trade_size_btc="1",
            trade_size_eth="1",
            cooldown_seconds=30,
            starting_equity_usdt="1000",
            taker_fee_bps="10",
            maker_fee_bps="2",
            execution_policy="post_only_maker",
            marketable_limit_buffer_bps="1",
            post_only_offset_bps="0",
            maker_timeout_bars=1,
            maker_fill_fraction="1",
            spread_bps="2",
            slippage_bps="5",
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

    candidate = payload["reports"][0]["candidates"][0]
    assert candidate["test_report"]["assumptions"]["execution_policy"] == "post_only_maker"
    assert candidate["test_report"]["assumptions"]["maker_fee_bps"] == "2"
    assert candidate["expectancy_calibration"]["cost_bps"] == "4"
    assert payload["criteria"]["cost_model"]["execution_policy"] == "post_only_maker"
    assert payload["criteria"]["cost_model"]["maker_fee_bps"] == "2"
