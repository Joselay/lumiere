from __future__ import annotations

import argparse
import asyncio
import json
from decimal import Decimal
from pathlib import Path

from lumiere.backtest import BacktestConfig, Backtester, CostModel
from lumiere.config import SUPPORTED_INST_IDS
from lumiere.historical_data import (
    HistoricalCandleRequest,
    OKXHistoricalDataClient,
    dataset_exists,
    load_dataset,
    parse_cli_datetime,
    save_dataset,
)
from lumiere.models import MarketCandle
from lumiere.strategy import MovingAverageCrossoverConfig, MovingAverageCrossoverStrategy
from lumiere.strategy_evaluation import (
    EvaluationConfig,
    MovingAverageCandidate,
    baseline_comparison,
    split_backtest_reports,
    walk_forward_backtest_reports,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a Lumiere OKX historical backtest")
    parser.add_argument("--inst-id", action="append", choices=SUPPORTED_INST_IDS, dest="inst_ids")
    parser.add_argument("--bar", default="1m")
    parser.add_argument("--limit", type=int, default=300, help="OKX candles per page")
    parser.add_argument("--start", help="inclusive UTC start time, e.g. 2026-01-01T00:00:00Z")
    parser.add_argument("--end", help="inclusive UTC end time, e.g. 2026-03-01T00:00:00Z")
    parser.add_argument("--cache-dir", default="data/historical")
    parser.add_argument(
        "--offline",
        action="store_true",
        help="load cached candles without OKX access",
    )
    parser.add_argument(
        "--refresh-cache",
        action="store_true",
        help="ignore existing cache and refetch",
    )
    parser.add_argument("--starting-equity-usdt", default="1000")
    parser.add_argument("--fast-window", type=int, default=5)
    parser.add_argument("--slow-window", type=int, default=20)
    parser.add_argument("--trade-size-btc", default="0.001")
    parser.add_argument("--trade-size-eth", default="0.01")
    parser.add_argument("--taker-fee-bps", default="10")
    parser.add_argument("--spread-bps", default="2")
    parser.add_argument("--slippage-bps", default="5")
    parser.add_argument("--market-impact-bps", default="0")
    parser.add_argument("--reject-every-n-orders", type=int, default=0)
    parser.add_argument("--train-fraction", default="0.6")
    parser.add_argument("--validation-fraction", default="0.2")
    parser.add_argument("--walk-forward-train-size", type=int, default=0)
    parser.add_argument("--walk-forward-test-size", type=int, default=0)
    parser.add_argument("--walk-forward-step-size", type=int, default=0)
    parser.add_argument("--no-walk-forward", action="store_true")
    return parser


async def run_backtest(args: argparse.Namespace) -> list[dict]:
    inst_ids = tuple(args.inst_ids or SUPPORTED_INST_IDS)
    start = parse_cli_datetime(args.start)
    end = parse_cli_datetime(args.end)
    cache_dir = Path(args.cache_dir)
    data_client = None if args.offline else OKXHistoricalDataClient(flag="1")
    cost_model = CostModel(
        taker_fee_bps=Decimal(args.taker_fee_bps),
        spread_bps=Decimal(args.spread_bps),
        slippage_bps=Decimal(args.slippage_bps),
        market_impact_bps=Decimal(args.market_impact_bps),
        reject_every_n_orders=args.reject_every_n_orders,
    )
    evaluation_config = EvaluationConfig(
        train_fraction=Decimal(args.train_fraction),
        starting_equity_usdt=Decimal(args.starting_equity_usdt),
        cost_model=cost_model,
    )
    reports = []
    for inst_id in inst_ids:
        candles, dataset_metadata = await _load_or_fetch_candles(
            inst_id=inst_id,
            bar=args.bar,
            limit=args.limit,
            start=start,
            end=end,
            cache_dir=cache_dir,
            offline=args.offline,
            refresh_cache=args.refresh_cache,
            data_client=data_client,
        )
        configured_size = args.trade_size_eth if inst_id.startswith("ETH-") else args.trade_size_btc
        trade_size = Decimal(configured_size)
        strategy = MovingAverageCrossoverStrategy(
            MovingAverageCrossoverConfig(
                inst_id=inst_id,
                fast_window=args.fast_window,
                slow_window=args.slow_window,
                trade_size_btc=trade_size,
            )
        )
        report = Backtester(
            strategy,
            BacktestConfig(
                starting_equity_usdt=Decimal(args.starting_equity_usdt),
                cost_model=cost_model,
            ),
        ).run(candles)
        candidate = MovingAverageCandidate(args.fast_window, args.slow_window, trade_size)
        payload = report.to_dict()
        payload["baseline_comparison"] = baseline_comparison(report)
        payload["dataset"] = None if dataset_metadata is None else dataset_metadata.to_json_dict()
        payload["split_reports"] = _safe_split_reports(
            inst_id,
            candles,
            candidate,
            evaluation_config,
            train_fraction=Decimal(args.train_fraction),
            validation_fraction=Decimal(args.validation_fraction),
        )
        payload["walk_forward_reports"] = _safe_walk_forward_reports(
            inst_id,
            candles,
            candidate,
            evaluation_config,
            train_size=args.walk_forward_train_size,
            test_size=args.walk_forward_test_size,
            step_size=args.walk_forward_step_size,
            enabled=not args.no_walk_forward,
        )
        reports.append(payload)
    return reports


async def _load_or_fetch_candles(
    *,
    inst_id: str,
    bar: str,
    limit: int,
    start,
    end,
    cache_dir: Path,
    offline: bool,
    refresh_cache: bool,
    data_client: OKXHistoricalDataClient | None,
):
    if not refresh_cache and dataset_exists(
        cache_dir,
        inst_id=inst_id,
        bar=bar,
        start=start,
        end=end,
    ):
        dataset = load_dataset(cache_dir, inst_id=inst_id, bar=bar, start=start, end=end)
        return list(dataset.candles), dataset.metadata
    if offline:
        dataset = load_dataset(cache_dir, inst_id=inst_id, bar=bar, start=start, end=end)
        return list(dataset.candles), dataset.metadata
    if data_client is None:
        raise RuntimeError("OKX data client is unavailable")

    request = HistoricalCandleRequest(inst_id=inst_id, bar=bar, limit=limit)
    if start is not None or end is not None:
        candles = await data_client.fetch_candles_paginated(request, start=start, end=end)
    else:
        candles = await data_client.fetch_candles(request)
    dataset = save_dataset(
        cache_dir,
        inst_id=inst_id,
        bar=bar,
        candles=candles,
        start=start,
        end=end,
    )
    return list(dataset.candles), dataset.metadata


def _safe_split_reports(
    inst_id: str,
    candles: list[MarketCandle],
    candidate: MovingAverageCandidate,
    config: EvaluationConfig,
    *,
    train_fraction: Decimal,
    validation_fraction: Decimal,
) -> list[dict]:
    if len(candles) < 3:
        return []
    return [
        report.to_dict()
        for report in split_backtest_reports(
            inst_id,
            candles,
            candidate,
            config,
            train_fraction=train_fraction,
            validation_fraction=validation_fraction,
        )
    ]


def _safe_walk_forward_reports(
    inst_id: str,
    candles: list[MarketCandle],
    candidate: MovingAverageCandidate,
    config: EvaluationConfig,
    *,
    train_size: int,
    test_size: int,
    step_size: int,
    enabled: bool,
) -> list[dict]:
    if not enabled or len(candles) < 5:
        return []
    resolved_train_size = train_size or max(2, int(len(candles) * 0.6))
    resolved_test_size = test_size or max(1, int(len(candles) * 0.2))
    if resolved_train_size + resolved_test_size > len(candles):
        return []
    return list(
        walk_forward_backtest_reports(
            inst_id,
            candles,
            candidate,
            config,
            train_size=resolved_train_size,
            test_size=resolved_test_size,
            step_size=step_size or None,
        )
    )


def main() -> None:
    args = build_parser().parse_args()
    reports = asyncio.run(run_backtest(args))
    print(json.dumps({"reports": reports}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
