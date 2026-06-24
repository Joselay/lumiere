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
from lumiere.risk import RiskConfig
from lumiere.strategy import TradingStrategy
from lumiere.strategy_evaluation import (
    EvaluationConfig,
    baseline_comparison,
    train_validation_test_split,
    walk_forward_splits,
)
from lumiere.strategy_factory import STRATEGY_NAMES, build_strategy


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
    parser.add_argument("--strategy", choices=STRATEGY_NAMES, default="moving_average_crossover")
    parser.add_argument("--fast-window", type=int, default=5)
    parser.add_argument("--slow-window", type=int, default=20)
    parser.add_argument("--trade-size-btc", default="0.001")
    parser.add_argument("--trade-size-eth", default="0.01")
    parser.add_argument("--rsi-period", type=int, default=14)
    parser.add_argument("--oversold-rsi", default="30")
    parser.add_argument("--overbought-rsi", default="70")
    parser.add_argument("--breakout-lookback", type=int, default=20)
    parser.add_argument("--breakout-atr-period", type=int, default=14)
    parser.add_argument("--breakout-atr-multiplier", default="0.5")
    parser.add_argument("--breakout-min-atr-pct", default="0.001")
    parser.add_argument("--taker-fee-bps", default="10")
    parser.add_argument("--maker-fee-bps", default="2")
    parser.add_argument("--spread-bps", default="2")
    parser.add_argument("--slippage-bps", default="5")
    parser.add_argument("--market-impact-bps", default="0")
    parser.add_argument("--reject-every-n-orders", type=int, default=0)
    parser.add_argument(
        "--execution-policy",
        choices=("market", "marketable_limit", "post_only_maker"),
        default="market",
    )
    parser.add_argument("--marketable-limit-buffer-bps", default="1")
    parser.add_argument("--post-only-offset-bps", default="0")
    parser.add_argument("--maker-timeout-bars", type=int, default=1)
    parser.add_argument("--maker-fill-fraction", default="1")
    parser.add_argument("--stop-loss-bps", default="0")
    parser.add_argument("--take-profit-bps", default="0")
    parser.add_argument("--trailing-stop-bps", default="0")
    parser.add_argument("--max-bars-in-trade", type=int, default=0)
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
        maker_fee_bps=Decimal(args.maker_fee_bps),
        spread_bps=Decimal(args.spread_bps),
        slippage_bps=Decimal(args.slippage_bps),
        market_impact_bps=Decimal(args.market_impact_bps),
        reject_every_n_orders=args.reject_every_n_orders,
        execution_policy=args.execution_policy,
        marketable_limit_buffer_bps=Decimal(args.marketable_limit_buffer_bps),
        post_only_offset_bps=Decimal(args.post_only_offset_bps),
        maker_timeout_bars=args.maker_timeout_bars,
        maker_fill_fraction=Decimal(args.maker_fill_fraction),
    )
    evaluation_config = EvaluationConfig(
        train_fraction=Decimal(args.train_fraction),
        starting_equity_usdt=Decimal(args.starting_equity_usdt),
        cost_model=cost_model,
        stop_loss_bps=_positive_decimal_or_none(args.stop_loss_bps),
        take_profit_bps=_positive_decimal_or_none(args.take_profit_bps),
        trailing_stop_bps=_positive_decimal_or_none(args.trailing_stop_bps),
        max_bars_in_trade=args.max_bars_in_trade or None,
        risk_config=RiskConfig(
            allowed_inst_ids=inst_ids,
            cooldown_seconds=0,
            max_position_by_inst_id={
                inst_id: Decimal("0.05" if inst_id.startswith("ETH-") else "0.005")
                for inst_id in inst_ids
            },
            min_order_by_inst_id={
                inst_id: Decimal("0.0001" if inst_id.startswith("ETH-") else "0.00001")
                for inst_id in inst_ids
            },
        ),
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
        strategy = _strategy_from_args(args, inst_id)
        report = _run_strategy_report(strategy, candles, evaluation_config)
        payload = report.to_dict()
        payload["baseline_comparison"] = baseline_comparison(report)
        payload["dataset"] = None if dataset_metadata is None else dataset_metadata.to_json_dict()
        payload["split_reports"] = _safe_split_reports(
            inst_id,
            candles,
            args,
            evaluation_config,
            train_fraction=Decimal(args.train_fraction),
            validation_fraction=Decimal(args.validation_fraction),
        )
        payload["walk_forward_reports"] = _safe_walk_forward_reports(
            inst_id,
            candles,
            args,
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
    args: argparse.Namespace,
    config: EvaluationConfig,
    *,
    train_fraction: Decimal,
    validation_fraction: Decimal,
) -> list[dict]:
    if len(candles) < 3:
        return []
    reports = []
    for split in train_validation_test_split(
        candles,
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
    ):
        report = _run_strategy_report(_strategy_from_args(args, inst_id), split.candles, config)
        payload = report.to_dict()
        payload["split_name"] = split.name
        payload["role"] = "in_sample" if split.name == "train" else "out_of_sample"
        payload["baseline_comparison"] = baseline_comparison(report)
        reports.append(payload)
    return reports


def _safe_walk_forward_reports(
    inst_id: str,
    candles: list[MarketCandle],
    args: argparse.Namespace,
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
    reports = []
    for window in walk_forward_splits(
        candles,
        train_size=resolved_train_size,
        test_size=resolved_test_size,
        step_size=step_size or None,
    ):
        train_report = _run_strategy_report(
            _strategy_from_args(args, inst_id),
            window.train,
            config,
        )
        test_report = _run_strategy_report(
            _strategy_from_args(args, inst_id),
            window.test,
            config,
        )
        reports.append(
            {
                "window": window.window,
                "train": train_report.to_dict()
                | {
                    "split_name": f"walk_forward_{window.window}_train",
                    "role": "in_sample",
                    "baseline_comparison": baseline_comparison(train_report),
                },
                "test": test_report.to_dict()
                | {
                    "split_name": f"walk_forward_{window.window}_test",
                    "role": "out_of_sample",
                    "baseline_comparison": baseline_comparison(test_report),
                },
            }
        )
    return reports


def _strategy_from_args(args: argparse.Namespace, inst_id: str) -> TradingStrategy:
    configured_size = args.trade_size_eth if inst_id.startswith("ETH-") else args.trade_size_btc
    dust_threshold = "0.0001" if inst_id.startswith("ETH-") else "0.00001"
    return build_strategy(
        args.strategy,
        inst_id=inst_id,
        trade_size_btc=Decimal(configured_size),
        dust_threshold_btc=Decimal(dust_threshold),
        fast_window=args.fast_window,
        slow_window=args.slow_window,
        rsi_period=args.rsi_period,
        oversold_rsi=Decimal(args.oversold_rsi),
        overbought_rsi=Decimal(args.overbought_rsi),
        breakout_lookback=args.breakout_lookback,
        breakout_atr_period=args.breakout_atr_period,
        breakout_atr_multiplier=Decimal(args.breakout_atr_multiplier),
        breakout_min_atr_pct=Decimal(args.breakout_min_atr_pct),
    )


def _run_strategy_report(
    strategy: TradingStrategy,
    candles: list[MarketCandle] | tuple[MarketCandle, ...],
    config: EvaluationConfig,
):
    return Backtester(
        strategy,
        BacktestConfig(
            starting_equity_usdt=config.starting_equity_usdt,
            cost_model=config.cost_model,
            stop_loss_bps=config.stop_loss_bps,
            take_profit_bps=config.take_profit_bps,
            trailing_stop_bps=config.trailing_stop_bps,
            max_bars_in_trade=config.max_bars_in_trade,
        ),
    ).run(candles)


def _positive_decimal_or_none(raw: str) -> Decimal | None:
    value = Decimal(raw)
    return value if value > 0 else None


def main() -> None:
    args = build_parser().parse_args()
    reports = asyncio.run(run_backtest(args))
    print(json.dumps({"reports": reports}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
