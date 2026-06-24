from __future__ import annotations

import argparse
import asyncio
import json
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from lumiere.backtest import CostModel
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
from lumiere.paper_gate import PerformanceGateConfig
from lumiere.strategy_evaluation import (
    EvaluationConfig,
    MovingAverageCandidate,
    evaluate_parameter_grid,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Optimize Lumiere MA strategy candidates")
    parser.add_argument("--inst-id", action="append", choices=SUPPORTED_INST_IDS, dest="inst_ids")
    parser.add_argument("--bar", action="append", default=None, help="OKX bar; repeat to test many")
    parser.add_argument("--limit", type=int, default=300, help="OKX candles per page")
    parser.add_argument("--start", help="inclusive UTC start time")
    parser.add_argument("--end", help="inclusive UTC end time")
    parser.add_argument("--cache-dir", default="data/historical")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--output-dir", default="reports/strategy_optimization")
    parser.add_argument("--fast-window", default="3:12", help="comma list or inclusive range a:b")
    parser.add_argument("--slow-window", default="10:40", help="comma list or inclusive range a:b")
    parser.add_argument("--trade-size-btc", default="0.001")
    parser.add_argument("--trade-size-eth", default="0.01")
    parser.add_argument("--cooldown-seconds", type=int, default=300)
    parser.add_argument("--starting-equity-usdt", default="1000")
    parser.add_argument("--taker-fee-bps", default="10")
    parser.add_argument("--spread-bps", default="2")
    parser.add_argument("--slippage-bps", default="5")
    parser.add_argument("--market-impact-bps", default="0")
    parser.add_argument("--reject-every-n-orders", type=int, default=0)
    parser.add_argument("--train-fraction", default="0.6")
    parser.add_argument("--min-trades", type=int, default=20)
    parser.add_argument("--min-net-pnl-usdt", default="0")
    parser.add_argument("--max-drawdown-usdt", default="0")
    parser.add_argument("--min-profit-factor", default="1")
    parser.add_argument("--max-train-test-pnl-ratio", default="4")
    parser.add_argument("--min-walk-forward-windows", type=int, default=0)
    parser.add_argument("--walk-forward-train-size", type=int, default=0)
    parser.add_argument("--walk-forward-test-size", type=int, default=0)
    parser.add_argument("--min-walk-forward-pass-rate", default="0.5")
    parser.add_argument("--min-stable-neighbors", type=int, default=0)
    parser.add_argument("--parameter-stability-radius", type=int, default=1)
    return parser


async def run_optimizer(args: argparse.Namespace) -> dict[str, Any]:
    inst_ids = tuple(args.inst_ids or SUPPORTED_INST_IDS)
    bars = tuple(args.bar or ("1m",))
    start = parse_cli_datetime(args.start)
    end = parse_cli_datetime(args.end)
    cache_dir = Path(args.cache_dir)
    data_client = None if args.offline else OKXHistoricalDataClient(flag="1")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = _evaluation_config(args)

    reports: list[dict[str, Any]] = []
    accepted_configs: list[dict[str, Any]] = []
    for inst_id in inst_ids:
        for bar in bars:
            candles, dataset = await _load_or_fetch_candles(
                inst_id=inst_id,
                bar=bar,
                limit=args.limit,
                start=start,
                end=end,
                cache_dir=cache_dir,
                offline=args.offline,
                refresh_cache=args.refresh_cache,
                data_client=data_client,
            )
            candidates = _candidate_grid(
                fast_windows=_parse_int_values(args.fast_window),
                slow_windows=_parse_int_values(args.slow_window),
                trade_size=_trade_size_for(inst_id, args),
            )
            evaluations = evaluate_parameter_grid(inst_id, candles, candidates, config)
            candidate_payloads = [evaluation.to_dict() for evaluation in evaluations]
            for evaluation in evaluations:
                if evaluation.accepted:
                    accepted_configs.append(
                        {
                            "inst_id": inst_id,
                            "bar": bar,
                            "strategy": "moving_average_crossover",
                            "fast_window": evaluation.candidate.fast_window,
                            "slow_window": evaluation.candidate.slow_window,
                            "trade_size_base": str(evaluation.candidate.trade_size_base),
                            "cooldown_seconds": args.cooldown_seconds,
                            "source_report": "optimizer",
                        }
                    )
            reports.append(
                {
                    "inst_id": inst_id,
                    "bar": bar,
                    "dataset": None if dataset is None else dataset.to_json_dict(),
                    "candidate_count": len(candidate_payloads),
                    "accepted_count": sum(1 for item in candidate_payloads if item["accepted"]),
                    "candidates": candidate_payloads,
                }
            )

    payload = {
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "criteria": _criteria_payload(args),
        "reports": reports,
        "accepted_configs": accepted_configs,
    }
    report_path = output_dir / "optimizer_report.json"
    accepted_path = output_dir / "accepted_candidates.json"
    report_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    accepted_path.write_text(
        json.dumps({"accepted_configs": accepted_configs}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    payload["artifacts"] = {
        "report_path": str(report_path),
        "accepted_candidates_path": str(accepted_path),
    }
    return payload


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
) -> tuple[list[MarketCandle], Any]:
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


def _evaluation_config(args: argparse.Namespace) -> EvaluationConfig:
    max_drawdown = Decimal(args.max_drawdown_usdt)
    min_profit_factor = args.min_profit_factor.strip().lower()
    return EvaluationConfig(
        train_fraction=Decimal(args.train_fraction),
        starting_equity_usdt=Decimal(args.starting_equity_usdt),
        cost_model=CostModel(
            taker_fee_bps=Decimal(args.taker_fee_bps),
            spread_bps=Decimal(args.spread_bps),
            slippage_bps=Decimal(args.slippage_bps),
            market_impact_bps=Decimal(args.market_impact_bps),
            reject_every_n_orders=args.reject_every_n_orders,
        ),
        performance_gate=PerformanceGateConfig(
            min_trades=args.min_trades,
            min_net_pnl_usdt=Decimal(args.min_net_pnl_usdt),
            max_drawdown_usdt=max_drawdown if max_drawdown > 0 else None,
            min_profit_factor=(
                None
                if min_profit_factor in {"", "none", "null", "0"}
                else Decimal(min_profit_factor)
            ),
        ),
        max_train_test_pnl_ratio=Decimal(args.max_train_test_pnl_ratio),
        min_walk_forward_windows=args.min_walk_forward_windows,
        walk_forward_train_size=args.walk_forward_train_size,
        walk_forward_test_size=args.walk_forward_test_size,
        min_walk_forward_pass_rate=Decimal(args.min_walk_forward_pass_rate),
        min_stable_neighbors=args.min_stable_neighbors,
        parameter_stability_radius=args.parameter_stability_radius,
    )


def _candidate_grid(
    *,
    fast_windows: tuple[int, ...],
    slow_windows: tuple[int, ...],
    trade_size: Decimal,
) -> tuple[MovingAverageCandidate, ...]:
    candidates = [
        MovingAverageCandidate(fast, slow, trade_size)
        for fast in fast_windows
        for slow in slow_windows
        if slow > fast
    ]
    if not candidates:
        raise ValueError("no valid MA candidates; every slow window must be greater than fast")
    return tuple(candidates)


def _parse_int_values(raw: str) -> tuple[int, ...]:
    values: list[int] = []
    for part in raw.split(","):
        token = part.strip()
        if not token:
            continue
        if ":" in token:
            start, end = token.split(":", maxsplit=1)
            values.extend(range(int(start), int(end) + 1))
        else:
            values.append(int(token))
    unique = tuple(sorted(set(values)))
    if not unique:
        raise ValueError("at least one window value is required")
    return unique


def _trade_size_for(inst_id: str, args: argparse.Namespace) -> Decimal:
    return Decimal(args.trade_size_eth if inst_id.startswith("ETH-") else args.trade_size_btc)


def _criteria_payload(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "fast_window": args.fast_window,
        "slow_window": args.slow_window,
        "cooldown_seconds": args.cooldown_seconds,
        "min_trades": args.min_trades,
        "min_net_pnl_usdt": args.min_net_pnl_usdt,
        "max_drawdown_usdt": args.max_drawdown_usdt,
        "min_profit_factor": args.min_profit_factor,
        "max_train_test_pnl_ratio": args.max_train_test_pnl_ratio,
        "min_walk_forward_windows": args.min_walk_forward_windows,
        "min_walk_forward_pass_rate": args.min_walk_forward_pass_rate,
        "min_stable_neighbors": args.min_stable_neighbors,
        "parameter_stability_radius": args.parameter_stability_radius,
        "require_baseline_outperformance": True,
        "cost_model": {
            "taker_fee_bps": args.taker_fee_bps,
            "spread_bps": args.spread_bps,
            "slippage_bps": args.slippage_bps,
            "market_impact_bps": args.market_impact_bps,
            "reject_every_n_orders": args.reject_every_n_orders,
        },
    }


def main() -> None:
    payload = asyncio.run(run_optimizer(build_parser().parse_args()))
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
