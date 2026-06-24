from __future__ import annotations

import argparse
import asyncio
import json
from datetime import UTC, datetime
from decimal import Decimal
from itertools import product
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
from lumiere.risk import RiskConfig
from lumiere.strategy_evaluation import (
    EvaluationConfig,
    OptimizerCandidate,
    StrategyCandidate,
    candidate_to_dict,
    evaluate_parameter_grid,
)
from lumiere.strategy_factory import STRATEGY_NAMES


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Optimize Lumiere strategy candidates")
    parser.add_argument("--inst-id", action="append", choices=SUPPORTED_INST_IDS, dest="inst_ids")
    parser.add_argument("--bar", action="append", default=None, help="OKX bar; repeat to test many")
    parser.add_argument(
        "--strategy",
        action="append",
        choices=STRATEGY_NAMES,
        dest="strategies",
        help="strategy to optimize; repeat or omit to test all",
    )
    parser.add_argument("--limit", type=int, default=300, help="OKX candles per page")
    parser.add_argument("--start", help="inclusive UTC start time")
    parser.add_argument("--end", help="inclusive UTC end time")
    parser.add_argument("--cache-dir", default="data/historical")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--output-dir", default="reports/strategy_optimization")
    parser.add_argument("--fast-window", default="3:12", help="comma list or inclusive range a:b")
    parser.add_argument("--slow-window", default="10:40", help="comma list or inclusive range a:b")
    parser.add_argument("--rsi-period", default="7,14,21")
    parser.add_argument("--oversold-rsi", default="25,30,35")
    parser.add_argument("--overbought-rsi", default="65,70,75")
    parser.add_argument("--breakout-lookback", default="10,20,40")
    parser.add_argument("--breakout-atr-period", default="7,14,21")
    parser.add_argument("--breakout-atr-multiplier", default="0.25,0.5,1")
    parser.add_argument("--breakout-min-atr-pct", default="0.0005,0.001,0.002")
    parser.add_argument("--stop-loss-bps", default="none")
    parser.add_argument("--take-profit-bps", default="none")
    parser.add_argument("--trailing-stop-bps", default="none")
    parser.add_argument("--max-bars-in-trade", default="none")
    parser.add_argument("--trade-size-btc", default="0.001")
    parser.add_argument("--trade-size-eth", default="0.01")
    parser.add_argument("--cooldown-seconds", type=int, default=300)
    parser.add_argument("--starting-equity-usdt", default="1000")
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
    parser.add_argument("--min-calibration-signals", type=int, default=3)
    parser.add_argument("--min-expected-edge-bps", default="0")
    parser.add_argument("--expectancy-horizon-bars", type=int, default=1)
    return parser


async def run_optimizer(args: argparse.Namespace) -> dict[str, Any]:
    inst_ids = tuple(args.inst_ids or SUPPORTED_INST_IDS)
    bars = tuple(args.bar or ("1m",))
    strategies = tuple(getattr(args, "strategies", None) or STRATEGY_NAMES)
    start = parse_cli_datetime(args.start)
    end = parse_cli_datetime(args.end)
    cache_dir = Path(args.cache_dir)
    data_client = None if args.offline else OKXHistoricalDataClient(flag="1")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = _evaluation_config(args, inst_ids=inst_ids)

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
                strategies=strategies,
                fast_windows=_parse_int_values(args.fast_window),
                slow_windows=_parse_int_values(args.slow_window),
                rsi_periods=_parse_int_values(getattr(args, "rsi_period", "14")),
                oversold_values=_parse_decimal_values(getattr(args, "oversold_rsi", "30")),
                overbought_values=_parse_decimal_values(getattr(args, "overbought_rsi", "70")),
                breakout_lookbacks=_parse_int_values(getattr(args, "breakout_lookback", "20")),
                breakout_atr_periods=_parse_int_values(
                    getattr(args, "breakout_atr_period", "14")
                ),
                breakout_atr_multipliers=_parse_decimal_values(
                    getattr(args, "breakout_atr_multiplier", "0.5")
                ),
                breakout_min_atr_pcts=_parse_decimal_values(
                    getattr(args, "breakout_min_atr_pct", "0.001")
                ),
                stop_loss_values=_parse_optional_decimal_values(
                    getattr(args, "stop_loss_bps", "none")
                ),
                take_profit_values=_parse_optional_decimal_values(
                    getattr(args, "take_profit_bps", "none")
                ),
                trailing_stop_values=_parse_optional_decimal_values(
                    getattr(args, "trailing_stop_bps", "none")
                ),
                max_bars_values=_parse_optional_int_values(
                    getattr(args, "max_bars_in_trade", "none")
                ),
                trade_size=_trade_size_for(inst_id, args),
            )
            evaluations = evaluate_parameter_grid(inst_id, candles, candidates, config)
            candidate_payloads = [evaluation.to_dict() for evaluation in evaluations]
            for evaluation in evaluations:
                if evaluation.accepted:
                    accepted_configs.append(
                        _accepted_config_payload(
                            inst_id=inst_id,
                            bar=bar,
                            evaluation=evaluation,
                            cooldown_seconds=args.cooldown_seconds,
                        )
                    )
            reports.append(
                {
                    "inst_id": inst_id,
                    "bar": bar,
                    "dataset": None if dataset is None else dataset.to_json_dict(),
                    "strategies": list(strategies),
                    "candidate_count": len(candidate_payloads),
                    "accepted_count": sum(1 for item in candidate_payloads if item["accepted"]),
                    "candidates": candidate_payloads,
                }
            )

    payload = {
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "criteria": _criteria_payload(args, strategies=strategies),
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


def _evaluation_config(args: argparse.Namespace, *, inst_ids: tuple[str, ...]) -> EvaluationConfig:
    max_drawdown = Decimal(args.max_drawdown_usdt)
    min_profit_factor = args.min_profit_factor.strip().lower()
    return EvaluationConfig(
        train_fraction=Decimal(args.train_fraction),
        starting_equity_usdt=Decimal(args.starting_equity_usdt),
        cost_model=CostModel(
            taker_fee_bps=Decimal(args.taker_fee_bps),
            maker_fee_bps=Decimal(getattr(args, "maker_fee_bps", "2")),
            spread_bps=Decimal(args.spread_bps),
            slippage_bps=Decimal(args.slippage_bps),
            market_impact_bps=Decimal(args.market_impact_bps),
            reject_every_n_orders=args.reject_every_n_orders,
            execution_policy=getattr(args, "execution_policy", "market"),
            marketable_limit_buffer_bps=Decimal(
                getattr(args, "marketable_limit_buffer_bps", "1")
            ),
            post_only_offset_bps=Decimal(getattr(args, "post_only_offset_bps", "0")),
            maker_timeout_bars=getattr(args, "maker_timeout_bars", 1),
            maker_fill_fraction=Decimal(getattr(args, "maker_fill_fraction", "1")),
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
        min_calibration_signals=getattr(args, "min_calibration_signals", 1),
        min_expected_edge_bps=Decimal(getattr(args, "min_expected_edge_bps", "0")),
        expectancy_horizon_bars=getattr(args, "expectancy_horizon_bars", 1),
        risk_config=RiskConfig(
            allowed_inst_ids=inst_ids,
            cooldown_seconds=args.cooldown_seconds,
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


def _candidate_grid(
    *,
    strategies: tuple[str, ...],
    fast_windows: tuple[int, ...],
    slow_windows: tuple[int, ...],
    rsi_periods: tuple[int, ...],
    oversold_values: tuple[Decimal, ...],
    overbought_values: tuple[Decimal, ...],
    breakout_lookbacks: tuple[int, ...],
    breakout_atr_periods: tuple[int, ...],
    breakout_atr_multipliers: tuple[Decimal, ...],
    breakout_min_atr_pcts: tuple[Decimal, ...],
    stop_loss_values: tuple[Decimal | None, ...],
    take_profit_values: tuple[Decimal | None, ...],
    trailing_stop_values: tuple[Decimal | None, ...],
    max_bars_values: tuple[int | None, ...],
    trade_size: Decimal,
) -> tuple[OptimizerCandidate, ...]:
    exit_combinations = tuple(
        product(stop_loss_values, take_profit_values, trailing_stop_values, max_bars_values)
    )
    candidates: list[OptimizerCandidate] = []
    if "moving_average_crossover" in strategies:
        for fast, slow, exits in product(fast_windows, slow_windows, exit_combinations):
            if slow <= fast:
                continue
            stop, take, trailing, max_bars = exits
            candidates.append(
                StrategyCandidate(
                    "moving_average_crossover",
                    trade_size,
                    fast_window=fast,
                    slow_window=slow,
                    stop_loss_bps=stop,
                    take_profit_bps=take,
                    trailing_stop_bps=trailing,
                    max_bars_in_trade=max_bars,
                )
            )
    if "rsi_mean_reversion" in strategies:
        for period, oversold, overbought, exits in product(
            rsi_periods,
            oversold_values,
            overbought_values,
            exit_combinations,
        ):
            if oversold >= overbought:
                continue
            stop, take, trailing, max_bars = exits
            candidates.append(
                StrategyCandidate(
                    "rsi_mean_reversion",
                    trade_size,
                    rsi_period=period,
                    oversold_rsi=oversold,
                    overbought_rsi=overbought,
                    stop_loss_bps=stop,
                    take_profit_bps=take,
                    trailing_stop_bps=trailing,
                    max_bars_in_trade=max_bars,
                )
            )
    if "volatility_breakout" in strategies:
        for lookback, atr_period, multiplier, min_atr_pct, exits in product(
            breakout_lookbacks,
            breakout_atr_periods,
            breakout_atr_multipliers,
            breakout_min_atr_pcts,
            exit_combinations,
        ):
            stop, take, trailing, max_bars = exits
            candidates.append(
                StrategyCandidate(
                    "volatility_breakout",
                    trade_size,
                    breakout_lookback=lookback,
                    breakout_atr_period=atr_period,
                    breakout_atr_multiplier=multiplier,
                    breakout_min_atr_pct=min_atr_pct,
                    stop_loss_bps=stop,
                    take_profit_bps=take,
                    trailing_stop_bps=trailing,
                    max_bars_in_trade=max_bars,
                )
            )
    if not candidates:
        raise ValueError("no valid optimizer candidates")
    return tuple(candidates)


def _accepted_config_payload(
    *,
    inst_id: str,
    bar: str,
    evaluation,
    cooldown_seconds: int,
) -> dict[str, Any]:
    candidate = candidate_to_dict(evaluation.candidate)
    edge = evaluation.expectancy.average_forward_return_after_cost_bps
    strategy_name = candidate["strategy"]
    env = {
        "STRATEGY_NAME": strategy_name,
        "OKX_INST_ID": inst_id,
        "ENGINE_CANDLE_BAR": bar,
    }
    env.update(_env_parameters(candidate))
    payload = {
        "inst_id": inst_id,
        "bar": bar,
        "strategy": strategy_name,
        "candidate": candidate,
        "trade_size_base": candidate["trade_size_base"],
        "cooldown_seconds": cooldown_seconds,
        "expected_edge_bps": None if edge is None else str(edge),
        "expected_edge_source": "historical_forward_return_after_costs",
        "expectancy_calibration": evaluation.expectancy.to_dict(),
        "optimizer_passed": True,
        "anti_overfit_gates": {
            "performance_gate": evaluation.gate.reason,
            "walk_forward_gate_count": len(evaluation.walk_forward_gates),
            "train_test_divergence_checked": True,
            "baseline_outperformance_checked": True,
            "parameter_stability_checked": True,
        },
        "rank_metrics": evaluation.to_dict()["rank_metrics"],
        "env": env,
        "source_report": "optimizer",
    }
    payload.update(candidate)
    return payload


def _env_parameters(candidate: dict[str, Any]) -> dict[str, str]:
    mapping = {
        "fast_window": "STRATEGY_FAST_WINDOW",
        "slow_window": "STRATEGY_SLOW_WINDOW",
        "rsi_period": "STRATEGY_RSI_PERIOD",
        "oversold_rsi": "STRATEGY_OVERSOLD_RSI",
        "overbought_rsi": "STRATEGY_OVERBOUGHT_RSI",
        "breakout_lookback": "STRATEGY_BREAKOUT_LOOKBACK",
        "breakout_atr_period": "STRATEGY_BREAKOUT_ATR_PERIOD",
        "breakout_atr_multiplier": "STRATEGY_BREAKOUT_ATR_MULTIPLIER",
        "breakout_min_atr_pct": "STRATEGY_BREAKOUT_MIN_ATR_PCT",
        "stop_loss_bps": "LIVE_STOP_LOSS_BPS",
        "take_profit_bps": "LIVE_TAKE_PROFIT_BPS",
        "trailing_stop_bps": "LIVE_TRAILING_STOP_BPS",
        "max_bars_in_trade": "LIVE_MAX_BARS_IN_TRADE",
    }
    return {env: str(candidate[key]) for key, env in mapping.items() if key in candidate}


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
        raise ValueError("at least one integer value is required")
    return unique


def _parse_optional_int_values(raw: str) -> tuple[int | None, ...]:
    if _is_none_grid(raw):
        return (None,)
    values: list[int | None] = []
    for token in _split_grid(raw):
        if token.lower() in {"none", "null", "off", "0"}:
            values.append(None)
        elif ":" in token:
            start, end = token.split(":", maxsplit=1)
            values.extend(range(int(start), int(end) + 1))
        else:
            values.append(int(token))
    return tuple(dict.fromkeys(values))


def _parse_decimal_values(raw: str) -> tuple[Decimal, ...]:
    values = tuple(Decimal(token) for token in _split_grid(raw))
    if not values:
        raise ValueError("at least one decimal value is required")
    return tuple(sorted(set(values)))


def _parse_optional_decimal_values(raw: str) -> tuple[Decimal | None, ...]:
    if _is_none_grid(raw):
        return (None,)
    values: list[Decimal | None] = []
    for token in _split_grid(raw):
        if token.lower() in {"none", "null", "off", "0"}:
            values.append(None)
        else:
            values.append(Decimal(token))
    return tuple(dict.fromkeys(values))


def _split_grid(raw: str) -> tuple[str, ...]:
    return tuple(token.strip() for token in raw.split(",") if token.strip())


def _is_none_grid(raw: str) -> bool:
    tokens = _split_grid(raw)
    return not tokens or all(token.lower() in {"none", "null", "off", "0"} for token in tokens)


def _trade_size_for(inst_id: str, args: argparse.Namespace) -> Decimal:
    return Decimal(args.trade_size_eth if inst_id.startswith("ETH-") else args.trade_size_btc)


def _criteria_payload(args: argparse.Namespace, *, strategies: tuple[str, ...]) -> dict[str, Any]:
    return {
        "strategies": list(strategies),
        "fast_window": args.fast_window,
        "slow_window": args.slow_window,
        "rsi_period": getattr(args, "rsi_period", "14"),
        "oversold_rsi": getattr(args, "oversold_rsi", "30"),
        "overbought_rsi": getattr(args, "overbought_rsi", "70"),
        "breakout_lookback": getattr(args, "breakout_lookback", "20"),
        "breakout_atr_period": getattr(args, "breakout_atr_period", "14"),
        "breakout_atr_multiplier": getattr(args, "breakout_atr_multiplier", "0.5"),
        "breakout_min_atr_pct": getattr(args, "breakout_min_atr_pct", "0.001"),
        "stop_loss_bps": getattr(args, "stop_loss_bps", "none"),
        "take_profit_bps": getattr(args, "take_profit_bps", "none"),
        "trailing_stop_bps": getattr(args, "trailing_stop_bps", "none"),
        "max_bars_in_trade": getattr(args, "max_bars_in_trade", "none"),
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
        "min_calibration_signals": getattr(args, "min_calibration_signals", 1),
        "min_expected_edge_bps": getattr(args, "min_expected_edge_bps", "0"),
        "expectancy_horizon_bars": getattr(args, "expectancy_horizon_bars", 1),
        "require_baseline_outperformance": True,
        "cost_model": {
            "taker_fee_bps": args.taker_fee_bps,
            "maker_fee_bps": getattr(args, "maker_fee_bps", "2"),
            "spread_bps": args.spread_bps,
            "slippage_bps": args.slippage_bps,
            "market_impact_bps": args.market_impact_bps,
            "reject_every_n_orders": args.reject_every_n_orders,
            "execution_policy": getattr(args, "execution_policy", "market"),
            "marketable_limit_buffer_bps": getattr(args, "marketable_limit_buffer_bps", "1"),
            "post_only_offset_bps": getattr(args, "post_only_offset_bps", "0"),
            "maker_timeout_bars": getattr(args, "maker_timeout_bars", 1),
            "maker_fill_fraction": getattr(args, "maker_fill_fraction", "1"),
        },
    }


def main() -> None:
    payload = asyncio.run(run_optimizer(build_parser().parse_args()))
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
