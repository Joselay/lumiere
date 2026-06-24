from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from lumiere.backtest import BacktestConfig, Backtester, CostModel
from lumiere.config import SUPPORTED_INST_IDS
from lumiere.historical_data import (
    HistoricalCandleRequest,
    HistoricalDatasetMetadata,
    OKXHistoricalDataClient,
    dataset_exists,
    load_dataset,
    parse_cli_datetime,
    save_dataset,
)
from lumiere.models import MarketCandle
from lumiere.paper_gate import PerformanceGateConfig, assess_report
from lumiere.risk import RiskConfig
from lumiere.strategy import TradingStrategy
from lumiere.strategy_evaluation import (
    EvaluationConfig,
    baseline_comparison,
    rank_metrics,
    train_validation_test_split,
    walk_forward_splits,
)
from lumiere.strategy_factory import STRATEGY_NAMES, build_strategy

DEFAULT_BARS = ("1m", "5m", "15m", "1H")
DEFAULT_STRATEGIES = STRATEGY_NAMES


@dataclass(frozen=True, slots=True)
class RegimeWindow:
    name: str
    candles: tuple[MarketCandle, ...]
    labels: tuple[str, ...]
    start: datetime
    end: datetime

    @property
    def signature(self) -> str:
        return "+".join(self.labels)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a long-horizon BTC/ETH multi-regime evidence matrix"
    )
    parser.add_argument("--inst-id", action="append", choices=SUPPORTED_INST_IDS, dest="inst_ids")
    parser.add_argument("--bar", action="append", default=None, help="OKX bar; repeat to test many")
    parser.add_argument("--strategy", action="append", choices=STRATEGY_NAMES, dest="strategies")
    parser.add_argument("--limit", type=int, default=300, help="OKX candles per page")
    parser.add_argument("--start", help="inclusive UTC start time")
    parser.add_argument("--end", help="inclusive UTC end time")
    parser.add_argument("--cache-dir", default="data/historical")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--output-dir", default="reports/evidence_matrix")
    parser.add_argument("--min-span-days", type=int, default=180)
    parser.add_argument("--max-span-days", type=int, default=730)
    parser.add_argument("--min-regime-passes", type=int, default=2)
    parser.add_argument("--regime-window-size", type=int, default=0)
    parser.add_argument("--min-regime-candles", type=int, default=30)
    parser.add_argument("--trend-threshold-bps", default="100")
    parser.add_argument("--high-volatility-bps", default="50")
    parser.add_argument("--drawdown-threshold-bps", default="300")
    parser.add_argument("--event-range-bps", default="250")
    parser.add_argument("--starting-equity-usdt", default="1000")
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
    parser.add_argument("--spread-bps", default="2")
    parser.add_argument("--slippage-bps", default="5")
    parser.add_argument("--market-impact-bps", default="0")
    parser.add_argument("--reject-every-n-orders", type=int, default=0)
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
    parser.add_argument("--min-trades", type=int, default=20)
    parser.add_argument("--min-net-pnl-usdt", default="0")
    parser.add_argument("--max-drawdown-usdt", default="0")
    parser.add_argument("--min-profit-factor", default="1")
    parser.add_argument("--allow-baseline-underperformance", action="store_true")
    return parser


async def run_evidence_matrix(args: argparse.Namespace) -> dict[str, Any]:
    inst_ids = tuple(args.inst_ids or SUPPORTED_INST_IDS)
    bars = tuple(args.bar or DEFAULT_BARS)
    strategies = tuple(args.strategies or DEFAULT_STRATEGIES)
    start = parse_cli_datetime(args.start)
    end = parse_cli_datetime(args.end)
    cache_dir = Path(args.cache_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_client = None if args.offline else OKXHistoricalDataClient(flag="1")
    evaluation_config = _evaluation_config(args, inst_ids=inst_ids)
    gate_config = _gate_config(args)

    matrix: list[dict[str, Any]] = []
    accepted_configs: list[dict[str, Any]] = []
    for inst_id in inst_ids:
        for bar in bars:
            candles, dataset_metadata = await _load_or_fetch_candles(
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
            dataset_horizon = _dataset_horizon_payload(
                dataset_metadata,
                requested_start=start,
                requested_end=end,
                min_span_days=args.min_span_days,
                max_span_days=args.max_span_days,
            )
            regime_windows = regime_windows_for(
                candles,
                window_size=args.regime_window_size or _default_regime_window_size(len(candles)),
                min_candles=args.min_regime_candles,
                trend_threshold_bps=Decimal(args.trend_threshold_bps),
                high_volatility_bps=Decimal(args.high_volatility_bps),
                drawdown_threshold_bps=Decimal(args.drawdown_threshold_bps),
                event_range_bps=Decimal(args.event_range_bps),
            )
            for strategy_name in strategies:
                row = _build_matrix_row(
                    inst_id=inst_id,
                    bar=bar,
                    strategy_name=strategy_name,
                    candles=candles,
                    dataset_metadata=dataset_metadata,
                    dataset_horizon=dataset_horizon,
                    regime_windows=regime_windows,
                    args=args,
                    evaluation_config=evaluation_config,
                    gate_config=gate_config,
                )
                matrix.append(row)
                if row["accepted"]:
                    accepted_configs.append(_accepted_config(row, args))

    payload = {
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "coverage_plan": {
            "inst_ids": list(inst_ids),
            "bars": list(bars),
            "strategies": list(strategies),
            "requested_start": None if start is None else start.isoformat(),
            "requested_end": None if end is None else end.isoformat(),
            "min_span_days": args.min_span_days,
            "max_span_days": args.max_span_days,
            "min_regime_passes": args.min_regime_passes,
        },
        "criteria": _criteria_payload(args),
        "matrix": matrix,
        "accepted_configs": accepted_configs,
    }
    matrix_path = output_dir / "evidence_matrix.json"
    accepted_path = output_dir / "accepted_configs.json"
    matrix_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    accepted_path.write_text(
        json.dumps({"accepted_configs": accepted_configs}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    payload["artifacts"] = {
        "matrix_path": str(matrix_path),
        "accepted_configs_path": str(accepted_path),
    }
    return payload


def regime_windows_for(
    candles: list[MarketCandle] | tuple[MarketCandle, ...],
    *,
    window_size: int,
    min_candles: int,
    trend_threshold_bps: Decimal,
    high_volatility_bps: Decimal,
    drawdown_threshold_bps: Decimal,
    event_range_bps: Decimal,
) -> tuple[RegimeWindow, ...]:
    ordered = tuple(sorted(candles, key=lambda candle: candle.ts))
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    if min_candles <= 0:
        raise ValueError("min_candles must be positive")
    windows: list[RegimeWindow] = []
    window_number = 1
    for index in range(0, len(ordered), window_size):
        window_candles = ordered[index : index + window_size]
        if len(window_candles) < min_candles:
            continue
        windows.append(
            RegimeWindow(
                name=f"regime_{window_number}",
                candles=window_candles,
                labels=label_market_regime(
                    window_candles,
                    trend_threshold_bps=trend_threshold_bps,
                    high_volatility_bps=high_volatility_bps,
                    drawdown_threshold_bps=drawdown_threshold_bps,
                    event_range_bps=event_range_bps,
                ),
                start=window_candles[0].ts,
                end=window_candles[-1].ts,
            )
        )
        window_number += 1
    return tuple(windows)


def label_market_regime(
    candles: tuple[MarketCandle, ...],
    *,
    trend_threshold_bps: Decimal,
    high_volatility_bps: Decimal,
    drawdown_threshold_bps: Decimal,
    event_range_bps: Decimal,
) -> tuple[str, ...]:
    if len(candles) < 2:
        return ("insufficient_regime_data",)
    start_price = candles[0].open
    end_price = candles[-1].close
    basis_price = start_price if start_price != 0 else Decimal("1")
    return_bps = (end_price - start_price) / basis_price * Decimal("10000")
    absolute_return_bps = abs(return_bps)
    average_range_bps = _average_range_bps(candles)
    max_drawdown_bps = _max_drawdown_bps(candles)
    max_range_bps = max(_range_bps(candle) for candle in candles)
    volume_positive = sum(1 for candle in candles if candle.volume > 0)

    direction = "bull" if return_bps > 0 else "bear" if return_bps < 0 else "flat"
    trend = "trend" if absolute_return_bps >= trend_threshold_bps else "range"
    volatility = "high_volatility" if average_range_bps >= high_volatility_bps else "low_volatility"
    drawdown = "drawdown" if max_drawdown_bps >= drawdown_threshold_bps else "shallow_drawdown"
    liquidity = (
        "spread_liquidity_ok"
        if volume_positive >= max(1, len(candles) // 2)
        else "low_liquidity_volume_proxy"
    )
    event = "event_period" if max_range_bps >= event_range_bps else "normal_period"
    return (direction, trend, volatility, drawdown, liquidity, event)


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
) -> tuple[list[MarketCandle], HistoricalDatasetMetadata]:
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
    candles = await data_client.fetch_candles_paginated(request, start=start, end=end)
    dataset = save_dataset(
        cache_dir,
        inst_id=inst_id,
        bar=bar,
        candles=candles,
        start=start,
        end=end,
    )
    return list(dataset.candles), dataset.metadata


def _build_matrix_row(
    *,
    inst_id: str,
    bar: str,
    strategy_name: str,
    candles: list[MarketCandle],
    dataset_metadata: HistoricalDatasetMetadata,
    dataset_horizon: dict[str, Any],
    regime_windows: tuple[RegimeWindow, ...],
    args: argparse.Namespace,
    evaluation_config: EvaluationConfig,
    gate_config: PerformanceGateConfig,
) -> dict[str, Any]:
    strategy = _strategy_from_args(args, strategy_name, inst_id)
    full_report = _run_strategy_report(strategy, candles, evaluation_config)
    split_reports = _split_reports(strategy_name, inst_id, candles, args, evaluation_config)
    walk_forward_reports = _walk_forward_reports(
        strategy_name,
        inst_id,
        candles,
        args,
        evaluation_config,
    )
    regime_reports = _regime_reports(
        strategy_name,
        inst_id,
        regime_windows,
        args,
        evaluation_config,
        gate_config,
    )
    acceptance = _acceptance_payload(
        full_report=full_report,
        regime_reports=regime_reports,
        dataset_horizon_ok=bool(dataset_horizon["ok"]),
        min_regime_passes=args.min_regime_passes,
        gate_config=gate_config,
        require_baseline_outperformance=not args.allow_baseline_underperformance,
    )
    return {
        "inst_id": inst_id,
        "bar": bar,
        "strategy": strategy_name,
        "parameters": strategy.describe(),
        "dataset": dataset_metadata.to_json_dict(),
        "dataset_horizon": dataset_horizon,
        "market_regimes": _regime_summary(regime_windows),
        "full_period": _report_summary(full_report),
        "split_reports": split_reports,
        "walk_forward_reports": walk_forward_reports,
        "regime_reports": regime_reports,
        "accepted": acceptance["accepted"],
        "rejection_reason": acceptance["rejection_reason"],
        "acceptance_checks": acceptance["checks"],
    }


def _split_reports(
    strategy_name: str,
    inst_id: str,
    candles: list[MarketCandle],
    args: argparse.Namespace,
    config: EvaluationConfig,
) -> list[dict[str, Any]]:
    if len(candles) < 3:
        return []
    payloads: list[dict[str, Any]] = []
    for split in train_validation_test_split(
        candles,
        train_fraction=Decimal(args.train_fraction),
        validation_fraction=Decimal(args.validation_fraction),
    ):
        report = _run_strategy_report(
            _strategy_from_args(args, strategy_name, inst_id),
            split.candles,
            config,
        )
        summary = _report_summary(report)
        summary["split_name"] = split.name
        summary["role"] = "in_sample" if split.name == "train" else "out_of_sample"
        payloads.append(summary)
    return payloads


def _walk_forward_reports(
    strategy_name: str,
    inst_id: str,
    candles: list[MarketCandle],
    args: argparse.Namespace,
    config: EvaluationConfig,
) -> list[dict[str, Any]]:
    if args.no_walk_forward or len(candles) < 5:
        return []
    train_size = args.walk_forward_train_size or max(2, int(len(candles) * 0.6))
    test_size = args.walk_forward_test_size or max(1, int(len(candles) * 0.2))
    if train_size + test_size > len(candles):
        return []
    payloads: list[dict[str, Any]] = []
    for window in walk_forward_splits(
        candles,
        train_size=train_size,
        test_size=test_size,
        step_size=args.walk_forward_step_size or None,
    ):
        train_report = _run_strategy_report(
            _strategy_from_args(args, strategy_name, inst_id),
            window.train,
            config,
        )
        test_report = _run_strategy_report(
            _strategy_from_args(args, strategy_name, inst_id),
            window.test,
            config,
        )
        payloads.append(
            {
                "window": window.window,
                "train": _report_summary(train_report)
                | {
                    "split_name": f"walk_forward_{window.window}_train",
                    "role": "in_sample",
                },
                "test": _report_summary(test_report)
                | {
                    "split_name": f"walk_forward_{window.window}_test",
                    "role": "out_of_sample",
                },
            }
        )
    return payloads


def _regime_reports(
    strategy_name: str,
    inst_id: str,
    regime_windows: tuple[RegimeWindow, ...],
    args: argparse.Namespace,
    config: EvaluationConfig,
    gate_config: PerformanceGateConfig,
) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for window in regime_windows:
        report = _run_strategy_report(
            _strategy_from_args(args, strategy_name, inst_id),
            window.candles,
            config,
        )
        gate = assess_report(report, gate_config)
        payloads.append(
            {
                "window": window.name,
                "start": window.start.isoformat(),
                "end": window.end.isoformat(),
                "labels": list(window.labels),
                "signature": window.signature,
                "gate": {"allowed": gate.allowed, "reason": gate.reason},
                "summary": _report_summary(report),
            }
        )
    return payloads


def _acceptance_payload(
    *,
    full_report,
    regime_reports: list[dict[str, Any]],
    dataset_horizon_ok: bool,
    min_regime_passes: int,
    gate_config: PerformanceGateConfig,
    require_baseline_outperformance: bool,
) -> dict[str, Any]:
    full_gate = assess_report(full_report, gate_config)
    baseline = baseline_comparison(full_report)
    no_trade_delta = Decimal(baseline["net_pnl_minus_no_trade_usdt"])
    buy_hold_delta = Decimal(baseline["net_pnl_minus_buy_and_hold_usdt"])
    baseline_passed = not require_baseline_outperformance or (
        no_trade_delta > 0 and buy_hold_delta > 0
    )
    passing_regime_signatures = sorted(
        {
            report["signature"]
            for report in regime_reports
            if report["gate"]["allowed"]
            and Decimal(report["summary"]["rank_metrics"]["net_pnl_usdt"])
            > gate_config.min_net_pnl_usdt
        }
    )
    regime_pass_count = len(passing_regime_signatures)
    checks = {
        "dataset_horizon_ok": dataset_horizon_ok,
        "full_period_gate": {"allowed": full_gate.allowed, "reason": full_gate.reason},
        "baseline_outperformance_passed": baseline_passed,
        "passing_regime_count": regime_pass_count,
        "passing_regime_signatures": passing_regime_signatures,
        "min_regime_passes": min_regime_passes,
    }
    rejection_reason = None
    if not dataset_horizon_ok:
        rejection_reason = "dataset_horizon_short"
    elif not full_gate.allowed:
        rejection_reason = full_gate.reason
    elif not baseline_passed:
        rejection_reason = "baseline_not_beaten"
    elif regime_pass_count < min_regime_passes:
        rejection_reason = "insufficient_regime_passes"
    return {
        "accepted": rejection_reason is None,
        "rejection_reason": rejection_reason,
        "checks": checks,
    }


def _report_summary(report) -> dict[str, Any]:
    return {
        "period_start": report.period_start.isoformat(),
        "period_end": report.period_end.isoformat(),
        "execution_timing": report.execution_timing,
        "rank_metrics": rank_metrics(report),
        "risk_rejection_count": report.risk_rejection_count,
        "risk_rejections": report.risk_rejections,
        "rejected_order_count": report.rejected_order_count,
        "baseline_comparison": baseline_comparison(report),
    }


def _regime_summary(regime_windows: tuple[RegimeWindow, ...]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    signatures: dict[str, int] = {}
    for window in regime_windows:
        signatures[window.signature] = signatures.get(window.signature, 0) + 1
        for label in window.labels:
            counts[label] = counts.get(label, 0) + 1
    return {
        "window_count": len(regime_windows),
        "label_counts": dict(sorted(counts.items())),
        "signature_counts": dict(sorted(signatures.items())),
    }


def _dataset_horizon_payload(
    metadata: HistoricalDatasetMetadata,
    *,
    requested_start: datetime | None,
    requested_end: datetime | None,
    min_span_days: int,
    max_span_days: int,
) -> dict[str, Any]:
    start = requested_start or metadata.start
    end = requested_end or metadata.end
    span_days = None
    if start is not None and end is not None:
        span_days = (end - start).total_seconds() / 86400
    too_short = span_days is None or span_days < min_span_days
    too_long = span_days is not None and max_span_days > 0 and span_days > max_span_days
    return {
        "ok": not too_short and not too_long,
        "span_days": span_days,
        "min_span_days": min_span_days,
        "max_span_days": max_span_days,
        "too_short": too_short,
        "too_long": too_long,
        "row_count": metadata.row_count,
        "checksum_sha256": metadata.checksum_sha256,
    }


def _strategy_from_args(
    args: argparse.Namespace,
    strategy_name: str,
    inst_id: str,
) -> TradingStrategy:
    configured_size = args.trade_size_eth if inst_id.startswith("ETH-") else args.trade_size_btc
    dust_threshold = "0.0001" if inst_id.startswith("ETH-") else "0.00001"
    return build_strategy(
        strategy_name,
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
            risk_config=config.risk_config,
        ),
    ).run(candles)


def _evaluation_config(args: argparse.Namespace, *, inst_ids: tuple[str, ...]) -> EvaluationConfig:
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


def _gate_config(args: argparse.Namespace) -> PerformanceGateConfig:
    max_drawdown = Decimal(args.max_drawdown_usdt)
    min_profit_factor = args.min_profit_factor.strip().lower()
    return PerformanceGateConfig(
        min_trades=args.min_trades,
        min_net_pnl_usdt=Decimal(args.min_net_pnl_usdt),
        max_drawdown_usdt=max_drawdown if max_drawdown > 0 else None,
        min_profit_factor=None
        if min_profit_factor in {"", "none", "null", "0"}
        else Decimal(min_profit_factor),
    )


def _criteria_payload(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "min_trades": args.min_trades,
        "min_net_pnl_usdt": args.min_net_pnl_usdt,
        "max_drawdown_usdt": args.max_drawdown_usdt,
        "min_profit_factor": args.min_profit_factor,
        "min_regime_passes": args.min_regime_passes,
        "require_baseline_outperformance": not args.allow_baseline_underperformance,
        "regime_thresholds": {
            "trend_threshold_bps": args.trend_threshold_bps,
            "high_volatility_bps": args.high_volatility_bps,
            "drawdown_threshold_bps": args.drawdown_threshold_bps,
            "event_range_bps": args.event_range_bps,
        },
        "cost_model": {
            "taker_fee_bps": args.taker_fee_bps,
            "spread_bps": args.spread_bps,
            "slippage_bps": args.slippage_bps,
            "market_impact_bps": args.market_impact_bps,
            "reject_every_n_orders": args.reject_every_n_orders,
        },
    }


def _accepted_config(row: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    parameters = row["parameters"]
    return {
        "inst_id": row["inst_id"],
        "bar": row["bar"],
        "strategy": row["strategy"],
        "parameters": parameters,
        "trade_size_base": parameters.get("trade_size_btc"),
        "source_report": "evidence_matrix",
        "min_regime_passes": args.min_regime_passes,
        "passing_regime_signatures": row["acceptance_checks"]["passing_regime_signatures"],
    }


def _default_regime_window_size(candle_count: int) -> int:
    return max(30, candle_count // 12) if candle_count else 30


def _average_range_bps(candles: tuple[MarketCandle, ...]) -> Decimal:
    return sum((_range_bps(candle) for candle in candles), Decimal("0")) / Decimal(len(candles))


def _range_bps(candle: MarketCandle) -> Decimal:
    basis = candle.close if candle.close != 0 else Decimal("1")
    return (candle.high - candle.low) / basis * Decimal("10000")


def _max_drawdown_bps(candles: tuple[MarketCandle, ...]) -> Decimal:
    peak = candles[0].close
    max_drawdown = Decimal("0")
    for candle in candles:
        peak = max(peak, candle.close)
        if peak == 0:
            continue
        drawdown = (peak - candle.close) / peak * Decimal("10000")
        max_drawdown = max(max_drawdown, drawdown)
    return max_drawdown


def _positive_decimal_or_none(raw: str) -> Decimal | None:
    value = Decimal(raw)
    return value if value > 0 else None


def main() -> None:
    payload = asyncio.run(run_evidence_matrix(build_parser().parse_args()))
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
