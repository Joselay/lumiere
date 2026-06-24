from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime, timedelta
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from lumiere.attribution import AttributionLedger

SEVERITY_ORDER = {"critical": 0, "high": 1, "medium": 2, "low": 3}

STAGE_THRESHOLDS = {
    "backtest": {
        "min_trades": 30,
        "min_profit_factor": Decimal("1.2"),
        "max_drawdown_pct": Decimal("0.05"),
        "min_observation_days": 0,
        "min_data_days": Decimal("30"),
        "min_net_pnl_usdt": Decimal("0"),
        "max_slippage_bps": Decimal("10"),
        "min_consecutive_gate_passes": 0,
        "require_live_config_match": False,
    },
    "paper": {
        "min_trades": 30,
        "min_profit_factor": Decimal("1.2"),
        "max_drawdown_pct": Decimal("0.05"),
        "min_observation_days": 14,
        "min_data_days": Decimal("30"),
        "min_net_pnl_usdt": Decimal("0"),
        "max_slippage_bps": Decimal("10"),
        "min_consecutive_gate_passes": 3,
        "require_live_config_match": True,
    },
    "small_demo": {
        "min_trades": 20,
        "min_profit_factor": Decimal("1.0"),
        "max_drawdown_pct": Decimal("0.05"),
        "min_observation_days": 7,
        "min_data_days": Decimal("30"),
        "min_net_pnl_usdt": Decimal("0"),
        "max_slippage_bps": Decimal("20"),
        "min_consecutive_gate_passes": 3,
        "require_live_config_match": True,
    },
    "larger_demo": {
        "min_trades": 20,
        "min_profit_factor": Decimal("1.0"),
        "max_drawdown_pct": Decimal("0.05"),
        "min_observation_days": 7,
        "min_data_days": Decimal("30"),
        "min_net_pnl_usdt": Decimal("0"),
        "max_slippage_bps": Decimal("20"),
        "min_consecutive_gate_passes": 3,
        "require_live_config_match": True,
    },
}


class _Blockers(list[dict[str, str]]):
    def add(self, severity: str, code: str, message: str) -> None:
        self.append({"severity": severity, "code": code, "message": message})

    def sorted(self) -> list[dict[str, str]]:
        return sorted(self, key=lambda item: (SEVERITY_ORDER[item["severity"]], item["code"]))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit Lumiere staged promotion readiness")
    parser.add_argument("--stage", choices=tuple(STAGE_THRESHOLDS), required=True)
    parser.add_argument(
        "--optimizer-report",
        default="reports/strategy_optimization/optimizer_report.json",
    )
    parser.add_argument("--backtest-report", default="reports/backtest_report.json")
    parser.add_argument("--paper-ledger", default="data/paper_trading.jsonl")
    parser.add_argument("--attribution-ledger", default="data/attribution.jsonl")
    parser.add_argument(
        "--current-config",
        help="JSON file containing runtime strategy/risk settings; defaults to environment",
    )
    parser.add_argument("--window-days", type=int, default=14)
    parser.add_argument("--max-artifact-age-hours", type=int, default=168)
    parser.add_argument("--human-approval", action="store_true")
    parser.add_argument("--proposed-size-multiple", default="1")
    parser.add_argument("--output", default="reports/promotion_evidence.json")
    return parser


def build_evidence_packet(args: argparse.Namespace) -> dict[str, Any]:
    stage = args.stage
    thresholds = STAGE_THRESHOLDS[stage]
    now = _normalise(getattr(args, "now", None) or datetime.now(tz=UTC))
    blockers = _Blockers()
    checks: dict[str, Any] = {
        "thresholds": {key: str(value) for key, value in thresholds.items()},
    }

    optimizer = _load_required_json(
        Path(args.optimizer_report), blockers, "optimizer_report", "critical"
    )
    accepted = _accepted_configs(optimizer)
    checks["has_accepted_candidate"] = bool(accepted)
    if not accepted:
        blockers.add(
            "critical",
            "no_accepted_candidate",
            "optimizer report must contain at least one accepted strategy candidate",
        )

    backtest_report_path = getattr(args, "backtest_report", "reports/backtest_report.json")
    backtest = _load_required_json(Path(backtest_report_path), blockers, "backtest_report", "high")
    checks["backtest_report_present"] = backtest is not None

    _audit_optimizer_artifact(
        optimizer,
        accepted,
        thresholds=thresholds,
        blockers=blockers,
        checks=checks,
        now=now,
        max_age=timedelta(hours=getattr(args, "max_artifact_age_hours", 168)),
    )

    attribution_report = None
    attribution_events: tuple[dict[str, Any], ...] = ()
    if stage != "backtest":
        attribution_path = Path(args.attribution_ledger)
        if not attribution_path.exists():
            blockers.add(
                "critical",
                "missing_attribution_ledger",
                "paper/demo stages require the attribution ledger",
            )
        else:
            attribution_ledger = AttributionLedger(attribution_path)
            attribution_events = attribution_ledger.events
            attribution_report = attribution_ledger.report(
                window=timedelta(days=getattr(args, "window_days", 14)),
                now=now,
            ).to_dict()
            _audit_attribution_report(
                attribution_report,
                thresholds=thresholds,
                blockers=blockers,
                checks=checks,
                stage=stage,
            )
            _audit_gate_history(
                attribution_events,
                thresholds=thresholds,
                blockers=blockers,
                checks=checks,
            )
            _audit_api_errors(attribution_events, blockers, checks)

        paper_ledger_path = getattr(args, "paper_ledger", "data/paper_trading.jsonl")
        paper_events = _load_required_events(Path(paper_ledger_path), blockers, "paper_ledger")
        _audit_observation_duration(
            (*paper_events, *attribution_events),
            thresholds=thresholds,
            blockers=blockers,
            checks=checks,
        )
        current_config = _load_current_config(getattr(args, "current_config", None))
        _audit_current_config(
            accepted,
            current_config,
            blockers=blockers,
            checks=checks,
        )
        if stage == "larger_demo":
            _audit_larger_demo_controls(current_config, blockers, checks)

    if stage == "larger_demo":
        if not getattr(args, "human_approval", False):
            blockers.add(
                "critical",
                "human_approval_missing",
                "larger demo promotion requires explicit human approval",
            )
        size_multiple = _decimal_or_none(getattr(args, "proposed_size_multiple", "1"))
        checks["size_increase_within_2x"] = size_multiple is not None and size_multiple <= 2
        if not checks["size_increase_within_2x"]:
            blockers.add(
                "high", "size_increase_too_large", "larger demo size increase must be <= 2x"
            )

    sorted_blockers = blockers.sorted()
    missing = [
        item["code"].removeprefix("missing_")
        for item in sorted_blockers
        if item["code"].startswith("missing_")
    ]
    go = not sorted_blockers
    return {
        "stage": stage,
        "go": go,
        "summary": "go" if go else f"no-go: {len(sorted_blockers)} blocker(s)",
        "blockers": sorted_blockers,
        "missing_evidence": missing,
        "checks": checks,
        "accepted_configs": accepted,
        "attribution_report": attribution_report,
    }


def _audit_optimizer_artifact(
    optimizer: dict[str, Any] | None,
    accepted: list[dict[str, Any]],
    *,
    thresholds: dict[str, Any],
    blockers: _Blockers,
    checks: dict[str, Any],
    now: datetime,
    max_age: timedelta,
) -> None:
    if optimizer is None:
        return
    generated_at = _parse_datetime_or_none(optimizer.get("generated_at"))
    checks["accepted_candidate_age_hours"] = None
    if generated_at is None:
        blockers.add(
            "high",
            "accepted_candidate_undated",
            "accepted candidate artifact must include generated_at",
        )
    else:
        age = now - generated_at
        checks["accepted_candidate_age_hours"] = str(age.total_seconds() / 3600)
        if age > max_age:
            blockers.add("high", "accepted_candidate_stale", "accepted candidate artifact is stale")

    data_days = _optimizer_data_days(optimizer)
    checks["optimizer_data_days"] = None if data_days is None else str(data_days)
    if data_days is None or data_days < thresholds["min_data_days"]:
        blockers.add(
            "high",
            "historical_data_too_short",
            f"accepted historical data must cover at least {thresholds['min_data_days']} days",
        )

    checks["accepted_candidate_thresholds_passed"] = _accepted_candidate_thresholds_pass(
        optimizer,
        thresholds,
    )
    if not checks["accepted_candidate_thresholds_passed"]:
        blockers.add(
            "high",
            "accepted_candidate_thresholds_failed",
            "accepted candidates must satisfy the runbook backtest/OOS thresholds",
        )


def _accepted_candidate_thresholds_pass(
    optimizer: dict[str, Any],
    thresholds: dict[str, Any],
) -> bool:
    for candidate in _optimizer_candidates(optimizer):
        if candidate.get("accepted") is not True or candidate.get("rejection_reason") not in {
            None,
            "",
        }:
            continue
        test_report = (
            candidate.get("test_report") if isinstance(candidate.get("test_report"), dict) else {}
        )
        metrics = test_report.get("metrics") if isinstance(test_report.get("metrics"), dict) else {}
        if (
            _decimal_or_none(metrics.get("net_pnl_usdt")) is None
            or _decimal_or_none(metrics.get("net_pnl_usdt")) <= thresholds["min_net_pnl_usdt"]
        ):
            continue
        if int(metrics.get("trade_count") or 0) < thresholds["min_trades"]:
            continue
        profit_factor = _profit_factor(metrics.get("profit_factor"))
        if profit_factor is None or profit_factor < thresholds["min_profit_factor"]:
            continue
        drawdown = _decimal_or_none(metrics.get("max_drawdown_usdt"))
        starting = _decimal_or_none(metrics.get("starting_equity_usdt"))
        if (
            drawdown is None
            or starting is None
            or drawdown > starting * thresholds["max_drawdown_pct"]
        ):
            continue
        baseline = (
            test_report.get("baseline_comparison")
            if isinstance(test_report.get("baseline_comparison"), dict)
            else {}
        )
        no_trade_delta = _decimal_or_none(baseline.get("net_pnl_minus_no_trade_usdt"))
        buy_hold_delta = _decimal_or_none(baseline.get("net_pnl_minus_buy_and_hold_usdt"))
        if no_trade_delta is not None and no_trade_delta <= 0:
            continue
        if buy_hold_delta is not None and buy_hold_delta <= 0:
            continue
        sharpe = _decimal_or_none(metrics.get("sharpe"))
        sortino = _decimal_or_none(metrics.get("sortino"))
        risk_adjusted_available = sharpe is not None or sortino is not None
        risk_adjusted_passed = (sharpe is not None and sharpe > Decimal("0.5")) or (
            sortino is not None and sortino > Decimal("0.75")
        )
        if risk_adjusted_available and not risk_adjusted_passed:
            continue
        slippage = _candidate_slippage_bps(test_report)
        if slippage is not None and slippage > thresholds["max_slippage_bps"]:
            continue
        walk_forward_gates = candidate.get("walk_forward_gates")
        if not isinstance(walk_forward_gates, list) or not walk_forward_gates:
            continue
        if not all(
            isinstance(gate, dict) and gate.get("allowed") is True for gate in walk_forward_gates
        ):
            continue
        return True
    return False


def _audit_attribution_report(
    report: dict[str, Any],
    *,
    thresholds: dict[str, Any],
    blockers: _Blockers,
    checks: dict[str, Any],
    stage: str,
) -> None:
    metrics = report["metrics"]
    trade_count = int(metrics["trade_count"])
    profit_factor = _profit_factor(metrics.get("profit_factor"))
    drawdown = Decimal(str(metrics["max_drawdown_usdt"]))
    starting = Decimal(str(metrics["starting_equity_usdt"]))
    net_pnl = Decimal(str(metrics["net_pnl_usdt"]))
    checks["attribution_present"] = True
    checks["min_trades_passed"] = trade_count >= thresholds["min_trades"]
    checks["profit_factor_passed"] = (
        profit_factor is not None and profit_factor >= thresholds["min_profit_factor"]
    )
    checks["drawdown_passed"] = drawdown <= starting * thresholds["max_drawdown_pct"]
    checks["net_pnl_passed"] = (
        net_pnl > thresholds["min_net_pnl_usdt"]
        if stage == "paper"
        else net_pnl >= thresholds["min_net_pnl_usdt"]
    )
    checks["alerts"] = report["alerts"]
    checks["fill_completeness"] = metrics.get("fill_completeness", {})
    checks["slippage_passed"] = _realized_slippage_passed(metrics, thresholds)

    if not checks["min_trades_passed"]:
        blockers.add("high", "trade_count_below_threshold", "observation window has too few trades")
    if not checks["profit_factor_passed"]:
        blockers.add(
            "high", "profit_factor_below_threshold", "profit factor is below the stage threshold"
        )
    if not checks["drawdown_passed"]:
        blockers.add("high", "drawdown_threshold_breached", "drawdown exceeds the stage threshold")
    if not checks["net_pnl_passed"]:
        blockers.add(
            "high", "net_pnl_threshold_failed", "net PnL does not satisfy the stage threshold"
        )
    for alert in report["alerts"]:
        blockers.add("high", f"{alert}_alert", f"attribution alert is active: {alert}")
    fill_completeness = metrics.get("fill_completeness") or {}
    if any(
        int(fill_completeness.get(key, 0) or 0)
        for key in ("orders_without_final_attribution", "filled_orders_without_fills")
    ):
        blockers.add(
            "critical", "fill_reconciliation_incomplete", "order/fill attribution is incomplete"
        )
    if not checks["slippage_passed"]:
        blockers.add(
            "high", "realized_slippage_above_threshold", "realized slippage exceeds stage bounds"
        )


def _audit_gate_history(
    events: tuple[dict[str, Any], ...],
    *,
    thresholds: dict[str, Any],
    blockers: _Blockers,
    checks: dict[str, Any],
) -> None:
    required = int(thresholds["min_consecutive_gate_passes"])
    account_events = sorted(
        (event for event in events if "performance_gate_passed" in event),
        key=lambda item: str(item.get("ts") or ""),
    )
    consecutive = 0
    for event in reversed(account_events):
        if event.get("performance_gate_passed") is True:
            consecutive += 1
            continue
        break
    checks["consecutive_gate_passes"] = consecutive
    if consecutive < required:
        blockers.add(
            "high",
            "consecutive_gate_history_short",
            f"performance gate must pass for {required} consecutive checks",
        )


def _audit_observation_duration(
    events: tuple[dict[str, Any], ...],
    *,
    thresholds: dict[str, Any],
    blockers: _Blockers,
    checks: dict[str, Any],
) -> None:
    timestamps = [_parse_datetime_or_none(event.get("ts")) for event in events]
    timestamps = [ts for ts in timestamps if ts is not None]
    days = Decimal("0")
    if len(timestamps) >= 2:
        days = Decimal(str((max(timestamps) - min(timestamps)).total_seconds())) / Decimal("86400")
    checks["observation_days"] = str(days)
    checks["paper_observation_days_passed"] = days >= Decimal(
        str(thresholds["min_observation_days"])
    )
    if not checks["paper_observation_days_passed"]:
        blockers.add(
            "high",
            "observation_duration_short",
            "observation window is shorter than the runbook stage minimum",
        )


def _audit_current_config(
    accepted: list[dict[str, Any]],
    current_config: dict[str, Any] | None,
    *,
    blockers: _Blockers,
    checks: dict[str, Any],
) -> None:
    checks["current_config_matches_accepted"] = False
    if current_config is None:
        blockers.add(
            "high", "current_config_missing", "current runtime strategy config is required"
        )
        return
    for item in accepted:
        if _accepted_config_matches_current(item, current_config):
            checks["current_config_matches_accepted"] = True
            return
    blockers.add(
        "high",
        "current_config_mismatch",
        "current runtime strategy config differs from accepted candidate",
    )


def _audit_larger_demo_controls(
    current_config: dict[str, Any] | None,
    blockers: _Blockers,
    checks: dict[str, Any],
) -> None:
    if current_config is None:
        checks["larger_demo_risk_controls_passed"] = False
        return
    performance_gate_required = _truthy(current_config.get("risk_require_performance_gate"))
    risk_per_trade = _percentage_fraction_or_none(current_config.get("risk_max_risk_per_trade_pct"))
    portfolio_cap = _percentage_fraction_or_none(
        current_config.get("risk_max_portfolio_exposure_pct")
    )
    checks["performance_gate_required_for_larger_demo"] = performance_gate_required
    checks["risk_per_trade_lte_1_pct"] = risk_per_trade is not None and risk_per_trade <= Decimal(
        "0.01"
    )
    checks["portfolio_exposure_cap_configured"] = portfolio_cap is not None and portfolio_cap <= 1
    checks["larger_demo_risk_controls_passed"] = (
        checks["performance_gate_required_for_larger_demo"]
        and checks["risk_per_trade_lte_1_pct"]
        and checks["portfolio_exposure_cap_configured"]
    )
    if not checks["performance_gate_required_for_larger_demo"]:
        blockers.add(
            "high",
            "performance_gate_not_required",
            "larger demo requires RISK_REQUIRE_PERFORMANCE_GATE=true",
        )
    if not checks["risk_per_trade_lte_1_pct"]:
        blockers.add(
            "high",
            "risk_per_trade_above_one_pct",
            "larger demo requires max risk per trade <= 1%",
        )
    if not checks["portfolio_exposure_cap_configured"]:
        blockers.add(
            "high",
            "portfolio_exposure_cap_missing",
            "larger demo requires a configured portfolio exposure cap",
        )


def _audit_api_errors(
    events: tuple[dict[str, Any], ...], blockers: _Blockers, checks: dict[str, Any]
) -> None:
    api_errors = [
        event
        for event in events
        if str(event.get("type") or "") in {"api_error", "okx_api_error", "error"}
    ]
    checks["api_error_count"] = len(api_errors)
    if len(api_errors) >= 3:
        blockers.add(
            "high", "api_error_cluster", "OKX API errors have clustered in the observation window"
        )


def _realized_slippage_passed(metrics: dict[str, Any], thresholds: dict[str, Any]) -> bool:
    slippage = _decimal_or_none(
        metrics.get("realized_slippage_bps") or metrics.get("average_slippage_bps")
    )
    return slippage is None or slippage <= thresholds["max_slippage_bps"]


def _candidate_slippage_bps(test_report: dict[str, Any]) -> Decimal | None:
    execution_quality = test_report.get("execution_quality")
    if isinstance(execution_quality, dict):
        value = _decimal_or_none(
            execution_quality.get("average_slippage_bps")
            or execution_quality.get("realized_slippage_bps")
            or execution_quality.get("slippage_bps")
        )
        if value is not None:
            return value
    assumptions = test_report.get("assumptions")
    if isinstance(assumptions, dict):
        return _decimal_or_none(assumptions.get("slippage_bps"))
    return None


def _optimizer_data_days(optimizer: dict[str, Any]) -> Decimal | None:
    spans: list[Decimal] = []
    for report in optimizer.get("reports", []):
        if not isinstance(report, dict):
            continue
        dataset = report.get("dataset") if isinstance(report.get("dataset"), dict) else {}
        start = _parse_datetime_or_none(dataset.get("start") or dataset.get("period_start"))
        end = _parse_datetime_or_none(dataset.get("end") or dataset.get("period_end"))
        if start is None or end is None:
            for candidate in report.get("candidates", []):
                if not isinstance(candidate, dict):
                    continue
                test_report = (
                    candidate.get("test_report")
                    if isinstance(candidate.get("test_report"), dict)
                    else {}
                )
                start = _parse_datetime_or_none(test_report.get("period_start"))
                end = _parse_datetime_or_none(test_report.get("period_end"))
        if start is not None and end is not None:
            spans.append(Decimal(str((end - start).total_seconds())) / Decimal("86400"))
    return max(spans) if spans else None


def _accepted_configs(optimizer: dict[str, Any] | None) -> list[dict[str, Any]]:
    if optimizer is None:
        return []
    raw = optimizer.get("accepted_configs", [])
    return [item for item in raw if isinstance(item, dict)] if isinstance(raw, list) else []


def _optimizer_candidates(optimizer: dict[str, Any]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for report in optimizer.get("reports", []):
        if isinstance(report, dict) and isinstance(report.get("candidates"), list):
            candidates.extend(item for item in report["candidates"] if isinstance(item, dict))
    return candidates


def _load_required_json(
    path: Path, blockers: _Blockers, name: str, severity: str
) -> dict[str, Any] | None:
    payload = _load_json(path)
    if payload is None:
        blockers.add(severity, f"missing_{name}", f"{name.replace('_', ' ')} is required")
    return payload


def _load_required_events(path: Path, blockers: _Blockers, name: str) -> tuple[dict[str, Any], ...]:
    if not path.exists():
        blockers.add("critical", f"missing_{name}", f"{name.replace('_', ' ')} is required")
        return ()
    return tuple(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line)


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _load_current_config(path: str | None) -> dict[str, Any] | None:
    if path:
        return _load_json(Path(path))
    source = _load_dotenv_values(Path(".env"))
    source.update(os.environ)
    env_config = {
        "strategy_name": source.get("STRATEGY_NAME"),
        "strategy_fast_window": source.get("STRATEGY_FAST_WINDOW"),
        "strategy_slow_window": source.get("STRATEGY_SLOW_WINDOW"),
        "strategy_rsi_period": source.get("STRATEGY_RSI_PERIOD"),
        "strategy_oversold_rsi": source.get("STRATEGY_OVERSOLD_RSI"),
        "strategy_overbought_rsi": source.get("STRATEGY_OVERBOUGHT_RSI"),
        "strategy_breakout_lookback": source.get("STRATEGY_BREAKOUT_LOOKBACK"),
        "strategy_breakout_atr_period": source.get("STRATEGY_BREAKOUT_ATR_PERIOD"),
        "strategy_breakout_atr_multiplier": source.get("STRATEGY_BREAKOUT_ATR_MULTIPLIER"),
        "strategy_breakout_min_atr_pct": source.get("STRATEGY_BREAKOUT_MIN_ATR_PCT"),
        "risk_require_performance_gate": source.get("RISK_REQUIRE_PERFORMANCE_GATE"),
        "risk_max_risk_per_trade_pct": source.get("RISK_MAX_RISK_PER_TRADE_PCT"),
        "risk_max_portfolio_exposure_pct": source.get("RISK_MAX_PORTFOLIO_EXPOSURE_PCT"),
    }
    return {key: value for key, value in env_config.items() if value not in {None, ""}} or None


def _load_dotenv_values(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", maxsplit=1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def _accepted_config_matches_current(item: dict[str, Any], current: dict[str, Any]) -> bool:
    candidate = (
        item.get("candidate") if isinstance(item.get("candidate"), dict) else item.get("parameters")
    )
    if not isinstance(candidate, dict):
        candidate = item
    strategy = item.get("strategy") or candidate.get("strategy")
    if strategy != current.get("strategy_name"):
        return False
    if strategy == "moving_average_crossover":
        return _int(candidate.get("fast_window")) == _int(
            current.get("strategy_fast_window")
        ) and _int(candidate.get("slow_window")) == _int(current.get("strategy_slow_window"))
    if strategy == "rsi_mean_reversion":
        return (
            _int(candidate.get("rsi_period")) == _int(current.get("strategy_rsi_period"))
            and _decimal_or_none(candidate.get("oversold_rsi"))
            == _decimal_or_none(current.get("strategy_oversold_rsi"))
            and _decimal_or_none(candidate.get("overbought_rsi"))
            == _decimal_or_none(current.get("strategy_overbought_rsi"))
        )
    if strategy == "volatility_breakout":
        return (
            _int(candidate.get("breakout_lookback"))
            == _int(current.get("strategy_breakout_lookback"))
            and _int(candidate.get("breakout_atr_period"))
            == _int(current.get("strategy_breakout_atr_period"))
            and _decimal_or_none(candidate.get("breakout_atr_multiplier"))
            == _decimal_or_none(current.get("strategy_breakout_atr_multiplier"))
            and _decimal_or_none(candidate.get("breakout_min_atr_pct"))
            == _decimal_or_none(current.get("strategy_breakout_min_atr_pct"))
        )
    return False


def _profit_factor(raw: object) -> Decimal | None:
    if raw == "Infinity":
        return Decimal("Infinity")
    return _decimal_or_none(raw)


def _truthy(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _percentage_fraction_or_none(value: object) -> Decimal | None:
    if value is None:
        return None
    raw = str(value).strip()
    if not raw:
        return None
    if raw.endswith("%"):
        percent = _decimal_or_none(raw[:-1])
        return None if percent is None else percent / Decimal("100")
    decimal = _decimal_or_none(raw)
    if decimal is None:
        return None
    return decimal / Decimal("100") if decimal > 1 else decimal


def _decimal_or_none(value: object) -> Decimal | None:
    try:
        if value is None:
            return None
        return Decimal(str(value))
    except InvalidOperation, TypeError, ValueError:
        return None


def _int(value: object) -> int | None:
    try:
        if value is None:
            return None
        return int(str(value))
    except TypeError, ValueError:
        return None


def _parse_datetime_or_none(value: object) -> datetime | None:
    if not value:
        return None
    try:
        return _normalise(datetime.fromisoformat(str(value).replace("Z", "+00:00")))
    except ValueError:
        return None


def _normalise(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def main() -> None:
    args = build_parser().parse_args()
    payload = build_evidence_packet(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not payload["go"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
