from __future__ import annotations

import json
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from lumiere.config import Settings


@dataclass(frozen=True, slots=True)
class RiskAuditCheck:
    name: str
    passed: bool
    message: str

    def to_dict(self) -> dict[str, str | bool]:
        return {"name": self.name, "passed": self.passed, "message": self.message}


@dataclass(frozen=True, slots=True)
class RiskAuditReport:
    checks: tuple[RiskAuditCheck, ...]

    @property
    def passed(self) -> bool:
        return all(check.passed for check in self.checks)

    def failures(self) -> tuple[RiskAuditCheck, ...]:
        return tuple(check for check in self.checks if not check.passed)

    def to_dict(self) -> dict[str, Any]:
        return {"passed": self.passed, "checks": [check.to_dict() for check in self.checks]}


def audit_settings(settings: Settings) -> RiskAuditReport:
    risk = settings.risk_config()
    optimizer_check = _optimizer_candidate_check(settings)
    research_mode = settings.demo_research_mode
    checks = [
        RiskAuditCheck(
            "telegram_access_restricted",
            bool(settings.allowed_chat_ids),
            "TELEGRAM_ALLOWED_CHAT_IDS must restrict bot control to known chat ids",
        ),
        RiskAuditCheck(
            "performance_gate_required",
            risk.performance_gate_required or research_mode,
            "RISK_REQUIRE_PERFORMANCE_GATE must be enabled unless DEMO_RESEARCH_MODE=true",
        ),
        RiskAuditCheck(
            "spread_guard_configured",
            risk.max_spread_bps is not None,
            "RISK_MAX_SPREAD_BPS must be set",
        ),
        RiskAuditCheck(
            "edge_cost_buffer_configured",
            risk.min_expected_edge_buffer_bps > 0,
            "RISK_MIN_EXPECTED_EDGE_BUFFER_BPS must be positive",
        ),
        RiskAuditCheck(
            "optimizer_candidate_approved",
            optimizer_check[0] or research_mode,
            optimizer_check[1]
            if not research_mode
            else "DEMO_RESEARCH_MODE=true: optimizer candidate is waived for demo research only",
        ),
        RiskAuditCheck(
            "drawdown_cap_configured",
            risk.max_drawdown_usdt is not None,
            "RISK_MAX_DRAWDOWN_USDT must be set",
        ),
        RiskAuditCheck(
            "risk_state_path_configured",
            bool(settings.risk_state_path.strip()),
            "RISK_STATE_PATH must persist high-water mark state",
        ),
        RiskAuditCheck(
            "portfolio_exposure_bounded",
            risk.max_portfolio_exposure_pct <= 1,
            "RISK_MAX_PORTFOLIO_EXPOSURE_PCT must be at most 100%",
        ),
        RiskAuditCheck(
            "per_trade_risk_configured",
            Decimal("0") < risk.max_risk_per_trade_pct <= Decimal("0.05"),
            "RISK_MAX_RISK_PER_TRADE_PCT must be positive and no more than 5%",
        ),
        RiskAuditCheck(
            "maker_execution_guard_configured",
            settings.okx_execution_policy != "post_only_maker"
            or (
                risk.max_maker_non_fill_rate is not None
                and risk.max_maker_adverse_selection_bps is not None
            ),
            "post-only maker mode requires RISK_MAX_MAKER_NON_FILL_RATE and "
            "RISK_MAX_MAKER_ADVERSE_SELECTION_BPS",
        ),
        RiskAuditCheck(
            "research_daily_trade_limit_configured",
            not research_mode or risk.max_daily_trades is not None,
            "DEMO_RESEARCH_MODE requires RISK_MAX_DAILY_TRADES to bound exploratory turnover",
        ),
    ]
    return RiskAuditReport(tuple(checks))


def assert_risk_audit_passes(settings: Settings) -> None:
    report = audit_settings(settings)
    if report.passed:
        return
    failures = "; ".join(f"{check.name}: {check.message}" for check in report.failures())
    raise RuntimeError(f"risk audit failed: {failures}")


def _optimizer_candidate_check(settings: Settings) -> tuple[bool, str]:
    path = Path(settings.optimizer_accepted_candidates_path)
    if not path.exists():
        return False, "OPTIMIZER_ACCEPTED_CANDIDATES_PATH must point to optimizer output"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return False, f"optimizer accepted-candidates file is unreadable: {exc}"
    configs = payload.get("accepted_configs", [])
    if not isinstance(configs, list):
        return False, "optimizer accepted-candidates file must contain accepted_configs"
    missing = [
        inst_id
        for inst_id in settings.enabled_inst_ids
        if not any(_candidate_matches_settings(item, settings, inst_id) for item in configs)
    ]
    if missing:
        return (
            False,
            "current live/demo strategy must match an optimizer-passed accepted candidate "
            f"with calibrated positive expected edge; missing {', '.join(missing)}",
        )
    return True, "current strategy has an optimizer-passed calibrated expected edge"


def _candidate_matches_settings(item: object, settings: Settings, inst_id: str) -> bool:
    if not isinstance(item, dict):
        return False
    candidate = item.get("candidate")
    if not isinstance(candidate, dict):
        candidate = item
    if item.get("inst_id") != inst_id:
        return False
    if (item.get("strategy") or candidate.get("strategy")) != settings.strategy_name:
        return False
    if item.get("optimizer_passed") is not True:
        return False
    if item.get("expected_edge_source") != "historical_forward_return_after_costs":
        return False
    edge = _decimal_or_none(item.get("expected_edge_bps"))
    if edge is None or edge <= 0:
        return False
    if not _candidate_parameters_match(candidate, settings):
        return False
    calibration = item.get("expectancy_calibration")
    return isinstance(calibration, dict) and calibration.get("calibrated") is True


def _candidate_parameters_match(candidate: dict[str, Any], settings: Settings) -> bool:
    if settings.strategy_name == "moving_average_crossover":
        return _int(candidate.get("fast_window")) == settings.strategy_fast_window and _int(
            candidate.get("slow_window")
        ) == settings.strategy_slow_window
    if settings.strategy_name == "rsi_mean_reversion":
        return (
            _int(candidate.get("rsi_period")) == settings.strategy_rsi_period
            and _decimal_or_none(candidate.get("oversold_rsi")) == settings.strategy_oversold_rsi
            and _decimal_or_none(candidate.get("overbought_rsi"))
            == settings.strategy_overbought_rsi
        )
    if settings.strategy_name == "volatility_breakout":
        return (
            _int(candidate.get("breakout_lookback")) == settings.strategy_breakout_lookback
            and _int(candidate.get("breakout_atr_period"))
            == settings.strategy_breakout_atr_period
            and _decimal_or_none(candidate.get("breakout_atr_multiplier"))
            == settings.strategy_breakout_atr_multiplier
            and _decimal_or_none(candidate.get("breakout_min_atr_pct"))
            == settings.strategy_breakout_min_atr_pct
        )
    return False


def _decimal_or_none(value: object) -> Decimal | None:
    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return None


def _int(value: object) -> int | None:
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return None


def main() -> None:
    settings = Settings()
    report = audit_settings(settings)
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    if not report.passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
