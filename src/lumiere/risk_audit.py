from __future__ import annotations

import json
from dataclasses import dataclass
from decimal import Decimal
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
    checks = [
        RiskAuditCheck(
            "telegram_access_restricted",
            bool(settings.allowed_chat_ids),
            "TELEGRAM_ALLOWED_CHAT_IDS must restrict bot control to known chat ids",
        ),
        RiskAuditCheck(
            "performance_gate_required",
            risk.performance_gate_required,
            "RISK_REQUIRE_PERFORMANCE_GATE must be enabled",
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
    ]
    return RiskAuditReport(tuple(checks))


def assert_risk_audit_passes(settings: Settings) -> None:
    report = audit_settings(settings)
    if report.passed:
        return
    failures = "; ".join(f"{check.name}: {check.message}" for check in report.failures())
    raise RuntimeError(f"risk audit failed: {failures}")


def main() -> None:
    settings = Settings()
    report = audit_settings(settings)
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    if not report.passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
