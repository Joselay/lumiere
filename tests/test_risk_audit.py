from __future__ import annotations

from decimal import Decimal

import pytest

from lumiere.config import Settings
from lumiere.risk_audit import assert_risk_audit_passes, audit_settings


def make_settings(**overrides) -> Settings:
    values = {
        "okx_api_key": "key",
        "okx_api_secret": "secret",
        "okx_passphrase": "passphrase",
        "telegram_bot_token": "token",
        "telegram_allowed_chat_ids": "123",
        "risk_require_performance_gate": True,
        "risk_max_spread_bps": Decimal("5"),
        "risk_min_expected_edge_buffer_bps": Decimal("1"),
        "risk_max_drawdown_usdt": Decimal("25"),
        "risk_max_risk_per_trade_pct": Decimal("0.01"),
        "risk_max_portfolio_exposure_pct": Decimal("1"),
        "risk_state_path": "data/risk_state.json",
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


def test_risk_audit_reports_dangerous_defaults_before_trading() -> None:
    settings = Settings(
        _env_file=None,
        okx_api_key="key",
        okx_api_secret="secret",
        okx_passphrase="passphrase",
        telegram_bot_token="token",
    )

    report = audit_settings(settings)

    assert not report.passed
    assert {check.name for check in report.failures()} >= {
        "telegram_access_restricted",
        "performance_gate_required",
        "spread_guard_configured",
        "edge_cost_buffer_configured",
        "drawdown_cap_configured",
        "per_trade_risk_configured",
    }
    with pytest.raises(RuntimeError, match="risk audit failed"):
        assert_risk_audit_passes(settings)


def test_risk_audit_passes_ready_configuration() -> None:
    settings = make_settings()

    report = audit_settings(settings)

    assert report.passed
    assert_risk_audit_passes(settings)


def test_risk_audit_rejects_oversized_risk_budget() -> None:
    settings = make_settings(risk_max_risk_per_trade_pct=Decimal("0.25"))

    report = audit_settings(settings)

    assert not report.passed
    assert "per_trade_risk_configured" in {check.name for check in report.failures()}
