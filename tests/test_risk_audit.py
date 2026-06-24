from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

import pytest

from lumiere.config import Settings
from lumiere.risk_audit import assert_risk_audit_passes, audit_settings


def make_settings(tmp_path: Path, **overrides) -> Settings:
    accepted_candidates = tmp_path / "accepted_candidates.json"
    accepted_candidates.write_text(
        json.dumps(
            {
                "accepted_configs": [
                    {
                        "inst_id": "BTC-USDT",
                        "strategy": "moving_average_crossover",
                        "candidate": {
                            "strategy": "moving_average_crossover",
                            "fast_window": 5,
                            "slow_window": 20,
                            "trade_size_base": "0.001",
                        },
                        "optimizer_passed": True,
                        "expected_edge_bps": "2.5",
                        "expected_edge_source": "historical_forward_return_after_costs",
                        "expectancy_calibration": {"calibrated": True},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
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
        "optimizer_accepted_candidates_path": str(accepted_candidates),
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
        "optimizer_candidate_approved",
        "drawdown_cap_configured",
        "per_trade_risk_configured",
    }
    with pytest.raises(RuntimeError, match="risk audit failed"):
        assert_risk_audit_passes(settings)


def test_risk_audit_passes_ready_configuration(tmp_path) -> None:
    settings = make_settings(tmp_path)

    report = audit_settings(settings)

    assert report.passed
    assert_risk_audit_passes(settings)


def test_risk_audit_rejects_oversized_risk_budget(tmp_path) -> None:
    settings = make_settings(tmp_path, risk_max_risk_per_trade_pct=Decimal("0.25"))

    report = audit_settings(settings)

    assert not report.passed
    assert "per_trade_risk_configured" in {check.name for check in report.failures()}


def test_risk_audit_allows_explicit_research_demo_without_optimizer_candidate(tmp_path) -> None:
    missing_candidates = tmp_path / "missing.json"
    settings = make_settings(
        tmp_path,
        demo_research_mode=True,
        risk_require_performance_gate=False,
        risk_max_daily_trades=20,
        optimizer_accepted_candidates_path=str(missing_candidates),
    )

    report = audit_settings(settings)

    assert report.passed
    assert_risk_audit_passes(settings)


def test_risk_audit_requires_daily_trade_limit_in_research_demo(tmp_path) -> None:
    missing_candidates = tmp_path / "missing.json"
    settings = make_settings(
        tmp_path,
        demo_research_mode=True,
        risk_require_performance_gate=False,
        risk_max_daily_trades=0,
        optimizer_accepted_candidates_path=str(missing_candidates),
    )

    report = audit_settings(settings)

    assert not report.passed
    assert "research_daily_trade_limit_configured" in {
        check.name for check in report.failures()
    }


def test_risk_audit_rejects_uncalibrated_optimizer_candidate(tmp_path) -> None:
    uncalibrated = tmp_path / "uncalibrated_candidates.json"
    uncalibrated.write_text(
        json.dumps(
            {
                "accepted_configs": [
                    {
                        "inst_id": "BTC-USDT",
                        "strategy": "moving_average_crossover",
                        "candidate": {
                            "strategy": "moving_average_crossover",
                            "fast_window": 5,
                            "slow_window": 20,
                        },
                        "optimizer_passed": True,
                        "expected_edge_bps": "0",
                        "expected_edge_source": "heuristic_indicator_distance",
                        "expectancy_calibration": {"calibrated": False},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    settings = make_settings(tmp_path, optimizer_accepted_candidates_path=str(uncalibrated))

    report = audit_settings(settings)

    assert not report.passed
    assert "optimizer_candidate_approved" in {check.name for check in report.failures()}
