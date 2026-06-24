from __future__ import annotations

import json
from argparse import Namespace
from datetime import UTC, datetime, timedelta
from decimal import Decimal

from lumiere.attribution import AttributionLedger
from lumiere.evidence_cli import build_evidence_packet
from lumiere.models import AccountSnapshot, DecisionAction


def _write_json(path, payload) -> str:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return str(path)


def _optimizer_payload(now: datetime, *, accepted_config: dict | None = None) -> dict:
    accepted = accepted_config or {
        "inst_id": "BTC-USDT",
        "strategy": "moving_average_crossover",
        "candidate": {"fast_window": 5, "slow_window": 20},
        "optimizer_passed": True,
        "expected_edge_bps": "2.5",
        "expected_edge_source": "historical_forward_return_after_costs",
        "expectancy_calibration": {"calibrated": True},
    }
    return {
        "generated_at": now.isoformat(),
        "accepted_configs": [accepted],
        "reports": [
            {
                "inst_id": "BTC-USDT",
                "dataset": {
                    "start": (now - timedelta(days=60)).isoformat(),
                    "end": now.isoformat(),
                    "sha256": "dataset-checksum",
                },
                "candidates": [
                    {
                        "accepted": True,
                        "rejection_reason": None,
                        "gate": {"allowed": True, "reason": "passed"},
                        "walk_forward_gates": [
                            {"allowed": True, "reason": "passed"},
                            {"allowed": True, "reason": "passed"},
                            {"allowed": True, "reason": "passed"},
                        ],
                        "test_report": {
                            "period_start": (now - timedelta(days=20)).isoformat(),
                            "period_end": now.isoformat(),
                            "metrics": {
                                "net_pnl_usdt": "12",
                                "trade_count": 35,
                                "profit_factor": "1.8",
                                "max_drawdown_usdt": "20",
                                "starting_equity_usdt": "1000",
                                "sharpe": 0.8,
                                "sortino": None,
                            },
                            "buy_and_hold_pnl_usdt": "5",
                            "no_trade_pnl_usdt": "0",
                            "baseline_comparison": {
                                "net_pnl_minus_no_trade_usdt": "12",
                                "net_pnl_minus_buy_and_hold_usdt": "7",
                            },
                            "execution_quality": {"average_slippage_bps": "5"},
                        },
                    }
                ],
            }
        ],
    }


def _write_profitable_attribution(path, now: datetime) -> None:
    ledger = AttributionLedger(path)
    start = now - timedelta(days=14)
    for day in (12, 13, 14):
        ledger.record_account(
            AccountSnapshot(
                equity_usdt=Decimal("1000"),
                available_usdt=Decimal("1000"),
                performance_gate_passed=True,
                performance_gate_reason="passed",
            ),
            ts=now - timedelta(days=15 - day),
        )
    for index in range(15):
        ts = start + timedelta(days=index)
        ledger.record_fill(
            inst_id="BTC-USDT",
            side=DecisionAction.BUY,
            size_base=Decimal("1"),
            price=Decimal("100"),
            fee=Decimal("0"),
            ts=ts,
            decision_price=Decimal("100"),
        )
        ledger.record_fill(
            inst_id="BTC-USDT",
            side=DecisionAction.SELL,
            size_base=Decimal("1"),
            price=Decimal("101"),
            fee=Decimal("0"),
            ts=ts + timedelta(minutes=1),
            decision_price=Decimal("101"),
        )


def _write_paper_ledger(path, now: datetime) -> None:
    start = now - timedelta(days=14)
    rows = [
        {"type": "simulated_fill", "ts": (start + timedelta(hours=12 * i)).isoformat()}
        for i in range(29)
    ]
    rows.append({"type": "simulated_fill", "ts": now.isoformat()})
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_readiness_auditor_allows_complete_paper_packet(tmp_path) -> None:
    now = datetime(2026, 6, 24, tzinfo=UTC)
    optimizer = _write_json(tmp_path / "optimizer.json", _optimizer_payload(now))
    backtest = _write_json(tmp_path / "backtest.json", {"reports": [{"ok": True}]})
    attribution = tmp_path / "attribution.jsonl"
    paper = tmp_path / "paper.jsonl"
    config = _write_json(
        tmp_path / "config.json",
        {
            "strategy_name": "moving_average_crossover",
            "strategy_fast_window": 5,
            "strategy_slow_window": 20,
        },
    )
    _write_profitable_attribution(attribution, now)
    _write_paper_ledger(paper, now)

    packet = build_evidence_packet(
        Namespace(
            stage="paper",
            optimizer_report=optimizer,
            backtest_report=backtest,
            paper_ledger=str(paper),
            attribution_ledger=str(attribution),
            current_config=config,
            window_days=14,
            max_artifact_age_hours=168,
            output=str(tmp_path / "evidence.json"),
            now=now,
        )
    )

    assert packet["go"] is True
    assert packet["blockers"] == []
    assert packet["checks"]["paper_observation_days_passed"] is True
    assert packet["checks"]["consecutive_gate_passes"] == 3


def test_readiness_auditor_reports_sorted_blockers_for_incomplete_packet(tmp_path) -> None:
    now = datetime(2026, 6, 24, tzinfo=UTC)
    stale_generated_at = now - timedelta(days=20)
    optimizer = _write_json(
        tmp_path / "optimizer.json",
        _optimizer_payload(
            stale_generated_at,
            accepted_config={
                "inst_id": "BTC-USDT",
                "strategy": "moving_average_crossover",
                "candidate": {"fast_window": 9, "slow_window": 20},
                "optimizer_passed": True,
                "expected_edge_bps": "2.5",
                "expected_edge_source": "historical_forward_return_after_costs",
                "expectancy_calibration": {"calibrated": True},
            },
        ),
    )
    attribution = tmp_path / "attribution.jsonl"
    AttributionLedger(attribution).record_fill(
        inst_id="BTC-USDT",
        side=DecisionAction.BUY,
        size_base=Decimal("1"),
        price=Decimal("100"),
        fee=Decimal("0"),
        ts=now,
        decision_price=Decimal("150"),
    )
    config = _write_json(
        tmp_path / "config.json",
        {
            "strategy_name": "moving_average_crossover",
            "strategy_fast_window": 5,
            "strategy_slow_window": 20,
        },
    )

    packet = build_evidence_packet(
        Namespace(
            stage="paper",
            optimizer_report=optimizer,
            backtest_report=str(tmp_path / "missing-backtest.json"),
            paper_ledger=str(tmp_path / "missing-paper.jsonl"),
            attribution_ledger=str(attribution),
            current_config=config,
            window_days=14,
            max_artifact_age_hours=168,
            output=str(tmp_path / "evidence.json"),
            now=now,
        )
    )

    assert packet["go"] is False
    assert [blocker["severity"] for blocker in packet["blockers"]] == sorted(
        [blocker["severity"] for blocker in packet["blockers"]],
        key={"critical": 0, "high": 1, "medium": 2, "low": 3}.get,
    )
    codes = {blocker["code"] for blocker in packet["blockers"]}
    assert codes >= {
        "missing_backtest_report",
        "missing_paper_ledger",
        "accepted_candidate_stale",
        "current_config_mismatch",
        "observation_duration_short",
        "consecutive_gate_history_short",
        "trade_count_below_threshold",
        "abnormal_slippage_alert",
    }
