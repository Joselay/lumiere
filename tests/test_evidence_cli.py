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


def _optimizer_report(now: datetime) -> dict:
    return {
        "generated_at": now.isoformat(),
        "accepted_configs": [
            {
                "inst_id": "BTC-USDT",
                "strategy": "moving_average_crossover",
                "candidate": {"fast_window": 5, "slow_window": 20},
                "optimizer_passed": True,
                "expected_edge_bps": "2.5",
                "expected_edge_source": "historical_forward_return_after_costs",
                "expectancy_calibration": {"calibrated": True},
            }
        ],
        "reports": [
            {
                "dataset": {
                    "start": (now - timedelta(days=60)).isoformat(),
                    "end": now.isoformat(),
                    "sha256": "fixture",
                },
                "candidates": [
                    {
                        "accepted": True,
                        "rejection_reason": None,
                        "gate": {"allowed": True},
                        "walk_forward_gates": [{"allowed": True}],
                        "test_report": {
                            "metrics": {
                                "net_pnl_usdt": "5",
                                "trade_count": 35,
                                "profit_factor": "1.5",
                                "max_drawdown_usdt": "10",
                                "starting_equity_usdt": "1000",
                                "sharpe": 0.7,
                            },
                            "baseline_comparison": {
                                "net_pnl_minus_no_trade_usdt": "5",
                                "net_pnl_minus_buy_and_hold_usdt": "2",
                            },
                            "execution_quality": {"average_slippage_bps": "5"},
                        },
                    }
                ],
            }
        ],
    }


def _write_attribution(path, now: datetime) -> None:
    ledger = AttributionLedger(path)
    for offset in range(3):
        ledger.record_account(
            AccountSnapshot(
                equity_usdt=Decimal("1000"),
                available_usdt=Decimal("1000"),
                performance_gate_passed=True,
                performance_gate_reason="passed",
            ),
            ts=now - timedelta(minutes=offset),
        )
    for index in range(15):
        ts = now - timedelta(days=14) + timedelta(hours=12 * index)
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


def test_evidence_packet_can_be_generated_without_live_credentials(tmp_path) -> None:
    now = datetime(2026, 6, 24, tzinfo=UTC)
    optimizer_report = _write_json(tmp_path / "optimizer.json", _optimizer_report(now))
    backtest_report = _write_json(tmp_path / "backtest.json", {"reports": [{"ok": True}]})
    ledger_path = tmp_path / "attribution.jsonl"
    paper_path = tmp_path / "paper.jsonl"
    config_path = _write_json(
        tmp_path / "config.json",
        {
            "strategy_name": "moving_average_crossover",
            "strategy_fast_window": 5,
            "strategy_slow_window": 20,
        },
    )
    _write_attribution(ledger_path, now)
    paper_path.write_text(
        "\n".join(
            json.dumps(
                {
                    "type": "simulated_fill",
                    "ts": (now - timedelta(days=14) + timedelta(hours=12 * i)).isoformat(),
                }
            )
            for i in range(29)
        )
        + "\n"
        + json.dumps({"type": "simulated_fill", "ts": now.isoformat()})
        + "\n",
        encoding="utf-8",
    )

    packet = build_evidence_packet(
        Namespace(
            stage="paper",
            optimizer_report=optimizer_report,
            backtest_report=backtest_report,
            paper_ledger=str(paper_path),
            attribution_ledger=str(ledger_path),
            current_config=config_path,
            window_days=14,
            max_artifact_age_hours=168,
            output=str(tmp_path / "evidence.json"),
            now=now,
        )
    )

    assert packet["missing_evidence"] == []
    assert packet["blockers"] == []
    assert packet["checks"]["min_trades_passed"] is True
    assert packet["checks"]["profit_factor_passed"] is True


def test_evidence_packet_rejects_missing_reports(tmp_path) -> None:
    packet = build_evidence_packet(
        Namespace(
            stage="paper",
            optimizer_report=str(tmp_path / "missing.json"),
            backtest_report=str(tmp_path / "missing-backtest.json"),
            paper_ledger=str(tmp_path / "missing-paper.jsonl"),
            attribution_ledger=str(tmp_path / "missing-ledger.jsonl"),
            current_config=str(tmp_path / "missing-config.json"),
            window_days=14,
            output=str(tmp_path / "evidence.json"),
        )
    )

    assert packet["go"] is False
    assert set(packet["missing_evidence"]) >= {
        "optimizer_report",
        "backtest_report",
        "attribution_ledger",
        "paper_ledger",
    }
