from __future__ import annotations

import json
from argparse import Namespace
from datetime import UTC, datetime
from decimal import Decimal

from lumiere.attribution import AttributionLedger
from lumiere.evidence_cli import build_evidence_packet
from lumiere.models import DecisionAction


def test_evidence_packet_can_be_generated_without_live_credentials(tmp_path) -> None:
    optimizer_report = tmp_path / "optimizer.json"
    optimizer_report.write_text(
        json.dumps({"accepted_configs": [{"inst_id": "BTC-USDT", "fast_window": 1}]}),
        encoding="utf-8",
    )
    ledger_path = tmp_path / "attribution.jsonl"
    ledger = AttributionLedger(ledger_path)
    ts = datetime.now(tz=UTC)
    for _ in range(15):
        ledger.record_fill(
            inst_id="BTC-USDT",
            side=DecisionAction.BUY,
            size_base=Decimal("1"),
            price=Decimal("100"),
            fee=Decimal("0"),
            ts=ts,
        )
        ledger.record_fill(
            inst_id="BTC-USDT",
            side=DecisionAction.SELL,
            size_base=Decimal("1"),
            price=Decimal("101"),
            fee=Decimal("0"),
            ts=ts,
        )

    packet = build_evidence_packet(
        Namespace(
            stage="paper",
            optimizer_report=str(optimizer_report),
            attribution_ledger=str(ledger_path),
            window_days=14,
            output=str(tmp_path / "evidence.json"),
        )
    )

    assert packet["missing_evidence"] == []
    assert packet["checks"]["min_trades_passed"] is True
    assert packet["checks"]["profit_factor_passed"] is True


def test_evidence_packet_rejects_missing_reports(tmp_path) -> None:
    packet = build_evidence_packet(
        Namespace(
            stage="paper",
            optimizer_report=str(tmp_path / "missing.json"),
            attribution_ledger=str(tmp_path / "missing-ledger.jsonl"),
            window_days=14,
            output=str(tmp_path / "evidence.json"),
        )
    )

    assert packet["go"] is False
    assert packet["missing_evidence"] == ["optimizer_report", "attribution_ledger"]
