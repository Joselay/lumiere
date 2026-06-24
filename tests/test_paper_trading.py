from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

from lumiere.backtest import CostModel
from lumiere.models import DecisionAction, MarketCandle, StrategyDecision
from lumiere.paper_gate import PerformanceGateConfig
from lumiere.paper_trading import PaperTradingConfig, PaperTradingLedger


def candle(price: str, ts: datetime) -> MarketCandle:
    value = Decimal(price)
    return MarketCandle(ts=ts, open=value, high=value, low=value, close=value)


def decision(action: DecisionAction) -> StrategyDecision:
    return StrategyDecision(action, "BTC-USDT", Decimal("1"), "test_signal")


def config(path, *, max_age: timedelta = timedelta(days=7)) -> PaperTradingConfig:
    return PaperTradingConfig(
        path=path,
        cost_model=CostModel(
            taker_fee_bps=Decimal("0"),
            spread_bps=Decimal("0"),
            slippage_bps=Decimal("0"),
        ),
        gate=PerformanceGateConfig(min_trades=2, min_profit_factor=Decimal("1")),
        max_evidence_age=max_age,
    )


def test_paper_gate_blocks_without_evidence(tmp_path) -> None:
    ledger = PaperTradingLedger(config(tmp_path / "paper.jsonl"))

    gate = ledger.gate_decision(now=datetime(2026, 1, 1, tzinfo=UTC))

    assert gate.allowed is False
    assert gate.reason == "paper_gate_no_evidence"


def test_paper_gate_passes_and_survives_restart(tmp_path) -> None:
    path = tmp_path / "paper.jsonl"
    ledger = PaperTradingLedger(config(path))
    start = datetime(2026, 1, 1, tzinfo=UTC)
    ledger.record_decision(decision(DecisionAction.BUY), candle("100", start), strategy_name="test")
    ledger.record_decision(
        decision(DecisionAction.SELL),
        candle("110", start + timedelta(minutes=1)),
        strategy_name="test",
    )

    restarted = PaperTradingLedger(config(path))
    gate = restarted.gate_decision(now=start + timedelta(minutes=2))

    assert gate.allowed is True
    assert gate.reason == "performance_gate_passed"


def test_paper_gate_decays_when_evidence_is_stale(tmp_path) -> None:
    path = tmp_path / "paper.jsonl"
    ledger = PaperTradingLedger(config(path, max_age=timedelta(minutes=5)))
    start = datetime(2026, 1, 1, tzinfo=UTC)
    ledger.record_decision(decision(DecisionAction.BUY), candle("100", start), strategy_name="test")
    ledger.record_decision(
        decision(DecisionAction.SELL),
        candle("110", start + timedelta(minutes=1)),
        strategy_name="test",
    )

    gate = ledger.gate_decision(now=start + timedelta(minutes=10))

    assert gate.allowed is False
    assert gate.reason == "paper_gate_decayed"
