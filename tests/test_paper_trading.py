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


def decision(
    action: DecisionAction,
    *,
    inst_id: str = "BTC-USDT",
    size: Decimal = Decimal("1"),
) -> StrategyDecision:
    return StrategyDecision(action, inst_id, size, "test_signal")


def config(
    path,
    *,
    max_age: timedelta = timedelta(days=7),
    starting_equity: Decimal = Decimal("1000"),
) -> PaperTradingConfig:
    return PaperTradingConfig(
        path=path,
        starting_equity_usdt=starting_equity,
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
    account = restarted.account_snapshot(now=start + timedelta(minutes=2))

    assert gate.allowed is True
    assert gate.reason == "performance_gate_passed"
    assert account.equity_usdt == Decimal("1010")
    assert account.available_usdt == Decimal("1010")
    assert account.daily_realized_pnl_usdt == Decimal("10")
    assert account.positions == ()


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


def test_paper_sell_without_inventory_is_rejected_and_not_counted(tmp_path) -> None:
    path = tmp_path / "paper.jsonl"
    ledger = PaperTradingLedger(config(path))
    start = datetime(2026, 1, 1, tzinfo=UTC)

    ledger.record_decision(
        decision(DecisionAction.SELL),
        candle("100", start),
        strategy_name="test",
    )

    account = ledger.account_snapshot(now=start + timedelta(minutes=1))
    gate = ledger.gate_decision(now=start + timedelta(minutes=1))
    events = ledger.events

    assert account.equity_usdt == Decimal("1000")
    assert account.available_usdt == Decimal("1000")
    assert account.daily_trade_count == 0
    assert account.positions == ()
    assert gate.reason == "paper_gate_no_fills"
    assert [event["type"] for event in events] == [
        "decision",
        "simulated_rejection",
        "portfolio_state",
    ]
    assert events[1]["reason"] == "paper_no_inventory"


def test_paper_portfolio_tracks_multiple_symbols_and_reloads(tmp_path) -> None:
    path = tmp_path / "paper.jsonl"
    ledger = PaperTradingLedger(config(path, starting_equity=Decimal("10000")))
    start = datetime(2026, 1, 1, tzinfo=UTC)

    ledger.record_decision(
        decision(DecisionAction.BUY, inst_id="BTC-USDT", size=Decimal("1")),
        candle("100", start),
        strategy_name="test",
    )
    ledger.record_decision(
        decision(DecisionAction.BUY, inst_id="ETH-USDT", size=Decimal("2")),
        candle("50", start + timedelta(minutes=1)),
        strategy_name="test",
    )

    restarted = PaperTradingLedger(config(path, starting_equity=Decimal("10000")))
    account = restarted.account_snapshot(
        mark_prices={"BTC-USDT": Decimal("110"), "ETH-USDT": Decimal("40")},
        now=start + timedelta(minutes=2),
    )

    assert account.available_usdt == Decimal("9800")
    assert account.equity_usdt == Decimal("9990")
    assert account.position_for("BTC-USDT").size_base == Decimal("1")  # type: ignore[union-attr]
    assert account.position_for("BTC-USDT").avg_px == Decimal("100")  # type: ignore[union-attr]
    assert account.position_for("BTC-USDT").unrealized_pnl_usdt == Decimal("10")  # type: ignore[union-attr]
    assert account.position_for("ETH-USDT").size_base == Decimal("2")  # type: ignore[union-attr]
    assert account.position_for("ETH-USDT").avg_px == Decimal("50")  # type: ignore[union-attr]
    assert account.position_for("ETH-USDT").unrealized_pnl_usdt == Decimal("-20")  # type: ignore[union-attr]
