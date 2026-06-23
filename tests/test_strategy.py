from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

from lumiere.models import AccountSnapshot, DecisionAction, MarketCandle, Position
from lumiere.strategy import MovingAverageCrossoverConfig, MovingAverageCrossoverStrategy


def candles(closes: list[str]) -> list[MarketCandle]:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    return [
        MarketCandle(
            ts=start + timedelta(minutes=i),
            open=Decimal(close),
            high=Decimal(close),
            low=Decimal(close),
            close=Decimal(close),
        )
        for i, close in enumerate(closes)
    ]


def test_strategy_is_deterministic_for_same_inputs() -> None:
    strategy = MovingAverageCrossoverStrategy(
        MovingAverageCrossoverConfig(fast_window=2, slow_window=3, trade_size_btc=Decimal("0.01"))
    )
    account = AccountSnapshot(equity_usdt=Decimal("1000"), available_usdt=Decimal("1000"))
    market = candles(["100", "101", "110"])

    first = strategy.decide(market, account)
    second = strategy.decide(market, account)

    assert first == second
    assert first.action is DecisionAction.BUY
    assert first.reason == "fast_ma_above_slow_ma_and_flat"


def test_strategy_sells_when_fast_average_crosses_below_slow_and_long() -> None:
    strategy = MovingAverageCrossoverStrategy(
        MovingAverageCrossoverConfig(fast_window=2, slow_window=3, trade_size_btc=Decimal("0.01"))
    )
    account = AccountSnapshot(
        equity_usdt=Decimal("1000"),
        available_usdt=Decimal("1000"),
        btc_position=Position(inst_id="BTC-USDT", size_btc=Decimal("0.004")),
    )

    decision = strategy.decide(candles(["110", "101", "100"]), account)

    assert decision.action is DecisionAction.SELL
    assert decision.size_btc == Decimal("0.004")


def test_strategy_holds_without_enough_candles() -> None:
    strategy = MovingAverageCrossoverStrategy(
        MovingAverageCrossoverConfig(fast_window=2, slow_window=3)
    )
    account = AccountSnapshot(equity_usdt=Decimal("1000"), available_usdt=Decimal("1000"))

    decision = strategy.decide(candles(["100", "101"]), account)

    assert decision.action is DecisionAction.HOLD
    assert decision.reason == "not_enough_candles"


def test_strategy_ignores_dust_position_for_sell_signal() -> None:
    strategy = MovingAverageCrossoverStrategy(
        MovingAverageCrossoverConfig(
            fast_window=2,
            slow_window=3,
            dust_threshold_btc=Decimal("0.00001"),
        )
    )
    account = AccountSnapshot(
        equity_usdt=Decimal("1000"),
        available_usdt=Decimal("1000"),
        btc_position=Position(inst_id="BTC-USDT", size_btc=Decimal("0.00000000268")),
    )

    decision = strategy.decide(candles(["110", "101", "100"]), account)

    assert decision.action is DecisionAction.HOLD
    assert decision.reason == "no_position_change"


def test_strategy_treats_dust_position_as_flat_for_buy_signal() -> None:
    strategy = MovingAverageCrossoverStrategy(
        MovingAverageCrossoverConfig(
            fast_window=2,
            slow_window=3,
            dust_threshold_btc=Decimal("0.00001"),
        )
    )
    account = AccountSnapshot(
        equity_usdt=Decimal("1000"),
        available_usdt=Decimal("1000"),
        btc_position=Position(inst_id="BTC-USDT", size_btc=Decimal("0.00000000268")),
    )

    decision = strategy.decide(candles(["100", "101", "110"]), account)

    assert decision.action is DecisionAction.BUY


def test_strategy_uses_position_for_its_own_symbol() -> None:
    strategy = MovingAverageCrossoverStrategy(
        MovingAverageCrossoverConfig(
            inst_id="ETH-USDT",
            fast_window=2,
            slow_window=3,
            trade_size_btc=Decimal("0.01"),
        )
    )
    account = AccountSnapshot(
        equity_usdt=Decimal("1000"),
        available_usdt=Decimal("1000"),
        positions=(Position(inst_id="ETH-USDT", size_btc=Decimal("0.02")),),
    )

    decision = strategy.decide(candles(["110", "101", "100"]), account)

    assert decision.action is DecisionAction.SELL
    assert decision.inst_id == "ETH-USDT"
    assert decision.size_btc == Decimal("0.01")
