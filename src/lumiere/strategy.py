from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from lumiere.models import AccountSnapshot, DecisionAction, MarketCandle, StrategyDecision


@dataclass(frozen=True, slots=True)
class MovingAverageCrossoverConfig:
    inst_id: str = "BTC-USDT"
    fast_window: int = 5
    slow_window: int = 20
    trade_size_btc: Decimal = Decimal("0.001")

    def __post_init__(self) -> None:
        if self.fast_window <= 0:
            raise ValueError("fast_window must be positive")
        if self.slow_window <= self.fast_window:
            raise ValueError("slow_window must be greater than fast_window")
        if self.trade_size_btc <= 0:
            raise ValueError("trade_size_btc must be positive")


class MovingAverageCrossoverStrategy:
    """Deterministic long-only BTC spot strategy.

    The strategy buys when the fast simple moving average is above the slow SMA and
    no BTC is held. It sells the configured size, capped by the current position,
    when the fast SMA is below the slow SMA and BTC is held. Equal averages hold.
    """

    name = "moving_average_crossover"

    def __init__(self, config: MovingAverageCrossoverConfig) -> None:
        self.config = config

    def describe(self) -> dict[str, str | int]:
        return {
            "name": self.name,
            "inst_id": self.config.inst_id,
            "fast_window": self.config.fast_window,
            "slow_window": self.config.slow_window,
            "trade_size_btc": str(self.config.trade_size_btc),
        }

    def decide(
        self,
        candles: list[MarketCandle],
        account: AccountSnapshot,
    ) -> StrategyDecision:
        if len(candles) < self.config.slow_window:
            return StrategyDecision.hold(
                self.config.inst_id,
                "not_enough_candles",
                {"candles": len(candles), "required": self.config.slow_window},
            )

        closes = [c.close for c in candles]
        fast_ma = simple_average(closes[-self.config.fast_window :])
        slow_ma = simple_average(closes[-self.config.slow_window :])
        position = account.btc_position_size
        inputs = {
            "fast_ma": str(fast_ma),
            "slow_ma": str(slow_ma),
            "position_btc": str(position),
            "fast_window": self.config.fast_window,
            "slow_window": self.config.slow_window,
        }

        if fast_ma > slow_ma and position <= 0:
            return StrategyDecision(
                action=DecisionAction.BUY,
                inst_id=self.config.inst_id,
                size_btc=self.config.trade_size_btc,
                reason="fast_ma_above_slow_ma_and_flat",
                inputs=inputs,
            )

        if fast_ma < slow_ma and position > 0:
            return StrategyDecision(
                action=DecisionAction.SELL,
                inst_id=self.config.inst_id,
                size_btc=min(self.config.trade_size_btc, position),
                reason="fast_ma_below_slow_ma_and_long",
                inputs=inputs,
            )

        return StrategyDecision.hold(self.config.inst_id, "no_position_change", inputs)


def simple_average(values: list[Decimal]) -> Decimal:
    if not values:
        raise ValueError("cannot average empty values")
    return sum(values, Decimal("0")) / Decimal(len(values))
