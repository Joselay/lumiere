from __future__ import annotations

from decimal import Decimal

from lumiere.strategies import (
    RsiMeanReversionConfig,
    RsiMeanReversionStrategy,
    VolatilityBreakoutConfig,
    VolatilityBreakoutStrategy,
)
from lumiere.strategy import (
    MovingAverageCrossoverConfig,
    MovingAverageCrossoverStrategy,
    TradingStrategy,
)

STRATEGY_NAMES = (
    "moving_average_crossover",
    "rsi_mean_reversion",
    "volatility_breakout",
)


def build_strategy(
    name: str,
    *,
    inst_id: str,
    trade_size_btc: Decimal,
    dust_threshold_btc: Decimal,
    fast_window: int = 5,
    slow_window: int = 20,
    rsi_period: int = 14,
    oversold_rsi: Decimal = Decimal("30"),
    overbought_rsi: Decimal = Decimal("70"),
    breakout_lookback: int = 20,
    breakout_atr_period: int = 14,
    breakout_atr_multiplier: Decimal = Decimal("0.5"),
    breakout_min_atr_pct: Decimal = Decimal("0.001"),
) -> TradingStrategy:
    if name == "moving_average_crossover":
        return MovingAverageCrossoverStrategy(
            MovingAverageCrossoverConfig(
                inst_id=inst_id,
                fast_window=fast_window,
                slow_window=slow_window,
                trade_size_btc=trade_size_btc,
                dust_threshold_btc=dust_threshold_btc,
            )
        )
    if name == "rsi_mean_reversion":
        return RsiMeanReversionStrategy(
            RsiMeanReversionConfig(
                inst_id=inst_id,
                rsi_period=rsi_period,
                oversold_rsi=oversold_rsi,
                overbought_rsi=overbought_rsi,
                trade_size_btc=trade_size_btc,
                dust_threshold_btc=dust_threshold_btc,
            )
        )
    if name == "volatility_breakout":
        return VolatilityBreakoutStrategy(
            VolatilityBreakoutConfig(
                inst_id=inst_id,
                lookback=breakout_lookback,
                atr_period=breakout_atr_period,
                atr_multiplier=breakout_atr_multiplier,
                min_atr_pct=breakout_min_atr_pct,
                trade_size_btc=trade_size_btc,
                dust_threshold_btc=dust_threshold_btc,
            )
        )
    raise ValueError(f"unsupported strategy: {name}")
