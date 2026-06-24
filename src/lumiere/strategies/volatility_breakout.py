from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from lumiere.models import AccountSnapshot, DecisionAction, MarketCandle, StrategyDecision


@dataclass(frozen=True, slots=True)
class VolatilityBreakoutConfig:
    inst_id: str = "BTC-USDT"
    lookback: int = 20
    atr_period: int = 14
    atr_multiplier: Decimal = Decimal("0.5")
    min_atr_pct: Decimal = Decimal("0.001")
    trade_size_btc: Decimal = Decimal("0.001")
    dust_threshold_btc: Decimal = Decimal("0.00001")
    max_spread_bps: Decimal | None = None

    def __post_init__(self) -> None:
        if self.lookback <= 1:
            raise ValueError("lookback must be greater than 1")
        if self.atr_period <= 0:
            raise ValueError("atr_period must be positive")
        if self.atr_multiplier < 0:
            raise ValueError("atr_multiplier cannot be negative")
        if self.min_atr_pct < 0:
            raise ValueError("min_atr_pct cannot be negative")
        if self.trade_size_btc <= 0:
            raise ValueError("trade_size_btc must be positive")
        if self.dust_threshold_btc < 0:
            raise ValueError("dust_threshold_btc cannot be negative")
        if self.max_spread_bps is not None and self.max_spread_bps <= 0:
            raise ValueError("max_spread_bps must be positive when configured")


class VolatilityBreakoutStrategy:
    """ATR breakout candidate for trending, sufficiently volatile, liquid regimes."""

    name = "volatility_breakout"
    allowed_regimes = ("trending", "high_volatility", "spread_liquidity_ok")

    def __init__(self, config: VolatilityBreakoutConfig) -> None:
        self.config = config

    def describe(self) -> dict[str, str | int | tuple[str, ...]]:
        return {
            "name": self.name,
            "inst_id": self.config.inst_id,
            "lookback": self.config.lookback,
            "atr_period": self.config.atr_period,
            "atr_multiplier": str(self.config.atr_multiplier),
            "min_atr_pct": str(self.config.min_atr_pct),
            "trade_size_btc": str(self.config.trade_size_btc),
            "dust_threshold_btc": str(self.config.dust_threshold_btc),
            "max_spread_bps": ""
            if self.config.max_spread_bps is None
            else str(self.config.max_spread_bps),
            "allowed_regimes": self.allowed_regimes,
        }

    def decide(
        self,
        candles: list[MarketCandle],
        account: AccountSnapshot,
    ) -> StrategyDecision:
        required = max(self.config.lookback + 1, self.config.atr_period + 1)
        if len(candles) < required:
            return StrategyDecision.hold(
                self.config.inst_id,
                "not_enough_candles",
                {"candles": len(candles), "required": required},
            )
        spread_ok = _spread_ok(account.spread_bps, self.config.max_spread_bps)
        atr = average_true_range(candles, self.config.atr_period)
        latest = candles[-1]
        previous_window = candles[-(self.config.lookback + 1) : -1]
        prior_high = max(candle.high for candle in previous_window)
        prior_low = min(candle.low for candle in previous_window)
        atr_pct = Decimal("0") if latest.close == 0 else atr / latest.close
        volatility_ok = atr_pct >= self.config.min_atr_pct
        position = account.position_size(self.config.inst_id)
        effective_position = (
            Decimal("0") if abs(position) < self.config.dust_threshold_btc else position
        )
        buy_trigger = prior_high + atr * self.config.atr_multiplier
        sell_trigger = prior_low - atr * self.config.atr_multiplier
        inputs = {
            "atr": str(atr),
            "atr_pct": str(atr_pct),
            "prior_high": str(prior_high),
            "prior_low": str(prior_low),
            "buy_trigger": str(buy_trigger),
            "sell_trigger": str(sell_trigger),
            "position_base": str(position),
            "effective_position_base": str(effective_position),
            "allowed_regimes": self.allowed_regimes,
            "regime": "trending_high_volatility" if volatility_ok else "low_volatility",
            "spread_liquidity_ok": spread_ok,
            "decision_price": str(latest.close),
            "volatility_bps": str(atr_pct * Decimal("10000")),
            "expected_edge_bps": str(atr_pct * Decimal("10000")),
        }
        if not spread_ok:
            return StrategyDecision.hold(self.config.inst_id, "spread_liquidity_not_ok", inputs)
        if not volatility_ok:
            return StrategyDecision.hold(self.config.inst_id, "volatility_regime_too_low", inputs)
        if latest.close > buy_trigger and effective_position <= 0:
            return StrategyDecision(
                action=DecisionAction.BUY,
                inst_id=self.config.inst_id,
                size_btc=self.config.trade_size_btc,
                reason="atr_breakout_above_prior_high",
                inputs=inputs,
            )
        if latest.close < sell_trigger and effective_position > 0:
            return StrategyDecision(
                action=DecisionAction.SELL,
                inst_id=self.config.inst_id,
                size_btc=min(self.config.trade_size_btc, effective_position),
                reason="atr_breakdown_exit",
                inputs=inputs,
            )
        return StrategyDecision.hold(self.config.inst_id, "no_breakout", inputs)


def average_true_range(candles: list[MarketCandle], period: int) -> Decimal:
    if len(candles) < period + 1:
        raise ValueError("not enough candles for ATR")
    ranges: list[Decimal] = []
    window = candles[-(period + 1) :]
    previous_close = window[0].close
    for candle in window[1:]:
        true_range = max(
            candle.high - candle.low,
            abs(candle.high - previous_close),
            abs(candle.low - previous_close),
        )
        ranges.append(true_range)
        previous_close = candle.close
    return sum(ranges, Decimal("0")) / Decimal(len(ranges))


def _spread_ok(spread_bps: Decimal | None, max_spread_bps: Decimal | None) -> bool:
    return max_spread_bps is None or (spread_bps is not None and spread_bps <= max_spread_bps)
