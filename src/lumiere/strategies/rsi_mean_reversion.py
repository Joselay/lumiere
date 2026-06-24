from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from lumiere.models import AccountSnapshot, DecisionAction, MarketCandle, StrategyDecision


@dataclass(frozen=True, slots=True)
class RsiMeanReversionConfig:
    inst_id: str = "BTC-USDT"
    rsi_period: int = 14
    oversold_rsi: Decimal = Decimal("30")
    overbought_rsi: Decimal = Decimal("70")
    trade_size_btc: Decimal = Decimal("0.001")
    dust_threshold_btc: Decimal = Decimal("0.00001")
    max_spread_bps: Decimal | None = None

    def __post_init__(self) -> None:
        if self.rsi_period <= 1:
            raise ValueError("rsi_period must be greater than 1")
        if self.oversold_rsi <= 0 or self.overbought_rsi >= 100:
            raise ValueError("RSI thresholds must be inside 0..100")
        if self.oversold_rsi >= self.overbought_rsi:
            raise ValueError("oversold_rsi must be below overbought_rsi")
        if self.trade_size_btc <= 0:
            raise ValueError("trade_size_btc must be positive")
        if self.dust_threshold_btc < 0:
            raise ValueError("dust_threshold_btc cannot be negative")
        if self.max_spread_bps is not None and self.max_spread_bps <= 0:
            raise ValueError("max_spread_bps must be positive when configured")


class RsiMeanReversionStrategy:
    """Long-only RSI mean-reversion candidate for ranging, liquid regimes."""

    name = "rsi_mean_reversion"
    allowed_regimes = ("ranging", "normal_volatility", "spread_liquidity_ok")

    def __init__(self, config: RsiMeanReversionConfig) -> None:
        self.config = config

    def describe(self) -> dict[str, str | int | tuple[str, ...]]:
        return {
            "name": self.name,
            "inst_id": self.config.inst_id,
            "rsi_period": self.config.rsi_period,
            "oversold_rsi": str(self.config.oversold_rsi),
            "overbought_rsi": str(self.config.overbought_rsi),
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
        if len(candles) < self.config.rsi_period + 1:
            return StrategyDecision.hold(
                self.config.inst_id,
                "not_enough_candles",
                {"candles": len(candles), "required": self.config.rsi_period + 1},
            )
        spread_ok = _spread_ok(account.spread_bps, self.config.max_spread_bps)
        rsi = relative_strength_index([candle.close for candle in candles], self.config.rsi_period)
        position = account.position_size(self.config.inst_id)
        effective_position = (
            Decimal("0") if abs(position) < self.config.dust_threshold_btc else position
        )
        inputs = {
            "rsi": str(rsi),
            "rsi_period": self.config.rsi_period,
            "position_base": str(position),
            "effective_position_base": str(effective_position),
            "allowed_regimes": self.allowed_regimes,
            "regime": "ranging" if rsi <= self.config.oversold_rsi else "neutral",
            "spread_liquidity_ok": spread_ok,
            "decision_price": str(candles[-1].close),
            "volatility_bps": str(abs(rsi - Decimal("50")) * Decimal("10")),
            "expected_edge_bps": str(abs(rsi - Decimal("50")) * Decimal("10")),
        }
        if not spread_ok:
            return StrategyDecision.hold(self.config.inst_id, "spread_liquidity_not_ok", inputs)
        if rsi <= self.config.oversold_rsi and effective_position <= 0:
            return StrategyDecision(
                action=DecisionAction.BUY,
                inst_id=self.config.inst_id,
                size_btc=self.config.trade_size_btc,
                reason="rsi_oversold_in_ranging_regime",
                inputs=inputs,
            )
        if rsi >= self.config.overbought_rsi and effective_position > 0:
            return StrategyDecision(
                action=DecisionAction.SELL,
                inst_id=self.config.inst_id,
                size_btc=min(self.config.trade_size_btc, effective_position),
                reason="rsi_overbought_exit",
                inputs=inputs,
            )
        return StrategyDecision.hold(self.config.inst_id, "rsi_neutral", inputs)


def relative_strength_index(closes: list[Decimal], period: int) -> Decimal:
    if len(closes) < period + 1:
        raise ValueError("not enough closes for RSI")
    gains = Decimal("0")
    losses = Decimal("0")
    window = closes[-(period + 1) :]
    for previous, current in zip(window[:-1], window[1:], strict=True):
        change = current - previous
        if change > 0:
            gains += change
        else:
            losses += abs(change)
    if losses == 0:
        return Decimal("100")
    average_gain = gains / Decimal(period)
    average_loss = losses / Decimal(period)
    rs = average_gain / average_loss
    return Decimal("100") - (Decimal("100") / (Decimal("1") + rs))


def _spread_ok(spread_bps: Decimal | None, max_spread_bps: Decimal | None) -> bool:
    return max_spread_bps is None or (spread_bps is not None and spread_bps <= max_spread_bps)
