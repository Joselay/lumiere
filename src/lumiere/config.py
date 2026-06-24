from __future__ import annotations

from decimal import Decimal
from functools import cached_property

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from lumiere.risk import RiskConfig
from lumiere.strategy import MovingAverageCrossoverConfig, TradingStrategy
from lumiere.strategy_factory import STRATEGY_NAMES, build_strategy

SUPPORTED_INST_IDS = ("BTC-USDT", "ETH-USDT")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    okx_api_key: str = Field(min_length=1)
    okx_api_secret: str = Field(min_length=1)
    okx_passphrase: str = Field(min_length=1)
    okx_flag: str = "1"
    okx_inst_id: str = "BTC-USDT"
    okx_inst_ids: str = ""
    okx_td_mode: str = "cash"
    okx_order_tag: str = "lumieredemo"

    telegram_bot_token: str = Field(min_length=1)
    telegram_allowed_chat_ids: str = ""

    log_level: str = "INFO"

    engine_poll_interval_seconds: float = 30.0
    engine_candle_bar: str = "1m"
    engine_candle_limit: int = 80

    strategy_name: str = "moving_average_crossover"
    strategy_fast_window: int = 5
    strategy_slow_window: int = 20
    strategy_trade_size_btc: Decimal = Decimal("0.001")
    strategy_trade_size_eth: Decimal = Decimal("0.01")
    strategy_dust_threshold_btc: Decimal = Decimal("0.00001")
    strategy_dust_threshold_eth: Decimal = Decimal("0.0001")
    strategy_rsi_period: int = 14
    strategy_oversold_rsi: Decimal = Decimal("30")
    strategy_overbought_rsi: Decimal = Decimal("70")
    strategy_breakout_lookback: int = 20
    strategy_breakout_atr_period: int = 14
    strategy_breakout_atr_multiplier: Decimal = Decimal("0.5")
    strategy_breakout_min_atr_pct: Decimal = Decimal("0.001")

    risk_max_position_btc: Decimal = Decimal("0.005")
    risk_max_position_eth: Decimal = Decimal("0.05")
    risk_min_order_btc: Decimal = Decimal("0.00001")
    risk_min_order_eth: Decimal = Decimal("0.0001")
    risk_max_daily_loss_usdt: Decimal = Decimal("25")
    risk_cooldown_seconds: int = 300
    risk_max_consecutive_failures: int = 3
    risk_max_drawdown_usdt: Decimal = Decimal("0")
    risk_max_daily_trades: int = 0
    risk_max_spread_bps: Decimal = Decimal("0")
    risk_require_performance_gate: bool = False

    @field_validator("okx_flag")
    @classmethod
    def demo_only(cls, value: str) -> str:
        if value != "1":
            raise ValueError("Lumiere only supports OKX demo trading; set OKX_FLAG=1")
        return value

    @field_validator("okx_inst_id")
    @classmethod
    def supported_inst_id(cls, value: str) -> str:
        _validate_supported_inst_ids((value,))
        return value

    @field_validator("okx_inst_ids")
    @classmethod
    def supported_inst_ids(cls, value: str) -> str:
        if value.strip():
            _validate_supported_inst_ids(_parse_inst_ids(value))
        return value

    @field_validator("okx_order_tag")
    @classmethod
    def okx_order_tag_format(cls, value: str) -> str:
        if len(value) > 16 or not value.isalnum():
            raise ValueError("OKX_ORDER_TAG must be alphanumeric and at most 16 characters")
        return value

    @field_validator("strategy_name")
    @classmethod
    def supported_strategy_name(cls, value: str) -> str:
        if value not in STRATEGY_NAMES:
            raise ValueError(f"unsupported strategy: {value}")
        return value

    @cached_property
    def enabled_inst_ids(self) -> tuple[str, ...]:
        inst_ids = (
            _parse_inst_ids(self.okx_inst_ids) if self.okx_inst_ids.strip() else (self.okx_inst_id,)
        )
        _validate_supported_inst_ids(inst_ids)
        return inst_ids

    @cached_property
    def allowed_chat_ids(self) -> set[int]:
        if not self.telegram_allowed_chat_ids.strip():
            return set()
        return {
            int(part.strip()) for part in self.telegram_allowed_chat_ids.split(",") if part.strip()
        }

    def strategy_config(self) -> MovingAverageCrossoverConfig:
        return self.strategy_configs()[0]

    def strategy_configs(self) -> tuple[MovingAverageCrossoverConfig, ...]:
        return tuple(
            MovingAverageCrossoverConfig(
                inst_id=inst_id,
                fast_window=self.strategy_fast_window,
                slow_window=self.strategy_slow_window,
                trade_size_btc=self._trade_size_for(inst_id),
                dust_threshold_btc=self._dust_threshold_for(inst_id),
            )
            for inst_id in self.enabled_inst_ids
        )

    def strategies(self) -> tuple[TradingStrategy, ...]:
        return tuple(
            build_strategy(
                self.strategy_name,
                inst_id=inst_id,
                trade_size_btc=self._trade_size_for(inst_id),
                dust_threshold_btc=self._dust_threshold_for(inst_id),
                fast_window=self.strategy_fast_window,
                slow_window=self.strategy_slow_window,
                rsi_period=self.strategy_rsi_period,
                oversold_rsi=self.strategy_oversold_rsi,
                overbought_rsi=self.strategy_overbought_rsi,
                breakout_lookback=self.strategy_breakout_lookback,
                breakout_atr_period=self.strategy_breakout_atr_period,
                breakout_atr_multiplier=self.strategy_breakout_atr_multiplier,
                breakout_min_atr_pct=self.strategy_breakout_min_atr_pct,
            )
            for inst_id in self.enabled_inst_ids
        )

    def risk_config(self) -> RiskConfig:
        return RiskConfig(
            demo_flag=self.okx_flag,
            allowed_inst_ids=self.enabled_inst_ids,
            max_position_btc=self.risk_max_position_btc,
            min_order_btc=self.risk_min_order_btc,
            max_daily_loss_usdt=self.risk_max_daily_loss_usdt,
            cooldown_seconds=self.risk_cooldown_seconds,
            max_consecutive_failures=self.risk_max_consecutive_failures,
            max_position_by_inst_id={
                inst_id: self._max_position_for(inst_id) for inst_id in self.enabled_inst_ids
            },
            min_order_by_inst_id={
                inst_id: self._min_order_for(inst_id) for inst_id in self.enabled_inst_ids
            },
            max_drawdown_usdt=_positive_decimal_or_none(self.risk_max_drawdown_usdt),
            max_daily_trades=self.risk_max_daily_trades if self.risk_max_daily_trades > 0 else None,
            max_spread_bps=_positive_decimal_or_none(self.risk_max_spread_bps),
            performance_gate_required=self.risk_require_performance_gate,
        )

    def _trade_size_for(self, inst_id: str) -> Decimal:
        if inst_id.startswith("ETH-"):
            return self.strategy_trade_size_eth
        return self.strategy_trade_size_btc

    def _dust_threshold_for(self, inst_id: str) -> Decimal:
        if inst_id.startswith("ETH-"):
            return self.strategy_dust_threshold_eth
        return self.strategy_dust_threshold_btc

    def _max_position_for(self, inst_id: str) -> Decimal:
        if inst_id.startswith("ETH-"):
            return self.risk_max_position_eth
        return self.risk_max_position_btc

    def _min_order_for(self, inst_id: str) -> Decimal:
        if inst_id.startswith("ETH-"):
            return self.risk_min_order_eth
        return self.risk_min_order_btc


def _positive_decimal_or_none(value: Decimal) -> Decimal | None:
    return value if value > 0 else None


def _parse_inst_ids(raw: str) -> tuple[str, ...]:
    inst_ids = tuple(part.strip().upper() for part in raw.split(",") if part.strip())
    if not inst_ids:
        raise ValueError("at least one OKX instrument id is required")
    return inst_ids


def _validate_supported_inst_ids(inst_ids: tuple[str, ...]) -> None:
    unsupported = [inst_id for inst_id in inst_ids if inst_id not in SUPPORTED_INST_IDS]
    if unsupported:
        supported = ", ".join(SUPPORTED_INST_IDS)
        raise ValueError(
            f"unsupported instrument(s): {', '.join(unsupported)}; supported symbols: {supported}"
        )
