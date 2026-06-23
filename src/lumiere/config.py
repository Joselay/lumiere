from __future__ import annotations

from decimal import Decimal
from functools import cached_property

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from lumiere.risk import RiskConfig
from lumiere.strategy import MovingAverageCrossoverConfig


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
    okx_td_mode: str = "cash"
    okx_order_tag: str = "lumiere-demo"

    telegram_bot_token: str = Field(min_length=1)
    telegram_allowed_chat_ids: str = ""

    engine_poll_interval_seconds: float = 30.0
    engine_candle_bar: str = "1m"
    engine_candle_limit: int = 80

    strategy_fast_window: int = 5
    strategy_slow_window: int = 20
    strategy_trade_size_btc: Decimal = Decimal("0.001")

    risk_max_position_btc: Decimal = Decimal("0.005")
    risk_max_daily_loss_usdt: Decimal = Decimal("25")
    risk_cooldown_seconds: int = 300
    risk_max_consecutive_failures: int = 3

    @field_validator("okx_flag")
    @classmethod
    def demo_only(cls, value: str) -> str:
        if value != "1":
            raise ValueError("Lumiere only supports OKX demo trading; set OKX_FLAG=1")
        return value

    @field_validator("okx_inst_id")
    @classmethod
    def btc_only(cls, value: str) -> str:
        if not value.startswith("BTC-"):
            raise ValueError("Lumiere is BTC-only; OKX_INST_ID must start with BTC-")
        return value

    @cached_property
    def allowed_chat_ids(self) -> set[int]:
        if not self.telegram_allowed_chat_ids.strip():
            return set()
        return {
            int(part.strip()) for part in self.telegram_allowed_chat_ids.split(",") if part.strip()
        }

    def strategy_config(self) -> MovingAverageCrossoverConfig:
        return MovingAverageCrossoverConfig(
            inst_id=self.okx_inst_id,
            fast_window=self.strategy_fast_window,
            slow_window=self.strategy_slow_window,
            trade_size_btc=self.strategy_trade_size_btc,
        )

    def risk_config(self) -> RiskConfig:
        return RiskConfig(
            demo_flag=self.okx_flag,
            allowed_inst_ids=(self.okx_inst_id,),
            max_position_btc=self.risk_max_position_btc,
            max_daily_loss_usdt=self.risk_max_daily_loss_usdt,
            cooldown_seconds=self.risk_cooldown_seconds,
            max_consecutive_failures=self.risk_max_consecutive_failures,
        )
