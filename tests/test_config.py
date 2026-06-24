from __future__ import annotations

from decimal import Decimal

import pytest

from lumiere.config import Settings
from lumiere.strategies import RsiMeanReversionStrategy


def make_settings(**overrides) -> Settings:
    values = {
        "okx_api_key": "key",
        "okx_api_secret": "secret",
        "okx_passphrase": "passphrase",
        "telegram_bot_token": "token",
        "okx_inst_ids": "BTC-USDT,ETH-USDT",
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


def test_settings_enables_btc_and_eth_symbols() -> None:
    settings = make_settings()

    assert settings.enabled_inst_ids == ("BTC-USDT", "ETH-USDT")
    assert [config.inst_id for config in settings.strategy_configs()] == ["BTC-USDT", "ETH-USDT"]
    assert settings.risk_config().allowed_inst_ids == ("BTC-USDT", "ETH-USDT")


def test_settings_uses_eth_specific_trade_and_risk_sizes() -> None:
    settings = make_settings()
    strategy_by_symbol = {config.inst_id: config for config in settings.strategy_configs()}
    risk = settings.risk_config()

    assert strategy_by_symbol["BTC-USDT"].trade_size_btc == Decimal("0.001")
    assert strategy_by_symbol["ETH-USDT"].trade_size_btc == Decimal("0.01")
    assert risk.max_position_for("BTC-USDT") == Decimal("0.005")
    assert risk.max_position_for("ETH-USDT") == Decimal("0.05")


def test_settings_rejects_unsupported_symbols() -> None:
    with pytest.raises(ValueError, match="supported symbols"):
        make_settings(okx_inst_ids="BTC-USDT,SOL-USDT")


def test_strategy_can_be_selected_by_config() -> None:
    settings = make_settings(strategy_name="rsi_mean_reversion", strategy_rsi_period=5)

    strategies = settings.strategies()

    assert all(isinstance(strategy, RsiMeanReversionStrategy) for strategy in strategies)
    assert strategies[0].config.rsi_period == 5


def test_settings_maps_optional_profitability_risk_controls() -> None:
    settings = make_settings(
        risk_max_drawdown_usdt=Decimal("50"),
        risk_max_daily_trades=10,
        risk_max_spread_bps=Decimal("8"),
        risk_require_performance_gate=True,
    )
    risk = settings.risk_config()

    assert risk.max_drawdown_usdt == Decimal("50")
    assert risk.max_daily_trades == 10
    assert risk.max_spread_bps == Decimal("8")
    assert risk.performance_gate_required is True
