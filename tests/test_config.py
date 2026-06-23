from __future__ import annotations

from decimal import Decimal

import pytest

from lumiere.config import Settings


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
