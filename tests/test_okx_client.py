from __future__ import annotations

from decimal import Decimal

from lumiere.okx_client import _parse_account_balances, _parse_btc_position


def test_parse_account_balances_includes_spot_btc_and_eth_as_positions() -> None:
    equity, available, positions = _parse_account_balances(
        [
            {
                "totalEq": "1000",
                "details": [
                    {"ccy": "USDT", "availBal": "900"},
                    {"ccy": "BTC", "cashBal": "0.002"},
                    {"ccy": "ETH", "cashBal": "0.05"},
                ],
            }
        ],
        ("BTC-USDT", "ETH-USDT"),
    )

    assert equity == Decimal("1000")
    assert available == Decimal("900")
    assert {position.inst_id: position.size_btc for position in positions} == {
        "BTC-USDT": Decimal("0.002"),
        "ETH-USDT": Decimal("0.05"),
    }


def test_parse_derivatives_position_uses_okx_positions_payload() -> None:
    position = _parse_btc_position(
        "BTC-USDT-SWAP",
        [{"instId": "BTC-USDT-SWAP", "pos": "2", "avgPx": "100000", "upl": "12.5"}],
    )

    assert position is not None
    assert position.size_btc == Decimal("2")
    assert position.avg_px == Decimal("100000")
    assert position.unrealized_pnl_usdt == Decimal("12.5")
