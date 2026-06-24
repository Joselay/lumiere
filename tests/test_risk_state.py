from __future__ import annotations

from decimal import Decimal

from lumiere.risk_state import RiskStateStore


def test_risk_state_persists_high_water_mark_and_max_drawdown(tmp_path) -> None:
    store = RiskStateStore(tmp_path / "risk_state.json")

    first = store.update(Decimal("1000"))
    second = store.update(Decimal("950"))
    third = RiskStateStore(tmp_path / "risk_state.json").update(Decimal("1100"))

    assert first.high_water_equity_usdt == Decimal("1000")
    assert second.high_water_equity_usdt == Decimal("1000")
    assert second.max_drawdown_usdt == Decimal("50")
    assert third.high_water_equity_usdt == Decimal("1100")
    assert third.max_drawdown_usdt == Decimal("50")
