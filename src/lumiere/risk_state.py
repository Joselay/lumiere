from __future__ import annotations

import json
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path


@dataclass(frozen=True, slots=True)
class RiskState:
    high_water_equity_usdt: Decimal
    max_drawdown_usdt: Decimal


class RiskStateStore:
    """Small JSON store for live account high-water mark and drawdown state."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> RiskState | None:
        if not self.path.exists():
            return None
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        return RiskState(
            high_water_equity_usdt=Decimal(str(payload["high_water_equity_usdt"])),
            max_drawdown_usdt=Decimal(str(payload["max_drawdown_usdt"])),
        )

    def update(self, equity_usdt: Decimal) -> RiskState:
        current = self.load()
        high_water = equity_usdt
        max_drawdown = Decimal("0")
        if current is not None:
            high_water = max(current.high_water_equity_usdt, equity_usdt)
            max_drawdown = current.max_drawdown_usdt
        drawdown = max(high_water - equity_usdt, Decimal("0"))
        state = RiskState(
            high_water_equity_usdt=high_water,
            max_drawdown_usdt=max(max_drawdown, drawdown),
        )
        self.path.write_text(
            json.dumps(
                {
                    "high_water_equity_usdt": str(state.high_water_equity_usdt),
                    "max_drawdown_usdt": str(state.max_drawdown_usdt),
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        return state
