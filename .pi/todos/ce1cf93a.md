{
  "id": "ce1cf93a",
  "title": "Generalize optimizer to all strategies, exits, timeframes, and expectancy calibration",
  "tags": [
    "profitability",
    "strategy",
    "optimizer"
  ],
  "status": "open",
  "created_at": "2026-06-24T02:47:31.125Z"
}

## Why
`lumiere-optimize` only searches moving-average fast/slow windows. RSI and volatility-breakout are selectable but not optimized, and `expected_edge_bps` is currently a heuristic (MA distance, RSI distance, ATR) rather than measured forward expectancy.

## Scope
- Add parameter grids for RSI, volatility breakout, exits, stop sizes, take-profit, trailing stop, max bars, bars/timeframes, and per-symbol sizing.
- Calibrate expected edge from historical conditional forward returns after fees/spread/slippage, not from indicator distance alone.
- Add regime-aware strategy selection: only trade a strategy when its historically validated regime classifier agrees.
- Penalize turnover, tail losses, drawdown duration, and parameter instability.
- Produce accepted candidate configs directly usable by `.env` or a config file.

## Ready when
No strategy can trade live/demo unless its expected edge was empirically calibrated and its optimizer result passed anti-overfit gates.
