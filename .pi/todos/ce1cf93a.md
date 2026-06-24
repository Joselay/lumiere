{
  "id": "ce1cf93a",
  "title": "Generalize optimizer to all strategies, exits, timeframes, and expectancy calibration",
  "tags": [
    "profitability",
    "strategy",
    "optimizer"
  ],
  "status": "closed",
  "created_at": "2026-06-24T02:47:31.125Z"
}

## Why
`lumiere-optimize` only searched moving-average fast/slow windows. RSI and volatility-breakout were selectable but not optimized, and `expected_edge_bps` was heuristic rather than measured forward expectancy.

## Completed
- Generalized optimizer candidate evaluation to support moving-average crossover, RSI mean-reversion, and volatility breakout candidates.
- Added optimizer grids for RSI settings, volatility breakout settings, optional stop-loss/take-profit/trailing/time exits, multiple bars/timeframes, and per-symbol sizing.
- Added empirical expectancy calibration from historical conditional forward returns after modeled fees, spread, slippage, and market impact.
- Added accepted candidate payloads with `.env`-ready settings, `expected_edge_bps`, `expected_edge_source=historical_forward_return_after_costs`, `optimizer_passed=true`, and anti-overfit metadata.
- Added ranking/penalty metrics for expected edge, turnover, largest loss/tail loss, drawdown duration, drawdown, risk rejections, profit factor, Sharpe/Sortino, and parameter stability.
- Added live/demo risk-audit enforcement requiring the current configured strategy to match an optimizer-passed accepted candidate with calibrated positive expected edge.
- Documented the generalized optimizer workflow in `README.md`.

## Verification
- `uv run ruff check .`
- `uv run pytest -q` → 118 passed
