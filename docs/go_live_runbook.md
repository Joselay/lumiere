# Lumiere staged go-live profitability runbook

Profit is never guaranteed. This runbook defines the minimum evidence required before increasing risk from historical research to paper trading, small demo size, and larger demo size.

## Stage gates

| Stage | Required evidence | Go / no-go thresholds |
| --- | --- | --- |
| 1. Backtest research | `lumiere-backtest` split + walk-forward reports and `lumiere-optimize` candidate report | Out-of-sample net PnL > 0 after costs; beats no-trade and buy-and-hold; profit factor >= 1.2; max drawdown <= 5% of starting equity; Sharpe > 0.5 or Sortino > 0.75 when available; >= 30 OOS trades; average slippage <= 10 bps; parameter-stability and walk-forward gates pass. |
| 2. Paper shadow | Persistent paper ledger and attribution report | Minimum 14 calendar days and 30 trades; net PnL > 0 after modeled fees/spread/slippage; profit factor >= 1.2; max drawdown <= 5%; no abnormal slippage alert; performance gate remains passed for 3 consecutive checks. |
| 3. Small demo | OKX demo with minimum configured size | Minimum 7 calendar days and 20 trades; live/demo attribution net PnL >= 0; realized slippage within 2x modeled slippage; no API-error cluster; no drawdown, spread, or market-regime rollback trigger. |
| 4. Larger demo | Explicit human approval with evidence packet | Previous stage still passes; increase size by no more than 2x; keep `RISK_REQUIRE_PERFORMANCE_GATE=true`; keep max risk per trade <= 1% and portfolio exposure <= configured cap. |

## Rollback / pause rules

Pause trading or reduce to the previous stage if any condition occurs:

- Daily loss or drawdown gate trips.
- Rolling performance gate fails or decays.
- Attribution alerts show negative rolling expectancy, abnormal slippage, spread spikes, or cost-gate rejections.
- Profit factor falls below 1.0 over the active observation window.
- Strategy trades outside its allowed market regime.
- OKX API errors repeat enough to pause the engine or order/fill attribution becomes incomplete.
- Human operator cannot explain recent wins/losses from `/performance` and the evidence packet.

## Conservative defaults

Defaults intentionally remain conservative: demo-only OKX flag, fixed small BTC/ETH sizes, market order type, performance gate disabled until intentionally enabled, max risk per trade disabled until configured, and no automatic size increase. Do not raise size until the evidence packet below passes the relevant stage.

## Evidence packet commands

Historical evidence:

```bash
uv run lumiere-backtest --inst-id BTC-USDT --inst-id ETH-USDT \
  --bar 1m --start 2026-01-01T00:00:00Z --end 2026-03-01T00:00:00Z
uv run lumiere-optimize --inst-id BTC-USDT --bar 1m \
  --start 2026-01-01T00:00:00Z --end 2026-03-01T00:00:00Z \
  --min-trades 30 --min-walk-forward-windows 3 --min-stable-neighbors 1
```

Promotion evidence packet without live credentials:

```bash
uv run lumiere-evidence \
  --optimizer-report reports/strategy_optimization/optimizer_report.json \
  --attribution-ledger data/attribution.jsonl \
  --stage paper --output reports/promotion_evidence.json
```

Attach the generated packet to the promotion decision. Promotion is rejected if `go` is false or any `missing_evidence` item remains.
