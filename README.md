# Lumiere

<p align="center">
  <img src="assets/btc.svg" alt="Bitcoin" width="56" height="56">
  <img src="assets/eth.svg" alt="Ethereum" width="56" height="56">
</p>

<p align="center">
  <strong>Telegram-controlled OKX demo trading automation for BTC and ETH.</strong>
</p>

Lumiere is a Python trading bot workspace for experimenting with automated OKX demo trading. The project is built to keep improving over time while clearly showing what the bot supports today.

## What is supported now

| Area | Current support |
| --- | --- |
| Exchange | OKX demo trading only (`OKX_FLAG=1`) |
| Symbols | `BTC-USDT`, `ETH-USDT` |
| Strategy | Config-selectable moving-average crossover, RSI mean-reversion, and ATR volatility-breakout strategies |
| Orders | OKX market buy/sell orders through `python-okx` |
| Telegram | `/start`, `/status`, `/strategy`, `/performance`, `/risk`, `/pause`, `/resume`, `/panic` |
| Risk controls | Demo guard, allowed symbols, min order size, max position size, cooldown, real fill-derived daily loss, max drawdown, daily trade limit, spread guard, performance gate, max consecutive failures |
| Profitability evidence | OKX SDK historical candle backtests with fees, spread, slippage, rejected-order modeling, PnL ledger metrics, and buy-and-hold/no-trade baselines |
| Observability | Pretty structured logs with configurable `LOG_LEVEL` |

## Bot showcase

- Runs multiple configured symbols from `OKX_INST_IDS`.
- Reads OKX candles, account balances, and positions.
- Sends order and risk notifications to Telegram.
- Supports operator controls for pause, resume, status checks, strategy inspection, and panic stop.
- Cancels open orders during panic stop.
- Keeps strategy and risk logic covered by tests so future changes are easier to make safely.
- Supports config-selected strategy modules via `STRATEGY_NAME=moving_average_crossover`, `rsi_mean_reversion`, or `volatility_breakout`; backtest reports include each strategy's allowed market regimes.

## Profitability and safety disclaimer

Lumiere can help collect evidence and enforce safeguards, but **profit is never guaranteed**. Backtests are not live trading: fees, spread, slippage, liquidity, rejected orders, and regime changes can make live/demo results worse than historical reports.

Current default assumptions for reports are conservative but configurable: 10 bps taker fee, 2 bps spread, 5 bps slippage, 0 bps market impact, and no synthetic rejected orders unless requested.

## Setup

```bash
uv sync
cp .env.example .env
# edit .env with your local credentials/settings
uv run lumiere-bot
```

Logs are pretty, timestamped, and colorized in a terminal. Set `LOG_LEVEL=DEBUG` in `.env` for more detail.

## Backtesting

Run a reproducible BTC/ETH report using OKX historical candle access through `python-okx`:

```bash
uv run lumiere-backtest --inst-id BTC-USDT --inst-id ETH-USDT --bar 1m --limit 300
```

For longer-horizon evidence, request an inclusive date range. The backtester paginates OKX history, saves a checksummed CSV plus metadata under `data/historical`, and can rerun fully offline from that cache:

```bash
uv run lumiere-backtest \
  --inst-id BTC-USDT --inst-id ETH-USDT \
  --bar 1m --start 2026-01-01T00:00:00Z --end 2026-03-01T00:00:00Z
uv run lumiere-backtest --offline --bar 1m --start 2026-01-01T00:00:00Z --end 2026-03-01T00:00:00Z
```

The JSON report includes full-period metrics plus chronological train/validation/test and rolling walk-forward reports. Each split shows in-sample vs out-of-sample net PnL after modeled costs, realized/unrealized PnL, equity curve, max drawdown, trade count, win rate, profit factor, Sharpe/Sortino when available, rejected order count, and buy-and-hold/no-trade baseline comparisons.

Optimize moving-average candidates with the same cached data and cost assumptions:

```bash
uv run lumiere-optimize \
  --inst-id BTC-USDT --bar 1m \
  --start 2026-01-01T00:00:00Z --end 2026-03-01T00:00:00Z \
  --fast-window 3:12 --slow-window 10:40 \
  --min-trades 20 --min-walk-forward-windows 3 --min-stable-neighbors 1
```

The optimizer writes `reports/strategy_optimization/optimizer_report.json` and `accepted_candidates.json`. Candidates are sorted by out-of-sample net PnL, drawdown, profit factor, Sharpe/Sortino, trade count, and win rate, and are rejected unless they pass out-of-sample gates, beat no-trade and buy-and-hold baselines, avoid train/test divergence, and satisfy any configured walk-forward and parameter-stability gates.

## Symbols

Configure OKX demo symbols with `OKX_INST_IDS`:

```env
OKX_INST_IDS=BTC-USDT,ETH-USDT
```

`OKX_INST_ID` remains available as a fallback when `OKX_INST_IDS` is empty.

## Telegram controls

```text
/start     show that the bot is online
/status       show engine, account, position, and risk status
/strategy     show active strategy configuration
/performance  show daily realized PnL, drawdown, trade count, spread, and gate state
/risk         show configured risk limits and current open risk inputs
/pause        pause trading
/resume       resume trading
/panic        stop the engine and cancel open orders
```

## Development

```bash
uv run pytest
uv run ruff check .
uv run ruff format .
```
