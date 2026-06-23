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
| Strategy | Moving-average crossover strategy implementation |
| Orders | OKX market buy/sell orders through `python-okx` |
| Telegram | `/start`, `/status`, `/strategy`, `/pause`, `/resume`, `/panic` |
| Risk controls | Demo guard, allowed symbols, min order size, max position size, cooldown, max daily loss, max consecutive failures |
| Observability | Pretty structured logs with configurable `LOG_LEVEL` |

## Bot showcase

- Runs multiple configured symbols from `OKX_INST_IDS`.
- Reads OKX candles, account balances, and positions.
- Sends order and risk notifications to Telegram.
- Supports operator controls for pause, resume, status checks, strategy inspection, and panic stop.
- Cancels open orders during panic stop.
- Keeps strategy and risk logic covered by tests so future changes are easier to make safely.

## Setup

```bash
uv sync
cp .env.example .env
# edit .env with your local credentials/settings
uv run lumiere-bot
```

Logs are pretty, timestamped, and colorized in a terminal. Set `LOG_LEVEL=DEBUG` in `.env` for more detail.

## Symbols

Configure OKX demo symbols with `OKX_INST_IDS`:

```env
OKX_INST_IDS=BTC-USDT,ETH-USDT
```

`OKX_INST_ID` remains available as a fallback when `OKX_INST_IDS` is empty.

## Telegram controls

```text
/start     show that the bot is online
/status    show engine, account, position, and risk status
/strategy  show active strategy configuration
/pause     pause trading
/resume    resume trading
/panic     stop the engine and cancel open orders
```

## Development

```bash
uv run pytest
uv run ruff check .
uv run ruff format .
```
