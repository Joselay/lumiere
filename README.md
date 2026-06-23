# Lumiere

Lumiere is a Python trading automation workspace for OKX demo BTC-USDT and ETH-USDT trading.

The bot's trading behavior is intentionally kept in source code and tests so it can be changed as the project evolves. This README avoids defining a fixed strategy, market, or final architecture.

## Setup

```bash
uv sync
cp .env.example .env
# edit .env with your local credentials/settings
uv run lumiere-bot
```

Logs are pretty, timestamped, and colorized in a terminal. Set `LOG_LEVEL=DEBUG` in `.env` for more detail.

## Symbols

Current supported OKX demo symbols:

```text
BTC-USDT
ETH-USDT
```

Enable symbols with `OKX_INST_IDS=BTC-USDT,ETH-USDT`. `OKX_INST_ID` remains as a single-symbol fallback.

## Telegram controls

```text
/start
/status
/strategy
/pause
/resume
/panic
```

## Development

```bash
uv run pytest
uv run ruff check .
uv run ruff format .
```
