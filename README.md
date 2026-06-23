# Lumiere

<p>
  <img src="assets/btc.svg" alt="Bitcoin" width="32" height="32">
  <img src="assets/eth.svg" alt="Ethereum" width="32" height="32">
</p>

Lumiere is a Python trading automation workspace for OKX demo trading.

The project is intended to evolve over time. Trading behavior and architecture live in the source code and tests.

## Setup

```bash
uv sync
cp .env.example .env
# edit .env with your local credentials/settings
uv run lumiere-bot
```

Logs are pretty, timestamped, and colorized in a terminal. Set `LOG_LEVEL=DEBUG` in `.env` for more detail.

## Symbols

Configure OKX demo symbols with `OKX_INST_IDS`. `OKX_INST_ID` remains available as a fallback.

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
