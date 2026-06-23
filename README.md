# Lumiere

Lumiere is a Python trading automation workspace.

The bot's trading behavior is intentionally kept in source code and tests so it can be changed as the project evolves. This README avoids defining a fixed strategy, market, or final architecture.

## Setup

```bash
uv sync
cp .env.example .env
# edit .env with your local credentials/settings
uv run lumiere-bot
```

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
