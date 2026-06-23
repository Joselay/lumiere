# Lumiere

Lumiere is a deterministic Python Telegram bot for automated BTC trading on the OKX demo environment.

## Safety defaults

- Demo trading only: `OKX_FLAG` must be `1`; startup fails otherwise.
- BTC only: `OKX_INST_ID` must start with `BTC-` and must be in the allowlist.
- No LLM/ML trade decisions: strategy decisions come from deterministic moving-average rules.
- Risk controls: max position size, max daily loss, cooldown between trades, repeated-failure stop.

## Setup

```bash
uv sync
cp .env.example .env
# edit .env with OKX demo credentials and Telegram bot token
uv run lumiere-bot
```

## Telegram commands

- `/start` — health check
- `/status` — account, position, and engine state
- `/strategy` — active deterministic strategy parameters
- `/pause` — pause automated order placement
- `/resume` — resume automated trading
- `/panic` — cancel open orders and stop trading

## Tests and linting

```bash
uv run pytest
uv run ruff check .
uv run ruff format .
```

## OKX SDK

This project is pinned to `python-okx==0.4.1`, the latest reviewed `okxapi/python-okx` version at commit `eeb851e` (`release/version_0.4.1`). It isolates synchronous REST SDK calls behind `lumiere.okx_client.OKXDemoClient`, called from async code using `asyncio.to_thread`.

The current SDK exposes async WebSocket clients as `okx.websocket.WsPrivateAsync` and `okx.websocket.WsPublicAsync`; older sample imports such as `WsPrivate` / `WsPublic` are intentionally not used.
