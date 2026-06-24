from __future__ import annotations

import asyncio

import structlog

from lumiere.config import Settings
from lumiere.engine import EngineConfig, TradingEngine
from lumiere.logging_config import configure_logging
from lumiere.okx_client import OKXDemoClient
from lumiere.risk import RiskManager
from lumiere.telegram_bot import run_bot

log = structlog.get_logger(__name__)


def build_engine(settings: Settings) -> TradingEngine:
    risk_manager = RiskManager(settings.risk_config())
    strategy = settings.strategies()
    client = OKXDemoClient(settings, risk_manager)
    return TradingEngine(
        client=client,
        strategy=strategy,
        risk_manager=risk_manager,
        config=EngineConfig(
            poll_interval_seconds=settings.engine_poll_interval_seconds,
            td_mode=settings.okx_td_mode,
        ),
    )


async def main() -> None:
    settings = Settings()
    configure_logging(settings.log_level)
    log.info(
        "lumiere_starting",
        symbols=settings.enabled_inst_ids,
        td_mode=settings.okx_td_mode,
        poll_interval_seconds=settings.engine_poll_interval_seconds,
    )
    engine = build_engine(settings)
    await run_bot(
        bot_token=settings.telegram_bot_token,
        engine=engine,
        allowed_chat_ids=settings.allowed_chat_ids,
    )


def run() -> None:
    asyncio.run(main())


if __name__ == "__main__":
    run()
