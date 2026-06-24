from __future__ import annotations

import asyncio
from pathlib import Path

import structlog

from lumiere.attribution import AttributionLedger
from lumiere.config import Settings
from lumiere.engine import TradingEngine
from lumiere.logging_config import configure_logging
from lumiere.okx_client import OKXDemoClient
from lumiere.paper_trading import PaperTradingLedger
from lumiere.risk import RiskManager
from lumiere.risk_audit import assert_risk_audit_passes
from lumiere.telegram_bot import run_bot

log = structlog.get_logger(__name__)


def build_engine(settings: Settings) -> TradingEngine:
    risk_manager = RiskManager(settings.risk_config())
    strategy = settings.strategies()
    client = OKXDemoClient(settings, risk_manager)
    paper_ledger = PaperTradingLedger(settings.paper_trading_config())
    attribution_ledger = AttributionLedger(Path(settings.attribution_ledger_path))
    return TradingEngine(
        client=client,
        strategy=strategy,
        risk_manager=risk_manager,
        config=settings.engine_config(),
        paper_ledger=paper_ledger,
        attribution_ledger=attribution_ledger,
    )


async def main() -> None:
    settings = Settings()
    assert_risk_audit_passes(settings)
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
