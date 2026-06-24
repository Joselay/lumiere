from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from types import SimpleNamespace

import pytest

from lumiere.engine import TradingEngine
from lumiere.models import MarketCandle
from lumiere.risk import RiskConfig, RiskManager
from lumiere.strategy import MovingAverageCrossoverConfig, MovingAverageCrossoverStrategy
from lumiere.telegram_bot import (
    BOT_COMMANDS,
    COMMAND_HELP_TEXT,
    TelegramNotifier,
    command_router,
    set_bot_commands,
)
from tests.fakes import DeterministicFakeExchange


class FakeBot:
    def __init__(self) -> None:
        self.commands = None
        self.sent_messages: list[tuple[int, str]] = []

    async def set_my_commands(self, commands) -> None:  # noqa: ANN001 - aiogram test double
        self.commands = commands

    async def send_message(self, *, chat_id: int, text: str) -> None:
        self.sent_messages.append((chat_id, text))


class FakeMessage:
    def __init__(self, chat_id: int) -> None:
        self.chat = SimpleNamespace(id=chat_id)
        self.answers: list[str] = []

    async def answer(self, text: str) -> None:
        self.answers.append(text)


def test_help_text_lists_every_telegram_command() -> None:
    for command in BOT_COMMANDS:
        assert f"/{command.command} — {command.description}" in COMMAND_HELP_TEXT


@pytest.mark.asyncio
async def test_set_bot_commands_registers_every_command() -> None:
    bot = FakeBot()

    await set_bot_commands(bot)

    assert [command.command for command in bot.commands] == [
        "start",
        "help",
        "status",
        "strategy",
        "performance",
        "risk",
        "pause",
        "resume",
        "panic",
    ]


@pytest.mark.asyncio
async def test_telegram_notifier_registers_allowed_chats_and_broadcasts() -> None:
    bot = FakeBot()
    notifier = TelegramNotifier(bot, allowed_chat_ids={1})

    notifier.register_chat(2)
    await notifier.send("hello")

    assert notifier.subscribed_chat_ids == {1}
    assert bot.sent_messages == [(1, "hello")]


@pytest.mark.asyncio
async def test_command_router_guards_and_dispatches_engine_commands() -> None:
    engine_notifier = TelegramNotifier(FakeBot(), allowed_chat_ids={42})
    engine = TradingEngine(
        client=DeterministicFakeExchange(
            {
                "BTC-USDT": [
                    MarketCandle(
                        ts=datetime(2026, 6, 24, tzinfo=UTC),
                        open=Decimal("100"),
                        high=Decimal("100"),
                        low=Decimal("100"),
                        close=Decimal("100"),
                    )
                ]
            }
        ),
        strategy=MovingAverageCrossoverStrategy(
            MovingAverageCrossoverConfig(fast_window=1, slow_window=2)
        ),
        risk_manager=RiskManager(RiskConfig(cooldown_seconds=0)),
        notifier=engine_notifier,
    )
    router = command_router(engine, allowed_chat_ids={42}, notifier=engine_notifier)

    unauthorized = FakeMessage(7)
    await router.message.handlers[0].callback(unauthorized)
    assert unauthorized.answers == ["Unauthorized chat"]

    message = FakeMessage(42)
    for handler in router.message.handlers[:8]:
        await handler.callback(message)
    await router.message.handlers[8].callback(message)
    await router.message.handlers[7].callback(message)

    joined = "\n".join(message.answers)
    assert "Lumiere is online" in joined
    assert "Lumiere commands" in joined
    assert "Status" in joined
    assert "Strategies" in joined
    assert "Performance" in joined
    assert "Risk" in joined
    assert "Trading paused" in joined
    assert "Trading resumed" in joined
    assert "Panic stop activated" in joined
    assert "Panic stop active; restart required" in joined
    assert 42 in engine_notifier.subscribed_chat_ids
