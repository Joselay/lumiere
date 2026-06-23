from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

from aiogram import Bot, Dispatcher, Router
from aiogram.filters import Command
from aiogram.types import Message

from lumiere.engine import TradingEngine


class TelegramNotifier:
    def __init__(self, bot: Bot, allowed_chat_ids: set[int]) -> None:
        self.bot = bot
        self.allowed_chat_ids = allowed_chat_ids
        self.subscribed_chat_ids: set[int] = set(allowed_chat_ids)

    def register_chat(self, chat_id: int) -> None:
        if not self.allowed_chat_ids or chat_id in self.allowed_chat_ids:
            self.subscribed_chat_ids.add(chat_id)

    async def send(self, text: str) -> None:
        for chat_id in self.subscribed_chat_ids:
            await self.bot.send_message(chat_id=chat_id, text=text)


AccessCheck = Callable[[Message], bool]


def allowed_chat_check(allowed_chat_ids: set[int]) -> AccessCheck:
    def check(message: Message) -> bool:
        return not allowed_chat_ids or message.chat.id in allowed_chat_ids

    return check


def command_router(
    engine: TradingEngine,
    allowed_chat_ids: set[int] | None = None,
    notifier: TelegramNotifier | None = None,
) -> Router:
    router = Router(name="lumiere_commands")
    is_allowed = allowed_chat_check(allowed_chat_ids or set())

    async def guard(message: Message, action: Callable[[], Awaitable[str | None]]) -> None:
        if not is_allowed(message):
            await message.answer("Unauthorized chat")
            return
        if notifier is not None:
            notifier.register_chat(message.chat.id)
        response = await action()
        if response:
            await message.answer(response)

    @router.message(Command("start"))
    async def start(message: Message) -> None:
        await guard(message, lambda: _text("Lumiere OKX demo BTC/ETH bot is online"))

    @router.message(Command("status"))
    async def status(message: Message) -> None:
        await guard(message, engine.status_text)

    @router.message(Command("strategy"))
    async def strategy(message: Message) -> None:
        async def show() -> str:
            sections = []
            for params in engine.describe_strategies():
                sections.append("\n".join(f"{key}={value}" for key, value in params.items()))
            return "\n\n".join(sections)

        await guard(message, show)

    @router.message(Command("performance"))
    async def performance(message: Message) -> None:
        await guard(message, engine.performance_text)

    @router.message(Command("risk"))
    async def risk(message: Message) -> None:
        await guard(message, engine.risk_text)

    @router.message(Command("pause"))
    async def pause(message: Message) -> None:
        async def do_pause() -> str:
            await engine.pause()
            return "Trading paused"

        await guard(message, do_pause)

    @router.message(Command("resume"))
    async def resume(message: Message) -> None:
        async def do_resume() -> str:
            await engine.resume()
            if engine.panic_stopped:
                return "Panic stop active; restart required"
            return "Trading resumed"

        await guard(message, do_resume)

    @router.message(Command("panic"))
    async def panic(message: Message) -> None:
        async def do_panic() -> str:
            await engine.panic()
            return "Panic stop activated"

        await guard(message, do_panic)

    return router


async def _text(value: str) -> str:
    return value


async def run_bot(
    bot_token: str,
    engine: TradingEngine,
    allowed_chat_ids: set[int],
) -> None:
    bot = Bot(token=bot_token)
    notifier = TelegramNotifier(bot, allowed_chat_ids)
    engine.notifier = notifier
    dispatcher = Dispatcher()
    dispatcher.include_router(command_router(engine, allowed_chat_ids, notifier))
    engine_task = asyncio.create_task(engine.run_forever())
    try:
        await dispatcher.start_polling(bot)
    finally:
        await engine.stop()
        await engine_task
        await bot.session.close()
