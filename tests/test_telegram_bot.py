from __future__ import annotations

from lumiere.telegram_bot import BOT_COMMANDS, COMMAND_HELP_TEXT, set_bot_commands


class FakeBot:
    def __init__(self) -> None:
        self.commands = None

    async def set_my_commands(self, commands) -> None:  # noqa: ANN001 - aiogram test double
        self.commands = commands


def test_help_text_lists_every_telegram_command() -> None:
    for command in BOT_COMMANDS:
        assert f"/{command.command} — {command.description}" in COMMAND_HELP_TEXT


def test_set_bot_commands_registers_every_command() -> None:
    bot = FakeBot()

    import asyncio

    asyncio.run(set_bot_commands(bot))

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
