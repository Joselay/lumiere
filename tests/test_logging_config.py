from __future__ import annotations

import logging
from io import StringIO

import structlog

from lumiere.logging_config import configure_logging


def test_configure_logging_renders_colorful_structlog_events() -> None:
    stream = StringIO()
    configure_logging("INFO", stream=stream, force_colors=True)

    structlog.get_logger("lumiere.test").info(
        "strategy_decision",
        inst_id="BTC-USDT",
        action="buy",
    )

    output = stream.getvalue()
    assert "\x1b[" in output
    assert "📈 strategy_decision" in output
    assert "BTC-USDT" in output
    assert "action" in output


def test_configure_logging_formats_standard_library_logs() -> None:
    stream = StringIO()
    configure_logging("INFO", stream=stream, force_colors=False)

    logging.getLogger("aiogram").info("Start polling")

    output = stream.getvalue()
    assert "Start polling" in output
    assert "aiogram" in output
    assert "✨" in output


def test_configure_logging_rejects_unknown_levels() -> None:
    stream = StringIO()

    try:
        configure_logging("NOPE", stream=stream)
    except ValueError as exc:
        assert "invalid log level" in str(exc)
    else:  # pragma: no cover - protects the assertion branch
        raise AssertionError("configure_logging should reject unknown levels")
