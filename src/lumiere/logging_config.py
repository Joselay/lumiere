from __future__ import annotations

import logging
import sys
from collections.abc import Callable, MutableMapping
from typing import Any, TextIO

import structlog

EventDict = MutableMapping[str, Any]
Processor = Callable[[Any, str, EventDict], EventDict]

_LEVEL_ICONS = {
    "debug": "🔎",
    "info": "✨",
    "warning": "⚠️",
    "error": "🔥",
    "critical": "🚨",
}
_EVENT_ICONS = {
    "engine_started": "🚀",
    "engine_stopped": "🛑",
    "engine_tick_skipped": "⏸️",
    "strategy_decision": "📈",
    "risk_blocked": "🛡️",
    "order_submitted": "✅",
    "engine_tick_failed": "💥",
    "trading_paused": "⏸️",
    "trading_resumed": "▶️",
    "panic_stop": "🚨",
    "lumiere_starting": "🌙",
}


def configure_logging(
    level: int | str = logging.INFO,
    *,
    stream: TextIO | None = None,
    force_colors: bool = False,
) -> None:
    """Configure colorful human-readable logs for Lumiere and dependency loggers."""

    stream = stream or sys.stdout
    level_number = _level_number(level)
    use_colors = force_colors or _supports_color(stream)

    timestamper = structlog.processors.TimeStamper(fmt="%H:%M:%S", utc=False)
    shared_processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        timestamper,
        _add_event_icon,
        structlog.processors.StackInfoRenderer(),
    ]

    renderer = structlog.dev.ConsoleRenderer(
        colors=use_colors,
        force_colors=force_colors,
        pad_event_to=24,
        sort_keys=False,
    )
    formatter = structlog.stdlib.ProcessorFormatter(
        processor=renderer,
        foreign_pre_chain=shared_processors,
    )

    handler = logging.StreamHandler(stream)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(level_number)

    # Keep noisy HTTP internals out of normal trading logs.
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=False,
    )


def _add_event_icon(_: Any, __: str, event_dict: EventDict) -> EventDict:
    level = str(event_dict.get("level", "")).lower()
    event = str(event_dict.get("event", ""))
    icon = _EVENT_ICONS.get(event, _LEVEL_ICONS.get(level, "•"))
    if event and not event.startswith(icon):
        event_dict["event"] = f"{icon} {event}"
    return event_dict


def _level_number(level: int | str) -> int:
    if isinstance(level, int):
        return level
    normalized = level.strip().upper()
    if normalized.isdigit():
        return int(normalized)
    level_number = logging.getLevelName(normalized)
    if isinstance(level_number, int):
        return level_number
    valid = ", ".join(logging.getLevelNamesMapping())
    raise ValueError(f"invalid log level {level!r}; expected one of: {valid}")


def _supports_color(stream: TextIO) -> bool:
    isatty = getattr(stream, "isatty", None)
    return bool(isatty and isatty())
