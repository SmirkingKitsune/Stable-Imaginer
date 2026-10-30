"""Structured logging.

structlog renders both our own events and stdlib loggers (uvicorn, alembic, httpx), so all
output shares one format. ``request_id``, ``conversation_id``, ``turn_id`` and ``job_id``
are bound as contextvars and appear on every line logged while they are in scope.
"""

import logging
import sys
from typing import Any

import structlog
from structlog.typing import Processor


def _console_renderer() -> Processor:
    try:
        return structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty())
    except SystemError:  # colors requested on a Windows console without colorama
        return structlog.dev.ConsoleRenderer(colors=False)


def configure_logging(level: str = "INFO", fmt: str = "console") -> None:
    shared: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
    ]
    final: list[Processor]
    if fmt == "json":
        final = [structlog.processors.format_exc_info, structlog.processors.JSONRenderer()]
    else:
        final = [_console_renderer()]

    structlog.configure(
        processors=[*shared, structlog.stdlib.ProcessorFormatter.wrap_for_formatter],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared,
        processors=[structlog.stdlib.ProcessorFormatter.remove_processors_meta, *final],
    )
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level)
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        uv_logger = logging.getLogger(name)
        uv_logger.handlers.clear()
        uv_logger.propagate = True
    # We log one line per request ourselves, with the request id.
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    # Chatty third-party loggers that would drown our DEBUG output.
    for name in ("aiosqlite", "asyncio", "httpx", "httpcore", "multipart", "PIL", "sse_starlette"):
        logging.getLogger(name).setLevel(logging.WARNING)


def get_logger(name: str, **initial: Any) -> structlog.stdlib.BoundLogger:
    logger: structlog.stdlib.BoundLogger = structlog.stdlib.get_logger(name, **initial)
    return logger
