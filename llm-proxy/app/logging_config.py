"""Logging configuration."""
import logging
import sys
from typing import Optional

import structlog


def setup_logging(
    level: str = "INFO",
    log_format: str = "json",
    include_body: bool = True,
    include_headers: bool = False,
) -> None:
    """Configure logging for the application."""

    # Set log level
    log_level = getattr(logging, level.upper(), logging.INFO)

    # Configure structlog
    shared_kwargs = {
        "level": log_level,
        "format": structlog.dev.ConsoleRenderer()
        if log_format != "json"
        else structlog.processors.JSONRenderer(),
    }

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars(),
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.dev.set_exc_info,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.UnicodeDecoder(),
            shared_kwargs["format"],
        ],
        wrapper_class=structlog.make_logging_logger_wrapper(
            logging.getLogger("llm_proxy")
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Configure standard logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=log_level,
    )


def get_logger(name: Optional[str] = None) -> structlog.BoundLogger:
    """Get a logger instance."""
    return structlog.get_logger(name)
