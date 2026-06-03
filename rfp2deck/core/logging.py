from __future__ import annotations

import logging
import os
import time
from contextlib import contextmanager
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Iterator

from rich.logging import RichHandler

# Sentinel attribute so repeated calls (e.g. Streamlit reruns) don't stack handlers.
_CONFIGURED_FLAG = "_rfp2deck_logging_configured"

# Third-party loggers that are very chatty at DEBUG/INFO; keep them at WARNING
# unless the user explicitly asks for verbose output.
_NOISY_LOGGERS = ("httpx", "httpcore", "openai", "urllib3", "faiss", "PIL")


def setup_logging(level: str | None = None, log_file: str | None = None) -> None:
    """Configure application logging once, idempotently.

    Console output goes through Rich (with tracebacks). When a log file is
    configured, logs are also written to a rotating file so issues that happen
    in a deployed/headless run can be inspected after the fact.

    Configuration can be overridden via environment variables:
      - ``LOG_LEVEL``  (default: INFO)
      - ``LOG_FILE``   (default: logs/rfp2deck.log; set empty to disable)
      - ``LOG_VERBOSE_LIBS`` (set to "1"/"true" to keep noisy libs at the app level)
    """
    root = logging.getLogger()

    level_name = (level or os.getenv("LOG_LEVEL", "INFO")).upper()
    log_level = getattr(logging, level_name, logging.INFO)

    # Idempotent: if already configured, just align the level and return.
    if getattr(root, _CONFIGURED_FLAG, False):
        root.setLevel(log_level)
        return

    root.setLevel(log_level)

    console_handler = RichHandler(rich_tracebacks=True, show_path=False)
    console_handler.setFormatter(logging.Formatter("%(name)s | %(message)s", datefmt="[%X]"))
    root.addHandler(console_handler)

    # File handler (rotating) for persistent debugging.
    if log_file is None:
        log_file = os.getenv("LOG_FILE", "logs/rfp2deck.log")
    if log_file:
        try:
            path = Path(log_file)
            path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = RotatingFileHandler(
                path, maxBytes=5_000_000, backupCount=3, encoding="utf-8"
            )
            file_handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s | %(levelname)-7s | %(name)s | %(funcName)s | %(message)s"
                )
            )
            root.addHandler(file_handler)
        except OSError as exc:  # pragma: no cover - filesystem edge cases
            root.warning("Could not set up file logging at %s: %s", log_file, exc)

    # Tame noisy third-party loggers unless verbose libs explicitly requested.
    if os.getenv("LOG_VERBOSE_LIBS", "").lower() not in ("1", "true", "yes"):
        for name in _NOISY_LOGGERS:
            logging.getLogger(name).setLevel(logging.WARNING)

    setattr(root, _CONFIGURED_FLAG, True)
    root.debug("Logging configured (level=%s, file=%s)", level_name, log_file or "<none>")


def get_logger(name: str) -> logging.Logger:
    """Return a module-scoped logger. Use ``get_logger(__name__)``."""
    return logging.getLogger(name)


@contextmanager
def log_duration(logger: logging.Logger, label: str, **context: object) -> Iterator[None]:
    """Log the start, duration, and any failure of a block of work.

    Example::

        with log_duration(log, "render deck", slides=len(plan.slides)):
            render(...)
    """
    ctx = " ".join(f"{k}={v}" for k, v in context.items())
    suffix = f" ({ctx})" if ctx else ""
    logger.info("START %s%s", label, suffix)
    start = time.perf_counter()
    try:
        yield
    except Exception:
        elapsed = time.perf_counter() - start
        logger.exception("FAILED %s after %.2fs%s", label, elapsed, suffix)
        raise
    else:
        elapsed = time.perf_counter() - start
        logger.info("DONE  %s in %.2fs%s", label, elapsed, suffix)
