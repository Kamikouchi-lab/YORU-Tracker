# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) YORU contributors — see LICENSE for details.

"""The YORU Tracker log file, next to YORU's own.

``~/.yoru/logs/yoru_tracker.log`` (the directory moves with ``YORU_HOME``,
exactly as YORU's ``yoru.log`` does), rotated at 5 MB with three backups.
Every error the GUI shows, every failed batch file, and tracker resets and
configuration loads go here.  Per-track events are added only when
``log_events`` is on in the tracker configuration.
"""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

LOG_FILENAME = "yoru_tracker.log"
_LOGGER_NAME = "yoru_tracker"
_configured = False


def log_file() -> Path:
    from yoru.libs.user_paths import get_log_dir

    return get_log_dir() / LOG_FILENAME


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """Attach the rotating file handler once; safe to call repeatedly."""
    global _configured
    logger = logging.getLogger(_LOGGER_NAME)
    logger.setLevel(level)
    if _configured:
        return logger
    _configured = True
    try:
        handler = RotatingFileHandler(log_file(), maxBytes=5 * 1024 * 1024,
                                      backupCount=3, encoding="utf-8")
        handler.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"))
        logger.addHandler(handler)
    except Exception:
        # No log file is no reason not to run.
        pass
    return logger


def log_exception(context: str, exc: BaseException) -> None:
    try:
        setup_logging().error("%s: %s", context, exc, exc_info=exc)
    except Exception:
        pass
