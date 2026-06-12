"""Project-wide logging configuration.

Exposes a single configured ``logger`` so every module logs with a
consistent format to both stdout and ``logs/run.log`` without registering
duplicate handlers when imported multiple times.
"""

import logging
import sys
from pathlib import Path

_LOG_DIR = Path("logs")
_LOG_DIR.mkdir(exist_ok=True)
_LOG_FILE = _LOG_DIR / "run.log"

_FORMAT = "[%(asctime)s] %(levelname)s %(name)s: %(message)s"


def get_logger(name: str = "sklearn_only") -> logging.Logger:
    """Return a configured logger, attaching handlers only once."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(_FORMAT)

    file_handler = logging.FileHandler(_LOG_FILE)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.propagate = False
    return logger


logger = get_logger()
