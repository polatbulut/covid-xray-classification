"""Console logging configuration shared by every command-line entry point."""

from __future__ import annotations

import logging
import sys
from typing import Final

_LOG_FORMAT: Final = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DATE_FORMAT: Final = "%Y-%m-%d %H:%M:%S"


def configure_logging(*, verbose: bool = False) -> None:
    """Send log records to stderr with a consistent format.

    Args:
        verbose: When ``True``, emit ``DEBUG`` records as well as ``INFO``.
    """
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
    root = logging.getLogger()
    for existing in list(root.handlers):
        root.removeHandler(existing)
    root.addHandler(handler)
    root.setLevel(logging.DEBUG if verbose else logging.INFO)
