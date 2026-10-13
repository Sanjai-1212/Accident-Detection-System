"""
Structured logging for the Accident Detection System.

All modules share a single session log file via the 'accident' root logger.
"""
import logging
import os
from datetime import datetime

_initialized = False
_log_file_path: str = ""


def _init_root_logger():
    """Set up the root 'accident' logger with console + file handlers."""
    global _initialized, _log_file_path
    if _initialized:
        return

    try:
        import config as cfg
        log_dir = cfg.LOG_DIR
        log_level = cfg.LOG_LEVEL
        log_to_file = cfg.LOG_TO_FILE
    except ImportError:
        log_dir = "logs"
        log_level = "INFO"
        log_to_file = False

    root = logging.getLogger("accident")
    root.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    console_fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)-22s | %(message)s",
        datefmt="%H:%M:%S",
    )
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(console_fmt)
    root.addHandler(ch)

    if log_to_file:
        os.makedirs(log_dir, exist_ok=True)
        _log_file_path = os.path.join(
            log_dir, f"session_{datetime.now():%Y%m%d_%H%M%S}.log"
        )
        try:
            file_fmt = logging.Formatter(
                "%(asctime)s | %(levelname)-8s | %(name)-22s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            fh = logging.FileHandler(_log_file_path, encoding="utf-8")
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(file_fmt)
            root.addHandler(fh)
        except OSError:
            pass  # Don't crash if log file creation fails

    root.propagate = False
    _initialized = True


def setup_logger(name: str = "main") -> logging.Logger:
    """Return a child logger under the 'accident' namespace.

    Example::

        logger = setup_logger("detector")  # → accident.detector
    """
    _init_root_logger()
    return logging.getLogger(f"accident.{name}")


def get_log_file_path() -> str:
    """Return the path of the current session log file."""
    _init_root_logger()
    return _log_file_path
