"""Structured file logging for CSFSeg runs."""

from __future__ import annotations

import logging
import time
import traceback
from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator


def make_run_id(now: datetime | None = None) -> str:
    """Return a compact run id that is stable for one command invocation."""

    stamp = now or datetime.now()
    return stamp.strftime("%Y%m%d-%H%M%S")


def setup_subject_logger(
    output_id: str,
    out_dir: str | Path,
    *,
    run_id: str | None = None,
) -> tuple[logging.Logger, Path, str]:
    """Create a detailed log file for one input.

    The logger is intentionally file-only. Terminal output stays in ``cli.py``
    so screen messages remain short while logs keep the full story.
    """

    resolved_out_dir = Path(out_dir).expanduser().resolve()
    resolved_run_id = run_id or make_run_id()
    log_path = resolved_out_dir / "logs" / "subjects" / f"{output_id}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger_name = f"csfseg.subject.{resolved_run_id}.{output_id}"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    return logger, log_path, resolved_run_id


def setup_run_logger(
    out_dir: str | Path,
    *,
    run_id: str | None = None,
) -> tuple[logging.Logger, Path, str]:
    """Create a batch-level log file."""

    resolved_out_dir = Path(out_dir).expanduser().resolve()
    resolved_run_id = run_id or make_run_id()
    log_path = resolved_out_dir / "logs" / f"run_{resolved_run_id}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger_name = f"csfseg.run.{resolved_run_id}"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    return logger, log_path, resolved_run_id


def close_logger(logger: logging.Logger | None) -> None:
    """Flush and close all handlers attached to a logger."""

    if logger is None:
        return
    for handler in list(logger.handlers):
        handler.flush()
        handler.close()
        logger.removeHandler(handler)


def log_run_header(
    logger: logging.Logger | None,
    *,
    title: str,
    values: Any,
) -> None:
    """Write a clearly marked identity block at the top of a run log."""

    if logger is None:
        return
    logger.info("===== %s =====", title)
    for key, value in _items(values):
        logger.info("%s: %s", key, _format_value(value))


def log_values(
    logger: logging.Logger | None,
    title: str,
    values: Any,
) -> None:
    """Write a named group of key/value details."""

    if logger is None:
        return
    logger.info("--- %s ---", title)
    for key, value in _items(values):
        logger.info("%s: %s", key, _format_value(value))


@contextmanager
def log_stage(logger: logging.Logger | None, stage: str) -> Iterator[None]:
    """Log stage start, success, elapsed time, and traceback on failure."""

    start = time.perf_counter()
    if logger is not None:
        logger.info("[START] %s", stage)
    try:
        yield
    except Exception as exc:
        elapsed = time.perf_counter() - start
        if logger is not None:
            logger.error("[ERROR] %s elapsed=%.3fs", stage, elapsed)
            logger.error("error_type: %s", type(exc).__name__)
            logger.error("error_message: %s", exc)
            logger.error("traceback:\n%s", traceback.format_exc())
        raise
    else:
        elapsed = time.perf_counter() - start
        if logger is not None:
            logger.info("[DONE] %s elapsed=%.3fs", stage, elapsed)


def _items(values: Any) -> list[tuple[str, Any]]:
    if values is None:
        return []
    if is_dataclass(values):
        values = asdict(values)
    if isinstance(values, dict):
        return [(str(key), value) for key, value in values.items()]
    return [(name, getattr(values, name)) for name in dir(values) if not name.startswith("_")]


def _format_value(value: Any) -> str:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_format_value(item) for item in value) + "]"
    if isinstance(value, dict):
        inner = ", ".join(f"{key}={_format_value(val)}" for key, val in value.items())
        return "{" + inner + "}"
    return str(value)


__all__ = [
    "close_logger",
    "log_run_header",
    "log_stage",
    "log_values",
    "make_run_id",
    "setup_run_logger",
    "setup_subject_logger",
]
