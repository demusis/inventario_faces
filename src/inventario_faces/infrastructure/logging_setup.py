from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from inventario_faces.utils.path_utils import ensure_directory
from inventario_faces.utils.serialization import to_serializable
from inventario_faces.utils.time_utils import utc_now


class StructuredEventLogger:
    def __init__(self, jsonl_path: Path) -> None:
        self._jsonl_path = jsonl_path
        ensure_directory(jsonl_path.parent)

    def write(self, event: str, **fields: Any) -> None:
        payload = {
            "timestamp_utc": utc_now().isoformat(),
            "event": event,
            **to_serializable(fields),
        }
        with self._jsonl_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, ensure_ascii=False) + "\n")


def build_file_logger(log_directory: Path, log_level: str) -> logging.Logger:
    ensure_directory(log_directory)
    logger_name = f"inventario_faces.{log_directory.as_posix()}"
    logger = logging.getLogger(logger_name)
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    logger.propagate = False

    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%SZ",
    )
    file_handler = logging.FileHandler(log_directory / "run.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def close_file_logger(logger: logging.Logger) -> None:
    handlers = list(logger.handlers)
    for handler in handlers:
        handler.flush()
        handler.close()
        logger.removeHandler(handler)
