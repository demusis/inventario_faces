from __future__ import annotations

import os
from pathlib import Path


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def file_io_path(path: Path) -> str:
    absolute_path = str(Path(path).expanduser().absolute())
    if os.name != "nt":
        return absolute_path
    if absolute_path.startswith("\\\\?\\"):
        return absolute_path
    if absolute_path.startswith("\\\\"):
        return "\\\\?\\UNC\\" + absolute_path[2:]
    return "\\\\?\\" + absolute_path


def safe_stem(value: str) -> str:
    allowed = []
    for character in value:
        if character.isalnum() or character in {"-", "_"}:
            allowed.append(character)
        else:
            allowed.append("_")
    return "".join(allowed).strip("_") or "item"
