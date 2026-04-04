from __future__ import annotations

import hashlib
from pathlib import Path

from inventario_faces.utils.path_utils import file_io_path


class HashingService:
    def sha512(self, path: Path, chunk_size: int = 1024 * 1024) -> str:
        hasher = hashlib.sha512()
        with open(file_io_path(path), "rb") as stream:
            while True:
                chunk = stream.read(chunk_size)
                if not chunk:
                    break
                hasher.update(chunk)
        return hasher.hexdigest()
