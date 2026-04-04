from __future__ import annotations

import os
from collections import Counter
from pathlib import Path

from inventario_faces.domain.config import MediaSettings
from inventario_faces.domain.entities import MediaType


class ScannerService:
    def __init__(self, settings: MediaSettings) -> None:
        self._settings = settings

    def classify(self, path: Path) -> MediaType:
        suffix = path.suffix.lower()
        if suffix in self._settings.image_extensions:
            return MediaType.IMAGE
        if suffix in self._settings.video_extensions:
            return MediaType.VIDEO
        return MediaType.OTHER

    def iter_scan(self, root_directory: Path, excluded_directories: set[Path] | None = None):
        root_directory = root_directory.resolve()
        excluded = {path.resolve() for path in (excluded_directories or set())}
        for current_root, directory_names, file_names in os.walk(root_directory, topdown=True):
            current_path = Path(current_root)
            directory_names[:] = [
                directory_name
                for directory_name in sorted(directory_names)
                if not self._is_excluded(current_path / directory_name, excluded)
            ]
            for file_name in sorted(file_names):
                candidate = current_path / file_name
                if not self._is_excluded(candidate, excluded):
                    yield candidate

    def scan(self, root_directory: Path, excluded_directories: set[Path] | None = None) -> list[Path]:
        return list(self.iter_scan(root_directory, excluded_directories))

    def media_files(self, root_directory: Path, excluded_directories: set[Path] | None = None) -> list[Path]:
        return [path for path in self.scan(root_directory, excluded_directories) if self.classify(path) != MediaType.OTHER]

    def summarize(
        self,
        root_directory: Path,
        excluded_directories: set[Path] | None = None,
    ) -> tuple[int, Counter[MediaType]]:
        total_files = 0
        media_counter: Counter[MediaType] = Counter()
        for path in self.iter_scan(root_directory, excluded_directories):
            total_files += 1
            media_counter[self.classify(path)] += 1
        return total_files, media_counter

    def _is_excluded(self, path: Path, excluded_directories: set[Path]) -> bool:
        resolved = path.resolve()
        return any(directory == resolved or directory in resolved.parents for directory in excluded_directories)
