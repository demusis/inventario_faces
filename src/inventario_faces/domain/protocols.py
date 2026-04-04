from __future__ import annotations

from pathlib import Path
from typing import Protocol

from inventario_faces.domain.entities import (
    DetectedFace,
    InventoryResult,
    MediaInfoTrack,
    ReportArtifacts,
    SampledFrame,
)


class FaceAnalyzer(Protocol):
    def analyze(self, frame: SampledFrame) -> list[DetectedFace]:
        ...


class ReportGenerator(Protocol):
    def generate(self, result: InventoryResult) -> ReportArtifacts:
        ...


class ProgressCallback(Protocol):
    def __call__(self, current: int, total: int, message: str) -> None:
        ...


class LogCallback(Protocol):
    def __call__(self, message: str) -> None:
        ...


class MediaInfoExtractor(Protocol):
    def extract(self, path: Path) -> tuple[tuple[MediaInfoTrack, ...], str | None]:
        ...


class PathOpener(Protocol):
    def __call__(self, path: Path) -> None:
        ...
