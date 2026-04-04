from __future__ import annotations

from pathlib import Path
from typing import Protocol

from inventario_faces.domain.entities import (
    DetectedFace,
    FaceSearchResult,
    InventoryResult,
    MediaInfoTrack,
    ReportArtifacts,
    SampledFrame,
)


class FaceAnalyzer(Protocol):
    def detect(self, frame: SampledFrame) -> list[DetectedFace]:
        ...

    def embed(self, frame: SampledFrame, detection: DetectedFace, reason: str = "keyframe") -> list[float]:
        ...

    def analyze(self, frame: SampledFrame) -> list[DetectedFace]:
        ...


class ReportGenerator(Protocol):
    def generate(self, result: InventoryResult) -> ReportArtifacts:
        ...


class FaceSearchReportGenerator(Protocol):
    def generate(self, result: FaceSearchResult) -> ReportArtifacts:
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
