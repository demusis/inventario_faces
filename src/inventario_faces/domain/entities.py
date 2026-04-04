from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any


class MediaType(StrEnum):
    IMAGE = "image"
    VIDEO = "video"
    OTHER = "other"


@dataclass(frozen=True)
class BoundingBox:
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)


@dataclass(frozen=True)
class SampledFrame:
    source_path: Path
    image_name: str
    frame_index: int | None
    timestamp_seconds: float | None
    bgr_pixels: Any


@dataclass(frozen=True)
class DetectedFace:
    bbox: BoundingBox
    detection_score: float
    embedding: list[float]
    crop_bgr: Any


@dataclass(frozen=True)
class MediaInfoAttribute:
    label: str
    value: str


@dataclass(frozen=True)
class MediaInfoTrack:
    track_type: str
    attributes: tuple[MediaInfoAttribute, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class FileRecord:
    path: Path
    media_type: MediaType
    sha512: str
    size_bytes: int
    discovered_at_utc: datetime
    modified_at_utc: datetime | None
    processing_error: str | None = None
    media_info_tracks: tuple[MediaInfoTrack, ...] = field(default_factory=tuple)
    media_info_error: str | None = None


@dataclass
class FaceOccurrence:
    occurrence_id: str
    source_path: Path
    sha512: str
    media_type: MediaType
    analysis_timestamp_utc: datetime
    frame_index: int | None
    frame_timestamp_seconds: float | None
    bbox: BoundingBox
    detection_score: float
    embedding: list[float]
    crop_path: Path | None
    context_image_path: Path | None = None
    cluster_id: str | None = None
    suggested_cluster_ids: list[str] = field(default_factory=list)


@dataclass
class FaceCluster:
    cluster_id: str
    occurrence_ids: list[str] = field(default_factory=list)
    centroid_embedding: list[float] = field(default_factory=list)
    representative_crop_path: Path | None = None
    candidate_cluster_ids: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ProcessingSummary:
    total_files: int
    media_files: int
    image_files: int
    video_files: int
    total_occurrences: int
    total_clusters: int
    probable_match_pairs: int
    total_detected_face_sizes: "FaceSizeStatistics" = field(default_factory=lambda: FaceSizeStatistics())
    selected_face_sizes: "FaceSizeStatistics" = field(default_factory=lambda: FaceSizeStatistics())


@dataclass(frozen=True)
class FaceSizeStatistics:
    count: int = 0
    min_pixels: float | None = None
    max_pixels: float | None = None
    mean_pixels: float | None = None
    stddev_pixels: float | None = None


@dataclass(frozen=True)
class ReportArtifacts:
    tex_path: Path
    pdf_path: Path | None
    docx_path: Path | None = None


@dataclass(frozen=True)
class InventoryResult:
    run_directory: Path
    started_at_utc: datetime
    finished_at_utc: datetime
    root_directory: Path
    files: list[FileRecord]
    occurrences: list[FaceOccurrence]
    clusters: list[FaceCluster]
    report: ReportArtifacts
    summary: ProcessingSummary
    logs_directory: Path
    manifest_path: Path
