from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from inventario_faces.domain.config import LikelihoodRatioSettings


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

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)


@dataclass(frozen=True)
class EnhancementMetadata:
    applied: bool = False
    strategy: str = "none"
    parameters: dict[str, Any] = field(default_factory=dict)
    brightness_before: float | None = None
    brightness_after: float | None = None
    note: str | None = None


@dataclass(frozen=True)
class FaceQualityMetrics:
    detection_score: float = 0.0
    sharpness: float = 0.0
    brightness: float = 0.0
    illumination: float = 0.0
    frontality: float = 0.0
    bbox_pixels: float = 0.0
    score: float = 0.0


@dataclass(frozen=True)
class SampledFrame:
    source_path: Path
    image_name: str
    frame_index: int | None
    timestamp_seconds: float | None
    bgr_pixels: Any
    original_bgr_pixels: Any | None = None
    enhancement_metadata: EnhancementMetadata | None = None


@dataclass
class DetectedFace:
    bbox: BoundingBox
    detection_score: float
    crop_bgr: Any
    embedding: list[float] = field(default_factory=list)
    landmarks: tuple[tuple[float, float], ...] = field(default_factory=tuple)
    biometric_landmarks: tuple[tuple[float, float], ...] = field(default_factory=tuple)
    quality_metrics: FaceQualityMetrics | None = None
    enhancement_metadata: EnhancementMetadata | None = None
    embedding_source: str | None = None


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
    crop_path: Path | None
    embedding: list[float] = field(default_factory=list)
    context_image_path: Path | None = None
    cluster_id: str | None = None
    suggested_cluster_ids: list[str] = field(default_factory=list)
    track_id: str | None = None
    keyframe_id: str | None = None
    quality_metrics: FaceQualityMetrics | None = None
    enhancement_metadata: EnhancementMetadata | None = None
    is_keyframe: bool = False
    track_position: int | None = None
    embedding_source: str | None = None
    biometric_landmarks: tuple[tuple[float, float], ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class KeyFrame:
    keyframe_id: str
    track_id: str
    occurrence_id: str
    source_path: Path
    frame_index: int | None
    timestamp_seconds: float | None
    selection_reasons: tuple[str, ...] = field(default_factory=tuple)
    quality_metrics: FaceQualityMetrics | None = None
    detection_score: float = 0.0
    crop_path: Path | None = None
    context_image_path: Path | None = None
    embedding: list[float] = field(default_factory=list)
    preview_path: Path | None = None
    biometric_landmarks: tuple[tuple[float, float], ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class TrackQualityStatistics:
    total_detections: int = 0
    keyframe_count: int = 0
    mean_detection_score: float = 0.0
    max_detection_score: float = 0.0
    mean_quality_score: float = 0.0
    best_quality_score: float = 0.0
    mean_sharpness: float = 0.0
    mean_brightness: float = 0.0
    mean_illumination: float = 0.0
    mean_frontality: float = 0.0
    duration_seconds: float = 0.0


@dataclass
class FaceTrack:
    track_id: str
    source_path: Path
    video_path: Path | None
    media_type: MediaType
    sha512: str
    start_frame: int | None
    end_frame: int | None
    start_time: float | None
    end_time: float | None
    occurrence_ids: list[str] = field(default_factory=list)
    keyframe_ids: list[str] = field(default_factory=list)
    representative_embeddings: list[list[float]] = field(default_factory=list)
    average_embedding: list[float] = field(default_factory=list)
    best_occurrence_id: str | None = None
    preview_path: Path | None = None
    top_crop_paths: list[Path] = field(default_factory=list)
    quality_statistics: TrackQualityStatistics = field(default_factory=TrackQualityStatistics)
    cluster_id: str | None = None
    candidate_cluster_ids: list[str] = field(default_factory=list)


@dataclass
class FaceCluster:
    cluster_id: str
    track_ids: list[str] = field(default_factory=list)
    occurrence_ids: list[str] = field(default_factory=list)
    centroid_embedding: list[float] = field(default_factory=list)
    representative_crop_path: Path | None = None
    representative_track_id: str | None = None
    preview_paths: list[Path] = field(default_factory=list)
    candidate_cluster_ids: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class SearchArtifacts:
    engine: str
    track_index_path: Path | None
    track_metadata_path: Path | None
    cluster_index_path: Path | None
    cluster_metadata_path: Path | None
    track_vector_count: int = 0
    cluster_vector_count: int = 0


@dataclass(frozen=True)
class ProcessingSummary:
    total_files: int
    media_files: int
    image_files: int
    video_files: int
    total_occurrences: int
    total_clusters: int
    probable_match_pairs: int
    total_tracks: int = 0
    total_keyframes: int = 0
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
class FaceSearchQuery:
    source_path: Path
    sha512: str
    detected_face_count: int
    selected_track_id: str
    selected_occurrence_id: str
    selected_keyframe_id: str | None
    crop_path: Path | None
    context_image_path: Path | None
    quality_score: float | None
    query_index: int = 1


@dataclass(frozen=True)
class FaceSearchQueryEvent:
    query_index: int
    source_path: Path
    status: str
    sha512: str | None = None
    detected_face_count: int | None = None
    selected_track_id: str | None = None
    selected_occurrence_id: str | None = None
    selected_keyframe_id: str | None = None
    crop_path: Path | None = None
    context_image_path: Path | None = None
    quality_score: float | None = None
    error_message: str | None = None
    error_type: str | None = None


@dataclass(frozen=True)
class FaceSearchMatch:
    rank: int
    cluster_id: str | None
    track_id: str
    occurrence_id: str | None
    cluster_score: float | None
    track_score: float
    occurrence_score: float | None
    source_path: Path
    frame_index: int | None
    timestamp_seconds: float | None
    track_start_time: float | None
    track_end_time: float | None
    crop_path: Path | None
    context_image_path: Path | None
    query_source_path: Path | None = None
    query_selected_track_id: str | None = None
    query_selected_occurrence_id: str | None = None


@dataclass(frozen=True)
class FaceSearchSummary:
    query_faces_detected: int
    compatible_clusters: int
    compatible_tracks: int
    compatible_occurrences: int
    compatibility_threshold: float
    query_image_count: int = 1
    query_faces_selected: int = 1
    query_images_rejected: int = 0


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
    tracks: list[FaceTrack] = field(default_factory=list)
    keyframes: list[KeyFrame] = field(default_factory=list)
    search: SearchArtifacts | None = None


@dataclass(frozen=True)
class FaceSearchResult:
    inventory_result: InventoryResult
    query: FaceSearchQuery | None
    matches: list[FaceSearchMatch]
    summary: FaceSearchSummary
    report: ReportArtifacts
    export_path: Path | None = None
    queries: list[FaceSearchQuery] = field(default_factory=list)
    query_events: list[FaceSearchQueryEvent] = field(default_factory=list)


@dataclass(frozen=True)
class FaceSetComparisonInput:
    set_label: str
    source_path: Path
    sha512: str
    detected_faces: int
    selected_faces: int
    tracks: int
    keyframes: int
    identity_label: str | None = None
    processing_error: str | None = None
    export_source_copy: Path | None = None


@dataclass(frozen=True)
class FaceSetComparisonEntry:
    entry_id: str
    set_label: str
    source_path: Path
    sha512: str
    track_id: str
    occurrence_id: str
    keyframe_id: str | None
    bbox: BoundingBox
    detection_score: float
    quality_score: float | None
    sharpness: float | None
    brightness: float | None
    illumination: float | None
    frontality: float | None
    embedding_dimension: int
    embedding_source: str | None
    crop_path: Path | None
    context_image_path: Path | None
    identity_label: str | None = None
    embedding: list[float] = field(default_factory=list)
    mesh_crop_path: Path | None = None
    mesh_context_path: Path | None = None
    selection_reasons: tuple[str, ...] = field(default_factory=tuple)
    biometric_landmarks: tuple[tuple[float, float], ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class FaceSetComparisonMatch:
    rank: int
    left_entry_id: str
    right_entry_id: str
    left_track_id: str
    right_track_id: str
    similarity: float
    classification: str
    left_quality_score: float | None = None
    right_quality_score: float | None = None
    likelihood_ratio: float | None = None
    log10_likelihood_ratio: float | None = None
    same_source_density: float | None = None
    different_source_density: float | None = None
    evidence_label: str | None = None


@dataclass(frozen=True)
class FaceSetComparisonCalibrationSummary:
    dataset_root: Path
    identity_count: int = 0
    processed_identities: int = 0
    input_images: int = 0
    processed_images: int = 0
    selected_faces: int = 0
    identities_with_selected_faces: int = 0
    genuine_pair_total: int = 0
    impostor_pair_total: int = 0
    genuine_score_count: int = 0
    impostor_score_count: int = 0
    support_ready: bool = False
    support_note: str | None = None
    score_min: float | None = None
    score_max: float | None = None
    density_method: str = "bounded_logit_kde"
    smoothing_note: str | None = None


@dataclass(frozen=True)
class FaceSetComparisonCalibration:
    summary: FaceSetComparisonCalibrationSummary
    inputs: list[FaceSetComparisonInput] = field(default_factory=list)
    entries: list[FaceSetComparisonEntry] = field(default_factory=list)
    genuine_scores: list[float] = field(default_factory=list)
    impostor_scores: list[float] = field(default_factory=list)
    procedure_details: tuple[str, ...] = field(default_factory=tuple)
    settings_snapshot: LikelihoodRatioSettings | None = None
    model_path: Path | None = None
    loaded_from_model: bool = False


@dataclass(frozen=True)
class FaceSetComparisonSummary:
    set_a_images: int = 0
    set_b_images: int = 0
    set_a_detected_faces: int = 0
    set_b_detected_faces: int = 0
    set_a_selected_faces: int = 0
    set_b_selected_faces: int = 0
    set_a_images_without_faces: int = 0
    set_b_images_without_faces: int = 0
    total_pair_comparisons: int = 0
    assignment_matches: int = 0
    candidate_matches: int = 0
    best_similarity: float | None = None
    worst_similarity: float | None = None
    mean_similarity: float | None = None
    median_similarity: float | None = None
    stddev_similarity: float | None = None
    q1_similarity: float | None = None
    q3_similarity: float | None = None
    mean_confidence_low: float | None = None
    mean_confidence_high: float | None = None
    candidate_threshold: float = 0.0
    assignment_threshold: float = 0.0
    likelihood_ratio_calibrated: bool = False
    calibrated_matches: int = 0
    mean_log10_likelihood_ratio: float | None = None
    median_log10_likelihood_ratio: float | None = None
    min_log10_likelihood_ratio: float | None = None
    max_log10_likelihood_ratio: float | None = None


@dataclass(frozen=True)
class FaceSetComparisonResult:
    run_directory: Path
    started_at_utc: datetime
    finished_at_utc: datetime
    logs_directory: Path
    export_directory: Path
    manifest_path: Path
    set_a_inputs: list[FaceSetComparisonInput] = field(default_factory=list)
    set_b_inputs: list[FaceSetComparisonInput] = field(default_factory=list)
    set_a_faces: list[FaceSetComparisonEntry] = field(default_factory=list)
    set_b_faces: list[FaceSetComparisonEntry] = field(default_factory=list)
    matches: list[FaceSetComparisonMatch] = field(default_factory=list)
    summary: FaceSetComparisonSummary = field(default_factory=FaceSetComparisonSummary)
    calibration: FaceSetComparisonCalibration | None = None
    procedure_details: tuple[str, ...] = field(default_factory=tuple)
