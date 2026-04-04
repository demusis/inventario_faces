from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class AppSettings:
    name: str
    output_directory_name: str
    report_title: str
    organization: str
    log_level: str = "INFO"
    mediainfo_directory: str | None = None


@dataclass(frozen=True)
class MediaSettings:
    image_extensions: tuple[str, ...]
    video_extensions: tuple[str, ...]


@dataclass(frozen=True)
class VideoSettings:
    sampling_interval_seconds: float
    max_frames_per_video: int | None = None
    keyframe_interval_seconds: float = 3.0
    significant_change_threshold: float = 0.18


@dataclass(frozen=True)
class FaceModelSettings:
    backend: str
    model_name: str
    det_size: tuple[int, int] | None
    minimum_face_quality: float = 0.6
    minimum_face_size_pixels: int = 40
    ctx_id: int = 0
    providers: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class ClusteringSettings:
    assignment_similarity: float
    candidate_similarity: float
    min_cluster_size: int = 1
    min_track_size: int = 1


@dataclass(frozen=True)
class TrackingSettings:
    iou_threshold: float = 0.15
    spatial_distance_threshold: float = 0.18
    embedding_similarity_threshold: float = 0.48
    minimum_total_match_score: float = 0.30
    geometry_weight: float = 0.45
    embedding_weight: float = 0.55
    max_missed_detections: int = 2
    confidence_margin: float = 0.05
    representative_embeddings_per_track: int = 5
    top_crops_per_track: int = 4
    quality_improvement_margin: float = 0.05


@dataclass(frozen=True)
class EnhancementSettings:
    enable_preprocessing: bool = True
    minimum_brightness_to_enhance: float = 0.36
    clahe_clip_limit: float = 2.0
    clahe_tile_grid_size: int = 8
    gamma: float = 1.0
    denoise_strength: int = 0


@dataclass(frozen=True)
class SearchSettings:
    enabled: bool = True
    prefer_faiss: bool = True
    coarse_top_k: int = 8
    refine_top_k: int = 12


@dataclass(frozen=True)
class ReportingSettings:
    compile_pdf: bool = True
    max_tracks_per_group: int = 8


@dataclass(frozen=True)
class ForensicsSettings:
    chain_of_custody_note: str


@dataclass(frozen=True)
class AppConfig:
    app: AppSettings
    media: MediaSettings
    video: VideoSettings
    face_model: FaceModelSettings
    clustering: ClusteringSettings
    reporting: ReportingSettings
    forensics: ForensicsSettings
    tracking: TrackingSettings = field(default_factory=TrackingSettings)
    enhancement: EnhancementSettings = field(default_factory=EnhancementSettings)
    search: SearchSettings = field(default_factory=SearchSettings)
