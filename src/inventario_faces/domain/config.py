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


@dataclass(frozen=True)
class ReportingSettings:
    max_gallery_faces_per_group: int
    compile_pdf: bool = True


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
