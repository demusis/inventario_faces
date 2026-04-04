from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

from inventario_faces.domain.config import (
    AppConfig,
    AppSettings,
    ClusteringSettings,
    EnhancementSettings,
    FaceModelSettings,
    ForensicsSettings,
    MediaSettings,
    ReportingSettings,
    SearchSettings,
    TrackingSettings,
    VideoSettings,
)
from inventario_faces.utils.path_utils import ensure_directory
from inventario_faces.utils.serialization import to_serializable


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        current = merged.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            merged[key] = _deep_merge(current, value)
        else:
            merged[key] = value
    return merged


def _candidate_default_paths() -> list[Path]:
    current_file = Path(__file__).resolve()
    return [
        current_file.parents[3] / "config" / "defaults.yaml",
        current_file.parents[1] / "config" / "defaults.yaml",
    ]


def locate_default_config() -> Path:
    for candidate in _candidate_default_paths():
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Arquivo de configuracao padrao nao encontrado.")


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        loaded = yaml.safe_load(stream) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Configuracao invalida em {path}")
    return loaded


def default_user_config_path() -> Path:
    appdata_directory = Path(os.getenv("APPDATA", Path.home() / "AppData" / "Roaming"))
    return appdata_directory / "InventarioFaces" / "config.yaml"


def load_app_config(config_path: Path | None = None) -> AppConfig:
    defaults = _load_yaml(locate_default_config())
    merged = defaults
    if config_path is not None:
        path = Path(config_path)
        if path.exists():
            merged = _deep_merge(defaults, _load_yaml(path))

    mediainfo_directory = merged["app"].get("mediainfo_directory")
    normalized_mediainfo_directory = (
        str(mediainfo_directory).strip() if mediainfo_directory not in (None, "") else None
    )

    return AppConfig(
        app=AppSettings(
            name=str(merged["app"]["name"]),
            output_directory_name=str(merged["app"]["output_directory_name"]),
            report_title=str(merged["app"]["report_title"]),
            organization=str(merged["app"]["organization"]),
            log_level=str(merged["app"].get("log_level", "INFO")),
            mediainfo_directory=normalized_mediainfo_directory,
        ),
        media=MediaSettings(
            image_extensions=tuple(str(item).lower() for item in merged["media"]["image_extensions"]),
            video_extensions=tuple(str(item).lower() for item in merged["media"]["video_extensions"]),
        ),
        video=VideoSettings(
            sampling_interval_seconds=float(merged["video"]["sampling_interval_seconds"]),
            max_frames_per_video=(
                int(merged["video"]["max_frames_per_video"])
                if merged["video"].get("max_frames_per_video") is not None
                else None
            ),
            keyframe_interval_seconds=float(merged["video"].get("keyframe_interval_seconds", 3.0)),
            significant_change_threshold=float(merged["video"].get("significant_change_threshold", 0.18)),
        ),
        face_model=FaceModelSettings(
            backend=str(merged["face_model"]["backend"]),
            model_name=str(merged["face_model"]["model_name"]),
            det_size=(
                tuple(int(item) for item in merged["face_model"]["det_size"])
                if merged["face_model"].get("det_size") is not None
                else None
            ),
            minimum_face_quality=float(merged["face_model"].get("minimum_face_quality", 0.6)),
            minimum_face_size_pixels=int(merged["face_model"].get("minimum_face_size_pixels", 40)),
            ctx_id=int(merged["face_model"].get("ctx_id", 0)),
            providers=tuple(str(item) for item in merged["face_model"].get("providers", [])),
        ),
        clustering=ClusteringSettings(
            assignment_similarity=float(merged["clustering"]["assignment_similarity"]),
            candidate_similarity=float(merged["clustering"]["candidate_similarity"]),
            min_cluster_size=int(merged["clustering"].get("min_cluster_size", 1)),
            min_track_size=int(merged["clustering"].get("min_track_size", 1)),
        ),
        reporting=ReportingSettings(
            compile_pdf=bool(merged["reporting"].get("compile_pdf", True)),
            max_tracks_per_group=int(merged["reporting"].get("max_tracks_per_group", 8)),
        ),
        forensics=ForensicsSettings(
            chain_of_custody_note=str(merged["forensics"]["chain_of_custody_note"]),
        ),
        tracking=TrackingSettings(
            iou_threshold=float(merged.get("tracking", {}).get("iou_threshold", 0.15)),
            spatial_distance_threshold=float(merged.get("tracking", {}).get("spatial_distance_threshold", 0.18)),
            embedding_similarity_threshold=float(merged.get("tracking", {}).get("embedding_similarity_threshold", 0.48)),
            minimum_total_match_score=float(merged.get("tracking", {}).get("minimum_total_match_score", 0.30)),
            geometry_weight=float(merged.get("tracking", {}).get("geometry_weight", 0.45)),
            embedding_weight=float(merged.get("tracking", {}).get("embedding_weight", 0.55)),
            max_missed_detections=int(merged.get("tracking", {}).get("max_missed_detections", 2)),
            confidence_margin=float(merged.get("tracking", {}).get("confidence_margin", 0.05)),
            representative_embeddings_per_track=int(
                merged.get("tracking", {}).get("representative_embeddings_per_track", 5)
            ),
            top_crops_per_track=int(merged.get("tracking", {}).get("top_crops_per_track", 4)),
            quality_improvement_margin=float(merged.get("tracking", {}).get("quality_improvement_margin", 0.05)),
        ),
        enhancement=EnhancementSettings(
            enable_preprocessing=bool(merged.get("enhancement", {}).get("enable_preprocessing", True)),
            minimum_brightness_to_enhance=float(
                merged.get("enhancement", {}).get("minimum_brightness_to_enhance", 0.36)
            ),
            clahe_clip_limit=float(merged.get("enhancement", {}).get("clahe_clip_limit", 2.0)),
            clahe_tile_grid_size=int(merged.get("enhancement", {}).get("clahe_tile_grid_size", 8)),
            gamma=float(merged.get("enhancement", {}).get("gamma", 1.0)),
            denoise_strength=int(merged.get("enhancement", {}).get("denoise_strength", 0)),
        ),
        search=SearchSettings(
            enabled=bool(merged.get("search", {}).get("enabled", True)),
            prefer_faiss=bool(merged.get("search", {}).get("prefer_faiss", True)),
            coarse_top_k=int(merged.get("search", {}).get("coarse_top_k", 8)),
            refine_top_k=int(merged.get("search", {}).get("refine_top_k", 12)),
        ),
    )


def save_app_config(config: AppConfig, path: Path) -> Path:
    payload = to_serializable(config)
    output_path = Path(path).expanduser()
    ensure_directory(output_path.parent)
    with output_path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(payload, stream, allow_unicode=True, sort_keys=False)
    return output_path
