from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

from inventario_faces.domain.config import (
    AppConfig,
    AppSettings,
    ClusteringSettings,
    DistributedSettings,
    EnhancementSettings,
    FaceModelSettings,
    ForensicsSettings,
    LikelihoodRatioSettings,
    MediaSettings,
    ReportingSettings,
    SearchSettings,
    TrackingSettings,
    VideoSettings,
)
from inventario_faces.utils.path_utils import ensure_directory
from inventario_faces.utils.serialization import to_serializable


def _coerce_bool(value: Any, field_name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in {0, 1}:
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y", "on"}:
            return True
        if normalized in {"false", "0", "no", "n", "off"}:
            return False
    raise ValueError(f"{field_name} deve ser booleano.")


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
    defaults_path = locate_default_config()
    defaults = _load_yaml(defaults_path)
    merged = defaults
    source_path = defaults_path
    if config_path is not None:
        path = Path(config_path)
        if path.exists():
            merged = _deep_merge(defaults, _load_yaml(path))
            source_path = path

    try:
        return AppConfig(
            app=AppSettings(
                name=str(merged["app"]["name"]),
                output_directory_name=str(merged["app"]["output_directory_name"]),
                report_title=str(merged["app"]["report_title"]),
                organization=str(merged["app"]["organization"]),
                log_level=str(merged["app"].get("log_level", "INFO")),
                use_local_temp_copy=_coerce_bool(
                    merged["app"].get("use_local_temp_copy", False),
                    "app.use_local_temp_copy",
                ),
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
                compile_pdf=_coerce_bool(merged["reporting"].get("compile_pdf", True), "reporting.compile_pdf"),
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
                enable_preprocessing=_coerce_bool(
                    merged.get("enhancement", {}).get("enable_preprocessing", True),
                    "enhancement.enable_preprocessing",
                ),
                minimum_brightness_to_enhance=float(
                    merged.get("enhancement", {}).get("minimum_brightness_to_enhance", 0.36)
                ),
                clahe_clip_limit=float(merged.get("enhancement", {}).get("clahe_clip_limit", 2.0)),
                clahe_tile_grid_size=int(merged.get("enhancement", {}).get("clahe_tile_grid_size", 8)),
                gamma=float(merged.get("enhancement", {}).get("gamma", 1.0)),
                denoise_strength=int(merged.get("enhancement", {}).get("denoise_strength", 0)),
            ),
            search=SearchSettings(
                enabled=_coerce_bool(merged.get("search", {}).get("enabled", True), "search.enabled"),
                prefer_faiss=_coerce_bool(
                    merged.get("search", {}).get("prefer_faiss", True),
                    "search.prefer_faiss",
                ),
                coarse_top_k=int(merged.get("search", {}).get("coarse_top_k", 8)),
                refine_top_k=int(merged.get("search", {}).get("refine_top_k", 12)),
            ),
            likelihood_ratio=LikelihoodRatioSettings(
                max_scores_per_distribution=int(
                    merged.get("likelihood_ratio", {}).get("max_scores_per_distribution", 20000)
                ),
                minimum_identities_with_faces=int(
                    merged.get("likelihood_ratio", {}).get("minimum_identities_with_faces", 2)
                ),
                minimum_same_source_scores=int(
                    merged.get("likelihood_ratio", {}).get("minimum_same_source_scores", 5)
                ),
                minimum_different_source_scores=int(
                    merged.get("likelihood_ratio", {}).get("minimum_different_source_scores", 5)
                ),
                minimum_unique_scores_per_distribution=int(
                    merged.get("likelihood_ratio", {}).get("minimum_unique_scores_per_distribution", 2)
                ),
                kde_bandwidth_scale=float(
                    merged.get("likelihood_ratio", {}).get("kde_bandwidth_scale", 1.0)
                ),
                kde_uniform_floor_weight=float(
                    merged.get("likelihood_ratio", {}).get("kde_uniform_floor_weight", 0.001)
                ),
                kde_min_density=float(
                    merged.get("likelihood_ratio", {}).get("kde_min_density", 1e-12)
                ),
            ),
            distributed=DistributedSettings(
                enabled=_coerce_bool(merged.get("distributed", {}).get("enabled", False), "distributed.enabled"),
                execution_label=str(merged.get("distributed", {}).get("execution_label", "compartilhado")),
                node_name=(
                    str(merged.get("distributed", {}).get("node_name")).strip()
                    if merged.get("distributed", {}).get("node_name") not in (None, "")
                    else None
                ),
                heartbeat_interval_seconds=int(
                    merged.get("distributed", {}).get("heartbeat_interval_seconds", 15)
                ),
                stale_lock_timeout_minutes=int(
                    merged.get("distributed", {}).get("stale_lock_timeout_minutes", 120)
                ),
                auto_finalize=_coerce_bool(
                    merged.get("distributed", {}).get("auto_finalize", True),
                    "distributed.auto_finalize",
                ),
                validate_partial_integrity=_coerce_bool(
                    merged.get("distributed", {}).get("validate_partial_integrity", True),
                    "distributed.validate_partial_integrity",
                ),
                auto_reprocess_invalid_partials=_coerce_bool(
                    merged.get("distributed", {}).get("auto_reprocess_invalid_partials", True),
                    "distributed.auto_reprocess_invalid_partials",
                ),
            ),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Configuracao invalida em {source_path}: {exc}") from exc


def load_default_app_config() -> AppConfig:
    """Carrega apenas os valores padrao oficiais do aplicativo."""

    return load_app_config(locate_default_config())


def save_app_config(config: AppConfig, path: Path) -> Path:
    payload = to_serializable(config)
    output_path = Path(path).expanduser()
    ensure_directory(output_path.parent)
    with output_path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(payload, stream, allow_unicode=True, sort_keys=False)
    return output_path
