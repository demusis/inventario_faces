from __future__ import annotations

import csv
import json
from pathlib import Path

from inventario_faces.domain.entities import (
    FaceCluster,
    FaceSearchResult,
    FaceOccurrence,
    FaceTrack,
    FileRecord,
    InventoryResult,
    KeyFrame,
    SearchArtifacts,
)
from inventario_faces.utils.path_utils import ensure_directory
from inventario_faces.utils.serialization import to_serializable


class ExportService:
    def __init__(self, run_directory: Path) -> None:
        self._inventory_directory = ensure_directory(run_directory / "inventory")

    @property
    def inventory_directory(self) -> Path:
        return self._inventory_directory

    def write_files_csv(self, files: list[FileRecord]) -> Path:
        output_path = self._inventory_directory / "files.csv"
        with output_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(
                [
                    "path",
                    "media_type",
                    "sha512",
                    "size_bytes",
                    "discovered_at_utc",
                    "modified_at_utc",
                    "processing_error",
                    "media_info_track_count",
                    "media_info_error",
                ]
            )
            for item in files:
                writer.writerow(
                    [
                        str(item.path),
                        item.media_type.value,
                        item.sha512,
                        item.size_bytes,
                        item.discovered_at_utc.isoformat(),
                        item.modified_at_utc.isoformat() if item.modified_at_utc else "",
                        item.processing_error or "",
                        len(item.media_info_tracks),
                        item.media_info_error or "",
                    ]
                )
        return output_path

    def write_occurrences_csv(self, occurrences: list[FaceOccurrence]) -> Path:
        output_path = self._inventory_directory / "occurrences.csv"
        with output_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(
                [
                    "occurrence_id",
                    "source_path",
                    "sha512",
                    "media_type",
                    "analysis_timestamp_utc",
                    "frame_index",
                    "frame_timestamp_seconds",
                    "bbox",
                    "detection_score",
                    "track_id",
                    "keyframe_id",
                    "is_keyframe",
                    "embedding_source",
                    "quality_score",
                    "sharpness",
                    "brightness",
                    "illumination",
                    "frontality",
                    "cluster_id",
                    "suggested_cluster_ids",
                    "crop_path",
                    "context_image_path",
                ]
            )
            for item in occurrences:
                writer.writerow(
                    [
                        item.occurrence_id,
                        str(item.source_path),
                        item.sha512,
                        item.media_type.value,
                        item.analysis_timestamp_utc.isoformat(),
                        item.frame_index,
                        item.frame_timestamp_seconds,
                        f"{item.bbox.x1:.2f},{item.bbox.y1:.2f},{item.bbox.x2:.2f},{item.bbox.y2:.2f}",
                        f"{item.detection_score:.6f}",
                        item.track_id or "",
                        item.keyframe_id or "",
                        "1" if item.is_keyframe else "0",
                        item.embedding_source or "",
                        f"{item.quality_metrics.score:.6f}" if item.quality_metrics is not None else "",
                        f"{item.quality_metrics.sharpness:.6f}" if item.quality_metrics is not None else "",
                        f"{item.quality_metrics.brightness:.6f}" if item.quality_metrics is not None else "",
                        f"{item.quality_metrics.illumination:.6f}" if item.quality_metrics is not None else "",
                        f"{item.quality_metrics.frontality:.6f}" if item.quality_metrics is not None else "",
                        item.cluster_id or "",
                        ";".join(item.suggested_cluster_ids),
                        str(item.crop_path) if item.crop_path else "",
                        str(item.context_image_path) if item.context_image_path else "",
                    ]
                )
        return output_path

    def write_tracks_csv(self, tracks: list[FaceTrack]) -> Path:
        output_path = self._inventory_directory / "tracks.csv"
        with output_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(
                [
                    "track_id",
                    "source_path",
                    "video_path",
                    "media_type",
                    "sha512",
                    "start_frame",
                    "end_frame",
                    "start_time",
                    "end_time",
                    "occurrence_count",
                    "keyframe_count",
                    "cluster_id",
                    "candidate_cluster_ids",
                    "best_occurrence_id",
                    "preview_path",
                    "top_crop_paths",
                    "mean_quality_score",
                    "best_quality_score",
                    "mean_sharpness",
                    "mean_brightness",
                    "mean_illumination",
                    "mean_frontality",
                    "duration_seconds",
                ]
            )
            for item in tracks:
                writer.writerow(
                    [
                        item.track_id,
                        str(item.source_path),
                        str(item.video_path) if item.video_path is not None else "",
                        item.media_type.value,
                        item.sha512,
                        item.start_frame,
                        item.end_frame,
                        item.start_time,
                        item.end_time,
                        len(item.occurrence_ids),
                        len(item.keyframe_ids),
                        item.cluster_id or "",
                        ";".join(item.candidate_cluster_ids),
                        item.best_occurrence_id or "",
                        str(item.preview_path) if item.preview_path is not None else "",
                        ";".join(str(path) for path in item.top_crop_paths),
                        f"{item.quality_statistics.mean_quality_score:.6f}",
                        f"{item.quality_statistics.best_quality_score:.6f}",
                        f"{item.quality_statistics.mean_sharpness:.6f}",
                        f"{item.quality_statistics.mean_brightness:.6f}",
                        f"{item.quality_statistics.mean_illumination:.6f}",
                        f"{item.quality_statistics.mean_frontality:.6f}",
                        f"{item.quality_statistics.duration_seconds:.6f}",
                    ]
                )
        return output_path

    def write_keyframes_csv(self, keyframes: list[KeyFrame]) -> Path:
        output_path = self._inventory_directory / "keyframes.csv"
        with output_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(
                [
                    "keyframe_id",
                    "track_id",
                    "occurrence_id",
                    "source_path",
                    "frame_index",
                    "timestamp_seconds",
                    "selection_reasons",
                    "detection_score",
                    "quality_score",
                    "crop_path",
                    "context_image_path",
                    "preview_path",
                ]
            )
            for item in keyframes:
                writer.writerow(
                    [
                        item.keyframe_id,
                        item.track_id,
                        item.occurrence_id,
                        str(item.source_path),
                        item.frame_index,
                        item.timestamp_seconds,
                        ";".join(item.selection_reasons),
                        f"{item.detection_score:.6f}",
                        f"{item.quality_metrics.score:.6f}" if item.quality_metrics is not None else "",
                        str(item.crop_path) if item.crop_path is not None else "",
                        str(item.context_image_path) if item.context_image_path is not None else "",
                        str(item.preview_path) if item.preview_path is not None else "",
                    ]
                )
        return output_path

    def write_clusters_json(self, clusters: list[FaceCluster]) -> Path:
        output_path = self._inventory_directory / "clusters.json"
        output_path.write_text(
            json.dumps(to_serializable(clusters), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return output_path

    def write_search_json(self, search: SearchArtifacts | None) -> Path:
        output_path = self._inventory_directory / "search.json"
        output_path.write_text(
            json.dumps(to_serializable(search), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return output_path

    def write_media_info_json(self, files: list[FileRecord]) -> Path:
        output_path = self._inventory_directory / "media_info.json"
        payload = [
            {
                "path": str(item.path),
                "media_type": item.media_type.value,
                "media_info_error": item.media_info_error,
                "media_info_tracks": to_serializable(item.media_info_tracks),
            }
            for item in files
            if item.media_type.value in {"image", "video"}
        ]
        output_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return output_path

    def write_manifest(self, result: InventoryResult) -> Path:
        output_path = self._inventory_directory / "manifest.json"
        output_path.write_text(
            json.dumps(to_serializable(result), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return output_path

    def write_face_search_json(self, result: FaceSearchResult) -> Path:
        output_path = self._inventory_directory / "face_search.json"
        output_path.write_text(
            json.dumps(to_serializable(result), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return output_path
