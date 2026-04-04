from __future__ import annotations

import csv
import json
from pathlib import Path

from inventario_faces.domain.entities import FaceCluster, FaceOccurrence, FileRecord, InventoryResult
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
                        item.cluster_id or "",
                        ";".join(item.suggested_cluster_ids),
                        str(item.crop_path) if item.crop_path else "",
                        str(item.context_image_path) if item.context_image_path else "",
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
