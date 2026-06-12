from __future__ import annotations

from datetime import datetime
from pathlib import Path

from inventario_faces.domain.entities import (
    BoundingBox,
    EnhancementMetadata,
    FaceOccurrence,
    FaceQualityMetrics,
    FaceTrack,
    FileRecord,
    KeyFrame,
    MediaInfoAttribute,
    MediaInfoTrack,
    MediaType,
    TrackQualityStatistics,
)
from inventario_faces.utils.time_utils import utc_now


def deserialize_partial_payload(payload: dict[str, object]) -> dict[str, object]:
    return {
        "file_record": deserialize_file_record(payload.get("file_record", {})),
        "occurrences": [deserialize_occurrence(item) for item in payload.get("occurrences", [])],
        "tracks": [deserialize_track(item) for item in payload.get("tracks", [])],
        "keyframes": [deserialize_keyframe(item) for item in payload.get("keyframes", [])],
        "raw_face_sizes": [float(item) for item in payload.get("raw_face_sizes", [])],
        "selected_face_sizes": [float(item) for item in payload.get("selected_face_sizes", [])],
    }


def deserialize_file_record(payload: object) -> FileRecord:
    data = payload if isinstance(payload, dict) else {}
    return FileRecord(
        path=Path(str(data.get("path", ""))),
        media_type=MediaType(str(data.get("media_type", MediaType.OTHER.value))),
        sha512=str(data.get("sha512", "")),
        size_bytes=int(data.get("size_bytes", 0)),
        discovered_at_utc=parse_datetime(data.get("discovered_at_utc")),
        modified_at_utc=parse_datetime_optional(data.get("modified_at_utc")),
        processing_error=(str(data["processing_error"]) if data.get("processing_error") else None),
        media_info_tracks=tuple(
            deserialize_media_info_track(item) for item in data.get("media_info_tracks", [])
        ),
        media_info_error=(str(data["media_info_error"]) if data.get("media_info_error") else None),
    )


def deserialize_occurrence(payload: object) -> FaceOccurrence:
    data = payload if isinstance(payload, dict) else {}
    return FaceOccurrence(
        occurrence_id=str(data.get("occurrence_id", "")),
        source_path=Path(str(data.get("source_path", ""))),
        sha512=str(data.get("sha512", "")),
        media_type=MediaType(str(data.get("media_type", MediaType.OTHER.value))),
        analysis_timestamp_utc=parse_datetime(data.get("analysis_timestamp_utc")),
        frame_index=(None if data.get("frame_index") is None else int(data.get("frame_index"))),
        frame_timestamp_seconds=(None if data.get("frame_timestamp_seconds") is None else float(data.get("frame_timestamp_seconds"))),
        bbox=deserialize_bbox(data.get("bbox", {})),
        detection_score=float(data.get("detection_score", 0.0)),
        crop_path=(Path(str(data["crop_path"])) if data.get("crop_path") else None),
        embedding=[float(item) for item in data.get("embedding", [])],
        context_image_path=(Path(str(data["context_image_path"])) if data.get("context_image_path") else None),
        cluster_id=(str(data["cluster_id"]) if data.get("cluster_id") else None),
        suggested_cluster_ids=[str(item) for item in data.get("suggested_cluster_ids", [])],
        track_id=(str(data["track_id"]) if data.get("track_id") else None),
        keyframe_id=(str(data["keyframe_id"]) if data.get("keyframe_id") else None),
        quality_metrics=deserialize_quality_metrics_optional(data.get("quality_metrics")),
        enhancement_metadata=deserialize_enhancement_optional(data.get("enhancement_metadata")),
        is_keyframe=bool(data.get("is_keyframe", False)),
        track_position=(None if data.get("track_position") is None else int(data.get("track_position"))),
        embedding_source=(str(data["embedding_source"]) if data.get("embedding_source") else None),
    )


def deserialize_track(payload: object) -> FaceTrack:
    data = payload if isinstance(payload, dict) else {}
    return FaceTrack(
        track_id=str(data.get("track_id", "")),
        source_path=Path(str(data.get("source_path", ""))),
        video_path=(Path(str(data["video_path"])) if data.get("video_path") else None),
        media_type=MediaType(str(data.get("media_type", MediaType.OTHER.value))),
        sha512=str(data.get("sha512", "")),
        start_frame=(None if data.get("start_frame") is None else int(data.get("start_frame"))),
        end_frame=(None if data.get("end_frame") is None else int(data.get("end_frame"))),
        start_time=(None if data.get("start_time") is None else float(data.get("start_time"))),
        end_time=(None if data.get("end_time") is None else float(data.get("end_time"))),
        occurrence_ids=[str(item) for item in data.get("occurrence_ids", [])],
        keyframe_ids=[str(item) for item in data.get("keyframe_ids", [])],
        representative_embeddings=[
            [float(value) for value in embedding]
            for embedding in data.get("representative_embeddings", [])
        ],
        average_embedding=[float(item) for item in data.get("average_embedding", [])],
        best_occurrence_id=(str(data["best_occurrence_id"]) if data.get("best_occurrence_id") else None),
        preview_path=(Path(str(data["preview_path"])) if data.get("preview_path") else None),
        top_crop_paths=[Path(str(item)) for item in data.get("top_crop_paths", [])],
        quality_statistics=deserialize_track_quality_statistics(data.get("quality_statistics")),
        cluster_id=(str(data["cluster_id"]) if data.get("cluster_id") else None),
        candidate_cluster_ids=[str(item) for item in data.get("candidate_cluster_ids", [])],
    )


def deserialize_keyframe(payload: object) -> KeyFrame:
    data = payload if isinstance(payload, dict) else {}
    return KeyFrame(
        keyframe_id=str(data.get("keyframe_id", "")),
        track_id=str(data.get("track_id", "")),
        occurrence_id=str(data.get("occurrence_id", "")),
        source_path=Path(str(data.get("source_path", ""))),
        frame_index=(None if data.get("frame_index") is None else int(data.get("frame_index"))),
        timestamp_seconds=(None if data.get("timestamp_seconds") is None else float(data.get("timestamp_seconds"))),
        selection_reasons=tuple(str(item) for item in data.get("selection_reasons", [])),
        quality_metrics=deserialize_quality_metrics_optional(data.get("quality_metrics")),
        detection_score=float(data.get("detection_score", 0.0)),
        crop_path=(Path(str(data["crop_path"])) if data.get("crop_path") else None),
        context_image_path=(Path(str(data["context_image_path"])) if data.get("context_image_path") else None),
        embedding=[float(item) for item in data.get("embedding", [])],
        preview_path=(Path(str(data["preview_path"])) if data.get("preview_path") else None),
    )


def deserialize_media_info_track(payload: object) -> MediaInfoTrack:
    data = payload if isinstance(payload, dict) else {}
    return MediaInfoTrack(
        track_type=str(data.get("track_type", "")),
        attributes=tuple(
            MediaInfoAttribute(
                label=str(item.get("label", "")),
                value=str(item.get("value", "")),
            )
            for item in data.get("attributes", [])
            if isinstance(item, dict)
        ),
    )


def deserialize_bbox(payload: object) -> BoundingBox:
    data = payload if isinstance(payload, dict) else {}
    return BoundingBox(
        x1=float(data.get("x1", 0.0)),
        y1=float(data.get("y1", 0.0)),
        x2=float(data.get("x2", 0.0)),
        y2=float(data.get("y2", 0.0)),
    )


def deserialize_quality_metrics_optional(payload: object) -> FaceQualityMetrics | None:
    if not isinstance(payload, dict):
        return None
    return FaceQualityMetrics(
        detection_score=float(payload.get("detection_score", 0.0)),
        sharpness=float(payload.get("sharpness", 0.0)),
        brightness=float(payload.get("brightness", 0.0)),
        illumination=float(payload.get("illumination", 0.0)),
        frontality=float(payload.get("frontality", 0.0)),
        bbox_pixels=float(payload.get("bbox_pixels", 0.0)),
        score=float(payload.get("score", 0.0)),
    )


def deserialize_enhancement_optional(payload: object) -> EnhancementMetadata | None:
    if not isinstance(payload, dict):
        return None
    return EnhancementMetadata(
        applied=bool(payload.get("applied", False)),
        strategy=str(payload.get("strategy", "none")),
        parameters={
            str(key): value
            for key, value in payload.get("parameters", {}).items()
        },
        brightness_before=(
            None if payload.get("brightness_before") is None else float(payload.get("brightness_before"))
        ),
        brightness_after=(
            None if payload.get("brightness_after") is None else float(payload.get("brightness_after"))
        ),
        note=(str(payload["note"]) if payload.get("note") else None),
    )


def deserialize_track_quality_statistics(payload: object) -> TrackQualityStatistics:
    data = payload if isinstance(payload, dict) else {}
    return TrackQualityStatistics(
        total_detections=int(data.get("total_detections", 0)),
        keyframe_count=int(data.get("keyframe_count", 0)),
        mean_detection_score=float(data.get("mean_detection_score", 0.0)),
        max_detection_score=float(data.get("max_detection_score", 0.0)),
        mean_quality_score=float(data.get("mean_quality_score", 0.0)),
        best_quality_score=float(data.get("best_quality_score", 0.0)),
        mean_sharpness=float(data.get("mean_sharpness", 0.0)),
        mean_brightness=float(data.get("mean_brightness", 0.0)),
        mean_illumination=float(data.get("mean_illumination", 0.0)),
        mean_frontality=float(data.get("mean_frontality", 0.0)),
        duration_seconds=float(data.get("duration_seconds", 0.0)),
    )


def parse_datetime(value: object) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str) and value:
        return datetime.fromisoformat(value)
    return utc_now()


def parse_datetime_optional(value: object) -> datetime | None:
    if value in (None, ""):
        return None
    return parse_datetime(value)
