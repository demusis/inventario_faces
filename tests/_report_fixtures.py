"""Construtores compartilhados de configuracao e resultado para testes de relatorio."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from inventario_faces.domain.config import (
    AppConfig,
    AppSettings,
    ClusteringSettings,
    FaceModelSettings,
    ForensicsSettings,
    MediaSettings,
    ReportingSettings,
    VideoSettings,
)
from inventario_faces.domain.entities import (
    BoundingBox,
    FaceCluster,
    FaceOccurrence,
    FaceSizeStatistics,
    FaceTrack,
    FileRecord,
    InventoryResult,
    KeyFrame,
    MediaInfoAttribute,
    MediaInfoTrack,
    MediaType,
    ProcessingSummary,
    ReportArtifacts,
    SearchArtifacts,
    TrackQualityStatistics,
)


def build_report_config(
    *,
    compile_pdf: bool = False,
    max_tracks_per_group: int = 8,
    chain_of_custody_note: str = "Arquivos originais preservados.",
) -> AppConfig:
    return AppConfig(
        app=AppSettings(
            name="Inventario Faces",
            output_directory_name="inventario_faces_output",
            report_title="Relatório Forense de Inventário Facial",
            organization="Laboratório Teste",
            log_level="INFO",
        ),
        media=MediaSettings(
            image_extensions=(".jpg", ".png"),
            video_extensions=(".mp4", ".avi"),
        ),
        video=VideoSettings(
            sampling_interval_seconds=1.0,
            max_frames_per_video=10,
        ),
        face_model=FaceModelSettings(
            backend="insightface",
            model_name="buffalo_l",
            det_size=(640, 640),
            minimum_face_quality=0.6,
            minimum_face_size_pixels=40,
            ctx_id=0,
            providers=("CPUExecutionProvider",),
        ),
        clustering=ClusteringSettings(
            assignment_similarity=0.52,
            candidate_similarity=0.44,
            min_cluster_size=1,
        ),
        reporting=ReportingSettings(
            compile_pdf=compile_pdf,
            max_tracks_per_group=max_tracks_per_group,
        ),
        forensics=ForensicsSettings(chain_of_custody_note=chain_of_custody_note),
    )


def build_inventory_result(
    root: Path,
    logs_dir: Path,
    *,
    track_count: int = 1,
    source_name: str = "evidencia.mp4",
) -> InventoryResult:
    timestamp = datetime.now(UTC)
    occurrences: list[FaceOccurrence] = []
    tracks: list[FaceTrack] = []
    keyframes: list[KeyFrame] = []
    track_ids: list[str] = []
    occurrence_ids: list[str] = []

    for index in range(1, track_count + 1):
        occurrence_id = f"O{index:06d}"
        track_id = f"T{index:06d}"
        keyframe_id = f"K{index:06d}"
        track_ids.append(track_id)
        occurrence_ids.append(occurrence_id)
        occurrences.append(
            FaceOccurrence(
                occurrence_id=occurrence_id,
                source_path=Path(source_name),
                sha512="a" * 128,
                media_type=MediaType.VIDEO,
                analysis_timestamp_utc=timestamp,
                frame_index=10 * index,
                frame_timestamp_seconds=10.0 * index,
                bbox=BoundingBox(10, 20, 60, 90),
                detection_score=0.83,
                embedding=[1.0, 0.0, 0.0],
                crop_path=None,
                context_image_path=None,
                cluster_id="I001",
                track_id=track_id,
            )
        )
        tracks.append(
            FaceTrack(
                track_id=track_id,
                source_path=Path(source_name),
                video_path=Path(source_name),
                media_type=MediaType.VIDEO,
                sha512="a" * 128,
                start_frame=10 * index,
                end_frame=10 * index,
                start_time=10.0 * index,
                end_time=10.0 * index,
                occurrence_ids=[occurrence_id],
                keyframe_ids=[keyframe_id],
                representative_embeddings=[[1.0, 0.0, 0.0]],
                average_embedding=[1.0, 0.0, 0.0],
                best_occurrence_id=occurrence_id,
                quality_statistics=TrackQualityStatistics(
                    total_detections=1,
                    keyframe_count=1,
                    mean_detection_score=0.83,
                    max_detection_score=0.83,
                    mean_quality_score=0.80,
                    best_quality_score=0.80,
                    mean_sharpness=0.70,
                    mean_brightness=0.55,
                    mean_illumination=0.90,
                    mean_frontality=0.88,
                    duration_seconds=0.0,
                ),
                cluster_id="I001",
            )
        )
        keyframes.append(
            KeyFrame(
                keyframe_id=keyframe_id,
                track_id=track_id,
                occurrence_id=occurrence_id,
                source_path=Path(source_name),
                frame_index=10 * index,
                timestamp_seconds=10.0 * index,
                selection_reasons=("track_start", "quality_peak"),
                detection_score=0.83,
                embedding=[1.0, 0.0, 0.0],
            )
        )

    cluster = FaceCluster(
        cluster_id="I001",
        track_ids=track_ids,
        occurrence_ids=occurrence_ids,
        centroid_embedding=[1.0, 0.0, 0.0],
    )
    file_record = FileRecord(
        path=Path(source_name),
        media_type=MediaType.VIDEO,
        sha512="a" * 128,
        size_bytes=123456,
        discovered_at_utc=timestamp,
        modified_at_utc=timestamp,
        media_info_tracks=(
            MediaInfoTrack(
                track_type="Geral",
                attributes=(
                    MediaInfoAttribute(label="Formato", value="MPEG-4"),
                    MediaInfoAttribute(label="Duração", value="00:00:10.000"),
                ),
            ),
        ),
    )
    summary = ProcessingSummary(
        total_files=1,
        media_files=1,
        image_files=0,
        video_files=1,
        total_occurrences=len(occurrences),
        total_clusters=1,
        probable_match_pairs=0,
        total_tracks=len(tracks),
        total_keyframes=len(keyframes),
        total_detected_face_sizes=FaceSizeStatistics(
            count=1, min_pixels=50, max_pixels=50, mean_pixels=50, stddev_pixels=0
        ),
        selected_face_sizes=FaceSizeStatistics(
            count=1, min_pixels=50, max_pixels=50, mean_pixels=50, stddev_pixels=0
        ),
    )
    return InventoryResult(
        run_directory=root,
        started_at_utc=timestamp,
        finished_at_utc=timestamp,
        root_directory=root,
        files=[file_record],
        occurrences=occurrences,
        clusters=[cluster],
        report=ReportArtifacts(tex_path=root / "report" / "relatorio_forense.tex", pdf_path=None),
        summary=summary,
        logs_directory=logs_dir,
        manifest_path=root / "inventory" / "manifest.json",
        tracks=tracks,
        keyframes=keyframes,
        search=SearchArtifacts(
            engine="numpy",
            track_index_path=None,
            track_metadata_path=None,
            cluster_index_path=None,
            cluster_metadata_path=None,
            track_vector_count=len(tracks),
            cluster_vector_count=1,
        ),
    )
