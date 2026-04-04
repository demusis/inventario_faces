from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from tests._bootstrap import PROJECT_ROOT  # noqa: F401
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
from inventario_faces.domain.entities import BoundingBox, DetectedFace, MediaType, SampledFrame
from inventario_faces.infrastructure.artifact_store import ArtifactStore
from inventario_faces.services.enhancement_service import EnhancementService
from inventario_faces.services.quality_service import FaceQualityService
from inventario_faces.services.tracking_service import FaceTrackingService


class _ProgrammedAnalyzer:
    def __init__(self, detections_by_frame: dict[int | None, list[DetectedFace]], embeddings: dict[tuple[float, float, float, float], list[float]]) -> None:
        self._detections_by_frame = detections_by_frame
        self._embeddings = embeddings

    def detect(self, frame: SampledFrame) -> list[DetectedFace]:
        return list(self._detections_by_frame.get(frame.frame_index, []))

    def embed(self, frame: SampledFrame, detection: DetectedFace, reason: str = "keyframe") -> list[float]:
        key = (
            detection.bbox.x1,
            detection.bbox.y1,
            detection.bbox.x2,
            detection.bbox.y2,
        )
        return list(self._embeddings.get(key, [1.0, 0.0, 0.0]))


class FaceTrackingServiceTests(unittest.TestCase):
    def test_tracking_preserves_persistent_face_and_reduces_redundant_keyframes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            service = FaceTrackingService(
                config=self._config(keyframe_interval_seconds=10.0),
                enhancement_service=EnhancementService(self._config().enhancement),
                quality_service=FaceQualityService(),
            )
            frames = self._frames(Path(temp_dir) / "sample.mp4", count=4)
            detections = {
                0: [self._detection(10, 10, 34, 34)],
                1: [self._detection(11, 10, 35, 34)],
                2: [self._detection(12, 10, 36, 34)],
                3: [self._detection(13, 10, 37, 34)],
            }
            analyzer = _ProgrammedAnalyzer(
                detections_by_frame=detections,
                embeddings={
                    (10.0, 10.0, 34.0, 34.0): [1.0, 0.0, 0.0],
                    (11.0, 10.0, 35.0, 34.0): [1.0, 0.0, 0.0],
                    (12.0, 10.0, 36.0, 34.0): [1.0, 0.0, 0.0],
                    (13.0, 10.0, 37.0, 34.0): [1.0, 0.0, 0.0],
                },
            )
            artifact_store = ArtifactStore(Path(temp_dir) / "run")

            result = service.process_media(
                source_path=Path(temp_dir) / "sample.mp4",
                sha512="hash",
                media_type=MediaType.VIDEO,
                frames=frames,
                analyzer=analyzer,
                artifact_store=artifact_store,
            )

            self.assertEqual(4, len(result.occurrences))
            self.assertEqual(1, len(result.tracks))
            self.assertEqual(1, len(result.keyframes))
            self.assertEqual(4, len(result.tracks[0].occurrence_ids))
            self.assertEqual(1, len(result.tracks[0].keyframe_ids))

    def test_tracking_supports_multiple_simultaneous_tracks(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            service = FaceTrackingService(
                config=self._config(),
                enhancement_service=EnhancementService(self._config().enhancement),
                quality_service=FaceQualityService(),
            )
            frames = self._frames(Path(temp_dir) / "pair.mp4", count=2)
            detections = {
                0: [self._detection(10, 10, 34, 34), self._detection(60, 10, 84, 34)],
                1: [self._detection(11, 10, 35, 34), self._detection(61, 10, 85, 34)],
            }
            analyzer = _ProgrammedAnalyzer(
                detections_by_frame=detections,
                embeddings={
                    (10.0, 10.0, 34.0, 34.0): [1.0, 0.0, 0.0],
                    (11.0, 10.0, 35.0, 34.0): [1.0, 0.0, 0.0],
                    (60.0, 10.0, 84.0, 34.0): [0.0, 1.0, 0.0],
                    (61.0, 10.0, 85.0, 34.0): [0.0, 1.0, 0.0],
                },
            )
            artifact_store = ArtifactStore(Path(temp_dir) / "run")

            result = service.process_media(
                source_path=Path(temp_dir) / "pair.mp4",
                sha512="hash",
                media_type=MediaType.VIDEO,
                frames=frames,
                analyzer=analyzer,
                artifact_store=artifact_store,
            )

            self.assertEqual(2, len(result.tracks))
            self.assertTrue(all(len(track.occurrence_ids) == 2 for track in result.tracks))
            self.assertEqual(result.tracks[0].start_time, result.tracks[1].start_time)
            self.assertEqual(result.tracks[0].end_time, result.tracks[1].end_time)

    def test_tracking_is_reproducible_for_same_input(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = self._config()
            frames = self._frames(Path(temp_dir) / "stable.mp4", count=3)
            detections = {
                0: [self._detection(10, 10, 34, 34)],
                1: [self._detection(12, 10, 36, 34)],
                2: [self._detection(14, 10, 38, 34)],
            }
            analyzer = _ProgrammedAnalyzer(
                detections_by_frame=detections,
                embeddings={
                    (10.0, 10.0, 34.0, 34.0): [1.0, 0.0, 0.0],
                    (12.0, 10.0, 36.0, 34.0): [1.0, 0.0, 0.0],
                    (14.0, 10.0, 38.0, 34.0): [1.0, 0.0, 0.0],
                },
            )

            first = FaceTrackingService(
                config=config,
                enhancement_service=EnhancementService(config.enhancement),
                quality_service=FaceQualityService(),
            ).process_media(
                source_path=Path(temp_dir) / "stable.mp4",
                sha512="hash",
                media_type=MediaType.VIDEO,
                frames=self._frames(Path(temp_dir) / "stable.mp4", count=3),
                analyzer=analyzer,
                artifact_store=ArtifactStore(Path(temp_dir) / "run1"),
            )
            second = FaceTrackingService(
                config=config,
                enhancement_service=EnhancementService(config.enhancement),
                quality_service=FaceQualityService(),
            ).process_media(
                source_path=Path(temp_dir) / "stable.mp4",
                sha512="hash",
                media_type=MediaType.VIDEO,
                frames=self._frames(Path(temp_dir) / "stable.mp4", count=3),
                analyzer=analyzer,
                artifact_store=ArtifactStore(Path(temp_dir) / "run2"),
            )

            self.assertEqual([track.track_id for track in first.tracks], [track.track_id for track in second.tracks])
            self.assertEqual([track.occurrence_ids for track in first.tracks], [track.occurrence_ids for track in second.tracks])
            self.assertEqual([keyframe.selection_reasons for keyframe in first.keyframes], [keyframe.selection_reasons for keyframe in second.keyframes])

    def test_tracking_starts_new_track_when_embedding_similarity_is_below_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = self._config()
            frames = self._frames(Path(temp_dir) / "ambiguous.mp4", count=2)
            detections = {
                0: [self._detection(10, 10, 34, 34)],
                1: [self._detection(22, 10, 46, 34)],
            }
            analyzer = _ProgrammedAnalyzer(
                detections_by_frame=detections,
                embeddings={
                    (10.0, 10.0, 34.0, 34.0): [1.0, 0.0, 0.0],
                    (22.0, 10.0, 46.0, 34.0): [0.0, 1.0, 0.0],
                },
            )

            result = FaceTrackingService(
                config=config,
                enhancement_service=EnhancementService(config.enhancement),
                quality_service=FaceQualityService(),
            ).process_media(
                source_path=Path(temp_dir) / "ambiguous.mp4",
                sha512="hash",
                media_type=MediaType.VIDEO,
                frames=frames,
                analyzer=analyzer,
                artifact_store=ArtifactStore(Path(temp_dir) / "run"),
            )

            self.assertEqual(2, len(result.tracks))

    def test_tracking_log_exposes_sample_count_and_real_frame_index(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = self._config()
            source_path = Path(temp_dir) / "sampled.mp4"
            frames = [
                SampledFrame(
                    source_path=source_path,
                    image_name="frame_000060",
                    frame_index=60,
                    timestamp_seconds=2.0,
                    bgr_pixels=np.zeros((100, 100, 3), dtype=np.uint8),
                    original_bgr_pixels=np.zeros((100, 100, 3), dtype=np.uint8),
                ),
                SampledFrame(
                    source_path=source_path,
                    image_name="frame_000120",
                    frame_index=120,
                    timestamp_seconds=4.0,
                    bgr_pixels=np.zeros((100, 100, 3), dtype=np.uint8),
                    original_bgr_pixels=np.zeros((100, 100, 3), dtype=np.uint8),
                ),
            ]
            analyzer = _ProgrammedAnalyzer(
                detections_by_frame={60: [], 120: []},
                embeddings={},
            )
            logs: list[str] = []

            FaceTrackingService(
                config=config,
                enhancement_service=EnhancementService(config.enhancement),
                quality_service=FaceQualityService(),
            ).process_media(
                source_path=source_path,
                sha512="hash",
                media_type=MediaType.VIDEO,
                frames=frames,
                analyzer=analyzer,
                artifact_store=ArtifactStore(Path(temp_dir) / "run"),
                text_callback=logs.append,
            )

            self.assertTrue(any("amostra=1" in line for line in logs))
            self.assertTrue(any("quadro_real=000060" in line for line in logs))
            self.assertTrue(any("instante=00:00:02" in line for line in logs))

    def _config(self, keyframe_interval_seconds: float = 3.0) -> AppConfig:
        return AppConfig(
            app=AppSettings(
                name="Inventario Faces",
                output_directory_name="inventario_faces_output",
                report_title="Relatorio Teste",
                organization="Lab Teste",
            ),
            media=MediaSettings(
                image_extensions=(".jpg", ".png"),
                video_extensions=(".mp4", ".avi"),
            ),
            video=VideoSettings(
                sampling_interval_seconds=1.0,
                max_frames_per_video=10,
                keyframe_interval_seconds=keyframe_interval_seconds,
                significant_change_threshold=0.90,
            ),
            face_model=FaceModelSettings(
                backend="fake",
                model_name="fake",
                det_size=(640, 640),
                minimum_face_quality=0.6,
                minimum_face_size_pixels=20,
            ),
            clustering=ClusteringSettings(
                assignment_similarity=0.5,
                candidate_similarity=0.4,
                min_cluster_size=1,
            ),
            reporting=ReportingSettings(
                compile_pdf=False,
            ),
            forensics=ForensicsSettings(
                chain_of_custody_note="Teste"
            ),
            tracking=TrackingSettings(
                iou_threshold=0.10,
                spatial_distance_threshold=0.25,
                embedding_similarity_threshold=0.45,
                minimum_total_match_score=0.20,
                max_missed_detections=1,
                confidence_margin=0.01,
            ),
            enhancement=EnhancementSettings(enable_preprocessing=False),
            search=SearchSettings(enabled=False),
        )

    def _frames(self, source_path: Path, count: int) -> list[SampledFrame]:
        frames: list[SampledFrame] = []
        for index in range(count):
            canvas = np.zeros((100, 100, 3), dtype=np.uint8)
            canvas[10:34, 10 + index : 34 + index] = (255, 255, 255)
            frames.append(
                SampledFrame(
                    source_path=source_path,
                    image_name=f"frame_{index:06d}",
                    frame_index=index,
                    timestamp_seconds=float(index),
                    bgr_pixels=canvas,
                    original_bgr_pixels=canvas,
                )
            )
        return frames

    def _detection(self, x1: int, y1: int, x2: int, y2: int) -> DetectedFace:
        crop = np.zeros((24, 24, 3), dtype=np.uint8)
        crop[:] = (220, 220, 220)
        return DetectedFace(
            bbox=BoundingBox(float(x1), float(y1), float(x2), float(y2)),
            detection_score=0.98,
            crop_bgr=crop,
            landmarks=(
                (x1 + 6.0, y1 + 9.0),
                (x1 + 18.0, y1 + 9.0),
                (x1 + 12.0, y1 + 14.0),
                (x1 + 8.0, y1 + 19.0),
                (x1 + 16.0, y1 + 19.0),
            ),
        )


if __name__ == "__main__":
    unittest.main()
