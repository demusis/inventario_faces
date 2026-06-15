from __future__ import annotations

import logging
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from tests._bootstrap import PROJECT_ROOT  # noqa: F401
from inventario_faces.domain.entities import MediaType, SampledFrame
from inventario_faces.infrastructure.config_loader import load_app_config
from inventario_faces.infrastructure.logging_setup import StructuredEventLogger
from inventario_faces.services.clustering_service import ClusteringService
from inventario_faces.services.face_set_comparison_service import FaceSetComparisonService
from inventario_faces.services.hashing_service import HashingService
from inventario_faces.services.lr_calibration import LikelihoodRatioCalibrator
from inventario_faces.services.scanner_service import ScannerService


def _sampled_frame(path: Path) -> SampledFrame:
    pixels = np.zeros((4, 4, 3), dtype=np.uint8)
    return SampledFrame(
        source_path=path,
        image_name=path.stem,
        frame_index=None,
        timestamp_seconds=None,
        bgr_pixels=pixels,
        original_bgr_pixels=pixels,
    )


class _RecordingMediaService:
    """Substitui o VideoService registrando qual rota de decodificacao foi usada."""

    def __init__(self) -> None:
        self.load_image_calls = 0
        self.sample_video_calls = 0

    def load_image(self, path: Path) -> SampledFrame:
        self.load_image_calls += 1
        return _sampled_frame(path)

    def sample_video(self, path: Path, metadata_callback=None) -> list[SampledFrame]:
        self.sample_video_calls += 1
        return [_sampled_frame(path)]


class _EmptyTrackingService:
    """process_media que nao seleciona faces, isolando o teste do backend facial."""

    def __init__(self) -> None:
        self.media_types: list[MediaType] = []

    def process_media(self, *, media_type, **_kwargs):
        self.media_types.append(media_type)
        return SimpleNamespace(
            occurrences=[],
            tracks=[],
            keyframes=[],
            raw_detection_count=0,
            selected_detection_count=0,
        )


class FaceSetComparisonVideoRoutingTests(unittest.TestCase):
    def setUp(self) -> None:
        self._config = load_app_config()
        self._media = _RecordingMediaService()
        self._tracking = _EmptyTrackingService()
        self._service = FaceSetComparisonService(
            config=self._config,
            scanner_service=ScannerService(self._config.media),
            hashing_service=HashingService(),
            media_service=self._media,
            tracking_service=self._tracking,
            face_analyzer_factory=lambda: None,
            lr_calibrator=LikelihoodRatioCalibrator(self._config),
            clustering_service=ClusteringService(self._config.clustering),
        )
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self._run_directory = Path(self._tmp.name)

    def _process(self, file_name: str):
        media_path = self._run_directory / file_name
        media_path.write_bytes(b"\x00")
        logs_directory = self._run_directory / "logs"
        logs_directory.mkdir(exist_ok=True)
        return self._service._process_comparison_input(
            set_label="A",
            image_path=media_path,
            index=1,
            total_images=1,
            analyzer=None,
            run_directory=self._run_directory,
            export_directory=self._run_directory / "export",
            entry_sequence=[0],
            event_logger=StructuredEventLogger(logs_directory / "events.jsonl"),
            text_logger=logging.getLogger("test_comparison_video"),
            log_callback=None,
        )

    def test_image_input_uses_load_image(self) -> None:
        self._process("foto.jpg")
        self.assertEqual(self._media.load_image_calls, 1)
        self.assertEqual(self._media.sample_video_calls, 0)
        self.assertEqual(self._tracking.media_types, [MediaType.IMAGE])

    def test_video_input_uses_sample_video(self) -> None:
        self._process("clipe.mp4")
        self.assertEqual(self._media.sample_video_calls, 1)
        self.assertEqual(self._media.load_image_calls, 0)
        self.assertEqual(self._tracking.media_types, [MediaType.VIDEO])

    def test_unsupported_media_is_rejected_in_normalization(self) -> None:
        text_path = self._run_directory / "nota.txt"
        text_path.write_bytes(b"x")
        with self.assertRaises(ValueError):
            self._service._normalize_face_set_paths([text_path], "Padrão")

    def test_video_path_is_accepted_in_normalization(self) -> None:
        video_path = self._run_directory / "clipe.mp4"
        video_path.write_bytes(b"\x00")
        normalized = self._service._normalize_face_set_paths([video_path], "Padrão")
        self.assertEqual(normalized, [video_path.resolve()])

    def test_no_comparison_override_reuses_global_media_service(self) -> None:
        self.assertIs(self._service._comparison_media_service, self._media)

    def test_comparison_frame_cap_builds_dedicated_media_service(self) -> None:
        from dataclasses import replace

        from inventario_faces.domain.config import ComparisonSettings
        from inventario_faces.services.video_service import VideoService

        config = replace(
            self._config,
            comparison=ComparisonSettings(max_frames_per_video=5, sampling_interval_seconds=2.0),
        )
        service = FaceSetComparisonService(
            config=config,
            scanner_service=ScannerService(config.media),
            hashing_service=HashingService(),
            media_service=self._media,
            tracking_service=self._tracking,
            face_analyzer_factory=lambda: None,
            lr_calibrator=LikelihoodRatioCalibrator(config),
            clustering_service=ClusteringService(config.clustering),
        )
        self.assertIsNot(service._comparison_media_service, self._media)
        self.assertIsInstance(service._comparison_media_service, VideoService)
        self.assertEqual(service._comparison_media_service._settings.max_frames_per_video, 5)
        self.assertEqual(service._comparison_media_service._settings.sampling_interval_seconds, 2.0)


if __name__ == "__main__":
    unittest.main()
