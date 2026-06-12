from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from tests._bootstrap import PROJECT_ROOT  # noqa: F401
from inventario_faces.domain.config import FaceModelSettings
from inventario_faces.domain.entities import BoundingBox, DetectedFace, SampledFrame
from inventario_faces.infrastructure.face_analyzer_insight import (
    FaceAnalyzerInitializationError,
    _has_local_model_bundle,
    _resolve_model_directory,
)
from inventario_faces.infrastructure.face_analyzer_onnx import (
    ARCFACE_DESTINATION_LANDMARKS,
    OnnxFaceAnalyzer,
    _distance2bbox,
    _distance2kps,
    _nms,
    estimate_similarity_transform,
    select_model_files,
)


def _apply_transform(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    homogeneous = np.hstack([points, np.ones((points.shape[0], 1))])
    return homogeneous @ matrix.T


class SimilarityTransformTests(unittest.TestCase):
    def test_recovers_known_rotation_scale_translation(self) -> None:
        source = np.asarray(
            [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0], [5.0, 5.0]],
            dtype=np.float64,
        )
        angle = math.radians(30.0)
        scale = 1.7
        rotation = np.asarray(
            [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]]
        )
        translation = np.asarray([12.0, -4.0])
        destination = scale * (source @ rotation.T) + translation

        matrix = estimate_similarity_transform(source, destination)
        recovered = _apply_transform(matrix, source)

        np.testing.assert_allclose(recovered, destination, atol=1e-9)

    def test_arcface_template_alignment_is_exact_for_template_points(self) -> None:
        matrix = estimate_similarity_transform(
            ARCFACE_DESTINATION_LANDMARKS, ARCFACE_DESTINATION_LANDMARKS
        )
        recovered = _apply_transform(matrix, ARCFACE_DESTINATION_LANDMARKS.astype(np.float64))
        np.testing.assert_allclose(recovered, ARCFACE_DESTINATION_LANDMARKS, atol=1e-5)


class ScrfdDecodeTests(unittest.TestCase):
    def test_distance2bbox_expands_around_anchor(self) -> None:
        points = np.asarray([[100.0, 200.0]])
        distance = np.asarray([[10.0, 20.0, 30.0, 40.0]])
        bbox = _distance2bbox(points, distance)
        np.testing.assert_allclose(bbox, [[90.0, 180.0, 130.0, 240.0]])

    def test_distance2kps_offsets_from_anchor(self) -> None:
        points = np.asarray([[100.0, 200.0]])
        distance = np.asarray([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]])
        kps = _distance2kps(points, distance)
        np.testing.assert_allclose(
            kps,
            [[101.0, 202.0, 103.0, 204.0, 105.0, 206.0, 107.0, 208.0, 109.0, 210.0]],
        )

    def test_nms_suppresses_heavily_overlapping_boxes(self) -> None:
        detections = np.asarray(
            [
                [0.0, 0.0, 100.0, 100.0, 0.9],
                [5.0, 5.0, 105.0, 105.0, 0.8],
                [200.0, 200.0, 300.0, 300.0, 0.7],
            ],
            dtype=np.float64,
        )
        keep = _nms(detections, iou_threshold=0.4)
        self.assertEqual([0, 2], keep)


class ModelSelectionTests(unittest.TestCase):
    def test_select_model_files_identifies_bundle_components(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for name in ("det_10g.onnx", "w600k_r50.onnx", "2d106det.onnx", "genderage.onnx"):
                (root / name).write_bytes(b"0")

            detector, recognizer, landmark = select_model_files(root)

            self.assertEqual("det_10g.onnx", detector.name)
            self.assertEqual("w600k_r50.onnx", recognizer.name)
            self.assertEqual("2d106det.onnx", landmark.name)

    def test_select_model_files_fails_without_recognizer(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "det_10g.onnx").write_bytes(b"0")

            with self.assertRaises(FaceAnalyzerInitializationError):
                select_model_files(root)


class _RecognizerSessionStub:
    def __init__(self) -> None:
        self.last_blob = None

    def run(self, output_names, feeds):
        blob = next(iter(feeds.values()))
        self.last_blob = blob
        assert blob.shape == (1, 3, 112, 112)
        return [np.asarray([[3.0, 4.0, 0.0]], dtype=np.float32)]


class EmbeddingPathTests(unittest.TestCase):
    def test_embed_aligns_and_normalizes(self) -> None:
        analyzer = OnnxFaceAnalyzer.__new__(OnnxFaceAnalyzer)
        analyzer.settings = FaceModelSettings(
            backend="onnx", model_name="buffalo_l", det_size=(640, 640)
        )
        stub = _RecognizerSessionStub()
        analyzer._recognizer_session = stub
        analyzer._recognizer_input_name = "input"

        frame = SampledFrame(
            source_path=Path("sample.jpg"),
            image_name="sample",
            frame_index=None,
            timestamp_seconds=None,
            bgr_pixels=np.zeros((200, 200, 3), dtype=np.uint8),
            original_bgr_pixels=np.zeros((200, 200, 3), dtype=np.uint8),
        )
        detection = DetectedFace(
            bbox=BoundingBox(40.0, 40.0, 160.0, 160.0),
            detection_score=0.9,
            crop_bgr=np.zeros((120, 120, 3), dtype=np.uint8),
            landmarks=((70.0, 80.0), (130.0, 80.0), (100.0, 110.0), (78.0, 135.0), (122.0, 135.0)),
        )

        embedding = analyzer.embed(frame, detection)

        self.assertEqual(3, len(embedding))
        self.assertAlmostEqual(1.0, sum(value ** 2 for value in embedding) ** 0.5, places=5)
        self.assertAlmostEqual(0.6, embedding[0], places=5)
        self.assertAlmostEqual(0.8, embedding[1], places=5)


class BackendFallbackTests(unittest.TestCase):
    def test_builder_falls_back_to_onnx_when_insightface_fails(self) -> None:
        from inventario_faces import app as app_module

        settings = FaceModelSettings(backend="insightface", model_name="buffalo_l", det_size=(640, 640))
        sentinel = object()

        with (
            mock.patch(
                "inventario_faces.infrastructure.face_analyzer_insight.InsightFaceAnalyzer",
                side_effect=FaceAnalyzerInitializationError("sem insightface"),
            ),
            mock.patch(
                "inventario_faces.infrastructure.face_analyzer_onnx.OnnxFaceAnalyzer",
                return_value=sentinel,
            ),
        ):
            analyzer = app_module._build_face_analyzer(settings)

        self.assertIs(sentinel, analyzer)

    def test_builder_uses_onnx_backend_when_configured(self) -> None:
        from inventario_faces import app as app_module

        settings = FaceModelSettings(backend="onnx", model_name="buffalo_l", det_size=(640, 640))
        sentinel = object()

        with mock.patch(
            "inventario_faces.infrastructure.face_analyzer_onnx.OnnxFaceAnalyzer",
            return_value=sentinel,
        ) as factory:
            analyzer = app_module._build_face_analyzer(settings)

        self.assertIs(sentinel, analyzer)
        factory.assert_called_once_with(settings)

    def test_builder_raises_when_both_backends_fail(self) -> None:
        from inventario_faces import app as app_module

        settings = FaceModelSettings(backend="insightface", model_name="buffalo_l", det_size=(640, 640))

        with (
            mock.patch(
                "inventario_faces.infrastructure.face_analyzer_insight.InsightFaceAnalyzer",
                side_effect=FaceAnalyzerInitializationError("sem insightface"),
            ),
            mock.patch(
                "inventario_faces.infrastructure.face_analyzer_onnx.OnnxFaceAnalyzer",
                side_effect=RuntimeError("sem modelos"),
            ),
        ):
            with self.assertRaises(FaceAnalyzerInitializationError) as context:
                app_module._build_face_analyzer(settings)

        self.assertIn("fallback ONNX", str(context.exception))


@unittest.skipUnless(
    _has_local_model_bundle(_resolve_model_directory("buffalo_l")),
    "modelos buffalo_l indisponiveis localmente",
)
class OnnxAnalyzerIntegrationTests(unittest.TestCase):
    def test_full_pipeline_runs_on_synthetic_image(self) -> None:
        settings = FaceModelSettings(
            backend="onnx",
            model_name="buffalo_l",
            det_size=(320, 320),
            providers=("CPUExecutionProvider",),
        )
        analyzer = OnnxFaceAnalyzer(settings)

        frame = SampledFrame(
            source_path=Path("sintetico.jpg"),
            image_name="sintetico",
            frame_index=None,
            timestamp_seconds=None,
            bgr_pixels=np.full((240, 320, 3), 96, dtype=np.uint8),
            original_bgr_pixels=np.full((240, 320, 3), 96, dtype=np.uint8),
        )

        detections = analyzer.detect(frame)
        self.assertEqual([], detections)

        detection = DetectedFace(
            bbox=BoundingBox(80.0, 60.0, 200.0, 200.0),
            detection_score=0.9,
            crop_bgr=np.zeros((140, 120, 3), dtype=np.uint8),
            landmarks=(),
        )
        embedding = analyzer.embed(frame, detection)
        self.assertEqual(512, len(embedding))
        self.assertAlmostEqual(
            1.0, sum(value ** 2 for value in embedding) ** 0.5, places=4
        )


if __name__ == "__main__":
    unittest.main()
