from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from tests._bootstrap import PROJECT_ROOT  # noqa: F401
from inventario_faces.domain.config import FaceModelSettings
from inventario_faces.domain.entities import BoundingBox, DetectedFace, SampledFrame
from inventario_faces.infrastructure.face_analyzer_insight import InsightFaceAnalyzer


class _RecognizerStub:
    def get(self, img, face) -> None:
        assert hasattr(face.kps, "shape")
        assert tuple(face.kps.shape) == (5, 2)
        face.embedding = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)


class InsightFaceAdapterTests(unittest.TestCase):
    def test_embed_converts_landmarks_to_numpy_matrix(self) -> None:
        analyzer = InsightFaceAnalyzer.__new__(InsightFaceAnalyzer)
        analyzer.settings = FaceModelSettings(backend="insightface", model_name="buffalo_l", det_size=(640, 640))
        analyzer._recognizer = _RecognizerStub()

        frame = SampledFrame(
            source_path=Path("sample.jpg"),
            image_name="sample",
            frame_index=None,
            timestamp_seconds=None,
            bgr_pixels=np.zeros((120, 120, 3), dtype=np.uint8),
            original_bgr_pixels=np.zeros((120, 120, 3), dtype=np.uint8),
        )
        detection = DetectedFace(
            bbox=BoundingBox(10.0, 10.0, 60.0, 60.0),
            detection_score=0.99,
            crop_bgr=np.zeros((50, 50, 3), dtype=np.uint8),
            landmarks=((20.0, 25.0), (45.0, 25.0), (32.0, 36.0), (23.0, 48.0), (41.0, 48.0)),
        )

        embedding = analyzer.embed(frame, detection)

        self.assertEqual([1.0, 0.0, 0.0], embedding)


if __name__ == "__main__":
    unittest.main()
