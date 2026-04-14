from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from tests._bootstrap import PROJECT_ROOT  # noqa: F401
from inventario_faces.domain.config import FaceModelSettings
from inventario_faces.domain.entities import BoundingBox, DetectedFace, SampledFrame
from inventario_faces.infrastructure.face_analyzer_insight import (
    InsightFaceAnalyzer,
    _collect_runtime_dll_directories,
    _runtime_search_roots,
)


class _RecognizerStub:
    def get(self, img, face) -> None:
        assert hasattr(face.kps, "shape")
        assert tuple(face.kps.shape) == (5, 2)
        face.embedding = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)


class InsightFaceAdapterTests(unittest.TestCase):
    def test_discover_providers_prioritizes_gpu_before_cpu(self) -> None:
        analyzer = InsightFaceAnalyzer.__new__(InsightFaceAnalyzer)

        providers = analyzer._discover_providers(
            [
                "AzureExecutionProvider",
                "CPUExecutionProvider",
                "DmlExecutionProvider",
                "CUDAExecutionProvider",
            ]
        )

        self.assertEqual(
            ["CUDAExecutionProvider", "DmlExecutionProvider", "CPUExecutionProvider"],
            providers,
        )

    def test_resolve_providers_filters_unavailable_configured_entries(self) -> None:
        analyzer = InsightFaceAnalyzer.__new__(InsightFaceAnalyzer)
        analyzer.settings = FaceModelSettings(
            backend="insightface",
            model_name="buffalo_l",
            det_size=(640, 640),
            providers=("CUDAExecutionProvider", "CPUExecutionProvider"),
        )

        providers = analyzer._resolve_providers(["CPUExecutionProvider"])

        self.assertEqual(["CPUExecutionProvider"], providers)

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

    def test_extract_biometric_landmarks_prefers_dense_landmarks_when_available(self) -> None:
        analyzer = InsightFaceAnalyzer.__new__(InsightFaceAnalyzer)

        class _FaceStub:
            landmark_2d_106 = np.asarray(
                [
                    [10.0, 12.0],
                    [20.0, 12.0],
                    [15.0, 20.0],
                    [12.0, 28.0],
                    [18.0, 28.0],
                    [15.0, 32.0],
                ],
                dtype=np.float32,
            )

        fallback = ((1.0, 1.0), (2.0, 2.0))
        points = analyzer._extract_biometric_landmarks(_FaceStub(), fallback)

        self.assertEqual(6, len(points))
        self.assertEqual((10.0, 12.0), points[0])

    def test_collect_runtime_dll_directories_finds_onnxruntime_and_nvidia_bins(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ort_capi = root / "onnxruntime" / "capi"
            nvrtc_bin = root / "nvidia" / "cuda_nvrtc" / "bin"
            cudnn_bin = root / "nvidia" / "cudnn" / "bin"
            ort_capi.mkdir(parents=True, exist_ok=True)
            nvrtc_bin.mkdir(parents=True, exist_ok=True)
            cudnn_bin.mkdir(parents=True, exist_ok=True)

            directories = _collect_runtime_dll_directories([root])

            self.assertEqual(
                [ort_capi, nvrtc_bin, cudnn_bin],
                directories,
            )

    def test_runtime_search_roots_include_site_package_locations(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            python_executable = temp_root / "Scripts" / "python.exe"
            python_executable.parent.mkdir(parents=True, exist_ok=True)
            python_executable.write_text("", encoding="utf-8")
            prefix_root = temp_root / "venv"
            prefix_root.mkdir(parents=True, exist_ok=True)
            site_packages = temp_root / "venv" / "Lib" / "site-packages"
            site_packages.mkdir(parents=True, exist_ok=True)
            user_site = temp_root / "usersite"
            user_site.mkdir(parents=True, exist_ok=True)

            with (
                mock.patch("inventario_faces.infrastructure.face_analyzer_insight.sys.executable", str(python_executable)),
                mock.patch("inventario_faces.infrastructure.face_analyzer_insight.sys.prefix", str(prefix_root)),
                mock.patch("inventario_faces.infrastructure.face_analyzer_insight.sys.base_prefix", str(prefix_root)),
                mock.patch(
                    "inventario_faces.infrastructure.face_analyzer_insight.site.getsitepackages",
                    return_value=[str(site_packages)],
                ),
                mock.patch(
                    "inventario_faces.infrastructure.face_analyzer_insight.site.getusersitepackages",
                    return_value=str(user_site),
                ),
                mock.patch(
                    "inventario_faces.infrastructure.face_analyzer_insight.sysconfig.get_paths",
                    return_value={"purelib": str(site_packages), "platlib": str(site_packages)},
                ),
            ):
                roots = _runtime_search_roots()

            self.assertIn(site_packages.resolve(), roots)
            self.assertIn(user_site.resolve(), roots)


if __name__ == "__main__":
    unittest.main()
