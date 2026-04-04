from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from pathlib import Path
from typing import Any
import warnings

import onnxruntime as ort

from inventario_faces.domain.config import FaceModelSettings
from inventario_faces.domain.entities import BoundingBox, DetectedFace, SampledFrame
from inventario_faces.utils.math_utils import l2_normalize


class FaceAnalyzerInitializationError(RuntimeError):
    """Erro ao inicializar o backend facial."""


def _cleanup_skimage_desktop_ini() -> None:
    """Remove arquivos desktop.ini indevidos do diretorio de plugins do skimage.

    Em alguns ambientes Windows, o Explorer grava ``desktop.ini`` dentro de
    ``skimage/io/_plugins``. O scikit-image tenta interpretar qualquer ``*.ini``
    nesse diretorio como definicao de plugin e falha no parse.
    """
    spec = importlib.util.find_spec("skimage")
    if spec is None or spec.origin is None:
        return

    package_directory = Path(spec.origin).resolve().parent
    plugin_directory = package_directory / "io" / "_plugins"
    for candidate in plugin_directory.glob("desktop.ini"):
        _remove_file_quietly(candidate)
    for candidate in plugin_directory.glob("Desktop.ini"):
        _remove_file_quietly(candidate)


def _remove_file_quietly(path: Path) -> None:
    if not path.exists():
        return
    try:
        path.unlink()
    except OSError:
        pass


def _is_skimage_plugin_parse_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return (
        "skimage" in message
        and "desktop.ini" in message
        and "parsing errors" in message
    )


def _patch_insightface_face_align() -> None:
    try:
        import numpy as np
        from insightface.utils import face_align
        from skimage import transform as trans
    except Exception:
        return

    if getattr(face_align, "_inventario_faces_from_estimate_patch", False):
        return
    if not hasattr(trans.SimilarityTransform, "from_estimate"):
        return

    arcface_dst = np.asarray(face_align.arcface_dst, dtype=np.float32)

    def estimate_norm(lmk, image_size=112, mode="arcface"):
        landmarks = np.asarray(lmk, dtype=np.float32)
        assert landmarks.shape == (5, 2)
        assert image_size % 112 == 0 or image_size % 128 == 0
        if image_size % 112 == 0:
            ratio = float(image_size) / 112.0
            diff_x = 0.0
        else:
            ratio = float(image_size) / 128.0
            diff_x = 8.0 * ratio
        dst = arcface_dst.copy() * ratio
        dst[:, 0] += diff_x
        tform = trans.SimilarityTransform.from_estimate(landmarks, dst)
        return tform.params[0:2, :]

    face_align.estimate_norm = estimate_norm
    face_align._inventario_faces_from_estimate_patch = True


@dataclass
class InsightFaceAnalyzer:
    settings: FaceModelSettings

    def __post_init__(self) -> None:
        providers = list(self.settings.providers) or self._discover_providers()
        self._detector, self._model_dir, self._recognizer = self._initialize_backend(providers)
        self._prepared_det_size: tuple[int, int] | None = None
        self._prepare_detector(self.settings.det_size or (640, 640))
        self._providers = providers

    def _initialize_backend(self, providers: list[str]) -> tuple[Any, Path, Any | None]:
        for attempt in range(2):
            _cleanup_skimage_desktop_ini()
            try:
                from insightface.app import FaceAnalysis
                from insightface.model_zoo import model_zoo
            except ImportError as exc:
                raise FaceAnalyzerInitializationError(
                    "InsightFace nao esta instalado. Use Python 3.11/3.12 e instale as dependencias faciais."
                ) from exc
            except Exception as exc:
                if attempt == 0 and _is_skimage_plugin_parse_error(exc):
                    _cleanup_skimage_desktop_ini()
                    continue
                raise FaceAnalyzerInitializationError(
                    f"Falha ao importar o InsightFace: {exc}"
                ) from exc

            try:
                _patch_insightface_face_align()
                detector = FaceAnalysis(
                    name=self.settings.model_name,
                    providers=providers,
                    allowed_modules=["detection"],
                )
                model_dir = Path(detector.model_dir)
                recognizer = self._load_recognizer(model_zoo, providers, model_dir)
                return detector, model_dir, recognizer
            except Exception as exc:
                if attempt == 0 and _is_skimage_plugin_parse_error(exc):
                    _cleanup_skimage_desktop_ini()
                    continue
                raise

        raise FaceAnalyzerInitializationError(
            "Falha ao inicializar o InsightFace apos saneamento automatico do skimage."
        )

    def _discover_providers(self) -> list[str]:
        available = ort.get_available_providers()
        prioritized = [
            provider
            for provider in ("CUDAExecutionProvider", "DmlExecutionProvider", "CPUExecutionProvider")
            if provider in available
        ]
        return prioritized or ["CPUExecutionProvider"]

    @property
    def providers(self) -> list[str]:
        return list(self._providers)

    def detect(self, frame: SampledFrame) -> list[DetectedFace]:
        self._ensure_detection_size(frame)
        faces = self._detector.get(frame.bgr_pixels)
        detections: list[DetectedFace] = []
        for face in faces:
            detection_score = float(getattr(face, "det_score", 0.0))
            x1, y1, x2, y2 = [float(value) for value in face.bbox]
            crop_source = frame.original_bgr_pixels if frame.original_bgr_pixels is not None else frame.bgr_pixels
            crop = self._crop(crop_source, x1, y1, x2, y2)
            raw_landmarks = getattr(face, "kps", None)
            landmarks = (
                tuple((float(point[0]), float(point[1])) for point in raw_landmarks)
                if raw_landmarks is not None
                else ()
            )

            detections.append(
                DetectedFace(
                    bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
                    detection_score=detection_score,
                    crop_bgr=crop,
                    landmarks=landmarks,
                    enhancement_metadata=frame.enhancement_metadata,
                )
            )
        return detections

    def embed(self, frame: SampledFrame, detection: DetectedFace, reason: str = "keyframe") -> list[float]:
        import numpy as np

        if self._recognizer is None:
            return []

        class _FaceProxy:
            def __init__(self, bbox: list[float], kps: Any | None) -> None:
                self.bbox = bbox
                self.kps = kps
                self.embedding: Any | None = None

        if detection.landmarks:
            landmarks = np.asarray(
                [[float(x), float(y)] for x, y in detection.landmarks],
                dtype=np.float32,
            )
        else:
            landmarks = self._fallback_landmarks(detection.bbox)
        face = _FaceProxy(
            bbox=[detection.bbox.x1, detection.bbox.y1, detection.bbox.x2, detection.bbox.y2],
            kps=landmarks,
        )
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"`estimate` is deprecated since version 0\.26.*",
                category=FutureWarning,
                module=r"insightface\.utils\.face_align",
            )
            self._recognizer.get(frame.bgr_pixels, face)
        raw_embedding = getattr(face, "embedding", None)
        if raw_embedding is None:
            return []
        return l2_normalize(raw_embedding)

    def analyze(self, frame: SampledFrame) -> list[DetectedFace]:
        detections = self.detect(frame)
        enriched: list[DetectedFace] = []
        for detection in detections:
            enriched.append(
                DetectedFace(
                    bbox=detection.bbox,
                    detection_score=detection.detection_score,
                    crop_bgr=detection.crop_bgr,
                    embedding=self.embed(frame, detection, reason="full_analysis"),
                    landmarks=detection.landmarks,
                    quality_metrics=detection.quality_metrics,
                    enhancement_metadata=detection.enhancement_metadata,
                    embedding_source="full_analysis",
                )
            )
        return enriched

    def _prepare_detector(self, det_size: tuple[int, int]) -> None:
        self._detector.prepare(ctx_id=self.settings.ctx_id, det_size=det_size)
        self._prepared_det_size = det_size

    def _ensure_detection_size(self, frame: SampledFrame) -> None:
        configured_size = self.settings.det_size
        if configured_size is not None:
            if self._prepared_det_size != configured_size:
                self._prepare_detector(configured_size)
            return

        height, width = frame.bgr_pixels.shape[:2]
        original_size = (max(32, int(width)), max(32, int(height)))
        if self._prepared_det_size != original_size:
            self._prepare_detector(original_size)

    def _crop(self, image: object, x1: float, y1: float, x2: float, y2: float) -> object:
        import numpy as np

        height, width = image.shape[:2]
        left = max(0, int(x1))
        top = max(0, int(y1))
        right = min(width, int(x2))
        bottom = min(height, int(y2))
        if right <= left or bottom <= top:
            return np.zeros((1, 1, 3), dtype=np.uint8)
        return image[top:bottom, left:right].copy()

    def _load_recognizer(self, model_zoo: Any, providers: list[str], model_dir: Path) -> Any | None:
        candidates = sorted(model_dir.glob("*.onnx"))
        for candidate in candidates:
            model = model_zoo.get_model(str(candidate), providers=providers)
            if getattr(model, "taskname", None) != "recognition":
                continue
            model.prepare(self.settings.ctx_id)
            return model
        return None

    def _fallback_landmarks(self, bbox: BoundingBox) -> Any:
        import numpy as np

        width = bbox.width
        height = bbox.height
        return np.asarray(
            [
                [bbox.x1 + width * 0.30, bbox.y1 + height * 0.38],
                [bbox.x1 + width * 0.70, bbox.y1 + height * 0.38],
                [bbox.x1 + width * 0.50, bbox.y1 + height * 0.56],
                [bbox.x1 + width * 0.36, bbox.y1 + height * 0.78],
                [bbox.x1 + width * 0.64, bbox.y1 + height * 0.78],
            ],
            dtype=np.float32,
        )
