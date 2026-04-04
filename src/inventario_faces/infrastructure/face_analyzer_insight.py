from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from pathlib import Path

import onnxruntime as ort

from inventario_faces.domain.config import FaceModelSettings
from inventario_faces.domain.entities import BoundingBox, DetectedFace, SampledFrame
from inventario_faces.utils.math_utils import l2_normalize


class FaceAnalyzerInitializationError(RuntimeError):
    """Erro ao inicializar o backend facial."""


def _cleanup_skimage_desktop_ini() -> None:
    """Remove arquivos desktop.ini indevidos do diretório de plugins do skimage.

    Em instalações Windows, alguns ambientes acabam gravando esse arquivo dentro
    de ``skimage/io/_plugins``. O scikit-image interpreta qualquer ``*.ini`` ali
    como configuração de plugin e falha no parse.
    """
    spec = importlib.util.find_spec("skimage")
    if spec is None or spec.origin is None:
        return

    package_directory = Path(spec.origin).resolve().parent
    plugin_directory = package_directory / "io" / "_plugins"
    desktop_ini = plugin_directory / "desktop.ini"
    if desktop_ini.exists():
        try:
            desktop_ini.unlink()
        except OSError:
            pass


@dataclass
class InsightFaceAnalyzer:
    settings: FaceModelSettings

    def __post_init__(self) -> None:
        _cleanup_skimage_desktop_ini()
        try:
            from insightface.app import FaceAnalysis
        except ImportError as exc:
            raise FaceAnalyzerInitializationError(
                "InsightFace nao esta instalado. Use Python 3.11/3.12 e instale as dependencias faciais."
            ) from exc

        providers = list(self.settings.providers) or self._discover_providers()
        self._analyzer = FaceAnalysis(name=self.settings.model_name, providers=providers)
        self._prepared_det_size: tuple[int, int] | None = None
        self._prepare_analyzer(self.settings.det_size or (640, 640))
        self._providers = providers

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

    def analyze(self, frame: SampledFrame) -> list[DetectedFace]:
        self._ensure_detection_size(frame)
        faces = self._analyzer.get(frame.bgr_pixels)
        detections: list[DetectedFace] = []
        for face in faces:
            detection_score = float(getattr(face, "det_score", 0.0))
            x1, y1, x2, y2 = [float(value) for value in face.bbox]
            crop = self._crop(frame.bgr_pixels, x1, y1, x2, y2)
            embedding = getattr(face, "normed_embedding", None)
            if embedding is None:
                embedding = l2_normalize(getattr(face, "embedding", []))
            else:
                embedding = l2_normalize(embedding.tolist())

            detections.append(
                DetectedFace(
                    bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
                    detection_score=detection_score,
                    embedding=embedding,
                    crop_bgr=crop,
                )
            )
        return detections

    def _prepare_analyzer(self, det_size: tuple[int, int]) -> None:
        self._analyzer.prepare(ctx_id=self.settings.ctx_id, det_size=det_size)
        self._prepared_det_size = det_size

    def _ensure_detection_size(self, frame: SampledFrame) -> None:
        configured_size = self.settings.det_size
        if configured_size is not None:
            if self._prepared_det_size != configured_size:
                self._prepare_analyzer(configured_size)
            return

        height, width = frame.bgr_pixels.shape[:2]
        original_size = (max(32, int(width)), max(32, int(height)))
        if self._prepared_det_size != original_size:
            self._prepare_analyzer(original_size)

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
