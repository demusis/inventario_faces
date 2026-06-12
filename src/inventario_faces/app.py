from __future__ import annotations

from dataclasses import dataclass
import os
import sys
from pathlib import Path
from threading import Condition, RLock, Thread
from typing import TYPE_CHECKING, Callable

from inventario_faces.domain.config import AppConfig, FaceModelSettings
from inventario_faces.infrastructure.config_loader import (
    default_user_config_path,
    load_default_app_config,
    load_app_config,
    save_app_config,
)

if TYPE_CHECKING:
    from inventario_faces.domain.protocols import FaceAnalyzer
    from inventario_faces.services.inventory_service import InventoryService


@dataclass
class _SharedFaceAnalyzerEntry:
    analyzer: FaceAnalyzer | None = None
    initializing: bool = False


_SHARED_FACE_ANALYZER_CACHE: dict[FaceModelSettings, _SharedFaceAnalyzerEntry] = {}
_SHARED_FACE_ANALYZER_CONDITION = Condition()


def _runtime_base_directory() -> Path:
    return Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parents[2]))


def _package_resource_directory() -> Path:
    frozen_base = getattr(sys, "_MEIPASS", None)
    if frozen_base is not None:
        return Path(frozen_base) / "inventario_faces"
    return Path(__file__).resolve().parent


def resolve_config_path() -> Path | None:
    raw_value = os.getenv("INVENTARIO_FACES_CONFIG")
    return Path(raw_value).expanduser().resolve() if raw_value else None


def resolve_persistent_config_path() -> Path:
    return resolve_config_path() or default_user_config_path()


def load_runtime_config() -> AppConfig:
    explicit_config_path = resolve_config_path()
    if explicit_config_path is not None:
        return load_app_config(explicit_config_path)
    persistent_path = resolve_persistent_config_path()
    return load_app_config(persistent_path if persistent_path.exists() else None)


def load_default_runtime_config() -> AppConfig:
    return load_default_app_config()


def persist_runtime_config(config: AppConfig) -> Path:
    return save_app_config(config, resolve_persistent_config_path())


def resolve_app_icon_path() -> Path | None:
    resource_directory = _package_resource_directory() / "assets"
    for file_name in ("app_icon.ico", "app_icon.png"):
        candidate = resource_directory / file_name
        if candidate.exists():
            return candidate
    return None


def configure_runtime_environment() -> None:
    runtime_base = _runtime_base_directory()
    root_candidates = [
        Path(sys.executable).resolve().parent,
        runtime_base,
    ]
    for candidate in root_candidates:
        if (candidate / "models").exists():
            os.environ.setdefault("INSIGHTFACE_HOME", str(candidate))
            break


def _build_face_analyzer(settings: FaceModelSettings) -> FaceAnalyzer:
    backend = (settings.backend or "insightface").strip().lower()

    if backend == "onnx":
        from inventario_faces.infrastructure.face_analyzer_onnx import OnnxFaceAnalyzer

        return OnnxFaceAnalyzer(settings)

    from inventario_faces.infrastructure.face_analyzer_insight import (
        FaceAnalyzerInitializationError,
        InsightFaceAnalyzer,
    )

    try:
        return InsightFaceAnalyzer(settings)
    except Exception as exc:
        # O pacote insightface pode estar ausente ou quebrado (ImportError,
        # cadeia skimage, incompatibilidade de versao dentro do FaceAnalysis);
        # os modelos ONNX continuam utilizaveis diretamente.
        from inventario_faces.infrastructure.face_analyzer_onnx import OnnxFaceAnalyzer

        try:
            return OnnxFaceAnalyzer(settings)
        except Exception as fallback_exc:
            raise FaceAnalyzerInitializationError(
                (
                    f"Backend insightface indisponivel ({exc}) e o fallback ONNX "
                    f"direto tambem falhou ({fallback_exc})."
                )
            ) from exc


class _SynchronizedFaceAnalyzer:
    def __init__(self, analyzer: FaceAnalyzer) -> None:
        self._analyzer = analyzer
        self._lock = RLock()

    def detect(self, frame) -> list:
        with self._lock:
            return self._analyzer.detect(frame)

    def embed(self, frame, detection, reason: str = "keyframe") -> list[float]:
        with self._lock:
            return self._analyzer.embed(frame, detection, reason=reason)

    def analyze(self, frame) -> list:
        with self._lock:
            return self._analyzer.analyze(frame)

    def __getattr__(self, name: str):
        return getattr(self._analyzer, name)


def clear_shared_face_analyzer_cache() -> None:
    with _SHARED_FACE_ANALYZER_CONDITION:
        _SHARED_FACE_ANALYZER_CACHE.clear()


def get_shared_face_analyzer(
    settings: FaceModelSettings,
    *,
    builder: Callable[[FaceModelSettings], FaceAnalyzer] | None = None,
) -> FaceAnalyzer:
    analyzer_builder = builder or _build_face_analyzer

    with _SHARED_FACE_ANALYZER_CONDITION:
        entry = _SHARED_FACE_ANALYZER_CACHE.setdefault(settings, _SharedFaceAnalyzerEntry())
        while entry.initializing and entry.analyzer is None:
            _SHARED_FACE_ANALYZER_CONDITION.wait()
        if entry.analyzer is not None:
            return entry.analyzer
        entry.initializing = True

    try:
        analyzer = _SynchronizedFaceAnalyzer(analyzer_builder(settings))
    except Exception:
        with _SHARED_FACE_ANALYZER_CONDITION:
            current = _SHARED_FACE_ANALYZER_CACHE.get(settings)
            if current is entry:
                _SHARED_FACE_ANALYZER_CACHE.pop(settings, None)
            entry.initializing = False
            _SHARED_FACE_ANALYZER_CONDITION.notify_all()
        raise

    with _SHARED_FACE_ANALYZER_CONDITION:
        entry.analyzer = analyzer
        entry.initializing = False
        _SHARED_FACE_ANALYZER_CONDITION.notify_all()
        return analyzer


def preload_shared_face_analyzer_async(settings: FaceModelSettings) -> bool:
    with _SHARED_FACE_ANALYZER_CONDITION:
        entry = _SHARED_FACE_ANALYZER_CACHE.setdefault(settings, _SharedFaceAnalyzerEntry())
        if entry.analyzer is not None or entry.initializing:
            return False

    Thread(
        target=lambda: get_shared_face_analyzer(settings),
        name=f"inventario_faces_model_preload_{settings.model_name}",
        daemon=True,
    ).start()
    return True


def build_inventory_service(config: AppConfig | None = None) -> InventoryService:
    from inventario_faces.infrastructure.latex_compiler import LatexCompiler
    from inventario_faces.infrastructure.media_info_service import MediaInfoService
    from inventario_faces.reporting.combined_face_search_report_generator import (
        CombinedFaceSearchReportGenerator,
    )
    from inventario_faces.reporting.combined_report_generator import CombinedReportGenerator
    from inventario_faces.reporting.docx_renderer import DocxReportGenerator
    from inventario_faces.reporting.face_search_docx_renderer import FaceSearchDocxReportGenerator
    from inventario_faces.reporting.face_search_latex_renderer import (
        FaceSearchLatexReportGenerator,
    )
    from inventario_faces.reporting.latex_renderer import LatexReportGenerator
    from inventario_faces.services.clustering_service import ClusteringService
    from inventario_faces.services.hashing_service import HashingService
    from inventario_faces.services.inventory_service import InventoryService
    from inventario_faces.services.scanner_service import ScannerService
    from inventario_faces.services.video_service import VideoService

    configure_runtime_environment()
    runtime_config = config or load_runtime_config()
    return InventoryService(
        config=runtime_config,
        scanner_service=ScannerService(runtime_config.media),
        hashing_service=HashingService(),
        media_service=VideoService(runtime_config.video),
        clustering_service=ClusteringService(runtime_config.clustering),
        report_generator=CombinedReportGenerator(
            latex_generator=LatexReportGenerator(runtime_config, LatexCompiler()),
            docx_generator=DocxReportGenerator(runtime_config),
        ),
        face_analyzer_factory=lambda: get_shared_face_analyzer(runtime_config.face_model),
        media_info_extractor=MediaInfoService(),
        face_search_report_generator=CombinedFaceSearchReportGenerator(
            latex_generator=FaceSearchLatexReportGenerator(runtime_config, LatexCompiler()),
            docx_generator=FaceSearchDocxReportGenerator(runtime_config),
        ),
    )
