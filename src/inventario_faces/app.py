from __future__ import annotations

import os
import sys
from pathlib import Path

from inventario_faces.domain.config import AppConfig
from inventario_faces.infrastructure.config_loader import (
    default_user_config_path,
    load_default_app_config,
    load_app_config,
    save_app_config,
)
from inventario_faces.infrastructure.face_analyzer_insight import InsightFaceAnalyzer
from inventario_faces.infrastructure.latex_compiler import LatexCompiler
from inventario_faces.infrastructure.media_info_service import MediaInfoService
from inventario_faces.reporting.combined_face_search_report_generator import CombinedFaceSearchReportGenerator
from inventario_faces.reporting.combined_report_generator import CombinedReportGenerator
from inventario_faces.reporting.docx_renderer import DocxReportGenerator
from inventario_faces.reporting.face_search_docx_renderer import FaceSearchDocxReportGenerator
from inventario_faces.reporting.face_search_latex_renderer import FaceSearchLatexReportGenerator
from inventario_faces.reporting.latex_renderer import LatexReportGenerator
from inventario_faces.services.clustering_service import ClusteringService
from inventario_faces.services.hashing_service import HashingService
from inventario_faces.services.inventory_service import InventoryService
from inventario_faces.services.scanner_service import ScannerService
from inventario_faces.services.video_service import VideoService


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


def build_inventory_service(config: AppConfig | None = None) -> InventoryService:
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
        face_analyzer_factory=lambda: InsightFaceAnalyzer(runtime_config.face_model),
        media_info_extractor=MediaInfoService(),
        face_search_report_generator=CombinedFaceSearchReportGenerator(
            latex_generator=FaceSearchLatexReportGenerator(runtime_config, LatexCompiler()),
            docx_generator=FaceSearchDocxReportGenerator(runtime_config),
        ),
    )
