from __future__ import annotations

from pathlib import Path
from typing import Callable

from PySide6.QtCore import QObject, Signal, Slot

from inventario_faces.infrastructure.logging_setup import summarize_exception
from inventario_faces.services.inventory_service import InventoryService


class _BaseWorker(QObject):
    progress_changed = Signal(int, str)
    log_message = Signal(str)
    completed = Signal(str)
    failed = Signal(str)

    def __init__(self, service: InventoryService, root_directory: Path, work_directory: Path | None = None) -> None:
        super().__init__()
        self._service = service
        self._root_directory = root_directory
        self._work_directory = work_directory

    def _execute(self, action: Callable[[], object]) -> None:
        try:
            result = action()
            report = getattr(result, "report")
            report_path = report.pdf_path or report.docx_path or report.tex_path
            self.completed.emit(str(report_path))
        except Exception as exc:
            self.failed.emit(summarize_exception(exc))

    def _on_progress(self, current: int, total: int, message: str) -> None:
        percent = 0 if total == 0 else int((current / total) * 100)
        self.progress_changed.emit(percent, message)


class InventoryWorker(_BaseWorker):
    def __init__(self, service: InventoryService, root_directory: Path, work_directory: Path | None = None) -> None:
        super().__init__(service, root_directory, work_directory)

    @Slot()
    def run(self) -> None:
        self._execute(
            lambda: self._service.run(
                self._root_directory,
                work_directory=self._work_directory,
                progress_callback=self._on_progress,
                log_callback=self.log_message.emit,
            )
        )


class FaceSearchWorker(_BaseWorker):
    def __init__(
        self,
        service: InventoryService,
        root_directory: Path,
        query_image_path: Path,
        work_directory: Path | None = None,
    ) -> None:
        super().__init__(service, root_directory, work_directory)
        self._query_image_path = query_image_path

    @Slot()
    def run(self) -> None:
        self._execute(
            lambda: self._service.run_face_search(
                self._root_directory,
                self._query_image_path,
                work_directory=self._work_directory,
                progress_callback=self._on_progress,
                log_callback=self.log_message.emit,
            )
        )


class DistributedHealthWorker(_BaseWorker):
    def __init__(self, service: InventoryService, root_directory: Path, work_directory: Path | None = None) -> None:
        super().__init__(service, root_directory, work_directory)

    @Slot()
    def run(self) -> None:
        self._execute(
            lambda: self._service.inspect_distributed_health(
                self._root_directory,
                work_directory=self._work_directory,
                log_callback=self.log_message.emit,
            )
        )


class FaceSetComparisonWorker(QObject):
    progress_changed = Signal(int, str)
    log_message = Signal(str)
    completed = Signal(object)
    failed = Signal(str)

    def __init__(
        self,
        service: InventoryService,
        set_a_paths: list[Path],
        set_b_paths: list[Path],
        work_directory: Path | None = None,
        calibration_root: Path | None = None,
        calibration_model_path: Path | None = None,
    ) -> None:
        super().__init__()
        self._service = service
        self._set_a_paths = list(set_a_paths)
        self._set_b_paths = list(set_b_paths)
        self._work_directory = work_directory
        self._calibration_root = calibration_root
        self._calibration_model_path = calibration_model_path

    def _on_progress(self, current: int, total: int, message: str) -> None:
        percent = 0 if total == 0 else int((current / total) * 100)
        self.progress_changed.emit(percent, message)

    @Slot()
    def run(self) -> None:
        try:
            result = self._service.compare_face_sets(
                self._set_a_paths,
                self._set_b_paths,
                work_directory=self._work_directory,
                calibration_root=self._calibration_root,
                calibration_model_path=self._calibration_model_path,
                progress_callback=self._on_progress,
                log_callback=self.log_message.emit,
            )
            self.completed.emit(result)
        except Exception as exc:
            self.failed.emit(summarize_exception(exc))
