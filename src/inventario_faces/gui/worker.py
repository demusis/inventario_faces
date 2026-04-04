from __future__ import annotations

from pathlib import Path
from typing import Callable

from PySide6.QtCore import QObject, Signal, Slot

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
            self.failed.emit(str(exc))

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
