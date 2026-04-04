from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QObject, Signal, Slot

from inventario_faces.services.inventory_service import InventoryService


class InventoryWorker(QObject):
    progress_changed = Signal(int, str)
    log_message = Signal(str)
    completed = Signal(str)
    failed = Signal(str)

    def __init__(self, service: InventoryService, root_directory: Path) -> None:
        super().__init__()
        self._service = service
        self._root_directory = root_directory

    @Slot()
    def run(self) -> None:
        try:
            result = self._service.run(
                self._root_directory,
                progress_callback=self._on_progress,
                log_callback=self.log_message.emit,
            )
            report_path = result.report.pdf_path or result.report.docx_path or result.report.tex_path
            self.completed.emit(str(report_path))
        except Exception as exc:
            self.failed.emit(str(exc))

    def _on_progress(self, current: int, total: int, message: str) -> None:
        percent = 0 if total == 0 else int((current / total) * 100)
        self.progress_changed.emit(percent, message)
