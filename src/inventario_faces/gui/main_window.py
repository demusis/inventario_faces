from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Callable

from PySide6.QtCore import QThread, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from inventario_faces import __version__
from inventario_faces.app import persist_runtime_config
from inventario_faces.domain.config import AppConfig
from inventario_faces.gui.config_dialog import ConfigDialog
from inventario_faces.gui.worker import InventoryWorker
from inventario_faces.services.inventory_service import InventoryService


class MainWindow(QMainWindow):
    def __init__(
        self,
        service_factory: Callable[[AppConfig | None], InventoryService],
        initial_config: AppConfig,
    ) -> None:
        super().__init__()
        self._service_factory = service_factory
        self._current_config = initial_config
        self._current_report_path: Path | None = None
        self._thread: QThread | None = None
        self._worker: InventoryWorker | None = None
        self._build_ui()

    def _build_ui(self) -> None:
        self.setWindowTitle(self._current_config.app.name)
        self.resize(920, 640)

        central_widget = QWidget(self)
        layout = QVBoxLayout(central_widget)

        selector_layout = QHBoxLayout()
        selector_layout.addWidget(QLabel("Pasta de entrada:"))
        self._folder_input = QLineEdit()
        self._folder_input.setPlaceholderText("Selecione o diretório raiz da evidência")
        self._browse_button = QPushButton("Selecionar Pasta")
        self._browse_button.clicked.connect(self._select_folder)
        selector_layout.addWidget(self._folder_input)
        selector_layout.addWidget(self._browse_button)

        controls_layout = QHBoxLayout()
        self._config_button = QPushButton("Configurações")
        self._config_button.clicked.connect(self._open_configuration_dialog)
        self._about_button = QPushButton("Sobre")
        self._about_button.clicked.connect(self._show_about_dialog)
        self._run_button = QPushButton("Executar")
        self._run_button.clicked.connect(self._start_processing)
        self._open_report_button = QPushButton("Abrir Relatório")
        self._open_report_button.setEnabled(False)
        self._open_report_button.clicked.connect(self._open_report)
        controls_layout.addWidget(self._config_button)
        controls_layout.addWidget(self._about_button)
        controls_layout.addWidget(self._run_button)
        controls_layout.addWidget(self._open_report_button)

        self._status_label = QLabel(
            "Fluxo forense auditável. Os resultados são probabilísticos e exigem revisão humana."
        )
        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 100)
        self._log_view = QPlainTextEdit()
        self._log_view.setReadOnly(True)

        layout.addLayout(selector_layout)
        layout.addLayout(controls_layout)
        layout.addWidget(self._status_label)
        layout.addWidget(self._progress_bar)
        layout.addWidget(QLabel("Log de execução:"))
        layout.addWidget(self._log_view)

        self.setCentralWidget(central_widget)

    def _select_folder(self) -> None:
        selected = QFileDialog.getExistingDirectory(self, "Selecionar Diretório")
        if selected:
            self._folder_input.setText(selected)

    def _open_configuration_dialog(self) -> None:
        dialog = ConfigDialog(self._current_config, self)
        if dialog.exec():
            self._current_config = dialog.selected_config
            self.setWindowTitle(self._current_config.app.name)
            self._status_label.setText("Configurações atualizadas para a próxima execução.")
            try:
                saved_path = persist_runtime_config(self._current_config)
            except Exception as exc:
                self._append_log(f"Falha ao persistir configurações: {exc}")
                QMessageBox.warning(self, "Persistência de configurações", str(exc))
                return
            self._append_log(f"Configurações persistidas em: {saved_path}")

    def _start_processing(self) -> None:
        folder = self._folder_input.text().strip()
        if not folder:
            QMessageBox.warning(self, "Diretório obrigatório", "Selecione uma pasta de entrada.")
            return

        root_directory = Path(folder)
        if not root_directory.exists():
            QMessageBox.warning(self, "Diretório inválido", "A pasta selecionada não existe.")
            return

        self._set_running_state(True)
        self._progress_bar.setValue(0)
        self._log_view.clear()
        self._append_log(f"Iniciando processamento de {root_directory}...")

        service = self._service_factory(self._current_config)
        self._thread = QThread(self)
        self._worker = InventoryWorker(service, root_directory)
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)
        self._worker.progress_changed.connect(self._on_progress_changed)
        self._worker.log_message.connect(self._append_log)
        self._worker.completed.connect(self._on_completed)
        self._worker.failed.connect(self._on_failed)
        self._worker.completed.connect(self._thread.quit)
        self._worker.failed.connect(self._thread.quit)
        self._thread.finished.connect(self._cleanup_thread)
        self._thread.start()

    def _on_progress_changed(self, value: int, message: str) -> None:
        self._progress_bar.setValue(value)
        self._status_label.setText(message)

    def _on_completed(self, report_path: str) -> None:
        self._current_report_path = Path(report_path)
        self._progress_bar.setValue(100)
        self._status_label.setText("Processamento concluído.")
        self._append_log(f"Relatório gerado em: {report_path}")
        self._open_report_button.setEnabled(True)
        self._set_running_state(False)

    def _on_failed(self, error: str) -> None:
        self._append_log(f"Falha: {error}")
        self._status_label.setText("Falha no processamento.")
        self._set_running_state(False)
        QMessageBox.critical(self, "Erro", error)

    def _open_report(self) -> None:
        if self._current_report_path is None:
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(self._current_report_path)))

    def _show_about_dialog(self) -> None:
        QMessageBox.about(
            self,
            "Sobre",
            (
                f"{self._current_config.app.name}\n"
                f"Versão {__version__}\n\n"
                "Aplicativo de código aberto para inventário facial forense assistido.\n"
                "Repositório oficial:\n"
                "https://github.com/demusis/inventario_faces"
            ),
        )

    def _append_log(self, message: str) -> None:
        timestamp = datetime.now().strftime("%H:%M:%S")
        self._log_view.appendPlainText(f"{timestamp} | {message}")

    def _cleanup_thread(self) -> None:
        if self._worker is not None:
            self._worker.deleteLater()
            self._worker = None
        if self._thread is not None:
            self._thread.deleteLater()
            self._thread = None

    def _set_running_state(self, running: bool) -> None:
        self._run_button.setEnabled(not running)
        self._config_button.setEnabled(not running)
        self._folder_input.setEnabled(not running)
        self._browse_button.setEnabled(not running)
        self._open_report_button.setEnabled((not running) and self._current_report_path is not None)
