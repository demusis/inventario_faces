from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Callable

from PySide6.QtCore import QThread, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QStyle,
    QVBoxLayout,
    QWidget,
)

from inventario_faces import __version__
from inventario_faces.app import load_default_runtime_config, persist_runtime_config
from inventario_faces.domain.config import AppConfig
from inventario_faces.gui.config_dialog import ConfigDialog
from inventario_faces.gui.distributed_monitor_dialog import DistributedMonitorDialog
from inventario_faces.gui.face_set_comparison_dialog import FaceSetComparisonDialog
from inventario_faces.gui.icon_utils import apply_standard_icon
from inventario_faces.gui.worker import FaceSearchWorker, InventoryWorker
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
        self._last_work_directory: Path | None = None
        self._thread: QThread | None = None
        self._worker: InventoryWorker | FaceSearchWorker | None = None
        self._monitor_dialog: DistributedMonitorDialog | None = None
        self._comparison_dialog: FaceSetComparisonDialog | None = None
        self._close_requested = False
        self._build_ui()

    def _build_ui(self) -> None:
        self.setWindowTitle(self._current_config.app.name)
        app_icon = QApplication.windowIcon()
        if not app_icon.isNull():
            self.setWindowIcon(app_icon)
        self.resize(920, 640)

        central_widget = QWidget(self)
        layout = QVBoxLayout(central_widget)

        selector_layout = QHBoxLayout()
        selector_layout.addWidget(QLabel("Pasta de entrada:"))
        self._folder_input = QLineEdit()
        self._folder_input.setPlaceholderText("Selecione o diretório raiz da evidência")
        self._browse_button = QPushButton("Selecionar Pasta")
        apply_standard_icon(self, self._browse_button, QStyle.SP_DirOpenIcon)
        self._browse_button.clicked.connect(self._select_folder)
        selector_layout.addWidget(self._folder_input)
        selector_layout.addWidget(self._browse_button)

        controls_layout = QHBoxLayout()
        self._config_button = QPushButton("Configurações")
        apply_standard_icon(self, self._config_button, QStyle.SP_FileDialogDetailedView)
        self._config_button.clicked.connect(self._open_configuration_dialog)
        self._about_button = QPushButton("Sobre")
        apply_standard_icon(self, self._about_button, QStyle.SP_MessageBoxInformation)
        self._about_button.clicked.connect(self._show_about_dialog)
        self._run_button = QPushButton("Criar inventário")
        apply_standard_icon(self, self._run_button, QStyle.SP_MediaPlay)
        self._run_button.clicked.connect(self._start_processing)
        self._face_search_button = QPushButton("Busca por faces")
        apply_standard_icon(self, self._face_search_button, QStyle.SP_FileDialogContentsView)
        self._face_search_button.clicked.connect(self._start_face_search)
        self._face_set_comparison_button = QPushButton("Comparar conjuntos")
        apply_standard_icon(self, self._face_set_comparison_button, QStyle.SP_FileDialogInfoView)
        self._face_set_comparison_button.clicked.connect(self._open_face_set_comparison_dialog)
        self._distributed_health_button = QPushButton("Monitor")
        apply_standard_icon(self, self._distributed_health_button, QStyle.SP_ComputerIcon)
        self._distributed_health_button.clicked.connect(self._inspect_distributed_health)
        self._open_report_button = QPushButton("Abrir Relatório")
        apply_standard_icon(self, self._open_report_button, QStyle.SP_DialogOpenButton)
        self._open_report_button.setEnabled(False)
        self._open_report_button.clicked.connect(self._open_report)
        controls_layout.addWidget(self._config_button)
        controls_layout.addWidget(self._run_button)
        controls_layout.addWidget(self._face_search_button)
        controls_layout.addWidget(self._face_set_comparison_button)
        controls_layout.addWidget(self._distributed_health_button)
        controls_layout.addWidget(self._open_report_button)
        controls_layout.addStretch(1)
        controls_layout.addWidget(self._about_button)

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
        dialog = ConfigDialog(
            self._current_config,
            self,
            default_config=load_default_runtime_config(),
        )
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
        root_directory = self._require_root_directory()
        if root_directory is None:
            return

        work_directory = self._choose_work_directory(
            caption="Selecionar diretorio de trabalho",
            root_directory=root_directory,
        )
        if work_directory is None:
            return

        service = self._service_factory(self._current_config)
        self._prepare_execution_logs(
            f"Iniciando processamento de {root_directory}...",
            f"Diretorio de trabalho: {work_directory}",
        )
        self._start_worker(InventoryWorker(service, root_directory, work_directory))

    def _start_face_search(self) -> None:
        root_directory = self._require_root_directory()
        if root_directory is None:
            return

        selected_paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Selecionar imagens para busca por faces",
            str(root_directory),
            self._image_file_filter(),
        )
        if not selected_paths:
            return

        query_image_paths = [Path(selected_path) for selected_path in selected_paths]
        invalid_paths = [path for path in query_image_paths if not path.exists()]
        if invalid_paths:
            QMessageBox.warning(
                self,
                "Arquivo inválido",
                "Uma ou mais imagens de consulta selecionadas não existem mais no disco.",
            )
            return

        work_directory = self._choose_work_directory(
            caption="Selecionar diretorio de trabalho",
            root_directory=root_directory,
        )
        if work_directory is None:
            return

        service = self._service_factory(self._current_config)
        self._prepare_execution_logs(
            f"Iniciando busca por faces em {root_directory}...",
            f"Imagens de consulta: {len(query_image_paths)}",
            *[
                f"Consulta {index}: {query_path}"
                for index, query_path in enumerate(query_image_paths, start=1)
            ],
            f"Diretorio de trabalho: {work_directory}",
        )
        self._start_worker(FaceSearchWorker(service, root_directory, query_image_paths, work_directory))

    def _inspect_distributed_health(self) -> None:
        if not self._current_config.distributed.enabled:
            QMessageBox.warning(
                self,
                "Modo distribuído desativado",
                "Ative o processamento compartilhado nas configurações para usar o monitor.",
            )
            return

        root_directory = self._require_root_directory()
        if root_directory is None:
            return

        work_directory = self._choose_work_directory(
            caption="Selecionar diretorio onde o processamento esta sendo gravado",
            root_directory=root_directory,
        )
        if work_directory is None:
            return

        self._append_log(f"Abrindo monitor distribuído em {root_directory}...")
        self._append_log(f"Diretório de trabalho: {work_directory}")

        if self._monitor_dialog is not None:
            self._monitor_dialog.close()
            self._monitor_dialog = None

        self._monitor_dialog = DistributedMonitorDialog(
            service_factory=lambda: self._service_factory(self._current_config),
            config=self._current_config,
            root_directory=root_directory,
            work_directory=work_directory,
            parent=self,
        )
        self._monitor_dialog.destroyed.connect(self._clear_monitor_dialog)
        self._monitor_dialog.show()
        self._monitor_dialog.raise_()
        self._monitor_dialog.activateWindow()

    def _open_face_set_comparison_dialog(self) -> None:
        initial_directory: Path | None = None
        folder = self._folder_input.text().strip()
        if folder:
            candidate = Path(folder)
            if candidate.exists():
                initial_directory = candidate

        if self._comparison_dialog is not None:
            self._comparison_dialog.show()
            self._comparison_dialog.raise_()
            self._comparison_dialog.activateWindow()
            return

        self._comparison_dialog = FaceSetComparisonDialog(
            service_factory=lambda: self._service_factory(self._current_config),
            config=self._current_config,
            initial_input_directory=initial_directory,
            initial_work_directory=self._last_work_directory or initial_directory,
            parent=self,
        )
        self._comparison_dialog.destroyed.connect(self._clear_comparison_dialog)
        self._comparison_dialog.show()
        self._comparison_dialog.raise_()
        self._comparison_dialog.activateWindow()

    def _prepare_execution_logs(self, *messages: str) -> None:
        self._set_running_state(True)
        self._current_report_path = None
        self._progress_bar.setValue(0)
        self._log_view.clear()
        self._open_report_button.setEnabled(False)
        for message in messages:
            self._append_log(message)

    def _select_work_directory(self, *, caption: str, initial_directory: Path) -> Path | None:
        selected = QFileDialog.getExistingDirectory(self, caption, str(initial_directory))
        if not selected:
            return None
        return Path(selected)

    def _require_root_directory(self) -> Path | None:
        folder = self._folder_input.text().strip()
        if not folder:
            QMessageBox.warning(self, "Diretório obrigatório", "Selecione uma pasta de entrada.")
            return None

        root_directory = Path(folder)
        if not root_directory.exists():
            QMessageBox.warning(self, "Diretório inválido", "A pasta selecionada não existe.")
            return None
        return root_directory

    def _choose_work_directory(self, *, caption: str, root_directory: Path) -> Path | None:
        work_directory = self._select_work_directory(
            caption=caption,
            initial_directory=self._last_work_directory or root_directory,
        )
        if work_directory is not None:
            self._last_work_directory = work_directory
        return work_directory

    def _start_worker(self, worker: InventoryWorker | FaceSearchWorker) -> None:
        self._thread = QThread(self)
        self._worker = worker
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
        self._append_log("[ERRO] Execucao interrompida.")
        self._append_log(error)
        self._status_label.setText("Falha no processamento.")
        self._set_running_state(False)
        short_error = error.splitlines()[0] if error else "Falha desconhecida."
        QMessageBox.critical(
            self,
            "Erro",
            f"{short_error}\n\nConsulte o log exibido na janela para detalhes.",
        )

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
        normalized = message.replace("\r\n", "\n").replace("\r", "\n")
        lines = normalized.split("\n")
        for line in lines:
            if line == "":
                self._log_view.appendPlainText("")
                continue
            timestamp = datetime.now().strftime("%H:%M:%S")
            self._log_view.appendPlainText(f"{timestamp} | {line}")

    def _cleanup_thread(self) -> None:
        if self._worker is not None:
            self._worker.deleteLater()
            self._worker = None
        if self._thread is not None:
            self._thread.deleteLater()
            self._thread = None
        self._set_running_state(False)

    def _clear_monitor_dialog(self, *_args: object) -> None:
        self._monitor_dialog = None
        if self._close_requested and not self._is_task_running():
            self._close_requested = False
            self.close()

    def _clear_comparison_dialog(self, *_args: object) -> None:
        self._comparison_dialog = None
        if self._close_requested and not self._is_task_running():
            self._close_requested = False
            self.close()

    def _set_running_state(self, running: bool) -> None:
        self._run_button.setEnabled(not running)
        self._face_search_button.setEnabled(not running)
        self._face_set_comparison_button.setEnabled(not running)
        self._distributed_health_button.setEnabled(True)
        self._config_button.setEnabled(not running)
        self._folder_input.setEnabled(not running)
        self._browse_button.setEnabled(not running)
        self._open_report_button.setEnabled((not running) and self._current_report_path is not None)

    def _image_file_filter(self) -> str:
        patterns = " ".join(f"*{extension}" for extension in self._current_config.media.image_extensions)
        return f"Imagens suportadas ({patterns});;Todos os arquivos (*)"

    def closeEvent(self, event) -> None:  # type: ignore[override]
        if self._is_task_running():
            QMessageBox.information(
                self,
                "Processamento em andamento",
                "Aguarde o termino do processamento antes de fechar o aplicativo.",
            )
            event.ignore()
            return

        if self._monitor_dialog is not None and not self._monitor_dialog.request_close():
            self._close_requested = True
            self._append_log("Aguardando encerramento do monitor antes de fechar o aplicativo.")
            event.ignore()
            return

        if self._comparison_dialog is not None and not self._comparison_dialog.request_close():
            self._close_requested = True
            self._append_log("Aguardando encerramento da comparacao antes de fechar o aplicativo.")
            event.ignore()
            return

        self._close_requested = False
        super().closeEvent(event)

    def _is_task_running(self) -> bool:
        return self._thread is not None
