from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Callable

from PySide6.QtCore import QObject, QThread, QTimer, Qt, QUrl, Signal, Slot
from PySide6.QtGui import QDesktopServices, QColor
from PySide6.QtWidgets import (
    QDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QProgressBar,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from inventario_faces.domain.config import AppConfig
from inventario_faces.infrastructure.distributed_coordination import (
    DistributedHealthSnapshot,
    DistributedNodeStatus,
    DistributedPartialValidation,
)
from inventario_faces.services.inventory_service import DistributedHealthResult, InventoryService
from inventario_faces.utils.time_utils import format_local_datetime


class _HealthPollWorker(QObject):
    completed = Signal(object)
    failed = Signal(str)

    def __init__(
        self,
        service_factory: Callable[[], InventoryService],
        root_directory: Path,
        work_directory: Path,
    ) -> None:
        super().__init__()
        self._service_factory = service_factory
        self._root_directory = root_directory
        self._work_directory = work_directory

    @Slot()
    def run(self) -> None:
        try:
            result = self._service_factory().inspect_distributed_health(
                self._root_directory,
                work_directory=self._work_directory,
            )
            self.completed.emit(result)
        except Exception as exc:
            self.failed.emit(str(exc))


class DistributedMonitorDialog(QDialog):
    def __init__(
        self,
        *,
        service_factory: Callable[[], InventoryService],
        config: AppConfig,
        root_directory: Path,
        work_directory: Path,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WA_DeleteOnClose, True)
        self._service_factory = service_factory
        self._config = config
        self._root_directory = Path(root_directory).resolve()
        self._work_directory = Path(work_directory).resolve()
        self._latest_result: DistributedHealthResult | None = None
        self._refresh_thread: QThread | None = None
        self._refresh_worker: _HealthPollWorker | None = None
        self._refresh_in_progress = False
        self._close_requested = False

        self.setWindowTitle("Monitor distribuido")
        self.resize(1180, 760)
        self.setMinimumSize(980, 620)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self.request_refresh)

        self._build_ui()
        self._apply_refresh_interval(max(3, self._config.distributed.heartbeat_interval_seconds))
        self.request_refresh()
        self._timer.start()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        header_group = QGroupBox("Contexto")
        header_layout = QGridLayout(header_group)
        header_layout.addWidget(QLabel("Pasta de evidencias:"), 0, 0)
        header_layout.addWidget(QLabel(str(self._root_directory)), 0, 1)
        header_layout.addWidget(QLabel("Diretorio de trabalho:"), 1, 0)
        header_layout.addWidget(QLabel(str(self._work_directory)), 1, 1)
        header_layout.addWidget(QLabel("Execucao compartilhada:"), 2, 0)
        header_layout.addWidget(QLabel(self._config.distributed.execution_label), 2, 1)
        layout.addWidget(header_group)

        controls_layout = QHBoxLayout()
        controls_layout.addWidget(QLabel("Atualizacao automatica (s):"))
        self._interval_spin = QSpinBox(self)
        self._interval_spin.setRange(3, 3600)
        self._interval_spin.valueChanged.connect(self._apply_refresh_interval)
        controls_layout.addWidget(self._interval_spin)

        self._refresh_button = QPushButton("Atualizar agora")
        self._refresh_button.clicked.connect(self.request_refresh)
        controls_layout.addWidget(self._refresh_button)

        self._pause_button = QPushButton("Pausar")
        self._pause_button.clicked.connect(self._toggle_timer)
        controls_layout.addWidget(self._pause_button)

        self._open_text_button = QPushButton("Abrir TXT")
        self._open_text_button.clicked.connect(self._open_text_report)
        self._open_text_button.setEnabled(False)
        controls_layout.addWidget(self._open_text_button)

        self._open_json_button = QPushButton("Abrir JSON")
        self._open_json_button.clicked.connect(self._open_json_report)
        self._open_json_button.setEnabled(False)
        controls_layout.addWidget(self._open_json_button)

        controls_layout.addStretch(1)
        self._last_update_label = QLabel("Ultima atualizacao: -")
        controls_layout.addWidget(self._last_update_label)
        layout.addLayout(controls_layout)

        summary_group = QGroupBox("Resumo")
        summary_layout = QGridLayout(summary_group)
        self._summary_total = self._create_metric_label(summary_layout, 0, "Arquivos no plano")
        self._summary_done = self._create_metric_label(summary_layout, 1, "Concluidos")
        self._summary_processing = self._create_metric_label(summary_layout, 2, "Em processamento")
        self._summary_pending = self._create_metric_label(summary_layout, 3, "Pendentes")
        self._summary_nodes = self._create_metric_label(summary_layout, 4, "Nos ativos")
        self._summary_partials = self._create_metric_label(summary_layout, 5, "Parciais com problema")
        layout.addWidget(summary_group)

        self._progress = QProgressBar(self)
        self._progress.setRange(0, 100)
        layout.addWidget(self._progress)

        nodes_group = QGroupBox("Nos e instancias")
        nodes_layout = QVBoxLayout(nodes_group)
        self._nodes_table = QTableWidget(self)
        self._nodes_table.setColumnCount(7)
        self._nodes_table.setHorizontalHeaderLabels(
            ["Host", "PID", "Heartbeat", "Ultimo heartbeat", "Fase", "Arquivo atual", "Idade"]
        )
        header = self._nodes_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Interactive)
        header.setStretchLastSection(True)
        self._nodes_table.setColumnWidth(0, 180)
        self._nodes_table.setColumnWidth(1, 90)
        self._nodes_table.setColumnWidth(2, 120)
        self._nodes_table.setColumnWidth(3, 170)
        self._nodes_table.setColumnWidth(4, 120)
        self._nodes_table.setColumnWidth(5, 330)
        self._nodes_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._nodes_table.setSelectionBehavior(QTableWidget.SelectRows)
        self._nodes_table.setAlternatingRowColors(True)
        nodes_layout.addWidget(self._nodes_table)
        layout.addWidget(nodes_group, stretch=2)

        partials_group = QGroupBox("Parciais com problema")
        partials_layout = QVBoxLayout(partials_group)
        self._partials_table = QTableWidget(self)
        self._partials_table.setColumnCount(3)
        self._partials_table.setHorizontalHeaderLabels(["Arquivo", "Status", "Detalhe"])
        partial_header = self._partials_table.horizontalHeader()
        partial_header.setSectionResizeMode(QHeaderView.Interactive)
        partial_header.setStretchLastSection(True)
        self._partials_table.setColumnWidth(0, 360)
        self._partials_table.setColumnWidth(1, 120)
        self._partials_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._partials_table.setSelectionBehavior(QTableWidget.SelectRows)
        self._partials_table.setAlternatingRowColors(True)
        partials_layout.addWidget(self._partials_table)
        layout.addWidget(partials_group, stretch=1)

        self._status_label = QLabel("Aguardando primeira leitura.")
        layout.addWidget(self._status_label)

    def _create_metric_label(self, layout: QGridLayout, column: int, title: str) -> QLabel:
        title_label = QLabel(title)
        value_label = QLabel("0")
        value_label.setAlignment(Qt.AlignCenter)
        value_label.setStyleSheet("font-size: 22px; font-weight: bold;")
        layout.addWidget(title_label, 0, column, alignment=Qt.AlignCenter)
        layout.addWidget(value_label, 1, column, alignment=Qt.AlignCenter)
        return value_label

    def _apply_refresh_interval(self, seconds: int) -> None:
        self._interval_spin.blockSignals(True)
        self._interval_spin.setValue(seconds)
        self._interval_spin.blockSignals(False)
        self._timer.setInterval(seconds * 1000)

    def _toggle_timer(self) -> None:
        if self._timer.isActive():
            self._timer.stop()
            self._pause_button.setText("Retomar")
            self._status_label.setText("Atualizacao automatica pausada.")
        else:
            self._timer.start()
            self._pause_button.setText("Pausar")
            self._status_label.setText(
                f"Atualizacao automatica ativa a cada {self._interval_spin.value()} segundo(s)."
            )

    def request_refresh(self) -> None:
        if self._refresh_in_progress:
            return
        self._refresh_in_progress = True
        self._refresh_button.setEnabled(False)
        self._status_label.setText("Consultando estado compartilhado...")

        self._refresh_thread = QThread(self)
        self._refresh_worker = _HealthPollWorker(
            self._service_factory,
            self._root_directory,
            self._work_directory,
        )
        self._refresh_worker.moveToThread(self._refresh_thread)

        self._refresh_thread.started.connect(self._refresh_worker.run)
        self._refresh_worker.completed.connect(self._on_refresh_completed)
        self._refresh_worker.failed.connect(self._on_refresh_failed)
        self._refresh_worker.completed.connect(self._refresh_thread.quit)
        self._refresh_worker.failed.connect(self._refresh_thread.quit)
        self._refresh_thread.finished.connect(self._cleanup_refresh)
        self._refresh_thread.start()

    def _on_refresh_completed(self, result: object) -> None:
        if not isinstance(result, DistributedHealthResult):
            self._on_refresh_failed("Resultado invalido do monitor.")
            return
        self._latest_result = result
        self._render_snapshot(result.health_snapshot)
        self._open_text_button.setEnabled(True)
        self._open_json_button.setEnabled(True)

    def _on_refresh_failed(self, error: str) -> None:
        self._status_label.setText(f"Falha no monitoramento: {error}")

    def _cleanup_refresh(self) -> None:
        if self._refresh_worker is not None:
            self._refresh_worker.deleteLater()
            self._refresh_worker = None
        if self._refresh_thread is not None:
            self._refresh_thread.deleteLater()
            self._refresh_thread = None
        self._refresh_in_progress = False
        self._refresh_button.setEnabled(True)
        if self._close_requested:
            self.close()

    def _render_snapshot(self, snapshot: DistributedHealthSnapshot) -> None:
        use_processable_counts = snapshot.processable_files > 0
        total_files = snapshot.processable_files if use_processable_counts else snapshot.total_files
        completed_files = (
            snapshot.processable_completed_files if use_processable_counts else snapshot.completed_files
        )
        active_claims = (
            snapshot.processable_active_claims if use_processable_counts else snapshot.active_claims
        )
        pending_files = (
            snapshot.processable_pending_files if use_processable_counts else snapshot.pending_files
        )

        self._summary_total.setText(str(total_files))
        self._summary_done.setText(str(completed_files))
        self._summary_processing.setText(str(active_claims))
        self._summary_pending.setText(str(pending_files))
        self._summary_nodes.setText(f"{snapshot.active_nodes} / {snapshot.active_nodes + snapshot.stale_nodes}")
        self._summary_partials.setText(str(snapshot.missing_partials + snapshot.corrupted_partials))

        percent = 0
        if total_files > 0:
            percent = int((completed_files / total_files) * 100)
        self._progress.setValue(percent)
        self._progress.setFormat(
            f"{percent}% concluido ({completed_files}/{max(total_files, 1)} arquivos)"
        )

        self._render_nodes(snapshot.nodes)
        self._render_partials(snapshot.partials)

        self._last_update_label.setText(f"Ultima atualizacao: {self._timestamp_now_label()}")
        self._status_label.setText(self._overall_status_label(snapshot))

    def _render_nodes(self, nodes: tuple[DistributedNodeStatus, ...]) -> None:
        self._nodes_table.clearSpans()
        self._nodes_table.clearContents()
        self._nodes_table.setRowCount(max(1, len(nodes)))
        if not nodes:
            self._nodes_table.setItem(0, 0, QTableWidgetItem("Nenhum no ativo localizado."))
            self._nodes_table.setSpan(0, 0, 1, self._nodes_table.columnCount())
            return

        for row, node in enumerate(nodes):
            heartbeat_text, heartbeat_color = self._heartbeat_status_label(node)
            self._set_table_item(self._nodes_table, row, 0, node.hostname)
            self._set_table_item(self._nodes_table, row, 1, "-" if node.pid is None else str(node.pid))
            self._set_table_item(self._nodes_table, row, 2, heartbeat_text, heartbeat_color)
            self._set_table_item(self._nodes_table, row, 3, self._heartbeat_timestamp_label(node))
            self._set_table_item(self._nodes_table, row, 4, self._phase_label(node.phase))
            self._set_table_item(
                self._nodes_table,
                row,
                5,
                node.current_relative_path or "-",
            )
            self._set_table_item(self._nodes_table, row, 6, self._age_label(node.age_seconds))

    def _render_partials(self, partials: tuple[DistributedPartialValidation, ...]) -> None:
        invalid_partials = [item for item in partials if not item.is_healthy]
        self._partials_table.clearSpans()
        self._partials_table.clearContents()
        self._partials_table.setRowCount(max(1, len(invalid_partials)))
        if not invalid_partials:
            self._partials_table.setItem(0, 0, QTableWidgetItem("Nenhum parcial com problema."))
            self._partials_table.setSpan(0, 0, 1, self._partials_table.columnCount())
            return

        for row, partial in enumerate(invalid_partials):
            color = QColor("#C0392B") if partial.status == "corrupt" else QColor("#AF601A")
            self._set_table_item(self._partials_table, row, 0, partial.entry.relative_path)
            self._set_table_item(self._partials_table, row, 1, partial.status, color)
            self._set_table_item(self._partials_table, row, 2, partial.detail)

    def _heartbeat_status_label(self, node: DistributedNodeStatus) -> tuple[str, QColor]:
        if node.is_stale:
            return "Stale", QColor("#C0392B")
        if node.age_seconds is None:
            return "Sem dado", QColor("#7F8C8D")
        warning_seconds = max(self._config.distributed.heartbeat_interval_seconds * 2, 10)
        if node.age_seconds <= warning_seconds:
            return "OK", QColor("#1E8449")
        return f"Atrasado ({int(node.age_seconds)}s)", QColor("#AF601A")

    def _heartbeat_timestamp_label(self, node: DistributedNodeStatus) -> str:
        if not node.last_heartbeat_utc:
            return "-"
        try:
            instant = datetime.fromisoformat(node.last_heartbeat_utc)
        except ValueError:
            return node.last_heartbeat_utc
        local_text = format_local_datetime(instant)
        if local_text.startswith("-"):
            return "-"
        return local_text.split(" (", maxsplit=1)[0].split(" ", maxsplit=1)[-1]

    def _phase_label(self, phase: str) -> str:
        labels = {
            "idle": "Ocioso",
            "planejamento": "Planejamento",
            "processando": "Processando",
            "consolidando": "Consolidando",
        }
        return labels.get(phase, phase or "-")

    def _age_label(self, age_seconds: float | None) -> str:
        if age_seconds is None:
            return "-"
        if age_seconds < 60:
            return f"{int(age_seconds)} s"
        return f"{age_seconds / 60:.1f} min"

    def _timestamp_now_label(self) -> str:
        return datetime.now().strftime("%H:%M:%S")

    def _overall_status_label(self, snapshot: DistributedHealthSnapshot) -> str:
        issues = snapshot.missing_partials + snapshot.corrupted_partials + snapshot.stale_nodes
        if issues == 0 and snapshot.active_claims >= 0:
            return (
                f"Lote monitorado sem alertas criticos. "
                f"Nos ativos: {snapshot.active_nodes}. "
                f"Arquivos em processamento: {snapshot.active_claims}. "
                f"Pendentes: {snapshot.pending_files}."
            )
        return (
            f"Atencao: {snapshot.stale_nodes} no(s) stale, "
            f"{snapshot.missing_partials} parcial(is) ausente(s), "
            f"{snapshot.corrupted_partials} parcial(is) corrompido(s). "
            f"Arquivos em processamento: {snapshot.active_claims}. "
            f"Pendentes: {snapshot.pending_files}."
        )

    def _set_table_item(
        self,
        table: QTableWidget,
        row: int,
        column: int,
        text: str,
        color: QColor | None = None,
    ) -> None:
        item = QTableWidgetItem(text)
        if color is not None:
            item.setForeground(color)
        table.setItem(row, column, item)

    def _open_text_report(self) -> None:
        if self._latest_result is None:
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(self._latest_result.report.tex_path)))

    def _open_json_report(self) -> None:
        if self._latest_result is None:
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(self._latest_result.json_path)))

    def request_close(self) -> bool:
        self._timer.stop()
        if self._refresh_thread is not None and self._refresh_thread.isRunning():
            self._close_requested = True
            self.hide()
            return False
        self.close()
        return True

    def closeEvent(self, event) -> None:  # type: ignore[override]
        self._timer.stop()
        if self._refresh_thread is not None and self._refresh_thread.isRunning():
            self._close_requested = True
            self.hide()
            event.ignore()
            return
        super().closeEvent(event)
