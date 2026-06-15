from __future__ import annotations

from datetime import datetime
from html import escape
from pathlib import Path
import zipfile

from PySide6.QtCore import QThread, QTimer, Qt, QUrl
from PySide6.QtGui import QColor, QDesktopServices
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
    QStyle,
)

import numpy as np

from inventario_faces.domain.config import AppConfig
from inventario_faces.domain.entities import (
    FaceSetComparisonEntry,
    FaceSetComparisonInput,
    FaceSetComparisonMatch,
    FaceSetComparisonResult,
)
from inventario_faces.gui.comparison_statistics import (
    _DistributionSeries,
    _GroupComparisonTestResult,
    _distribution_card_title,
    _distribution_help_html,
    _distribution_popup_html,
    _distribution_series_label,
    _distribution_zone_short_label,
    _expanded_score_range,
    _group_comparison_direction_text,
    _group_comparison_significance_text,
    _likelihood_ratio_selection_html,
    _mann_whitney_group_comparison,
    _summary_help_html,
    _summary_popup_html,
)
from inventario_faces.gui.comparison_widgets import (
    AdaptiveImageLabel,
    LikelihoodRatioDensityWidget,
    SimilarityDistributionWidget,
)
from inventario_faces.gui.face_set_comparison_help import build_face_set_comparison_help_html
from inventario_faces.gui.icon_utils import apply_standard_icon
from inventario_faces.gui.worker import FaceSetComparisonWorker
from inventario_faces.utils.density_utils import fit_score_density_model, score_density_method_label


class FaceSetComparisonDialog(QDialog):
    def __init__(
        self,
        *,
        service_factory,
        config: AppConfig,
        initial_input_directory: Path | None = None,
        initial_work_directory: Path | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._service_factory = service_factory
        self._config = config
        self._initial_input_directory = (
            Path(initial_input_directory).resolve() if initial_input_directory else Path.cwd()
        )
        self._initial_work_directory = (
            Path(initial_work_directory).resolve()
            if initial_work_directory is not None
            else self._initial_input_directory
        )
        self._latest_result: FaceSetComparisonResult | None = None
        self._entry_by_id: dict[str, FaceSetComparisonEntry] = {}
        self._thread: QThread | None = None
        self._worker: FaceSetComparisonWorker | None = None
        self._run_started_at: datetime | None = None
        self._last_signal_at: datetime | None = None
        self._activity_detail = ""
        self._activity_state = "idle"
        self._activity_frame_index = 0
        self._activity_timer = QTimer(self)
        self._activity_timer.setInterval(1000)
        self._activity_timer.timeout.connect(self._refresh_activity_panel)

        self.setWindowTitle("Comparação entre grupos faciais")
        self.setWindowFlag(Qt.WindowMaximizeButtonHint, True)
        self.setAttribute(Qt.WA_DeleteOnClose, True)
        self.resize(1280, 820)
        self.setMinimumSize(980, 620)
        self._apply_styles()
        self._build_ui()
        self._work_directory_input.setText(str(self._initial_work_directory))

    def _apply_styles(self) -> None:
        self.setStyleSheet(
            """
            QDialog { background:#eef3f8; }
            QGroupBox {
                font-weight:600; border:1px solid #d4dde7; border-radius:12px;
                margin-top:14px; padding-top:14px; background:#f9fbfe; color:#0f172a;
            }
            QGroupBox::title {
                subcontrol-origin:margin; left:12px; padding:0 4px; color:#334155;
            }
            QListWidget, QTableWidget, QPlainTextEdit, QLineEdit, QTextBrowser {
                background:#ffffff; color:#0f172a; border:1px solid #d7e0ea; border-radius:10px;
                selection-background-color:#cfe6ff; selection-color:#0f172a; gridline-color:#e7edf4;
            }
            QHeaderView::section {
                background:#eef4fa; color:#0f172a; border:none; border-bottom:1px solid #d7e0ea;
                padding:7px; font-weight:600;
            }
            QPushButton {
                background:#ffffff; color:#0f172a; border:1px solid #c9d5e3; border-radius:9px;
                padding:7px 12px;
            }
            QPushButton:hover { background:#f4f7fb; }
            QPushButton#PrimaryButton {
                background:#0f766e; color:#ffffff; border-color:#0f766e; font-weight:700;
            }
            QPushButton#PrimaryButton:hover { background:#115e59; }
            QFrame#ToolBarFrame {
                background:#f9fbfe; border:1px solid #d7e0ea; border-radius:12px;
            }
            QFrame#MetricCard {
                background:#ffffff; border:1px solid #d9e3ee; border-radius:12px;
            }
            QLabel#MetricTitle { color:#64748b; font-size:11px; font-weight:600; }
            QLabel#MetricValue { color:#0f172a; font-size:20px; font-weight:700; }
            QLabel#MetricDetail { color:#475569; font-size:11px; }
            QLabel#SectionHint { color:#64748b; }
            QLabel#ActivityBadge {
                color:#0f172a; font-weight:700; background:#e2e8f0;
                border:1px solid #cbd5e1; border-radius:999px; padding:5px 12px;
            }
            QLabel#PreviewInfo {
                background:#f8fafc; color:#0f172a; border:1px solid #d7e0ea;
                border-radius:10px; padding:10px;
            }
            """
        )

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        top = QSplitter(Qt.Horizontal, self)
        top.setChildrenCollapsible(False)
        top.addWidget(self._build_set_group("Padrão", "A"))
        top.addWidget(self._build_set_group("Questionado", "B"))
        top.setStretchFactor(0, 1)
        top.setStretchFactor(1, 1)
        layout.addWidget(top, stretch=2)

        layout.addWidget(self._build_execution_bar())
        layout.addWidget(self._build_results_bar())

        status_layout = QHBoxLayout()
        self._status_label = QLabel("Selecione os grupos Padrão e Questionado para iniciar a comparação.")
        self._status_label.setObjectName("SectionHint")
        self._progress_bar = QProgressBar(self)
        self._progress_bar.setRange(0, 100)
        status_layout.addWidget(self._status_label, 3)
        status_layout.addWidget(self._progress_bar, 2)
        layout.addLayout(status_layout)

        activity_layout = QHBoxLayout()
        self._activity_badge = QLabel("Pronto", self)
        self._activity_badge.setObjectName("ActivityBadge")
        self._activity_meta_label = QLabel(
            "Aguardando nova execucao. O mesmo log exibido aqui tambem e gravado em run.log e events.jsonl.",
            self,
        )
        self._activity_meta_label.setObjectName("SectionHint")
        activity_layout.addWidget(self._activity_badge)
        activity_layout.addWidget(self._activity_meta_label, 1)
        layout.addLayout(activity_layout)

        self._quick_summary_label = QLabel("Os resultados detalhados ficarão disponíveis nos botões acima.")
        self._quick_summary_label.setObjectName("SectionHint")
        layout.addWidget(self._quick_summary_label)

        log_group = QGroupBox("Log da comparação", self)
        log_layout = QVBoxLayout(log_group)
        self._log_view = QPlainTextEdit(self)
        self._log_view.setReadOnly(True)
        self._log_view.setLineWrapMode(QPlainTextEdit.NoWrap)
        log_layout.addWidget(self._log_view)
        layout.addWidget(log_group, stretch=3)

    def _build_execution_bar(self) -> QWidget:
        frame = QFrame(self)
        frame.setObjectName("ToolBarFrame")
        layout = QGridLayout(frame)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setHorizontalSpacing(8)
        layout.setVerticalSpacing(6)
        layout.addWidget(QLabel("Diretório de trabalho:"))
        self._work_directory_input = QLineEdit(self)
        layout.addWidget(self._work_directory_input, 0, 1)
        self._browse_work_directory_button = QPushButton("Selecionar")
        apply_standard_icon(self, self._browse_work_directory_button, QStyle.SP_DirOpenIcon)
        self._browse_work_directory_button.clicked.connect(self._select_work_directory)
        layout.addWidget(self._browse_work_directory_button, 0, 2)
        self._run_button = QPushButton("Comparar conjuntos")
        apply_standard_icon(self, self._run_button, QStyle.SP_MediaPlay)
        self._run_button.setObjectName("PrimaryButton")
        self._run_button.clicked.connect(self._start_comparison)
        layout.addWidget(self._run_button, 0, 3)
        self._help_button = QPushButton("Ajuda")
        apply_standard_icon(self, self._help_button, QStyle.SP_DialogHelpButton)
        self._help_button.clicked.connect(self._open_help_popup)
        layout.addWidget(self._help_button, 0, 4)
        self._export_button = QPushButton("Exportar ZIP")
        apply_standard_icon(self, self._export_button, QStyle.SP_DialogSaveButton)
        self._export_button.setEnabled(False)
        self._export_button.clicked.connect(self._export_results_zip)
        layout.addWidget(self._export_button, 0, 5)
        self._open_export_button = QPushButton("Abrir execução")
        apply_standard_icon(self, self._open_export_button, QStyle.SP_DialogOpenButton)
        self._open_export_button.setEnabled(False)
        self._open_export_button.clicked.connect(self._open_export_directory)
        layout.addWidget(self._open_export_button, 0, 6)
        self._calibration_directory_input = QLineEdit(self)
        self._calibration_directory_input.setPlaceholderText("Uma subpasta por identidade rotulada")
        self._browse_calibration_directory_button = QPushButton("Selecionar base")
        apply_standard_icon(self, self._browse_calibration_directory_button, QStyle.SP_DirOpenIcon)
        self._browse_calibration_directory_button.clicked.connect(self._select_calibration_directory)
        layout.addWidget(QLabel("Base de calibração LR (opcional):"), 1, 0)
        layout.addWidget(self._calibration_directory_input, 1, 1, 1, 5)
        layout.addWidget(self._browse_calibration_directory_button, 1, 6)
        self._calibration_model_input = QLineEdit(self)
        self._calibration_model_input.setPlaceholderText("Arquivo JSON com modelo LR já calculado")
        self._browse_calibration_model_button = QPushButton("Carregar modelo")
        apply_standard_icon(self, self._browse_calibration_model_button, QStyle.SP_DialogOpenButton)
        self._browse_calibration_model_button.clicked.connect(self._select_calibration_model)
        layout.addWidget(QLabel("Modelo de calibração LR (opcional):"), 2, 0)
        layout.addWidget(self._calibration_model_input, 2, 1, 1, 5)
        layout.addWidget(self._browse_calibration_model_button, 2, 6)
        layout.setColumnStretch(1, 1)
        return frame

    def _build_results_bar(self) -> QGroupBox:
        group = QGroupBox("Resultados", self)
        layout = QHBoxLayout(group)
        layout.setSpacing(8)
        layout.addWidget(QLabel("Reamostragens:"))
        self._bootstrap_resamples_input = QSpinBox(self)
        self._bootstrap_resamples_input.setRange(200, 50000)
        self._bootstrap_resamples_input.setSingleStep(500)
        self._bootstrap_resamples_input.setValue(5000)
        self._bootstrap_resamples_input.setMaximumWidth(110)
        layout.addWidget(self._bootstrap_resamples_input)
        layout.addWidget(QLabel("Significância (%):"))
        self._significance_input = QDoubleSpinBox(self)
        self._significance_input.setRange(0.1, 20.0)
        self._significance_input.setDecimals(2)
        self._significance_input.setSingleStep(0.5)
        self._significance_input.setValue(5.0)
        self._significance_input.setMaximumWidth(96)
        layout.addWidget(self._significance_input)
        buttons = [
            ("Resumo estatístico", self._open_summary_popup, QStyle.SP_FileDialogInfoView),
            ("Distribuição", self._open_distribution_popup, QStyle.SP_FileDialogListView),
            ("Entradas processadas", self._open_inputs_popup, QStyle.SP_FileIcon),
            ("Correspondências", self._open_matches_popup, QStyle.SP_FileDialogDetailedView),
            ("Malha biométrica", self._open_preview_popup, QStyle.SP_DesktopIcon),
        ]
        self._result_buttons: list[QPushButton] = []
        for label, handler, icon in buttons:
            button = QPushButton(label)
            apply_standard_icon(self, button, icon)
            button.setEnabled(False)
            button.clicked.connect(handler)
            layout.addWidget(button)
            self._result_buttons.append(button)
        likelihood_button = QPushButton("Razão de verossimilhança")
        apply_standard_icon(self, likelihood_button, QStyle.SP_FileDialogInfoView)
        likelihood_button.setEnabled(False)
        likelihood_button.clicked.connect(self._open_likelihood_ratio_popup)
        layout.addWidget(likelihood_button)
        self._result_buttons.append(likelihood_button)
        self._save_calibration_model_button = QPushButton("Salvar modelo LR")
        apply_standard_icon(self, self._save_calibration_model_button, QStyle.SP_DialogSaveButton)
        self._save_calibration_model_button.setEnabled(False)
        self._save_calibration_model_button.clicked.connect(self._save_calibration_model)
        layout.addWidget(self._save_calibration_model_button)
        layout.addStretch(1)
        return group

    def _build_set_group(self, title: str, set_label: str) -> QGroupBox:
        group = QGroupBox(title, self)
        layout = QVBoxLayout(group)
        list_widget = QListWidget(self)
        list_widget.setSelectionMode(QAbstractItemView.ExtendedSelection)
        list_widget.setAlternatingRowColors(True)
        layout.addWidget(list_widget, stretch=1)

        controls = QHBoxLayout()
        add_button = QPushButton("Adicionar mídias")
        apply_standard_icon(self, add_button, QStyle.SP_DialogOpenButton)
        add_button.clicked.connect(lambda: self._add_images(set_label))
        controls.addWidget(add_button)
        remove_button = QPushButton("Remover selecionadas")
        apply_standard_icon(self, remove_button, QStyle.SP_TrashIcon)
        remove_button.clicked.connect(lambda: self._remove_selected_images(set_label))
        controls.addWidget(remove_button)
        clear_button = QPushButton("Limpar")
        apply_standard_icon(self, clear_button, QStyle.SP_LineEditClearButton)
        clear_button.clicked.connect(lambda: self._clear_images(set_label))
        controls.addWidget(clear_button)
        layout.addLayout(controls)

        count_label = QLabel("0 arquivo(s) selecionado(s)")
        count_label.setObjectName("SectionHint")
        layout.addWidget(count_label)

        if set_label == "A":
            self._set_a_list = list_widget
            self._set_a_count_label = count_label
            self._set_a_add_button = add_button
            self._set_a_remove_button = remove_button
            self._set_a_clear_button = clear_button
        else:
            self._set_b_list = list_widget
            self._set_b_count_label = count_label
            self._set_b_add_button = add_button
            self._set_b_remove_button = remove_button
            self._set_b_clear_button = clear_button
        return group

    def _list_widget_for_set(self, set_label: str) -> QListWidget:
        return self._set_a_list if set_label == "A" else self._set_b_list

    def _count_label_for_set(self, set_label: str) -> QLabel:
        return self._set_a_count_label if set_label == "A" else self._set_b_count_label

    def _group_label(self, set_label: str) -> str:
        if set_label == "A":
            return "Padrão"
        if set_label == "B":
            return "Questionado"
        if set_label == "CAL":
            return "Calibração LR"
        return set_label

    def _add_images(self, set_label: str) -> None:
        selected_paths, _ = QFileDialog.getOpenFileNames(
            self,
            f"Selecionar mídias para {self._group_label(set_label)}",
            str(self._initial_input_directory),
            self._media_file_filter(),
        )
        if not selected_paths:
            return
        list_widget = self._list_widget_for_set(set_label)
        existing = {list_widget.item(index).text() for index in range(list_widget.count())}
        for path in selected_paths:
            if path not in existing:
                list_widget.addItem(QListWidgetItem(path))
        self._initial_input_directory = Path(selected_paths[0]).resolve().parent
        self._update_set_count(set_label)

    def _remove_selected_images(self, set_label: str) -> None:
        list_widget = self._list_widget_for_set(set_label)
        for item in list_widget.selectedItems():
            list_widget.takeItem(list_widget.row(item))
        self._update_set_count(set_label)

    def _clear_images(self, set_label: str) -> None:
        self._list_widget_for_set(set_label).clear()
        self._update_set_count(set_label)

    def _update_set_count(self, set_label: str) -> None:
        list_widget = self._list_widget_for_set(set_label)
        self._count_label_for_set(set_label).setText(f"{list_widget.count()} arquivo(s) selecionado(s)")

    def _select_work_directory(self) -> None:
        selected = QFileDialog.getExistingDirectory(
            self,
            "Selecionar diretório de trabalho",
            self._work_directory_input.text().strip() or str(self._initial_work_directory),
        )
        if selected:
            self._work_directory_input.setText(selected)

    def _select_calibration_directory(self) -> None:
        selected = QFileDialog.getExistingDirectory(
            self,
            "Selecionar base de calibração LR",
            self._calibration_directory_input.text().strip() or str(self._initial_input_directory),
        )
        if selected:
            self._calibration_directory_input.setText(selected)

    def _select_calibration_model(self) -> None:
        selected, _ = QFileDialog.getOpenFileName(
            self,
            "Selecionar modelo de calibração LR",
            self._calibration_model_input.text().strip() or str(self._initial_input_directory),
            "Modelo de calibração LR (*.json);;Arquivos JSON (*.json);;Todos os arquivos (*)",
        )
        if selected:
            self._calibration_model_input.setText(selected)

    def _selected_paths(self, set_label: str) -> list[Path]:
        list_widget = self._list_widget_for_set(set_label)
        return [Path(list_widget.item(index).text()) for index in range(list_widget.count())]

    def _start_comparison(self) -> None:
        set_a_paths = self._selected_paths("A")
        set_b_paths = self._selected_paths("B")
        if not set_a_paths:
            QMessageBox.warning(self, "Padrão vazio", "Selecione ao menos uma mídia (imagem ou vídeo) em Padrão.")
            return
        if not set_b_paths:
            QMessageBox.warning(self, "Questionado vazio", "Selecione ao menos uma mídia (imagem ou vídeo) em Questionado.")
            return

        work_directory_text = self._work_directory_input.text().strip()
        work_directory = Path(work_directory_text).resolve() if work_directory_text else set_a_paths[0].parent.resolve()
        work_directory.mkdir(parents=True, exist_ok=True)
        self._work_directory_input.setText(str(work_directory))
        calibration_directory_text = self._calibration_directory_input.text().strip()
        calibration_root = Path(calibration_directory_text).resolve() if calibration_directory_text else None
        calibration_model_text = self._calibration_model_input.text().strip()
        calibration_model_path = Path(calibration_model_text).resolve() if calibration_model_text else None

        self._latest_result = None
        self._entry_by_id.clear()
        self._reset_after_new_run()
        self._set_activity_state("running", "Inicializando comparação entre os conjuntos selecionados.")
        self._append_log(
            "[Monitor] O texto exibido nesta janela também é gravado em run.log e events.jsonl da execução."
        )
        self._status_label.setText("Inicializando comparação entre os conjuntos selecionados.")
        self._append_log(f"Padrão: {len(set_a_paths)} mídia(s)")
        self._append_log(f"Questionado: {len(set_b_paths)} mídia(s)")
        self._append_log(f"Diretório de trabalho: {work_directory}")

        if calibration_root is not None:
            self._append_log(f"Base de calibração LR: {calibration_root}")
        if calibration_model_path is not None:
            self._append_log(f"Modelo de calibração LR: {calibration_model_path}")
            if calibration_root is not None:
                self._append_log("Modelo salvo informado: a base rotulada será ignorada nesta execução.")

        self._thread = QThread(self)
        self._worker = FaceSetComparisonWorker(
            self._service_factory(),
            set_a_paths,
            set_b_paths,
            work_directory=work_directory,
            calibration_root=calibration_root,
            calibration_model_path=calibration_model_path,
        )
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress_changed.connect(self._on_progress_changed)
        self._worker.log_message.connect(self._append_log)
        self._worker.completed.connect(self._on_completed)
        self._worker.failed.connect(self._on_failed)
        self._worker.completed.connect(self._thread.quit)
        self._worker.failed.connect(self._thread.quit)
        self._thread.finished.connect(self._cleanup_thread)
        self._set_running_state(True)
        self._thread.start()

    def _reset_after_new_run(self) -> None:
        self._log_view.clear()
        self._progress_bar.setValue(0)
        self._quick_summary_label.setText("Os resultados detalhados ficarão disponíveis nos botões acima.")
        self._set_result_buttons_enabled(False)
        self._save_calibration_model_button.setEnabled(False)

    def _on_progress_changed(self, value: int, message: str) -> None:
        self._progress_bar.setValue(value)
        self._status_label.setText(message)
        self._mark_activity(message)

    def _on_completed(self, result: object) -> None:
        if not isinstance(result, FaceSetComparisonResult):
            self._on_failed("Resultado inválido da comparação.")
            return
        self._latest_result = result
        self._entry_by_id = {
            entry.entry_id: entry
            for entry in [*result.set_a_faces, *result.set_b_faces]
        }
        self._progress_bar.setValue(100)
        self._status_label.setText("Comparação concluída.")
        self._set_activity_state(
            "completed",
            "Comparação concluída. Logs e artefatos foram gravados na pasta da execução.",
        )
        self._append_log(f"Exportação disponível em: {result.export_directory}")
        self._set_result_buttons_enabled(True)
        self._save_calibration_model_button.setEnabled(result.calibration is not None)
        if result.calibration is not None:
            if result.calibration.model_path is not None:
                self._calibration_model_input.setText(str(result.calibration.model_path))
            self._append_log(
                f"Modelo LR exportado em: {result.export_directory / 'face_set_comparison_calibration_model.json'}"
            )
        summary = result.summary
        support, _ = self._has_statistical_support([item.similarity for item in result.matches])
        lr_state = (
            "disponível"
            if summary.likelihood_ratio_calibrated
            else "não calibrada"
            if result.calibration is not None
            else "desabilitada"
        )
        self._quick_summary_label.setText(
            (
                f"Comparações: {summary.total_pair_comparisons} | "
                f"Atribuições: {summary.assignment_matches} | "
                f"Candidatas: {summary.candidate_matches} | "
                f"Inferência estatística: {'disponível' if support else 'indisponível'}"
            )
        )

        self._quick_summary_label.setText(f"{self._quick_summary_label.text()} | LR: {lr_state}")

    def _on_failed(self, error: str) -> None:
        self._set_activity_state(
            "failed",
            "Falha na comparação. Consulte o log desta janela para identificar a etapa interrompida.",
        )
        self._append_log("[ERRO] Comparação interrompida.")
        self._append_log(error)
        self._status_label.setText("Falha na comparação.")
        short_error = error.splitlines()[0] if error else "Falha desconhecida."
        QMessageBox.critical(
            self,
            "Erro",
            f"{short_error}\n\nConsulte o log exibido na janela para detalhes.",
        )

    def _set_result_buttons_enabled(self, enabled: bool) -> None:
        for button in self._result_buttons:
            button.setEnabled(enabled)
        self._export_button.setEnabled(enabled)
        self._open_export_button.setEnabled(enabled)

    def _require_result(self, title: str) -> FaceSetComparisonResult | None:
        if self._latest_result is not None:
            return self._latest_result
        QMessageBox.information(self, title, "Execute a comparação antes de abrir este resultado.")
        return None

    def _create_popup(self, title: str, width: int = 980, height: int = 720) -> QDialog:
        dialog = QDialog(self)
        dialog.setWindowTitle(title)
        dialog.setWindowFlag(Qt.WindowMaximizeButtonHint, True)
        dialog.resize(width, height)
        dialog.setMinimumSize(760, 520)
        dialog.setStyleSheet(self.styleSheet())
        return dialog

    def _create_metric_card(self, title: str, value: str, detail: str | None = None) -> QFrame:
        card = QFrame(self)
        card.setObjectName("MetricCard")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(4)
        card.setMinimumHeight(82 if detail else 74)
        title_label = QLabel(title)
        title_label.setObjectName("MetricTitle")
        title_label.setWordWrap(True)
        value_label = QLabel(value)
        value_label.setObjectName("MetricValue")
        value_label.setWordWrap(True)
        layout.addWidget(title_label)
        layout.addWidget(value_label)
        if detail:
            detail_label = QLabel(detail)
            detail_label.setObjectName("MetricDetail")
            detail_label.setWordWrap(True)
            layout.addWidget(detail_label)
        return card

    def _bootstrap_alpha(self) -> float:
        return self._significance_input.value() / 100.0

    def _bootstrap_resamples(self) -> int:
        return int(self._bootstrap_resamples_input.value())

    def _quality_group_comparison_test(self, result: FaceSetComparisonResult) -> _GroupComparisonTestResult:
        left_values = [
            float(entry.quality_score)
            for entry in result.set_a_faces
            if entry.quality_score is not None
        ]
        right_values = [
            float(entry.quality_score)
            for entry in result.set_b_faces
            if entry.quality_score is not None
        ]
        return _mann_whitney_group_comparison(
            left_values,
            right_values,
            alpha=self._bootstrap_alpha(),
            metric_label="qualidade facial",
            left_label="Padrão",
            right_label="Questionado",
        )

    def _group_comparison_summary_lines(
        self,
        test_result: _GroupComparisonTestResult,
        *,
        significance_percent: float,
    ) -> list[str]:
        if not test_result.available:
            return [
                "Teste não paramétrico entre Padrão e Questionado:",
                test_result.note or "Teste U de Mann-Whitney indisponível.",
            ]

        direction = _group_comparison_direction_text(test_result)
        significance_label = _group_comparison_significance_text(
            test_result,
            significance_percent=significance_percent,
        )
        return [
            "Teste não paramétrico entre Padrão e Questionado:",
            (
                "U de Mann-Whitney bilateral sobre a distribuição de qualidade facial "
                "das faces selecionadas em Padrão e Questionado."
            ),
            (
                f"n: {test_result.left_label} {test_result.left_count} | "
                f"{test_result.right_label} {test_result.right_count}"
            ),
            (
                f"Medianas: {test_result.left_label} {self._format_optional_float(test_result.left_median)} | "
                f"{test_result.right_label} {self._format_optional_float(test_result.right_median)}"
            ),
            f"U: {self._format_optional_float(test_result.u_statistic)}",
            f"p-valor bilateral: {self._format_p_value(test_result.p_value)}",
            (
                "Correlação bisserial de postos: "
                f"{self._format_optional_float(test_result.rank_biserial)}"
            ),
            (
                f"Probabilidade de superioridade comum ({test_result.left_label} > {test_result.right_label}): "
                f"{self._format_optional_float(test_result.common_language_effect)}"
            ),
            f"Interpretação: {significance_label}; {direction}.",
        ]

    def _has_statistical_support(self, values: list[float], minimum_count: int = 5) -> tuple[bool, str | None]:
        if len(values) < minimum_count:
            return False, f"Amostra insuficiente para inferência: são necessárias ao menos {minimum_count} repetições."
        if len({round(value, 8) for value in values}) < 2:
            return False, "Variabilidade insuficiente: as similaridades são praticamente constantes."
        return True, None

    def _bootstrap_mean_interval(
        self,
        values: list[float],
        *,
        resamples: int,
        alpha: float,
    ) -> tuple[float, float]:
        array = np.asarray(values, dtype=np.float64)
        rng = np.random.default_rng(20260409)
        means = np.empty(resamples, dtype=np.float64)
        sample_size = len(array)
        for index in range(resamples):
            sample = rng.choice(array, size=sample_size, replace=True)
            means[index] = float(np.mean(sample))
        low_q, high_q = np.quantile(means, [alpha / 2.0, 1.0 - (alpha / 2.0)])
        return float(low_q), float(high_q)

    def _kde_curve(
        self,
        values: list[float],
        *,
        lower: float,
        upper: float,
        points: int = 256,
        bandwidth_scale: float = 1.0,
        density_method: str | None = None,
        uniform_floor_weight: float = 0.0,
        min_density: float = 0.0,
    ) -> tuple[tuple[float, ...], tuple[float, ...]]:
        model = fit_score_density_model(
            values,
            method=density_method or self._config.likelihood_ratio.density_estimator,
            bandwidth_scale=bandwidth_scale,
        )
        return model.curve(
            lower=lower,
            upper=upper,
            points=points,
            uniform_floor_weight=uniform_floor_weight,
            min_density=min_density,
        )

    def _match_groups(self, result: FaceSetComparisonResult) -> dict[str, list[float]]:
        groups: dict[str, list[float]] = {
            "assignment": [],
            "candidate": [],
            "below_threshold": [],
        }
        for match in result.matches:
            groups.setdefault(match.classification, []).append(match.similarity)
        return groups

    def _distribution_analysis(
        self,
        result: FaceSetComparisonResult,
    ) -> tuple[list[_DistributionSeries], tuple[bool, str | None], dict[str, float | None]]:
        groups = self._match_groups(result)
        all_values = [item.similarity for item in result.matches]
        support, note = self._has_statistical_support(all_values)
        lower, upper = _expanded_score_range(
            all_values,
            observed_score=result.summary.best_similarity,
            minimum_span=0.2,
        )
        resamples = self._bootstrap_resamples()
        alpha = self._bootstrap_alpha()
        overall_stats: dict[str, float | None] = {
            "mean": None,
            "median": None,
            "q1": None,
            "q3": None,
            "ci_low": None,
            "ci_high": None,
        }
        if support:
            overall_stats["mean"] = float(np.mean(all_values))
            overall_stats["median"] = float(np.median(all_values))
            quartiles = np.quantile(np.asarray(all_values, dtype=np.float64), [0.25, 0.75], method="linear")
            overall_stats["q1"] = float(quartiles[0])
            overall_stats["q3"] = float(quartiles[1])
            ci_low, ci_high = self._bootstrap_mean_interval(all_values, resamples=resamples, alpha=alpha)
            overall_stats["ci_low"] = ci_low
            overall_stats["ci_high"] = ci_high

        palettes = {
            "assignment": (_distribution_series_label("assignment"), "#0f766e"),
            "candidate": (_distribution_series_label("candidate"), "#b45309"),
            "below_threshold": (_distribution_series_label("below_threshold"), "#7c3aed"),
        }
        series_list: list[_DistributionSeries] = []
        for classification, values in groups.items():
            label, color = palettes.get(classification, (classification, "#334155"))
            class_support, class_note = self._has_statistical_support(values)
            if class_support:
                kde_x, kde_y = self._kde_curve(
                    values,
                    lower=lower,
                    upper=upper,
                    density_method=self._config.likelihood_ratio.density_estimator,
                    bandwidth_scale=self._config.likelihood_ratio.kde_bandwidth_scale,
                )
                mean_value = float(np.mean(values))
                median_value = float(np.median(values))
                quartiles = np.quantile(np.asarray(values, dtype=np.float64), [0.25, 0.75], method="linear")
                ci_low, ci_high = self._bootstrap_mean_interval(values, resamples=resamples, alpha=alpha)
                series_list.append(
                    _DistributionSeries(
                        label=label,
                        classification=classification,
                        color=color,
                        values=tuple(values),
                        sufficient=True,
                        kde_x=kde_x,
                        kde_y=kde_y,
                        mean=mean_value,
                        median=median_value,
                        q1=float(quartiles[0]),
                        q3=float(quartiles[1]),
                        ci_low=ci_low,
                        ci_high=ci_high,
                    )
                )
            else:
                series_list.append(
                    _DistributionSeries(
                        label=label,
                        classification=classification,
                        color=color,
                        values=tuple(values),
                        sufficient=False,
                        note=class_note or "Amostra insuficiente.",
                    )
                )
        return series_list, (support, note), overall_stats

    def _open_summary_popup(self) -> None:
        result = self._require_result("Resumo estatístico")
        if result is None:
            return
        summary = result.summary
        series_list, (support, note), overall_stats = self._distribution_analysis(result)
        significance = self._significance_input.value()
        confidence_level = max(0.0, 100.0 - significance)
        group_test = self._quality_group_comparison_test(result)
        below_threshold_count = max(
            0,
            summary.total_pair_comparisons - summary.assignment_matches - summary.candidate_matches,
        )
        dialog = self._create_popup("Resumo estatístico", 1080, 860)
        layout = QVBoxLayout(dialog)
        layout.setSpacing(10)

        metrics = QGridLayout()
        metrics.setHorizontalSpacing(12)
        metrics.setVerticalSpacing(12)
        best_zone = _distribution_zone_short_label(
            summary.best_similarity,
            candidate_threshold=summary.candidate_threshold,
            assignment_threshold=summary.assignment_threshold,
        )
        cards = [
            (
                "Pares comparados",
                str(summary.total_pair_comparisons),
                "combinações Padrão x Questionado efetivamente avaliadas",
            ),
            (
                "Faces aproveitadas",
                f"Padrão {summary.set_a_selected_faces} | Questionado {summary.set_b_selected_faces}",
                "selecionadas para gerar os pares e sustentar o teste de qualidade",
            ),
            (
                "Maior similaridade observada",
                self._format_optional_float(summary.best_similarity),
                best_zone,
            ),
            (
                "Pares por faixa decisória",
                (
                    f"atribuição {summary.assignment_matches} | "
                    f"candidata {summary.candidate_matches} | "
                    f"abaixo do limiar {below_threshold_count}"
                ),
                "classificação dos pares pelos limiares adotados",
            ),
            (
                "Limiar candidata / atribuição",
                f"{summary.candidate_threshold:.4f} / {summary.assignment_threshold:.4f}",
                "limiares decisórios desta execução",
            ),
            (
                "Média dos escores",
                self._format_optional_float(overall_stats["mean"]),
                "todos os pares Padrão x Questionado",
            ),
            (
                "Mediana dos escores",
                self._format_optional_float(overall_stats["median"]),
                "valor central da distribuição observada",
            ),
            (
                "Faixa interquartil (Q1-Q3)",
                f"{self._format_optional_float(overall_stats['q1'])} / {self._format_optional_float(overall_stats['q3'])}",
                "metade central dos escores observados",
            ),
            (
                f"IC bootstrap da média ({confidence_level:.2f}%)",
                f"{self._format_optional_float(overall_stats['ci_low'])} .. {self._format_optional_float(overall_stats['ci_high'])}",
                "intervalo de confiança da média geral",
            ),
        ]
        for index, (title, value, detail) in enumerate(cards):
            metrics.addWidget(self._create_metric_card(title, value, detail), index // 3, index % 3)
        layout.addLayout(metrics)

        subtitle = QLabel(
            (
                "Este painel resume os escores globais dos pares Padrão x Questionado e, em separado, "
                "a comparação da qualidade facial das faces selecionadas em Padrão e Questionado. "
                f"O nível de significância adotado ({significance:.2f}%) é explicado no texto técnico abaixo."
            ),
            dialog,
        )
        subtitle.setWordWrap(True)
        subtitle.setObjectName("SectionHint")
        layout.addWidget(subtitle)

        browser = QTextBrowser(dialog)
        browser.setOpenExternalLinks(True)
        browser.setHtml(
            _summary_popup_html(
                summary,
                overall_stats,
                group_test,
                series_list,
                significance_percent=significance,
                bootstrap_resamples=self._bootstrap_resamples(),
                procedure_details=result.procedure_details,
                support_note=None if support else note,
            )
        )
        browser.setMinimumHeight(320)
        layout.addWidget(browser, stretch=1)
        actions = QHBoxLayout()
        actions.addStretch(1)
        help_button = QPushButton("Ajuda", dialog)
        apply_standard_icon(self, help_button, QStyle.SP_DialogHelpButton)
        help_button.clicked.connect(self._open_summary_help_popup)
        actions.addWidget(help_button)
        close_button = QPushButton("Fechar", dialog)
        apply_standard_icon(self, close_button, QStyle.SP_DialogCloseButton)
        close_button.clicked.connect(dialog.accept)
        actions.addWidget(close_button)
        layout.addLayout(actions)
        dialog.exec()

    def _open_summary_help_popup(self) -> None:
        dialog = self._create_popup("Ajuda do resumo estatístico", 1020, 780)
        layout = QVBoxLayout(dialog)
        intro = QLabel(
            (
                "Este painel explica como interpretar os cards numéricos, o IC bootstrap da média "
                "e o teste de comparação da qualidade facial entre Padrão e Questionado."
            ),
            dialog,
        )
        intro.setWordWrap(True)
        intro.setObjectName("SectionHint")
        layout.addWidget(intro)
        browser = QTextBrowser(dialog)
        browser.setHtml(_summary_help_html())
        layout.addWidget(browser, stretch=1)
        actions = QHBoxLayout()
        actions.addStretch(1)
        close_button = QPushButton("Fechar", dialog)
        apply_standard_icon(self, close_button, QStyle.SP_DialogCloseButton)
        close_button.clicked.connect(dialog.accept)
        actions.addWidget(close_button)
        layout.addLayout(actions)
        dialog.exec()

    def _open_help_popup(self) -> None:
        dialog = self._create_popup("Ajuda da comparação entre grupos faciais", 1120, 860)
        layout = QVBoxLayout(dialog)
        intro = QLabel(
            (
                "Este painel reúne orientação operacional, descrição técnica do pipeline e critérios "
                "para leitura dos resultados produzidos nesta janela."
            ),
            dialog,
        )
        intro.setWordWrap(True)
        intro.setObjectName("SectionHint")
        layout.addWidget(intro)
        browser = QTextBrowser(dialog)
        browser.setOpenExternalLinks(True)
        browser.setHtml(self._comparison_help_html())
        layout.addWidget(browser, stretch=1)
        dialog.exec()

    def _comparison_help_html(self) -> str:
        return build_face_set_comparison_help_html(self._config)

    def _open_distribution_help_popup(self) -> None:
        dialog = self._create_popup('Ajuda da distribuição de similaridades', 1020, 780)
        layout = QVBoxLayout(dialog)
        intro = QLabel(
            (
                'Este painel explica como ler as curvas, as linhas médias, os limiares de decisão '
                'e o posicionamento do melhor escore observado.'
            ),
            dialog,
        )
        intro.setWordWrap(True)
        intro.setObjectName('SectionHint')
        layout.addWidget(intro)
        browser = QTextBrowser(dialog)
        browser.setHtml(_distribution_help_html())
        layout.addWidget(browser, stretch=1)
        actions = QHBoxLayout()
        actions.addStretch(1)
        close_button = QPushButton('Fechar', dialog)
        apply_standard_icon(self, close_button, QStyle.SP_DialogCloseButton)
        close_button.clicked.connect(dialog.accept)
        actions.addWidget(close_button)
        layout.addLayout(actions)
        dialog.exec()

    def _open_distribution_popup(self) -> None:
        result = self._require_result('Distribuição de similaridades')
        if result is None:
            return
        summary = result.summary
        series_list, (support, note), overall_stats = self._distribution_analysis(result)
        significance = self._significance_input.value()
        confidence_level = max(0.0, 100.0 - significance)
        dialog = self._create_popup('Distribuição de similaridades', 1180, 880)
        layout = QVBoxLayout(dialog)
        layout.setSpacing(10)

        def _series_by_classification(classification: str) -> _DistributionSeries | None:
            return next((series for series in series_list if series.classification == classification), None)

        assignment_series = _series_by_classification('assignment')
        candidate_series = _series_by_classification('candidate')
        below_series = _series_by_classification('below_threshold')
        best_zone_short = _distribution_zone_short_label(
            summary.best_similarity,
            candidate_threshold=summary.candidate_threshold,
            assignment_threshold=summary.assignment_threshold,
        )

        metrics = QGridLayout()
        metrics.setHorizontalSpacing(12)
        metrics.setVerticalSpacing(12)
        cards = [
            ('Comparações', str(summary.total_pair_comparisons), 'pares Padrão x Questionado comparados'),
            ('Melhor escore', self._format_optional_float(summary.best_similarity), best_zone_short),
            ('Média geral', self._format_optional_float(overall_stats['mean']), 'todos os pares Padrão x Questionado'),
            (
                f'IC bootstrap ({confidence_level:.2f}%)',
                f"{self._format_optional_float(overall_stats['ci_low'])} .. {self._format_optional_float(overall_stats['ci_high'])}",
                'intervalo de confiança da média geral',
            ),
            (
                _distribution_card_title('assignment'),
                self._format_optional_float(assignment_series.mean if assignment_series is not None else None),
                'pares classificados na faixa de atribuição',
            ),
            (
                _distribution_card_title('candidate'),
                self._format_optional_float(candidate_series.mean if candidate_series is not None else None),
                'pares classificados na faixa candidata',
            ),
            (
                _distribution_card_title('below_threshold'),
                self._format_optional_float(below_series.mean if below_series is not None else None),
                'pares abaixo do limiar candidato',
            ),
            ('Limiar cand./atrib.', f'{summary.candidate_threshold:.4f} / {summary.assignment_threshold:.4f}', 'faixa candidata / faixa de atribuição'),
        ]
        for index, (title, value, detail) in enumerate(cards):
            metrics.addWidget(self._create_metric_card(title, value, detail), index // 4, index % 4)
        layout.addLayout(metrics)

        subtitle = QLabel(
            "Todas as curvas abaixo representam pares Padrão x Questionado; as cores separam apenas as faixas decisórias.",
            dialog,
        )
        subtitle.setWordWrap(True)
        subtitle.setObjectName('SectionHint')
        layout.addWidget(subtitle)

        if support:
            widget = SimilarityDistributionWidget(dialog)
            widget.set_distribution(
                series_list,
                candidate_threshold=summary.candidate_threshold,
                assignment_threshold=summary.assignment_threshold,
                observed_score=summary.best_similarity,
                mean_value=overall_stats['mean'],
                ci_low=overall_stats['ci_low'],
                ci_high=overall_stats['ci_high'],
            )
            layout.addWidget(widget, stretch=2)
        else:
            info = QTextBrowser(dialog)
            info.setHtml(
                '<p><b>A curva de densidade não será exibida para esta comparação.</b></p>'
                f'<p>{escape(note or "Amostra insuficiente.")}</p>'
                f'<p>Comparações disponíveis: <b>{summary.total_pair_comparisons}</b></p>'
            )
            info.setMaximumHeight(160)
            layout.addWidget(info)

        caption = QTextBrowser(dialog)
        caption.setOpenExternalLinks(True)
        caption.setHtml(
            _distribution_popup_html(
                summary,
                series_list,
                overall_stats,
                significance_percent=significance,
                bootstrap_resamples=self._bootstrap_resamples(),
            )
        )
        caption.setMinimumHeight(240)
        layout.addWidget(caption, stretch=1)

        actions = QHBoxLayout()
        actions.addStretch(1)
        help_button = QPushButton('Ajuda', dialog)
        apply_standard_icon(self, help_button, QStyle.SP_DialogHelpButton)
        help_button.clicked.connect(self._open_distribution_help_popup)
        actions.addWidget(help_button)
        close_button = QPushButton('Fechar', dialog)
        apply_standard_icon(self, close_button, QStyle.SP_DialogCloseButton)
        close_button.clicked.connect(dialog.accept)
        actions.addWidget(close_button)
        layout.addLayout(actions)
        dialog.exec()

    def _likelihood_ratio_series(
        self,
        result: FaceSetComparisonResult,
    ) -> tuple[list[_DistributionSeries], str | None]:
        calibration = result.calibration
        if calibration is None:
            return [], "Nenhuma base de calibração LR foi informada para esta execução."
        summary = calibration.summary
        if not summary.support_ready:
            return [], summary.support_note or "A base de calibração não teve suporte suficiente."
        settings = calibration.settings_snapshot or self._config.likelihood_ratio

        lower, upper = _expanded_score_range(
            [*calibration.genuine_scores, *calibration.impostor_scores],
            observed_score=result.summary.best_similarity,
            minimum_span=0.2,
        )
        same_x, same_y = self._kde_curve(
            calibration.genuine_scores,
            lower=lower,
            upper=upper,
            bandwidth_scale=settings.kde_bandwidth_scale,
            density_method=settings.density_estimator,
            uniform_floor_weight=settings.kde_uniform_floor_weight,
            min_density=settings.kde_min_density,
        )
        diff_x, diff_y = self._kde_curve(
            calibration.impostor_scores,
            lower=lower,
            upper=upper,
            bandwidth_scale=settings.kde_bandwidth_scale,
            density_method=settings.density_estimator,
            uniform_floor_weight=settings.kde_uniform_floor_weight,
            min_density=settings.kde_min_density,
        )
        return (
            [
                _DistributionSeries(
                    label="Padrão/Questionado, mesma origem (H1)",
                    classification="same_source",
                    color="#0f766e",
                    values=tuple(calibration.genuine_scores),
                    sufficient=True,
                    kde_x=same_x,
                    kde_y=same_y,
                    mean=float(np.mean(calibration.genuine_scores)),
                    median=float(np.median(calibration.genuine_scores)),
                ),
                _DistributionSeries(
                    label="Padrão/Questionado, origem distinta (H2)",
                    classification="different_source",
                    color="#b91c1c",
                    values=tuple(calibration.impostor_scores),
                    sufficient=True,
                    kde_x=diff_x,
                    kde_y=diff_y,
                    mean=float(np.mean(calibration.impostor_scores)),
                    median=float(np.median(calibration.impostor_scores)),
                ),
            ],
            None,
        )

    def _open_likelihood_ratio_popup(self) -> None:
        result = self._require_result("Razão de verossimilhança")
        if result is None:
            return
        dialog = self._create_popup("Razão de verossimilhança", 1120, 820)
        layout = QVBoxLayout(dialog)
        calibration = result.calibration
        if calibration is None:
            info = QTextBrowser(dialog)
            info.setPlainText(
                "\n".join(
                    [
                        "Esta execução não foi calibrada por razão de verossimilhança.",
                        "Para habilitar a LR, selecione uma base rotulada com uma subpasta por identidade ou carregue um modelo salvo antes de comparar os conjuntos.",
                    ]
                )
            )
            layout.addWidget(info)
            dialog.exec()
            return

        calibration_summary = calibration.summary
        metrics = QGridLayout()
        cards = [
            ("Identidades", str(calibration_summary.identity_count)),
            ("Faces na base", str(calibration_summary.selected_faces)),
            (
                "Scores Padrão/Questionado (mesma origem)",
                f"{calibration_summary.genuine_score_count}/{calibration_summary.genuine_pair_total}",
            ),
            (
                "Scores Padrão/Questionado (origem distinta)",
                f"{calibration_summary.impostor_score_count}/{calibration_summary.impostor_pair_total}",
            ),
            ("Ajuste de densidade", "pronto" if calibration_summary.support_ready else "indisponível"),
            ("Pares calibrados", str(result.summary.calibrated_matches)),
            ("Média log10(LR)", self._format_optional_float(result.summary.mean_log10_likelihood_ratio)),
            ("Mediana log10(LR)", self._format_optional_float(result.summary.median_log10_likelihood_ratio)),
        ]
        cards[2] = ("Scores Padrão/Questionado (mesma origem)", cards[2][1])
        cards[3] = ("Scores Padrão/Questionado (origem distinta)", cards[3][1])
        for index, (title, value) in enumerate(cards):
            metrics.addWidget(self._create_metric_card(title, value), index // 4, index % 4)
        layout.addLayout(metrics)

        info = QTextBrowser(dialog)
        info_lines = [
            f"Diretório da base: {calibration_summary.dataset_root}",
            f"Observação: {calibration_summary.support_note or 'Ajuste calibrado com suporte suficiente.'}",
        ]
        if calibration.model_path is not None:
            info_lines.append(
                (
                    f"Modelo carregado de: {calibration.model_path}"
                    if calibration.loaded_from_model
                    else f"Modelo salvo em: {calibration.model_path}"
                )
            )
        if calibration_summary.smoothing_note:
            info_lines.append(f"Estabilização: {calibration_summary.smoothing_note}")
        if result.matches:
            info_lines.append(
                "Linha tracejada azul: acompanha a linha selecionada na tabela de confrontos "
                "e, ao abrir esta janela, inicia no primeiro item do ranking."
            )
            info_lines.append(
                "Barras translúcidas: histograma dos scores brutos observados; "
                "curvas preenchidas: densidade suavizada usada na calibração LR."
            )
            info_lines.append(
                "Leitura do LR no gráfico: no score da linha tracejada, o sistema calcula "
                "LR = altura da curva H1 / altura da curva H2."
            )
        if calibration.settings_snapshot is not None:
            settings = calibration.settings_snapshot
            info_lines.append(
                "Parâmetros LR: "
                f"amostra_max={settings.max_scores_per_distribution} | "
                f"min_identidades={settings.minimum_identities_with_faces} | "
                f"min_mesma_origem={settings.minimum_same_source_scores} | "
                f"min_origem_distinta={settings.minimum_different_source_scores} | "
                f"min_distintos={settings.minimum_unique_scores_per_distribution}"
            )
            info_lines.append(
                "Parâmetros do estimador: "
                f"metodo={score_density_method_label(settings.density_estimator)} | "
                f"banda_x={settings.kde_bandwidth_scale:.3f} | "
                f"piso_uniforme={settings.kde_uniform_floor_weight:.4%} | "
                f"densidade_minima={settings.kde_min_density:.1e}"
            )
        info.setPlainText("\n".join(info_lines))
        info.setMaximumHeight(160)
        layout.addWidget(info)

        series, note = self._likelihood_ratio_series(result)
        if series:
            density_widget = LikelihoodRatioDensityWidget(dialog)
            density_widget.set_series(series, observed_score=result.summary.best_similarity)
            layout.addWidget(density_widget, stretch=1)
            selection_label = QLabel(dialog)
            selection_label.setObjectName("SectionHint")
            selection_label.setTextFormat(Qt.RichText)
            selection_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
            selection_label.setWordWrap(True)
            layout.addWidget(selection_label)
        else:
            density_info = QTextBrowser(dialog)
            density_info.setPlainText(note or "A densidade calibrada não está disponível.")
            layout.addWidget(density_info)
            selection_label = None

        table = self._create_matches_table(dialog)
        self._populate_matches_table(table, result)
        layout.addWidget(table, stretch=1)
        if series:
            table.itemSelectionChanged.connect(
                lambda: self._sync_likelihood_ratio_selection(
                    result=result,
                    table=table,
                    density_widget=density_widget,
                    selection_label=selection_label,
                    series=series,
                )
            )
            if result.matches:
                table.selectRow(0)
            self._sync_likelihood_ratio_selection(
                result=result,
                table=table,
                density_widget=density_widget,
                selection_label=selection_label,
                series=series,
            )
        dialog.exec()

    def _open_inputs_popup(self) -> None:
        result = self._require_result("Entradas processadas")
        if result is None:
            return
        dialog = self._create_popup("Entradas processadas", 1120, 760)
        layout = QVBoxLayout(dialog)
        table = self._create_inputs_table(dialog)
        self._populate_inputs_table(table, [*result.set_a_inputs, *result.set_b_inputs])
        layout.addWidget(table)
        dialog.exec()

    def _open_matches_popup(self) -> None:
        result = self._require_result("Correspondências")
        if result is None:
            return
        dialog = self._create_popup("Correspondências", 1120, 760)
        layout = QVBoxLayout(dialog)
        table = self._create_matches_table(dialog)
        self._populate_matches_table(table, result)
        layout.addWidget(table)
        dialog.exec()

    def _open_preview_popup(self) -> None:
        result = self._require_result("Pré-visualização da malha biométrica")
        if result is None:
            return
        dialog = self._create_popup("Pré-visualização da malha biométrica", 1280, 860)
        layout = QVBoxLayout(dialog)
        splitter = QSplitter(Qt.Vertical, dialog)
        splitter.setChildrenCollapsible(False)
        table = self._create_matches_table(dialog)
        self._populate_matches_table(table, result)
        splitter.addWidget(table)

        previews = QWidget(dialog)
        previews_layout = QHBoxLayout(previews)
        previews_layout.setSpacing(10)
        left_card, left_label, left_info = self._build_preview_card_widget("Padrão", dialog)
        right_card, right_label, right_info = self._build_preview_card_widget("Questionado", dialog)
        previews_layout.addWidget(left_card, stretch=1)
        previews_layout.addWidget(right_card, stretch=1)
        splitter.addWidget(previews)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 3)
        layout.addWidget(splitter)

        def render_match(row: int) -> None:
            if row < 0 or row >= len(result.matches):
                left_label.set_image_path(None)
                right_label.set_image_path(None)
                left_info.setText("Selecione uma correspondência para visualizar a mídia de Padrão.")
                right_info.setText("Selecione uma correspondência para visualizar a mídia de Questionado.")
                return
            match = result.matches[row]
            left_entry = self._entry_by_id.get(match.left_entry_id)
            right_entry = self._entry_by_id.get(match.right_entry_id)
            if left_entry is not None:
                if self._supports_live_mesh_preview(left_entry.source_path):
                    left_label.set_mesh_image(
                        left_entry.source_path,
                        landmarks=left_entry.biometric_landmarks,
                        bbox=left_entry.bbox,
                    )
                elif self._supports_live_mesh_preview(left_entry.crop_path):
                    left_label.set_mesh_image(
                        left_entry.crop_path,
                        landmarks=left_entry.biometric_landmarks,
                        bbox=left_entry.bbox,
                        translate=(-left_entry.bbox.x1, -left_entry.bbox.y1),
                    )
                else:
                    left_label.set_image_path(left_entry.mesh_context_path or left_entry.mesh_crop_path)
            else:
                left_label.set_image_path(None)
            if right_entry is not None:
                if self._supports_live_mesh_preview(right_entry.source_path):
                    right_label.set_mesh_image(
                        right_entry.source_path,
                        landmarks=right_entry.biometric_landmarks,
                        bbox=right_entry.bbox,
                    )
                elif self._supports_live_mesh_preview(right_entry.crop_path):
                    right_label.set_mesh_image(
                        right_entry.crop_path,
                        landmarks=right_entry.biometric_landmarks,
                        bbox=right_entry.bbox,
                        translate=(-right_entry.bbox.x1, -right_entry.bbox.y1),
                    )
                else:
                    right_label.set_image_path(right_entry.mesh_context_path or right_entry.mesh_crop_path)
            else:
                right_label.set_image_path(None)
            left_info.setText(self._preview_text(left_entry, match, "A"))
            right_info.setText(self._preview_text(right_entry, match, "B"))

        def on_selection_changed() -> None:
            selected = table.selectedRanges()
            if not selected:
                render_match(-1)
                return
            row = selected[0].topRow()
            rank_item = table.item(row, 0)
            match_row = int(rank_item.data(Qt.UserRole)) if rank_item and rank_item.data(Qt.UserRole) is not None else row
            render_match(match_row)

        table.itemSelectionChanged.connect(on_selection_changed)
        if result.matches:
            table.selectRow(0)
            render_match(0)
        else:
            render_match(-1)
        dialog.exec()

    def _create_inputs_table(self, parent: QWidget) -> QTableWidget:
        table = QTableWidget(parent)
        table.setColumnCount(7)
        table.setHorizontalHeaderLabels(
            ["Grupo", "Arquivo", "Detectadas", "Selecionadas", "Tracks", "Keyframes", "Estado"]
        )
        header = table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        for column in (2, 3, 4, 5):
            header.setSectionResizeMode(column, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(6, QHeaderView.Stretch)
        table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        table.setSelectionBehavior(QAbstractItemView.SelectRows)
        table.setSelectionMode(QAbstractItemView.SingleSelection)
        table.setAlternatingRowColors(True)
        return table

    def _create_matches_table(self, parent: QWidget) -> QTableWidget:
        table = QTableWidget(parent)
        table.setColumnCount(10)
        table.setHorizontalHeaderLabels(
            [
                "Rank",
                "Similaridade",
                "Classe",
                "log10(LR)",
                "Evidência",
                "Arquivo Padrão",
                "Qualidade Padrão",
                "Arquivo Questionado",
                "Qualidade Questionado",
                "LR",
            ]
        )
        header = table.horizontalHeader()
        for column in (0, 1, 2, 3, 6, 8, 9):
            header.setSectionResizeMode(column, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.Stretch)
        header.setSectionResizeMode(5, QHeaderView.Stretch)
        header.setSectionResizeMode(7, QHeaderView.Stretch)
        table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        table.setSelectionBehavior(QAbstractItemView.SelectRows)
        table.setSelectionMode(QAbstractItemView.SingleSelection)
        table.setAlternatingRowColors(True)
        return table

    def _populate_inputs_table(
        self,
        table: QTableWidget,
        inputs: list[FaceSetComparisonInput],
    ) -> None:
        table.setRowCount(len(inputs))
        for row, item in enumerate(inputs):
            self._set_table_item(table, row, 0, self._group_label(item.set_label))
            self._set_table_item(table, row, 1, item.source_path.name, tooltip=str(item.source_path))
            self._set_table_item(table, row, 2, str(item.detected_faces))
            self._set_table_item(table, row, 3, str(item.selected_faces))
            self._set_table_item(table, row, 4, str(item.tracks))
            self._set_table_item(table, row, 5, str(item.keyframes))
            state_text = "OK" if not item.processing_error else item.processing_error.splitlines()[0]
            state_item = self._set_table_item(table, row, 6, state_text, tooltip=item.processing_error or "")
            if item.processing_error:
                state_item.setForeground(QColor("#b91c1c"))
        table.resizeRowsToContents()

    def _populate_matches_table(self, table: QTableWidget, result: FaceSetComparisonResult) -> None:
        table.setRowCount(len(result.matches))
        for row, match in enumerate(result.matches):
            left = self._entry_by_id.get(match.left_entry_id)
            right = self._entry_by_id.get(match.right_entry_id)
            rank_item = self._set_table_item(table, row, 0, str(match.rank))
            rank_item.setData(Qt.UserRole, row)
            self._set_table_item(table, row, 1, f"{match.similarity:.4f}")
            class_item = self._set_table_item(table, row, 2, self._classification_label(match.classification))
            if match.classification == "assignment":
                class_item.setForeground(QColor("#0f766e"))
            elif match.classification == "candidate":
                class_item.setForeground(QColor("#b45309"))
            self._set_table_item(table, row, 3, self._format_optional_float(match.log10_likelihood_ratio))
            evidence_item = self._set_table_item(table, row, 4, match.evidence_label or "-")
            if match.log10_likelihood_ratio is not None:
                if match.log10_likelihood_ratio >= 0.5:
                    evidence_item.setForeground(QColor("#0f766e"))
                elif match.log10_likelihood_ratio <= -0.5:
                    evidence_item.setForeground(QColor("#b91c1c"))
            self._set_table_item(table, row, 5, left.source_path.name if left else "-", tooltip=str(left.source_path) if left else "")
            self._set_table_item(table, row, 6, self._format_optional_float(match.left_quality_score))
            self._set_table_item(table, row, 7, right.source_path.name if right else "-", tooltip=str(right.source_path) if right else "")
            self._set_table_item(table, row, 8, self._format_optional_float(match.right_quality_score))
            self._set_table_item(table, row, 9, self._format_optional_float(match.likelihood_ratio))
        table.resizeRowsToContents()

    def _build_preview_card_widget(self, title: str, parent: QWidget) -> tuple[QWidget, AdaptiveImageLabel, QLabel]:
        card = QWidget(parent)
        layout = QVBoxLayout(card)
        layout.addWidget(QLabel(title))
        image = AdaptiveImageLabel(parent)
        info = QLabel(parent)
        info.setObjectName("PreviewInfo")
        info.setTextFormat(Qt.RichText)
        info.setWordWrap(True)
        layout.addWidget(image, stretch=1)
        layout.addWidget(info)
        return card, image, info

    def _supports_live_mesh_preview(self, path: Path | None) -> bool:
        if path is None or not path.exists():
            return False
        return path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

    def _preview_text(
        self,
        entry: FaceSetComparisonEntry | None,
        match: FaceSetComparisonMatch | None,
        set_label: str,
    ) -> str:
        if entry is None:
            return f"Selecione uma correspondência para visualizar a mídia de {self._group_label(set_label)}."
        similarity = self._format_optional_float(match.similarity) if match is not None else "-"
        classification = self._classification_label(match.classification) if match is not None else "-"
        return (
            f"<b>Arquivo</b>: {entry.source_path.name}<br>"
            f"<span style='color:#475569'>{entry.source_path}</span><br><br>"
            f"<b>Similaridade</b>: {similarity}<br>"
            f"<b>Classe</b>: {classification}<br>"
            f"<b>log10(LR)</b>: {self._format_optional_float(match.log10_likelihood_ratio) if match is not None else '-'}<br>"
            f"<b>Evidência</b>: {match.evidence_label if match is not None and match.evidence_label else '-'}<br>"
            f"<b>Qualidade</b>: {self._format_optional_float(entry.quality_score)}<br>"
            f"<b>Detecção</b>: {entry.detection_score:.4f}<br>"
            f"<b>Keyframe</b>: {entry.keyframe_id or '-'}"
        )

    def _set_table_item(
        self,
        table: QTableWidget,
        row: int,
        column: int,
        text: str,
        *,
        tooltip: str | None = None,
    ) -> QTableWidgetItem:
        item = QTableWidgetItem(text)
        item.setToolTip(tooltip or text)
        table.setItem(row, column, item)
        return item

    def _format_optional_float(self, value: float | None) -> str:
        return "-" if value is None else f"{value:.4f}"

    def _format_p_value(self, value: float | None) -> str:
        if value is None:
            return "-"
        if value < 1e-4:
            return f"{value:.2e}"
        return f"{value:.6f}"

    def _selected_match_from_table(
        self,
        *,
        table: QTableWidget,
        result: FaceSetComparisonResult,
    ) -> FaceSetComparisonMatch | None:
        selected = table.selectedRanges()
        if not selected:
            return result.matches[0] if result.matches else None
        row = selected[0].topRow()
        rank_item = table.item(row, 0)
        match_index = int(rank_item.data(Qt.UserRole)) if rank_item and rank_item.data(Qt.UserRole) is not None else row
        if 0 <= match_index < len(result.matches):
            return result.matches[match_index]
        return result.matches[0] if result.matches else None

    def _sync_likelihood_ratio_selection(
        self,
        *,
        result: FaceSetComparisonResult,
        table: QTableWidget,
        density_widget: LikelihoodRatioDensityWidget,
        selection_label: QLabel | None,
        series: list[_DistributionSeries],
    ) -> None:
        match = self._selected_match_from_table(table=table, result=result)
        observed_score = match.similarity if match is not None else None
        density_widget.set_series(series, observed_score=observed_score)
        if selection_label is None:
            return
        if match is None:
            selection_label.setText(_likelihood_ratio_selection_html(None))
            return
        left = self._entry_by_id.get(match.left_entry_id)
        right = self._entry_by_id.get(match.right_entry_id)
        left_name = left.source_path.name if left is not None else "-"
        right_name = right.source_path.name if right is not None else "-"
        selection_label.setText(
            _likelihood_ratio_selection_html(
                match,
                left_name=left_name,
                right_name=right_name,
            )
        )

    def _classification_label(self, classification: str) -> str:
        if classification == "assignment":
            return "Atribuição"
        if classification == "candidate":
            return "Candidata"
        return "Abaixo do limiar"

    def _append_log(self, message: str) -> None:
        normalized = message.replace("\r\n", "\n").replace("\r", "\n")
        for line in normalized.split("\n"):
            if line == "":
                self._log_view.appendPlainText("")
                continue
            timestamp = datetime.now().strftime("%H:%M:%S")
            self._log_view.appendPlainText(f"{timestamp} | {line}")
            self._mark_activity(line)

    def _set_activity_state(self, state: str, detail: str) -> None:
        now = datetime.now()
        self._activity_state = state
        self._activity_detail = detail.strip()
        self._last_signal_at = now
        if state == "running":
            self._run_started_at = now
            self._activity_frame_index = 0
            if not self._activity_timer.isActive():
                self._activity_timer.start()
        elif self._activity_timer.isActive():
            self._activity_timer.stop()
        self._refresh_activity_panel()

    def _mark_activity(self, message: str | None = None) -> None:
        self._last_signal_at = datetime.now()
        if message is not None:
            detail = message.strip()
            if detail:
                self._activity_detail = detail
        if self._activity_state == "running":
            self._refresh_activity_panel()

    def _refresh_activity_panel(self) -> None:
        now = datetime.now()
        if self._activity_state == "running":
            if self._run_started_at is None:
                self._run_started_at = now
            if self._last_signal_at is None:
                self._last_signal_at = now
            frames = ("•", "••", "•••")
            frame = frames[self._activity_frame_index % len(frames)]
            self._activity_frame_index += 1
            elapsed = self._format_elapsed(now - self._run_started_at)
            idle = self._format_elapsed(now - self._last_signal_at)
            detail = self._activity_detail or self._status_label.text().strip() or "Processando."
            self._activity_badge.setText(f"Processando {frame}")
            self._activity_meta_label.setText(
                f"{detail} | decorrido {elapsed} | último sinal há {idle} | "
                "o mesmo texto é gravado em run.log e events.jsonl."
            )
            return
        if self._activity_state == "completed":
            duration = (
                self._format_elapsed(now - self._run_started_at)
                if self._run_started_at is not None
                else "-"
            )
            self._activity_badge.setText("Concluído")
            self._activity_meta_label.setText(f"{self._activity_detail} | duração total {duration}.")
            return
        if self._activity_state == "failed":
            duration = (
                self._format_elapsed(now - self._run_started_at)
                if self._run_started_at is not None
                else "-"
            )
            self._activity_badge.setText("Falha")
            self._activity_meta_label.setText(f"{self._activity_detail} | tempo decorrido {duration}.")
            return
        self._activity_badge.setText("Pronto")
        self._activity_meta_label.setText(
            "Aguardando nova execução. O mesmo log exibido aqui também é gravado em run.log e events.jsonl."
        )

    def _format_elapsed(self, delta) -> str:
        total_seconds = max(0, int(delta.total_seconds()))
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    def _open_export_directory(self) -> None:
        if self._latest_result is None:
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(self._latest_result.run_directory)))

    def _export_results_zip(self) -> None:
        if self._latest_result is None:
            return
        default_path = str(self._latest_result.run_directory / f"{self._latest_result.run_directory.name}.zip")
        selected_path, _ = QFileDialog.getSaveFileName(
            self,
            "Salvar pacote ZIP",
            default_path,
            "Arquivos ZIP (*.zip)",
        )
        if not selected_path:
            return
        target_path = Path(selected_path)
        if target_path.suffix.lower() != ".zip":
            target_path = target_path.with_suffix(".zip")
        target_path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(target_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for path in sorted(self._latest_result.run_directory.rglob("*")):
                if path.is_dir():
                    continue
                archive.write(path, arcname=path.relative_to(self._latest_result.run_directory))
        self._append_log(f"Pacote ZIP exportado em: {target_path}")
        QMessageBox.information(self, "Exportação concluída", f"Pacote criado em:\n{target_path}")

    def _save_calibration_model(self) -> None:
        result = self._require_result("Salvar modelo LR")
        if result is None or result.calibration is None:
            return
        default_path = result.export_directory / "face_set_comparison_calibration_model.json"
        selected_path, _ = QFileDialog.getSaveFileName(
            self,
            "Salvar modelo de calibração LR",
            str(default_path),
            "Modelo de calibração LR (*.json);;Arquivos JSON (*.json)",
        )
        if not selected_path:
            return
        target_path = Path(selected_path)
        if target_path.suffix.lower() != ".json":
            target_path = target_path.with_suffix(".json")
        saved_path = self._service_factory().save_face_set_comparison_calibration_model(
            result.calibration,
            target_path,
        )
        self._calibration_model_input.setText(str(saved_path))
        self._append_log(f"Modelo LR salvo em: {saved_path}")
        QMessageBox.information(self, "Modelo salvo", f"Modelo de calibração LR salvo em:\n{saved_path}")

    def _cleanup_thread(self) -> None:
        if self._worker is not None:
            self._worker.deleteLater()
            self._worker = None
        if self._thread is not None:
            self._thread.deleteLater()
            self._thread = None
        if self._activity_state == "running":
            self._set_activity_state("idle", "Aguardando nova execução.")
        self._set_running_state(False)

    def _set_running_state(self, running: bool) -> None:
        self._run_button.setEnabled(not running)
        self._browse_work_directory_button.setEnabled(not running)
        self._work_directory_input.setEnabled(not running)
        self._browse_calibration_directory_button.setEnabled(not running)
        self._calibration_directory_input.setEnabled(not running)
        self._browse_calibration_model_button.setEnabled(not running)
        self._calibration_model_input.setEnabled(not running)
        self._set_a_add_button.setEnabled(not running)
        self._set_a_remove_button.setEnabled(not running)
        self._set_a_clear_button.setEnabled(not running)
        self._set_b_add_button.setEnabled(not running)
        self._set_b_remove_button.setEnabled(not running)
        self._set_b_clear_button.setEnabled(not running)
        if running:
            self._set_result_buttons_enabled(False)
            self._save_calibration_model_button.setEnabled(False)

    def _media_file_filter(self) -> str:
        image_patterns = " ".join(f"*{extension}" for extension in self._config.media.image_extensions)
        video_patterns = " ".join(f"*{extension}" for extension in self._config.media.video_extensions)
        all_patterns = f"{image_patterns} {video_patterns}".strip()
        return (
            f"Mídias suportadas ({all_patterns});;"
            f"Imagens ({image_patterns});;"
            f"Vídeos ({video_patterns});;"
            "Todos os arquivos (*)"
        )

    def request_close(self) -> bool:
        if self._thread is not None:
            QMessageBox.information(
                self,
                "Comparação em andamento",
                "Aguarde o término da comparação antes de fechar esta janela.",
            )
            return False
        self.close()
        return True

    def closeEvent(self, event) -> None:  # type: ignore[override]
        if self._thread is not None:
            QMessageBox.information(
                self,
                "Comparação em andamento",
                "Aguarde o término da comparação antes de fechar esta janela.",
            )
            event.ignore()
            return
        super().closeEvent(event)


