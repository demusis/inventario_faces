from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from html import escape
from pathlib import Path
import zipfile

from PySide6.QtCore import QPointF, QThread, QTimer, Qt, QUrl
from PySide6.QtGui import QColor, QDesktopServices, QPainter, QPen, QPixmap, QPolygonF
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
)

import numpy as np
from scipy.stats import gaussian_kde

from inventario_faces.domain.config import AppConfig
from inventario_faces.domain.entities import (
    FaceSetComparisonEntry,
    FaceSetComparisonInput,
    FaceSetComparisonMatch,
    FaceSetComparisonResult,
)
from inventario_faces.gui.worker import FaceSetComparisonWorker


class AdaptiveImageLabel(QLabel):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("Sem imagem", parent)
        self._pixmap = QPixmap()
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(320, 240)
        self.setStyleSheet(
            "background:#f8fafc; border:1px solid #d7e0ea; border-radius:10px; color:#64748b;"
        )

    def set_image_path(self, path: Path | None) -> None:
        if path is None or not path.exists():
            self._pixmap = QPixmap()
            self.setPixmap(QPixmap())
            self.setText("Sem imagem")
            return
        pixmap = QPixmap(str(path))
        if pixmap.isNull():
            self._pixmap = QPixmap()
            self.setPixmap(QPixmap())
            self.setText(f"Não foi possível abrir {path.name}")
            return
        self._pixmap = pixmap
        self._refresh()

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        self._refresh()

    def _refresh(self) -> None:
        if self._pixmap.isNull():
            return
        self.setText("")
        self.setPixmap(
            self._pixmap.scaled(
                self.contentsRect().size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
        )


@dataclass(frozen=True)
class _DistributionSeries:
    label: str
    classification: str
    color: str
    values: tuple[float, ...]
    sufficient: bool
    note: str | None = None
    kde_x: tuple[float, ...] = ()
    kde_y: tuple[float, ...] = ()
    mean: float | None = None
    median: float | None = None
    q1: float | None = None
    q3: float | None = None
    ci_low: float | None = None
    ci_high: float | None = None


class SimilarityDistributionWidget(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._series: list[_DistributionSeries] = []
        self._candidate = 0.0
        self._assignment = 0.0
        self._overall_ci_low: float | None = None
        self._overall_ci_high: float | None = None
        self._overall_mean: float | None = None
        self._show_threshold_markers = True
        self._show_mean_marker = True
        self.setMinimumHeight(240)

    def set_distribution(
        self,
        series: list[_DistributionSeries],
        *,
        candidate_threshold: float,
        assignment_threshold: float,
        mean_value: float | None,
        ci_low: float | None,
        ci_high: float | None,
        show_threshold_markers: bool = True,
        show_mean_marker: bool = True,
    ) -> None:
        self._series = list(series)
        self._candidate = candidate_threshold
        self._assignment = assignment_threshold
        self._overall_mean = mean_value
        self._overall_ci_low = ci_low
        self._overall_ci_high = ci_high
        self._show_threshold_markers = show_threshold_markers
        self._show_mean_marker = show_mean_marker
        self.update()

    def paintEvent(self, event) -> None:  # type: ignore[override]
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.fillRect(self.rect(), QColor("#ffffff"))

        outer = self.rect().adjusted(12, 12, -12, -12)
        painter.setPen(QPen(QColor("#d7e0ea"), 1))
        painter.drawRoundedRect(outer, 10, 10)

        plot = outer.adjusted(42, 18, -16, -36)
        if plot.width() <= 0 or plot.height() <= 0:
            return

        painter.setPen(QPen(QColor("#e7edf4"), 1))
        for ratio in (0.25, 0.5, 0.75):
            y = plot.bottom() - (plot.height() * ratio)
            painter.drawLine(plot.left(), int(y), plot.right(), int(y))

        drawable_series = [series for series in self._series if series.sufficient and series.kde_x and series.kde_y]
        if not drawable_series:
            painter.setPen(QPen(QColor("#64748b"), 1))
            painter.drawText(plot, Qt.AlignCenter, "A distribuição não será exibida sem repetição suficiente e variabilidade.")
            return

        all_values = [value for series in drawable_series for value in series.values]
        lower = min(0.0, min(all_values), self._candidate, self._assignment, self._overall_ci_low or 0.0)
        upper = max(1.0, max(all_values), self._candidate, self._assignment, self._overall_ci_high or 1.0)
        if upper - lower < 0.2:
            lower -= 0.1
            upper += 0.1

        if self._overall_ci_low is not None and self._overall_ci_high is not None:
            x1 = self._map_x(self._overall_ci_low, plot, lower, upper)
            x2 = self._map_x(self._overall_ci_high, plot, lower, upper)
            painter.fillRect(
                int(min(x1, x2)),
                plot.top(),
                int(abs(x2 - x1)),
                plot.height(),
                QColor(15, 118, 110, 32),
            )

        max_density = max(max(series.kde_y) for series in drawable_series if series.kde_y) or 1.0
        legend_x = plot.left() + 10
        legend_y = plot.top() + 8

        for series in drawable_series:
            color = QColor(series.color)
            points = [
                QPointF(
                    self._map_x(x, plot, lower, upper),
                    plot.bottom() - ((y / max_density) * plot.height()),
                )
                for x, y in zip(series.kde_x, series.kde_y)
            ]
            fill_polygon = [QPointF(points[0].x(), plot.bottom()), *points, QPointF(points[-1].x(), plot.bottom())]
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(color.red(), color.green(), color.blue(), 36))
            painter.drawPolygon(QPolygonF(fill_polygon))
            painter.setBrush(Qt.NoBrush)
            painter.setPen(QPen(color, 2))
            painter.drawPolyline(QPolygonF(points))
            painter.fillRect(legend_x, legend_y, 12, 12, color)
            painter.setPen(QPen(QColor("#334155"), 1))
            painter.drawText(legend_x + 18, legend_y - 1, 180, 14, Qt.AlignLeft | Qt.AlignVCenter, series.label)
            legend_y += 18

        self._draw_marker(painter, plot, lower, upper, self._candidate, QColor("#f59e0b"), "cand.")
        self._draw_marker(painter, plot, lower, upper, self._assignment, QColor("#0f766e"), "atrib.")
        if self._overall_mean is not None:
            self._draw_marker(painter, plot, lower, upper, self._overall_mean, QColor("#1e293b"), "média")

        painter.setPen(QPen(QColor("#475569"), 1))
        painter.drawLine(plot.left(), plot.bottom(), plot.right(), plot.bottom())
        for ratio in (0.0, 0.25, 0.5, 0.75, 1.0):
            value = lower + ((upper - lower) * ratio)
            x = self._map_x(value, plot, lower, upper)
            painter.drawText(int(x - 20), outer.bottom() - 12, 44, 14, Qt.AlignCenter, f"{value:.2f}")

    def _map_x(self, value: float, plot, lower: float, upper: float) -> float:
        if upper <= lower:
            return float(plot.left())
        return plot.left() + (plot.width() * ((value - lower) / (upper - lower)))

    def _draw_marker(
        self,
        painter: QPainter,
        plot,
        lower: float,
        upper: float,
        value: float,
        color: QColor,
        label: str,
    ) -> None:
        x = self._map_x(value, plot, lower, upper)
        pen = QPen(color, 2)
        pen.setStyle(Qt.DashLine)
        painter.setPen(pen)
        painter.drawLine(int(x), plot.top(), int(x), plot.bottom())
        painter.setPen(QPen(color, 1))
        painter.drawText(int(x - 26), plot.top() - 4, 52, 14, Qt.AlignCenter, label)


class LikelihoodRatioDensityWidget(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._series: list[_DistributionSeries] = []
        self.setMinimumHeight(260)

    def set_series(self, series: list[_DistributionSeries]) -> None:
        self._series = list(series)
        self.update()

    def paintEvent(self, event) -> None:  # type: ignore[override]
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.fillRect(self.rect(), QColor("#ffffff"))

        outer = self.rect().adjusted(12, 12, -12, -12)
        painter.setPen(QPen(QColor("#d7e0ea"), 1))
        painter.drawRoundedRect(outer, 10, 10)

        plot = outer.adjusted(42, 18, -16, -36)
        if plot.width() <= 0 or plot.height() <= 0:
            return

        drawable_series = [series for series in self._series if series.sufficient and series.kde_x and series.kde_y]
        if not drawable_series:
            painter.setPen(QPen(QColor("#64748b"), 1))
            painter.drawText(plot, Qt.AlignCenter, "A densidade calibrada nao esta disponivel.")
            return

        all_values = [value for series in drawable_series for value in series.values]
        lower = min(-1.0, min(all_values))
        upper = max(1.0, max(all_values))
        if upper - lower < 0.2:
            lower -= 0.1
            upper += 0.1

        painter.setPen(QPen(QColor("#e7edf4"), 1))
        for ratio in (0.25, 0.5, 0.75):
            y = plot.bottom() - (plot.height() * ratio)
            painter.drawLine(plot.left(), int(y), plot.right(), int(y))

        max_density = max(max(series.kde_y) for series in drawable_series if series.kde_y) or 1.0
        legend_x = plot.left() + 10
        legend_y = plot.top() + 8

        for series in drawable_series:
            color = QColor(series.color)
            points = [
                QPointF(
                    self._map_x(x, plot, lower, upper),
                    plot.bottom() - ((y / max_density) * plot.height()),
                )
                for x, y in zip(series.kde_x, series.kde_y)
            ]
            fill_polygon = [QPointF(points[0].x(), plot.bottom()), *points, QPointF(points[-1].x(), plot.bottom())]
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(color.red(), color.green(), color.blue(), 32))
            painter.drawPolygon(QPolygonF(fill_polygon))
            painter.setBrush(Qt.NoBrush)
            painter.setPen(QPen(color, 2))
            painter.drawPolyline(QPolygonF(points))
            painter.fillRect(legend_x, legend_y, 12, 12, color)
            painter.setPen(QPen(QColor("#334155"), 1))
            painter.drawText(legend_x + 18, legend_y - 1, 240, 14, Qt.AlignLeft | Qt.AlignVCenter, series.label)
            legend_y += 18

        painter.setPen(QPen(QColor("#475569"), 1))
        painter.drawLine(plot.left(), plot.bottom(), plot.right(), plot.bottom())
        for ratio in (0.0, 0.25, 0.5, 0.75, 1.0):
            value = lower + ((upper - lower) * ratio)
            x = self._map_x(value, plot, lower, upper)
            painter.drawText(int(x - 20), outer.bottom() - 12, 44, 14, Qt.AlignCenter, f"{value:.2f}")

    def _map_x(self, value: float, plot, lower: float, upper: float) -> float:
        if upper <= lower:
            return float(plot.left())
        return plot.left() + (plot.width() * ((value - lower) / (upper - lower)))


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
                background:#ffffff; border:1px solid #d9e3ee; border-radius:10px;
            }
            QLabel#MetricTitle { color:#64748b; font-size:11px; }
            QLabel#MetricValue { color:#0f172a; font-size:18px; font-weight:700; }
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
        self._browse_work_directory_button.clicked.connect(self._select_work_directory)
        layout.addWidget(self._browse_work_directory_button, 0, 2)
        self._run_button = QPushButton("Comparar conjuntos")
        self._run_button.setObjectName("PrimaryButton")
        self._run_button.clicked.connect(self._start_comparison)
        layout.addWidget(self._run_button, 0, 3)
        self._help_button = QPushButton("Ajuda")
        self._help_button.clicked.connect(self._open_help_popup)
        layout.addWidget(self._help_button, 0, 4)
        self._export_button = QPushButton("Exportar ZIP")
        self._export_button.setEnabled(False)
        self._export_button.clicked.connect(self._export_results_zip)
        layout.addWidget(self._export_button, 0, 5)
        self._open_export_button = QPushButton("Abrir execução")
        self._open_export_button.setEnabled(False)
        self._open_export_button.clicked.connect(self._open_export_directory)
        layout.addWidget(self._open_export_button, 0, 6)
        self._calibration_directory_input = QLineEdit(self)
        self._calibration_directory_input.setPlaceholderText("Uma subpasta por identidade rotulada")
        self._browse_calibration_directory_button = QPushButton("Selecionar base")
        self._browse_calibration_directory_button.clicked.connect(self._select_calibration_directory)
        layout.addWidget(QLabel("Base de calibração LR (opcional):"), 1, 0)
        layout.addWidget(self._calibration_directory_input, 1, 1, 1, 5)
        layout.addWidget(self._browse_calibration_directory_button, 1, 6)
        self._calibration_model_input = QLineEdit(self)
        self._calibration_model_input.setPlaceholderText("Arquivo JSON com modelo LR já calculado")
        self._browse_calibration_model_button = QPushButton("Carregar modelo")
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
            ("Resumo estatístico", self._open_summary_popup),
            ("Distribuição", self._open_distribution_popup),
            ("Entradas processadas", self._open_inputs_popup),
            ("Correspondências", self._open_matches_popup),
            ("Malha biométrica", self._open_preview_popup),
        ]
        self._result_buttons: list[QPushButton] = []
        for label, handler in buttons:
            button = QPushButton(label)
            button.setEnabled(False)
            button.clicked.connect(handler)
            layout.addWidget(button)
            self._result_buttons.append(button)
        likelihood_button = QPushButton("Razão de verossimilhança")
        likelihood_button.setEnabled(False)
        likelihood_button.clicked.connect(self._open_likelihood_ratio_popup)
        layout.addWidget(likelihood_button)
        self._result_buttons.append(likelihood_button)
        self._save_calibration_model_button = QPushButton("Salvar modelo LR")
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
        add_button = QPushButton("Adicionar imagens")
        add_button.clicked.connect(lambda: self._add_images(set_label))
        controls.addWidget(add_button)
        remove_button = QPushButton("Remover selecionadas")
        remove_button.clicked.connect(lambda: self._remove_selected_images(set_label))
        controls.addWidget(remove_button)
        clear_button = QPushButton("Limpar")
        clear_button.clicked.connect(lambda: self._clear_images(set_label))
        controls.addWidget(clear_button)
        layout.addLayout(controls)

        count_label = QLabel("0 imagem(ns) selecionada(s)")
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
            f"Selecionar imagens do grupo {self._group_label(set_label)}",
            str(self._initial_input_directory),
            self._image_file_filter(),
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
        self._count_label_for_set(set_label).setText(f"{list_widget.count()} imagem(ns) selecionada(s)")

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
            QMessageBox.warning(self, "Padrão vazio", "Selecione ao menos uma imagem no grupo Padrão.")
            return
        if not set_b_paths:
            QMessageBox.warning(self, "Questionado vazio", "Selecione ao menos uma imagem no grupo Questionado.")
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
        self._append_log(f"Padrão: {len(set_a_paths)} imagem(ns)")
        self._append_log(f"Questionado: {len(set_b_paths)} imagem(ns)")
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

    def _create_metric_card(self, title: str, value: str) -> QFrame:
        card = QFrame(self)
        card.setObjectName("MetricCard")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(10, 8, 10, 8)
        title_label = QLabel(title)
        title_label.setObjectName("MetricTitle")
        value_label = QLabel(value)
        value_label.setObjectName("MetricValue")
        layout.addWidget(title_label)
        layout.addWidget(value_label)
        return card

    def _bootstrap_alpha(self) -> float:
        return self._significance_input.value() / 100.0

    def _bootstrap_resamples(self) -> int:
        return int(self._bootstrap_resamples_input.value())

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
    ) -> tuple[tuple[float, ...], tuple[float, ...]]:
        array = np.asarray(values, dtype=np.float64)
        if abs(bandwidth_scale - 1.0) <= 1e-12:
            kde = gaussian_kde(array)
        else:
            kde = gaussian_kde(array, bw_method=lambda model: model.scotts_factor() * bandwidth_scale)
        grid = np.linspace(lower, upper, points)
        density = kde(grid)
        return tuple(float(value) for value in grid), tuple(float(value) for value in density)

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
        lower = min(0.0, min(all_values)) if all_values else 0.0
        upper = max(1.0, max(all_values)) if all_values else 1.0
        if upper - lower < 0.2:
            lower -= 0.1
            upper += 0.1
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
            "assignment": ("Atribuição", "#0f766e"),
            "candidate": ("Candidata", "#b45309"),
            "below_threshold": ("Abaixo do limiar", "#7c3aed"),
        }
        series_list: list[_DistributionSeries] = []
        for classification, values in groups.items():
            label, color = palettes.get(classification, (classification, "#334155"))
            class_support, class_note = self._has_statistical_support(values)
            if class_support:
                kde_x, kde_y = self._kde_curve(values, lower=lower, upper=upper)
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
        dialog = self._create_popup("Resumo estatístico", 980, 760)
        layout = QVBoxLayout(dialog)
        if support:
            metrics = QGridLayout()
            cards = [
                ("Comparações", str(summary.total_pair_comparisons)),
                (
                    "Faces selecionadas",
                    f"Padrão {summary.set_a_selected_faces} | Questionado {summary.set_b_selected_faces}",
                ),
                ("Atribuições", str(summary.assignment_matches)),
                ("Candidatas", str(summary.candidate_matches)),
                ("Média", self._format_optional_float(overall_stats["mean"])),
                ("Mediana", self._format_optional_float(overall_stats["median"])),
                ("Q1 / Q3", f"{self._format_optional_float(overall_stats['q1'])} / {self._format_optional_float(overall_stats['q3'])}"),
                ("IC bootstrap", f"{self._format_optional_float(overall_stats['ci_low'])} .. {self._format_optional_float(overall_stats['ci_high'])}"),
            ]
            for index, (title, value) in enumerate(cards):
                metrics.addWidget(self._create_metric_card(title, value), index // 4, index % 4)
            layout.addLayout(metrics)
        else:
            info = QTextBrowser(dialog)
            info.setPlainText(
                "\n".join(
                    [
                        "A análise estatística inferencial não será apresentada para esta comparação.",
                        note or "Amostra insuficiente.",
                        f"Comparações disponíveis: {summary.total_pair_comparisons}",
                        (
                            f"Faces selecionadas: Padrão {summary.set_a_selected_faces} | "
                            f"Questionado {summary.set_b_selected_faces}"
                        ),
                    ]
                )
            )
            info.setMaximumHeight(180)
            layout.addWidget(info)
        class_notes = QTextBrowser(dialog)
        class_notes.setPlainText(
            "\n".join(
                [
                    f"{series.label}: n={len(series.values)} | {'OK' if series.sufficient else series.note}"
                    for series in series_list
                ]
            )
        )
        class_notes.setMaximumHeight(140)
        layout.addWidget(class_notes)
        layout.addWidget(QLabel("Procedimento e configuração usada"))
        browser = QTextBrowser(dialog)
        browser.setPlainText("\n".join(result.procedure_details))
        layout.addWidget(browser, stretch=1)
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
        image_extensions = ", ".join(self._config.media.image_extensions)
        det_size = (
            f"{self._config.face_model.det_size[0]}x{self._config.face_model.det_size[1]}"
            if self._config.face_model.det_size is not None
            else "resolução original do quadro"
        )
        providers = (
            ", ".join(self._config.face_model.providers)
            if self._config.face_model.providers
            else "seleção automática com preferência por GPU e fallback para CPU"
        )
        return f"""
<html>
<head>
<style>
body {{
    font-family: 'Segoe UI', sans-serif;
    color: #0f172a;
    line-height: 1.45;
}}
h1 {{
    font-size: 22px;
    color: #0f172a;
    margin: 0 0 10px 0;
}}
h2 {{
    font-size: 17px;
    color: #0f766e;
    margin: 18px 0 6px 0;
}}
h3 {{
    font-size: 14px;
    color: #334155;
    margin: 14px 0 4px 0;
}}
p, li {{
    font-size: 13px;
}}
code {{
    background: #f1f5f9;
    padding: 1px 4px;
    border-radius: 4px;
}}
table {{
    border-collapse: collapse;
    width: 100%;
    margin-top: 8px;
}}
th, td {{
    border: 1px solid #d7e0ea;
    padding: 6px 8px;
    text-align: left;
    vertical-align: top;
}}
th {{
    background: #eef4fa;
}}
</style>
</head>
<body>
<h1>Ajuda da comparação entre grupos faciais</h1>
<p>
Esta janela compara dois conjuntos de imagens faciais, normalmente um conjunto de referência
(<b>Padrão</b>) e um conjunto sob exame (<b>Questionado</b>). O sistema processa as imagens,
seleciona faces elegíveis, compara as representações faciais e organiza o resultado em saídas auditáveis.
</p>

<h2>Objetivo da janela</h2>
<ul>
    <li>Comparar diretamente dois grupos de imagens faciais.</li>
    <li>Classificar os pares por nível de interesse: atribuição, candidata ou abaixo do limiar.</li>
    <li>Exibir medidas descritivas e inferenciais quando houver repetição e variabilidade suficientes.</li>
    <li>Aplicar razão de verossimilhança calibrada quando houver base LR ou modelo LR salvo.</li>
    <li>Gerar artefatos exportáveis, logs e trilha de auditoria da execução.</li>
</ul>

<h2>Fluxo recomendado de uso</h2>
<ol>
    <li>Adicione as imagens do grupo <b>Padrão</b>.</li>
    <li>Adicione as imagens do grupo <b>Questionado</b>.</li>
    <li>Confirme o <b>Diretório de trabalho</b>.</li>
    <li>Se necessário, informe a <b>Base de calibração LR</b> ou carregue um <b>Modelo de calibração LR</b>.</li>
    <li>Clique em <b>Comparar conjuntos</b>.</li>
    <li>Ao final, revise <b>Resumo estatístico</b>, <b>Correspondências</b> e, se houver, <b>Razão de verossimilhança</b>.</li>
    <li>Se a calibração for útil para execuções futuras, preserve o JSON salvo automaticamente ou use <b>Salvar modelo LR</b>.</li>
</ol>

<h2>Controles principais</h2>
<table>
    <tr><th>Controle</th><th>Função</th><th>Orientação</th></tr>
    <tr>
        <td><b>Padrão</b></td>
        <td>Grupo de referência usado como fonte das faces do conjunto A.</td>
        <td>Use imagens representativas e bem documentadas. Extensões aceitas nesta configuração: <code>{escape(image_extensions)}</code>.</td>
    </tr>
    <tr>
        <td><b>Questionado</b></td>
        <td>Grupo examinado, tratado como conjunto B na comparação.</td>
        <td>Mantenha apenas material pertinente à hipótese examinada para evitar inflar o número de pares sem necessidade.</td>
    </tr>
    <tr>
        <td><b>Adicionar imagens</b></td>
        <td>Inclui novos arquivos no grupo.</td>
        <td>Ideal para montar o conjunto de forma incremental.</td>
    </tr>
    <tr>
        <td><b>Remover selecionadas</b></td>
        <td>Retira apenas os itens marcados na lista.</td>
        <td>Útil para limpar erros de seleção sem reiniciar tudo.</td>
    </tr>
    <tr>
        <td><b>Limpar</b></td>
        <td>Esvazia completamente o grupo correspondente.</td>
        <td>Use quando quiser reiniciar a montagem do conjunto.</td>
    </tr>
    <tr>
        <td><b>Diretório de trabalho</b></td>
        <td>Local em que a execução grava logs, tabelas, gráficos, JSONs, ZIP e demais artefatos.</td>
        <td>Escolha uma pasta com espaço suficiente e preservável para auditoria posterior.</td>
    </tr>
    <tr>
        <td><b>Base de calibração LR (opcional)</b></td>
        <td>Diretório com uma subpasta por identidade rotulada, usado para estimar as distribuições de mesma origem e de origem distinta.</td>
        <td>É a opção mais completa, mas também a mais custosa em tempo de processamento.</td>
    </tr>
    <tr>
        <td><b>Modelo de calibração LR (opcional)</b></td>
        <td>Arquivo JSON com o modelo LR já calculado em execução anterior.</td>
        <td>Se a base de calibração não mudou, esta é a forma recomendada de reaproveitar a LR sem recalcular tudo.</td>
    </tr>
    <tr>
        <td><b>Comparar conjuntos</b></td>
        <td>Inicia o pipeline da comparação.</td>
        <td>Durante a execução, os controles de entrada são bloqueados para manter a consistência do procedimento.</td>
    </tr>
    <tr>
        <td><b>Ajuda</b></td>
        <td>Abre este painel explicativo.</td>
        <td>Use-o como referência rápida operacional e interpretativa.</td>
    </tr>
    <tr>
        <td><b>Exportar ZIP</b></td>
        <td>Compacta o diretório de execução concluída.</td>
        <td>Útil para preservação do procedimento e circulação controlada dos artefatos.</td>
    </tr>
    <tr>
        <td><b>Abrir execução</b></td>
        <td>Abre a pasta da execução atual.</td>
        <td>Permite acessar diretamente os arquivos exportados, logs e modelos LR.</td>
    </tr>
</table>

<h2>Barra de resultados</h2>
<table>
    <tr><th>Controle</th><th>Uso</th><th>Leitura</th></tr>
    <tr>
        <td><b>Reamostragens</b></td>
        <td>Quantidade de amostras bootstrap usada nos intervalos de confiança.</td>
        <td>Valores maiores tendem a estabilizar a inferência, porém aumentam o custo computacional.</td>
    </tr>
    <tr>
        <td><b>Significância (%)</b></td>
        <td>Nível de significância aplicado à inferência bootstrap.</td>
        <td>Por exemplo, 5% produz um intervalo bilateral aproximado de 95%.</td>
    </tr>
    <tr>
        <td><b>Resumo estatístico</b></td>
        <td>Mostra contagens, média, mediana, quartis, IC bootstrap e procedimento.</td>
        <td>É o melhor ponto de partida para avaliar robustez quantitativa da execução.</td>
    </tr>
    <tr>
        <td><b>Distribuição</b></td>
        <td>Exibe curvas KDE por classe decisória e marcadores dos limiares.</td>
        <td>Ajuda a visualizar dispersão, separação entre classes e estabilidade dos scores.</td>
    </tr>
    <tr>
        <td><b>Entradas processadas</b></td>
        <td>Lista arquivo a arquivo com detectadas, selecionadas, tracks, keyframes e estado.</td>
        <td>É a saída primária para auditoria operacional e checagem de aproveitamento das imagens.</td>
    </tr>
    <tr>
        <td><b>Correspondências</b></td>
        <td>Mostra o ranking dos pares comparados.</td>
        <td>É a saída central para revisão pericial; deve ser lida junto com a inspeção visual.</td>
    </tr>
    <tr>
        <td><b>Malha biométrica</b></td>
        <td>Mostra a tabela de correspondências com as imagens/derivados associados ao par selecionado.</td>
        <td>Serve para revisar contexto, recorte, qualidade, keyframe e coerência visual do par.</td>
    </tr>
    <tr>
        <td><b>Razão de verossimilhança</b></td>
        <td>Exibe estado da calibração LR, densidades H1/H2 e a tabela com <code>LR</code>, <code>log10(LR)</code> e evidência.</td>
        <td>Use quando a execução estiver calibrada e a base ou modelo LR forem tecnicamente compatíveis com o caso.</td>
    </tr>
    <tr>
        <td><b>Salvar modelo LR</b></td>
        <td>Salva manualmente o modelo LR corrente em JSON.</td>
        <td>Disponível apenas depois de uma execução com calibração LR.</td>
    </tr>
</table>

<h2>Pipeline técnico resumido</h2>
<h3>1. Preparação e leitura das imagens</h3>
<p>
As imagens dos dois grupos são abertas individualmente e registradas como entradas processadas. A comparação não trabalha
com a “pasta inteira” como uma única unidade, mas com as faces elegíveis encontradas em cada arquivo.
</p>

<h3>2. Detecção, filtros e extração facial</h3>
<p>
O backend atual é <code>{escape(self._config.face_model.backend)}</code>, com modelo
<code>{escape(self._config.face_model.model_name)}</code>, tamanho de detecção
<code>{escape(det_size)}</code>, qualidade mínima
<code>{self._config.face_model.minimum_face_quality:.2f}</code>, tamanho mínimo de face
<code>{self._config.face_model.minimum_face_size_pixels}px</code>, <code>ctx_id={self._config.face_model.ctx_id}</code>
e providers configurados como <code>{escape(providers)}</code>.
</p>
<p>
Cada imagem passa por detecção facial, filtros de qualidade e tamanho, seleção de ocorrências elegíveis e geração de embeddings.
Faces inelegíveis podem ser descartadas antes da comparação propriamente dita.
</p>

<h3>3. Comparação entre Padrão e Questionado</h3>
<p>
Depois da seleção das faces válidas, o sistema compara cada face elegível do grupo Padrão com cada face elegível do grupo
Questionado. Em termos práticos, o número de comparações cresce aproximadamente com
<code>faces_padrão × faces_questionado</code>.
</p>

<h3>4. Similaridade e classes decisórias</h3>
<p>
Cada par recebe uma similaridade facial. Em seguida, o sistema o classifica segundo os limiares de decisão da configuração:
</p>
<ul>
    <li><b>Atribuição</b>: resultado que atinge o limiar principal de atribuição.</li>
    <li><b>Candidata</b>: resultado abaixo da atribuição, mas acima do limiar de sugestão investigativa.</li>
    <li><b>Abaixo do limiar</b>: resultado que não atingiu o patamar mínimo de interesse definido.</li>
</ul>
<p>
Na configuração carregada nesta sessão, o limiar de atribuição é
<code>{self._config.clustering.assignment_similarity:.2f}</code> e o limiar de sugestão é
<code>{self._config.clustering.candidate_similarity:.2f}</code>.
</p>

<h3>5. Inferência estatística</h3>
<p>
Se houver número suficiente de repetições e variabilidade entre os valores, a janela calcula média, mediana, quartis,
intervalo de confiança bootstrap e densidades KDE. Se a amostra for pequena demais ou quase constante, a interface informa
que a inferência não pôde ser apresentada.
</p>

<h3>6. Calibração por razão de verossimilhança (LR)</h3>
<p>
Quando uma base rotulada ou um modelo LR salvo é informado, o sistema pode calibrar a interpretação do score. Em vez de
usar apenas a similaridade bruta, ele estima o quanto aquele valor é compatível com:
</p>
<ul>
    <li><b>H1</b>: mesma origem.</li>
    <li><b>H2</b>: origem distinta.</li>
</ul>
<p>
Se a base rotulada for usada, o sistema gera scores de mesma origem e origem distinta a partir das subpastas-identidade,
ajusta densidades e calcula <code>LR</code> e <code>log10(LR)</code> para os pares do caso.
Se um modelo salvo for carregado, essa etapa é reaproveitada sem recalcular toda a base.
</p>
<p><b>Importante:</b> se a base rotulada e o modelo salvo forem informados ao mesmo tempo, o <b>modelo salvo tem prioridade</b>.</p>

<h2>Interpretação dos campos principais</h2>
<h3>Similaridade</h3>
<p>
É a medida comparativa bruta entre os embeddings das duas faces. Similaridade alta indica maior proximidade no espaço do
modelo, mas não equivale, isoladamente, a identificação conclusiva.
</p>

<h3>Classe</h3>
<p>
A classe mostra a posição do par em relação aos limiares. <b>Atribuição</b> representa um resultado mais forte do que
<b>Candidata</b>, porém ambos exigem revisão humana. <b>Abaixo do limiar</b> indica que o par não atingiu o patamar mínimo
de interesse configurado.
</p>

<h3>Qualidade</h3>
<p>
As colunas de qualidade ajudam a julgar o valor prático do par. Um score alto vindo de faces ruins pede cautela extra; um
score moderado vindo de material ruim também pode estar artificialmente deprimido pela qualidade da entrada.
</p>

<h3>Resumo estatístico</h3>
<p>
Use esta janela para verificar quantos pares foram comparados, quantos ultrapassaram os limiares, se houve suporte
suficiente para inferência e como a distribuição geral se comporta.
</p>

<h3>Distribuição</h3>
<p>
As curvas KDE ajudam a enxergar concentração, separação e dispersão dos scores. Curvas muito sobrepostas sugerem menor
separação prática; curvas mais apartadas indicam comportamento mais estável dos resultados.
</p>

<h3>Entradas processadas</h3>
<p>
É a visão mais importante para auditoria do processamento. Antes de interpretar qualquer ranking, vale conferir se as
imagens relevantes foram realmente aproveitadas, quantas faces foram detectadas e se houve erro em algum arquivo.
</p>

<h3>Correspondências</h3>
<p>
O ranking organiza os pares mais relevantes da comparação. Em geral, a revisão começa pelos primeiros itens, mas a ordem
não substitui a análise pericial do contexto e da qualidade do material.
</p>

<h3>Malha biométrica</h3>
<p>
Esta visualização integra números e imagem. Ela serve para inspeção qualitativa do par selecionado, verificando contexto,
recorte, qualidade, keyframe e coerência geral do material exibido.
</p>

<h3>LR, log10(LR) e evidência</h3>
<ul>
    <li><code>LR &gt; 1</code> favorece a hipótese de mesma origem.</li>
    <li><code>LR &lt; 1</code> favorece a hipótese de origem distinta.</li>
    <li><code>log10(LR) = 0</code> é aproximadamente neutro entre H1 e H2.</li>
    <li><code>log10(LR) &gt; 0</code> favorece mesma origem; quanto maior, mais forte o suporte.</li>
    <li><code>log10(LR) &lt; 0</code> favorece origem distinta; quanto menor, mais forte o suporte para H2.</li>
</ul>
<p>
O rótulo textual de <b>Evidência</b> resume essa magnitude. Ainda assim, a força da LR depende da adequação da base ou do
modelo LR ao pipeline realmente usado no caso.
</p>

<h2>Quando reutilizar um modelo LR salvo</h2>
<ul>
    <li>Quando a base de calibração é a mesma.</li>
    <li>Quando backend, modelo facial, thresholds e condições operacionais continuam compatíveis.</li>
    <li>Quando o objetivo é comparar novos grupos Padrão/Questionado sem recalcular a calibração inteira.</li>
</ul>
<p>
Se houver mudança material na base rotulada, no pipeline ou na configuração, o mais prudente é recalibrar.
</p>

<h2>Cautelas periciais e boas práticas</h2>
<ul>
    <li>Não trate o resultado automático como prova conclusiva de identidade.</li>
    <li>Combine leitura numérica, revisão visual e contexto do caso.</li>
    <li>Leve em conta a qualidade das faces antes de valorar a força de um par.</li>
    <li>Ao usar LR, confirme se a calibração estava realmente disponível e com suporte suficiente.</li>
    <li>Preserve o diretório de execução ou o ZIP exportado como parte da trilha de auditoria.</li>
</ul>

<h2>Arquivos gerados e rastreabilidade</h2>
<p>
Cada execução grava um diretório próprio com logs, tabelas, resumos, artefatos de imagem e, quando aplicável, arquivos da
calibração LR. O botão <b>Abrir execução</b> leva diretamente a essa pasta, e <b>Exportar ZIP</b> permite empacotá-la.
</p>

<h2>Se a distribuição ou a LR não aparecerem</h2>
<p>
As saídas inferenciais podem ficar indisponíveis quando houver amostra insuficiente, variabilidade muito baixa, calibração
ausente ou base/modelo LR sem suporte suficiente. Nesses casos, os popups e o log informam o motivo de forma explícita.
</p>
</body>
</html>
"""

    def _open_distribution_popup(self) -> None:
        result = self._require_result("Distribuição de similaridades")
        if result is None:
            return
        summary = result.summary
        series_list, (support, note), overall_stats = self._distribution_analysis(result)
        dialog = self._create_popup("Distribuição de similaridades", 1040, 760)
        layout = QVBoxLayout(dialog)
        if support:
            widget = SimilarityDistributionWidget(dialog)
            widget.set_distribution(
                series_list,
                candidate_threshold=summary.candidate_threshold,
                assignment_threshold=summary.assignment_threshold,
                mean_value=overall_stats["mean"],
                ci_low=overall_stats["ci_low"],
                ci_high=overall_stats["ci_high"],
            )
            layout.addWidget(widget, stretch=1)
        else:
            info = QTextBrowser(dialog)
            info.setPlainText(
                "\n".join(
                    [
                        "A distribuição KDE não será exibida para esta comparação.",
                        note or "Amostra insuficiente.",
                        f"Comparações disponíveis: {summary.total_pair_comparisons}",
                    ]
                )
            )
            layout.addWidget(info)
        caption = QTextBrowser(dialog)
        caption.setPlainText(
            "\n".join(
                [
                    "Curvas KDE separadas por classe de decisão.",
                    f"Reamostragens bootstrap: {self._bootstrap_resamples()}",
                    f"Nível de significância: {self._significance_input.value():.2f}%",
                    f"Média: {self._format_optional_float(overall_stats['mean'])}",
                    f"Mediana: {self._format_optional_float(overall_stats['median'])}",
                    f"Q1: {self._format_optional_float(overall_stats['q1'])}",
                    f"Q3: {self._format_optional_float(overall_stats['q3'])}",
                    (
                        "IC bootstrap da média: "
                        f"{self._format_optional_float(overall_stats['ci_low'])} .. "
                        f"{self._format_optional_float(overall_stats['ci_high'])}"
                    ),
                    f"Limiar candidato: {summary.candidate_threshold:.4f}",
                    f"Limiar de atribuição: {summary.assignment_threshold:.4f}",
                    "",
                    *[
                        f"{series.label}: n={len(series.values)} | "
                        f"{'KDE exibida' if series.sufficient else series.note}"
                        for series in series_list
                    ],
                ]
            )
        )
        caption.setMaximumHeight(240)
        layout.addWidget(caption)
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

        lower = min(-1.0, min([*calibration.genuine_scores, *calibration.impostor_scores]))
        upper = max(1.0, max([*calibration.genuine_scores, *calibration.impostor_scores]))
        same_x, same_y = self._kde_curve(
            calibration.genuine_scores,
            lower=lower,
            upper=upper,
            bandwidth_scale=settings.kde_bandwidth_scale,
        )
        diff_x, diff_y = self._kde_curve(
            calibration.impostor_scores,
            lower=lower,
            upper=upper,
            bandwidth_scale=settings.kde_bandwidth_scale,
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
            ("Ajuste KDE", "pronto" if calibration_summary.support_ready else "indisponível"),
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
                "Parâmetros KDE: "
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
            density_widget.set_series(series)
            layout.addWidget(density_widget, stretch=1)
        else:
            density_info = QTextBrowser(dialog)
            density_info.setPlainText(note or "A densidade calibrada não está disponível.")
            layout.addWidget(density_info)

        table = self._create_matches_table(dialog)
        self._populate_matches_table(table, result)
        layout.addWidget(table, stretch=1)
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
                left_info.setText("Selecione uma correspondência para visualizar a imagem do grupo Padrão.")
                right_info.setText("Selecione uma correspondência para visualizar a imagem do grupo Questionado.")
                return
            match = result.matches[row]
            left_entry = self._entry_by_id.get(match.left_entry_id)
            right_entry = self._entry_by_id.get(match.right_entry_id)
            left_label.set_image_path(
                left_entry.mesh_context_path if left_entry and left_entry.mesh_context_path else left_entry.mesh_crop_path if left_entry else None
            )
            right_label.set_image_path(
                right_entry.mesh_context_path if right_entry and right_entry.mesh_context_path else right_entry.mesh_crop_path if right_entry else None
            )
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

    def _preview_text(
        self,
        entry: FaceSetComparisonEntry | None,
        match: FaceSetComparisonMatch | None,
        set_label: str,
    ) -> str:
        if entry is None:
            return f"Selecione uma correspondência para visualizar a imagem do grupo {self._group_label(set_label)}."
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

    def _image_file_filter(self) -> str:
        patterns = " ".join(f"*{extension}" for extension in self._config.media.image_extensions)
        return f"Imagens suportadas ({patterns});;Todos os arquivos (*)"

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
