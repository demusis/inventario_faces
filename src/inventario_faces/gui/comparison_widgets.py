from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QPainter, QPen, QPixmap, QPolygonF
from PySide6.QtWidgets import QLabel, QWidget

from inventario_faces.domain.entities import BoundingBox
from inventario_faces.gui.comparison_statistics import (
    _DistributionSeries,
    _expanded_score_range,
    _histogram_density,
)
from inventario_faces.infrastructure.face_mesh_renderer import build_face_mesh_geometry


class AdaptiveImageLabel(QLabel):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("Sem imagem", parent)
        self._pixmap = QPixmap()
        self._mesh_points: tuple[tuple[int, int], ...] = ()
        self._mesh_edges: tuple[tuple[tuple[int, int], tuple[int, int]], ...] = ()
        self._mesh_bbox: QRectF | None = None
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(320, 240)
        self.setStyleSheet(
            "background:#f8fafc; border:1px solid #d7e0ea; border-radius:10px; color:#64748b;"
        )

    def set_image_path(self, path: Path | None) -> None:
        self._mesh_points = ()
        self._mesh_edges = ()
        self._mesh_bbox = None
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

    def set_mesh_image(
        self,
        path: Path | None,
        *,
        landmarks: tuple[tuple[float, float], ...] = (),
        bbox: BoundingBox | None = None,
        translate: tuple[float, float] = (0.0, 0.0),
        draw_bbox: bool = False,
    ) -> None:
        self.set_image_path(path)
        if self._pixmap.isNull():
            return
        points, edges = build_face_mesh_geometry(
            landmarks,
            width=self._pixmap.width(),
            height=self._pixmap.height(),
            translate=translate,
        )
        self._mesh_points = tuple(points)
        self._mesh_edges = tuple(edges)
        if draw_bbox and bbox is not None:
            self._mesh_bbox = QRectF(
                float(bbox.x1 + translate[0]),
                float(bbox.y1 + translate[1]),
                float(max(0.0, bbox.width)),
                float(max(0.0, bbox.height)),
            )
        else:
            self._mesh_bbox = None
        self.update()

    def paintEvent(self, event) -> None:  # type: ignore[override]
        super().paintEvent(event)
        if self._pixmap.isNull() or (not self._mesh_points and self._mesh_bbox is None):
            return

        target = self._pixmap_target_rect()
        if target.isEmpty():
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)

        mesh_pen = QPen(QColor(34, 197, 94), 1.0)
        mesh_pen.setCosmetic(True)
        painter.setPen(mesh_pen)
        for start, end in self._mesh_edges:
            painter.drawLine(
                self._map_point_to_target(start, target),
                self._map_point_to_target(end, target),
            )

        if self._mesh_bbox is not None:
            bbox_pen = QPen(QColor(220, 38, 38), 1.0)
            bbox_pen.setCosmetic(True)
            painter.setPen(bbox_pen)
            painter.setBrush(Qt.NoBrush)
            painter.drawRect(self._map_rect_to_target(self._mesh_bbox, target))

        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(250, 204, 21))
        for point in self._mesh_points:
            mapped = self._map_point_to_target(point, target)
            painter.drawEllipse(mapped, 1.6, 1.6)

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

    def _pixmap_target_rect(self) -> QRectF:
        if self._pixmap.isNull():
            return QRectF()
        contents = QRectF(self.contentsRect())
        scaled = self._pixmap.size().scaled(self.contentsRect().size(), Qt.KeepAspectRatio)
        x = contents.left() + max(0.0, (contents.width() - scaled.width()) / 2.0)
        y = contents.top() + max(0.0, (contents.height() - scaled.height()) / 2.0)
        return QRectF(x, y, float(scaled.width()), float(scaled.height()))

    def _map_point_to_target(self, point: tuple[int, int], target: QRectF) -> QPointF:
        if self._pixmap.isNull() or self._pixmap.width() == 0 or self._pixmap.height() == 0:
            return QPointF()
        return QPointF(
            target.left() + (float(point[0]) / float(self._pixmap.width())) * target.width(),
            target.top() + (float(point[1]) / float(self._pixmap.height())) * target.height(),
        )

    def _map_rect_to_target(self, rect: QRectF, target: QRectF) -> QRectF:
        if self._pixmap.isNull() or self._pixmap.width() == 0 or self._pixmap.height() == 0:
            return QRectF()
        scale_x = target.width() / float(self._pixmap.width())
        scale_y = target.height() / float(self._pixmap.height())
        return QRectF(
            target.left() + (rect.left() * scale_x),
            target.top() + (rect.top() * scale_y),
            rect.width() * scale_x,
            rect.height() * scale_y,
        )


class SimilarityDistributionWidget(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._series: list[_DistributionSeries] = []
        self._candidate = 0.0
        self._assignment = 0.0
        self._observed_score: float | None = None
        self._overall_ci_low: float | None = None
        self._overall_ci_high: float | None = None
        self._overall_mean: float | None = None
        self._show_threshold_markers = True
        self._show_mean_marker = True
        self.setMinimumHeight(360)

    def set_distribution(
        self,
        series: list[_DistributionSeries],
        *,
        candidate_threshold: float,
        assignment_threshold: float,
        observed_score: float | None,
        mean_value: float | None,
        ci_low: float | None,
        ci_high: float | None,
        show_threshold_markers: bool = True,
        show_mean_marker: bool = True,
    ) -> None:
        self._series = list(series)
        self._candidate = candidate_threshold
        self._assignment = assignment_threshold
        self._observed_score = observed_score
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
        painter.fillRect(self.rect(), QColor('#ffffff'))

        outer = self.rect().adjusted(12, 12, -12, -12)
        painter.setPen(QPen(QColor('#d7e0ea'), 1))
        painter.drawRoundedRect(outer, 10, 10)

        drawable_series = [series for series in self._series if series.sufficient and series.kde_x and series.kde_y]
        marker_count = len([series for series in drawable_series if series.mean is not None])
        if self._overall_mean is not None and self._show_mean_marker:
            marker_count += 1
        if self._observed_score is not None:
            marker_count += 1
        marker_row_count = self._marker_badge_row_count(marker_count)
        legend_bottom = self._draw_distribution_legend(painter, outer, drawable_series)
        top_margin = max(116, (legend_bottom - outer.top()) + 18 + (marker_row_count * 24) + 12)
        plot = outer.adjusted(56, top_margin, -18, -72)
        if plot.width() <= 0 or plot.height() <= 0:
            return
        if not drawable_series:
            painter.setPen(QPen(QColor('#64748b'), 1))
            painter.drawText(plot, Qt.AlignCenter, 'A distribuição não será exibida sem repetição suficiente e variabilidade.')
            return

        all_values = [value for series in drawable_series for value in series.values]
        axis_values = [
            *all_values,
            self._candidate,
            self._assignment,
            *[series.mean for series in drawable_series if series.mean is not None],
        ]
        if self._overall_ci_low is not None:
            axis_values.append(self._overall_ci_low)
        if self._overall_ci_high is not None:
            axis_values.append(self._overall_ci_high)
        if self._observed_score is not None:
            axis_values.append(self._observed_score)
        if self._overall_mean is not None:
            axis_values.append(self._overall_mean)
        lower, upper = _expanded_score_range(axis_values, minimum_span=0.2)

        if self._show_threshold_markers:
            candidate_x = self._map_x(self._candidate, plot, lower, upper)
            assignment_x = self._map_x(self._assignment, plot, lower, upper)
            self._fill_zone(painter, plot, float(plot.left()), candidate_x, QColor(124, 58, 237, 18))
            self._fill_zone(painter, plot, candidate_x, assignment_x, QColor(180, 83, 9, 18))
            self._fill_zone(painter, plot, assignment_x, float(plot.right()), QColor(15, 118, 110, 16))

        if self._overall_ci_low is not None and self._overall_ci_high is not None:
            x1 = self._map_x(self._overall_ci_low, plot, lower, upper)
            x2 = self._map_x(self._overall_ci_high, plot, lower, upper)
            painter.fillRect(
                int(min(x1, x2)),
                plot.top(),
                int(abs(x2 - x1)),
                plot.height(),
                QColor(71, 85, 105, 26),
            )

        painter.setPen(QPen(QColor('#e7edf4'), 1))
        for ratio in (0.0, 0.25, 0.5, 0.75, 1.0):
            y = plot.bottom() - (plot.height() * ratio)
            painter.drawLine(plot.left(), int(y), plot.right(), int(y))
            painter.setPen(QPen(QColor('#64748b'), 1))
            painter.drawText(outer.left() + 4, int(y - 8), 42, 16, Qt.AlignRight | Qt.AlignVCenter, f'{int(ratio * 100)}%')
            painter.setPen(QPen(QColor('#e7edf4'), 1))

        max_density = max(max(series.kde_y) for series in drawable_series if series.kde_y) or 1.0
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
            painter.setBrush(QColor(color.red(), color.green(), color.blue(), 44))
            painter.drawPolygon(QPolygonF(fill_polygon))
            painter.setBrush(Qt.NoBrush)
            painter.setPen(QPen(color, 2))
            painter.drawPolyline(QPolygonF(points))

        if self._show_threshold_markers:
            self._draw_vertical_marker(
                painter,
                plot,
                self._map_x(self._candidate, plot, lower, upper),
                QColor('#b45309'),
                dashed=False,
                width=1,
            )
            self._draw_vertical_marker(
                painter,
                plot,
                self._map_x(self._assignment, plot, lower, upper),
                QColor('#0f766e'),
                dashed=False,
                width=1,
            )

        marker_badges: list[tuple[float, QColor, str]] = []
        for series in drawable_series:
            if series.mean is None:
                continue
            self._draw_vertical_marker(
                painter,
                plot,
                self._map_x(series.mean, plot, lower, upper),
                QColor(series.color),
                dashed=True,
                width=2,
            )
            marker_badges.append(
                (
                    series.mean,
                    QColor(series.color),
                    f'{self._series_mean_short_label(series)} {series.mean:.4f}',
                )
            )

        if self._overall_mean is not None and self._show_mean_marker:
            self._draw_vertical_marker(
                painter,
                plot,
                self._map_x(self._overall_mean, plot, lower, upper),
                QColor('#1e293b'),
                dashed=True,
                width=2,
            )
            marker_badges.append((self._overall_mean, QColor('#1e293b'), f'μ geral {self._overall_mean:.4f}'))
        if self._observed_score is not None:
            self._draw_vertical_marker(
                painter,
                plot,
                self._map_x(self._observed_score, plot, lower, upper),
                QColor('#2563eb'),
                dashed=False,
                width=2,
            )
            marker_badges.append((self._observed_score, QColor('#2563eb'), f'melhor {self._observed_score:.4f}'))

        self._draw_marker_badges(painter, plot, lower, upper, marker_badges)
        if self._show_threshold_markers:
            self._draw_bottom_badges(
                painter,
                plot,
                lower,
                upper,
                [
                    (self._candidate, QColor('#b45309'), f'limiar cand. {self._candidate:.4f}'),
                    (self._assignment, QColor('#0f766e'), f'limiar atrib. {self._assignment:.4f}'),
                ],
            )

        painter.setPen(QPen(QColor('#475569'), 1))
        painter.drawLine(plot.left(), plot.bottom(), plot.right(), plot.bottom())
        for ratio in (0.0, 0.25, 0.5, 0.75, 1.0):
            value = lower + ((upper - lower) * ratio)
            x = self._map_x(value, plot, lower, upper)
            painter.drawText(int(x - 20), outer.bottom() - 12, 44, 14, Qt.AlignCenter, f'{value:.2f}')

    def _map_x(self, value: float, plot, lower: float, upper: float) -> float:
        if upper <= lower:
            return float(plot.left())
        return plot.left() + (plot.width() * ((value - lower) / (upper - lower)))

    def _draw_vertical_marker(
        self,
        painter: QPainter,
        plot,
        x: float,
        color: QColor,
        *,
        dashed: bool,
        width: int,
    ) -> None:
        pen = QPen(color, width)
        pen.setStyle(Qt.DashLine if dashed else Qt.SolidLine)
        painter.setPen(pen)
        painter.drawLine(int(x), plot.top(), int(x), plot.bottom())

    def _fill_zone(self, painter: QPainter, plot, left: float, right: float, color: QColor) -> None:
        if right <= left:
            return
        painter.fillRect(QRectF(left, plot.top(), right - left, plot.height()), color)

    def _draw_distribution_legend(
        self,
        painter: QPainter,
        outer,
        drawable_series: list[_DistributionSeries],
    ) -> int:
        legend_left = outer.left() + 16
        legend_top = outer.top() + 16
        legend_right = outer.right() - 16
        cursor_x = legend_left
        cursor_y = legend_top
        row_height = 20
        for series in drawable_series:
            width = self._legend_color_item_width(painter, series.label)
            if cursor_x + width > legend_right:
                cursor_x = legend_left
                cursor_y += row_height
            self._draw_color_legend_item(painter, cursor_x, cursor_y, QColor(series.color), series.label)
            cursor_x += width

        cursor_x = legend_left
        cursor_y += row_height + 4
        items = [
            (QColor('#1e293b'), 'Linhas tracejadas = médias / centros de gravidade', True),
            (QColor('#b45309'), 'Linhas sólidas = limiares', False),
            (QColor('#2563eb'), 'Linha azul = melhor escore', False),
        ]
        for color, label, dashed in items:
            width = self._legend_line_item_width(painter, label)
            if cursor_x + width > legend_right:
                cursor_x = legend_left
                cursor_y += row_height
            self._draw_line_legend_item(painter, cursor_x, cursor_y, color, label, dashed=dashed)
            cursor_x += width

        band_label = 'Faixa cinza = IC bootstrap da média geral'
        band_width = self._legend_band_item_width(painter, band_label)
        if cursor_x + band_width > legend_right:
            cursor_x = legend_left
            cursor_y += row_height
        self._draw_band_legend_item(painter, cursor_x, cursor_y, QColor(71, 85, 105, 26), band_label)
        return cursor_y + row_height

    def _legend_color_item_width(self, painter: QPainter, label: str) -> int:
        return painter.fontMetrics().horizontalAdvance(label) + 34

    def _legend_line_item_width(self, painter: QPainter, label: str) -> int:
        return painter.fontMetrics().horizontalAdvance(label) + 46

    def _legend_band_item_width(self, painter: QPainter, label: str) -> int:
        return painter.fontMetrics().horizontalAdvance(label) + 42

    def _draw_color_legend_item(self, painter: QPainter, x: int, y: int, color: QColor, label: str) -> None:
        painter.fillRect(x, y + 2, 12, 12, color)
        painter.setPen(QPen(QColor('#334155'), 1))
        painter.drawText(x + 18, y, self._legend_color_item_width(painter, label), 16, Qt.AlignLeft | Qt.AlignVCenter, label)

    def _draw_line_legend_item(
        self,
        painter: QPainter,
        x: int,
        y: int,
        color: QColor,
        label: str,
        *,
        dashed: bool,
    ) -> None:
        pen = QPen(color, 2)
        pen.setStyle(Qt.DashLine if dashed else Qt.SolidLine)
        painter.setPen(pen)
        painter.drawLine(x, y + 8, x + 18, y + 8)
        painter.setPen(QPen(QColor('#334155'), 1))
        painter.drawText(x + 24, y, self._legend_line_item_width(painter, label), 16, Qt.AlignLeft | Qt.AlignVCenter, label)

    def _draw_band_legend_item(self, painter: QPainter, x: int, y: int, color: QColor, label: str) -> None:
        painter.setPen(QPen(QColor('#94a3b8'), 1))
        painter.setBrush(color)
        painter.drawRoundedRect(QRectF(x, y + 2, 16, 12), 3, 3)
        painter.setBrush(Qt.NoBrush)
        painter.setPen(QPen(QColor('#334155'), 1))
        painter.drawText(x + 24, y, self._legend_band_item_width(painter, label), 16, Qt.AlignLeft | Qt.AlignVCenter, label)

    def _series_mean_short_label(self, series: _DistributionSeries) -> str:
        mapping = {
            'assignment': 'μ PxQ atrib.',
            'candidate': 'μ PxQ cand.',
            'below_threshold': 'μ PxQ abaixo',
        }
        return mapping.get(series.classification, f'μ {series.label.lower()}')

    def _draw_marker_badges(
        self,
        painter: QPainter,
        plot,
        lower: float,
        upper: float,
        markers: list[tuple[float, QColor, str]],
    ) -> None:
        if not markers:
            return
        metrics = painter.fontMetrics()
        row_count = self._marker_badge_row_count(len(markers))
        row_tops = [plot.top() - (24 * (row_count - index)) - 4 for index in range(row_count)]
        last_right = [plot.left() - 9999.0 for _ in row_tops]
        for value, color, text in sorted(markers, key=lambda item: item[0]):
            x = self._map_x(value, plot, lower, upper)
            width = metrics.horizontalAdvance(text) + 16
            left = max(plot.left() + 4, min(x - (width / 2), plot.right() - width - 4))
            chosen_row: int | None = None
            for index in range(len(row_tops)):
                if left > last_right[index] + 8:
                    chosen_row = index
                    break
            if chosen_row is None:
                chosen_row = min(range(len(row_tops)), key=lambda index: last_right[index])
                left = min(max(left, last_right[chosen_row] + 8), plot.right() - width - 4)
            rect = QRectF(left, row_tops[chosen_row], width, 18)
            painter.setPen(QPen(color, 1))
            painter.setBrush(QColor(255, 255, 255, 235))
            painter.drawRoundedRect(rect, 8, 8)
            painter.setPen(QPen(color, 1))
            painter.drawText(rect, Qt.AlignCenter, text)
            last_right[chosen_row] = rect.right()

    def _marker_badge_row_count(self, marker_count: int) -> int:
        if marker_count <= 2:
            return 1
        if marker_count <= 5:
            return 2
        return 3

    def _draw_bottom_badges(
        self,
        painter: QPainter,
        plot,
        lower: float,
        upper: float,
        markers: list[tuple[float, QColor, str]],
    ) -> None:
        metrics = painter.fontMetrics()
        row_tops = [plot.bottom() + 8, plot.bottom() + 28]
        last_right = [plot.left() - 9999.0 for _ in row_tops]
        for value, color, text in sorted(markers, key=lambda item: item[0]):
            x = self._map_x(value, plot, lower, upper)
            width = metrics.horizontalAdvance(text) + 16
            left = max(plot.left() + 4, min(x - (width / 2), plot.right() - width - 4))
            chosen_row = 0
            for index in range(len(row_tops)):
                if left > last_right[index] + 8:
                    chosen_row = index
                    break
            else:
                chosen_row = min(range(len(row_tops)), key=lambda index: last_right[index])
                left = min(max(left, last_right[chosen_row] + 8), plot.right() - width - 4)
            rect = QRectF(left, row_tops[chosen_row], width, 18)
            painter.setPen(QPen(color, 1))
            painter.setBrush(QColor(255, 255, 255, 238))
            painter.drawRoundedRect(rect, 8, 8)
            painter.setPen(QPen(color, 1))
            painter.drawText(rect, Qt.AlignCenter, text)
            last_right[chosen_row] = rect.right()

class LikelihoodRatioDensityWidget(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._series: list[_DistributionSeries] = []
        self._observed_score: float | None = None
        self.setMinimumHeight(260)

    def set_series(self, series: list[_DistributionSeries], *, observed_score: float | None = None) -> None:
        self._series = list(series)
        self._observed_score = observed_score
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
        lower, upper = _expanded_score_range(
            all_values,
            observed_score=self._observed_score,
            minimum_span=0.2,
        )

        painter.setPen(QPen(QColor("#e7edf4"), 1))
        for ratio in (0.25, 0.5, 0.75):
            y = plot.bottom() - (plot.height() * ratio)
            painter.drawLine(plot.left(), int(y), plot.right(), int(y))

        histogram_series: list[tuple[_DistributionSeries, tuple[float, ...], tuple[float, ...]]] = []
        histogram_max_density = 0.0
        for series in drawable_series:
            edges, histogram = _histogram_density(series.values, lower=lower, upper=upper)
            histogram_series.append((series, edges, histogram))
            if histogram:
                histogram_max_density = max(histogram_max_density, max(histogram))

        curve_max_density = max(max(series.kde_y) for series in drawable_series if series.kde_y) or 1.0
        max_density = max(curve_max_density, histogram_max_density, 1.0)
        legend_x = plot.left() + 10
        legend_y = plot.top() + 8

        for series, edges, histogram in histogram_series:
            if not edges or not histogram:
                continue
            color = QColor(series.color)
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(color.red(), color.green(), color.blue(), 26))
            for left_edge, right_edge, density in zip(edges[:-1], edges[1:], histogram):
                if density <= 0.0:
                    continue
                x1 = self._map_x(left_edge, plot, lower, upper)
                x2 = self._map_x(right_edge, plot, lower, upper)
                top = plot.bottom() - ((density / max_density) * plot.height())
                rect = QRectF(min(x1, x2), top, max(1.0, abs(x2 - x1)), plot.bottom() - top)
                painter.drawRect(rect)

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

        painter.setBrush(QColor(148, 163, 184, 38))
        painter.setPen(QPen(QColor("#94a3b8"), 1))
        painter.drawRect(QRectF(legend_x, legend_y, 12, 12))
        painter.setPen(QPen(QColor("#334155"), 1))
        painter.drawText(
            legend_x + 18,
            legend_y - 1,
            240,
            14,
            Qt.AlignLeft | Qt.AlignVCenter,
            "barras: histograma bruto",
        )

        if self._observed_score is not None:
            self._draw_marker(painter, plot, lower, upper, self._observed_score, QColor("#2563eb"))

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
    ) -> None:
        x = self._map_x(value, plot, lower, upper)
        pen = QPen(color, 2)
        pen.setStyle(Qt.DashLine)
        painter.setPen(pen)
        painter.drawLine(int(x), plot.top(), int(x), plot.bottom())
