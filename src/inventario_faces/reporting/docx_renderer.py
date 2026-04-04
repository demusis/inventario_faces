from __future__ import annotations

from docx import Document
from docx.enum.table import WD_ALIGN_VERTICAL, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt

from inventario_faces.domain.config import AppConfig
from inventario_faces.domain.entities import FaceTrack, FileRecord, InventoryResult, KeyFrame, MediaInfoAttribute, ReportArtifacts
from inventario_faces.reporting.report_context import (
    keyframes_by_track,
    mean_pairwise_track_similarity,
    tracks_by_cluster,
)
from inventario_faces.reporting.report_support import (
    candidate_cluster_map,
    format_group_similarity,
    inventory_methodology_items,
    keyframe_reference_text,
    media_track_type_label,
    software_reference_abnt_text,
    technical_parameter_items,
)
from inventario_faces.utils.latex import format_seconds
from inventario_faces.utils.path_utils import ensure_directory
from inventario_faces.utils.time_utils import format_local_datetime


class DocxReportGenerator:
    def __init__(self, config: AppConfig) -> None:
        self._config = config

    def generate(self, result: InventoryResult) -> ReportArtifacts:
        report_directory = ensure_directory(result.run_directory / "report")
        docx_path = report_directory / "relatorio_forense.docx"
        document = self._build_document(result)
        document.save(docx_path)
        return ReportArtifacts(
            tex_path=report_directory / "relatorio_forense.tex",
            pdf_path=None,
            docx_path=docx_path,
        )

    def _build_document(self, result: InventoryResult) -> Document:
        document = Document()
        self._configure_styles(document)
        self._add_report_header(document, self._config.app.report_title, result.finished_at_utc)

        self._add_heading(document, "Resumo Executivo", 1)
        self._add_paragraph(
            document,
            (
                f"Foram catalogados {result.summary.total_files} arquivo(s), com "
                f"{result.summary.total_occurrences} detecção(ões), {result.summary.total_tracks} track(s), "
                f"{result.summary.total_keyframes} keyframe(s) e {result.summary.total_clusters} grupo(s) "
                f"de possíveis correspondências."
            ),
        )
        self._add_notice_box(
            document,
            "Advertência pericial",
            (
                "Correspondência facial automatizada não constitui prova conclusiva de identidade; "
                "os resultados são probabilísticos e exigem validação humana especializada."
            ),
        )
        self._add_notice_box(document, "Rastreabilidade forense", self._config.forensics.chain_of_custody_note)

        self._add_heading(document, "Metodologia", 1)
        for item in self._methodology_items(result):
            self._add_list_item(document, item)

        self._add_heading(document, "Resultados por Grupo", 1)
        self._add_group_sections(document, result)

        self._add_heading(document, "Estatísticas de tamanho das faces", 1)
        self._add_face_size_statistics(document, result)

        self._add_heading(document, "Anexo técnico", 1)
        self._add_heading(document, "Hashes SHA-512", 2)
        self._add_hashes_table(document, result.files)
        self._add_heading(document, "Metadados técnicos da mídia", 2)
        self._add_media_metadata_table(document, result.files)
        self._add_heading(document, "Parâmetros e melhorias aplicadas", 2)
        for item in self._technical_parameter_items(result):
            self._add_list_item(document, item)
        self._add_list_item(document, f"Referência do software: {software_reference_abnt_text()}")
        self._add_heading(document, "Registros de execução", 2)
        self._add_log_excerpt(document, result.logs_directory / "run.log")
        return document

    def _configure_styles(self, document: Document) -> None:
        document.styles["Normal"].font.name = "Cambria"
        document.styles["Normal"].font.size = Pt(11)

    def _add_report_header(self, document: Document, title_text: str, finished_at_utc) -> None:
        title = document.add_paragraph()
        title.alignment = WD_ALIGN_PARAGRAPH.CENTER
        title_run = title.add_run(title_text)
        title_run.bold = True
        title_run.font.size = Pt(16)

        meta = document.add_paragraph()
        meta.alignment = WD_ALIGN_PARAGRAPH.CENTER
        meta_run = meta.add_run(
            f"{self._config.app.organization} | Emitido em {format_local_datetime(finished_at_utc)}"
        )
        meta_run.font.size = Pt(10)

    def _methodology_items(self, result: InventoryResult) -> list[str]:
        return inventory_methodology_items(self._config, result.search)

    def _technical_parameter_items(self, result: InventoryResult) -> list[str]:
        return technical_parameter_items(self._config, result.search)

    def _add_group_sections(self, document: Document, result: InventoryResult) -> None:
        grouped_tracks = tracks_by_cluster(result)
        keyframes_map = keyframes_by_track(result)
        candidate_map = candidate_cluster_map(result.clusters)
        if not result.clusters:
            self._add_paragraph(document, "Nenhum grupo consolidado.")
            return
        for cluster in result.clusters:
            tracks = grouped_tracks.get(cluster.cluster_id, [])
            keyframe_count = sum(len(keyframes_map.get(track.track_id, [])) for track in tracks)
            mean_similarity = mean_pairwise_track_similarity(tracks)
            self._add_heading(document, f"Grupo {cluster.cluster_id}", 2)
            self._add_paragraph(
                document,
                (
                    f"Tracks={len(tracks)}; keyframes={keyframe_count}; ocorrências={len(cluster.occurrence_ids)}; "
                    f"similaridade média entre tracks={format_group_similarity(mean_similarity, len(tracks))}; "
                    "a tabela abaixo consolida as ocorrências representativas do grupo."
                ),
            )
            related_groups = candidate_map.get(cluster.cluster_id, [])
            self._add_paragraph(
                document,
                "Relações intergrupos: "
                + (", ".join(related_groups) if related_groups else "nenhuma acima do limiar configurado."),
            )
            self._add_group_track_table(document, tracks, keyframes_map)

    def _add_group_track_table(
        self,
        document: Document,
        tracks: list[FaceTrack],
        keyframes_map: dict[str, list[KeyFrame]],
    ) -> None:
        if not tracks:
            self._add_paragraph(document, "Nenhum track consolidado para este grupo.")
            return

        table = document.add_table(rows=1, cols=4)
        table.style = "Table Grid"
        table.alignment = WD_TABLE_ALIGNMENT.CENTER
        headers = table.rows[0].cells
        self._set_cell_text(headers[0], "Track", alignment=WD_ALIGN_PARAGRAPH.CENTER, bold=True)
        self._set_cell_text(headers[1], "Recorte", alignment=WD_ALIGN_PARAGRAPH.CENTER, bold=True)
        self._set_cell_text(headers[2], "Quadro de origem", alignment=WD_ALIGN_PARAGRAPH.CENTER, bold=True)
        self._set_cell_text(headers[3], "Metadados", alignment=WD_ALIGN_PARAGRAPH.CENTER, bold=True)

        for track in tracks[: self._config.reporting.max_tracks_per_group]:
            row = table.add_row().cells
            keyframe = self._select_representative_keyframe(track, keyframes_map)
            crop_path = keyframe.preview_path if keyframe is not None and keyframe.preview_path is not None else track.preview_path
            context_path = keyframe.context_image_path if keyframe is not None else None

            self._set_cell_text(row[0], track.track_id, alignment=WD_ALIGN_PARAGRAPH.CENTER)
            self._add_cell_image(row[1], crop_path, width=Inches(1.35))
            self._add_cell_image(row[2], context_path, width=Inches(2.2))

            metadata_lines = [
                f"Origem: {track.source_path.name}",
                f"Intervalo: {self._track_interval(track)}",
                f"Frames: {self._frame_interval(track)}",
                f"Detecções: {len(track.occurrence_ids)}",
                f"Keyframes: {len(track.keyframe_ids)}",
                f"Qualidade média: {track.quality_statistics.mean_quality_score:.3f}",
            ]
            if keyframe is not None:
                metadata_lines.append(keyframe_reference_text(keyframe))
            self._set_cell_text(
                row[3],
                "\n".join(metadata_lines),
                alignment=WD_ALIGN_PARAGRAPH.LEFT,
            )

        document.add_paragraph()

    def _add_face_size_statistics(self, document: Document, result: InventoryResult) -> None:
        table = document.add_table(rows=1, cols=4)
        table.style = "Table Grid"
        table.alignment = WD_TABLE_ALIGNMENT.CENTER
        headers = table.rows[0].cells
        self._set_cell_text(headers[0], "Conjunto", alignment=WD_ALIGN_PARAGRAPH.CENTER, bold=True)
        self._set_cell_text(headers[1], "Quantidade", alignment=WD_ALIGN_PARAGRAPH.CENTER, bold=True)
        self._set_cell_text(headers[2], "Média (px)", alignment=WD_ALIGN_PARAGRAPH.CENTER, bold=True)
        self._set_cell_text(headers[3], "Desvio padrão (px)", alignment=WD_ALIGN_PARAGRAPH.CENTER, bold=True)

        rows = [
            (
                "Todas as faces detectadas",
                result.summary.total_detected_face_sizes.count,
                result.summary.total_detected_face_sizes.mean_pixels,
                result.summary.total_detected_face_sizes.stddev_pixels,
            ),
            (
                "Faces filtradas e mantidas",
                result.summary.selected_face_sizes.count,
                result.summary.selected_face_sizes.mean_pixels,
                result.summary.selected_face_sizes.stddev_pixels,
            ),
        ]
        for label, count, mean_pixels, stddev_pixels in rows:
            cells = table.add_row().cells
            self._set_cell_text(cells[0], label)
            self._set_cell_text(cells[1], str(count), alignment=WD_ALIGN_PARAGRAPH.CENTER)
            self._set_cell_text(cells[2], self._format_face_size_value(mean_pixels), alignment=WD_ALIGN_PARAGRAPH.CENTER)
            self._set_cell_text(
                cells[3],
                self._format_face_size_value(stddev_pixels),
                alignment=WD_ALIGN_PARAGRAPH.CENTER,
            )
        document.add_paragraph()

    def _select_representative_keyframe(
        self,
        track: FaceTrack,
        keyframes_map: dict[str, list[KeyFrame]],
    ) -> KeyFrame | None:
        keyframes = keyframes_map.get(track.track_id, [])
        if not keyframes:
            return None
        for keyframe in keyframes:
            if track.best_occurrence_id is not None and keyframe.occurrence_id == track.best_occurrence_id:
                return keyframe
        return keyframes[0]

    def _track_interval(self, track: FaceTrack) -> str:
        return f"{format_seconds(track.start_time)} - {format_seconds(track.end_time)}"

    def _frame_interval(self, track: FaceTrack) -> str:
        start = "-" if track.start_frame is None else f"{track.start_frame:06d}"
        end = "-" if track.end_frame is None else f"{track.end_frame:06d}"
        return f"{start} - {end}"

    def _add_heading(self, document: Document, text: str, level: int) -> None:
        document.add_heading(text, level=level)

    def _add_paragraph(self, document: Document, text: str) -> None:
        document.add_paragraph(text)

    def _add_list_item(self, document: Document, text: str) -> None:
        paragraph = document.add_paragraph(style="List Number")
        paragraph.add_run(text)

    def _add_notice_box(self, document: Document, title: str, body: str) -> None:
        table = document.add_table(rows=1, cols=1)
        table.style = "Table Grid"
        table.alignment = WD_TABLE_ALIGNMENT.CENTER
        cell = table.cell(0, 0)
        paragraph = cell.paragraphs[0]
        title_run = paragraph.add_run(f"{title}. ")
        title_run.bold = True
        paragraph.add_run(body)
        document.add_paragraph()

    def _add_hashes_table(self, document: Document, files: list[FileRecord]) -> None:
        table = document.add_table(rows=1, cols=2)
        table.style = "Table Grid"
        table.alignment = WD_TABLE_ALIGNMENT.CENTER
        self._set_cell_text(table.rows[0].cells[0], "Arquivo", alignment=WD_ALIGN_PARAGRAPH.CENTER, bold=True)
        self._set_cell_text(table.rows[0].cells[1], "SHA-512", alignment=WD_ALIGN_PARAGRAPH.CENTER, bold=True)
        for item in files:
            row = table.add_row().cells
            self._set_cell_text(row[0], str(item.path))
            self._set_cell_text(row[1], item.sha512)
        document.add_paragraph()

    def _add_media_metadata_table(self, document: Document, files: list[FileRecord]) -> None:
        table = document.add_table(rows=1, cols=2)
        table.style = "Table Grid"
        table.alignment = WD_TABLE_ALIGNMENT.CENTER
        table.autofit = False
        self._set_cell_text(table.rows[0].cells[0], "Arquivo", alignment=WD_ALIGN_PARAGRAPH.CENTER, bold=True)
        self._set_cell_text(table.rows[0].cells[1], "Características", alignment=WD_ALIGN_PARAGRAPH.CENTER, bold=True)
        for cell in table.columns[0].cells:
            cell.width = Inches(4.2)
        for cell in table.columns[1].cells:
            cell.width = Inches(2.4)

        media_files = [item for item in files if item.media_type.value in {"image", "video"}]
        if not media_files:
            row = table.add_row().cells
            self._set_cell_text(row[0], "-")
            self._set_cell_text(row[1], "Nenhuma mídia elegível para extração interna de metadados.")
            document.add_paragraph()
            return

        for item in media_files:
            if item.media_info_tracks:
                row = table.add_row().cells
                self._set_cell_text(row[0], str(item.path))
                self._set_cell_text(row[1], self._media_info_file_description(item.media_info_tracks))
            else:
                row = table.add_row().cells
                self._set_cell_text(row[0], str(item.path))
                self._set_cell_text(row[1], item.media_info_error or "Metadados técnicos indisponíveis.")
        document.add_paragraph()

    def _media_info_file_description(self, tracks: tuple[object, ...]) -> str:
        descriptions: list[str] = []
        show_type_label = len(tracks) > 1
        for track in tracks:
            track_title = media_track_type_label(track.track_type)
            description = self._media_info_track_description(
                track.attributes,
                track_title=track_title,
                show_type_label=show_type_label and track_title != "Arquivo",
            )
            descriptions.append(description)
        return "\n\n".join(descriptions)

    def _add_log_excerpt(self, document: Document, log_path: Path, max_lines: int = 60) -> None:
        if not log_path.exists():
            self._add_paragraph(document, "Registro principal não encontrado.")
            return
        lines = log_path.read_text(encoding="utf-8", errors="ignore").splitlines()[-max_lines:]
        self._add_paragraph(document, "\n".join(lines))

    def _set_cell_text(
        self,
        cell,
        text: str,
        *,
        alignment: WD_ALIGN_PARAGRAPH = WD_ALIGN_PARAGRAPH.LEFT,
        bold: bool = False,
    ) -> None:
        cell.text = ""
        cell.vertical_alignment = WD_ALIGN_VERTICAL.TOP
        paragraph = cell.paragraphs[0]
        paragraph.alignment = alignment
        run = paragraph.add_run(text)
        run.bold = bold

    def _add_cell_image(self, cell, image_path: Path | None, *, width: Inches) -> None:
        cell.text = ""
        cell.vertical_alignment = WD_ALIGN_VERTICAL.TOP
        paragraph = cell.paragraphs[0]
        paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        if image_path is None or not image_path.exists():
            paragraph.add_run("Artefato não disponível")
            return
        paragraph.add_run().add_picture(str(image_path), width=width)

    def _format_face_size_value(self, value: float | None) -> str:
        return "-" if value is None else f"{value:.2f}"

    def _media_info_track_description(
        self,
        attributes: tuple[MediaInfoAttribute, ...],
        *,
        track_title: str | None = None,
        show_type_label: bool = False,
    ) -> str:
        if not attributes:
            return "Nenhuma característica disponível."
        lines: list[str] = []
        if show_type_label and track_title:
            lines.append(track_title)
        lines.extend(f"{attribute.label}: {attribute.value}" for attribute in attributes)
        return "\n".join(lines)
