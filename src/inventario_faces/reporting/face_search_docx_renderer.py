from __future__ import annotations

from pathlib import Path

from docx import Document
from docx.enum.table import WD_ALIGN_VERTICAL, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt

from inventario_faces.domain.config import AppConfig
from inventario_faces.domain.entities import FaceSearchMatch, FaceSearchResult, ReportArtifacts
from inventario_faces.reporting.report_support import (
    face_search_methodology_items,
    software_reference_abnt_text,
    technical_parameter_items,
)
from inventario_faces.utils.latex import format_seconds
from inventario_faces.utils.path_utils import ensure_directory
from inventario_faces.utils.time_utils import format_local_datetime


class FaceSearchDocxReportGenerator:
    def __init__(self, config: AppConfig) -> None:
        self._config = config

    def generate(self, result: FaceSearchResult) -> ReportArtifacts:
        report_directory = ensure_directory(result.inventory_result.run_directory / "report")
        docx_path = report_directory / "relatorio_busca_por_face.docx"
        document = self._build_document(result)
        document.save(docx_path)
        return ReportArtifacts(
            tex_path=report_directory / "relatorio_busca_por_face.tex",
            pdf_path=None,
            docx_path=docx_path,
        )

    def _build_document(self, result: FaceSearchResult) -> Document:
        document = Document()
        document.styles["Normal"].font.name = "Cambria"
        document.styles["Normal"].font.size = Pt(11)

        self._add_report_header(
            document,
            f"{self._config.app.report_title} - Busca por Face",
            result.inventory_result.finished_at_utc,
        )

        self._add_heading(document, "Resumo Executivo", 1)
        self._add_paragraph(
            document,
            (
                f"A consulta utilizou o arquivo {result.query.source_path} e retornou "
                f"{result.summary.compatible_tracks} track(s) compatíveis, distribuídos em "
                f"{result.summary.compatible_clusters} grupo(s), com limiar probabilístico de "
                f"{result.summary.compatibility_threshold:.2f}."
            ),
        )
        self._add_notice(
            document,
            "Advertência pericial",
            "A busca abaixo aponta apenas compatibilidade probabilística e não constitui prova conclusiva de identidade.",
        )

        self._add_heading(document, "Metodologia da Busca", 1)
        for item in face_search_methodology_items(result.query.selected_track_id):
            self._add_list_item(document, item)

        self._add_heading(document, "Face Consultada", 1)
        self._add_query_table(document, result)

        self._add_heading(document, "Resultados Compatíveis", 1)
        self._add_matches_table(document, result.matches)

        self._add_heading(document, "Anexo técnico", 1)
        self._add_heading(document, "Parâmetros e melhorias aplicadas", 2)
        for item in technical_parameter_items(self._config, result.inventory_result.search):
            self._add_list_item(document, item)
        self._add_list_item(document, f"Referência do software: {software_reference_abnt_text()}")
        self._add_heading(document, "Rastreabilidade da busca", 2)
        report_path = result.inventory_result.report.pdf_path or result.inventory_result.report.docx_path or result.inventory_result.report.tex_path
        self._add_paragraph(document, f"Inventário de referência usado na busca: {report_path}")
        self._add_paragraph(document, f"Exportação estruturada da busca: {result.export_path or '-'}")
        return document

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

    def _add_query_table(self, document: Document, result: FaceSearchResult) -> None:
        table = document.add_table(rows=1, cols=3)
        table.style = "Table Grid"
        table.alignment = WD_TABLE_ALIGNMENT.CENTER
        self._set_cell_text(table.rows[0].cells[0], "Recorte da consulta", alignment=WD_ALIGN_PARAGRAPH.CENTER, bold=True)
        self._set_cell_text(table.rows[0].cells[1], "Quadro de origem da consulta", alignment=WD_ALIGN_PARAGRAPH.CENTER, bold=True)
        self._set_cell_text(table.rows[0].cells[2], "Metadados da consulta", alignment=WD_ALIGN_PARAGRAPH.CENTER, bold=True)

        row = table.add_row().cells
        self._add_cell_image(row[0], result.query.crop_path, width=Inches(1.5))
        self._add_cell_image(row[1], result.query.context_image_path, width=Inches(2.5))
        quality = "-" if result.query.quality_score is None else f"{result.query.quality_score:.3f}"
        self._set_cell_text(
            row[2],
            "\n".join(
                [
                    f"Arquivo consultado: {result.query.source_path}",
                    f"SHA-512: {result.query.sha512}",
                    f"Faces elegíveis detectadas: {result.query.detected_face_count}",
                    f"Track selecionado: {result.query.selected_track_id}",
                    f"Ocorrência selecionada: {result.query.selected_occurrence_id}",
                    f"Keyframe de referência: {result.query.selected_keyframe_id or '-'}",
                    f"Qualidade da face consultada: {quality}",
                ]
            ),
        )
        document.add_paragraph()

    def _add_matches_table(self, document: Document, matches: list[FaceSearchMatch]) -> None:
        if not matches:
            self._add_paragraph(document, "Nenhum track compatível superou o limiar probabilístico configurado para a consulta.")
            return

        table = document.add_table(rows=1, cols=6)
        table.style = "Table Grid"
        table.alignment = WD_TABLE_ALIGNMENT.CENTER
        headers = table.rows[0].cells
        for index, label in enumerate(("Pos.", "Grupo", "Track", "Recorte", "Quadro de origem", "Metadados")):
            self._set_cell_text(headers[index], label, alignment=WD_ALIGN_PARAGRAPH.CENTER, bold=True)

        for match in matches:
            row = table.add_row().cells
            self._set_cell_text(row[0], str(match.rank), alignment=WD_ALIGN_PARAGRAPH.CENTER)
            self._set_cell_text(row[1], match.cluster_id or "-", alignment=WD_ALIGN_PARAGRAPH.CENTER)
            self._set_cell_text(row[2], match.track_id, alignment=WD_ALIGN_PARAGRAPH.CENTER)
            self._add_cell_image(row[3], match.crop_path, width=Inches(1.2))
            self._add_cell_image(row[4], match.context_image_path, width=Inches(2.1))
            self._set_cell_text(
                row[5],
                "\n".join(self._match_metadata_lines(match)),
            )
        document.add_paragraph()

    def _match_metadata_lines(self, match: FaceSearchMatch) -> list[str]:
        occurrence_score = "-" if match.occurrence_score is None else f"{match.occurrence_score:.3f}"
        cluster_score = "-" if match.cluster_score is None else f"{match.cluster_score:.3f}"
        return [
            f"Origem: {match.source_path.name}",
            f"Ocorrência: {match.occurrence_id or '-'}",
            f"Instante da ocorrência: {format_seconds(match.timestamp_seconds)}",
            f"Intervalo do track: {format_seconds(match.track_start_time)} - {format_seconds(match.track_end_time)}",
            f"Quadro da ocorrência: {match.frame_index:06d}" if match.frame_index is not None else "Quadro da ocorrência: -",
            f"Similaridade do grupo: {cluster_score}",
            f"Similaridade do track: {match.track_score:.3f}",
            f"Similaridade da ocorrência: {occurrence_score}",
        ]

    def _add_heading(self, document: Document, text: str, level: int) -> None:
        document.add_heading(text, level=level)

    def _add_paragraph(self, document: Document, text: str) -> None:
        document.add_paragraph(text)

    def _add_list_item(self, document: Document, text: str) -> None:
        paragraph = document.add_paragraph(style="List Number")
        paragraph.add_run(text)

    def _add_notice(self, document: Document, title: str, body: str) -> None:
        table = document.add_table(rows=1, cols=1)
        table.style = "Table Grid"
        table.alignment = WD_TABLE_ALIGNMENT.CENTER
        cell = table.cell(0, 0)
        paragraph = cell.paragraphs[0]
        title_run = paragraph.add_run(f"{title}. ")
        title_run.bold = True
        paragraph.add_run(body)
        document.add_paragraph()

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
