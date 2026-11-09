from __future__ import annotations

from pathlib import Path

from docx import Document
from docx.enum.table import WD_ALIGN_VERTICAL, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt

from inventario_faces.domain.config import AppConfig
from inventario_faces.domain.entities import (
    FaceSetComparisonEntry,
    FaceSetComparisonInput,
    FaceSetComparisonMatch,
    FaceSetComparisonResult,
)
from inventario_faces.reporting.report_support import (
    software_reference_abnt_text,
    technical_parameter_items,
)
from inventario_faces.utils.time_utils import format_local_datetime

REPORT_FILE_NAME = "relatorio_comparacao.docx"

_CLASS_LABELS = {
    "assignment": "Atribuição",
    "candidate": "Candidata",
    "below_threshold": "Abaixo do limiar",
}


class FaceSetComparisonDocxReportGenerator:
    """Relatório DOCX da comparação Padrão × Questionado (busca 1:N)."""

    # Quantidade de pares exibidos no ranking e no confronto visual lado a lado.
    MAX_RANKED_PAIRS = 50
    MAX_VISUAL_PAIRS = 10

    def __init__(self, config: AppConfig) -> None:
        self._config = config

    def generate(
        self,
        result: FaceSetComparisonResult,
        media_verdicts: dict[str, str],
        output_directory: Path,
    ) -> Path:
        """Gera o relatório.

        ``media_verdicts`` mapeia cada mídia do Questionado (caminho em texto) para a
        faixa decisória do veredito: ``assignment``, ``candidate`` ou ``below_threshold``.
        """

        docx_path = output_directory / REPORT_FILE_NAME
        self._build_document(result, media_verdicts).save(docx_path)
        return docx_path

    # ------------------------------------------------------------------ documento

    def _build_document(self, result: FaceSetComparisonResult, media_verdicts: dict[str, str]) -> Document:
        document = Document()
        document.styles["Normal"].font.name = "Cambria"
        document.styles["Normal"].font.size = Pt(11)
        entries_by_id = {entry.entry_id: entry for entry in [*result.set_a_faces, *result.set_b_faces]}
        summary = result.summary

        self._add_header(document, result)

        document.add_heading("Resumo e resultado", level=1)
        self._add_summary(document, result, media_verdicts)
        self._add_notice(
            document,
            "Advertência pericial",
            "A comparação facial automatizada expressa compatibilidade probabilística entre "
            "representações biométricas e não constitui identificação conclusiva. O resultado "
            "deve ser interpretado por perito, em conjunto com o exame morfológico e as demais "
            "evidências do caso.",
        )
        if not summary.reference_is_homogeneous:
            self._add_notice(
                document,
                "Alerta sobre o Padrão",
                f"As faces do Padrão formaram {summary.set_a_individuals} agrupamentos, quando o "
                "esperado é 1. Pode haver imagem de outra pessoa ou face de baixa qualidade na "
                "referência; revise o material antes de valorar o resultado.",
            )

        document.add_heading("Material examinado", level=1)
        self._add_inputs_table(document, "Padrão (referência)", result.set_a_inputs)
        self._add_inputs_table(document, "Questionado", result.set_b_inputs)

        document.add_heading("Faces selecionadas", level=1)
        self._add_entries_table(document, "Padrão", result.set_a_faces)
        self._add_entries_table(document, "Questionado", result.set_b_faces)

        document.add_heading("Confronto par a par", level=1)
        self._add_matches_table(document, result.matches, entries_by_id)
        self._add_visual_pairs(document, result.matches, entries_by_id)

        document.add_heading("Estatística descritiva das similaridades", level=1)
        self._add_statistics(document, result)

        document.add_heading("Metodologia", level=1)
        for item in self._methodology_items(result):
            self._add_list_item(document, item)

        document.add_heading("Anexo técnico", level=1)
        document.add_heading("Parâmetros de processamento", level=2)
        for item in technical_parameter_items(self._config, None):
            self._add_list_item(document, item)
        comparison = self._config.comparison
        self._add_list_item(
            document,
            "Comparação: faixas por similaridade candidata="
            f"{self._config.clustering.candidate_similarity:.2f} e atribuição="
            f"{self._config.clustering.assignment_similarity:.2f}; veredito por LR (quando calibrada) "
            f"candidata log10(LR)≥{comparison.verdict_candidate_log10_lr:.1f} e atribuição "
            f"log10(LR)≥{comparison.verdict_assignment_log10_lr:.1f}; máximo de quadros por vídeo="
            f"{comparison.max_frames_per_video or 'herdado de vídeo'}; intervalo de amostragem="
            f"{comparison.sampling_interval_seconds or 'herdado de vídeo'}.",
        )
        self._add_list_item(document, f"Referência do software: {software_reference_abnt_text()}")
        document.add_heading("Rastreabilidade", level=2)
        self._add_paragraph(document, f"Diretório da execução: {result.run_directory}")
        self._add_paragraph(document, f"Artefatos estruturados (CSV/JSON): {result.export_directory}")
        self._add_paragraph(document, f"Logs (texto e eventos JSONL): {result.logs_directory}")
        self._add_paragraph(
            document,
            f"Início: {format_local_datetime(result.started_at_utc)} | "
            f"Término: {format_local_datetime(result.finished_at_utc)}",
        )
        return document

    def _add_header(self, document: Document, result: FaceSetComparisonResult) -> None:
        title = document.add_paragraph()
        title.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = title.add_run(f"{self._config.app.report_title} - Comparação entre Grupos Faciais")
        run.bold = True
        run.font.size = Pt(16)
        meta = document.add_paragraph()
        meta.alignment = WD_ALIGN_PARAGRAPH.CENTER
        meta_run = meta.add_run(
            f"{self._config.app.organization} | Emitido em {format_local_datetime(result.finished_at_utc)}"
        )
        meta_run.font.size = Pt(10)

    def _add_summary(
        self,
        document: Document,
        result: FaceSetComparisonResult,
        media_verdicts: dict[str, str],
    ) -> None:
        summary = result.summary
        basis = (
            "razão de verossimilhança (LR) calibrada, com atribuição para "
            f"log10(LR) ≥ {self._config.comparison.verdict_assignment_log10_lr:.1f} e candidata para "
            f"log10(LR) ≥ {self._config.comparison.verdict_candidate_log10_lr:.1f}"
            if summary.verdict_uses_likelihood_ratio
            else "similaridade cosseno, com atribuição para similaridade ≥ "
            f"{summary.assignment_threshold:.2f} e candidata para similaridade ≥ "
            f"{summary.candidate_threshold:.2f} (sem calibração LR)"
        )
        best = "-" if summary.best_similarity is None else f"{summary.best_similarity:.4f}"
        self._add_paragraph(
            document,
            f"O grupo Padrão (referência de um único indivíduo) reuniu {summary.set_a_images} mídia(s), "
            f"com {summary.set_a_selected_faces} face(s) selecionada(s). O grupo Questionado reuniu "
            f"{summary.set_b_images} mídia(s), com {summary.set_b_selected_faces} face(s) selecionada(s). "
            f"Foram realizadas {summary.total_pair_comparisons} comparação(ões) par a par; a maior "
            f"similaridade observada foi {best}. O veredito por mídia foi governado por {basis}.",
        )

        table = document.add_table(rows=1, cols=3)
        table.style = "Table Grid"
        table.alignment = WD_TABLE_ALIGNMENT.CENTER
        for cell, label in zip(table.rows[0].cells, ("Mídia do Questionado", "Resultado", "Maior similaridade")):
            self._set_cell_text(cell, label, bold=True, alignment=WD_ALIGN_PARAGRAPH.CENTER)
        best_by_media = self._best_similarity_by_media(result)
        for record in result.set_b_inputs:
            media = str(record.source_path)
            verdict = media_verdicts.get(media)
            if record.processing_error:
                label = "Não analisada (falha de processamento)"
            elif verdict is None:
                label = "Não analisada (sem face selecionada)"
            else:
                label = _CLASS_LABELS.get(verdict, verdict)
            similarity = best_by_media.get(media)
            row = table.add_row().cells
            self._set_cell_text(row[0], record.source_path.name)
            self._set_cell_text(row[1], label, alignment=WD_ALIGN_PARAGRAPH.CENTER, bold=verdict == "assignment")
            self._set_cell_text(
                row[2],
                "-" if similarity is None else f"{similarity:.4f}",
                alignment=WD_ALIGN_PARAGRAPH.CENTER,
            )
        document.add_paragraph()

    def _add_inputs_table(self, document: Document, title: str, inputs: list[FaceSetComparisonInput]) -> None:
        document.add_heading(title, level=2)
        if not inputs:
            self._add_paragraph(document, "Nenhuma mídia informada.")
            return
        for index, record in enumerate(inputs, start=1):
            rows = [
                ("Mídia", f"{index}/{len(inputs)} - {record.source_path.name}"),
                ("Caminho", str(record.source_path)),
                ("SHA-512", record.sha512 or "-"),
                (
                    "Processamento",
                    f"faces detectadas: {record.detected_faces}; faces selecionadas: "
                    f"{record.selected_faces}; trilhas (tracks): {record.tracks}",
                ),
            ]
            if record.still_image_margin_px:
                rows.append(
                    (
                        "Observação",
                        "Nenhuma face detectada na imagem original (recorte facial justo); detecção "
                        f"repetida com margem neutra de {record.still_image_margin_px}px em cada lado.",
                    )
                )
            if record.processing_error:
                rows.append(("Falha", record.processing_error))
            table = document.add_table(rows=0, cols=2)
            table.style = "Table Grid"
            for label, value in rows:
                cells = table.add_row().cells
                self._set_cell_text(cells[0], label, bold=True)
                self._set_cell_text(cells[1], value, size=Pt(8) if label == "SHA-512" else None)
            self._set_column_widths(table, (Inches(1.3), Inches(5.2)))
            document.add_paragraph()

    def _add_entries_table(self, document: Document, title: str, entries: list[FaceSetComparisonEntry]) -> None:
        document.add_heading(title, level=2)
        if not entries:
            self._add_paragraph(document, "Nenhuma face selecionada.")
            return
        table = document.add_table(rows=1, cols=4)
        table.style = "Table Grid"
        table.alignment = WD_TABLE_ALIGNMENT.CENTER
        for cell, label in zip(table.rows[0].cells, ("Face", "Recorte", "Malha biométrica", "Métricas")):
            self._set_cell_text(cell, label, bold=True, alignment=WD_ALIGN_PARAGRAPH.CENTER)
        for entry in entries:
            row = table.add_row().cells
            self._set_cell_text(row[0], f"{entry.entry_id}\n{entry.source_path.name}")
            self._add_cell_image(row[1], entry.crop_path, width=Inches(1.3))
            self._add_cell_image(row[2], entry.mesh_crop_path, width=Inches(1.3))
            self._set_cell_text(row[3], "\n".join(self._entry_metric_lines(entry)))
        document.add_paragraph()

    def _entry_metric_lines(self, entry: FaceSetComparisonEntry) -> list[str]:
        def fmt(value: float | None) -> str:
            return "-" if value is None else f"{value:.3f}"

        bbox = entry.bbox
        return [
            f"Confiança de detecção: {entry.detection_score:.3f}",
            f"Qualidade: {fmt(entry.quality_score)}",
            f"Nitidez: {fmt(entry.sharpness)}",
            f"Iluminação: {fmt(entry.illumination)}",
            f"Frontalidade: {fmt(entry.frontality)}",
            f"Caixa: ({bbox.x1:.0f}, {bbox.y1:.0f}) - ({bbox.x2:.0f}, {bbox.y2:.0f})",
            f"Embedding: {entry.embedding_dimension} dimensões",
        ]

    def _add_matches_table(
        self,
        document: Document,
        matches: list[FaceSetComparisonMatch],
        entries_by_id: dict[str, FaceSetComparisonEntry],
    ) -> None:
        if not matches:
            self._add_paragraph(document, "Nenhum par foi comparado.")
            return
        shown = matches[: self.MAX_RANKED_PAIRS]
        if len(matches) > len(shown):
            self._add_paragraph(
                document,
                f"Exibidos os {len(shown)} pares de maior similaridade, de um total de {len(matches)}. "
                "A lista completa está em face_set_comparison_matches.csv.",
            )
        has_lr = any(match.log10_likelihood_ratio is not None for match in shown)
        headers = ["Pos.", "Face Padrão", "Face Questionado", "Similaridade", "Faixa"]
        if has_lr:
            headers.extend(["log10(LR)", "Leitura da LR"])
        table = document.add_table(rows=1, cols=len(headers))
        table.style = "Table Grid"
        table.alignment = WD_TABLE_ALIGNMENT.CENTER
        for cell, label in zip(table.rows[0].cells, headers):
            self._set_cell_text(cell, label, bold=True, alignment=WD_ALIGN_PARAGRAPH.CENTER)
        for match in shown:
            row = table.add_row().cells
            left = entries_by_id.get(match.left_entry_id)
            right = entries_by_id.get(match.right_entry_id)
            self._set_cell_text(row[0], str(match.rank), alignment=WD_ALIGN_PARAGRAPH.CENTER)
            self._set_cell_text(row[1], self._entry_label(match.left_entry_id, left))
            self._set_cell_text(row[2], self._entry_label(match.right_entry_id, right))
            self._set_cell_text(row[3], f"{match.similarity:.4f}", alignment=WD_ALIGN_PARAGRAPH.CENTER)
            self._set_cell_text(
                row[4],
                _CLASS_LABELS.get(match.classification, match.classification),
                alignment=WD_ALIGN_PARAGRAPH.CENTER,
            )
            if has_lr:
                lr_text = "-" if match.log10_likelihood_ratio is None else f"{match.log10_likelihood_ratio:.2f}"
                if match.outside_calibration_support:
                    lr_text += " (extrapolada)"
                self._set_cell_text(row[5], lr_text, alignment=WD_ALIGN_PARAGRAPH.CENTER)
                self._set_cell_text(row[6], match.evidence_label or "-")
        document.add_paragraph()

    def _add_visual_pairs(
        self,
        document: Document,
        matches: list[FaceSetComparisonMatch],
        entries_by_id: dict[str, FaceSetComparisonEntry],
    ) -> None:
        shown = matches[: self.MAX_VISUAL_PAIRS]
        if not shown:
            return
        document.add_heading("Confronto visual dos pares de maior similaridade", level=2)
        for match in shown:
            left = entries_by_id.get(match.left_entry_id)
            right = entries_by_id.get(match.right_entry_id)
            self._add_paragraph(
                document,
                f"Par {match.rank}: {match.left_entry_id} × {match.right_entry_id} | "
                f"similaridade {match.similarity:.4f} | "
                f"{_CLASS_LABELS.get(match.classification, match.classification)}",
            )
            table = document.add_table(rows=3, cols=2)
            table.style = "Table Grid"
            table.alignment = WD_TABLE_ALIGNMENT.CENTER
            self._set_cell_text(table.cell(0, 0), "Padrão", bold=True, alignment=WD_ALIGN_PARAGRAPH.CENTER)
            self._set_cell_text(table.cell(0, 1), "Questionado", bold=True, alignment=WD_ALIGN_PARAGRAPH.CENTER)
            for column, entry in enumerate((left, right)):
                self._add_cell_image(table.cell(1, column), entry.crop_path if entry else None, width=Inches(2.2))
                self._add_cell_image(table.cell(2, column), entry.mesh_crop_path if entry else None, width=Inches(2.2))
            document.add_paragraph()

    def _add_statistics(self, document: Document, result: FaceSetComparisonResult) -> None:
        summary = result.summary

        def fmt(value: float | None) -> str:
            return "-" if value is None else f"{value:.4f}"

        rows = [
            ("Comparações par a par", str(summary.total_pair_comparisons)),
            ("Pares na faixa de atribuição (similaridade)", str(summary.assignment_matches)),
            ("Pares na faixa candidata (similaridade)", str(summary.candidate_matches)),
            ("Maior similaridade", fmt(summary.best_similarity)),
            ("Menor similaridade", fmt(summary.worst_similarity)),
            ("Média", fmt(summary.mean_similarity)),
            ("Mediana", fmt(summary.median_similarity)),
            ("Desvio padrão", fmt(summary.stddev_similarity)),
            ("Limiar candidata", f"{summary.candidate_threshold:.4f}"),
            ("Limiar atribuição", f"{summary.assignment_threshold:.4f}"),
        ]
        if summary.likelihood_ratio_calibrated:
            rows.extend(
                [
                    ("Pares com LR calibrada", str(summary.calibrated_matches)),
                    ("Maior log10(LR)", fmt(summary.max_log10_likelihood_ratio)),
                    ("Mediana de log10(LR)", fmt(summary.median_log10_likelihood_ratio)),
                    ("Pares com LR extrapolada", str(summary.extrapolated_matches)),
                ]
            )
        if summary.total_pair_comparisons < 2:
            self._add_paragraph(
                document,
                "Com uma única comparação, medidas de dispersão não se aplicam; o resultado "
                "decorre do valor individual do par.",
            )
        table = document.add_table(rows=0, cols=2)
        table.style = "Table Grid"
        for label, value in rows:
            row = table.add_row().cells
            self._set_cell_text(row[0], label)
            self._set_cell_text(row[1], value, alignment=WD_ALIGN_PARAGRAPH.CENTER)
        document.add_paragraph()

    def _methodology_items(self, result: FaceSetComparisonResult) -> list[str]:
        face_model = self._config.face_model
        items = [
            "Cada mídia foi preservada: o hash SHA-512 foi calculado sobre o arquivo original e todo "
            "pré-processamento foi aplicado apenas a cópias em memória.",
            f"A detecção facial e a extração de embeddings usaram o modelo {face_model.model_name} "
            "(os backends insightface e onnx executam os mesmos arquivos de modelo; o backend efetivo "
            f"fica registrado no log), com qualidade mínima de detecção {face_model.minimum_face_quality:.2f} "
            f"e tamanho mínimo de face {face_model.minimum_face_size_pixels}px.",
            "As faces do Padrão foram tratadas como referência de um único indivíduo; um agrupamento de "
            "controle verifica se a referência é homogênea. O Questionado não é agrupado: cada face é "
            "confrontada diretamente com a referência.",
            "A comparação foi feita par a par pela similaridade cosseno entre embeddings normalizados.",
        ]
        if face_model.still_image_margin_ratio > 0:
            items.append(
                "Imagens estáticas sem nenhuma face detectada são reprocessadas com margem neutra de "
                f"{face_model.still_image_margin_ratio:.2f} do maior lado em cada borda, para recuperar "
                "recortes faciais justos que excedem a escala do detector. O conteúdo original não é "
                "reamostrado e a aplicação fica registrada por mídia."
            )
        if result.calibration is not None:
            items.append(
                "A razão de verossimilhança foi estimada como p(escore | mesma origem) / "
                "p(escore | origem distinta), a partir de base rotulada ou modelo de calibração salvo."
            )
        return items

    # ------------------------------------------------------------------ auxiliares

    def _best_similarity_by_media(self, result: FaceSetComparisonResult) -> dict[str, float]:
        source_by_entry = {entry.entry_id: str(entry.source_path) for entry in result.set_b_faces}
        best: dict[str, float] = {}
        for match in result.matches:
            media = source_by_entry.get(match.right_entry_id)
            if media is not None and match.similarity > best.get(media, float("-inf")):
                best[media] = match.similarity
        return best

    def _entry_label(self, entry_id: str, entry: FaceSetComparisonEntry | None) -> str:
        return entry_id if entry is None else f"{entry_id}\n{entry.source_path.name}"

    def _add_paragraph(self, document: Document, text: str) -> None:
        document.add_paragraph(text)

    def _add_list_item(self, document: Document, text: str) -> None:
        document.add_paragraph(style="List Bullet").add_run(text)

    def _add_notice(self, document: Document, title: str, body: str) -> None:
        table = document.add_table(rows=1, cols=1)
        table.style = "Table Grid"
        table.alignment = WD_TABLE_ALIGNMENT.CENTER
        paragraph = table.cell(0, 0).paragraphs[0]
        paragraph.add_run(f"{title}. ").bold = True
        paragraph.add_run(body)
        document.add_paragraph()

    def _set_cell_text(
        self,
        cell,
        text: str,
        *,
        alignment: WD_ALIGN_PARAGRAPH = WD_ALIGN_PARAGRAPH.LEFT,
        bold: bool = False,
        size: Pt | None = None,
    ) -> None:
        cell.text = ""
        cell.vertical_alignment = WD_ALIGN_VERTICAL.TOP
        paragraph = cell.paragraphs[0]
        paragraph.alignment = alignment
        run = paragraph.add_run(text)
        run.bold = bold
        if size is not None:
            run.font.size = size

    def _set_column_widths(self, table, widths: tuple) -> None:
        table.autofit = False
        for row in table.rows:
            for cell, width in zip(row.cells, widths):
                cell.width = width

    def _add_cell_image(self, cell, image_path: Path | None, *, width: Inches) -> None:
        cell.text = ""
        paragraph = cell.paragraphs[0]
        paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        if image_path is None or not Path(image_path).exists():
            paragraph.add_run("Imagem indisponível")
            return
        try:
            paragraph.add_run().add_picture(str(image_path), width=width)
        except Exception:
            paragraph.add_run("Imagem indisponível")
