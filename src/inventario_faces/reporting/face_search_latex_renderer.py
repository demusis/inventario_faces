from __future__ import annotations

import os
from pathlib import Path

from inventario_faces.domain.config import AppConfig
from inventario_faces.domain.entities import (
    FaceSearchMatch,
    FaceSearchQuery,
    FaceSearchQueryEvent,
    FaceSearchResult,
    ReportArtifacts,
)
from inventario_faces.infrastructure.latex_compiler import LatexCompilationError, LatexCompiler
from inventario_faces.reporting.report_support import (
    face_search_methodology_items,
    software_reference_abnt_latex,
    technical_parameter_items,
)
from inventario_faces.utils.latex import break_monospace_text, break_wrappable_text, escape_latex, format_seconds
from inventario_faces.utils.path_utils import ensure_directory
from inventario_faces.utils.time_utils import format_local_datetime


class FaceSearchLatexReportGenerator:
    def __init__(self, config: AppConfig, compiler: LatexCompiler) -> None:
        self._config = config
        self._compiler = compiler

    def generate(self, result: FaceSearchResult) -> ReportArtifacts:
        report_directory = ensure_directory(result.inventory_result.run_directory / "report")
        tex_path = report_directory / "relatorio_busca_por_face.tex"
        tex_path.write_text(self._render_tex(result, tex_path), encoding="utf-8")

        pdf_path: Path | None = None
        if self._config.reporting.compile_pdf:
            try:
                pdf_path = self._compiler.compile(tex_path)
            except LatexCompilationError as exc:
                warning_path = report_directory / "relatorio_busca_por_face_pdf_erro.txt"
                warning_path.write_text(str(exc), encoding="utf-8")
        return ReportArtifacts(tex_path=tex_path, pdf_path=pdf_path)

    def _render_tex(self, result: FaceSearchResult, tex_path: Path) -> str:
        template = self._load_template()
        body = "\n".join(
            [
                self._executive_summary(result),
                self._methodology(result),
                self._query_face_section(result, tex_path),
                self._matches_section(result, tex_path),
                self._technical_appendix(result),
            ]
        )
        return (
            template.replace("@@TITLE@@", escape_latex(f"{self._config.app.report_title} - Busca por Faces"))
            .replace("@@ORGANIZATION@@", escape_latex(self._config.app.organization))
            .replace("@@DATE@@", escape_latex(format_local_datetime(result.inventory_result.finished_at_utc)))
            .replace("@@BODY@@", body)
        )

    def _load_template(self) -> str:
        template_path = Path(__file__).resolve().parent / "templates" / "forensic_report_template.tex"
        return template_path.read_text(encoding="utf-8")

    def _executive_summary(self, result: FaceSearchResult) -> str:
        summary = result.summary
        queries = self._queries(result)
        if not queries:
            summary_text = (
                rf"Esta busca facial recebeu {summary.query_image_count} arquivo(s) de consulta. Nenhum deles gerou face de refer\^encia "
                rf"utiliz\'avel para a etapa de busca vetorial. Todos os eventos das consultas, inclusive erros de leitura, corrup\c{{c}}\~ao "
                rf"e descartes por crit\'erios t\'ecnicos, foram individualizados na se\c{{c}}\~ao de arquivos de consulta informados. "
            )
            search_outcome = (
                rf"O diret\'orio \texttt{{{break_wrappable_text(str(result.inventory_result.root_directory))}}} foi processado, "
                rf"mas a busca vetorial n\~ao foi executada por aus\^encia de refer\^encia facial v\'alida."
            )
        elif len(queries) == 1 and summary.query_image_count == 1:
            query = queries[0]
            summary_text = (
                rf"Esta busca facial utilizou como consulta o arquivo \texttt{{{break_wrappable_text(str(query.source_path))}}}. "
                rf"Foram detectada(s) {summary.query_faces_detected} face(s) eleg\'iveis na imagem de consulta e a face selecionada "
                rf"automaticamente para a pesquisa corresponde ao track \texttt{{{escape_latex(query.selected_track_id)}}}. "
            )
            search_outcome = (
                rf"A varredura no diret\'orio \texttt{{{break_wrappable_text(str(result.inventory_result.root_directory))}}} retornou "
                rf"{summary.compatible_tracks} track(s) compat\'iveis, distribu\'idos em {summary.compatible_clusters} grupo(s), com "
                rf"{summary.compatible_occurrences} ocorr\^encia(s) internas acima do limiar probabil\'istico de {summary.compatibility_threshold:.2f}."
            )
        else:
            summary_text = (
                rf"Esta busca facial recebeu {summary.query_image_count} arquivo(s) de consulta. "
                rf"{summary.query_faces_selected} arquivo(s) geraram face(s) de refer\^encia utiliz\'avel(is), totalizando "
                rf"{summary.query_faces_detected} face(s) eleg\'iveis nas consultas v\'alidas. "
                rf"{summary.query_images_rejected} arquivo(s) foram descartados, permanecendo individualizados com o motivo do descarte "
                rf"na se\c{{c}}\~ao de arquivos de consulta informados. "
            )
            search_outcome = (
                rf"A varredura no diret\'orio \texttt{{{break_wrappable_text(str(result.inventory_result.root_directory))}}} retornou "
                rf"{summary.compatible_tracks} track(s) compat\'iveis, distribu\'idos em {summary.compatible_clusters} grupo(s), com "
                rf"{summary.compatible_occurrences} ocorr\^encia(s) internas acima do limiar probabil\'istico de {summary.compatibility_threshold:.2f}."
            )
        return rf"""
\section{{Resumo Executivo}}
{summary_text}{search_outcome}

\begin{{center}}
\fcolorbox{{yellow!40!black}}{{yellow!8}}{{\begin{{minipage}}{{0.92\linewidth}}
\textbf{{Advert\^encia pericial.}} A busca abaixo aponta apenas \textbf{{compatibilidade probabil\'istica}} com a face consultada. O resultado n\~ao constitui prova conclusiva de identidade e deve ser validado por especialista humano em conjunto com outros elementos do caso.
\end{{minipage}}}}
\end{{center}}
"""

    def _methodology(self, result: FaceSearchResult) -> str:
        queries = self._queries(result)
        items = "\n".join(
            rf"\item {escape_latex(item)}"
            for item in face_search_methodology_items(
                result.summary.query_image_count,
                [query.selected_track_id for query in queries],
            )
        )
        return rf"""
\section{{Metodologia da Busca}}
\begin{{enumerate}}[leftmargin=*,label=\arabic*.]
{items}
\end{{enumerate}}
"""

    def _query_face_section(self, result: FaceSearchResult, tex_path: Path) -> str:
        rows = "\n".join(self._query_event_row(event, tex_path) for event in self._query_events(result))
        return rf"""
\section{{Arquivos de Consulta Informados}}
\begingroup
\setlength{{\tabcolsep}}{{4pt}}
\begin{{scriptsize}}
\begin{{longtable}}{{@{{}}C{{0.05\linewidth}}T{{0.10\linewidth}}G{{0.14\linewidth}}G{{0.22\linewidth}}L{{0.41\linewidth}}@{{}}}}
\toprule
Pos. & Situa\c{{c}}\~ao & Recorte da consulta & Quadro de origem da consulta & Metadados da consulta \\
\midrule
\endhead
{rows}
\bottomrule
\end{{longtable}}
\end{{scriptsize}}
\endgroup
"""

    def _query_event_row(self, event: FaceSearchQueryEvent, tex_path: Path) -> str:
        crop_cell = self._table_image_cell(event.crop_path, tex_path, width="0.32\\linewidth", height="0.20\\textheight")
        context_cell = self._table_image_cell(event.context_image_path, tex_path, width="0.50\\linewidth", height="0.20\\textheight")
        quality = "-" if event.quality_score is None else f"{event.quality_score:.3f}"
        detected_faces = "-" if event.detected_face_count is None else str(event.detected_face_count)
        status_label = "Selecionada" if event.status == "selected" else "Descartada"
        metadata = r"\begin{minipage}[t]{\linewidth}\raggedright\setlength{\parskip}{0.25em}\vspace{0pt}" + r" \par ".join(
            [
                rf"Consulta: {event.query_index}",
                rf"Arquivo consultado: \texttt{{{break_wrappable_text(str(event.source_path))}}}",
                (
                    rf"SHA-512: \texttt{{{break_monospace_text(event.sha512)}}}"
                    if event.sha512 is not None
                    else r"SHA-512: -"
                ),
                rf"Situa\c{{c}}\~ao da consulta: {status_label}",
                rf"Faces eleg\'iveis detectadas na consulta: {detected_faces}",
                (
                    rf"Track selecionado: \texttt{{{escape_latex(event.selected_track_id)}}}"
                    if event.selected_track_id
                    else r"Track selecionado: -"
                ),
                (
                    rf"Ocorr\^encia selecionada: \texttt{{{escape_latex(event.selected_occurrence_id)}}}"
                    if event.selected_occurrence_id
                    else r"Ocorr\^encia selecionada: -"
                ),
                (
                    rf"Keyframe de refer\^encia da consulta: \texttt{{{escape_latex(event.selected_keyframe_id)}}}"
                    if event.selected_keyframe_id
                    else r"Keyframe de refer\^encia da consulta: -"
                ),
                rf"Qualidade da face consultada: {quality}",
                (
                    rf"Evento reportado: {escape_latex(event.error_type)} - {escape_latex(event.error_message)}"
                    if event.error_message is not None and event.error_type is not None
                    else (
                        rf"Evento reportado: {escape_latex(event.error_message)}"
                        if event.error_message is not None
                        else r"Evento reportado: processamento conclu\'ido sem erro."
                    )
                ),
            ]
        ) + r"\end{minipage}"
        return rf"{event.query_index} & {status_label} & {crop_cell} & {context_cell} & {metadata} \\"

    def _matches_section(self, result: FaceSearchResult, tex_path: Path) -> str:
        if not result.matches:
            return r"""
\section{Resultados Compat\'iveis}
\textit{Nenhum track compat\'ivel superou o limiar probabil\'istico configurado para as consultas.}
"""
        rows = "\n".join(self._match_row(match, tex_path) for match in result.matches)
        return rf"""
\section{{Resultados Compat\'iveis}}
\begin{{scriptsize}}
\begin{{longtable}}{{C{{0.05\linewidth}}T{{0.08\linewidth}}T{{0.12\linewidth}}G{{0.14\linewidth}}G{{0.24\linewidth}}L{{0.26\linewidth}}}}
\toprule
Pos. & Grupo & Track & Recorte & Quadro de origem & Metadados \\
\midrule
\endhead
{rows}
\bottomrule
\end{{longtable}}
\end{{scriptsize}}
"""

    def _match_row(self, match: FaceSearchMatch, tex_path: Path) -> str:
        crop_cell = self._table_image_cell(match.crop_path, tex_path, width="0.92\\linewidth", height="0.16\\textheight")
        context_cell = self._table_image_cell(match.context_image_path, tex_path, width="0.95\\linewidth", height="0.16\\textheight")
        occurrence_score = "-" if match.occurrence_score is None else f"{match.occurrence_score:.3f}"
        cluster_score = "-" if match.cluster_score is None else f"{match.cluster_score:.3f}"
        occurrence_id = match.occurrence_id or "-"
        metadata = r"\begin{minipage}[t]{\linewidth}\raggedright\setlength{\parskip}{0.25em}\vspace{0pt}" + r" \par ".join(
            [
                (
                    rf"Consulta que sustentou o score: \texttt{{{break_wrappable_text(str(match.query_source_path))}}}"
                    if match.query_source_path is not None
                    else r"Consulta que sustentou o score: -"
                ),
                (
                    rf"Track da consulta: \texttt{{{escape_latex(match.query_selected_track_id)}}}"
                    if match.query_selected_track_id
                    else r"Track da consulta: -"
                ),
                (
                    rf"Ocorr\^encia da consulta: \texttt{{{escape_latex(match.query_selected_occurrence_id)}}}"
                    if match.query_selected_occurrence_id
                    else r"Ocorr\^encia da consulta: -"
                ),
                rf"Origem: \texttt{{{break_wrappable_text(match.source_path.name)}}}",
                rf"Ocorr\^encia: \texttt{{{escape_latex(occurrence_id)}}}",
                f"Instante da ocorr\\^encia: {format_seconds(match.timestamp_seconds)}",
                f"Intervalo do track: {format_seconds(match.track_start_time)} - {format_seconds(match.track_end_time)}",
                (
                    rf"Quadro da ocorr\^encia: \texttt{{{match.frame_index:06d}}}"
                    if match.frame_index is not None
                    else r"Quadro da ocorr\^encia: -"
                ),
                f"Similaridade do grupo: {cluster_score}",
                f"Similaridade do track: {match.track_score:.3f}",
                f"Similaridade da ocorr\\^encia: {occurrence_score}",
            ]
        ) + r"\end{minipage}"
        group_label = "-" if match.cluster_id is None else rf"\texttt{{{escape_latex(match.cluster_id)}}}"
        track_label = rf"\texttt{{{escape_latex(match.track_id)}}}"
        return rf"{match.rank} & {group_label} & {track_label} & {crop_cell} & {context_cell} & {metadata} \\"

    def _queries(self, result: FaceSearchResult) -> list[FaceSearchQuery]:
        if result.queries:
            return list(result.queries)
        return [result.query] if result.query is not None else []

    def _query_events(self, result: FaceSearchResult) -> list[FaceSearchQueryEvent]:
        if result.query_events:
            return list(result.query_events)
        return [
            FaceSearchQueryEvent(
                query_index=query.query_index,
                source_path=query.source_path,
                status="selected",
                sha512=query.sha512,
                detected_face_count=query.detected_face_count,
                selected_track_id=query.selected_track_id,
                selected_occurrence_id=query.selected_occurrence_id,
                selected_keyframe_id=query.selected_keyframe_id,
                crop_path=query.crop_path,
                context_image_path=query.context_image_path,
                quality_score=query.quality_score,
            )
            for query in self._queries(result)
        ]

    def _technical_appendix(self, result: FaceSearchResult) -> str:
        parameter_items = "\n".join(
            rf"\item {escape_latex(item)}"
            for item in technical_parameter_items(self._config, result.inventory_result.search)
        )
        export_path = "-" if result.export_path is None else str(result.export_path)
        report_path = result.inventory_result.report.pdf_path or result.inventory_result.report.docx_path or result.inventory_result.report.tex_path
        return rf"""
\section{{Anexo T\'ecnico}}
\begin{{itemize}}[leftmargin=*]
\item Invent\'ario de refer\^encia usado na busca: \texttt{{{break_wrappable_text(str(report_path))}}}.
\item Exporta\c{{c}}\~ao estruturada da busca: \texttt{{{break_wrappable_text(export_path)}}}.
{parameter_items}
\item Refer\^encia do software: {software_reference_abnt_latex()}
\end{{itemize}}
"""

    def _table_image_cell(self, artifact_path: Path | None, tex_path: Path, width: str, height: str) -> str:
        content = self._include_image(artifact_path, tex_path, width=width, height=height)
        return r"\begin{minipage}[t]{\linewidth}\centering\vspace{0pt}" + content + r"\end{minipage}"

    def _include_image(self, artifact_path: Path | None, tex_path: Path, width: str, height: str) -> str:
        if artifact_path is None:
            return r"\textit{Artefato n\~ao dispon\'ivel}"
        return rf"\includegraphics[width={width},height={height},keepaspectratio]{{{self._relative_to_tex(artifact_path, tex_path)}}}"

    def _relative_to_tex(self, artifact_path: Path, tex_path: Path) -> str:
        return Path(os.path.relpath(artifact_path, start=tex_path.parent)).as_posix()
