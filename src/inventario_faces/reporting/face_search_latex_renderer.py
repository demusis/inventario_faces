from __future__ import annotations

import os
from pathlib import Path

from inventario_faces.domain.config import AppConfig
from inventario_faces.domain.entities import FaceSearchMatch, FaceSearchResult, ReportArtifacts
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
            template.replace("@@TITLE@@", escape_latex(f"{self._config.app.report_title} - Busca por Face"))
            .replace("@@ORGANIZATION@@", escape_latex(self._config.app.organization))
            .replace("@@DATE@@", escape_latex(format_local_datetime(result.inventory_result.finished_at_utc)))
            .replace("@@BODY@@", body)
        )

    def _load_template(self) -> str:
        template_path = Path(__file__).resolve().parent / "templates" / "forensic_report_template.tex"
        return template_path.read_text(encoding="utf-8")

    def _executive_summary(self, result: FaceSearchResult) -> str:
        summary = result.summary
        query = result.query
        return rf"""
\section{{Resumo Executivo}}
Esta busca facial utilizou como consulta o arquivo \texttt{{{break_wrappable_text(str(query.source_path))}}}. Foram detectada(s) {summary.query_faces_detected} face(s) eleg\'iveis na imagem de consulta e a face selecionada automaticamente para a pesquisa corresponde ao track \texttt{{{escape_latex(query.selected_track_id)}}}. A varredura no diret\'orio \texttt{{{break_wrappable_text(str(result.inventory_result.root_directory))}}} retornou {summary.compatible_tracks} track(s) compat\'iveis, distribu\'idos em {summary.compatible_clusters} grupo(s), com {summary.compatible_occurrences} ocorr\^encia(s) internas acima do limiar probabil\'istico de {summary.compatibility_threshold:.2f}.

\begin{{center}}
\fcolorbox{{yellow!40!black}}{{yellow!8}}{{\begin{{minipage}}{{0.92\linewidth}}
\textbf{{Advert\^encia pericial.}} A busca abaixo aponta apenas \textbf{{compatibilidade probabil\'istica}} com a face consultada. O resultado n\~ao constitui prova conclusiva de identidade e deve ser validado por especialista humano em conjunto com outros elementos do caso.
\end{{minipage}}}}
\end{{center}}
"""

    def _methodology(self, result: FaceSearchResult) -> str:
        items = "\n".join(
            rf"\item {escape_latex(item)}"
            for item in face_search_methodology_items(result.query.selected_track_id)
        )
        return rf"""
\section{{Metodologia da Busca}}
\begin{{enumerate}}[leftmargin=*,label=\arabic*.]
{items}
\end{{enumerate}}
"""

    def _query_face_section(self, result: FaceSearchResult, tex_path: Path) -> str:
        query = result.query
        crop_cell = self._table_image_cell(query.crop_path, tex_path, width="0.36\\linewidth", height="0.22\\textheight")
        context_cell = self._table_image_cell(query.context_image_path, tex_path, width="0.54\\linewidth", height="0.22\\textheight")
        quality = "-" if query.quality_score is None else f"{query.quality_score:.3f}"
        metadata = r"\begin{minipage}[t]{\linewidth}\raggedright\setlength{\parskip}{0.25em}\vspace{0pt}" + r" \par ".join(
            [
                rf"Arquivo consultado: \texttt{{{break_wrappable_text(str(query.source_path))}}}",
                rf"SHA-512: \texttt{{{break_monospace_text(query.sha512)}}}",
                rf"Faces eleg\'iveis detectadas na consulta: {query.detected_face_count}",
                rf"Track selecionado: \texttt{{{escape_latex(query.selected_track_id)}}}",
                rf"Ocorr\^encia selecionada: \texttt{{{escape_latex(query.selected_occurrence_id)}}}",
                (
                    rf"Keyframe de refer\^encia da consulta: \texttt{{{escape_latex(query.selected_keyframe_id)}}}"
                    if query.selected_keyframe_id
                    else r"Keyframe de refer\^encia da consulta: -"
                ),
                rf"Qualidade da face consultada: {quality}",
            ]
        ) + r"\end{minipage}"
        return rf"""
\section{{Face Consultada}}
\begingroup
\setlength{{\tabcolsep}}{{4pt}}
\begin{{scriptsize}}
\begin{{longtable}}{{@{{}}G{{0.18\linewidth}}G{{0.28\linewidth}}L{{0.44\linewidth}}@{{}}}}
\toprule
Recorte da consulta & Quadro de origem da consulta & Metadados da consulta \\
\midrule
\endhead
{crop_cell} & {context_cell} & {metadata} \\
\bottomrule
\end{{longtable}}
\end{{scriptsize}}
\endgroup
"""

    def _matches_section(self, result: FaceSearchResult, tex_path: Path) -> str:
        if not result.matches:
            return r"""
\section{Resultados Compat\'iveis}
\textit{Nenhum track compat\'ivel superou o limiar probabil\'istico configurado para a consulta.}
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
