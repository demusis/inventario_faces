from __future__ import annotations

import os
from pathlib import Path

from inventario_faces.domain.config import AppConfig
from inventario_faces.domain.entities import FileRecord, FaceTrack, InventoryResult, KeyFrame, MediaInfoAttribute, ReportArtifacts
from inventario_faces.infrastructure.latex_compiler import LatexCompilationError, LatexCompiler
from inventario_faces.reporting.report_context import (
    keyframes_by_track,
    mean_pairwise_track_similarity,
    tracks_by_cluster,
)
from inventario_faces.reporting.report_support import (
    candidate_cluster_map,
    format_group_similarity,
    inventory_methodology_items,
    keyframe_reference_lines,
    media_track_type_label,
    software_reference_abnt_latex,
    technical_parameter_items,
    track_frame_interval_text,
    track_interval_text,
)
from inventario_faces.utils.latex import break_monospace_text, break_wrappable_text, escape_latex
from inventario_faces.utils.path_utils import ensure_directory
from inventario_faces.utils.time_utils import format_local_datetime


class LatexReportGenerator:
    def __init__(self, config: AppConfig, compiler: LatexCompiler) -> None:
        self._config = config
        self._compiler = compiler

    def generate(self, result: InventoryResult) -> ReportArtifacts:
        report_directory = ensure_directory(result.run_directory / "report")
        tex_path = report_directory / "relatorio_forense.tex"
        tex_path.write_text(self._render_tex(result, tex_path), encoding="utf-8")

        pdf_path: Path | None = None
        if self._config.reporting.compile_pdf:
            try:
                pdf_path = self._compiler.compile(tex_path)
            except LatexCompilationError as exc:
                warning_path = report_directory / "relatorio_forense_pdf_erro.txt"
                warning_path.write_text(str(exc), encoding="utf-8")
        return ReportArtifacts(tex_path=tex_path, pdf_path=pdf_path)

    def _render_tex(self, result: InventoryResult, tex_path: Path) -> str:
        template = self._load_template()
        body = "\n".join(
            [
                self._executive_summary(result),
                self._methodology(result),
                self._group_results(result, tex_path),
                self._face_size_statistics_section(result),
                self._technical_appendix(result),
            ]
        )
        return (
            template.replace("@@TITLE@@", escape_latex(self._config.app.report_title))
            .replace("@@ORGANIZATION@@", escape_latex(self._config.app.organization))
            .replace("@@DATE@@", escape_latex(format_local_datetime(result.finished_at_utc)))
            .replace("@@BODY@@", body)
        )

    def _load_template(self) -> str:
        template_path = Path(__file__).resolve().parent / "templates" / "forensic_report_template.tex"
        return template_path.read_text(encoding="utf-8")

    def _executive_summary(self, result: InventoryResult) -> str:
        summary = result.summary
        return rf"""
\section{{Resumo Executivo}}
O presente relat\'orio registra o processamento automatizado do diret\'orio \texttt{{{break_wrappable_text(str(result.root_directory))}}}. Foram catalogados {summary.total_files} arquivo(s), dos quais {summary.media_files} m\'idia(s) suportada(s) foram submetida(s) a detec\c{{c}}\~ao facial. O processamento registrou {summary.total_occurrences} detec\c{{c}}\~ao(\~oes), consolidadas em {summary.total_tracks} track(s) faciais e {summary.total_keyframes} keyframe(s) audit\'avel(eis). O agrupamento por track resultou em {summary.total_clusters} grupo(s) de poss\'iveis correspond\^encias, com {summary.probable_match_pairs} rela\c{{c}}\~ao(\~oes) intergrupos assinalada(s) como probabil\'isticas.

\begin{{center}}
\fcolorbox{{yellow!40!black}}{{yellow!8}}{{\begin{{minipage}}{{0.92\linewidth}}
\textbf{{Advert\^encia pericial.}} Correspond\^encia facial automatizada \textbf{{n\~ao constitui prova conclusiva de identidade}}. Os resultados apresentados s\~ao probabil\'isticos, dependem da qualidade do material e exigem valida\c{{c}}\~ao humana especializada em conjunto com outros elementos de prova.
\end{{minipage}}}}
\end{{center}}

\begin{{center}}
\fcolorbox{{blue!50!black}}{{blue!4}}{{\begin{{minipage}}{{0.92\linewidth}}
\textbf{{Rastreabilidade forense.}} {escape_latex(self._config.forensics.chain_of_custody_note)} O pipeline tamb\'em preserva logs estruturados, hashes SHA-512, par\^ametros operacionais, keyframes selecionados e metadados de aprimoramento aplicados.
\end{{minipage}}}}
\end{{center}}
"""

    def _methodology(self, result: InventoryResult) -> str:
        items = "\n".join(
            rf"\item {escape_latex(item)}"
            for item in inventory_methodology_items(self._config, result.search)
        )
        return rf"""
\section{{Metodologia}}
\begin{{enumerate}}[leftmargin=*,label=\arabic*.]
{items}
\end{{enumerate}}
"""

    def _group_results(self, result: InventoryResult, tex_path: Path) -> str:
        grouped_tracks = tracks_by_cluster(result)
        keyframes_map = keyframes_by_track(result)
        candidate_map = candidate_cluster_map(result.clusters)
        sections = []
        for cluster in result.clusters:
            tracks = grouped_tracks.get(cluster.cluster_id, [])
            keyframe_count = sum(len(keyframes_map.get(track.track_id, [])) for track in tracks)
            mean_similarity = mean_pairwise_track_similarity(tracks)
            related_groups = candidate_map.get(cluster.cluster_id, [])
            related_text = (
                rf"\textbf{{Rela\c{{c}}\~oes intergrupos.}} "
                + ", ".join(rf"\texttt{{{escape_latex(item)}}}" for item in related_groups)
                + "."
                if related_groups
                else r"\textbf{Rela\c{c}\~oes intergrupos.} Nenhuma acima do limiar configurado."
            )
            sections.append(
                rf"""
\subsection{{Grupo \texttt{{{escape_latex(cluster.cluster_id)}}}}}
\textbf{{Resumo do grupo.}} tracks={len(tracks)}; keyframes={keyframe_count}; ocorr\^encias={len(cluster.occurrence_ids)}; similaridade m\'edia entre tracks={escape_latex(format_group_similarity(mean_similarity, len(tracks)))}.

{related_text}

{self._group_tracks_table(tracks, keyframes_map, tex_path)}
"""
            )
        if not sections:
            sections.append(r"\textit{Nenhum grupo com track representativo foi consolidado.}")
        return "\\section{Resultados por Grupo}\n" + "\n".join(sections)

    def _group_tracks_table(
        self,
        tracks: list[FaceTrack],
        keyframes_map: dict[str, list[KeyFrame]],
        tex_path: Path,
    ) -> str:
        if not tracks:
            return r"\textit{Nenhum track consolidado para este grupo.}"

        rows = "\n".join(
            self._group_track_row(track, keyframes_map, tex_path)
            for track in tracks[: self._config.reporting.max_tracks_per_group]
        )
        return rf"""
\begingroup
\setlength{{\tabcolsep}}{{4pt}}
\renewcommand{{\arraystretch}}{{1.18}}
\begin{{scriptsize}}
\begin{{longtable}}{{T{{0.12\linewidth}}G{{0.16\linewidth}}G{{0.26\linewidth}}L{{0.34\linewidth}}}}
\toprule
Track & Recorte & Quadro de origem & Metadados \\
\midrule
\endhead
{rows}
\bottomrule
\end{{longtable}}
\end{{scriptsize}}
\endgroup
"""

    def _group_track_row(
        self,
        track: FaceTrack,
        keyframes_map: dict[str, list[KeyFrame]],
        tex_path: Path,
    ) -> str:
        keyframe = self._select_representative_keyframe(track, keyframes_map)
        crop_path = keyframe.preview_path if keyframe is not None and keyframe.preview_path is not None else track.preview_path
        context_path = keyframe.context_image_path if keyframe is not None else None
        crop_cell = self._table_image_cell(crop_path, tex_path, width="0.92\\linewidth", height="0.16\\textheight")
        context_cell = self._table_image_cell(context_path, tex_path, width="0.95\\linewidth", height="0.17\\textheight")
        metadata_lines = [
            rf"Arquivo de origem: \texttt{{{break_wrappable_text(track.source_path.name)}}}",
            f"Intervalo temporal do track: {track_interval_text(track)}",
            rf"Faixa de quadros do track: \texttt{{{break_wrappable_text(track_frame_interval_text(track))}}}",
            f"Ocorr\\^encias faciais no track: {len(track.occurrence_ids)}",
            f"Quadros de refer\\^encia do track: {len(track.keyframe_ids)}",
            f"Qualidade facial m\\'edia do track: {track.quality_statistics.mean_quality_score:.3f}",
        ]
        if keyframe is not None:
            metadata_lines.extend(escape_latex(line) for line in keyframe_reference_lines(keyframe))
        metadata = (
            r"\begin{minipage}[t]{\linewidth}\raggedright\setlength{\parskip}{0.25em}\vspace{0pt}"
            + r" \par ".join(metadata_lines)
            + r"\end{minipage}"
        )
        track_label = (
            r"\begin{minipage}[t]{\linewidth}\raggedright\vspace{0pt}\texttt{"
            + escape_latex(track.track_id)
            + r"}\end{minipage}"
        )
        return rf"{track_label} & {crop_cell} & {context_cell} & {metadata} \\"

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

    def _table_image_cell(self, artifact_path: Path | None, tex_path: Path, width: str, height: str) -> str:
        content = self._include_image(artifact_path, tex_path, width=width, height=height)
        return r"\begin{minipage}[t]{\linewidth}\centering\vspace{0pt}" + content + r"\end{minipage}"

    def _technical_appendix(self, result: InventoryResult) -> str:
        hash_rows = "\n".join(
            rf"\texttt{{{break_wrappable_text(str(item.path))}}} & \texttt{{{break_monospace_text(item.sha512)}}} \\"
            for item in result.files
        ) or r"\multicolumn{2}{c}{Nenhum arquivo catalogado.}\\"
        media_info_rows = self._media_info_rows(result.files)
        parameter_items = "\n".join(
            rf"\item {escape_latex(item)}"
            for item in technical_parameter_items(self._config, result.search)
        )
        log_excerpt = self._format_log_excerpt(result.logs_directory / "run.log")
        return rf"""
\section{{Anexo T\'ecnico}}
\subsection{{Hashes SHA-512}}
\begin{{scriptsize}}
\begin{{longtable}}{{T{{0.43\linewidth}}T{{0.49\linewidth}}}}
\toprule
Arquivo & SHA-512 \\
\midrule
\endhead
{hash_rows}
\bottomrule
\end{{longtable}}
\end{{scriptsize}}

\subsection{{Metadados T\'ecnicos da M\'idia}}
\begin{{scriptsize}}
\begin{{longtable}}{{@{{}}T{{0.44\linewidth}}L{{0.42\linewidth}}@{{}}}}
\toprule
Arquivo & Caracter\'isticas \\
\midrule
\endhead
{media_info_rows}
\bottomrule
\end{{longtable}}
\end{{scriptsize}}

\subsection{{Par\^ametros e Melhorias Aplicadas}}
\begin{{itemize}}[leftmargin=*]
{parameter_items}
\item Refer\^encia do software: {software_reference_abnt_latex()}
\end{{itemize}}

\subsection{{Registros de Execu\c{{c}}\~ao}}
\begingroup
\ttfamily
\scriptsize
\raggedright
{log_excerpt}
\par
\endgroup
"""

    def _face_size_statistics_section(self, result: InventoryResult) -> str:
        rows = "\n".join(
            [
                self._face_size_row(
                    "Todas as faces detectadas",
                    result.summary.total_detected_face_sizes.count,
                    result.summary.total_detected_face_sizes.mean_pixels,
                    result.summary.total_detected_face_sizes.stddev_pixels,
                ),
                self._face_size_row(
                    "Faces filtradas e mantidas",
                    result.summary.selected_face_sizes.count,
                    result.summary.selected_face_sizes.mean_pixels,
                    result.summary.selected_face_sizes.stddev_pixels,
                ),
            ]
        )
        return rf"""
\section{{Estat\'isticas de Tamanho das Faces}}
\begingroup
\setlength{{\tabcolsep}}{{3pt}}
\begin{{scriptsize}}
\begin{{longtable}}{{@{{}}L{{0.38\linewidth}}C{{0.16\linewidth}}C{{0.17\linewidth}}C{{0.19\linewidth}}@{{}}}}
\toprule
Conjunto & Quantidade & M\'edia (px) & Desvio padr\~ao (px) \\
\midrule
\endhead
{rows}
\bottomrule
\end{{longtable}}
\end{{scriptsize}}
\endgroup
"""

    def _include_image(self, artifact_path: Path | None, tex_path: Path, width: str, height: str) -> str:
        if artifact_path is None:
            return r"\textit{Artefato n\~ao dispon\'ivel}"
        return rf"\includegraphics[width={width},height={height},keepaspectratio]{{{self._relative_to_tex(artifact_path, tex_path)}}}"

    def _relative_to_tex(self, artifact_path: Path, tex_path: Path) -> str:
        return Path(os.path.relpath(artifact_path, start=tex_path.parent)).as_posix()

    def _format_log_excerpt(self, log_path: Path, max_lines: int = 60) -> str:
        if not log_path.exists():
            return r"\noindent\texttt{Registro principal n\~ao encontrado.}"
        lines = log_path.read_text(encoding="utf-8", errors="ignore").splitlines()[-max_lines:]
        return "\n".join(rf"\noindent\texttt{{{break_wrappable_text(line)}}}\par" for line in lines)

    def _media_info_rows(self, files: list[FileRecord]) -> str:
        media_files = [item for item in files if item.media_type.value in {"image", "video"}]
        if not media_files:
            return r"\multicolumn{2}{c}{Nenhuma m\'idia eleg\'ivel para extra\c{c}\~ao interna de metadados.}\\"
        rows: list[str] = []
        for item in media_files:
            path_text = self._table_text_cell(
                rf"\texttt{{{break_wrappable_text(str(item.path))}}}",
                monospace=True,
            )
            if item.media_info_tracks:
                description = self._media_info_file_description(item.media_info_tracks)
                rows.append(rf"{path_text} & {description} \\")
            else:
                rows.append(
                    rf"{path_text} & {self._table_text_cell(escape_latex(item.media_info_error or 'Metadados tecnicos indisponiveis.'))} \\"
                )
        return "\n".join(rows)

    def _media_info_file_description(self, tracks: tuple[object, ...]) -> str:
        descriptions: list[str] = []
        show_type_label = len(tracks) > 1
        for track in tracks:
            track_title = media_track_type_label(track.track_type)
            block = self._media_info_track_description(
                track.attributes,
                track_title=track_title,
                show_type_label=show_type_label and track_title != "Arquivo",
            )
            descriptions.append(block)
        return r"\par ".join(descriptions)

    def _media_info_track_description(
        self,
        attributes: tuple[MediaInfoAttribute, ...],
        *,
        track_title: str | None = None,
        show_type_label: bool = False,
    ) -> str:
        if not attributes:
            return r"\textit{Nenhuma caracter\'istica dispon\'ivel.}"
        parts: list[str] = []
        if show_type_label and track_title:
            parts.append(rf"\textbf{{{escape_latex(track_title)}}}")
        parts.extend(
            [
            rf"\textbf{{{escape_latex(attribute.label)}}}: {break_wrappable_text(attribute.value)}"
            for attribute in attributes
            ]
        )
        return (
            r"\begin{minipage}[t]{\linewidth}\raggedright\setlength{\parskip}{0.25em}\vspace{0pt}"
            + r" \par ".join(parts)
            + r"\end{minipage}"
        )

    def _table_text_cell(self, content: str, *, monospace: bool = False) -> str:
        font_command = r"\ttfamily\scriptsize " if monospace else ""
        return (
            r"\begin{minipage}[t]{\linewidth}\raggedright\vspace{0pt}"
            + font_command
            + content
            + r"\end{minipage}"
        )

    def _face_size_row(
        self,
        label: str,
        count: int,
        mean_pixels: float | None,
        stddev_pixels: float | None,
    ) -> str:
        mean_text = "-" if mean_pixels is None else f"{mean_pixels:.2f}"
        stddev_text = "-" if stddev_pixels is None else f"{stddev_pixels:.2f}"
        return rf"{escape_latex(label)} & {count} & {mean_text} & {stddev_text} \\"
