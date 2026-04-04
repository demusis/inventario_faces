from __future__ import annotations

from collections import defaultdict
import os
from pathlib import Path

from inventario_faces import __version__
from inventario_faces.domain.config import AppConfig
from inventario_faces.domain.entities import (
    FileRecord,
    FaceCluster,
    FaceOccurrence,
    InventoryResult,
    MediaInfoAttribute,
    ReportArtifacts,
)
from inventario_faces.infrastructure.latex_compiler import LatexCompiler
from inventario_faces.utils.latex import (
    break_monospace_text,
    break_wrappable_text,
    escape_latex,
    format_seconds,
)
from inventario_faces.utils.path_utils import ensure_directory

PROJECT_URL = "https://github.com/demusis/inventario_faces"


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
            pdf_path = self._compiler.compile(tex_path)
        return ReportArtifacts(tex_path=tex_path, pdf_path=pdf_path)

    def _render_tex(self, result: InventoryResult, tex_path: Path) -> str:
        template = self._load_template()
        body = "\n".join(
            [
                self._executive_summary(result),
                self._methodology(result),
                self._results(result, tex_path),
                self._technical_appendix(result),
            ]
        )
        return (
            template.replace("@@TITLE@@", escape_latex(self._config.app.report_title))
            .replace("@@ORGANIZATION@@", escape_latex(self._config.app.organization))
            .replace("@@DATE@@", escape_latex(result.finished_at_utc.strftime("%Y-%m-%d %H:%M:%SZ")))
            .replace("@@BODY@@", body)
        )

    def _load_template(self) -> str:
        template_path = Path(__file__).resolve().parent / "templates" / "forensic_report_template.tex"
        return template_path.read_text(encoding="utf-8")

    def _executive_summary(self, result: InventoryResult) -> str:
        summary = result.summary
        total_files = self._count_phrase(summary.total_files, "arquivo", "arquivos")
        supported_media = self._count_phrase(summary.media_files, "m\\'idia suportada", "m\\'idias suportadas")
        images = self._count_phrase(summary.image_files, "imagem", "imagens")
        videos = self._count_phrase(summary.video_files, "v\\'ideo", "v\\'ideos")
        occurrences = self._count_phrase(summary.total_occurrences, "ocorr\\^encia facial", "ocorr\\^encias faciais")
        individuals = self._count_phrase(
            summary.total_clusters,
            "poss\\'ivel indiv\\'iduo",
            "poss\\'iveis indiv\\'iduos",
        )
        probable_pairs = self._count_phrase(
            summary.probable_match_pairs,
            "par de poss\\'iveis indiv\\'iduos",
            "pares de poss\\'iveis indiv\\'iduos",
        )
        eligibility = "eleg\\'ivel" if summary.media_files == 1 else "eleg\\'iveis"
        root_directory = break_wrappable_text(str(result.root_directory))
        return rf"""
\section{{Resumo Executivo}}
O presente relat\'orio registra o processamento automatizado do diret\'orio \texttt{{{root_directory}}}. Foram catalogados e submetidos a c\'alculo de hash {total_files}. No conjunto catalogado, h\'a {supported_media} {eligibility} para an\'alise facial automatizada, com a seguinte composi\c{{c}}\~ao: {images} e {videos}. A etapa de detec\c{{c}}\~ao registrou {occurrences}. A etapa de agrupamento resultou em {individuals}. Foram assinalados {probable_pairs} como possivelmente correlatos, sem car\'ater conclusivo de identifica\c{{c}}\~ao.

\begin{{center}}
\fcolorbox{{yellow!40!black}}{{yellow!8}}{{\begin{{minipage}}{{0.92\linewidth}}
\textbf{{Advert\^encia pericial.}} Os resultados s\~ao probabil\'isticos e n\~ao constituem identifica\c{{c}}\~ao conclusiva de indiv\'iduos. Qualquer infer\^encia deve ser submetida \`a revis\~ao humana especializada e correlacionada com outros elementos de prova.
\end{{minipage}}}}
\end{{center}}

\begin{{center}}
\fcolorbox{{blue!50!black}}{{blue!4}}{{\begin{{minipage}}{{0.92\linewidth}}
\textbf{{Cadeia de cust\'odia.}} {escape_latex(self._config.forensics.chain_of_custody_note)}
\end{{minipage}}}}
\end{{center}}
"""

    def _count_phrase(self, count: int, singular: str, plural: str) -> str:
        label = singular if count == 1 else plural
        return f"{count} {label}"

    def _methodology(self, result: InventoryResult) -> str:
        max_frames = self._config.video.max_frames_per_video
        max_frames_text = str(max_frames) if max_frames is not None else "sem limite"
        run_directory = break_wrappable_text(str(result.run_directory))
        detection_size = self._detection_size_label()
        return rf"""
\section{{Metodologia}}
\begin{{enumerate}}[leftmargin=*,label=\arabic*.]
\item Varredura recursiva do diret\'orio de entrada, com registro individual de caminho, tamanho e hash SHA-512 para cada arquivo encontrado.
\item Classifica\c{{c}}\~ao de m\'idia em imagem, v\'ideo ou outro formato, sem altera\c{{c}}\~ao dos arquivos originais.
\item Para v\'ideos, amostragem temporal a cada {self._config.video.sampling_interval_seconds:.2f} segundos, limitada a {escape_latex(max_frames_text)} quadros por arquivo.
\item Detec\c{{c}}\~ao facial e extra\c{{c}}\~ao de vetores de caracter\'isticas normalizados com mecanismo configurado em \texttt{{{escape_latex(self._config.face_model.backend)}}} / \texttt{{{escape_latex(self._config.face_model.model_name)}}}, com tamanho de detec\c{{c}}\~ao definido em {detection_size}.
\item Sele\c{{c}}\~ao de faces condicionada a qualidade m\'inima de {self._config.face_model.minimum_face_quality:.2f}, aferida pela pontua\c{{c}}\~ao de detec\c{{c}}\~ao do mecanismo facial.
\item Sele\c{{c}}\~ao complementar por tamanho m\'inimo da face, exigindo ao menos {self._config.face_model.minimum_face_size_pixels} pixels na menor dimens\~ao da caixa delimitadora.
\item Agrupamento incremental por similaridade de cosseno, com limiar de atribui\c{{c}}\~ao {self._config.clustering.assignment_similarity:.2f} e limiar de sugest\~ao entre poss\'iveis indiv\'iduos {self._config.clustering.candidate_similarity:.2f}.
\item Consolida\c{{c}}\~ao de vest\'igios forenses, registros e artefatos derivados em diret\'orio dedicado de execu\c{{c}}\~ao: \texttt{{{run_directory}}}.
\end{{enumerate}}
"""

    def _results(self, result: InventoryResult, tex_path: Path) -> str:
        occurrence_map: dict[str, list[FaceOccurrence]] = defaultdict(list)
        for occurrence in result.occurrences:
            if occurrence.cluster_id is not None:
                occurrence_map[occurrence.cluster_id].append(occurrence)

        individuals_table = "\n".join(
            rf"\texttt{{{escape_latex(cluster.cluster_id)}}} & {len(cluster.occurrence_ids)} & \texttt{{{escape_latex(', '.join(cluster.candidate_cluster_ids) or '-')}}} \\"
            for cluster in result.clusters
        ) or r"\multicolumn{3}{c}{Nenhum poss\'ivel indiv\'iduo identificado.}\\"

        individual_sections = "\n".join(
            self._cluster_section(cluster, occurrence_map[cluster.cluster_id], tex_path)
            for cluster in result.clusters
        )
        face_stats_table = self._face_statistics_table(result)

        return rf"""
\section{{Resultados}}
\subsection{{Estat\'isticas de tamanho das faces}}
{face_stats_table}

\subsection{{Poss\'iveis indiv\'iduos}}
\begin{{longtable}}{{S{{0.16\linewidth}}S{{0.16\linewidth}}L{{0.48\linewidth}}}}
\toprule
Identificador & Ocorr\^encias & Poss\'iveis indiv\'iduos correlatos \\
\midrule
\endhead
{individuals_table}
\bottomrule
\end{{longtable}}

{individual_sections}
"""

    def _face_statistics_table(self, result: InventoryResult) -> str:
        return rf"""
\begingroup
\setlength{{\tabcolsep}}{{4pt}}
\renewcommand{{\arraystretch}}{{1.1}}
\begin{{scriptsize}}
\begin{{longtable}}{{L{{0.27\linewidth}}S{{0.10\linewidth}}S{{0.10\linewidth}}S{{0.10\linewidth}}S{{0.10\linewidth}}S{{0.10\linewidth}}}}
\toprule
Conjunto & N\'umero de faces & M\'inimo (px) & M\'aximo (px) & M\'edia (px) & Desvio padr\~ao\\(px) \\
\midrule
\endhead
Faces detectadas antes dos filtros & {self._stats_row(result.summary.total_detected_face_sizes)} \\
Faces selecionadas ap\'os os filtros & {self._stats_row(result.summary.selected_face_sizes)} \\
\bottomrule
\end{{longtable}}
\end{{scriptsize}}
\endgroup
"""

    def _cluster_section(
        self,
        cluster: FaceCluster,
        occurrences: list[FaceOccurrence],
        tex_path: Path,
    ) -> str:
        correlates = rf"\texttt{{{escape_latex(', '.join(cluster.candidate_cluster_ids) or '-')}}}"
        gallery_rows = "\n".join(
            self._gallery_row(item, tex_path)
            for item in occurrences[: self._config.reporting.max_gallery_faces_per_group]
        ) or r"\multicolumn{3}{c}{Nenhuma ilustra\c{c}\~ao dispon\'ivel para este poss\'ivel indiv\'iduo.}\\"

        details_rows = "\n".join(
            rf"\texttt{{{escape_latex(item.occurrence_id)}}} & \texttt{{{break_wrappable_text(item.source_path.name)}}} & {format_seconds(item.frame_timestamp_seconds)} & {item.detection_score:.3f} \\"
            for item in occurrences
        ) or r"\multicolumn{4}{c}{Nenhuma ocorr\^encia detalhada.}\\"

        return rf"""
\subsection{{Poss\'ivel indiv\'iduo \texttt{{{escape_latex(cluster.cluster_id)}}}}}
\textbf{{Total de ocorr\^encias:}} {len(occurrences)}\\
\textbf{{Poss\'iveis indiv\'iduos correlatos:}} {correlates}

\paragraph{{Galeria comparativa}}
\begingroup
\renewcommand{{\arraystretch}}{{1.2}}
\begin{{longtable}}{{G{{0.20\linewidth}}G{{0.42\linewidth}}L{{0.28\linewidth}}}}
\toprule
Recorte facial & Imagem ou quadro de origem & Dados da ocorr\^encia \\
\midrule
\endhead
{gallery_rows}
\bottomrule
\end{{longtable}}
\endgroup

\paragraph{{Ocorr\^encias do poss\'ivel indiv\'iduo}}
\begin{{longtable}}{{T{{0.18\linewidth}}T{{0.40\linewidth}}S{{0.17\linewidth}}S{{0.12\linewidth}}}}
\toprule
Ocorr\^encia & Arquivo & Marca temporal & Pontua\c{{c}}\~ao \\
\midrule
\endhead
{details_rows}
\bottomrule
\end{{longtable}}
"""

    def _gallery_row(self, occurrence: FaceOccurrence, tex_path: Path) -> str:
        crop_cell = self._top_aligned_visual_cell(
            self._include_image(
                occurrence.crop_path,
                tex_path,
                width="0.90\\linewidth",
                height="0.18\\textheight",
            )
        )
        context_cell = self._top_aligned_visual_cell(
            self._include_image(
                occurrence.context_image_path,
                tex_path,
                width="0.95\\linewidth",
                height="0.18\\textheight",
            )
        )
        bbox_text = break_wrappable_text(
            f"{occurrence.bbox.x1:.1f},{occurrence.bbox.y1:.1f},{occurrence.bbox.x2:.1f},{occurrence.bbox.y2:.1f}",
            {",", "."},
        )
        metadata = " \\par ".join(
            [
                rf"\texttt{{{escape_latex(occurrence.occurrence_id)}}}",
                rf"Arquivo: \texttt{{{break_wrappable_text(occurrence.source_path.name)}}}",
                f"Marca temporal: {format_seconds(occurrence.frame_timestamp_seconds)}",
                f"Pontua\\c{{c}}\\~ao: {occurrence.detection_score:.3f}",
                rf"Caixa delimitadora: \texttt{{{bbox_text}}}",
            ]
        )
        metadata_cell = (
            r"\begin{minipage}[t]{\linewidth}\raggedright\setlength{\parskip}{0.35em}\vspace{0pt}"
            + metadata
            + r"\end{minipage}"
        )
        return rf"{crop_cell} & {context_cell} & {metadata_cell} \\"

    def _include_image(self, artifact_path: Path | None, tex_path: Path, width: str, height: str) -> str:
        if artifact_path is None:
            return r"\textit{Artefato n\~ao dispon\'ivel}"
        return rf"\includegraphics[width={width},height={height},keepaspectratio]{{{self._relative_to_tex(artifact_path, tex_path)}}}"

    def _top_aligned_visual_cell(self, content: str) -> str:
        return r"\begin{minipage}[t]{\linewidth}\centering\vspace{0pt}" + content + r"\end{minipage}"

    def _technical_appendix(self, result: InventoryResult) -> str:
        hash_rows = "\n".join(
            rf"\texttt{{{break_wrappable_text(str(item.path))}}} & \texttt{{{break_monospace_text(item.sha512)}}} \\"
            for item in result.files
        ) or r"\multicolumn{2}{c}{Nenhum arquivo catalogado.}\\"
        media_info_rows = self._media_info_rows(result.files)
        log_excerpt = self._format_log_excerpt(result.logs_directory / "run.log")
        detection_size = self._detection_size_label()
        providers = break_wrappable_text(", ".join(self._config.face_model.providers) or "automatico")
        image_extensions = break_wrappable_text(", ".join(self._config.media.image_extensions))
        video_extensions = break_wrappable_text(", ".join(self._config.media.video_extensions))
        max_frames_text = escape_latex(
            str(self._config.video.max_frames_per_video)
            if self._config.video.max_frames_per_video is not None
            else "sem limite"
        )
        mediainfo_directory = break_wrappable_text(
            self._config.app.mediainfo_directory or "nao configurado; resolucao automatica pelo PATH do sistema"
        )
        return rf"""
\section{{Anexo t\'ecnico}}
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

\subsection{{Caracter\'isticas de imagens e v\'ideos (MediaInfo)}}
\begin{{scriptsize}}
\begin{{longtable}}{{T{{0.29\linewidth}}S{{0.12\linewidth}}L{{0.50\linewidth}}}}
\toprule
Arquivo & Fluxo & Caracter\'isticas \\
\midrule
\endhead
{media_info_rows}
\bottomrule
\end{{longtable}}
\end{{scriptsize}}

\subsection{{Par\^ametros do modelo e da execu\c{{c}}\~ao}}
\begin{{itemize}}[leftmargin=*]
\item Aplica\c{{c}}\~ao: vers\~ao=\texttt{{{escape_latex(__version__)}}}; diret\'orio de sa\'ida=\texttt{{{break_wrappable_text(self._config.app.output_directory_name)}}}; n\'ivel de log=\texttt{{{escape_latex(self._config.app.log_level)}}}.
\item M\'idias: extens\~oes de imagem=\texttt{{{image_extensions}}}; extens\~oes de v\'ideo=\texttt{{{video_extensions}}}.
\item V\'ideo: intervalo de amostragem={self._config.video.sampling_interval_seconds:.2f} s; m\'aximo de quadros por arquivo={max_frames_text}.
\item An\'alise facial: mecanismo=\texttt{{{escape_latex(self._config.face_model.backend)}}}; modelo=\texttt{{{escape_latex(self._config.face_model.model_name)}}}; tamanho de detec\c{{c}}\~ao={detection_size}; qualidade m\'inima={self._config.face_model.minimum_face_quality:.2f}; tamanho m\'inimo da face={self._config.face_model.minimum_face_size_pixels} px; contexto={self._config.face_model.ctx_id}; mecanismos de execu\c{{c}}\~ao=\texttt{{{providers}}}.
\item Agrupamento: limiar de atribui\c{{c}}\~ao={self._config.clustering.assignment_similarity:.2f}; limiar de sugest\~ao entre poss\'iveis indiv\'iduos={self._config.clustering.candidate_similarity:.2f}; tamanho m\'inimo do grupo={self._config.clustering.min_cluster_size}.
\item Relat\'orio: faces m\'aximas por poss\'ivel indiv\'iduo na galeria={self._config.reporting.max_gallery_faces_per_group}; compila\c{{c}}\~ao autom\'atica do PDF={'sim' if self._config.reporting.compile_pdf else 'nao'}.
\item MediaInfo: {self._mediainfo_status(result.files)}; diret\'orio configurado=\texttt{{{mediainfo_directory}}}.
\item Software utilizado: aplica\c{{c}}\~ao de c\'odigo aberto dispon\'ivel em \url{{{PROJECT_URL}}}.
\end{{itemize}}

\subsection{{Registros de execu\c{{c}}\~ao}}
\begingroup
\ttfamily
\scriptsize
\raggedright
{log_excerpt}
\par
\endgroup
"""

    def _format_log_excerpt(self, log_path: Path, max_lines: int = 60) -> str:
        if not log_path.exists():
            return r"\noindent\texttt{Registro principal n\~ao encontrado.}"
        lines = log_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        excerpt = lines[-max_lines:]
        if not excerpt:
            return r"\noindent\texttt{Nenhum registro dispon\'ivel.}"
        return "\n".join(
            rf"\noindent\texttt{{{break_wrappable_text(line)}}}\par"
            for line in excerpt
        )

    def _relative_to_tex(self, artifact_path: Path, tex_path: Path) -> str:
        return Path(os.path.relpath(artifact_path, start=tex_path.parent)).as_posix()

    def _media_info_rows(self, files: list[FileRecord]) -> str:
        media_files = [item for item in files if item.media_type.value in {"image", "video"}]
        if not media_files:
            return r"\multicolumn{3}{c}{Nenhuma m\'idia eleg\'ivel para extra\c{c}\~ao via MediaInfo.}\\"

        rows: list[str] = []
        for item in media_files:
            path_text = rf"\texttt{{{break_wrappable_text(str(item.path))}}}"
            if item.media_info_tracks:
                first_row = True
                for track in item.media_info_tracks:
                    characteristics = self._media_info_track_description(track.attributes)
                    file_cell = path_text if first_row else ""
                    first_row = False
                    rows.append(rf"{file_cell} & {escape_latex(track.track_type)} & {characteristics} \\")
                continue

            message = escape_latex(item.media_info_error or "MediaInfo nao disponivel para este arquivo.")
            rows.append(rf"{path_text} & - & {message} \\")
        return "\n".join(rows)

    def _media_info_track_description(self, attributes: tuple[MediaInfoAttribute, ...]) -> str:
        if not attributes:
            return r"\textit{Nenhuma caracter\'istica dispon\'ivel.}"
        parts = [
            rf"\textbf{{{escape_latex(attribute.label)}}}: {break_wrappable_text(attribute.value)}"
            for attribute in attributes
        ]
        return (
            r"\begin{minipage}[t]{\linewidth}\raggedright\setlength{\parskip}{0.25em}\vspace{0pt}"
            + r" \par ".join(parts)
            + r"\end{minipage}"
        )

    def _detection_size_label(self) -> str:
        if self._config.face_model.det_size is None:
            return "resolu\\c{c}\\~ao original do arquivo ou quadro"
        return f"{self._config.face_model.det_size[0]} x {self._config.face_model.det_size[1]}"

    def _stats_row(self, statistics) -> str:
        if statistics.count == 0:
            return r"0 & - & - & - & -"
        return (
            f"{statistics.count} & "
            f"{statistics.min_pixels:.1f} & "
            f"{statistics.max_pixels:.1f} & "
            f"{statistics.mean_pixels:.1f} & "
            f"{statistics.stddev_pixels:.1f}"
        )

    def _mediainfo_status(self, files: list[FileRecord]) -> str:
        media_files = [item for item in files if item.media_type.value in {"image", "video"}]
        if not media_files:
            return "nenhuma m\\'idia eleg\\'ivel para coleta"
        if any(item.media_info_tracks for item in media_files):
            return (
                "coleta executada para m\\'idias eleg\\'iveis, "
                "com resultados detalhados na subse\\c{c}\\~ao anterior"
            )
        error_messages = [item.media_info_error for item in media_files if item.media_info_error]
        if error_messages:
            return (
                rf"coleta n\~ao realizada ou indispon\'ivel no ambiente; "
                rf"motivo predominante=\texttt{{{break_wrappable_text(error_messages[0])}}}"
            )
        return "coleta n\\~ao realizada"
