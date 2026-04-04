from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from inventario_faces.domain.config import AppConfig
from inventario_faces.domain.entities import (
    FaceSizeStatistics,
    FaceOccurrence,
    FileRecord,
    InventoryResult,
    MediaType,
    ProcessingSummary,
    ReportArtifacts,
    SampledFrame,
)
from inventario_faces.domain.protocols import (
    FaceAnalyzer,
    LogCallback,
    MediaInfoExtractor,
    ProgressCallback,
    ReportGenerator,
)
from inventario_faces.infrastructure.artifact_store import ArtifactStore
from inventario_faces.infrastructure.logging_setup import (
    StructuredEventLogger,
    build_file_logger,
    close_file_logger,
)
from inventario_faces.services.clustering_service import ClusteringService
from inventario_faces.services.export_service import ExportService
from inventario_faces.services.hashing_service import HashingService
from inventario_faces.services.scanner_service import ScannerService
from inventario_faces.services.video_service import VideoService
from inventario_faces.utils.path_utils import ensure_directory
from inventario_faces.utils.time_utils import as_utc, utc_now


@dataclass(frozen=True)
class FrameProcessingSummary:
    raw_detection_count: int
    selected_detection_count: int
    raw_face_sizes: tuple[float, ...]
    selected_face_sizes: tuple[float, ...]
    occurrence_ids: tuple[str, ...]


class InventoryService:
    def __init__(
        self,
        config: AppConfig,
        scanner_service: ScannerService,
        hashing_service: HashingService,
        media_service: VideoService,
        clustering_service: ClusteringService,
        report_generator: ReportGenerator,
        face_analyzer_factory: Callable[[], FaceAnalyzer],
        media_info_extractor: MediaInfoExtractor | None = None,
    ) -> None:
        self._config = config
        self._scanner_service = scanner_service
        self._hashing_service = hashing_service
        self._media_service = media_service
        self._clustering_service = clustering_service
        self._report_generator = report_generator
        self._face_analyzer_factory = face_analyzer_factory
        self._media_info_extractor = media_info_extractor

    def run(
        self,
        root_directory: Path,
        progress_callback: ProgressCallback | None = None,
        log_callback: LogCallback | None = None,
    ) -> InventoryResult:
        root_directory = Path(root_directory).resolve()
        if not root_directory.exists():
            raise FileNotFoundError(f"Diretorio nao encontrado: {root_directory}")

        started_at_utc = utc_now()
        output_root = root_directory / self._config.app.output_directory_name
        total_files, media_counter = self._scanner_service.summarize(
            root_directory,
            excluded_directories={output_root},
        )
        run_directory = ensure_directory(
            output_root / f"run_{started_at_utc.strftime('%Y%m%d_%H%M%S')}"
        )
        logs_directory = ensure_directory(run_directory / "logs")
        text_logger = build_file_logger(logs_directory, self._config.app.log_level)
        event_logger = StructuredEventLogger(logs_directory / "events.jsonl")
        artifact_store = ArtifactStore(run_directory)
        export_service = ExportService(run_directory)

        try:
            self._emit_progress(progress_callback, 0, total_files, "Inicializando analise")
            self._emit_log(text_logger, log_callback, f"Diretorio analisado: {root_directory}")
            self._emit_log(text_logger, log_callback, f"Diretorio de execucao: {run_directory}")
            self._emit_log(text_logger, log_callback, f"Diretorio de logs: {logs_directory}")
            self._emit_log(
                text_logger,
                log_callback,
                "Modo de processamento em fluxo ativado para reduzir uso de memoria em acervos extensos.",
            )
            self._emit_log(
                text_logger,
                log_callback,
                (
                    "Varredura concluida: "
                    f"{total_files} arquivos localizados "
                    f"({media_counter[MediaType.IMAGE]} imagens, "
                    f"{media_counter[MediaType.VIDEO]} videos, "
                    f"{media_counter[MediaType.OTHER]} outros formatos)."
                ),
            )
            self._emit_log(
                text_logger,
                log_callback,
                "Inicializando mecanismo facial e carregando configuracao operacional.",
            )
            analyzer = self._face_analyzer_factory()
            providers = list(getattr(analyzer, "providers", []))
            for line in self._configuration_log_lines(providers):
                self._emit_log(text_logger, log_callback, line)
            event_logger.write(
                "run_started",
                root_directory=root_directory,
                run_directory=run_directory,
                total_files=total_files,
                image_files=media_counter[MediaType.IMAGE],
                video_files=media_counter[MediaType.VIDEO],
                other_files=media_counter[MediaType.OTHER],
                providers=providers,
                configuration=self._config,
            )

            file_records: list[FileRecord] = []
            occurrences: list[FaceOccurrence] = []
            total_detected_face_sizes: list[float] = []
            selected_face_sizes: list[float] = []

            for index, file_path in enumerate(
                self._scanner_service.iter_scan(root_directory, excluded_directories={output_root}),
                start=1,
            ):
                media_type = self._scanner_service.classify(file_path)
                file_prefix = f"[Arquivo {index}/{total_files}]"
                self._emit_progress(
                    progress_callback,
                    index - 1,
                    total_files,
                    f"Processando {file_path.name}",
                )
                self._emit_log(
                    text_logger,
                    log_callback,
                    f"{file_prefix} Inicio do processamento | tipo={self._media_type_label(media_type)} | caminho={file_path}",
                )
                if len(str(file_path)) >= 240:
                    self._emit_log(
                        text_logger,
                        log_callback,
                        f"{file_prefix} Caminho extenso detectado; rotinas de leitura robusta para Windows foram habilitadas.",
                    )

                discovered_at_utc = utc_now()
                sha512 = self._hashing_service.sha512(file_path)
                stat = file_path.stat()
                modified_at_utc = as_utc(stat.st_mtime)
                processing_error: str | None = None
                media_info_error: str | None = None
                media_info_tracks = ()
                initial_occurrence_count = len(occurrences)

                self._emit_log(
                    text_logger,
                    log_callback,
                    (
                        f"{file_prefix} Hash SHA-512 calculado | "
                        f"tamanho={self._format_size_bytes(stat.st_size)} | "
                        f"alteracao={modified_at_utc.isoformat() if modified_at_utc else '-'} | "
                        f"hash={self._short_hash(sha512)}"
                    ),
                )

                if media_type in {MediaType.IMAGE, MediaType.VIDEO}:
                    media_info_tracks, media_info_error = self._extract_media_info(file_path)
                    if media_info_tracks:
                        self._emit_log(
                            text_logger,
                            log_callback,
                            (
                                f"{file_prefix} MediaInfo extraido | "
                                f"fluxos={len(media_info_tracks)}"
                            ),
                        )
                    elif media_info_error:
                        self._emit_log(
                            text_logger,
                            log_callback,
                            f"{file_prefix} MediaInfo indisponivel | motivo={media_info_error}",
                        )

                try:
                    if media_type == MediaType.IMAGE:
                        frame_summary = self._process_frame(
                            analyzer=analyzer,
                            frame=self._media_service.load_image(file_path),
                            sha512=sha512,
                            media_type=media_type,
                            occurrences=occurrences,
                            artifact_store=artifact_store,
                            event_logger=event_logger,
                            logger=text_logger,
                            log_callback=log_callback,
                            file_prefix=file_prefix,
                        )
                        self._emit_log(
                            text_logger,
                            log_callback,
                            (
                                f"{file_prefix} Imagem analisada | "
                                f"faces detectadas={frame_summary.raw_detection_count} | "
                                f"faces selecionadas={frame_summary.selected_detection_count}"
                            ),
                        )
                        total_detected_face_sizes.extend(frame_summary.raw_face_sizes)
                        selected_face_sizes.extend(frame_summary.selected_face_sizes)
                    elif media_type == MediaType.VIDEO:
                        self._emit_log(
                            text_logger,
                            log_callback,
                            (
                                f"{file_prefix} Video identificado | "
                                f"intervalo de amostragem={self._config.video.sampling_interval_seconds:.2f}s | "
                                f"limite de quadros="
                                f"{self._config.video.max_frames_per_video if self._config.video.max_frames_per_video is not None else 'sem limite'}"
                            ),
                        )
                        sampled_frames = 0
                        frames_with_faces = 0
                        detected_faces = 0
                        selected_faces = 0
                        for frame in self._media_service.sample_video(file_path):
                            sampled_frames += 1
                            self._emit_progress(
                                progress_callback,
                                index - 1,
                                total_files,
                                f"Processando {file_path.name} | quadro amostrado {sampled_frames}",
                            )
                            frame_summary = self._process_frame(
                                analyzer=analyzer,
                                frame=frame,
                                sha512=sha512,
                                media_type=media_type,
                                occurrences=occurrences,
                                artifact_store=artifact_store,
                                event_logger=event_logger,
                                logger=text_logger,
                                log_callback=log_callback,
                                file_prefix=file_prefix,
                            )
                            total_detected_face_sizes.extend(frame_summary.raw_face_sizes)
                            selected_face_sizes.extend(frame_summary.selected_face_sizes)
                            detected_faces += frame_summary.raw_detection_count
                            selected_faces += frame_summary.selected_detection_count
                            if frame_summary.selected_detection_count > 0:
                                frames_with_faces += 1
                        self._emit_log(
                            text_logger,
                            log_callback,
                            (
                                f"{file_prefix} Video analisado | "
                                f"quadros amostrados={sampled_frames} | "
                                f"quadros com faces={frames_with_faces} | "
                                f"faces detectadas={detected_faces} | "
                                f"faces selecionadas={selected_faces}"
                            ),
                        )
                    else:
                        self._emit_log(
                            text_logger,
                            log_callback,
                            (
                                f"{file_prefix} Arquivo fora do escopo da analise facial. "
                                "O item foi mantido apenas no inventario forense."
                            ),
                        )

                    event_logger.write(
                        "file_processed",
                        path=file_path,
                        media_type=media_type,
                        sha512=sha512,
                        size_bytes=stat.st_size,
                        occurrences_generated=len(occurrences) - initial_occurrence_count,
                        media_info_tracks=media_info_tracks,
                        media_info_error=media_info_error,
                    )
                    self._emit_log(
                        text_logger,
                        log_callback,
                        (
                            f"{file_prefix} Processamento concluido | "
                            f"novas ocorrencias={len(occurrences) - initial_occurrence_count}"
                        ),
                    )
                except Exception as exc:
                    processing_error = str(exc)
                    self._emit_log(
                        text_logger,
                        log_callback,
                        f"{file_prefix} Erro de processamento: {processing_error}",
                    )
                    text_logger.exception("Falha ao processar %s", file_path)
                    event_logger.write(
                        "file_processing_error",
                        path=file_path,
                        media_type=media_type,
                        sha512=sha512,
                        error=processing_error,
                        media_info_tracks=media_info_tracks,
                        media_info_error=media_info_error,
                    )

                file_records.append(
                    FileRecord(
                        path=file_path,
                        media_type=media_type,
                        sha512=sha512,
                        size_bytes=stat.st_size,
                        discovered_at_utc=discovered_at_utc,
                        modified_at_utc=modified_at_utc,
                        processing_error=processing_error,
                        media_info_tracks=media_info_tracks,
                        media_info_error=media_info_error,
                    )
                )
                self._emit_progress(progress_callback, index, total_files, f"Concluido: {file_path.name}")

            self._emit_log(
                text_logger,
                log_callback,
                f"[Agrupamento] Iniciando consolidacao de {len(occurrences)} ocorrencias faciais em possiveis individuos.",
            )
            clusters = self._clustering_service.cluster(occurrences)
            finished_at_utc = utc_now()
            summary = self._build_summary(file_records, occurrences, clusters, total_detected_face_sizes, selected_face_sizes)
            self._emit_log(
                text_logger,
                log_callback,
                (
                    f"[Agrupamento] Concluido | possiveis individuos={summary.total_clusters} | "
                    f"pares possivelmente correlatos={summary.probable_match_pairs}"
                ),
            )
            self._emit_log(
                text_logger,
                log_callback,
                self._face_size_statistics_log_line(
                    "Faces detectadas antes dos filtros",
                    summary.total_detected_face_sizes,
                ),
            )
            self._emit_log(
                text_logger,
                log_callback,
                self._face_size_statistics_log_line(
                    "Faces selecionadas apos os filtros",
                    summary.selected_face_sizes,
                ),
            )
            report_stub = ReportArtifacts(
                tex_path=run_directory / "report" / "relatorio_forense.tex",
                pdf_path=None,
                docx_path=run_directory / "report" / "relatorio_forense.docx",
            )
            manifest_path = export_service.inventory_directory / "manifest.json"
            preliminary_result = InventoryResult(
                run_directory=run_directory,
                started_at_utc=started_at_utc,
                finished_at_utc=finished_at_utc,
                root_directory=root_directory,
                files=file_records,
                occurrences=occurrences,
                clusters=clusters,
                report=report_stub,
                summary=summary,
                logs_directory=logs_directory,
                manifest_path=manifest_path,
            )

            self._emit_log(
                text_logger,
                log_callback,
                f"[Exportacao] Gravando inventario estruturado em {export_service.inventory_directory}.",
            )
            files_csv_path = export_service.write_files_csv(file_records)
            occurrences_csv_path = export_service.write_occurrences_csv(occurrences)
            clusters_json_path = export_service.write_clusters_json(clusters)
            media_info_json_path = export_service.write_media_info_json(file_records)
            self._emit_log(
                text_logger,
                log_callback,
                (
                    f"[Exportacao] Artefatos estruturados gerados | "
                    f"arquivos={files_csv_path.name} | ocorrencias={occurrences_csv_path.name} | "
                    f"grupos={clusters_json_path.name} | mediainfo={media_info_json_path.name}"
                ),
            )
            self._emit_log(text_logger, log_callback, "[Relatorio] Gerando relatorio tecnico.")
            report_artifacts = self._report_generator.generate(preliminary_result)
            self._emit_log(
                text_logger,
                log_callback,
                (
                    f"[Relatorio] Artefatos gerados | TEX={report_artifacts.tex_path} | "
                    f"PDF={report_artifacts.pdf_path if report_artifacts.pdf_path is not None else 'nao compilado'} | "
                    f"DOCX={report_artifacts.docx_path if report_artifacts.docx_path is not None else 'nao gerado'}"
                ),
            )

            result = InventoryResult(
                run_directory=run_directory,
                started_at_utc=started_at_utc,
                finished_at_utc=finished_at_utc,
                root_directory=root_directory,
                files=file_records,
                occurrences=occurrences,
                clusters=clusters,
                report=report_artifacts,
                summary=summary,
                logs_directory=logs_directory,
                manifest_path=manifest_path,
            )
            manifest_output_path = export_service.write_manifest(result)
            self._emit_log(
                text_logger,
                log_callback,
                f"[Exportacao] Manifesto consolidado em {manifest_output_path}.",
            )

            event_logger.write(
                "run_finished",
                summary=summary,
                report_pdf=report_artifacts.pdf_path,
                report_tex=report_artifacts.tex_path,
                report_docx=report_artifacts.docx_path,
            )
            self._emit_log(
                text_logger,
                log_callback,
                (
                    "Processamento concluido. "
                    f"Arquivos catalogados={summary.total_files} | "
                    f"midias suportadas={summary.media_files} | "
                    f"faces detectadas={summary.total_occurrences} | "
                    f"possiveis individuos={summary.total_clusters}."
                ),
            )
            return result
        finally:
            close_file_logger(text_logger)

    def _process_frame(
        self,
        analyzer: FaceAnalyzer,
        frame: SampledFrame,
        sha512: str,
        media_type: MediaType,
        occurrences: list[FaceOccurrence],
        artifact_store: ArtifactStore,
        event_logger: StructuredEventLogger,
        logger: logging.Logger,
        log_callback: LogCallback | None,
        file_prefix: str,
    ) -> FrameProcessingSummary:
        minimum_face_quality = self._config.face_model.minimum_face_quality
        minimum_face_size_pixels = self._config.face_model.minimum_face_size_pixels
        raw_detections = analyzer.analyze(frame)
        raw_face_sizes = tuple(min(detection.bbox.width, detection.bbox.height) for detection in raw_detections)
        quality_filtered_detections = [
            detection for detection in raw_detections if detection.detection_score >= minimum_face_quality
        ]
        discarded_low_quality = len(raw_detections) - len(quality_filtered_detections)
        detections = [
            detection
            for detection in quality_filtered_detections
            if min(detection.bbox.width, detection.bbox.height) >= minimum_face_size_pixels
        ]
        selected_sizes = tuple(min(detection.bbox.width, detection.bbox.height) for detection in detections)
        discarded_small_faces = len(quality_filtered_detections) - len(detections)
        occurrence_ids: list[str] = []
        if detections and media_type == MediaType.VIDEO:
            artifact_store.save_frame(frame.image_name, frame.bgr_pixels)

        if discarded_low_quality > 0:
            self._emit_log(
                logger,
                log_callback,
                (
                    f"{file_prefix} {self._frame_description(frame)} | "
                    f"faces descartadas por qualidade insuficiente={discarded_low_quality} | "
                    f"limiar={minimum_face_quality:.3f}"
                ),
            )
            event_logger.write(
                "face_rejected_low_quality",
                source_path=frame.source_path,
                frame_index=frame.frame_index,
                frame_timestamp_seconds=frame.timestamp_seconds,
                discarded_count=discarded_low_quality,
                minimum_face_quality=minimum_face_quality,
                sha512=sha512,
            )

        if discarded_small_faces > 0:
            self._emit_log(
                logger,
                log_callback,
                (
                    f"{file_prefix} {self._frame_description(frame)} | "
                    f"faces descartadas por tamanho insuficiente={discarded_small_faces} | "
                    f"minimo={minimum_face_size_pixels}px"
                ),
            )
            event_logger.write(
                "face_rejected_small_size",
                source_path=frame.source_path,
                frame_index=frame.frame_index,
                frame_timestamp_seconds=frame.timestamp_seconds,
                discarded_count=discarded_small_faces,
                minimum_face_size_pixels=minimum_face_size_pixels,
                sha512=sha512,
            )

        if detections:
            self._emit_log(
                logger,
                log_callback,
                (
                    f"{file_prefix} {self._frame_description(frame)} | "
                    f"faces detectadas={len(raw_detections)} | "
                    f"faces selecionadas={len(detections)}"
                ),
            )

        for detection in detections:
            occurrence_id = f"O{len(occurrences) + 1:06d}"
            crop_path = artifact_store.save_crop(occurrence_id, detection.crop_bgr)
            context_image_path = artifact_store.save_context(
                occurrence_id,
                frame.image_name,
                frame.bgr_pixels,
                detection.bbox,
            )
            occurrence = FaceOccurrence(
                occurrence_id=occurrence_id,
                source_path=frame.source_path,
                sha512=sha512,
                media_type=media_type,
                analysis_timestamp_utc=utc_now(),
                frame_index=frame.frame_index,
                frame_timestamp_seconds=frame.timestamp_seconds,
                bbox=detection.bbox,
                detection_score=detection.detection_score,
                embedding=detection.embedding,
                crop_path=crop_path,
                context_image_path=context_image_path,
            )
            occurrences.append(occurrence)
            occurrence_ids.append(occurrence_id)
            event_logger.write(
                "face_detected",
                occurrence_id=occurrence_id,
                source_path=frame.source_path,
                frame_index=frame.frame_index,
                frame_timestamp_seconds=frame.timestamp_seconds,
                detection_score=detection.detection_score,
                bbox=detection.bbox,
                sha512=sha512,
            )
            self._emit_log(
                logger,
                log_callback,
                (
                    f"{file_prefix} Ocorrencia {occurrence_id} registrada | "
                    f"pontuacao={detection.detection_score:.3f} | "
                    f"caixa={detection.bbox.x1:.1f},{detection.bbox.y1:.1f},"
                    f"{detection.bbox.x2:.1f},{detection.bbox.y2:.1f}"
                ),
            )

        return FrameProcessingSummary(
            raw_detection_count=len(raw_detections),
            selected_detection_count=len(detections),
            raw_face_sizes=raw_face_sizes,
            selected_face_sizes=selected_sizes,
            occurrence_ids=tuple(occurrence_ids),
        )

    def _build_summary(
        self,
        file_records: list[FileRecord],
        occurrences: list[FaceOccurrence],
        clusters: list[object],
        total_detected_face_sizes: list[float],
        selected_face_sizes: list[float],
    ) -> ProcessingSummary:
        media_counter = Counter(record.media_type for record in file_records)
        probable_pairs = {
            tuple(sorted((cluster.cluster_id, candidate)))
            for cluster in clusters
            for candidate in cluster.candidate_cluster_ids
        }
        return ProcessingSummary(
            total_files=len(file_records),
            media_files=media_counter[MediaType.IMAGE] + media_counter[MediaType.VIDEO],
            image_files=media_counter[MediaType.IMAGE],
            video_files=media_counter[MediaType.VIDEO],
            total_occurrences=len(occurrences),
            total_clusters=len(clusters),
            probable_match_pairs=len(probable_pairs),
            total_detected_face_sizes=self._calculate_face_size_statistics(total_detected_face_sizes),
            selected_face_sizes=self._calculate_face_size_statistics(selected_face_sizes),
        )

    def _extract_media_info(self, file_path: Path) -> tuple[tuple[object, ...], str | None]:
        if self._media_info_extractor is None:
            return (), "MediaInfo nao configurado."
        return self._media_info_extractor.extract(file_path)

    def _emit_progress(
        self,
        progress_callback: ProgressCallback | None,
        current: int,
        total: int,
        message: str,
    ) -> None:
        if progress_callback is not None:
            progress_callback(current, total, message)

    def _emit_log(
        self,
        logger: logging.Logger,
        log_callback: LogCallback | None,
        message: str,
    ) -> None:
        logger.info(message)
        if log_callback is not None:
            log_callback(message)

    def _configuration_log_lines(self, providers: list[str]) -> list[str]:
        provider_label = ", ".join(providers) if providers else "selecao automatica"
        max_frames = (
            str(self._config.video.max_frames_per_video)
            if self._config.video.max_frames_per_video is not None
            else "sem limite"
        )
        return [
            (
                "[Configuracao] Aplicacao | "
                f"nome={self._config.app.name} | "
                f"saida={self._config.app.output_directory_name} | "
                f"nivel de log={self._config.app.log_level}"
            ),
            (
                "[Configuracao] Midias | "
                f"imagens={', '.join(self._config.media.image_extensions)} | "
                f"videos={', '.join(self._config.media.video_extensions)}"
            ),
            (
                "[Configuracao] Video | "
                f"intervalo de amostragem={self._config.video.sampling_interval_seconds:.2f}s | "
                f"maximo de quadros por video={max_frames}"
            ),
            (
                "[Configuracao] Analise facial | "
                f"mecanismo={self._config.face_model.backend} | "
                f"modelo={self._config.face_model.model_name} | "
                f"tamanho de deteccao={self._detection_size_label()} | "
                f"qualidade minima={self._config.face_model.minimum_face_quality:.3f} | "
                f"tamanho minimo da face={self._config.face_model.minimum_face_size_pixels}px | "
                f"contexto={self._config.face_model.ctx_id} | "
                f"mecanismos de execucao={provider_label}"
            ),
            (
                "[Configuracao] Agrupamento | "
                f"limiar de atribuicao={self._config.clustering.assignment_similarity:.3f} | "
                f"limiar de sugestao={self._config.clustering.candidate_similarity:.3f} | "
                f"tamanho minimo do grupo={self._config.clustering.min_cluster_size}"
            ),
            (
                "[Configuracao] Relatorio | "
                f"faces na galeria por possivel individuo={self._config.reporting.max_gallery_faces_per_group} | "
                f"compilar PDF={'sim' if self._config.reporting.compile_pdf else 'nao'}"
            ),
        ]

    def _media_type_label(self, media_type: MediaType) -> str:
        labels = {
            MediaType.IMAGE: "imagem",
            MediaType.VIDEO: "video",
            MediaType.OTHER: "outro",
        }
        return labels[media_type]

    def _short_hash(self, sha512: str) -> str:
        if len(sha512) <= 24:
            return sha512
        return f"{sha512[:12]}...{sha512[-12:]}"

    def _format_size_bytes(self, size_bytes: int) -> str:
        units = ["B", "KB", "MB", "GB", "TB"]
        size = float(size_bytes)
        for unit in units:
            if size < 1024.0 or unit == units[-1]:
                if unit == "B":
                    return f"{int(size)} {unit}"
                return f"{size:.2f} {unit}"
            size /= 1024.0
        return f"{size_bytes} B"

    def _frame_description(self, frame: SampledFrame) -> str:
        if frame.frame_index is None:
            return "Imagem integral analisada"
        return (
            f"Quadro {frame.frame_index:06d} | "
            f"marca temporal={self._format_seconds(frame.timestamp_seconds)}"
        )

    def _format_seconds(self, value: float | None) -> str:
        if value is None:
            return "-"
        total_seconds = int(value)
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    def _calculate_face_size_statistics(self, face_sizes: list[float]) -> FaceSizeStatistics:
        if not face_sizes:
            return FaceSizeStatistics()
        count = len(face_sizes)
        mean_value = sum(face_sizes) / count
        variance = sum((size - mean_value) ** 2 for size in face_sizes) / count
        return FaceSizeStatistics(
            count=count,
            min_pixels=min(face_sizes),
            max_pixels=max(face_sizes),
            mean_pixels=mean_value,
            stddev_pixels=variance ** 0.5,
        )

    def _face_size_statistics_log_line(self, label: str, statistics: FaceSizeStatistics) -> str:
        if statistics.count == 0:
            return f"[Estatisticas] {label} | quantidade=0"
        return (
            f"[Estatisticas] {label} | "
            f"quantidade={statistics.count} | "
            f"minimo={statistics.min_pixels:.1f}px | "
            f"maximo={statistics.max_pixels:.1f}px | "
            f"media={statistics.mean_pixels:.1f}px | "
            f"desvio padrao={statistics.stddev_pixels:.1f}px"
        )

    def _detection_size_label(self) -> str:
        if self._config.face_model.det_size is None:
            return "resolucao original"
        return f"{self._config.face_model.det_size[0]}x{self._config.face_model.det_size[1]}"
