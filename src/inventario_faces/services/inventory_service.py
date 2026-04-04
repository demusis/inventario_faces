from __future__ import annotations

import logging
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Callable

from inventario_faces.domain.config import AppConfig
from inventario_faces.domain.entities import (
    FaceSearchMatch,
    FaceSearchQuery,
    FaceSearchResult,
    FaceSearchSummary,
    FaceOccurrence,
    FaceSizeStatistics,
    FaceTrack,
    FileRecord,
    InventoryResult,
    KeyFrame,
    MediaType,
    ProcessingSummary,
    ReportArtifacts,
)
from inventario_faces.domain.protocols import (
    FaceAnalyzer,
    FaceSearchReportGenerator,
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
from inventario_faces.services.enhancement_service import EnhancementService
from inventario_faces.services.export_service import ExportService
from inventario_faces.services.hashing_service import HashingService
from inventario_faces.services.quality_service import FaceQualityService
from inventario_faces.services.scanner_service import ScannerService
from inventario_faces.services.search_service import SearchIndexService
from inventario_faces.services.tracking_service import FaceTrackingService, TrackingResult
from inventario_faces.services.video_service import VideoService
from inventario_faces.utils.path_utils import ensure_directory
from inventario_faces.utils.time_utils import as_utc, utc_now


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
        tracking_service: FaceTrackingService | None = None,
        search_service: SearchIndexService | None = None,
        face_search_report_generator: FaceSearchReportGenerator | None = None,
    ) -> None:
        self._config = config
        self._scanner_service = scanner_service
        self._hashing_service = hashing_service
        self._media_service = media_service
        self._clustering_service = clustering_service
        self._report_generator = report_generator
        self._face_analyzer_factory = face_analyzer_factory
        self._media_info_extractor = media_info_extractor
        self._tracking_service = tracking_service or FaceTrackingService(
            config=config,
            enhancement_service=EnhancementService(config.enhancement),
            quality_service=FaceQualityService(),
        )
        self._search_service = search_service or SearchIndexService(config.search)
        self._face_search_report_generator = face_search_report_generator

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
            self._emit_log(text_logger, log_callback, "Pipeline orientado a tracks ativado.")
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
                configuration=self._config,
                providers=providers,
            )

            file_records: list[FileRecord] = []
            occurrences: list[FaceOccurrence] = []
            tracks: list[FaceTrack] = []
            keyframes: list[KeyFrame] = []
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

                discovered_at_utc = utc_now()
                sha512 = self._hashing_service.sha512(file_path)
                stat = file_path.stat()
                modified_at_utc = as_utc(stat.st_mtime)
                processing_error: str | None = None
                media_info_tracks = ()
                media_info_error: str | None = None

                if media_type in {MediaType.IMAGE, MediaType.VIDEO}:
                    media_info_tracks, media_info_error = self._extract_media_info(file_path)

                try:
                    tracking_result: TrackingResult | None = None
                    if media_type == MediaType.IMAGE:
                        tracking_result = self._tracking_service.process_media(
                            source_path=file_path,
                            sha512=sha512,
                            media_type=media_type,
                            frames=[self._media_service.load_image(file_path)],
                            analyzer=analyzer,
                            artifact_store=artifact_store,
                            id_namespace=f"{index:04d}",
                            event_callback=lambda event, fields: event_logger.write(event, **fields),
                            text_callback=lambda message: self._emit_log(text_logger, log_callback, message),
                        )
                    elif media_type == MediaType.VIDEO:
                        tracking_result = self._tracking_service.process_media(
                            source_path=file_path,
                            sha512=sha512,
                            media_type=media_type,
                            frames=self._media_service.sample_video(file_path),
                            analyzer=analyzer,
                            artifact_store=artifact_store,
                            id_namespace=f"{index:04d}",
                            event_callback=lambda event, fields: event_logger.write(event, **fields),
                            text_callback=lambda message: self._emit_log(text_logger, log_callback, message),
                        )

                    if tracking_result is not None:
                        occurrences.extend(tracking_result.occurrences)
                        tracks.extend(tracking_result.tracks)
                        keyframes.extend(tracking_result.keyframes)
                        total_detected_face_sizes.extend(tracking_result.raw_face_sizes)
                        selected_face_sizes.extend(tracking_result.selected_face_sizes)
                        self._emit_log(
                            text_logger,
                            log_callback,
                            (
                                f"{file_prefix} Midia analisada | "
                                f"deteccoes={tracking_result.raw_detection_count} | "
                                f"selecionadas={tracking_result.selected_detection_count} | "
                                f"tracks={len(tracking_result.tracks)} | "
                                f"keyframes={len(tracking_result.keyframes)} | "
                                f"embeddings_calculados={tracking_result.embedded_detection_count}"
                            ),
                        )
                    elif media_type == MediaType.OTHER:
                        self._emit_log(
                            text_logger,
                            log_callback,
                            f"{file_prefix} Arquivo fora do escopo da analise facial.",
                        )

                    event_logger.write(
                        "file_processed",
                        path=file_path,
                        media_type=media_type,
                        sha512=sha512,
                        size_bytes=stat.st_size,
                        media_info_tracks=media_info_tracks,
                        media_info_error=media_info_error,
                    )
                except Exception as exc:
                    processing_error = str(exc)
                    text_logger.exception("Falha ao processar %s", file_path)
                    self._emit_log(
                        text_logger,
                        log_callback,
                        f"{file_prefix} Erro de processamento: {processing_error}",
                    )
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
                f"[Agrupamento] Consolidando {len(tracks)} tracks em possiveis grupos.",
            )
            clusters = self._clustering_service.cluster(tracks)
            self._propagate_cluster_membership(occurrences, tracks)
            search_artifacts = self._search_service.build(run_directory, tracks, clusters)

            finished_at_utc = utc_now()
            summary = self._build_summary(
                file_records=file_records,
                occurrences=occurrences,
                tracks=tracks,
                keyframes=keyframes,
                clusters=clusters,
                total_detected_face_sizes=total_detected_face_sizes,
                selected_face_sizes=selected_face_sizes,
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
                tracks=tracks,
                keyframes=keyframes,
                search=search_artifacts,
            )

            files_csv_path = export_service.write_files_csv(file_records)
            occurrences_csv_path = export_service.write_occurrences_csv(occurrences)
            tracks_csv_path = export_service.write_tracks_csv(tracks)
            keyframes_csv_path = export_service.write_keyframes_csv(keyframes)
            clusters_json_path = export_service.write_clusters_json(clusters)
            media_info_json_path = export_service.write_media_info_json(file_records)
            search_json_path = export_service.write_search_json(search_artifacts)
            self._emit_log(
                text_logger,
                log_callback,
                (
                    f"[Exportacao] Inventario atualizado | arquivos={files_csv_path.name} | "
                    f"ocorrencias={occurrences_csv_path.name} | tracks={tracks_csv_path.name} | "
                    f"keyframes={keyframes_csv_path.name} | grupos={clusters_json_path.name} | "
                    f"metadados={media_info_json_path.name} | busca={search_json_path.name}"
                ),
            )

            report_artifacts = self._report_generator.generate(preliminary_result)
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
                tracks=tracks,
                keyframes=keyframes,
                search=search_artifacts,
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
                search=search_artifacts,
            )
            return result
        finally:
            close_file_logger(text_logger)

    def run_face_search(
        self,
        root_directory: Path,
        query_image_path: Path,
        progress_callback: ProgressCallback | None = None,
        log_callback: LogCallback | None = None,
    ) -> FaceSearchResult:
        query_path = Path(query_image_path).resolve()
        if not query_path.exists():
            raise FileNotFoundError(f"Imagem de consulta nao encontrada: {query_path}")

        def inventory_progress(current: int, total: int, message: str) -> None:
            if progress_callback is None:
                return
            scaled = 0 if total == 0 else int((current / total) * 85)
            progress_callback(scaled, 100, message)

        inventory_result = self.run(
            root_directory,
            progress_callback=inventory_progress,
            log_callback=log_callback,
        )

        if progress_callback is not None:
            progress_callback(88, 100, "Analisando imagem de consulta")

        logger = build_file_logger(inventory_result.logs_directory, self._config.app.log_level)
        event_logger = StructuredEventLogger(inventory_result.logs_directory / "events.jsonl")
        export_service = ExportService(inventory_result.run_directory)
        try:
            self._emit_log(logger, log_callback, f"[Busca por face] Imagem de consulta: {query_path}")
            analyzer = self._face_analyzer_factory()
            query_sha512 = self._hashing_service.sha512(query_path)
            query_tracking = self._tracking_service.process_media(
                source_path=query_path,
                sha512=query_sha512,
                media_type=MediaType.IMAGE,
                frames=[self._media_service.load_image(query_path)],
                analyzer=analyzer,
                artifact_store=ArtifactStore(inventory_result.run_directory / "face_search_query"),
                id_namespace="Q",
                event_callback=lambda event, fields: event_logger.write(event, **fields),
                text_callback=lambda message: self._emit_log(logger, log_callback, message),
            )
            query_track, query_occurrence = self._select_query_face(query_tracking)
            query_keyframe = next(
                (item for item in query_tracking.keyframes if item.track_id == query_track.track_id),
                None,
            )
            event_logger.write(
                "face_search_query_selected",
                query_path=query_path,
                query_track_id=query_track.track_id,
                query_occurrence_id=query_occurrence.occurrence_id,
                query_keyframe_id=query_keyframe.keyframe_id if query_keyframe is not None else None,
                detected_faces=len(query_tracking.tracks),
            )

            if progress_callback is not None:
                progress_callback(94, 100, "Executando busca vetorial")

            raw_hits = self._search_service.search(
                query_track.average_embedding,
                inventory_result.tracks,
                inventory_result.clusters,
                inventory_result.occurrences,
            )
            matches = self._resolve_face_search_matches(inventory_result, raw_hits)
            summary = FaceSearchSummary(
                query_faces_detected=len(query_tracking.tracks),
                compatible_clusters=len({item.cluster_id for item in matches if item.cluster_id is not None}),
                compatible_tracks=len(matches),
                compatible_occurrences=len([item for item in matches if item.occurrence_id is not None]),
                compatibility_threshold=self._config.clustering.candidate_similarity,
            )

            if self._face_search_report_generator is None:
                raise RuntimeError("Gerador de relatório de busca por face nao configurado.")

            preliminary = FaceSearchResult(
                inventory_result=inventory_result,
                query=FaceSearchQuery(
                    source_path=query_path,
                    sha512=query_sha512,
                    detected_face_count=len(query_tracking.tracks),
                    selected_track_id=query_track.track_id,
                    selected_occurrence_id=query_occurrence.occurrence_id,
                    selected_keyframe_id=query_keyframe.keyframe_id if query_keyframe is not None else None,
                    crop_path=query_occurrence.crop_path,
                    context_image_path=query_occurrence.context_image_path,
                    quality_score=(
                        query_occurrence.quality_metrics.score
                        if query_occurrence.quality_metrics is not None
                        else None
                    ),
                ),
                matches=matches,
                summary=summary,
                report=ReportArtifacts(
                    tex_path=inventory_result.run_directory / "report" / "relatorio_busca_por_face.tex",
                    pdf_path=None,
                    docx_path=inventory_result.run_directory / "report" / "relatorio_busca_por_face.docx",
                ),
                export_path=export_service.inventory_directory / "face_search.json",
            )

            if progress_callback is not None:
                progress_callback(97, 100, "Gerando relatórios da busca")

            report_artifacts = self._face_search_report_generator.generate(preliminary)
            final_result = replace(preliminary, report=report_artifacts)
            export_service.write_face_search_json(final_result)
            self._emit_log(
                logger,
                log_callback,
                (
                    f"[Busca por face] Compatibilidades encontradas | grupos={summary.compatible_clusters} | "
                    f"tracks={summary.compatible_tracks} | ocorrencias={summary.compatible_occurrences}"
                ),
            )
            event_logger.write(
                "face_search_finished",
                query_path=query_path,
                compatible_clusters=summary.compatible_clusters,
                compatible_tracks=summary.compatible_tracks,
                compatible_occurrences=summary.compatible_occurrences,
                report_pdf=report_artifacts.pdf_path,
                report_tex=report_artifacts.tex_path,
                report_docx=report_artifacts.docx_path,
                export_path=final_result.export_path,
            )
            if progress_callback is not None:
                progress_callback(100, 100, "Busca por face concluída")
            return final_result
        finally:
            close_file_logger(logger)

    def _propagate_cluster_membership(
        self,
        occurrences: list[FaceOccurrence],
        tracks: list[FaceTrack],
    ) -> None:
        track_map = {track.track_id: track for track in tracks}
        for occurrence in occurrences:
            if occurrence.track_id is None:
                continue
            track = track_map.get(occurrence.track_id)
            if track is None:
                continue
            occurrence.cluster_id = track.cluster_id
            occurrence.suggested_cluster_ids = list(track.candidate_cluster_ids)

    def _build_summary(
        self,
        file_records: list[FileRecord],
        occurrences: list[FaceOccurrence],
        tracks: list[FaceTrack],
        keyframes: list[KeyFrame],
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
            total_tracks=len(tracks),
            total_keyframes=len(keyframes),
            total_detected_face_sizes=self._calculate_face_size_statistics(total_detected_face_sizes),
            selected_face_sizes=self._calculate_face_size_statistics(selected_face_sizes),
        )

    def _select_query_face(self, tracking_result: TrackingResult) -> tuple[FaceTrack, FaceOccurrence]:
        if not tracking_result.tracks:
            raise ValueError("Nenhuma face elegivel foi encontrada na imagem de consulta.")
        occurrence_map = {
            occurrence.track_id: occurrence
            for occurrence in tracking_result.occurrences
            if occurrence.track_id is not None
        }
        ranked_tracks = sorted(
            tracking_result.tracks,
            key=lambda track: (
                track.quality_statistics.best_quality_score,
                track.quality_statistics.mean_detection_score,
                len(track.occurrence_ids),
            ),
            reverse=True,
        )
        for track in ranked_tracks:
            if not track.average_embedding:
                continue
            occurrence = occurrence_map.get(track.track_id)
            if occurrence is not None:
                return track, occurrence
        raise ValueError("A face de consulta nao gerou embedding utilizavel para a busca.")

    def _resolve_face_search_matches(
        self,
        result: InventoryResult,
        raw_hits: dict[str, list[object]],
    ) -> list[FaceSearchMatch]:
        compatibility_threshold = self._config.clustering.candidate_similarity
        track_hits = [
            hit for hit in raw_hits.get("tracks", [])
            if getattr(hit, "score", -1.0) >= compatibility_threshold
        ]
        if not track_hits:
            return []

        cluster_scores = {
            hit.entity_id: hit.score
            for hit in raw_hits.get("clusters", [])
            if getattr(hit, "score", -1.0) >= compatibility_threshold
        }
        occurrence_hits = {
            hit.entity_id: hit
            for hit in raw_hits.get("occurrences", [])
            if getattr(hit, "score", -1.0) >= compatibility_threshold
        }

        tracks_by_id = {track.track_id: track for track in result.tracks}
        occurrences_by_id = {occurrence.occurrence_id: occurrence for occurrence in result.occurrences}
        keyframes_by_track: dict[str, list[KeyFrame]] = {}
        for keyframe in result.keyframes:
            keyframes_by_track.setdefault(keyframe.track_id, []).append(keyframe)

        resolved: list[FaceSearchMatch] = []
        for rank, hit in enumerate(track_hits, start=1):
            track = tracks_by_id.get(hit.entity_id)
            if track is None:
                continue
            occurrence = self._best_match_occurrence(track, occurrences_by_id, occurrence_hits)
            keyframe = self._representative_keyframe(track, keyframes_by_track)
            crop_path = (
                occurrence.crop_path
                if occurrence is not None and occurrence.crop_path is not None
                else keyframe.preview_path if keyframe is not None else track.preview_path
            )
            context_path = (
                occurrence.context_image_path
                if occurrence is not None and occurrence.context_image_path is not None
                else keyframe.context_image_path if keyframe is not None else None
            )
            occurrence_hit = occurrence_hits.get(occurrence.occurrence_id) if occurrence is not None else None
            resolved.append(
                FaceSearchMatch(
                    rank=rank,
                    cluster_id=track.cluster_id,
                    track_id=track.track_id,
                    occurrence_id=occurrence.occurrence_id if occurrence is not None else None,
                    cluster_score=cluster_scores.get(track.cluster_id or ""),
                    track_score=hit.score,
                    occurrence_score=occurrence_hit.score if occurrence_hit is not None else None,
                    source_path=track.source_path,
                    frame_index=(
                        occurrence.frame_index
                        if occurrence is not None
                        else keyframe.frame_index if keyframe is not None else None
                    ),
                    timestamp_seconds=(
                        occurrence.frame_timestamp_seconds
                        if occurrence is not None
                        else keyframe.timestamp_seconds if keyframe is not None else None
                    ),
                    track_start_time=track.start_time,
                    track_end_time=track.end_time,
                    crop_path=crop_path,
                    context_image_path=context_path,
                )
            )
        return resolved

    def _best_match_occurrence(
        self,
        track: FaceTrack,
        occurrences_by_id: dict[str, FaceOccurrence],
        occurrence_hits: dict[str, object],
    ) -> FaceOccurrence | None:
        ranked_hits = sorted(
            (
                occurrence_hits[occurrence_id]
                for occurrence_id in track.occurrence_ids
                if occurrence_id in occurrence_hits
            ),
            key=lambda item: item.score,
            reverse=True,
        )
        if ranked_hits:
            return occurrences_by_id.get(ranked_hits[0].entity_id)
        if track.best_occurrence_id is not None:
            return occurrences_by_id.get(track.best_occurrence_id)
        for occurrence_id in track.occurrence_ids:
            occurrence = occurrences_by_id.get(occurrence_id)
            if occurrence is not None:
                return occurrence
        return None

    def _representative_keyframe(
        self,
        track: FaceTrack,
        keyframes_by_track: dict[str, list[KeyFrame]],
    ) -> KeyFrame | None:
        keyframes = keyframes_by_track.get(track.track_id, [])
        if not keyframes:
            return None
        for keyframe in keyframes:
            if track.best_occurrence_id is not None and keyframe.occurrence_id == track.best_occurrence_id:
                return keyframe
        return keyframes[0]

    def _extract_media_info(self, file_path: Path) -> tuple[tuple[object, ...], str | None]:
        if self._media_info_extractor is None:
            return (), "Extrator interno de metadados nao configurado."
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
        return [
            (
                "[Configuracao] Video | "
                f"amostragem={self._config.video.sampling_interval_seconds:.2f}s | "
                f"intervalo de keyframe={self._config.video.keyframe_interval_seconds:.2f}s | "
                f"mudanca significativa={self._config.video.significant_change_threshold:.2f}"
            ),
            (
                "[Configuracao] Tracking | "
                f"iou={self._config.tracking.iou_threshold:.2f} | "
                f"distancia={self._config.tracking.spatial_distance_threshold:.2f} | "
                f"embedding={self._config.tracking.embedding_similarity_threshold:.2f} | "
                f"perda maxima={self._config.tracking.max_missed_detections}"
            ),
            (
                "[Configuracao] Analise facial | "
                f"backend={self._config.face_model.backend} | "
                f"modelo={self._config.face_model.model_name} | "
                f"qualidade minima={self._config.face_model.minimum_face_quality:.2f} | "
                f"face minima={self._config.face_model.minimum_face_size_pixels}px | "
                f"provedores={provider_label}"
            ),
            (
                "[Configuracao] Clustering | "
                f"atribuicao={self._config.clustering.assignment_similarity:.2f} | "
                f"sugestao={self._config.clustering.candidate_similarity:.2f} | "
                f"grupo minimo={self._config.clustering.min_cluster_size}"
            ),
            (
                "[Configuracao] Busca | "
                f"habilitada={'sim' if self._config.search.enabled else 'nao'} | "
                f"preferir_faiss={'sim' if self._config.search.prefer_faiss else 'nao'} | "
                f"coarse={self._config.search.coarse_top_k} | "
                f"refino={self._config.search.refine_top_k}"
            ),
        ]

    def _media_type_label(self, media_type: MediaType) -> str:
        labels = {
            MediaType.IMAGE: "imagem",
            MediaType.VIDEO: "video",
            MediaType.OTHER: "outro",
        }
        return labels[media_type]

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
