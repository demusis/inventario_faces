from __future__ import annotations

import logging
import time
from collections.abc import Iterable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from inventario_faces.domain.config import AppConfig
from inventario_faces.domain.entities import (
    FaceOccurrence,
    FaceSearchMatch,
    FaceSearchQuery,
    FaceSearchQueryEvent,
    FaceSearchResult,
    FaceSearchSummary,
    FaceTrack,
    InventoryResult,
    KeyFrame,
    MediaType,
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
    summarize_exception,
)
from inventario_faces.services.clustering_service import ClusteringService
from inventario_faces.services.export_service import ExportService
from inventario_faces.services.hashing_service import HashingService
from inventario_faces.services.pipeline_support import (
    cleanup_processing_input,
    emit_exception,
    emit_log,
    emit_progress,
    frames_with_original_source,
    prepare_processing_input,
)
from inventario_faces.services.scanner_service import ScannerService
from inventario_faces.services.search_service import SearchIndexService
from inventario_faces.services.tracking_service import FaceTrackingService, TrackingResult
from inventario_faces.services.video_service import VideoService

if TYPE_CHECKING:
    from inventario_faces.services.inventory_service import InventoryService


@dataclass(frozen=True)
class PreparedFaceSearchQuery:
    query: FaceSearchQuery
    embedding: list[float]


class FaceSearchService:
    """Executa a busca por faces de consulta contra as tracks indexadas do inventário."""

    def __init__(
        self,
        config: AppConfig,
        scanner_service: ScannerService,
        hashing_service: HashingService,
        media_service: VideoService,
        clustering_service: ClusteringService,
        tracking_service: FaceTrackingService,
        search_service: SearchIndexService,
        face_analyzer_factory: Callable[[], FaceAnalyzer],
        report_generator: ReportGenerator,
        face_search_report_generator: FaceSearchReportGenerator | None = None,
        media_info_extractor: MediaInfoExtractor | None = None,
    ) -> None:
        self._config = config
        self._scanner_service = scanner_service
        self._hashing_service = hashing_service
        self._media_service = media_service
        self._clustering_service = clustering_service
        self._tracking_service = tracking_service
        self._search_service = search_service
        self._face_analyzer_factory = face_analyzer_factory
        self._report_generator = report_generator
        self._face_search_report_generator = face_search_report_generator
        self._media_info_extractor = media_info_extractor

    def _build_inventory_service(self, config: AppConfig) -> "InventoryService":
        from inventario_faces.services.inventory_service import InventoryService

        return InventoryService(
            config=config,
            scanner_service=self._scanner_service,
            hashing_service=self._hashing_service,
            media_service=self._media_service,
            clustering_service=self._clustering_service,
            report_generator=self._report_generator,
            face_analyzer_factory=self._face_analyzer_factory,
            media_info_extractor=self._media_info_extractor,
            tracking_service=self._tracking_service,
            search_service=self._search_service,
            face_search_report_generator=self._face_search_report_generator,
        )

    def run_face_search(
        self,
        root_directory: Path,
        query_image_paths: Path | Iterable[Path],
        work_directory: Path | None = None,
        progress_callback: ProgressCallback | None = None,
        log_callback: LogCallback | None = None,
    ) -> FaceSearchResult:
        """Processa o acervo e pesquisa uma ou mais faces de consulta contra as tracks indexadas."""

        normalized_query_paths = self._normalize_face_search_query_paths(query_image_paths)

        def inventory_progress(current: int, total: int, message: str) -> None:
            if progress_callback is None:
                return
            scaled = 0 if total == 0 else int((current / total) * 85)
            progress_callback(scaled, 100, message)

        if self._config.distributed.enabled:
            local_service = self._build_inventory_service(
                replace(
                    self._config,
                    distributed=replace(self._config.distributed, enabled=False),
                )
            )
            inventory_result = local_service.run(
                root_directory,
                work_directory=work_directory,
                progress_callback=inventory_progress,
                log_callback=log_callback,
            )
        else:
            inventory_result = self._build_inventory_service(self._config).run(
                root_directory,
                work_directory=work_directory,
                progress_callback=inventory_progress,
                log_callback=log_callback,
            )

        if progress_callback is not None:
            progress_callback(88, 100, "Preparando consultas faciais")

        logger = build_file_logger(inventory_result.logs_directory, self._config.app.log_level)
        event_logger = StructuredEventLogger(inventory_result.logs_directory / "events.jsonl")
        export_service = ExportService(inventory_result.run_directory)
        try:
            emit_log(
                logger,
                log_callback,
                f"[Logs] Texto={inventory_result.logs_directory / 'run.log'} | eventos={inventory_result.logs_directory / 'events.jsonl'}",
            )
            emit_log(
                logger,
                log_callback,
                f"[Busca por faces] Imagens de consulta informadas: {len(normalized_query_paths)}",
            )
            for index, query_path in enumerate(normalized_query_paths, start=1):
                emit_log(
                    logger,
                    log_callback,
                    f"[Busca por faces] Consulta {index}/{len(normalized_query_paths)}: {query_path}",
                )
            emit_log(
                logger,
                log_callback,
                f"[Backend facial] Inicializando modelo {self._config.face_model.model_name} para busca por faces.",
            )
            analyzer_started_at = time.perf_counter()
            analyzer = self._face_analyzer_factory()
            analyzer_elapsed = time.perf_counter() - analyzer_started_at
            emit_log(
                logger,
                log_callback,
                (
                    f"[Backend facial] Modelo pronto | "
                    f"providers={', '.join(getattr(analyzer, 'providers', [])) or 'desconhecido'} | "
                    f"tempo={analyzer_elapsed:.2f}s"
                ),
            )
            prepared_queries, detected_face_total, rejected_queries, query_events = self._prepare_face_search_queries(
                query_paths=normalized_query_paths,
                inventory_result=inventory_result,
                analyzer=analyzer,
                logger=logger,
                log_callback=log_callback,
                event_logger=event_logger,
                progress_callback=progress_callback,
            )

            if progress_callback is not None:
                progress_callback(
                    94,
                    100,
                    "Executando busca vetorial das consultas"
                    if prepared_queries
                    else "Nenhuma consulta válida; gerando relatório auditável",
                )

            query_hit_sets = (
                [
                    (
                        prepared.query,
                        self._search_service.search(
                            prepared.embedding,
                            inventory_result.tracks,
                            inventory_result.clusters,
                            inventory_result.occurrences,
                        ),
                    )
                    for prepared in prepared_queries
                ]
                if prepared_queries
                else []
            )
            matches = self._resolve_face_search_matches(inventory_result, query_hit_sets)
            summary = FaceSearchSummary(
                query_faces_detected=detected_face_total,
                compatible_clusters=len({item.cluster_id for item in matches if item.cluster_id is not None}),
                compatible_tracks=len(matches),
                compatible_occurrences=len([item for item in matches if item.occurrence_id is not None]),
                compatibility_threshold=self._config.clustering.candidate_similarity,
                query_image_count=len(normalized_query_paths),
                query_faces_selected=len(prepared_queries),
                query_images_rejected=rejected_queries,
            )

            if self._face_search_report_generator is None:
                raise RuntimeError("Gerador de relatório de busca por faces nao configurado.")

            preliminary = FaceSearchResult(
                inventory_result=inventory_result,
                query=prepared_queries[0].query if prepared_queries else None,
                matches=matches,
                summary=summary,
                report=ReportArtifacts(
                    tex_path=inventory_result.run_directory / "report" / "relatorio_busca_por_face.tex",
                    pdf_path=None,
                    docx_path=inventory_result.run_directory / "report" / "relatorio_busca_por_face.docx",
                ),
                export_path=export_service.inventory_directory / "face_search.json",
                queries=[prepared.query for prepared in prepared_queries],
                query_events=query_events,
            )

            if progress_callback is not None:
                progress_callback(97, 100, "Gerando relatórios da busca por faces")

            report_artifacts = self._face_search_report_generator.generate(preliminary)
            final_result = replace(preliminary, report=report_artifacts)
            export_service.write_face_search_json(final_result)
            emit_log(
                logger,
                log_callback,
                (
                    f"[Busca por faces] Consultas válidas={len(prepared_queries)} | "
                    f"consultas descartadas={rejected_queries} | faces_detectadas={detected_face_total}"
                ),
            )
            emit_log(
                logger,
                log_callback,
                (
                    f"[Busca por faces] Compatibilidades encontradas | grupos={summary.compatible_clusters} | "
                    f"tracks={summary.compatible_tracks} | ocorrencias={summary.compatible_occurrences}"
                ),
            )
            if not prepared_queries:
                emit_log(
                    logger,
                    log_callback,
                    "[Busca por faces] Nenhuma consulta válida foi selecionada; a busca vetorial foi pulada e o relatório auditável foi gerado com os eventos das consultas.",
                )
            event_logger.write(
                "face_search_finished",
                query_count=len(normalized_query_paths),
                selected_queries=len(prepared_queries),
                rejected_queries=rejected_queries,
                compatible_clusters=summary.compatible_clusters,
                compatible_tracks=summary.compatible_tracks,
                compatible_occurrences=summary.compatible_occurrences,
                report_pdf=report_artifacts.pdf_path,
                report_tex=report_artifacts.tex_path,
                report_docx=report_artifacts.docx_path,
                export_path=final_result.export_path,
            )
            if progress_callback is not None:
                progress_callback(100, 100, "Busca por faces concluída")
            return final_result
        except Exception as exc:
            error_summary, traceback_text = emit_exception(
                logger,
                log_callback,
                "[Busca por faces] Falha fatal",
                exc,
                include_traceback_in_callback=True,
            )
            event_logger.write(
                "face_search_failed",
                query_count=len(normalized_query_paths),
                error=error_summary,
                error_type=type(exc).__name__,
                traceback=traceback_text,
            )
            raise
        finally:
            close_file_logger(logger)

    def _normalize_face_search_query_paths(
        self,
        query_image_paths: Path | Iterable[Path],
    ) -> list[Path]:
        if isinstance(query_image_paths, Path):
            raw_paths = [query_image_paths]
        else:
            raw_paths = [Path(path) for path in query_image_paths]

        if not raw_paths:
            raise ValueError("Selecione ao menos uma imagem de consulta para a busca por faces.")

        normalized: list[Path] = []
        seen_paths: set[Path] = set()
        for raw_path in raw_paths:
            resolved = Path(raw_path).expanduser().resolve()
            if resolved in seen_paths:
                continue
            seen_paths.add(resolved)
            normalized.append(resolved)
        return normalized

    def _prepare_face_search_queries(
        self,
        *,
        query_paths: list[Path],
        inventory_result: InventoryResult,
        analyzer: FaceAnalyzer,
        logger: logging.Logger,
        log_callback: LogCallback | None,
        event_logger: StructuredEventLogger,
        progress_callback: ProgressCallback | None,
    ) -> tuple[list[PreparedFaceSearchQuery], int, int, list[FaceSearchQueryEvent]]:
        prepared_queries: list[PreparedFaceSearchQuery] = []
        query_events: list[FaceSearchQueryEvent] = []
        detected_face_total = 0
        rejected_queries = 0
        total_queries = len(query_paths)

        for index, query_path in enumerate(query_paths, start=1):
            progress_value = 88 + int(((index - 1) / max(1, total_queries)) * 6)
            emit_progress(
                progress_callback,
                progress_value,
                100,
                f"Analisando consultas faciais ({index}/{total_queries})",
            )
            processing_query_path: Path | None = None
            query_cleanup_path: Path | None = None
            query_sha512: str | None = None
            detected_faces_in_query: int | None = None
            try:
                processing_query_path, query_sha512, query_cleanup_path = prepare_processing_input(
                    config=self._config,
                    hashing_service=self._hashing_service,
                    file_path=query_path,
                    media_type=MediaType.IMAGE,
                    file_prefix=f"[Busca por faces] Consulta {index}/{total_queries}",
                    text_logger=logger,
                    log_callback=log_callback,
                )
                query_tracking = self._tracking_service.process_media(
                    source_path=query_path,
                    sha512=query_sha512,
                    media_type=MediaType.IMAGE,
                    frames=frames_with_original_source(
                        [self._media_service.load_image(processing_query_path)],
                        query_path,
                    ),
                    analyzer=analyzer,
                    artifact_store=ArtifactStore(
                        inventory_result.run_directory / "face_search_queries" / f"query_{index:04d}"
                    ),
                    id_namespace=f"Q{index:03d}",
                    event_callback=lambda event, fields: event_logger.write(event, **fields),
                    text_callback=lambda message: emit_log(logger, log_callback, message),
                )
                detected_faces_in_query = len(query_tracking.tracks)
                detected_face_total += detected_faces_in_query
                query_track, query_occurrence = self._select_query_face(query_tracking)
                query_keyframe = next(
                    (item for item in query_tracking.keyframes if item.track_id == query_track.track_id),
                    None,
                )
                query = FaceSearchQuery(
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
                    query_index=index,
                )
                prepared_queries.append(
                    PreparedFaceSearchQuery(
                        query=query,
                        embedding=list(query_track.average_embedding),
                    )
                )
                query_events.append(
                    FaceSearchQueryEvent(
                        query_index=index,
                        source_path=query_path,
                        status="selected",
                        sha512=query_sha512,
                        detected_face_count=detected_faces_in_query,
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
                    )
                )
                event_logger.write(
                    "face_search_query_selected",
                    query_index=index,
                    query_path=query_path,
                    query_track_id=query_track.track_id,
                    query_occurrence_id=query_occurrence.occurrence_id,
                    query_keyframe_id=query_keyframe.keyframe_id if query_keyframe is not None else None,
                    detected_faces=len(query_tracking.tracks),
                )
                emit_log(
                    logger,
                    log_callback,
                    (
                        f"[Busca por faces] Consulta {index}/{total_queries} selecionada | "
                        f"track={query_track.track_id} | faces_elegiveis={detected_faces_in_query}"
                    ),
                )
            except Exception as exc:
                rejected_queries += 1
                reason = summarize_exception(exc)
                query_events.append(
                    FaceSearchQueryEvent(
                        query_index=index,
                        source_path=query_path,
                        status="rejected",
                        sha512=query_sha512,
                        detected_face_count=detected_faces_in_query,
                        error_message=reason,
                        error_type=type(exc).__name__,
                    )
                )
                emit_log(
                    logger,
                    log_callback,
                    f"[Busca por faces] Consulta {index}/{total_queries} descartada: {reason}",
                )
                event_logger.write(
                    "face_search_query_rejected",
                    query_index=index,
                    query_path=query_path,
                    error=reason,
                    error_type=type(exc).__name__,
                )
            finally:
                if query_cleanup_path is not None:
                    cleanup_processing_input(query_cleanup_path)

        return prepared_queries, detected_face_total, rejected_queries, query_events

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
        query_hit_sets: list[tuple[FaceSearchQuery, dict[str, list[object]]]],
    ) -> list[FaceSearchMatch]:
        compatibility_threshold = self._config.clustering.candidate_similarity
        best_track_hits: dict[str, tuple[float, FaceSearchQuery, dict[str, list[object]]]] = {}
        for query, raw_hits in query_hit_sets:
            for hit in raw_hits.get("tracks", []):
                score = float(getattr(hit, "score", -1.0))
                if score < compatibility_threshold:
                    continue
                current = best_track_hits.get(hit.entity_id)
                if current is None or score > current[0]:
                    best_track_hits[hit.entity_id] = (score, query, raw_hits)

        if not best_track_hits:
            return []

        ranked_track_hits = sorted(
            best_track_hits.items(),
            key=lambda item: item[1][0],
            reverse=True,
        )[: self._config.search.refine_top_k]

        tracks_by_id = {track.track_id: track for track in result.tracks}
        occurrences_by_id = {occurrence.occurrence_id: occurrence for occurrence in result.occurrences}
        keyframes_by_track: dict[str, list[KeyFrame]] = {}
        for keyframe in result.keyframes:
            keyframes_by_track.setdefault(keyframe.track_id, []).append(keyframe)

        resolved: list[FaceSearchMatch] = []
        for rank, (track_id, (track_score, query, raw_hits)) in enumerate(ranked_track_hits, start=1):
            track = tracks_by_id.get(track_id)
            if track is None:
                continue
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
                    track_score=track_score,
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
                    query_source_path=query.source_path,
                    query_selected_track_id=query.selected_track_id,
                    query_selected_occurrence_id=query.selected_occurrence_id,
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
