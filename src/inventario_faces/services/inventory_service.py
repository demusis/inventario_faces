from __future__ import annotations

import json
import logging
import shutil
import tempfile
from collections import Counter
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from inventario_faces.domain.config import AppConfig
from inventario_faces.domain.entities import (
    BoundingBox,
    EnhancementMetadata,
    FaceSearchMatch,
    FaceSearchQuery,
    FaceSearchResult,
    FaceSearchSummary,
    FaceOccurrence,
    FaceQualityMetrics,
    FaceSizeStatistics,
    FaceTrack,
    FileRecord,
    InventoryResult,
    KeyFrame,
    MediaInfoAttribute,
    MediaInfoTrack,
    MediaType,
    ProcessingSummary,
    ReportArtifacts,
    SearchArtifacts,
    TrackQualityStatistics,
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
from inventario_faces.infrastructure.distributed_coordination import (
    DistributedClaim,
    DistributedCoordinator,
    DistributedExecutionSnapshot,
    DistributedHealthSnapshot,
    DistributedNodeHeartbeat,
    DistributedPartialValidation,
    DistributedPlanEntry,
)
from inventario_faces.infrastructure.logging_setup import (
    StructuredEventLogger,
    build_file_logger,
    close_file_logger,
    format_exception_traceback,
    summarize_exception,
)
from inventario_faces.services.clustering_service import ClusteringService
from inventario_faces.services.enhancement_service import EnhancementService
from inventario_faces.services.export_service import ExportService
from inventario_faces.services.hashing_service import HashingService
from inventario_faces.services.quality_service import FaceQualityService
from inventario_faces.services.scanner_service import ScannerService
from inventario_faces.services.search_service import SearchIndexService
from inventario_faces.services.tracking_service import FaceTrackingService, TrackingResult
from inventario_faces.services.video_service import VideoSamplingInfo, VideoService
from inventario_faces.utils.latex import format_seconds
from inventario_faces.utils.path_utils import ensure_directory, file_io_path, safe_stem
from inventario_faces.utils.serialization import to_serializable
from inventario_faces.utils.time_utils import as_utc, utc_now


@dataclass(frozen=True)
class ProcessedFileBundle:
    file_record: FileRecord
    tracking_result: TrackingResult | None = None


@dataclass(frozen=True)
class DistributedHealthResult:
    run_directory: Path
    report: ReportArtifacts
    health_snapshot: DistributedHealthSnapshot
    json_path: Path


@dataclass(frozen=True)
class LocalResumeContext:
    run_directory: Path
    state_path: Path
    partials_directory: Path
    plan_entries: tuple[DistributedPlanEntry, ...]
    completed_items: tuple[dict[str, Any], ...]
    resumed: bool


class InventoryService:
    """Orquestra o inventário facial, a busca e a exportação dos artefatos auditáveis."""

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
        work_directory: Path | None = None,
        progress_callback: ProgressCallback | None = None,
        log_callback: LogCallback | None = None,
    ) -> InventoryResult:
        """Executa o pipeline completo de inventário em um diretório de evidências."""

        if self._config.distributed.enabled:
            return self._run_distributed(
                root_directory,
                work_directory=work_directory,
                progress_callback=progress_callback,
                log_callback=log_callback,
            )

        root_directory = Path(root_directory).resolve()
        if not root_directory.exists():
            raise FileNotFoundError(f"Diretorio nao encontrado: {root_directory}")

        started_at_utc = utc_now()
        work_root = self._resolve_work_directory(root_directory, work_directory)
        output_root = self._resolve_output_root(root_directory, work_root)
        planned_files = self.list_planned_files(root_directory, work_root)
        total_files = len(planned_files)
        media_counter: Counter[MediaType] = Counter(media_type for _, media_type in planned_files)
        local_resume = self._prepare_local_resume_context(
            root_directory=root_directory,
            work_root=work_root,
            output_root=output_root,
            planned_files=planned_files,
            started_at_utc=started_at_utc,
        )
        run_directory = local_resume.run_directory
        logs_directory = ensure_directory(run_directory / "logs")
        text_logger = build_file_logger(logs_directory, self._config.app.log_level)
        event_logger = StructuredEventLogger(logs_directory / "events.jsonl")
        artifact_store = ArtifactStore(run_directory)
        export_service = ExportService(run_directory)

        try:
            self._emit_progress(progress_callback, 0, total_files, "Inicializando analise")
            self._emit_log(text_logger, log_callback, f"Diretorio analisado: {root_directory}")
            self._emit_log(text_logger, log_callback, f"Diretorio de trabalho: {work_root}")
            self._emit_log(text_logger, log_callback, f"Diretorio de execucao: {run_directory}")
            self._emit_log(
                text_logger,
                log_callback,
                f"[Logs] Texto={logs_directory / 'run.log'} | eventos={logs_directory / 'events.jsonl'}",
            )
            if local_resume.resumed:
                self._emit_log(
                    text_logger,
                    log_callback,
                    "[Retomada local] Execucao local incompleta localizada; reaproveitando itens ja concluidos.",
                )
            self._emit_log(text_logger, log_callback, "Pipeline orientado a tracks ativado.")
            self._emit_log(
                text_logger,
                log_callback,
                (
                    f"[Backend facial] Inicializando modelo {self._config.face_model.model_name}. "
                    "No primeiro uso, o bundle local pode ser preparado automaticamente."
                ),
            )
            analyzer = self._face_analyzer_factory()
            providers = list(getattr(analyzer, "providers", []))
            self._emit_log(
                text_logger,
                log_callback,
                (
                    f"[Backend facial] Modelo pronto | "
                    f"diretorio={getattr(analyzer, '_model_dir', '-')} | "
                    f"providers={', '.join(providers) if providers else 'desconhecido'}"
                ),
            )
            for line in self._configuration_log_lines(providers):
                self._emit_log(text_logger, log_callback, line)
            for line in self._planned_file_log_lines(planned_files):
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
                resumed=local_resume.resumed,
            )

            file_records: list[FileRecord] = []
            occurrences: list[FaceOccurrence] = []
            tracks: list[FaceTrack] = []
            keyframes: list[KeyFrame] = []
            total_detected_face_sizes: list[float] = []
            selected_face_sizes: list[float] = []
            resumed_payloads, resumed_paths = self._load_local_completed_payloads(
                context=local_resume,
                text_logger=text_logger,
                log_callback=log_callback,
            )
            for payload in resumed_payloads:
                partial = self._deserialize_partial_payload(payload)
                file_records.append(partial["file_record"])
                occurrences.extend(partial["occurrences"])
                tracks.extend(partial["tracks"])
                keyframes.extend(partial["keyframes"])
                total_detected_face_sizes.extend(partial["raw_face_sizes"])
                selected_face_sizes.extend(partial["selected_face_sizes"])

            if resumed_paths:
                self._emit_log(
                    text_logger,
                    log_callback,
                    (
                        f"[Retomada local] Itens reaproveitados={len(resumed_paths)} | "
                        f"pendentes={max(0, total_files - len(resumed_paths))}"
                    ),
                )
                self._emit_progress(
                    progress_callback,
                    len(resumed_paths),
                    total_files,
                    f"Retomada local: {len(resumed_paths)}/{total_files} arquivo(s) reaproveitado(s)",
                )

            attempted_count = len(resumed_paths)
            successful_count = len(resumed_paths)

            for entry in local_resume.plan_entries:
                if entry.relative_path in resumed_paths:
                    continue
                self._emit_progress(
                    progress_callback,
                    attempted_count,
                    total_files,
                    f"Processando {entry.source_path.name}",
                )
                bundle = self._process_file_bundle(
                    index=entry.index,
                    total_files=total_files,
                    file_path=entry.source_path,
                    media_type=entry.media_type,
                    analyzer=analyzer,
                    artifact_store=artifact_store,
                    event_logger=event_logger,
                    text_logger=text_logger,
                    log_callback=log_callback,
                )
                attempted_count += 1
                file_records.append(bundle.file_record)
                tracking_result = bundle.tracking_result
                if tracking_result is not None:
                    occurrences.extend(tracking_result.occurrences)
                    tracks.extend(tracking_result.tracks)
                    keyframes.extend(tracking_result.keyframes)
                    total_detected_face_sizes.extend(tracking_result.raw_face_sizes)
                    selected_face_sizes.extend(tracking_result.selected_face_sizes)
                if bundle.file_record.processing_error is None:
                    self._checkpoint_local_completed_bundle(
                        context=local_resume,
                        entry=entry,
                        bundle=bundle,
                    )
                    successful_count += 1
                self._emit_progress(progress_callback, attempted_count, total_files, f"Concluido: {entry.source_path.name}")

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
            self._emit_log(
                text_logger,
                log_callback,
                (
                    f"[Resumo] arquivos={summary.total_files} | midias={summary.media_files} | "
                    f"ocorrencias={summary.total_occurrences} | tracks={summary.total_tracks} | "
                    f"keyframes={summary.total_keyframes} | grupos={summary.total_clusters} | "
                    f"pares_probabilisticos={summary.probable_match_pairs}"
                ),
            )
            event_logger.write(
                "run_finished",
                summary=summary,
                report_pdf=report_artifacts.pdf_path,
                report_tex=report_artifacts.tex_path,
                report_docx=report_artifacts.docx_path,
                search=search_artifacts,
                resumed=local_resume.resumed,
                resumed_items=len(resumed_paths),
            )
            if successful_count >= total_files:
                self._mark_local_resume_finished(local_resume, finished_at_utc)
            else:
                self._mark_local_resume_pending(local_resume, finished_at_utc)
            return result
        except Exception as exc:
            error_summary, traceback_text = self._emit_exception(
                text_logger,
                log_callback,
                "[Execucao] Falha fatal do inventario",
                exc,
                include_traceback_in_callback=True,
            )
            event_logger.write(
                "run_failed",
                error=error_summary,
                error_type=type(exc).__name__,
                traceback=traceback_text,
                resumed=local_resume.resumed,
            )
            self._mark_local_resume_pending(local_resume, utc_now())
            raise
        finally:
            close_file_logger(text_logger)

    def _run_distributed(
        self,
        root_directory: Path,
        work_directory: Path | None = None,
        progress_callback: ProgressCallback | None = None,
        log_callback: LogCallback | None = None,
    ) -> InventoryResult:
        root_directory = Path(root_directory).resolve()
        if not root_directory.exists():
            raise FileNotFoundError(f"Diretorio nao encontrado: {root_directory}")

        started_at_utc = utc_now()
        work_root = self._resolve_work_directory(root_directory, work_directory)
        planned_files = self.list_planned_files(root_directory, work_root)
        total_files = len(planned_files)
        media_counter = Counter(media_type for _, media_type in planned_files)
        output_root = self._resolve_output_root(root_directory, work_root)
        run_directory = ensure_directory(
            output_root / f"cluster_{safe_stem(self._config.distributed.execution_label)}"
        )
        logs_directory = ensure_directory(run_directory / "logs")
        coordinator = DistributedCoordinator(root_directory, run_directory, self._config.distributed)
        text_logger = build_file_logger(
            logs_directory,
            self._config.app.log_level,
            file_name=f"node_{coordinator.node_id}.log",
        )
        event_logger = StructuredEventLogger(logs_directory / f"events_{coordinator.node_id}.jsonl")
        artifact_store = ArtifactStore(run_directory / "artifacts" / coordinator.node_id)
        heartbeat = DistributedNodeHeartbeat(coordinator, total_files)

        try:
            self._emit_progress(progress_callback, 0, total_files, "Inicializando analise distribuida")
            self._emit_log(text_logger, log_callback, f"Diretorio analisado: {root_directory}")
            self._emit_log(text_logger, log_callback, f"Diretorio de trabalho compartilhado: {work_root}")
            self._emit_log(text_logger, log_callback, f"Diretorio compartilhado de execucao: {run_directory}")
            self._emit_log(
                text_logger,
                log_callback,
                (
                    f"[Distribuicao] Modo multi-instancia ativado | execucao={self._config.distributed.execution_label} | "
                    f"no={coordinator.hostname}:{coordinator.pid}"
                ),
            )
            analyzer = self._face_analyzer_factory()
            providers = list(getattr(analyzer, "providers", []))
            for line in self._configuration_log_lines(providers):
                self._emit_log(text_logger, log_callback, line)
            for line in self._planned_file_log_lines(planned_files):
                self._emit_log(text_logger, log_callback, line)

            plan_entries = coordinator.load_or_create_plan(planned_files)
            heartbeat.start()
            heartbeat.update("planejamento")
            event_logger.write(
                "distributed_run_started",
                root_directory=root_directory,
                run_directory=run_directory,
                total_files=total_files,
                image_files=media_counter[MediaType.IMAGE],
                video_files=media_counter[MediaType.VIDEO],
                other_files=media_counter[MediaType.OTHER],
                node_id=coordinator.node_id,
                hostname=coordinator.hostname,
                pid=coordinator.pid,
            )

            for ordinal, entry in enumerate(plan_entries, start=1):
                self._emit_progress(progress_callback, ordinal - 1, total_files, f"Avaliando {entry.source_path.name}")
                claim_result = coordinator.try_claim(entry)
                if claim_result.status == "completed":
                    self._emit_log(
                        text_logger,
                        log_callback,
                        f"[Distribuicao {ordinal}/{total_files}] SALTANDO (Concluido): {entry.source_path.name}",
                    )
                    continue
                if claim_result.status == "busy":
                    self._emit_log(
                        text_logger,
                        log_callback,
                        (
                            f"[Distribuicao {ordinal}/{total_files}] OCUPADO: {entry.source_path.name} "
                            f"(por {claim_result.detail or 'outro no'})"
                        ),
                    )
                    continue

                claim = claim_result.claim
                if claim is None:
                    continue

                heartbeat.update("processando", entry)
                try:
                    bundle = self._process_file_bundle(
                        index=entry.index,
                        total_files=total_files,
                        file_path=entry.source_path,
                        media_type=entry.media_type,
                        analyzer=analyzer,
                        artifact_store=artifact_store,
                        event_logger=event_logger,
                        text_logger=text_logger,
                        log_callback=log_callback,
                    )
                    partial_path = coordinator.write_partial_payload(
                        entry,
                        self._serialize_partial_bundle(bundle),
                        file_sha512=bundle.file_record.sha512,
                    )
                    tracking_result = bundle.tracking_result
                    coordinator.mark_completed(
                        entry,
                        partial_path=partial_path,
                        sha512=bundle.file_record.sha512,
                        occurrence_count=(len(tracking_result.occurrences) if tracking_result is not None else 0),
                        track_count=(len(tracking_result.tracks) if tracking_result is not None else 0),
                        keyframe_count=(len(tracking_result.keyframes) if tracking_result is not None else 0),
                        processing_error=bundle.file_record.processing_error,
                    )
                    self._emit_log(
                        text_logger,
                        log_callback,
                        f"[Distribuicao] Item sincronizado no manifesto compartilhado: {entry.source_path.name}",
                    )
                finally:
                    coordinator.release_claim(claim)
                    heartbeat.update("planejamento")

            snapshot = coordinator.snapshot(total_files)
            health = coordinator.inspect_health(total_files=total_files)
            finished_at_utc = utc_now()
            if snapshot.is_complete and self._config.distributed.auto_finalize:
                finalize_lock = coordinator.try_acquire_finalize_lock()
                if finalize_lock is not None:
                    heartbeat.update("consolidando")
                    try:
                        return self._finalize_distributed_inventory(
                            root_directory=root_directory,
                            work_directory=work_root,
                            run_directory=run_directory,
                            logs_directory=logs_directory,
                            started_at_utc=started_at_utc,
                            finished_at_utc=finished_at_utc,
                            coordinator=coordinator,
                            event_logger=event_logger,
                            text_logger=text_logger,
                            log_callback=log_callback,
                        )
                    finally:
                        coordinator.release_finalize_lock()
                self._emit_log(
                    text_logger,
                    log_callback,
                    "[Distribuicao] Todos os arquivos ja foram concluidos; outra instancia esta consolidando o relatorio final.",
                )

            status_path = self._write_distributed_status_file(
                run_directory=run_directory,
                snapshot=snapshot,
                total_media_files=media_counter[MediaType.IMAGE] + media_counter[MediaType.VIDEO],
                health=health,
            )
            self._write_distributed_health_files(
                run_directory=run_directory,
                health=health,
            )
            summary = ProcessingSummary(
                total_files=total_files,
                media_files=media_counter[MediaType.IMAGE] + media_counter[MediaType.VIDEO],
                image_files=media_counter[MediaType.IMAGE],
                video_files=media_counter[MediaType.VIDEO],
                total_occurrences=0,
                total_clusters=0,
                probable_match_pairs=0,
                total_tracks=0,
                total_keyframes=0,
            )
            self._emit_log(
                text_logger,
                log_callback,
                (
                    f"[Distribuicao] Trabalho parcial concluido | concluidos={snapshot.completed_files}/{snapshot.total_files} | "
                    f"em_processamento={snapshot.active_claims} | pendentes={snapshot.pending_files}"
                ),
            )
            for line in self._distributed_health_log_lines(health):
                self._emit_log(text_logger, log_callback, line)
            return InventoryResult(
                run_directory=run_directory,
                started_at_utc=started_at_utc,
                finished_at_utc=finished_at_utc,
                root_directory=root_directory,
                files=[],
                occurrences=[],
                clusters=[],
                report=ReportArtifacts(tex_path=status_path, pdf_path=None, docx_path=None),
                summary=summary,
                logs_directory=logs_directory,
                manifest_path=run_directory / "inventory" / "manifest.json",
                tracks=[],
                keyframes=[],
                search=SearchArtifacts(
                    engine="pending",
                    track_index_path=None,
                    track_metadata_path=None,
                    cluster_index_path=None,
                    cluster_metadata_path=None,
                ),
            )
        finally:
            heartbeat.stop()
            close_file_logger(text_logger)

    def list_planned_files(
        self,
        root_directory: Path,
        work_directory: Path | None = None,
    ) -> list[tuple[Path, MediaType]]:
        """Lista, em ordem determinística, os arquivos que entrarão no processamento."""

        root_directory = Path(root_directory).resolve()
        output_root = self._output_root_path(root_directory, work_directory)
        return [
            (path, self._scanner_service.classify(path))
            for path in self._scanner_service.iter_scan(
                root_directory,
                excluded_directories={output_root},
            )
        ]

    def _prepare_local_resume_context(
        self,
        *,
        root_directory: Path,
        work_root: Path,
        output_root: Path,
        planned_files: list[tuple[Path, MediaType]],
        started_at_utc: datetime,
    ) -> LocalResumeContext:
        plan_entries = tuple(
            DistributedPlanEntry(
                index=index,
                source_path=path,
                media_type=media_type,
                relative_path=str(path.resolve().relative_to(root_directory)).replace("\\", "/"),
            )
            for index, (path, media_type) in enumerate(planned_files, start=1)
        )
        config_digest = self._config_digest()
        planned_signature = self._plan_signature(plan_entries)

        for candidate in sorted(output_root.glob("run_*"), reverse=True):
            state_path = candidate / "runtime" / "local_resume_state.json"
            if not state_path.exists():
                continue
            state = self._load_json_file(state_path)
            if not isinstance(state, dict):
                continue
            if str(state.get("mode", "")) != "local_resume":
                continue
            if state.get("completed_at_utc") not in (None, ""):
                continue
            if str(state.get("root_directory", "")) != str(root_directory):
                continue
            if str(state.get("work_directory", "")) != str(work_root):
                continue
            if str(state.get("config_digest", "")) != config_digest:
                continue
            if str(state.get("plan_signature", "")) != planned_signature:
                continue
            return LocalResumeContext(
                run_directory=candidate,
                state_path=state_path,
                partials_directory=ensure_directory(candidate / "runtime" / "partials"),
                plan_entries=plan_entries,
                completed_items=tuple(
                    item for item in state.get("completed", []) if isinstance(item, dict)
                ),
                resumed=True,
            )

        run_directory = ensure_directory(output_root / f"run_{started_at_utc.strftime('%Y%m%d_%H%M%S')}")
        state_path = run_directory / "runtime" / "local_resume_state.json"
        partials_directory = ensure_directory(run_directory / "runtime" / "partials")
        self._write_local_resume_state(
            state_path,
            {
                "schema_version": 1,
                "mode": "local_resume",
                "root_directory": str(root_directory),
                "work_directory": str(work_root),
                "config_digest": config_digest,
                "plan_signature": planned_signature,
                "started_at_utc": started_at_utc.isoformat(),
                "updated_at_utc": started_at_utc.isoformat(),
                "completed_at_utc": None,
                "plan": [self._serialize_plan_entry(entry) for entry in plan_entries],
                "completed": [],
            },
        )
        return LocalResumeContext(
            run_directory=run_directory,
            state_path=state_path,
            partials_directory=partials_directory,
            plan_entries=plan_entries,
            completed_items=(),
            resumed=False,
        )

    def _load_local_completed_payloads(
        self,
        *,
        context: LocalResumeContext,
        text_logger: logging.Logger,
        log_callback: LogCallback | None,
    ) -> tuple[list[dict[str, object]], set[str]]:
        plan_map = {entry.relative_path: entry for entry in context.plan_entries}
        reusable_payloads: list[dict[str, object]] = []
        reusable_paths: set[str] = set()
        retained_items: list[dict[str, Any]] = []

        for item in context.completed_items:
            relative_path = str(item.get("relative_path", ""))
            entry = plan_map.get(relative_path)
            if entry is None:
                continue
            payload = self._load_local_resume_payload(entry=entry, item=item)
            if payload is None:
                self._emit_log(
                    text_logger,
                    log_callback,
                    f"[Retomada local] Parcial invalido descartado e agendado para reprocessamento: {relative_path}",
                )
                continue
            reusable_payloads.append(payload)
            reusable_paths.add(relative_path)
            retained_items.append(item)

        if len(retained_items) != len(context.completed_items):
            state = self._load_json_file(context.state_path)
            if isinstance(state, dict):
                state["completed"] = retained_items
                state["updated_at_utc"] = utc_now().isoformat()
                self._write_local_resume_state(context.state_path, state)

        return reusable_payloads, reusable_paths

    def _load_local_resume_payload(
        self,
        *,
        entry: DistributedPlanEntry,
        item: dict[str, Any],
    ) -> dict[str, object] | None:
        partial_path_raw = item.get("partial_path")
        if partial_path_raw in (None, ""):
            return None
        partial_path = Path(str(partial_path_raw))
        if not partial_path.exists() or not entry.source_path.exists():
            return None

        try:
            source_stat = entry.source_path.stat()
        except FileNotFoundError:
            return None

        expected_size = item.get("size_bytes")
        if expected_size is not None and int(expected_size) != source_stat.st_size:
            return None

        expected_modified = item.get("modified_at_utc")
        current_modified = as_utc(source_stat.st_mtime)
        if (
            expected_modified not in (None, "")
            and current_modified is not None
            and str(expected_modified) != current_modified.isoformat()
        ):
            return None

        state_payload = self._load_json_file(partial_path)
        if not isinstance(state_payload, dict):
            return None
        payload = state_payload.get("payload")
        if not isinstance(payload, dict):
            return None
        entry_payload = state_payload.get("entry")
        if isinstance(entry_payload, dict) and str(entry_payload.get("relative_path", "")) != entry.relative_path:
            return None
        payload_digest = str(state_payload.get("payload_sha256", ""))
        if payload_digest and payload_digest != self._payload_digest(payload):
            return None
        return payload

    def _checkpoint_local_completed_bundle(
        self,
        *,
        context: LocalResumeContext,
        entry: DistributedPlanEntry,
        bundle: ProcessedFileBundle,
    ) -> None:
        payload = self._serialize_partial_bundle(bundle)
        partial_path = context.partials_directory / f"{entry.lock_stem}.json"
        self._write_json_atomic(
            partial_path,
            {
                "schema_version": 1,
                "entry": self._serialize_plan_entry(entry),
                "payload_sha256": self._payload_digest(payload),
                "payload": payload,
            },
        )

        state = self._load_json_file(context.state_path)
        if not isinstance(state, dict):
            return
        completed_items = [
            item
            for item in state.get("completed", [])
            if isinstance(item, dict) and item.get("relative_path") != entry.relative_path
        ]
        completed_items.append(
            {
                "index": entry.index,
                "relative_path": entry.relative_path,
                "partial_path": str(partial_path),
                "sha512": bundle.file_record.sha512,
                "size_bytes": bundle.file_record.size_bytes,
                "modified_at_utc": (
                    bundle.file_record.modified_at_utc.isoformat()
                    if bundle.file_record.modified_at_utc is not None
                    else None
                ),
                "completed_at_utc": utc_now().isoformat(),
            }
        )
        completed_items.sort(key=lambda item: int(item.get("index", 0)))
        state["completed"] = completed_items
        state["updated_at_utc"] = utc_now().isoformat()
        self._write_local_resume_state(context.state_path, state)

    def _mark_local_resume_finished(self, context: LocalResumeContext, finished_at_utc: datetime) -> None:
        state = self._load_json_file(context.state_path)
        if not isinstance(state, dict):
            return
        state["updated_at_utc"] = finished_at_utc.isoformat()
        state["completed_at_utc"] = finished_at_utc.isoformat()
        self._write_local_resume_state(context.state_path, state)

    def _mark_local_resume_pending(self, context: LocalResumeContext, updated_at_utc: datetime) -> None:
        state = self._load_json_file(context.state_path)
        if not isinstance(state, dict):
            return
        state["updated_at_utc"] = updated_at_utc.isoformat()
        state["completed_at_utc"] = None
        self._write_local_resume_state(context.state_path, state)

    def _serialize_plan_entry(self, entry: DistributedPlanEntry) -> dict[str, object]:
        return {
            "index": entry.index,
            "relative_path": entry.relative_path,
            "media_type": entry.media_type.value,
            "source_path": str(entry.source_path),
        }

    def _plan_signature(self, plan_entries: tuple[DistributedPlanEntry, ...]) -> str:
        return self._payload_digest(
            [
                {
                    "index": entry.index,
                    "relative_path": entry.relative_path,
                    "media_type": entry.media_type.value,
                }
                for entry in plan_entries
            ]
        )

    def _config_digest(self) -> str:
        return self._payload_digest(to_serializable(self._config))

    def _payload_digest(self, payload: object) -> str:
        import hashlib

        serialized = json.dumps(
            to_serializable(payload),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def _load_json_file(self, path: Path) -> object:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            return None

    def _write_local_resume_state(self, path: Path, payload: dict[str, Any]) -> None:
        self._write_json_atomic(path, payload)

    def _write_json_atomic(self, path: Path, payload: object) -> None:
        ensure_directory(path.parent)
        serialized = json.dumps(to_serializable(payload), indent=2, ensure_ascii=False)
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f"{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            stream.write(serialized)
            temporary_path = Path(stream.name)
        temporary_path.replace(path)

    def inspect_distributed_health(
        self,
        root_directory: Path,
        work_directory: Path | None = None,
        log_callback: LogCallback | None = None,
    ) -> DistributedHealthResult:
        """Inspeciona a saude operacional do lote compartilhado sem reprocessar as evidencias."""

        if not self._config.distributed.enabled:
            raise ValueError("Ative o modo distribuido para inspecionar a saude da execucao compartilhada.")

        root_directory = Path(root_directory).resolve()
        if not root_directory.exists():
            raise FileNotFoundError(f"Diretorio nao encontrado: {root_directory}")

        work_root = self._resolve_work_directory(root_directory, work_directory)
        run_directory = ensure_directory(
            self._resolve_output_root(root_directory, work_root)
            / f"cluster_{safe_stem(self._config.distributed.execution_label)}"
        )
        logs_directory = ensure_directory(run_directory / "logs")
        coordinator = DistributedCoordinator(root_directory, run_directory, self._config.distributed)
        logger = build_file_logger(
            logs_directory,
            self._config.app.log_level,
            file_name=f"health_{coordinator.node_id}.log",
        )

        try:
            planned_files = self.list_planned_files(root_directory, work_root)
            plan_entries = coordinator.load_or_create_plan(planned_files)
            health = coordinator.inspect_health(total_files=len(plan_entries))
            text_path, json_path = self._write_distributed_health_files(
                run_directory=run_directory,
                health=health,
            )
            for line in self._distributed_health_log_lines(health):
                self._emit_log(logger, log_callback, line)
            return DistributedHealthResult(
                run_directory=run_directory,
                report=ReportArtifacts(tex_path=text_path, pdf_path=None, docx_path=None),
                health_snapshot=health,
                json_path=json_path,
            )
        finally:
            close_file_logger(logger)

    def run_face_search(
        self,
        root_directory: Path,
        query_image_path: Path,
        work_directory: Path | None = None,
        progress_callback: ProgressCallback | None = None,
        log_callback: LogCallback | None = None,
    ) -> FaceSearchResult:
        """Processa o acervo e pesquisa uma face de consulta contra as tracks indexadas."""

        query_path = Path(query_image_path).resolve()
        if not query_path.exists():
            raise FileNotFoundError(f"Imagem de consulta nao encontrada: {query_path}")

        def inventory_progress(current: int, total: int, message: str) -> None:
            if progress_callback is None:
                return
            scaled = 0 if total == 0 else int((current / total) * 85)
            progress_callback(scaled, 100, message)

        if self._config.distributed.enabled:
            local_service = InventoryService(
                config=replace(
                    self._config,
                    distributed=replace(self._config.distributed, enabled=False),
                ),
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
            inventory_result = local_service.run(
                root_directory,
                work_directory=work_directory,
                progress_callback=inventory_progress,
                log_callback=log_callback,
            )
        else:
            inventory_result = self.run(
                root_directory,
                work_directory=work_directory,
                progress_callback=inventory_progress,
                log_callback=log_callback,
            )

        if progress_callback is not None:
            progress_callback(88, 100, "Analisando imagem de consulta")

        logger = build_file_logger(inventory_result.logs_directory, self._config.app.log_level)
        event_logger = StructuredEventLogger(inventory_result.logs_directory / "events.jsonl")
        export_service = ExportService(inventory_result.run_directory)
        try:
            self._emit_log(
                logger,
                log_callback,
                f"[Logs] Texto={inventory_result.logs_directory / 'run.log'} | eventos={inventory_result.logs_directory / 'events.jsonl'}",
            )
            self._emit_log(logger, log_callback, f"[Busca por face] Imagem de consulta: {query_path}")
            analyzer = self._face_analyzer_factory()
            processing_query_path, query_sha512, query_cleanup_path = self._prepare_processing_input(
                file_path=query_path,
                media_type=MediaType.IMAGE,
                file_prefix="[Busca por face]",
                text_logger=logger,
                log_callback=log_callback,
            )
            try:
                query_tracking = self._tracking_service.process_media(
                    source_path=query_path,
                    sha512=query_sha512,
                    media_type=MediaType.IMAGE,
                    frames=self._frames_with_original_source(
                        [self._media_service.load_image(processing_query_path)],
                        query_path,
                    ),
                    analyzer=analyzer,
                    artifact_store=ArtifactStore(inventory_result.run_directory / "face_search_query"),
                    id_namespace="Q",
                    event_callback=lambda event, fields: event_logger.write(event, **fields),
                    text_callback=lambda message: self._emit_log(logger, log_callback, message),
                )
            finally:
                self._cleanup_processing_input(query_cleanup_path)
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
        except Exception as exc:
            error_summary, traceback_text = self._emit_exception(
                logger,
                log_callback,
                "[Busca por face] Falha fatal",
                exc,
                include_traceback_in_callback=True,
            )
            event_logger.write(
                "face_search_failed",
                query_path=query_path,
                error=error_summary,
                error_type=type(exc).__name__,
                traceback=traceback_text,
            )
            raise
        finally:
            close_file_logger(logger)

    def _process_file_bundle(
        self,
        *,
        index: int,
        total_files: int,
        file_path: Path,
        media_type: MediaType,
        analyzer: FaceAnalyzer,
        artifact_store: ArtifactStore,
        event_logger: StructuredEventLogger,
        text_logger: logging.Logger,
        log_callback: LogCallback | None,
    ) -> ProcessedFileBundle:
        file_prefix = f"[Arquivo {index}/{total_files}]"
        self._emit_log(
            text_logger,
            log_callback,
            f"{file_prefix} Inicio do processamento | tipo={self._media_type_label(media_type)} | caminho={file_path}",
        )

        discovered_at_utc = utc_now()
        stat = file_path.stat()
        modified_at_utc = as_utc(stat.st_mtime)
        sha512 = ""
        processing_error: str | None = None
        media_info_tracks = ()
        media_info_error: str | None = None
        tracking_result: TrackingResult | None = None
        current_stage = "preparacao de entrada"

        try:
            self._emit_log(
                text_logger,
                log_callback,
                f"{file_prefix} [Etapa] Preparando entrada e calculando hash SHA-512.",
            )
            processing_path, sha512, cleanup_path = self._prepare_processing_input(
                file_path=file_path,
                media_type=media_type,
                file_prefix=file_prefix,
                text_logger=text_logger,
                log_callback=log_callback,
            )
            self._emit_log(
                text_logger,
                log_callback,
                f"{file_prefix} [Etapa] Entrada preparada | sha512={sha512[:16]}...",
            )

            if media_type in {MediaType.IMAGE, MediaType.VIDEO}:
                current_stage = "extracao de metadados"
                self._emit_log(
                    text_logger,
                    log_callback,
                    f"{file_prefix} [Etapa] Extraindo metadados tecnicos da midia.",
                )
                media_info_tracks, media_info_error = self._extract_media_info(processing_path)
                if media_info_error is not None:
                    self._emit_log(
                        text_logger,
                        log_callback,
                        f"{file_prefix} Metadados tecnicos indisponiveis: {media_info_error}",
                    )

            frames = None
            if media_type == MediaType.IMAGE:
                current_stage = "carregamento de imagem"
                self._emit_log(
                    text_logger,
                    log_callback,
                    f"{file_prefix} [Etapa] Carregando imagem para analise.",
                )
                frames = self._frames_with_original_source(
                    [self._media_service.load_image(processing_path)],
                    file_path,
                )
            elif media_type == MediaType.VIDEO:
                current_stage = "amostragem de video"
                self._emit_log(
                    text_logger,
                    log_callback,
                    f"{file_prefix} [Etapa] Amostrando quadros do video.",
                )
                sampled_frames = self._media_service.sample_video(
                    processing_path,
                    metadata_callback=lambda info: self._emit_log(
                        text_logger,
                        log_callback,
                        self._format_video_sampling_log(file_prefix, info),
                    ),
                )
                frames = self._frames_with_original_source(sampled_frames, file_path)

            if frames is not None:
                current_stage = "deteccao, tracking e embeddings"
                self._emit_log(
                    text_logger,
                    log_callback,
                    f"{file_prefix} [Etapa] Executando deteccao, tracking e embeddings faciais.",
                )
                tracking_result = self._tracking_service.process_media(
                    source_path=file_path,
                    sha512=sha512,
                    media_type=media_type,
                    frames=frames,
                    analyzer=analyzer,
                    artifact_store=artifact_store,
                    id_namespace=f"{index:05d}",
                    event_callback=lambda event, fields: event_logger.write(event, **fields),
                    text_callback=lambda message: self._emit_log(text_logger, log_callback, message),
                )

            if tracking_result is not None:
                self._emit_log(
                    text_logger,
                    log_callback,
                    (
                        f"{file_prefix} Midia analisada | "
                        f"amostras={tracking_result.sampled_frames} | "
                        f"quadros_com_face={tracking_result.frames_with_faces} | "
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
            processing_error, traceback_text = self._emit_exception(
                text_logger,
                log_callback,
                (
                    f"{file_prefix} Erro de processamento | etapa={current_stage} | "
                    f"arquivo={file_path}"
                ),
                exc,
            )
            event_logger.write(
                "file_processing_error",
                path=file_path,
                media_type=media_type,
                sha512=sha512,
                error=processing_error,
                error_type=type(exc).__name__,
                error_stage=current_stage,
                traceback=traceback_text,
                media_info_tracks=media_info_tracks,
                media_info_error=media_info_error,
            )
        finally:
            if "cleanup_path" in locals():
                self._cleanup_processing_input(cleanup_path)

        file_record = FileRecord(
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
        return ProcessedFileBundle(file_record=file_record, tracking_result=tracking_result)

    def _resolve_work_directory(
        self,
        root_directory: Path,
        work_directory: Path | None,
    ) -> Path:
        if work_directory is None:
            return Path(root_directory).resolve()
        return Path(work_directory).resolve()

    def _resolve_output_root(
        self,
        root_directory: Path,
        work_directory: Path | None,
    ) -> Path:
        return ensure_directory(self._output_root_path(root_directory, work_directory))

    def _output_root_path(
        self,
        root_directory: Path,
        work_directory: Path | None,
    ) -> Path:
        work_root = self._resolve_work_directory(root_directory, work_directory)
        return work_root / self._config.app.output_directory_name

    def _prepare_processing_input(
        self,
        *,
        file_path: Path,
        media_type: MediaType,
        file_prefix: str,
        text_logger: logging.Logger,
        log_callback: LogCallback | None,
    ) -> tuple[Path, str, Path | None]:
        if not self._config.app.use_local_temp_copy or media_type not in {MediaType.IMAGE, MediaType.VIDEO}:
            sha512 = self._hashing_service.sha512(file_path)
            self._emit_log(
                text_logger,
                log_callback,
                (
                    f"{file_prefix} Entrada original mantida | "
                    f"origem={file_path} | sha512={sha512[:16]}..."
                ),
            )
            return file_path, sha512, None

        temporary_directory = Path(tempfile.mkdtemp(prefix="inventario_faces_media_"))
        temporary_path = temporary_directory / file_path.name
        sha512 = self._copy_file_with_sha512(file_path, temporary_path)
        self._emit_log(
            text_logger,
            log_callback,
            (
                f"{file_prefix} Copia temporaria local preparada | "
                f"origem={file_path} | copia_local={temporary_path}"
            ),
        )
        return temporary_path, sha512, temporary_directory

    def _copy_file_with_sha512(self, source_path: Path, target_path: Path, chunk_size: int = 1024 * 1024) -> str:
        import hashlib

        hasher = hashlib.sha512()
        ensure_directory(target_path.parent)
        with open(file_io_path(source_path), "rb") as source_stream, open(file_io_path(target_path), "wb") as target_stream:
            while True:
                chunk = source_stream.read(chunk_size)
                if not chunk:
                    break
                hasher.update(chunk)
                target_stream.write(chunk)
        return hasher.hexdigest()

    def _frames_with_original_source(
        self,
        frames: object,
        original_source_path: Path,
    ) -> object:
        return (
            replace(frame, source_path=original_source_path)
            for frame in frames
        )

    def _cleanup_processing_input(self, cleanup_path: Path | None) -> None:
        if cleanup_path is None:
            return
        shutil.rmtree(cleanup_path, ignore_errors=True)

    def _serialize_partial_bundle(self, bundle: ProcessedFileBundle) -> dict[str, object]:
        tracking_result = bundle.tracking_result
        return {
            "file_record": to_serializable(bundle.file_record),
            "occurrences": to_serializable(tracking_result.occurrences if tracking_result is not None else []),
            "tracks": to_serializable(tracking_result.tracks if tracking_result is not None else []),
            "keyframes": to_serializable(tracking_result.keyframes if tracking_result is not None else []),
            "raw_face_sizes": list(tracking_result.raw_face_sizes) if tracking_result is not None else [],
            "selected_face_sizes": list(tracking_result.selected_face_sizes) if tracking_result is not None else [],
        }

    def _write_distributed_status_file(
        self,
        *,
        run_directory: Path,
        snapshot: DistributedExecutionSnapshot,
        total_media_files: int,
        health: DistributedHealthSnapshot,
    ) -> Path:
        report_directory = ensure_directory(run_directory / "report")
        status_path = report_directory / "progresso_distribuido.txt"
        lines = [
            "Inventario Faces - Progresso Distribuido",
            f"Execucao compartilhada: {self._config.distributed.execution_label}",
            f"Arquivos no plano: {snapshot.processable_files}",
            f"Arquivos varridos no plano: {snapshot.total_files}",
            f"Midias suportadas no plano: {total_media_files}",
            f"Concluidos: {snapshot.processable_completed_files}",
            f"Em processamento: {snapshot.processable_active_claims}",
            f"Pendentes: {snapshot.processable_pending_files}",
            f"Parciais integros: {health.healthy_partials}",
            f"Parciais ausentes: {health.missing_partials}",
            f"Parciais corrompidos: {health.corrupted_partials}",
            f"Nos ativos: {health.active_nodes}",
            f"Nos stale: {health.stale_nodes}",
            f"Claims stale: {health.stale_claims}",
            (
                "Status: consolidacao final concluida."
                if snapshot.is_complete
                else "Status: processamento parcial; aguarde as demais instancias para o relatorio consolidado."
            ),
        ]
        status_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return status_path

    def _write_distributed_health_files(
        self,
        *,
        run_directory: Path,
        health: DistributedHealthSnapshot,
    ) -> tuple[Path, Path]:
        report_directory = ensure_directory(run_directory / "report")
        text_path = report_directory / "saude_distribuida.txt"
        json_path = report_directory / "saude_distribuida.json"

        lines = [
            "Inventario Faces - Saude da Execucao Distribuida",
            f"Execucao compartilhada: {self._config.distributed.execution_label}",
            f"Arquivos no plano: {health.processable_files}",
            f"Arquivos varridos no plano: {health.total_files}",
            f"Concluidos no manifesto: {health.processable_completed_files}",
            f"Em processamento: {health.processable_active_claims}",
            f"Pendentes: {health.processable_pending_files}",
            f"Parciais integros: {health.healthy_partials}",
            f"Parciais ausentes: {health.missing_partials}",
            f"Parciais corrompidos: {health.corrupted_partials}",
            f"Claims stale: {health.stale_claims}",
            f"Nos ativos: {health.active_nodes}",
            f"Nos stale: {health.stale_nodes}",
            f"Lock de finalizacao ativo: {'sim' if health.finalize_lock_active else 'nao'}",
            f"Recuperacao necessaria: {'sim' if health.recovery_needed else 'nao'}",
            "",
            "Parciais com problema:",
        ]
        invalid_partials = [item for item in health.partials if not item.is_healthy]
        if invalid_partials:
            lines.extend(
                f"- {item.entry.relative_path} | status={item.status} | detalhe={item.detail}"
                for item in invalid_partials
            )
        else:
            lines.append("- nenhum")

        lines.extend(["", "Nos observados:"])
        if health.nodes:
            lines.extend(
                (
                    f"- {node.hostname}:{node.pid if node.pid is not None else '-'} | "
                    f"fase={node.phase} | stale={'sim' if node.is_stale else 'nao'} | "
                    f"arquivo={node.current_relative_path or '-'}"
                )
                for node in health.nodes
            )
        else:
            lines.append("- nenhum heartbeat localizado")

        text_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        json_path.write_text(
            json.dumps(to_serializable(health), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return text_path, json_path

    def _distributed_health_log_lines(
        self,
        health: DistributedHealthSnapshot,
    ) -> list[str]:
        lines = [
            (
                "[Distribuicao] Saude do lote | "
                f"planejados_processaveis={health.processable_files} | "
                f"concluidos_processaveis={health.processable_completed_files} | "
                f"em_processamento={health.processable_active_claims} | "
                f"pendentes={health.processable_pending_files} | "
                f"parciais_integros={health.healthy_partials} | "
                f"parciais_ausentes={health.missing_partials} | "
                f"parciais_corrompidos={health.corrupted_partials} | "
                f"claims_stale={health.stale_claims} | "
                f"nos_ativos={health.active_nodes} | nos_stale={health.stale_nodes}"
            )
        ]
        for item in health.partials:
            if item.is_healthy:
                continue
            lines.append(
                (
                    "[Distribuicao] Parcial com problema | "
                    f"arquivo={item.entry.relative_path} | status={item.status} | detalhe={item.detail}"
                )
            )
        return lines

    def _recover_invalid_distributed_partials(
        self,
        *,
        validations: list[DistributedPartialValidation],
        total_files: int,
        run_directory: Path,
        coordinator: DistributedCoordinator,
        event_logger: StructuredEventLogger,
        text_logger: logging.Logger,
        log_callback: LogCallback | None,
    ) -> int:
        if not validations:
            return 0

        analyzer = self._face_analyzer_factory()
        artifact_store = ArtifactStore(run_directory / "artifacts" / coordinator.node_id / "recovery")
        recovered = 0
        for validation in validations:
            entry = validation.entry
            self._emit_log(
                text_logger,
                log_callback,
                (
                    "[Distribuicao] Recuperando parcial invalido | "
                    f"arquivo={entry.relative_path} | status={validation.status} | detalhe={validation.detail}"
                ),
            )
            event_logger.write(
                "distributed_partial_recovery_started",
                relative_path=entry.relative_path,
                status=validation.status,
                detail=validation.detail,
            )
            bundle = self._process_file_bundle(
                index=entry.index,
                total_files=total_files,
                file_path=entry.source_path,
                media_type=entry.media_type,
                analyzer=analyzer,
                artifact_store=artifact_store,
                event_logger=event_logger,
                text_logger=text_logger,
                log_callback=log_callback,
            )
            previous_sha512 = str(validation.manifest_item.get("sha512", ""))
            if previous_sha512 and previous_sha512 != bundle.file_record.sha512:
                event_logger.write(
                    "distributed_partial_recovery_source_changed",
                    relative_path=entry.relative_path,
                    previous_sha512=previous_sha512,
                    current_sha512=bundle.file_record.sha512,
                )
                raise RuntimeError(
                    f"O arquivo de origem mudou antes da recuperacao do parcial distribuido: {entry.relative_path}"
                )
            partial_path = coordinator.write_partial_payload(
                entry,
                self._serialize_partial_bundle(bundle),
                file_sha512=bundle.file_record.sha512,
            )
            tracking_result = bundle.tracking_result
            coordinator.mark_completed(
                entry,
                partial_path=partial_path,
                sha512=bundle.file_record.sha512,
                occurrence_count=(len(tracking_result.occurrences) if tracking_result is not None else 0),
                track_count=(len(tracking_result.tracks) if tracking_result is not None else 0),
                keyframe_count=(len(tracking_result.keyframes) if tracking_result is not None else 0),
                processing_error=bundle.file_record.processing_error,
            )
            event_logger.write(
                "distributed_partial_recovery_finished",
                relative_path=entry.relative_path,
                partial_path=partial_path,
                track_count=(len(tracking_result.tracks) if tracking_result is not None else 0),
                keyframe_count=(len(tracking_result.keyframes) if tracking_result is not None else 0),
            )
            recovered += 1
        return recovered

    def _finalize_distributed_inventory(
        self,
        *,
        root_directory: Path,
        work_directory: Path | None,
        run_directory: Path,
        logs_directory: Path,
        started_at_utc,
        finished_at_utc,
        coordinator: DistributedCoordinator,
        event_logger: StructuredEventLogger,
        text_logger: logging.Logger,
        log_callback: LogCallback | None,
    ) -> InventoryResult:
        self._emit_log(
            text_logger,
            log_callback,
            "[Distribuicao] Iniciando consolidacao final dos resultados parciais.",
        )
        planned_files = self.list_planned_files(root_directory, work_directory)
        plan_entries = coordinator.load_or_create_plan(planned_files)
        health = coordinator.inspect_health(total_files=len(plan_entries))
        for line in self._distributed_health_log_lines(health):
            self._emit_log(text_logger, log_callback, line)

        if self._config.distributed.validate_partial_integrity and health.recovery_needed:
            self._emit_log(
                text_logger,
                log_callback,
                (
                    "[Distribuicao] Foram detectados parciais ausentes ou corrompidos; "
                    "iniciando recuperacao automatica antes da consolidacao final."
                ),
            )
            if not self._config.distributed.auto_reprocess_invalid_partials:
                raise RuntimeError(
                    "A consolidacao final encontrou parciais invalidos, mas a recuperacao automatica esta desativada."
                )
            recovered = self._recover_invalid_distributed_partials(
                validations=[item for item in health.partials if not item.is_healthy],
                total_files=len(plan_entries),
                run_directory=run_directory,
                coordinator=coordinator,
                event_logger=event_logger,
                text_logger=text_logger,
                log_callback=log_callback,
            )
            self._emit_log(
                text_logger,
                log_callback,
                f"[Distribuicao] Recuperacao automatica concluida | itens_recuperados={recovered}",
            )
            health = coordinator.inspect_health(total_files=len(plan_entries))
            for line in self._distributed_health_log_lines(health):
                self._emit_log(text_logger, log_callback, line)
            if health.recovery_needed:
                raise RuntimeError(
                    "Persistem parciais invalidos apos a recuperacao automatica; revise a saude distribuida do lote."
                )

        self._write_distributed_health_files(
            run_directory=run_directory,
            health=health,
        )
        payloads = [
            item.payload
            for item in health.partials
            if item.is_healthy and item.payload is not None
        ]
        file_records: list[FileRecord] = []
        occurrences: list[FaceOccurrence] = []
        tracks: list[FaceTrack] = []
        keyframes: list[KeyFrame] = []
        total_detected_face_sizes: list[float] = []
        selected_face_sizes: list[float] = []

        for payload in payloads:
            partial = self._deserialize_partial_payload(payload)
            file_records.append(partial["file_record"])
            occurrences.extend(partial["occurrences"])
            tracks.extend(partial["tracks"])
            keyframes.extend(partial["keyframes"])
            total_detected_face_sizes.extend(partial["raw_face_sizes"])
            selected_face_sizes.extend(partial["selected_face_sizes"])

        self._emit_log(
            text_logger,
            log_callback,
            f"[Agrupamento] Consolidando {len(tracks)} tracks em possiveis grupos.",
        )
        clusters = self._clustering_service.cluster(tracks)
        self._propagate_cluster_membership(occurrences, tracks)
        search_artifacts = self._search_service.build(run_directory, tracks, clusters)
        summary = self._build_summary(
            file_records=file_records,
            occurrences=occurrences,
            tracks=tracks,
            keyframes=keyframes,
            clusters=clusters,
            total_detected_face_sizes=total_detected_face_sizes,
            selected_face_sizes=selected_face_sizes,
        )

        export_service = ExportService(run_directory)
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
        self._emit_log(
            text_logger,
            log_callback,
            (
                f"[Resumo] arquivos={summary.total_files} | midias={summary.media_files} | "
                f"ocorrencias={summary.total_occurrences} | tracks={summary.total_tracks} | "
                f"keyframes={summary.total_keyframes} | grupos={summary.total_clusters} | "
                f"pares_probabilisticos={summary.probable_match_pairs}"
            ),
        )
        event_logger.write(
            "distributed_run_finished",
            summary=summary,
            report_pdf=report_artifacts.pdf_path,
            report_tex=report_artifacts.tex_path,
            report_docx=report_artifacts.docx_path,
            search=search_artifacts,
        )
        self._write_distributed_health_files(
            run_directory=run_directory,
            health=coordinator.inspect_health(total_files=len(plan_entries)),
        )
        return result

    def _deserialize_partial_payload(self, payload: dict[str, object]) -> dict[str, object]:
        return {
            "file_record": self._deserialize_file_record(payload.get("file_record", {})),
            "occurrences": [self._deserialize_occurrence(item) for item in payload.get("occurrences", [])],
            "tracks": [self._deserialize_track(item) for item in payload.get("tracks", [])],
            "keyframes": [self._deserialize_keyframe(item) for item in payload.get("keyframes", [])],
            "raw_face_sizes": [float(item) for item in payload.get("raw_face_sizes", [])],
            "selected_face_sizes": [float(item) for item in payload.get("selected_face_sizes", [])],
        }

    def _deserialize_file_record(self, payload: object) -> FileRecord:
        data = payload if isinstance(payload, dict) else {}
        return FileRecord(
            path=Path(str(data.get("path", ""))),
            media_type=MediaType(str(data.get("media_type", MediaType.OTHER.value))),
            sha512=str(data.get("sha512", "")),
            size_bytes=int(data.get("size_bytes", 0)),
            discovered_at_utc=self._parse_datetime(data.get("discovered_at_utc")),
            modified_at_utc=self._parse_datetime_optional(data.get("modified_at_utc")),
            processing_error=(str(data["processing_error"]) if data.get("processing_error") else None),
            media_info_tracks=tuple(
                self._deserialize_media_info_track(item) for item in data.get("media_info_tracks", [])
            ),
            media_info_error=(str(data["media_info_error"]) if data.get("media_info_error") else None),
        )

    def _deserialize_occurrence(self, payload: object) -> FaceOccurrence:
        data = payload if isinstance(payload, dict) else {}
        return FaceOccurrence(
            occurrence_id=str(data.get("occurrence_id", "")),
            source_path=Path(str(data.get("source_path", ""))),
            sha512=str(data.get("sha512", "")),
            media_type=MediaType(str(data.get("media_type", MediaType.OTHER.value))),
            analysis_timestamp_utc=self._parse_datetime(data.get("analysis_timestamp_utc")),
            frame_index=(None if data.get("frame_index") is None else int(data.get("frame_index"))),
            frame_timestamp_seconds=(None if data.get("frame_timestamp_seconds") is None else float(data.get("frame_timestamp_seconds"))),
            bbox=self._deserialize_bbox(data.get("bbox", {})),
            detection_score=float(data.get("detection_score", 0.0)),
            crop_path=(Path(str(data["crop_path"])) if data.get("crop_path") else None),
            embedding=[float(item) for item in data.get("embedding", [])],
            context_image_path=(Path(str(data["context_image_path"])) if data.get("context_image_path") else None),
            cluster_id=(str(data["cluster_id"]) if data.get("cluster_id") else None),
            suggested_cluster_ids=[str(item) for item in data.get("suggested_cluster_ids", [])],
            track_id=(str(data["track_id"]) if data.get("track_id") else None),
            keyframe_id=(str(data["keyframe_id"]) if data.get("keyframe_id") else None),
            quality_metrics=self._deserialize_quality_metrics_optional(data.get("quality_metrics")),
            enhancement_metadata=self._deserialize_enhancement_optional(data.get("enhancement_metadata")),
            is_keyframe=bool(data.get("is_keyframe", False)),
            track_position=(None if data.get("track_position") is None else int(data.get("track_position"))),
            embedding_source=(str(data["embedding_source"]) if data.get("embedding_source") else None),
        )

    def _deserialize_track(self, payload: object) -> FaceTrack:
        data = payload if isinstance(payload, dict) else {}
        return FaceTrack(
            track_id=str(data.get("track_id", "")),
            source_path=Path(str(data.get("source_path", ""))),
            video_path=(Path(str(data["video_path"])) if data.get("video_path") else None),
            media_type=MediaType(str(data.get("media_type", MediaType.OTHER.value))),
            sha512=str(data.get("sha512", "")),
            start_frame=(None if data.get("start_frame") is None else int(data.get("start_frame"))),
            end_frame=(None if data.get("end_frame") is None else int(data.get("end_frame"))),
            start_time=(None if data.get("start_time") is None else float(data.get("start_time"))),
            end_time=(None if data.get("end_time") is None else float(data.get("end_time"))),
            occurrence_ids=[str(item) for item in data.get("occurrence_ids", [])],
            keyframe_ids=[str(item) for item in data.get("keyframe_ids", [])],
            representative_embeddings=[
                [float(value) for value in embedding]
                for embedding in data.get("representative_embeddings", [])
            ],
            average_embedding=[float(item) for item in data.get("average_embedding", [])],
            best_occurrence_id=(str(data["best_occurrence_id"]) if data.get("best_occurrence_id") else None),
            preview_path=(Path(str(data["preview_path"])) if data.get("preview_path") else None),
            top_crop_paths=[Path(str(item)) for item in data.get("top_crop_paths", [])],
            quality_statistics=self._deserialize_track_quality_statistics(data.get("quality_statistics")),
            cluster_id=(str(data["cluster_id"]) if data.get("cluster_id") else None),
            candidate_cluster_ids=[str(item) for item in data.get("candidate_cluster_ids", [])],
        )

    def _deserialize_keyframe(self, payload: object) -> KeyFrame:
        data = payload if isinstance(payload, dict) else {}
        return KeyFrame(
            keyframe_id=str(data.get("keyframe_id", "")),
            track_id=str(data.get("track_id", "")),
            occurrence_id=str(data.get("occurrence_id", "")),
            source_path=Path(str(data.get("source_path", ""))),
            frame_index=(None if data.get("frame_index") is None else int(data.get("frame_index"))),
            timestamp_seconds=(None if data.get("timestamp_seconds") is None else float(data.get("timestamp_seconds"))),
            selection_reasons=tuple(str(item) for item in data.get("selection_reasons", [])),
            quality_metrics=self._deserialize_quality_metrics_optional(data.get("quality_metrics")),
            detection_score=float(data.get("detection_score", 0.0)),
            crop_path=(Path(str(data["crop_path"])) if data.get("crop_path") else None),
            context_image_path=(Path(str(data["context_image_path"])) if data.get("context_image_path") else None),
            embedding=[float(item) for item in data.get("embedding", [])],
            preview_path=(Path(str(data["preview_path"])) if data.get("preview_path") else None),
        )

    def _deserialize_media_info_track(self, payload: object) -> MediaInfoTrack:
        data = payload if isinstance(payload, dict) else {}
        return MediaInfoTrack(
            track_type=str(data.get("track_type", "")),
            attributes=tuple(
                MediaInfoAttribute(
                    label=str(item.get("label", "")),
                    value=str(item.get("value", "")),
                )
                for item in data.get("attributes", [])
                if isinstance(item, dict)
            ),
        )

    def _deserialize_bbox(self, payload: object) -> BoundingBox:
        data = payload if isinstance(payload, dict) else {}
        return BoundingBox(
            x1=float(data.get("x1", 0.0)),
            y1=float(data.get("y1", 0.0)),
            x2=float(data.get("x2", 0.0)),
            y2=float(data.get("y2", 0.0)),
        )

    def _deserialize_quality_metrics_optional(self, payload: object) -> FaceQualityMetrics | None:
        if not isinstance(payload, dict):
            return None
        return FaceQualityMetrics(
            detection_score=float(payload.get("detection_score", 0.0)),
            sharpness=float(payload.get("sharpness", 0.0)),
            brightness=float(payload.get("brightness", 0.0)),
            illumination=float(payload.get("illumination", 0.0)),
            frontality=float(payload.get("frontality", 0.0)),
            bbox_pixels=float(payload.get("bbox_pixels", 0.0)),
            score=float(payload.get("score", 0.0)),
        )

    def _deserialize_enhancement_optional(self, payload: object) -> EnhancementMetadata | None:
        if not isinstance(payload, dict):
            return None
        return EnhancementMetadata(
            applied=bool(payload.get("applied", False)),
            strategy=str(payload.get("strategy", "none")),
            parameters={
                str(key): value
                for key, value in payload.get("parameters", {}).items()
            },
            brightness_before=(
                None if payload.get("brightness_before") is None else float(payload.get("brightness_before"))
            ),
            brightness_after=(
                None if payload.get("brightness_after") is None else float(payload.get("brightness_after"))
            ),
            note=(str(payload["note"]) if payload.get("note") else None),
        )

    def _deserialize_track_quality_statistics(self, payload: object) -> TrackQualityStatistics:
        data = payload if isinstance(payload, dict) else {}
        return TrackQualityStatistics(
            total_detections=int(data.get("total_detections", 0)),
            keyframe_count=int(data.get("keyframe_count", 0)),
            mean_detection_score=float(data.get("mean_detection_score", 0.0)),
            max_detection_score=float(data.get("max_detection_score", 0.0)),
            mean_quality_score=float(data.get("mean_quality_score", 0.0)),
            best_quality_score=float(data.get("best_quality_score", 0.0)),
            mean_sharpness=float(data.get("mean_sharpness", 0.0)),
            mean_brightness=float(data.get("mean_brightness", 0.0)),
            mean_illumination=float(data.get("mean_illumination", 0.0)),
            mean_frontality=float(data.get("mean_frontality", 0.0)),
            duration_seconds=float(data.get("duration_seconds", 0.0)),
        )

    def _parse_datetime(self, value: object) -> datetime:
        if isinstance(value, datetime):
            return value
        if isinstance(value, str) and value:
            return datetime.fromisoformat(value)
        return utc_now()

    def _parse_datetime_optional(self, value: object) -> datetime | None:
        if value in (None, ""):
            return None
        return self._parse_datetime(value)

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

    def _emit_exception(
        self,
        logger: logging.Logger,
        log_callback: LogCallback | None,
        context_message: str,
        exc: BaseException,
        *,
        include_traceback_in_callback: bool = False,
    ) -> tuple[str, str]:
        summary = summarize_exception(exc)
        traceback_text = format_exception_traceback(exc)
        logger.exception("%s | %s", context_message, summary)
        if log_callback is not None:
            log_callback(f"{context_message} | {summary}")
            if include_traceback_in_callback:
                log_callback("[Traceback] Inicio")
                log_callback(traceback_text)
                log_callback("[Traceback] Fim")
        return summary, traceback_text

    def _configuration_log_lines(self, providers: list[str]) -> list[str]:
        provider_label = ", ".join(providers) if providers else "selecao automatica"
        max_frames_label = (
            "sem limite"
            if self._config.video.max_frames_per_video is None
            else str(self._config.video.max_frames_per_video)
        )
        image_extensions = ", ".join(self._config.media.image_extensions)
        video_extensions = ", ".join(self._config.media.video_extensions)
        return [
            (
                "[Configuracao] Midias | "
                f"imagens={image_extensions} | videos={video_extensions}"
            ),
            (
                "[Configuracao] Video | "
                f"amostragem={self._config.video.sampling_interval_seconds:.2f}s | "
                f"max_quadros={max_frames_label} | "
                f"intervalo de keyframe={self._config.video.keyframe_interval_seconds:.2f}s | "
                f"mudanca significativa={self._config.video.significant_change_threshold:.2f}"
            ),
            (
                "[Configuracao] Tracking | "
                f"iou={self._config.tracking.iou_threshold:.2f} | "
                f"distancia={self._config.tracking.spatial_distance_threshold:.2f} | "
                f"embedding={self._config.tracking.embedding_similarity_threshold:.2f} | "
                f"score minimo={self._config.tracking.minimum_total_match_score:.2f} | "
                f"pesos=geo:{self._config.tracking.geometry_weight:.2f}/emb:{self._config.tracking.embedding_weight:.2f} | "
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
                f"grupo minimo={self._config.clustering.min_cluster_size} | "
                f"track minimo={self._config.clustering.min_track_size}"
            ),
            (
                "[Configuracao] Busca | "
                f"habilitada={'sim' if self._config.search.enabled else 'nao'} | "
                f"preferir_faiss={'sim' if self._config.search.prefer_faiss else 'nao'} | "
                f"coarse={self._config.search.coarse_top_k} | "
                f"refino={self._config.search.refine_top_k}"
            ),
            (
                "[Configuracao] Distribuicao | "
                f"habilitada={'sim' if self._config.distributed.enabled else 'nao'} | "
                f"execucao={self._config.distributed.execution_label} | "
                f"heartbeat={self._config.distributed.heartbeat_interval_seconds}s | "
                f"timeout_lock={self._config.distributed.stale_lock_timeout_minutes}min | "
                f"auto_finalizar={'sim' if self._config.distributed.auto_finalize else 'nao'} | "
                f"validar_parciais={'sim' if self._config.distributed.validate_partial_integrity else 'nao'} | "
                f"auto_recuperar={'sim' if self._config.distributed.auto_reprocess_invalid_partials else 'nao'}"
            ),
            (
                "[Configuracao] Aprimoramento | "
                f"pre_processamento={'sim' if self._config.enhancement.enable_preprocessing else 'nao'} | "
                f"brilho minimo={self._config.enhancement.minimum_brightness_to_enhance:.2f} | "
                f"gamma={self._config.enhancement.gamma:.2f} | "
                f"denoise={self._config.enhancement.denoise_strength}"
            ),
            (
                "[Configuracao] Relatorio | "
                f"pdf={'sim' if self._config.reporting.compile_pdf else 'nao'} | "
                f"max_tracks_por_grupo={self._config.reporting.max_tracks_per_group}"
            ),
            (
                "[Configuracao] Operacao | "
                f"copia_temporaria_local={'sim' if self._config.app.use_local_temp_copy else 'nao'}"
            ),
        ]

    def _planned_file_log_lines(self, planned_files: list[tuple[Path, MediaType]]) -> list[str]:
        if not planned_files:
            return ["[Planejamento] Nenhum arquivo localizado para processamento."]
        lines = [
            f"[Planejamento] Arquivos previstos para processamento: {len(planned_files)}"
        ]
        lines.extend(
            f"[Planejamento {index}/{len(planned_files)}] tipo={self._media_type_label(media_type)} | caminho={path}"
            for index, (path, media_type) in enumerate(planned_files, start=1)
        )
        return lines

    def _format_video_sampling_log(self, file_prefix: str, info: VideoSamplingInfo) -> str:
        fps_text = f"{info.fps:.2f}" if info.fps > 0 else "-"
        total_frames_text = "-" if info.total_frames is None else str(info.total_frames)
        duration_text = format_seconds(info.duration_seconds)
        interval_text = (
            "-"
            if info.actual_sampling_interval_seconds is None
            else f"{info.actual_sampling_interval_seconds:.2f}s"
        )
        planned_samples_text = (
            "-"
            if info.planned_sample_count is None
            else str(info.planned_sample_count)
        )
        max_samples_text = (
            "sem limite"
            if info.max_sample_count is None
            else str(info.max_sample_count)
        )
        return (
            f"{file_prefix} Video | fps={fps_text} | quadros_totais={total_frames_text} | "
            f"duracao={duration_text} | passo_amostragem={info.frame_step} quadro(s) | "
            f"intervalo_aprox={interval_text} | amostras_planejadas={planned_samples_text} | "
            f"limite_amostras={max_samples_text}"
        )

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
