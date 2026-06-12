from __future__ import annotations

import logging
import math
import shutil
import statistics
import time
from dataclasses import replace
from pathlib import Path
from typing import Callable

from inventario_faces.domain.config import AppConfig
from inventario_faces.domain.entities import (
    FaceOccurrence,
    FaceSetComparisonCalibration,
    FaceSetComparisonEntry,
    FaceSetComparisonInput,
    FaceSetComparisonMatch,
    FaceSetComparisonResult,
    FaceSetComparisonSummary,
    FaceTrack,
    KeyFrame,
    MediaType,
)
from inventario_faces.domain.protocols import (
    FaceAnalyzer,
    LogCallback,
    MediaInfoExtractor,
    ProgressCallback,
)
from inventario_faces.infrastructure.artifact_store import ArtifactStore
from inventario_faces.infrastructure.face_mesh_renderer import (
    draw_face_mesh,
    load_bgr_image,
    save_bgr_image,
)
from inventario_faces.infrastructure.logging_setup import (
    StructuredEventLogger,
    build_file_logger,
    close_file_logger,
)
from inventario_faces.services.export_service import ExportService
from inventario_faces.services.hashing_service import HashingService
from inventario_faces.services.lr_calibration import LikelihoodRatioCalibrator
from inventario_faces.services.pipeline_support import (
    cleanup_processing_input,
    configuration_log_lines,
    emit_exception,
    emit_log,
    emit_progress,
    frames_with_original_source,
    prepare_processing_input,
)
from inventario_faces.services.scanner_service import ScannerService
from inventario_faces.services.tracking_service import FaceTrackingService
from inventario_faces.services.video_service import VideoService
from inventario_faces.utils.math_utils import cosine_similarity
from inventario_faces.utils.path_utils import ensure_directory, file_io_path, safe_stem
from inventario_faces.utils.time_utils import utc_now


class FaceSetComparisonService:
    """Compara dois conjuntos de imagens faciais usando o pipeline de seleção e embedding do inventário."""

    def __init__(
        self,
        config: AppConfig,
        scanner_service: ScannerService,
        hashing_service: HashingService,
        media_service: VideoService,
        tracking_service: FaceTrackingService,
        face_analyzer_factory: Callable[[], FaceAnalyzer],
        lr_calibrator: LikelihoodRatioCalibrator,
        media_info_extractor: MediaInfoExtractor | None = None,
    ) -> None:
        self._config = config
        self._scanner_service = scanner_service
        self._hashing_service = hashing_service
        self._media_service = media_service
        self._tracking_service = tracking_service
        self._face_analyzer_factory = face_analyzer_factory
        self._lr_calibrator = lr_calibrator
        self._media_info_extractor = media_info_extractor

    def compare_face_sets(
        self,
        set_a_paths: list[Path],
        set_b_paths: list[Path],
        work_directory: Path | None = None,
        calibration_root: Path | None = None,
        calibration_model_path: Path | None = None,
        progress_callback: ProgressCallback | None = None,
        log_callback: LogCallback | None = None,
    ) -> FaceSetComparisonResult:
        """Compara dois conjuntos de imagens faciais usando o mesmo pipeline de selecao e embedding."""

        normalized_a = self._normalize_face_set_paths(set_a_paths, "Padrão")
        normalized_b = self._normalize_face_set_paths(set_b_paths, "Questionado")
        normalized_calibration_root = self._normalize_calibration_root(calibration_root)
        normalized_calibration_model_path = self._lr_calibrator._normalize_calibration_model_path(calibration_model_path)
        calibration_plan = (
            self._discover_likelihood_ratio_calibration_plan(normalized_calibration_root)
            if normalized_calibration_root is not None and normalized_calibration_model_path is None
            else []
        )
        started_at_utc = utc_now()
        work_root = self._resolve_comparison_work_directory(normalized_a, normalized_b, work_directory)
        output_root = ensure_directory(work_root / self._config.app.output_directory_name)
        run_directory = ensure_directory(output_root / f"comparison_{started_at_utc.strftime('%Y%m%d_%H%M%S')}")
        logs_directory = ensure_directory(run_directory / "logs")
        text_logger = build_file_logger(logs_directory, self._config.app.log_level)
        event_logger = StructuredEventLogger(logs_directory / "events.jsonl")
        export_service = ExportService(run_directory)
        total_inputs = len(normalized_a) + len(normalized_b)
        total_calibration_inputs = sum(len(paths) for _, paths in calibration_plan)
        total_steps = max(
            1,
            total_inputs
            + total_calibration_inputs
            + 2
            + (1 if calibration_plan or normalized_calibration_model_path is not None else 0),
        )

        try:
            emit_progress(progress_callback, 0, total_steps, "Inicializando comparacao entre conjuntos")
            emit_log(text_logger, log_callback, f"Diretorio de trabalho: {work_root}")
            emit_log(text_logger, log_callback, f"Diretorio de execucao: {run_directory}")
            emit_log(
                text_logger,
                log_callback,
                f"[Logs] Texto={logs_directory / 'run.log'} | eventos={logs_directory / 'events.jsonl'}",
            )
            emit_log(
                text_logger,
                log_callback,
                (
                    f"[Comparacao] Padrão={len(normalized_a)} imagem(ns) | "
                    f"Questionado={len(normalized_b)} imagem(ns)"
                ),
            )
            loaded_calibration: FaceSetComparisonCalibration | None = None
            if normalized_calibration_model_path is not None:
                loaded_calibration = self._lr_calibrator.load_face_set_comparison_calibration_model(
                    normalized_calibration_model_path
                )
                emit_log(
                    text_logger,
                    log_callback,
                    (
                        f"[Calibracao LR] Modelo salvo={normalized_calibration_model_path} | "
                        f"mesma_origem={len(loaded_calibration.genuine_scores)} | "
                        f"origem_distinta={len(loaded_calibration.impostor_scores)}"
                    ),
                )
                if normalized_calibration_root is not None:
                    emit_log(
                        text_logger,
                        log_callback,
                        "[Calibracao LR] A base rotulada foi ignorada porque um modelo salvo foi informado.",
                    )
            elif normalized_calibration_root is not None:
                emit_log(
                    text_logger,
                    log_callback,
                    (
                        f"[Calibracao LR] Base rotulada={normalized_calibration_root} | "
                        f"identidades={len(calibration_plan)} | imagens={total_calibration_inputs}"
                    ),
                )
            emit_log(
                text_logger,
                log_callback,
                (
                    f"[Backend facial] Inicializando modelo {self._config.face_model.model_name} "
                    "para comparacao direta entre conjuntos."
                ),
            )
            analyzer_started_at = time.perf_counter()
            analyzer = self._face_analyzer_factory()
            analyzer_elapsed = time.perf_counter() - analyzer_started_at
            providers = list(getattr(analyzer, "providers", []))
            available_providers = list(getattr(analyzer, "available_providers", []))
            using_gpu = bool(getattr(analyzer, "using_gpu", False))
            emit_log(
                text_logger,
                log_callback,
                (
                    f"[Backend facial] Modelo pronto | "
                    f"diretorio={getattr(analyzer, '_model_dir', '-')} | "
                    f"providers={', '.join(providers) if providers else 'desconhecido'} | "
                    f"disponiveis={', '.join(available_providers) if available_providers else 'desconhecido'} | "
                    f"gpu={'sim' if using_gpu else 'nao'} | "
                    f"ctx_id={self._config.face_model.ctx_id} | "
                    f"tempo={analyzer_elapsed:.2f}s"
                ),
            )

            procedure_details = tuple(
                self._comparison_procedure_lines(
                    providers=providers,
                    set_a_paths=normalized_a,
                    set_b_paths=normalized_b,
                    calibration_root=normalized_calibration_root,
                    calibration_plan=calibration_plan,
                    calibration_model_path=normalized_calibration_model_path,
                )
            )
            for line in procedure_details:
                emit_log(text_logger, log_callback, line)

            event_logger.write(
                "face_set_comparison_started",
                run_directory=run_directory,
                work_directory=work_root,
                set_a_paths=normalized_a,
                set_b_paths=normalized_b,
                calibration_root=normalized_calibration_root,
                calibration_model_path=normalized_calibration_model_path,
                configuration=self._config,
                providers=providers,
            )

            set_a_inputs: list[FaceSetComparisonInput] = []
            set_b_inputs: list[FaceSetComparisonInput] = []
            set_a_faces: list[FaceSetComparisonEntry] = []
            set_b_faces: list[FaceSetComparisonEntry] = []
            calibration_inputs: list[FaceSetComparisonInput] = []
            calibration_entries: list[FaceSetComparisonEntry] = []
            entry_sequence = [0]
            completed_inputs = 0

            for index, image_path in enumerate(normalized_a, start=1):
                input_record, entries = self._process_comparison_input(
                    set_label="A",
                    image_path=image_path,
                    index=index,
                    total_images=len(normalized_a),
                    analyzer=analyzer,
                    run_directory=run_directory,
                    export_directory=export_service.comparison_directory,
                    entry_sequence=entry_sequence,
                    event_logger=event_logger,
                    text_logger=text_logger,
                    log_callback=log_callback,
                )
                set_a_inputs.append(input_record)
                set_a_faces.extend(entries)
                completed_inputs += 1
                emit_progress(
                    progress_callback,
                    completed_inputs,
                    total_steps,
                    f"Padrão processado ({index}/{len(normalized_a)})",
                )

            for index, image_path in enumerate(normalized_b, start=1):
                input_record, entries = self._process_comparison_input(
                    set_label="B",
                    image_path=image_path,
                    index=index,
                    total_images=len(normalized_b),
                    analyzer=analyzer,
                    run_directory=run_directory,
                    export_directory=export_service.comparison_directory,
                    entry_sequence=entry_sequence,
                    event_logger=event_logger,
                    text_logger=text_logger,
                    log_callback=log_callback,
                )
                set_b_inputs.append(input_record)
                set_b_faces.extend(entries)
                completed_inputs += 1
                emit_progress(
                    progress_callback,
                    completed_inputs,
                    total_steps,
                    f"Questionado processado ({index}/{len(normalized_b)})",
                )

            if not set_a_faces:
                raise RuntimeError("Nenhuma face valida foi selecionada no grupo Padrão.")
            if not set_b_faces:
                raise RuntimeError("Nenhuma face valida foi selecionada no grupo Questionado.")

            calibration: FaceSetComparisonCalibration | None = loaded_calibration
            if calibration_plan:
                calibration_entry_sequence = [0]
                for identity_index, (identity_label, identity_paths) in enumerate(calibration_plan, start=1):
                    for image_index, image_path in enumerate(identity_paths, start=1):
                        input_record, entries = self._process_calibration_input(
                            identity_label=identity_label,
                            image_path=image_path,
                            identity_index=identity_index,
                            image_index=image_index,
                            total_images=len(identity_paths),
                            analyzer=analyzer,
                            run_directory=run_directory,
                            export_directory=export_service.comparison_directory,
                            entry_sequence=calibration_entry_sequence,
                            event_logger=event_logger,
                            text_logger=text_logger,
                            log_callback=log_callback,
                        )
                        calibration_inputs.append(input_record)
                        calibration_entries.extend(entries)
                        completed_inputs += 1
                        emit_progress(
                            progress_callback,
                            completed_inputs,
                            total_steps,
                            (
                                f"Calibracao LR: {identity_label} "
                                f"({image_index}/{len(identity_paths)})"
                            ),
                        )

            emit_progress(progress_callback, completed_inputs + 1, total_steps, "Calculando similaridades")
            matches = self._build_face_set_comparison_matches(
                set_a_faces,
                set_b_faces,
                progress_callback=progress_callback,
                progress_current=completed_inputs + 1,
                progress_total=total_steps,
                text_logger=text_logger,
                log_callback=log_callback,
                event_logger=event_logger,
            )
            if calibration_plan:
                emit_progress(
                    progress_callback,
                    completed_inputs + 2,
                    total_steps,
                    "Calibrando razao de verossimilhanca",
                )
                calibration = self._lr_calibrator._build_face_set_comparison_calibration(
                    dataset_root=normalized_calibration_root,
                    calibration_plan=calibration_plan,
                    inputs=calibration_inputs,
                    entries=calibration_entries,
                    progress_callback=progress_callback,
                    progress_current=completed_inputs + 2,
                    progress_total=total_steps,
                    text_logger=text_logger,
                    log_callback=log_callback,
                    event_logger=event_logger,
                )
                for line in self._lr_calibrator._calibration_summary_log_lines(calibration.summary):
                    emit_log(text_logger, log_callback, line)
                if calibration.summary.support_ready:
                    matches = self._lr_calibrator._apply_face_set_likelihood_ratio_calibration(
                        matches,
                        calibration,
                        progress_callback=progress_callback,
                        progress_current=completed_inputs + 2,
                        progress_total=total_steps,
                        text_logger=text_logger,
                        log_callback=log_callback,
                        event_logger=event_logger,
                    )
            elif calibration is not None:
                emit_progress(
                    progress_callback,
                    completed_inputs + 2,
                    total_steps,
                    "Aplicando modelo de calibracao LR salvo",
                )
                for line in self._lr_calibrator._calibration_summary_log_lines(calibration.summary):
                    emit_log(text_logger, log_callback, line)
                if calibration.summary.support_ready:
                    matches = self._lr_calibrator._apply_face_set_likelihood_ratio_calibration(
                        matches,
                        calibration,
                        progress_callback=progress_callback,
                        progress_current=completed_inputs + 2,
                        progress_total=total_steps,
                        text_logger=text_logger,
                        log_callback=log_callback,
                        event_logger=event_logger,
                    )
            calibration_model_export_path = export_service.comparison_directory / "face_set_comparison_calibration_model.json"
            if calibration is not None and not calibration.loaded_from_model:
                calibration = replace(calibration, model_path=calibration_model_export_path)
            summary = self._build_face_set_comparison_summary(
                set_a_inputs=set_a_inputs,
                set_b_inputs=set_b_inputs,
                matches=matches,
            )
            finished_at_utc = utc_now()
            manifest_path = export_service.comparison_directory / "face_set_comparison.json"
            result = FaceSetComparisonResult(
                run_directory=run_directory,
                started_at_utc=started_at_utc,
                finished_at_utc=finished_at_utc,
                logs_directory=logs_directory,
                export_directory=export_service.comparison_directory,
                manifest_path=manifest_path,
                set_a_inputs=set_a_inputs,
                set_b_inputs=set_b_inputs,
                set_a_faces=set_a_faces,
                set_b_faces=set_b_faces,
                matches=matches,
                summary=summary,
                calibration=calibration,
                procedure_details=procedure_details,
            )

            export_service.write_face_set_comparison_inputs_csv("face_set_comparison_inputs_a.csv", set_a_inputs)
            export_service.write_face_set_comparison_inputs_csv("face_set_comparison_inputs_b.csv", set_b_inputs)
            export_service.write_face_set_comparison_entries_csv("face_set_comparison_entries_a.csv", set_a_faces)
            export_service.write_face_set_comparison_entries_csv("face_set_comparison_entries_b.csv", set_b_faces)
            if calibration is not None:
                if calibration.inputs:
                    export_service.write_face_set_comparison_inputs_csv(
                        "face_set_comparison_calibration_inputs.csv",
                        calibration.inputs,
                    )
                if calibration.entries:
                    export_service.write_face_set_comparison_entries_csv(
                        "face_set_comparison_calibration_entries.csv",
                        calibration.entries,
                    )
                export_service.write_face_set_comparison_calibration_scores_csv(calibration)
                self._lr_calibrator.save_face_set_comparison_calibration_model(calibration, calibration_model_export_path)
            export_service.write_face_set_comparison_matches_csv(result)
            export_service.write_face_set_comparison_summary_text(result)
            export_service.write_face_set_comparison_json(result)

            for line in self._comparison_summary_log_lines(summary):
                emit_log(text_logger, log_callback, line)

            event_logger.write(
                "face_set_comparison_finished",
                run_directory=run_directory,
                set_a_faces=len(set_a_faces),
                set_b_faces=len(set_b_faces),
                total_pair_comparisons=summary.total_pair_comparisons,
                assignment_matches=summary.assignment_matches,
                candidate_matches=summary.candidate_matches,
                likelihood_ratio_calibrated=summary.likelihood_ratio_calibrated,
                export_directory=export_service.comparison_directory,
                manifest_path=manifest_path,
            )
            emit_progress(progress_callback, total_steps, total_steps, "Comparacao concluida")
            return result
        except Exception as exc:
            error_summary, traceback_text = emit_exception(
                text_logger,
                log_callback,
                "[Comparacao] Falha fatal",
                exc,
                include_traceback_in_callback=True,
            )
            event_logger.write(
                "face_set_comparison_failed",
                error=error_summary,
                error_type=type(exc).__name__,
                traceback=traceback_text,
                set_a_paths=normalized_a,
                set_b_paths=normalized_b,
            )
            raise
        finally:
            close_file_logger(text_logger)

    def _normalize_face_set_paths(self, paths: list[Path], set_name: str) -> list[Path]:
        normalized: list[Path] = []
        seen: set[Path] = set()
        if not paths:
            raise ValueError(f"{set_name} precisa conter ao menos uma imagem.")
        for raw_path in paths:
            candidate = Path(raw_path).resolve()
            if not candidate.exists():
                raise FileNotFoundError(f"Imagem nao encontrada em {set_name}: {candidate}")
            if candidate in seen:
                continue
            seen.add(candidate)
            normalized.append(candidate)
        if not normalized:
            raise ValueError(f"{set_name} precisa conter ao menos uma imagem valida.")
        return normalized

    def _resolve_comparison_work_directory(
        self,
        set_a_paths: list[Path],
        set_b_paths: list[Path],
        work_directory: Path | None,
    ) -> Path:
        if work_directory is not None:
            return Path(work_directory).resolve()
        return set_a_paths[0].parent.resolve() if set_a_paths else set_b_paths[0].parent.resolve()

    def _comparison_group_label(self, set_label: str) -> str:
        if set_label == "A":
            return "Padrão"
        if set_label == "B":
            return "Questionado"
        if set_label == "CAL":
            return "Calibração LR"
        return set_label

    def _normalize_calibration_root(self, calibration_root: Path | None) -> Path | None:
        if calibration_root is None:
            return None
        candidate = Path(calibration_root).resolve()
        if not candidate.exists():
            raise FileNotFoundError(f"Base de calibracao nao encontrada: {candidate}")
        if not candidate.is_dir():
            raise NotADirectoryError(f"Base de calibracao invalida: {candidate}")
        return candidate

    def _discover_likelihood_ratio_calibration_plan(
        self,
        calibration_root: Path,
    ) -> list[tuple[str, list[Path]]]:
        identity_directories = sorted(
            path
            for path in calibration_root.iterdir()
            if path.is_dir()
        )
        if not identity_directories:
            raise ValueError(
                "A base de calibracao precisa conter subdiretorios imediatos, um por identidade rotulada."
            )

        plan: list[tuple[str, list[Path]]] = []
        for directory in identity_directories:
            image_paths = [
                path.resolve()
                for path in self._scanner_service.scan(directory)
                if self._scanner_service.classify(path) == MediaType.IMAGE
            ]
            if image_paths:
                plan.append((directory.name, image_paths))

        if len(plan) < 2:
            raise ValueError(
                "A base de calibracao precisa conter ao menos duas identidades com imagens faciais."
            )
        return plan

    def _comparison_procedure_lines(
        self,
        *,
        providers: list[str],
        set_a_paths: list[Path],
        set_b_paths: list[Path],
        calibration_root: Path | None = None,
        calibration_plan: list[tuple[str, list[Path]]] | None = None,
        calibration_model_path: Path | None = None,
    ) -> list[str]:
        lines = [
            "[Procedimento] Cada imagem e submetida ao mesmo fluxo do inventario: preparo de entrada, deteccao, filtros, tracking e embeddings.",
            (
                "[Procedimento] A comparacao final e par-a-par entre as faces selecionadas dos grupos Padrão e Questionado, "
                "usando similaridade cosseno dos embeddings."
            ),
            (
                "[Procedimento] As classificacoes usam os limiares do agrupamento atual: "
                f"candidato>={self._config.clustering.candidate_similarity:.2f} e "
                f"atribuicao>={self._config.clustering.assignment_similarity:.2f}."
            ),
            f"[Planejamento] Padrão possui {len(set_a_paths)} imagem(ns).",
            f"[Planejamento] Questionado possui {len(set_b_paths)} imagem(ns).",
        ]
        lines.extend(
            f"[Planejamento Padrão {index}/{len(set_a_paths)}] caminho={path}"
            for index, path in enumerate(set_a_paths, start=1)
        )
        lines.extend(
            f"[Planejamento Questionado {index}/{len(set_b_paths)}] caminho={path}"
            for index, path in enumerate(set_b_paths, start=1)
        )
        if calibration_model_path is not None:
            lines.append(
                (
                    "[Calibracao LR] Um modelo salvo foi carregado e reutilizado sem reprocessar a base rotulada. "
                    "A LR reaproveita os scores de mesma origem e de origem distinta ja estimados."
                )
            )
            lines.append(f"[Calibracao LR] Modelo salvo={calibration_model_path}")
            if calibration_root is not None:
                lines.append(
                    "[Calibracao LR] A base rotulada informada foi ignorada porque o modelo salvo tem prioridade."
                )
        elif calibration_root is not None and calibration_plan:
            lines.append(
                (
                    "[Calibracao LR] A base rotulada e processada com o mesmo pipeline. "
                    "Cada subdiretorio imediato representa uma identidade para gerar scores Padrão/Questionado de mesma origem e de origem distinta."
                )
            )
            lines.append(
                (
                    f"[Calibracao LR] Base={calibration_root} | identidades={len(calibration_plan)} | "
                    f"imagens={sum(len(paths) for _, paths in calibration_plan)} | "
                    f"metodo={self._config.likelihood_ratio.density_estimator}"
                )
            )
            lines.extend(
                (
                    f"[Calibracao LR {index}/{len(calibration_plan)}] identidade={identity_label} | "
                    f"imagens={len(identity_paths)}"
                )
                for index, (identity_label, identity_paths) in enumerate(calibration_plan, start=1)
            )
            lines.append(
                (
                    "[Calibracao LR] A razao de verossimilhanca e calculada como "
                    "p(score|mesma_origem) / p(score|origem_diferente), com estimador nao parametrico "
                    "de densidade estabilizado por piso uniforme."
                )
            )
        lines.extend(configuration_log_lines(self._config, providers))
        return lines

    def _process_comparison_input(
        self,
        *,
        set_label: str,
        image_path: Path,
        index: int,
        total_images: int,
        analyzer: FaceAnalyzer,
        run_directory: Path,
        export_directory: Path,
        entry_sequence: list[int],
        event_logger: StructuredEventLogger,
        text_logger: logging.Logger,
        log_callback: LogCallback | None,
    ) -> tuple[FaceSetComparisonInput, list[FaceSetComparisonEntry]]:
        file_prefix = f"[Comparacao {self._comparison_group_label(set_label)} {index}/{total_images}]"
        emit_log(text_logger, log_callback, f"{file_prefix} Iniciando imagem {image_path}")
        source_copy_path: Path | None = None
        sha512 = ""
        current_stage = "preparacao de entrada"
        processing_error: str | None = None
        entries: list[FaceSetComparisonEntry] = []

        try:
            current_stage = "copia de exportacao"
            source_copy_path = self._copy_comparison_input(export_directory, set_label, index, image_path)
            current_stage = "preparacao de entrada"
            processing_path, sha512, cleanup_path = prepare_processing_input(
                config=self._config,
                hashing_service=self._hashing_service,
                file_path=image_path,
                media_type=MediaType.IMAGE,
                file_prefix=file_prefix,
                text_logger=text_logger,
                log_callback=log_callback,
            )
            try:
                current_stage = "carregamento da imagem"
                frame = self._media_service.load_image(processing_path)
                current_stage = "deteccao, tracking e embeddings"
                tracking_result = self._tracking_service.process_media(
                    source_path=image_path,
                    sha512=sha512,
                    media_type=MediaType.IMAGE,
                    frames=frames_with_original_source([frame], image_path),
                    analyzer=analyzer,
                    artifact_store=ArtifactStore(
                        run_directory
                        / "comparison_artifacts"
                        / f"set_{set_label.lower()}"
                        / f"{index:04d}_{safe_stem(image_path.stem)}"
                    ),
                    id_namespace=f"C{set_label}{index:04d}",
                    event_callback=lambda event, fields: event_logger.write(event, **fields),
                    text_callback=lambda message: emit_log(text_logger, log_callback, message),
                )
            finally:
                cleanup_processing_input(cleanup_path)

            input_record = FaceSetComparisonInput(
                set_label=set_label,
                source_path=image_path,
                sha512=sha512,
                detected_faces=tracking_result.raw_detection_count,
                selected_faces=tracking_result.selected_detection_count,
                tracks=len(tracking_result.tracks),
                keyframes=len(tracking_result.keyframes),
                export_source_copy=source_copy_path,
            )
            emit_log(
                text_logger,
                log_callback,
                (
                    f"{file_prefix} Resultado | deteccoes={tracking_result.raw_detection_count} | "
                    f"selecionadas={tracking_result.selected_detection_count} | "
                    f"tracks={len(tracking_result.tracks)} | keyframes={len(tracking_result.keyframes)}"
                ),
            )

            for track in tracking_result.tracks:
                occurrence = self._best_track_occurrence(track, tracking_result.occurrences)
                if occurrence is None:
                    continue
                keyframe = next((item for item in tracking_result.keyframes if item.track_id == track.track_id), None)
                entry_sequence[0] += 1
                entry_id = f"C{set_label}_{entry_sequence[0]:06d}"
                mesh_crop_path, mesh_context_path = self._render_comparison_mesh_artifacts(
                    export_directory=export_directory,
                    set_label=set_label,
                    entry_id=entry_id,
                    occurrence=occurrence,
                )
                entries.append(
                    self._create_face_set_comparison_entry(
                        entry_id=entry_id,
                        set_label=set_label,
                        track=track,
                        occurrence=occurrence,
                        keyframe=keyframe,
                        mesh_crop_path=mesh_crop_path,
                        mesh_context_path=mesh_context_path,
                    )
                )

            event_logger.write(
                "comparison_input_processed",
                set_label=set_label,
                source_path=image_path,
                sha512=sha512,
                detected_faces=input_record.detected_faces,
                selected_faces=input_record.selected_faces,
                tracks=input_record.tracks,
                keyframes=input_record.keyframes,
                exported_source_copy=source_copy_path,
            )
            return input_record, entries
        except Exception as exc:
            processing_error, traceback_text = emit_exception(
                text_logger,
                log_callback,
                f"{file_prefix} Erro de processamento | etapa={current_stage} | arquivo={image_path}",
                exc,
            )
            event_logger.write(
                "comparison_input_failed",
                set_label=set_label,
                source_path=image_path,
                sha512=sha512,
                error=processing_error,
                error_type=type(exc).__name__,
                error_stage=current_stage,
                traceback=traceback_text,
            )
            return (
                FaceSetComparisonInput(
                    set_label=set_label,
                    source_path=image_path,
                    sha512=sha512,
                    detected_faces=0,
                    selected_faces=0,
                    tracks=0,
                    keyframes=0,
                    processing_error=processing_error,
                    export_source_copy=source_copy_path,
                ),
                [],
            )

    def _copy_comparison_input(
        self,
        export_directory: Path,
        set_label: str,
        index: int,
        source_path: Path,
    ) -> Path:
        target_directory = ensure_directory(export_directory / "inputs" / f"set_{set_label.lower()}")
        target_path = target_directory / f"{index:04d}_{safe_stem(source_path.stem)}{source_path.suffix.lower()}"
        shutil.copy2(file_io_path(source_path), file_io_path(target_path))
        return target_path

    def _process_calibration_input(
        self,
        *,
        identity_label: str,
        image_path: Path,
        identity_index: int,
        image_index: int,
        total_images: int,
        analyzer: FaceAnalyzer,
        run_directory: Path,
        export_directory: Path,
        entry_sequence: list[int],
        event_logger: StructuredEventLogger,
        text_logger: logging.Logger,
        log_callback: LogCallback | None,
    ) -> tuple[FaceSetComparisonInput, list[FaceSetComparisonEntry]]:
        file_prefix = f"[Calibracao LR {identity_label} {image_index}/{total_images}]"
        emit_log(text_logger, log_callback, f"{file_prefix} Iniciando imagem {image_path}")
        source_copy_path: Path | None = None
        sha512 = ""
        current_stage = "preparacao de entrada"
        processing_error: str | None = None
        entries: list[FaceSetComparisonEntry] = []

        try:
            current_stage = "copia de exportacao"
            source_copy_path = self._copy_calibration_input(
                export_directory=export_directory,
                identity_label=identity_label,
                identity_index=identity_index,
                image_index=image_index,
                source_path=image_path,
            )
            current_stage = "preparacao de entrada"
            processing_path, sha512, cleanup_path = prepare_processing_input(
                config=self._config,
                hashing_service=self._hashing_service,
                file_path=image_path,
                media_type=MediaType.IMAGE,
                file_prefix=file_prefix,
                text_logger=text_logger,
                log_callback=log_callback,
            )
            try:
                current_stage = "carregamento da imagem"
                frame = self._media_service.load_image(processing_path)
                current_stage = "deteccao, tracking e embeddings"
                tracking_result = self._tracking_service.process_media(
                    source_path=image_path,
                    sha512=sha512,
                    media_type=MediaType.IMAGE,
                    frames=frames_with_original_source([frame], image_path),
                    analyzer=analyzer,
                    artifact_store=ArtifactStore(
                        run_directory
                        / "comparison_artifacts"
                        / "calibration"
                        / f"{identity_index:03d}_{safe_stem(identity_label)}"
                        / f"{image_index:04d}_{safe_stem(image_path.stem)}"
                    ),
                    id_namespace=f"CLR{identity_index:03d}_{image_index:04d}",
                    event_callback=lambda event, fields: event_logger.write(event, **fields),
                    text_callback=lambda message: emit_log(text_logger, log_callback, message),
                )
            finally:
                cleanup_processing_input(cleanup_path)

            input_record = FaceSetComparisonInput(
                set_label="CAL",
                source_path=image_path,
                sha512=sha512,
                detected_faces=tracking_result.raw_detection_count,
                selected_faces=tracking_result.selected_detection_count,
                tracks=len(tracking_result.tracks),
                keyframes=len(tracking_result.keyframes),
                identity_label=identity_label,
                export_source_copy=source_copy_path,
            )
            emit_log(
                text_logger,
                log_callback,
                (
                    f"{file_prefix} Resultado | deteccoes={tracking_result.raw_detection_count} | "
                    f"selecionadas={tracking_result.selected_detection_count} | "
                    f"tracks={len(tracking_result.tracks)} | keyframes={len(tracking_result.keyframes)}"
                ),
            )

            for track in tracking_result.tracks:
                occurrence = self._best_track_occurrence(track, tracking_result.occurrences)
                if occurrence is None:
                    continue
                keyframe = next((item for item in tracking_result.keyframes if item.track_id == track.track_id), None)
                entry_sequence[0] += 1
                entry_id = f"CLR_{entry_sequence[0]:06d}"
                entries.append(
                    self._create_face_set_comparison_entry(
                        entry_id=entry_id,
                        set_label="CAL",
                        track=track,
                        occurrence=occurrence,
                        keyframe=keyframe,
                        mesh_crop_path=None,
                        mesh_context_path=None,
                        identity_label=identity_label,
                    )
                )

            event_logger.write(
                "comparison_calibration_input_processed",
                identity_label=identity_label,
                source_path=image_path,
                sha512=sha512,
                detected_faces=input_record.detected_faces,
                selected_faces=input_record.selected_faces,
                tracks=input_record.tracks,
                keyframes=input_record.keyframes,
                exported_source_copy=source_copy_path,
            )
            return input_record, entries
        except Exception as exc:
            processing_error, traceback_text = emit_exception(
                text_logger,
                log_callback,
                f"{file_prefix} Erro de processamento | etapa={current_stage} | arquivo={image_path}",
                exc,
            )
            event_logger.write(
                "comparison_calibration_input_failed",
                identity_label=identity_label,
                source_path=image_path,
                sha512=sha512,
                error=processing_error,
                error_type=type(exc).__name__,
                error_stage=current_stage,
                traceback=traceback_text,
            )
            return (
                FaceSetComparisonInput(
                    set_label="CAL",
                    source_path=image_path,
                    sha512=sha512,
                    detected_faces=0,
                    selected_faces=0,
                    tracks=0,
                    keyframes=0,
                    identity_label=identity_label,
                    processing_error=processing_error,
                    export_source_copy=source_copy_path,
                ),
                [],
            )

    def _copy_calibration_input(
        self,
        *,
        export_directory: Path,
        identity_label: str,
        identity_index: int,
        image_index: int,
        source_path: Path,
    ) -> Path:
        target_directory = ensure_directory(
            export_directory
            / "calibration"
            / "inputs"
            / f"{identity_index:03d}_{safe_stem(identity_label)}"
        )
        target_path = (
            target_directory
            / f"{image_index:04d}_{safe_stem(source_path.stem)}{source_path.suffix.lower()}"
        )
        shutil.copy2(file_io_path(source_path), file_io_path(target_path))
        return target_path

    def _best_track_occurrence(
        self,
        track: FaceTrack,
        occurrences: list[FaceOccurrence],
    ) -> FaceOccurrence | None:
        occurrences_by_id = {item.occurrence_id: item for item in occurrences}
        if track.best_occurrence_id is not None and track.best_occurrence_id in occurrences_by_id:
            return occurrences_by_id[track.best_occurrence_id]
        for occurrence_id in track.occurrence_ids:
            if occurrence_id in occurrences_by_id:
                return occurrences_by_id[occurrence_id]
        return None

    def _render_comparison_mesh_artifacts(
        self,
        *,
        export_directory: Path,
        set_label: str,
        entry_id: str,
        occurrence: FaceOccurrence,
    ) -> tuple[Path | None, Path | None]:
        mesh_directory = ensure_directory(export_directory / "mesh" / f"set_{set_label.lower()}")
        mesh_crop_path: Path | None = None
        mesh_context_path: Path | None = None

        if occurrence.crop_path is not None:
            crop_mesh = draw_face_mesh(
                load_bgr_image(occurrence.crop_path),
                occurrence.biometric_landmarks,
                bbox=occurrence.bbox,
                translate=(-occurrence.bbox.x1, -occurrence.bbox.y1),
                draw_bbox=False,
            )
            mesh_crop_path = mesh_directory / f"{entry_id}_crop_mesh.jpg"
            save_bgr_image(mesh_crop_path, crop_mesh)

        if occurrence.context_image_path is not None:
            context_mesh = draw_face_mesh(
                load_bgr_image(occurrence.source_path),
                occurrence.biometric_landmarks,
                bbox=occurrence.bbox,
                draw_bbox=False,
            )
            mesh_context_path = mesh_directory / f"{entry_id}_context_mesh.jpg"
            save_bgr_image(mesh_context_path, context_mesh)

        return mesh_crop_path, mesh_context_path

    def _create_face_set_comparison_entry(
        self,
        *,
        entry_id: str,
        set_label: str,
        track: FaceTrack,
        occurrence: FaceOccurrence,
        keyframe: KeyFrame | None,
        mesh_crop_path: Path | None,
        mesh_context_path: Path | None,
        identity_label: str | None = None,
    ) -> FaceSetComparisonEntry:
        quality = occurrence.quality_metrics
        return FaceSetComparisonEntry(
            entry_id=entry_id,
            set_label=set_label,
            source_path=occurrence.source_path,
            sha512=occurrence.sha512,
            track_id=track.track_id,
            occurrence_id=occurrence.occurrence_id,
            keyframe_id=keyframe.keyframe_id if keyframe is not None else None,
            bbox=occurrence.bbox,
            detection_score=occurrence.detection_score,
            quality_score=quality.score if quality is not None else None,
            sharpness=quality.sharpness if quality is not None else None,
            brightness=quality.brightness if quality is not None else None,
            illumination=quality.illumination if quality is not None else None,
            frontality=quality.frontality if quality is not None else None,
            identity_label=identity_label,
            embedding=list(occurrence.embedding),
            embedding_dimension=len(occurrence.embedding),
            embedding_source=occurrence.embedding_source,
            crop_path=occurrence.crop_path,
            context_image_path=occurrence.context_image_path,
            mesh_crop_path=mesh_crop_path,
            mesh_context_path=mesh_context_path,
            selection_reasons=keyframe.selection_reasons if keyframe is not None else (),
            biometric_landmarks=occurrence.biometric_landmarks,
        )

    def _build_face_set_comparison_matches(
        self,
        set_a_faces: list[FaceSetComparisonEntry],
        set_b_faces: list[FaceSetComparisonEntry],
        *,
        progress_callback: ProgressCallback | None = None,
        progress_current: int | None = None,
        progress_total: int | None = None,
        text_logger: logging.Logger | None = None,
        log_callback: LogCallback | None = None,
        event_logger: StructuredEventLogger | None = None,
    ) -> list[FaceSetComparisonMatch]:
        ranked_pairs: list[FaceSetComparisonMatch] = []
        expected_pairs = len(set_a_faces) * len(set_b_faces)
        heartbeat_interval_pairs = 5000
        heartbeat_interval_seconds = 5.0
        last_heartbeat = time.monotonic()
        if text_logger is not None:
            emit_log(
                text_logger,
                log_callback,
                (
                    f"[Comparacao] Similaridades | faces_padrao={len(set_a_faces)} | "
                    f"faces_questionado={len(set_b_faces)} | pares_previstos={expected_pairs}"
                ),
            )
        if event_logger is not None:
            event_logger.write(
                "face_set_comparison_matching_started",
                set_a_faces=len(set_a_faces),
                set_b_faces=len(set_b_faces),
                expected_pairs=expected_pairs,
            )

        compared_pairs = 0
        for left in set_a_faces:
            for right in set_b_faces:
                similarity = cosine_similarity(left.embedding, right.embedding)
                ranked_pairs.append(
                    FaceSetComparisonMatch(
                        rank=0,
                        left_entry_id=left.entry_id,
                        right_entry_id=right.entry_id,
                        left_track_id=left.track_id,
                        right_track_id=right.track_id,
                        similarity=similarity,
                        classification=self._classify_comparison_similarity(similarity),
                        left_quality_score=left.quality_score,
                        right_quality_score=right.quality_score,
                    )
                )
                compared_pairs += 1
                should_emit_heartbeat = (
                    compared_pairs < expected_pairs
                    and (
                        compared_pairs % heartbeat_interval_pairs == 0
                        or (time.monotonic() - last_heartbeat) >= heartbeat_interval_seconds
                    )
                )
                if should_emit_heartbeat:
                    if (
                        progress_callback is not None
                        and progress_current is not None
                        and progress_total is not None
                    ):
                        emit_progress(
                            progress_callback,
                            progress_current,
                            progress_total,
                            f"Calculando similaridades ({compared_pairs}/{expected_pairs} pares)",
                        )
                    if text_logger is not None:
                        emit_log(
                            text_logger,
                            log_callback,
                            f"[Comparacao] Similaridades em andamento | pares={compared_pairs}/{expected_pairs}",
                        )
                    if event_logger is not None:
                        event_logger.write(
                            "face_set_comparison_matching_progress",
                            processed_pairs=compared_pairs,
                            expected_pairs=expected_pairs,
                        )
                    last_heartbeat = time.monotonic()
        ranked_pairs.sort(key=lambda item: item.similarity, reverse=True)
        if text_logger is not None:
            emit_log(
                text_logger,
                log_callback,
                f"[Comparacao] Similaridades concluidas | pares={expected_pairs} | ranking={len(ranked_pairs)}",
            )
        if event_logger is not None:
            event_logger.write(
                "face_set_comparison_matching_completed",
                processed_pairs=expected_pairs,
                ranked_pairs=len(ranked_pairs),
            )
        return [
            replace(item, rank=index)
            for index, item in enumerate(ranked_pairs, start=1)
        ]

    def _classify_comparison_similarity(self, similarity: float) -> str:
        if similarity >= self._config.clustering.assignment_similarity:
            return "assignment"
        if similarity >= self._config.clustering.candidate_similarity:
            return "candidate"
        return "below_threshold"

    def _build_face_set_comparison_summary(
        self,
        *,
        set_a_inputs: list[FaceSetComparisonInput],
        set_b_inputs: list[FaceSetComparisonInput],
        matches: list[FaceSetComparisonMatch],
    ) -> FaceSetComparisonSummary:
        similarity_values = [item.similarity for item in matches]
        calibrated_values = [item.log10_likelihood_ratio for item in matches if item.log10_likelihood_ratio is not None]
        q1_similarity: float | None = None
        q3_similarity: float | None = None
        mean_confidence_low: float | None = None
        mean_confidence_high: float | None = None
        mean_similarity = statistics.fmean(similarity_values) if similarity_values else None
        median_similarity = statistics.median(similarity_values) if similarity_values else None
        stddev_similarity = (
            statistics.pstdev(similarity_values)
            if len(similarity_values) > 1
            else 0.0 if similarity_values else None
        )
        if similarity_values:
            if len(similarity_values) == 1:
                q1_similarity = similarity_values[0]
                q3_similarity = similarity_values[0]
                mean_confidence_low = similarity_values[0]
                mean_confidence_high = similarity_values[0]
            else:
                quartiles = statistics.quantiles(similarity_values, n=4, method="inclusive")
                q1_similarity = quartiles[0]
                q3_similarity = quartiles[2]
                sample_stddev = statistics.stdev(similarity_values)
                confidence_margin = 1.96 * sample_stddev / math.sqrt(len(similarity_values))
                mean_confidence_low = max(-1.0, min(1.0, mean_similarity - confidence_margin))
                mean_confidence_high = max(-1.0, min(1.0, mean_similarity + confidence_margin))
        return FaceSetComparisonSummary(
            set_a_images=len(set_a_inputs),
            set_b_images=len(set_b_inputs),
            set_a_detected_faces=sum(item.detected_faces for item in set_a_inputs),
            set_b_detected_faces=sum(item.detected_faces for item in set_b_inputs),
            set_a_selected_faces=sum(item.selected_faces for item in set_a_inputs),
            set_b_selected_faces=sum(item.selected_faces for item in set_b_inputs),
            set_a_images_without_faces=len([item for item in set_a_inputs if item.selected_faces == 0]),
            set_b_images_without_faces=len([item for item in set_b_inputs if item.selected_faces == 0]),
            total_pair_comparisons=len(matches),
            assignment_matches=len([item for item in matches if item.classification == "assignment"]),
            candidate_matches=len([item for item in matches if item.classification == "candidate"]),
            best_similarity=max(similarity_values) if similarity_values else None,
            worst_similarity=min(similarity_values) if similarity_values else None,
            mean_similarity=mean_similarity,
            median_similarity=median_similarity,
            stddev_similarity=stddev_similarity,
            q1_similarity=q1_similarity,
            q3_similarity=q3_similarity,
            mean_confidence_low=mean_confidence_low,
            mean_confidence_high=mean_confidence_high,
            candidate_threshold=self._config.clustering.candidate_similarity,
            assignment_threshold=self._config.clustering.assignment_similarity,
            likelihood_ratio_calibrated=bool(calibrated_values),
            calibrated_matches=len(calibrated_values),
            extrapolated_matches=len(
                [item for item in matches if item.outside_calibration_support]
            ),
            mean_log10_likelihood_ratio=(
                statistics.fmean(calibrated_values) if calibrated_values else None
            ),
            median_log10_likelihood_ratio=(
                statistics.median(calibrated_values) if calibrated_values else None
            ),
            min_log10_likelihood_ratio=min(calibrated_values) if calibrated_values else None,
            max_log10_likelihood_ratio=max(calibrated_values) if calibrated_values else None,
        )

    def _comparison_summary_log_lines(self, summary: FaceSetComparisonSummary) -> list[str]:
        lines = [
            (
                f"[Comparacao] Faces selecionadas | padrao={summary.set_a_selected_faces} | "
                f"questionado={summary.set_b_selected_faces}"
            ),
            (
                f"[Comparacao] Estatisticas | pares={summary.total_pair_comparisons} | "
                f"atribuicao={summary.assignment_matches} | candidatos={summary.candidate_matches}"
            ),
            (
                f"[Comparacao] Similaridade | melhor={summary.best_similarity if summary.best_similarity is not None else '-'} | "
                f"media={summary.mean_similarity if summary.mean_similarity is not None else '-'} | "
                f"mediana={summary.median_similarity if summary.median_similarity is not None else '-'} | "
                f"desvio={summary.stddev_similarity if summary.stddev_similarity is not None else '-'} | "
                f"ic95%=[{summary.mean_confidence_low if summary.mean_confidence_low is not None else '-'}, "
                f"{summary.mean_confidence_high if summary.mean_confidence_high is not None else '-'}]"
            ),
        ]
        if summary.likelihood_ratio_calibrated:
            lines.append(
                (
                    f"[Comparacao] LR calibrada | pares={summary.calibrated_matches} | "
                    f"media_log10lr={summary.mean_log10_likelihood_ratio if summary.mean_log10_likelihood_ratio is not None else '-'} | "
                    f"mediana_log10lr={summary.median_log10_likelihood_ratio if summary.median_log10_likelihood_ratio is not None else '-'} | "
                    f"intervalo=[{summary.min_log10_likelihood_ratio if summary.min_log10_likelihood_ratio is not None else '-'}, "
                    f"{summary.max_log10_likelihood_ratio if summary.max_log10_likelihood_ratio is not None else '-'}]"
                )
            )
            if summary.extrapolated_matches:
                lines.append(
                    (
                        f"[Comparacao][ALERTA] {summary.extrapolated_matches} par(es) com escore fora do "
                        "intervalo de scores usado na calibracao; o LR desses pares e extrapolado."
                    )
                )
        return lines
