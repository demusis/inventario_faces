from __future__ import annotations

import json
import logging
import math
import time
from collections.abc import Iterable
from dataclasses import replace
from pathlib import Path

import numpy as np

from inventario_faces.domain.config import AppConfig, LikelihoodRatioSettings
from inventario_faces.domain.entities import (
    FaceSetComparisonCalibration,
    FaceSetComparisonCalibrationSummary,
    FaceSetComparisonEntry,
    FaceSetComparisonInput,
    FaceSetComparisonMatch,
)
from inventario_faces.domain.protocols import LogCallback, ProgressCallback
from inventario_faces.infrastructure.logging_setup import StructuredEventLogger
from inventario_faces.services.pipeline_support import (
    emit_log,
    emit_progress,
    write_json_atomic,
)
from inventario_faces.utils.density_utils import (
    fit_score_density_model,
    score_density_method_label,
    stabilize_score_density,
)
from inventario_faces.utils.math_utils import cosine_similarity
from inventario_faces.utils.time_utils import utc_now


class LikelihoodRatioCalibrator:
    """Calibra e aplica a razão de verossimilhança da comparação entre conjuntos faciais."""

    def __init__(self, config: AppConfig) -> None:
        self._config = config

    def _normalize_calibration_model_path(self, calibration_model_path: Path | None) -> Path | None:
        if calibration_model_path is None:
            return None
        candidate = Path(calibration_model_path).resolve()
        if not candidate.exists():
            raise FileNotFoundError(f"Modelo de calibracao nao encontrado: {candidate}")
        if not candidate.is_file():
            raise IsADirectoryError(f"Modelo de calibracao invalido: {candidate}")
        return candidate

    def save_face_set_comparison_calibration_model(
        self,
        calibration: FaceSetComparisonCalibration,
        output_path: Path,
    ) -> Path:
        candidate = Path(output_path).resolve()
        if candidate.exists() and candidate.is_dir():
            raise IsADirectoryError(f"O destino do modelo de calibracao precisa ser um arquivo: {candidate}")
        payload = {
            "schema": "face_set_comparison_calibration_model",
            "version": 1,
            "saved_at_utc": utc_now(),
            "summary": calibration.summary,
            "genuine_scores": calibration.genuine_scores,
            "impostor_scores": calibration.impostor_scores,
            "procedure_details": calibration.procedure_details,
            "settings_snapshot": calibration.settings_snapshot,
            "loaded_from_model": calibration.loaded_from_model,
            "source_model_path": calibration.model_path,
        }
        write_json_atomic(candidate, payload)
        return candidate

    def load_face_set_comparison_calibration_model(
        self,
        model_path: Path,
    ) -> FaceSetComparisonCalibration:
        candidate = self._normalize_calibration_model_path(model_path)
        if candidate is None:
            raise ValueError("O caminho do modelo de calibracao precisa ser informado.")
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Modelo de calibracao invalido: {candidate}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"Modelo de calibracao invalido: {candidate}")
        if payload.get("schema") != "face_set_comparison_calibration_model":
            raise ValueError(f"Arquivo nao reconhecido como modelo de calibracao LR: {candidate}")

        summary_payload = payload.get("summary")
        if not isinstance(summary_payload, dict):
            raise ValueError(f"Resumo ausente no modelo de calibracao: {candidate}")

        def _optional_float(value: object) -> float | None:
            if value in (None, ""):
                return None
            return float(value)

        def _optional_text(value: object) -> str | None:
            if value in (None, ""):
                return None
            return str(value)

        def _float_list(field_name: str) -> list[float]:
            values = payload.get(field_name, [])
            if not isinstance(values, list):
                raise ValueError(f"Campo invalido no modelo de calibracao: {field_name}")
            return [float(item) for item in values]

        procedure_payload = payload.get("procedure_details", [])
        if not isinstance(procedure_payload, list):
            raise ValueError("Campo invalido no modelo de calibracao: procedure_details")

        dataset_root_value = summary_payload.get("dataset_root")
        if dataset_root_value in (None, ""):
            raise ValueError(f"Modelo de calibracao sem dataset_root: {candidate}")

        summary = FaceSetComparisonCalibrationSummary(
            dataset_root=Path(str(dataset_root_value)),
            identity_count=int(summary_payload.get("identity_count", 0)),
            processed_identities=int(summary_payload.get("processed_identities", 0)),
            input_images=int(summary_payload.get("input_images", 0)),
            processed_images=int(summary_payload.get("processed_images", 0)),
            selected_faces=int(summary_payload.get("selected_faces", 0)),
            identities_with_selected_faces=int(summary_payload.get("identities_with_selected_faces", 0)),
            genuine_pair_total=int(summary_payload.get("genuine_pair_total", 0)),
            impostor_pair_total=int(summary_payload.get("impostor_pair_total", 0)),
            genuine_score_count=int(summary_payload.get("genuine_score_count", 0)),
            impostor_score_count=int(summary_payload.get("impostor_score_count", 0)),
            support_ready=bool(summary_payload.get("support_ready", False)),
            support_note=_optional_text(summary_payload.get("support_note")),
            score_min=_optional_float(summary_payload.get("score_min")),
            score_max=_optional_float(summary_payload.get("score_max")),
            density_method=str(summary_payload.get("density_method") or "bounded_logit_kde"),
            smoothing_note=_optional_text(summary_payload.get("smoothing_note")),
        )
        settings_snapshot = self._deserialize_likelihood_ratio_settings(
            payload.get("settings_snapshot"),
            legacy_density_method=summary.density_method,
        )
        return FaceSetComparisonCalibration(
            summary=summary,
            genuine_scores=_float_list("genuine_scores"),
            impostor_scores=_float_list("impostor_scores"),
            procedure_details=tuple(str(item) for item in procedure_payload),
            settings_snapshot=settings_snapshot,
            model_path=candidate,
            loaded_from_model=True,
        )

    def _deserialize_likelihood_ratio_settings(
        self,
        payload: object,
        *,
        legacy_density_method: str | None = None,
    ) -> LikelihoodRatioSettings | None:
        if payload in (None, ""):
            if legacy_density_method in (None, ""):
                return None
            return LikelihoodRatioSettings(density_estimator=str(legacy_density_method))
        if not isinstance(payload, dict):
            raise ValueError("Campo invalido no modelo de calibracao: settings_snapshot")
        return LikelihoodRatioSettings(
            max_scores_per_distribution=int(payload.get("max_scores_per_distribution", 20000)),
            minimum_identities_with_faces=int(payload.get("minimum_identities_with_faces", 2)),
            minimum_same_source_scores=int(payload.get("minimum_same_source_scores", 5)),
            minimum_different_source_scores=int(payload.get("minimum_different_source_scores", 5)),
            minimum_unique_scores_per_distribution=int(
                payload.get("minimum_unique_scores_per_distribution", 2)
            ),
            density_estimator=str(
                payload.get("density_estimator", legacy_density_method or "bounded_logit_kde")
            ),
            kde_bandwidth_scale=float(payload.get("kde_bandwidth_scale", 1.0)),
            kde_uniform_floor_weight=float(payload.get("kde_uniform_floor_weight", 0.001)),
            kde_min_density=float(payload.get("kde_min_density", 1e-12)),
        )

    def _likelihood_ratio_smoothing_note(self, settings: LikelihoodRatioSettings) -> str:
        return (
            f"{score_density_method_label(settings.density_estimator)} estabilizada com piso uniforme de "
            f"{settings.kde_uniform_floor_weight:.4%} e densidade minima de "
            f"{settings.kde_min_density:.1e}."
        )

    def _likelihood_ratio_density_procedure_detail(self, settings: LikelihoodRatioSettings) -> str:
        return (
            "[Calibracao LR] A densidade de cada score foi estimada com "
            f"{score_density_method_label(settings.density_estimator)} "
            f"(banda x{settings.kde_bandwidth_scale:.3f}) e piso uniforme para evitar densidades nulas."
        )

    def migrate_face_set_comparison_calibration_model(
        self,
        model_path: Path,
        output_path: Path,
        *,
        target_settings: LikelihoodRatioSettings | None = None,
    ) -> Path:
        source_model = self.load_face_set_comparison_calibration_model(model_path)
        source_path = self._normalize_calibration_model_path(model_path)
        if source_path is None:
            raise ValueError("O caminho do modelo de calibracao precisa ser informado.")
        settings = target_settings or self._config.likelihood_ratio
        source_settings = source_model.settings_snapshot
        source_method = (
            source_settings.density_estimator
            if source_settings is not None
            else source_model.summary.density_method
        )
        retained_details = [
            line
            for line in source_model.procedure_details
            if "A densidade de cada score foi estimada com " not in line
            and "Modelo migrado sem reprocessar imagens" not in line
        ]
        migrated_details = tuple(
            [
                *retained_details,
                self._likelihood_ratio_density_procedure_detail(settings),
                (
                    "[Calibracao LR] Modelo migrado sem reprocessar imagens; "
                    f"scores reaproveitados de {source_path} | "
                    f"metodo_origem={source_method} | metodo_destino={settings.density_estimator}."
                ),
            ]
        )
        migrated_summary = replace(
            source_model.summary,
            density_method=settings.density_estimator,
            smoothing_note=self._likelihood_ratio_smoothing_note(settings),
        )
        migrated_model = replace(
            source_model,
            summary=migrated_summary,
            procedure_details=migrated_details,
            settings_snapshot=settings,
            model_path=source_path,
            loaded_from_model=False,
        )
        return self.save_face_set_comparison_calibration_model(migrated_model, output_path)

    def _build_face_set_comparison_calibration(
        self,
        *,
        dataset_root: Path | None,
        calibration_plan: list[tuple[str, list[Path]]],
        inputs: list[FaceSetComparisonInput],
        entries: list[FaceSetComparisonEntry],
        progress_callback: ProgressCallback | None = None,
        progress_current: int | None = None,
        progress_total: int | None = None,
        text_logger: logging.Logger | None = None,
        log_callback: LogCallback | None = None,
        event_logger: StructuredEventLogger | None = None,
    ) -> FaceSetComparisonCalibration:
        if dataset_root is None:
            raise ValueError("A raiz da base de calibracao precisa ser informada para construir a LR.")

        lr_settings = self._config.likelihood_ratio
        entries_by_identity: dict[str, list[FaceSetComparisonEntry]] = {}
        for entry in entries:
            if entry.identity_label:
                entries_by_identity.setdefault(entry.identity_label, []).append(entry)

        genuine_pair_total, impostor_pair_total = self._estimate_calibration_pair_totals(entries_by_identity)
        if text_logger is not None:
            emit_log(
                text_logger,
                log_callback,
                (
                    f"[Calibracao LR] Consolidando distribuicoes | "
                    f"identidades_com_faces={len(entries_by_identity)} | faces={len(entries)}"
                ),
            )
            emit_log(
                text_logger,
                log_callback,
                (
                    f"[Calibracao LR] Pares previstos | "
                    f"mesma_origem={genuine_pair_total} | origem_distinta={impostor_pair_total} | "
                    f"limite_amostra={lr_settings.max_scores_per_distribution}"
                ),
            )
        if (
            progress_callback is not None
            and progress_current is not None
            and progress_total is not None
        ):
            emit_progress(
                progress_callback,
                progress_current,
                progress_total,
                (
                    "Calibracao LR: preparando pares estatisticos "
                    f"({len(entries_by_identity)} identidades com faces)"
                ),
            )
        if event_logger is not None:
            event_logger.write(
                "comparison_calibration_pair_plan",
                identities_with_faces=len(entries_by_identity),
                selected_faces=len(entries),
                genuine_pair_total=genuine_pair_total,
                impostor_pair_total=impostor_pair_total,
                sample_limit=lr_settings.max_scores_per_distribution,
            )

        genuine_scores, genuine_pair_total = self._sample_calibration_scores(
            self._iter_same_source_scores(entries_by_identity),
            max_scores=lr_settings.max_scores_per_distribution,
            phase_label="Mesma origem",
            distribution_key="same_source",
            expected_total_pairs=genuine_pair_total,
            progress_callback=progress_callback,
            progress_current=progress_current,
            progress_total=progress_total,
            text_logger=text_logger,
            log_callback=log_callback,
            event_logger=event_logger,
        )
        impostor_scores, impostor_pair_total = self._sample_calibration_scores(
            self._iter_different_source_scores(entries_by_identity),
            max_scores=lr_settings.max_scores_per_distribution,
            phase_label="Origem distinta",
            distribution_key="different_source",
            expected_total_pairs=impostor_pair_total,
            progress_callback=progress_callback,
            progress_current=progress_current,
            progress_total=progress_total,
            text_logger=text_logger,
            log_callback=log_callback,
            event_logger=event_logger,
        )

        support_ready, support_note = self._has_likelihood_ratio_support(
            genuine_scores=genuine_scores,
            impostor_scores=impostor_scores,
            identities_with_faces=len(entries_by_identity),
        )
        all_scores = [*genuine_scores, *impostor_scores]
        summary = FaceSetComparisonCalibrationSummary(
            dataset_root=dataset_root,
            identity_count=len(calibration_plan),
            processed_identities=len([1 for _, paths in calibration_plan if paths]),
            input_images=len(inputs),
            processed_images=len([item for item in inputs if not item.processing_error]),
            selected_faces=len(entries),
            identities_with_selected_faces=len(entries_by_identity),
            genuine_pair_total=genuine_pair_total,
            impostor_pair_total=impostor_pair_total,
            genuine_score_count=len(genuine_scores),
            impostor_score_count=len(impostor_scores),
            support_ready=support_ready,
            support_note=support_note,
            score_min=min(all_scores) if all_scores else None,
            score_max=max(all_scores) if all_scores else None,
            density_method=lr_settings.density_estimator,
            smoothing_note=self._likelihood_ratio_smoothing_note(lr_settings),
        )
        procedure_details = (
            "[Calibracao LR] Cada subdiretorio imediato foi tratado como uma identidade rotulada.",
            "[Calibracao LR] Scores Padrão/Questionado de mesma origem: pares entre faces da mesma identidade.",
            "[Calibracao LR] Scores Padrão/Questionado de origem distinta: pares entre identidades diferentes.",
            self._likelihood_ratio_density_procedure_detail(lr_settings),
        )
        return FaceSetComparisonCalibration(
            summary=summary,
            inputs=inputs,
            entries=entries,
            genuine_scores=genuine_scores,
            impostor_scores=impostor_scores,
            procedure_details=procedure_details,
            settings_snapshot=lr_settings,
        )

    def _estimate_calibration_pair_totals(
        self,
        entries_by_identity: dict[str, list[FaceSetComparisonEntry]],
    ) -> tuple[int, int]:
        face_counts = [len(entries) for entries in entries_by_identity.values() if entries]
        genuine_pair_total = sum(count * (count - 1) // 2 for count in face_counts)
        total_faces = sum(face_counts)
        all_pairs = total_faces * (total_faces - 1) // 2
        impostor_pair_total = max(0, all_pairs - genuine_pair_total)
        return genuine_pair_total, impostor_pair_total

    def _sample_calibration_scores(
        self,
        values: Iterable[float],
        *,
        max_scores: int,
        phase_label: str | None = None,
        distribution_key: str | None = None,
        expected_total_pairs: int | None = None,
        progress_callback: ProgressCallback | None = None,
        progress_current: int | None = None,
        progress_total: int | None = None,
        text_logger: logging.Logger | None = None,
        log_callback: LogCallback | None = None,
        event_logger: StructuredEventLogger | None = None,
    ) -> tuple[list[float], int]:
        rng = np.random.default_rng(20260409)
        sampled: list[float] = []
        total_seen = 0
        heartbeat_interval_pairs = 250000
        heartbeat_interval_seconds = 5.0
        last_heartbeat = time.monotonic()
        expected_display = expected_total_pairs if expected_total_pairs is not None else "-"

        if phase_label and text_logger is not None:
            emit_log(
                text_logger,
                log_callback,
                (
                    f"[Calibracao LR] {phase_label} | iniciando amostragem | "
                    f"pares_previstos={expected_display} | limite_amostra={max_scores}"
                ),
            )
        if distribution_key and event_logger is not None:
            event_logger.write(
                "comparison_calibration_distribution_started",
                distribution=distribution_key,
                expected_pairs=expected_total_pairs,
                sample_limit=max_scores,
            )

        for value in values:
            if len(sampled) < max_scores:
                sampled.append(value)
            else:
                replacement_index = int(rng.integers(0, total_seen + 1))
                if replacement_index < max_scores:
                    sampled[replacement_index] = value
            total_seen += 1

            should_emit_heartbeat = (
                phase_label is not None
                and total_seen != expected_total_pairs
                and (
                    total_seen % heartbeat_interval_pairs == 0
                    or (time.monotonic() - last_heartbeat) >= heartbeat_interval_seconds
                )
            )
            if should_emit_heartbeat:
                progress_message = (
                    f"Calibracao LR: {phase_label.lower()} {total_seen}/{expected_display} pares"
                )
                if (
                    progress_callback is not None
                    and progress_current is not None
                    and progress_total is not None
                ):
                    emit_progress(
                        progress_callback,
                        progress_current,
                        progress_total,
                        progress_message,
                    )
                if text_logger is not None:
                    coverage_text = ""
                    if expected_total_pairs:
                        coverage_text = f" | cobertura={((total_seen / expected_total_pairs) * 100.0):.1f}%"
                    emit_log(
                        text_logger,
                        log_callback,
                        (
                            f"[Calibracao LR] {phase_label} em andamento | "
                            f"pares={total_seen}/{expected_display} | amostrados={len(sampled)}"
                            f"{coverage_text}"
                        ),
                    )
                if distribution_key and event_logger is not None:
                    event_logger.write(
                        "comparison_calibration_distribution_progress",
                        distribution=distribution_key,
                        processed_pairs=total_seen,
                        expected_pairs=expected_total_pairs,
                        sampled_scores=len(sampled),
                    )
                last_heartbeat = time.monotonic()

        if phase_label and text_logger is not None:
            emit_log(
                text_logger,
                log_callback,
                (
                    f"[Calibracao LR] {phase_label} concluida | "
                    f"pares={total_seen}/{expected_display} | amostrados={len(sampled)}"
                ),
            )
        if distribution_key and event_logger is not None:
            event_logger.write(
                "comparison_calibration_distribution_completed",
                distribution=distribution_key,
                processed_pairs=total_seen,
                expected_pairs=expected_total_pairs,
                sampled_scores=len(sampled),
            )
        return sampled, total_seen

    def _iter_same_source_scores(
        self,
        entries_by_identity: dict[str, list[FaceSetComparisonEntry]],
    ) -> Iterable[float]:
        for entries in entries_by_identity.values():
            if len(entries) < 2:
                continue
            for index, left in enumerate(entries):
                for right in entries[index + 1 :]:
                    yield cosine_similarity(left.embedding, right.embedding)

    def _iter_different_source_scores(
        self,
        entries_by_identity: dict[str, list[FaceSetComparisonEntry]],
    ) -> Iterable[float]:
        identity_labels = sorted(entries_by_identity)
        for index, left_label in enumerate(identity_labels):
            left_entries = entries_by_identity[left_label]
            for right_label in identity_labels[index + 1 :]:
                right_entries = entries_by_identity[right_label]
                for left in left_entries:
                    for right in right_entries:
                        yield cosine_similarity(left.embedding, right.embedding)

    def _has_likelihood_ratio_support(
        self,
        *,
        genuine_scores: list[float],
        impostor_scores: list[float],
        identities_with_faces: int,
    ) -> tuple[bool, str | None]:
        settings = self._config.likelihood_ratio
        if identities_with_faces < settings.minimum_identities_with_faces:
            return (
                False,
                "Sao necessarias ao menos "
                f"{settings.minimum_identities_with_faces} identidades com faces selecionadas na base de calibracao.",
            )
        if len(genuine_scores) < settings.minimum_same_source_scores:
            return (
                False,
                "A base de calibracao nao gerou scores Padrão/Questionado de mesma origem "
                f"suficientes para ajustar a densidade (minimo={settings.minimum_same_source_scores}).",
            )
        if len(impostor_scores) < settings.minimum_different_source_scores:
            return (
                False,
                "A base de calibracao nao gerou scores Padrão/Questionado de origem distinta "
                f"suficientes para ajustar a densidade (minimo={settings.minimum_different_source_scores}).",
            )
        if len({round(value, 8) for value in genuine_scores}) < settings.minimum_unique_scores_per_distribution:
            return (
                False,
                "Os scores Padrão/Questionado de mesma origem nao apresentam variabilidade suficiente "
                f"(minimo de valores distintos={settings.minimum_unique_scores_per_distribution}).",
            )
        if len({round(value, 8) for value in impostor_scores}) < settings.minimum_unique_scores_per_distribution:
            return (
                False,
                "Os scores Padrão/Questionado de origem distinta nao apresentam variabilidade suficiente "
                f"(minimo de valores distintos={settings.minimum_unique_scores_per_distribution}).",
            )
        return True, None

    def _calibration_summary_log_lines(
        self,
        summary: FaceSetComparisonCalibrationSummary,
    ) -> list[str]:
        lines = [
            (
                f"[Calibracao LR] Resumo | identidades={summary.identity_count} | "
                f"faces={summary.selected_faces} | mesma_origem={summary.genuine_score_count}/{summary.genuine_pair_total} | "
                f"origem_distinta={summary.impostor_score_count}/{summary.impostor_pair_total}"
            )
        ]
        if summary.support_ready:
            lines.append(
                (
                    f"[Calibracao LR] Ajuste pronto | metodo={summary.density_method} | "
                    f"faixa=[{summary.score_min if summary.score_min is not None else '-'}, "
                    f"{summary.score_max if summary.score_max is not None else '-'}]"
                )
            )
        else:
            lines.append(
                f"[Calibracao LR] Ajuste indisponivel | motivo={summary.support_note or 'amostra insuficiente'}"
            )
        return lines

    def _apply_face_set_likelihood_ratio_calibration(
        self,
        matches: list[FaceSetComparisonMatch],
        calibration: FaceSetComparisonCalibration,
        *,
        progress_callback: ProgressCallback | None = None,
        progress_current: int | None = None,
        progress_total: int | None = None,
        text_logger: logging.Logger | None = None,
        log_callback: LogCallback | None = None,
        event_logger: StructuredEventLogger | None = None,
    ) -> list[FaceSetComparisonMatch]:
        summary = calibration.summary
        if not summary.support_ready:
            return matches

        settings = calibration.settings_snapshot or self._config.likelihood_ratio
        total_matches = len(matches)
        heartbeat_interval_pairs = 5000
        heartbeat_interval_seconds = 5.0
        last_heartbeat = time.monotonic()
        if text_logger is not None:
            emit_log(
                text_logger,
                log_callback,
                (
                    f"[Calibracao LR] Ajustando densidades | "
                    f"mesma_origem={len(calibration.genuine_scores)} | "
                    f"origem_distinta={len(calibration.impostor_scores)} | "
                    f"metodo={settings.density_estimator} | "
                    f"banda_x={settings.kde_bandwidth_scale:.3f}"
                ),
            )
        if (
            progress_callback is not None
            and progress_current is not None
            and progress_total is not None
        ):
            emit_progress(
                progress_callback,
                progress_current,
                progress_total,
                f"Calibracao LR: ajustando densidades ({settings.density_estimator})",
            )
        if event_logger is not None:
            event_logger.write(
                "comparison_calibration_application_started",
                match_count=total_matches,
                genuine_scores=len(calibration.genuine_scores),
                impostor_scores=len(calibration.impostor_scores),
            )

        genuine_array = np.asarray(calibration.genuine_scores, dtype=np.float64)
        impostor_array = np.asarray(calibration.impostor_scores, dtype=np.float64)
        same_source_density_model = self._build_score_density_model(genuine_array, settings=settings)
        different_source_density_model = self._build_score_density_model(impostor_array, settings=settings)
        calibration_scores = calibration.genuine_scores + calibration.impostor_scores
        support_lower = min(calibration_scores) if calibration_scores else None
        support_upper = max(calibration_scores) if calibration_scores else None

        calibrated_matches: list[FaceSetComparisonMatch] = []
        extrapolated_count = 0
        for index, match in enumerate(matches, start=1):
            same_source_density = self._stabilized_score_density(
                same_source_density_model,
                match.similarity,
                settings=settings,
            )
            different_source_density = self._stabilized_score_density(
                different_source_density_model,
                match.similarity,
                settings=settings,
            )
            likelihood_ratio = same_source_density / different_source_density
            log10_likelihood_ratio = math.log10(likelihood_ratio)
            outside_support = bool(
                support_lower is not None
                and support_upper is not None
                and (match.similarity < support_lower or match.similarity > support_upper)
            )
            evidence_label = self._likelihood_ratio_evidence_label(log10_likelihood_ratio)
            if outside_support:
                extrapolated_count += 1
                evidence_label = f"{evidence_label} (escore fora do intervalo de calibracao)"
            calibrated_matches.append(
                replace(
                    match,
                    likelihood_ratio=likelihood_ratio,
                    log10_likelihood_ratio=log10_likelihood_ratio,
                    same_source_density=same_source_density,
                    different_source_density=different_source_density,
                    evidence_label=evidence_label,
                    outside_calibration_support=outside_support,
                )
            )
            should_emit_heartbeat = (
                index < total_matches
                and (
                    index % heartbeat_interval_pairs == 0
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
                        f"Calibracao LR: aplicando densidades ({index}/{total_matches} pares)",
                    )
                if text_logger is not None:
                    emit_log(
                        text_logger,
                        log_callback,
                        f"[Calibracao LR] Aplicacao em andamento | pares_calibrados={index}/{total_matches}",
                    )
                if event_logger is not None:
                    event_logger.write(
                        "comparison_calibration_application_progress",
                        calibrated_pairs=index,
                        match_count=total_matches,
                    )
                last_heartbeat = time.monotonic()
        if text_logger is not None:
            emit_log(
                text_logger,
                log_callback,
                f"[Calibracao LR] Aplicacao de LR concluida | pares_calibrados={len(calibrated_matches)}",
            )
        if extrapolated_count and text_logger is not None:
            emit_log(
                text_logger,
                log_callback,
                (
                    f"[Calibracao LR][ALERTA] {extrapolated_count} par(es) com escore fora do "
                    f"intervalo de calibracao [{support_lower:.4f}, {support_upper:.4f}]; "
                    "o LR desses pares e uma extrapolacao das densidades e deve ser "
                    "interpretado com cautela."
                ),
            )
        if event_logger is not None:
            event_logger.write(
                "comparison_calibration_application_completed",
                calibrated_pairs=len(calibrated_matches),
                match_count=total_matches,
                extrapolated_pairs=extrapolated_count,
                calibration_score_min=support_lower,
                calibration_score_max=support_upper,
            )
        return calibrated_matches

    def _build_score_density_model(
        self,
        values: np.ndarray,
        *,
        settings: LikelihoodRatioSettings,
    ):
        return fit_score_density_model(
            values,
            method=settings.density_estimator,
            bandwidth_scale=settings.kde_bandwidth_scale,
        )

    def _stabilized_score_density(
        self,
        model,
        score: float,
        *,
        settings: LikelihoodRatioSettings | None = None,
    ) -> float:
        lr_settings = settings or self._config.likelihood_ratio
        clipped_score = max(-1.0, min(1.0, score))
        raw_density = float(model.evaluate_raw([clipped_score])[0])
        stabilized = stabilize_score_density(
            raw_density,
            uniform_floor_weight=lr_settings.kde_uniform_floor_weight,
            min_density=lr_settings.kde_min_density,
        )
        return float(stabilized[0])

    def _likelihood_ratio_evidence_label(self, log10_likelihood_ratio: float) -> str:
        if log10_likelihood_ratio >= 3.0:
            return "Suporte extremamente forte para mesma origem"
        if log10_likelihood_ratio >= 2.0:
            return "Suporte muito forte para mesma origem"
        if log10_likelihood_ratio >= 1.0:
            return "Suporte forte para mesma origem"
        if log10_likelihood_ratio >= 0.5:
            return "Suporte moderado para mesma origem"
        if log10_likelihood_ratio > -0.5:
            return "Evidencia limitada ou inconclusiva"
        if log10_likelihood_ratio > -1.0:
            return "Suporte moderado para origem diferente"
        if log10_likelihood_ratio > -2.0:
            return "Suporte forte para origem diferente"
        return "Suporte muito forte para origem diferente"
