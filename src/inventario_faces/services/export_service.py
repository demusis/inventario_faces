from __future__ import annotations

import csv
import json
from pathlib import Path

from inventario_faces.domain.entities import (
    FaceCluster,
    FaceSetComparisonCalibration,
    FaceSetComparisonEntry,
    FaceSetComparisonInput,
    FaceSetComparisonResult,
    FaceSearchResult,
    FaceOccurrence,
    FaceTrack,
    FileRecord,
    InventoryResult,
    KeyFrame,
    SearchArtifacts,
)
from inventario_faces.utils.path_utils import ensure_directory
from inventario_faces.utils.serialization import to_serializable


class ExportService:
    def __init__(self, run_directory: Path) -> None:
        self._inventory_directory = ensure_directory(run_directory / "inventory")
        self._comparison_directory = ensure_directory(run_directory / "comparison")

    def _comparison_group_label(self, set_label: str) -> str:
        if set_label == "A":
            return "Padrão"
        if set_label == "B":
            return "Questionado"
        if set_label == "CAL":
            return "Calibração LR"
        return set_label

    def _calibration_distribution_label(self, distribution: str) -> str:
        if distribution == "same_source":
            return "padrao_questionado_mesma_origem"
        if distribution == "different_source":
            return "padrao_questionado_origem_distinta"
        return distribution

    @property
    def inventory_directory(self) -> Path:
        return self._inventory_directory

    @property
    def comparison_directory(self) -> Path:
        return self._comparison_directory

    def write_files_csv(self, files: list[FileRecord]) -> Path:
        output_path = self._inventory_directory / "files.csv"
        with output_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(
                [
                    "path",
                    "media_type",
                    "sha512",
                    "size_bytes",
                    "discovered_at_utc",
                    "modified_at_utc",
                    "processing_error",
                    "media_info_track_count",
                    "media_info_error",
                ]
            )
            for item in files:
                writer.writerow(
                    [
                        str(item.path),
                        item.media_type.value,
                        item.sha512,
                        item.size_bytes,
                        item.discovered_at_utc.isoformat(),
                        item.modified_at_utc.isoformat() if item.modified_at_utc else "",
                        item.processing_error or "",
                        len(item.media_info_tracks),
                        item.media_info_error or "",
                    ]
                )
        return output_path

    def write_occurrences_csv(self, occurrences: list[FaceOccurrence]) -> Path:
        output_path = self._inventory_directory / "occurrences.csv"
        with output_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(
                [
                    "occurrence_id",
                    "source_path",
                    "sha512",
                    "media_type",
                    "analysis_timestamp_utc",
                    "frame_index",
                    "frame_timestamp_seconds",
                    "bbox",
                    "detection_score",
                    "track_id",
                    "keyframe_id",
                    "is_keyframe",
                    "embedding_source",
                    "quality_score",
                    "sharpness",
                    "brightness",
                    "illumination",
                    "frontality",
                    "cluster_id",
                    "suggested_cluster_ids",
                    "crop_path",
                    "context_image_path",
                ]
            )
            for item in occurrences:
                writer.writerow(
                    [
                        item.occurrence_id,
                        str(item.source_path),
                        item.sha512,
                        item.media_type.value,
                        item.analysis_timestamp_utc.isoformat(),
                        item.frame_index,
                        item.frame_timestamp_seconds,
                        f"{item.bbox.x1:.2f},{item.bbox.y1:.2f},{item.bbox.x2:.2f},{item.bbox.y2:.2f}",
                        f"{item.detection_score:.6f}",
                        item.track_id or "",
                        item.keyframe_id or "",
                        "1" if item.is_keyframe else "0",
                        item.embedding_source or "",
                        f"{item.quality_metrics.score:.6f}" if item.quality_metrics is not None else "",
                        f"{item.quality_metrics.sharpness:.6f}" if item.quality_metrics is not None else "",
                        f"{item.quality_metrics.brightness:.6f}" if item.quality_metrics is not None else "",
                        f"{item.quality_metrics.illumination:.6f}" if item.quality_metrics is not None else "",
                        f"{item.quality_metrics.frontality:.6f}" if item.quality_metrics is not None else "",
                        item.cluster_id or "",
                        ";".join(item.suggested_cluster_ids),
                        str(item.crop_path) if item.crop_path else "",
                        str(item.context_image_path) if item.context_image_path else "",
                    ]
                )
        return output_path

    def write_tracks_csv(self, tracks: list[FaceTrack]) -> Path:
        output_path = self._inventory_directory / "tracks.csv"
        with output_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(
                [
                    "track_id",
                    "source_path",
                    "video_path",
                    "media_type",
                    "sha512",
                    "start_frame",
                    "end_frame",
                    "start_time",
                    "end_time",
                    "occurrence_count",
                    "keyframe_count",
                    "cluster_id",
                    "candidate_cluster_ids",
                    "best_occurrence_id",
                    "preview_path",
                    "top_crop_paths",
                    "mean_quality_score",
                    "best_quality_score",
                    "mean_sharpness",
                    "mean_brightness",
                    "mean_illumination",
                    "mean_frontality",
                    "duration_seconds",
                ]
            )
            for item in tracks:
                writer.writerow(
                    [
                        item.track_id,
                        str(item.source_path),
                        str(item.video_path) if item.video_path is not None else "",
                        item.media_type.value,
                        item.sha512,
                        item.start_frame,
                        item.end_frame,
                        item.start_time,
                        item.end_time,
                        len(item.occurrence_ids),
                        len(item.keyframe_ids),
                        item.cluster_id or "",
                        ";".join(item.candidate_cluster_ids),
                        item.best_occurrence_id or "",
                        str(item.preview_path) if item.preview_path is not None else "",
                        ";".join(str(path) for path in item.top_crop_paths),
                        f"{item.quality_statistics.mean_quality_score:.6f}",
                        f"{item.quality_statistics.best_quality_score:.6f}",
                        f"{item.quality_statistics.mean_sharpness:.6f}",
                        f"{item.quality_statistics.mean_brightness:.6f}",
                        f"{item.quality_statistics.mean_illumination:.6f}",
                        f"{item.quality_statistics.mean_frontality:.6f}",
                        f"{item.quality_statistics.duration_seconds:.6f}",
                    ]
                )
        return output_path

    def write_keyframes_csv(self, keyframes: list[KeyFrame]) -> Path:
        output_path = self._inventory_directory / "keyframes.csv"
        with output_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(
                [
                    "keyframe_id",
                    "track_id",
                    "occurrence_id",
                    "source_path",
                    "frame_index",
                    "timestamp_seconds",
                    "selection_reasons",
                    "detection_score",
                    "quality_score",
                    "crop_path",
                    "context_image_path",
                    "preview_path",
                ]
            )
            for item in keyframes:
                writer.writerow(
                    [
                        item.keyframe_id,
                        item.track_id,
                        item.occurrence_id,
                        str(item.source_path),
                        item.frame_index,
                        item.timestamp_seconds,
                        ";".join(item.selection_reasons),
                        f"{item.detection_score:.6f}",
                        f"{item.quality_metrics.score:.6f}" if item.quality_metrics is not None else "",
                        str(item.crop_path) if item.crop_path is not None else "",
                        str(item.context_image_path) if item.context_image_path is not None else "",
                        str(item.preview_path) if item.preview_path is not None else "",
                    ]
                )
        return output_path

    def write_clusters_json(self, clusters: list[FaceCluster]) -> Path:
        output_path = self._inventory_directory / "clusters.json"
        output_path.write_text(
            json.dumps(to_serializable(clusters), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return output_path

    def write_search_json(self, search: SearchArtifacts | None) -> Path:
        output_path = self._inventory_directory / "search.json"
        output_path.write_text(
            json.dumps(to_serializable(search), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return output_path

    def write_media_info_json(self, files: list[FileRecord]) -> Path:
        output_path = self._inventory_directory / "media_info.json"
        payload = [
            {
                "path": str(item.path),
                "media_type": item.media_type.value,
                "media_info_error": item.media_info_error,
                "media_info_tracks": to_serializable(item.media_info_tracks),
            }
            for item in files
            if item.media_type.value in {"image", "video"}
        ]
        output_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return output_path

    def write_manifest(self, result: InventoryResult) -> Path:
        output_path = self._inventory_directory / "manifest.json"
        output_path.write_text(
            json.dumps(to_serializable(result), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return output_path

    def write_face_search_json(self, result: FaceSearchResult) -> Path:
        output_path = self._inventory_directory / "face_search.json"
        output_path.write_text(
            json.dumps(to_serializable(result), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return output_path

    def write_face_set_comparison_json(self, result: FaceSetComparisonResult) -> Path:
        output_path = self._comparison_directory / "face_set_comparison.json"
        output_path.write_text(
            json.dumps(to_serializable(result), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return output_path

    def write_face_set_comparison_inputs_csv(
        self,
        filename: str,
        inputs: list[FaceSetComparisonInput],
    ) -> Path:
        output_path = self._comparison_directory / filename
        with output_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(
                [
                    "set_label",
                    "source_path",
                    "sha512",
                    "detected_faces",
                    "selected_faces",
                    "tracks",
                    "keyframes",
                    "identity_label",
                    "processing_error",
                    "export_source_copy",
                ]
            )
            for item in inputs:
                writer.writerow(
                    [
                        self._comparison_group_label(item.set_label),
                        str(item.source_path),
                        item.sha512,
                        item.detected_faces,
                        item.selected_faces,
                        item.tracks,
                        item.keyframes,
                        item.identity_label or "",
                        item.processing_error or "",
                        str(item.export_source_copy) if item.export_source_copy is not None else "",
                    ]
                )
        return output_path

    def write_face_set_comparison_entries_csv(
        self,
        filename: str,
        entries: list[FaceSetComparisonEntry],
    ) -> Path:
        output_path = self._comparison_directory / filename
        with output_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(
                [
                    "entry_id",
                    "set_label",
                    "source_path",
                    "sha512",
                    "track_id",
                    "occurrence_id",
                    "keyframe_id",
                    "bbox",
                    "detection_score",
                    "quality_score",
                    "sharpness",
                    "brightness",
                    "illumination",
                    "frontality",
                    "embedding_dimension",
                    "embedding_source",
                    "identity_label",
                    "crop_path",
                    "context_image_path",
                    "mesh_crop_path",
                    "mesh_context_path",
                    "selection_reasons",
                    "biometric_landmark_count",
                ]
            )
            for item in entries:
                writer.writerow(
                    [
                        item.entry_id,
                        self._comparison_group_label(item.set_label),
                        str(item.source_path),
                        item.sha512,
                        item.track_id,
                        item.occurrence_id,
                        item.keyframe_id or "",
                        f"{item.bbox.x1:.2f},{item.bbox.y1:.2f},{item.bbox.x2:.2f},{item.bbox.y2:.2f}",
                        f"{item.detection_score:.6f}",
                        f"{item.quality_score:.6f}" if item.quality_score is not None else "",
                        f"{item.sharpness:.6f}" if item.sharpness is not None else "",
                        f"{item.brightness:.6f}" if item.brightness is not None else "",
                        f"{item.illumination:.6f}" if item.illumination is not None else "",
                        f"{item.frontality:.6f}" if item.frontality is not None else "",
                        item.embedding_dimension,
                        item.embedding_source or "",
                        item.identity_label or "",
                        str(item.crop_path) if item.crop_path is not None else "",
                        str(item.context_image_path) if item.context_image_path is not None else "",
                        str(item.mesh_crop_path) if item.mesh_crop_path is not None else "",
                        str(item.mesh_context_path) if item.mesh_context_path is not None else "",
                        ";".join(item.selection_reasons),
                        len(item.biometric_landmarks),
                    ]
                )
        return output_path

    def write_face_set_comparison_matches_csv(self, result: FaceSetComparisonResult) -> Path:
        output_path = self._comparison_directory / "face_set_comparison_matches.csv"
        with output_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(
                [
                    "rank",
                    "left_entry_id",
                    "right_entry_id",
                    "left_track_id",
                    "right_track_id",
                    "similarity",
                    "classification",
                    "left_quality_score",
                    "right_quality_score",
                    "likelihood_ratio",
                    "log10_likelihood_ratio",
                    "same_source_density",
                    "different_source_density",
                    "evidence_label",
                ]
            )
            for item in result.matches:
                writer.writerow(
                    [
                        item.rank,
                        item.left_entry_id,
                        item.right_entry_id,
                        item.left_track_id,
                        item.right_track_id,
                        f"{item.similarity:.6f}",
                        item.classification,
                        f"{item.left_quality_score:.6f}" if item.left_quality_score is not None else "",
                        f"{item.right_quality_score:.6f}" if item.right_quality_score is not None else "",
                        f"{item.likelihood_ratio:.12f}" if item.likelihood_ratio is not None else "",
                        f"{item.log10_likelihood_ratio:.6f}" if item.log10_likelihood_ratio is not None else "",
                        f"{item.same_source_density:.12f}" if item.same_source_density is not None else "",
                        f"{item.different_source_density:.12f}" if item.different_source_density is not None else "",
                        item.evidence_label or "",
                    ]
                )
        return output_path

    def write_face_set_comparison_calibration_scores_csv(
        self,
        calibration: FaceSetComparisonCalibration,
    ) -> Path:
        output_path = self._comparison_directory / "face_set_comparison_calibration_scores.csv"
        with output_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["distribution", "index", "score"])
            for distribution, scores in (
                ("same_source", calibration.genuine_scores),
                ("different_source", calibration.impostor_scores),
            ):
                for index, score in enumerate(scores, start=1):
                    writer.writerow([self._calibration_distribution_label(distribution), index, f"{score:.6f}"])
        return output_path

    def write_face_set_comparison_summary_text(self, result: FaceSetComparisonResult) -> Path:
        output_path = self._comparison_directory / "face_set_comparison_summary.txt"
        summary = result.summary
        def _fmt(value: float | None) -> str:
            return "-" if value is None else f"{value:.4f}"

        lines = [
            "Inventario Faces - Comparacao entre conjuntos",
            f"Execucao: {result.run_directory}",
            f"Exportacao: {result.export_directory}",
            "",
            "Procedimento:",
            *result.procedure_details,
            "",
            "Resumo:",
            f"Padrão - imagens: {summary.set_a_images}",
            f"Questionado - imagens: {summary.set_b_images}",
            f"Padrão - faces detectadas: {summary.set_a_detected_faces}",
            f"Questionado - faces detectadas: {summary.set_b_detected_faces}",
            f"Padrão - faces selecionadas: {summary.set_a_selected_faces}",
            f"Questionado - faces selecionadas: {summary.set_b_selected_faces}",
            f"Padrão - imagens sem face valida: {summary.set_a_images_without_faces}",
            f"Questionado - imagens sem face valida: {summary.set_b_images_without_faces}",
            f"Comparacoes par-a-par: {summary.total_pair_comparisons}",
            f"Compatibilidades por atribuicao: {summary.assignment_matches}",
            f"Compatibilidades candidatas: {summary.candidate_matches}",
            f"Melhor similaridade: {_fmt(summary.best_similarity)}",
            f"Pior similaridade: {_fmt(summary.worst_similarity)}",
            f"Media de similaridade: {_fmt(summary.mean_similarity)}",
            f"Mediana de similaridade: {_fmt(summary.median_similarity)}",
            f"Desvio padrao de similaridade: {_fmt(summary.stddev_similarity)}",
            f"Primeiro quartil (Q1): {_fmt(summary.q1_similarity)}",
            f"Terceiro quartil (Q3): {_fmt(summary.q3_similarity)}",
            f"IC95% da media - limite inferior: {_fmt(summary.mean_confidence_low)}",
            f"IC95% da media - limite superior: {_fmt(summary.mean_confidence_high)}",
            f"Limiar candidato: {summary.candidate_threshold:.4f}",
            f"Limiar atribuicao: {summary.assignment_threshold:.4f}",
        ]
        if summary.likelihood_ratio_calibrated:
            lines.extend(
                [
                    "",
                    "Razao de verossimilhanca calibrada:",
                    f"Pares calibrados: {summary.calibrated_matches}",
                    f"Media de log10(LR): {_fmt(summary.mean_log10_likelihood_ratio)}",
                    f"Mediana de log10(LR): {_fmt(summary.median_log10_likelihood_ratio)}",
                    f"Menor log10(LR): {_fmt(summary.min_log10_likelihood_ratio)}",
                    f"Maior log10(LR): {_fmt(summary.max_log10_likelihood_ratio)}",
                ]
            )
        calibration = result.calibration
        if calibration is not None:
            calibration_summary = calibration.summary
            lines.extend(
                [
                    "",
                    "Base de calibracao LR:",
                    f"Diretorio: {calibration_summary.dataset_root}",
                    f"Identidades previstas: {calibration_summary.identity_count}",
                    f"Imagens processadas: {calibration_summary.processed_images}/{calibration_summary.input_images}",
                    f"Faces selecionadas: {calibration_summary.selected_faces}",
                    (
                        "Scores Padrão/Questionado, mesma origem, utilizados/possiveis: "
                        f"{calibration_summary.genuine_score_count}/{calibration_summary.genuine_pair_total}"
                    ),
                    (
                        "Scores Padrão/Questionado, origem distinta, utilizados/possiveis: "
                        f"{calibration_summary.impostor_score_count}/{calibration_summary.impostor_pair_total}"
                    ),
                    f"Metodo de densidade: {calibration_summary.density_method}",
                    f"Ajuste pronto: {'sim' if calibration_summary.support_ready else 'nao'}",
                ]
            )
            if calibration.model_path is not None:
                lines.append(
                    (
                        f"Modelo salvo carregado de: {calibration.model_path}"
                        if calibration.loaded_from_model
                        else f"Modelo salvo em: {calibration.model_path}"
                    )
                )
            if calibration_summary.support_note:
                lines.append(f"Observacao: {calibration_summary.support_note}")
            if calibration_summary.smoothing_note:
                lines.append(f"Estabilizacao: {calibration_summary.smoothing_note}")
            if calibration.settings_snapshot is not None:
                settings = calibration.settings_snapshot
                lines.extend(
                    [
                        (
                            "Parametros LR: "
                            f"amostra_max={settings.max_scores_per_distribution} | "
                            f"min_identidades={settings.minimum_identities_with_faces} | "
                            f"min_mesma_origem={settings.minimum_same_source_scores} | "
                            f"min_origem_distinta={settings.minimum_different_source_scores} | "
                            f"min_distintos={settings.minimum_unique_scores_per_distribution}"
                        ),
                        (
                            "Parametros KDE: "
                            f"banda_x={settings.kde_bandwidth_scale:.3f} | "
                            f"piso_uniforme={settings.kde_uniform_floor_weight:.4%} | "
                            f"densidade_minima={settings.kde_min_density:.1e}"
                        ),
                    ]
                )
        output_path.write_text("\n".join(lines), encoding="utf-8")
        return output_path
