from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import re

from tests._bootstrap import PROJECT_ROOT  # noqa: F401
from inventario_faces.domain.config import (
    AppConfig,
    AppSettings,
    ClusteringSettings,
    DistributedSettings,
    EnhancementSettings,
    FaceModelSettings,
    ForensicsSettings,
    LikelihoodRatioSettings,
    MediaSettings,
    ReportingSettings,
    SearchSettings,
    TrackingSettings,
    VideoSettings,
)
from inventario_faces.infrastructure.config_loader import load_app_config, locate_default_config, save_app_config


class ConfigPersistenceTests(unittest.TestCase):
    def test_save_and_load_app_config_round_trip(self) -> None:
        config = AppConfig(
            app=AppSettings(
                name="Inventario Faces",
                output_directory_name="saida_personalizada",
                report_title="Relatorio Persistido",
                organization="Laboratorio Teste",
                log_level="DEBUG",
                use_local_temp_copy=True,
            ),
            media=MediaSettings(
                image_extensions=(".jpg", ".png"),
                video_extensions=(".mp4", ".avi"),
            ),
            video=VideoSettings(
                sampling_interval_seconds=1.5,
                max_frames_per_video=120,
                keyframe_interval_seconds=4.5,
                significant_change_threshold=0.22,
            ),
            face_model=FaceModelSettings(
                backend="insightface",
                model_name="buffalo_l",
                det_size=(800, 800),
                minimum_face_quality=0.72,
                minimum_face_size_pixels=56,
                ctx_id=-1,
                providers=("CPUExecutionProvider",),
            ),
            clustering=ClusteringSettings(
                assignment_similarity=0.61,
                candidate_similarity=0.47,
                min_cluster_size=2,
                min_track_size=2,
            ),
            reporting=ReportingSettings(
                compile_pdf=False,
                max_tracks_per_group=9,
            ),
            forensics=ForensicsSettings(
                chain_of_custody_note="Nota de teste"
            ),
            tracking=TrackingSettings(
                iou_threshold=0.17,
                spatial_distance_threshold=0.21,
                embedding_similarity_threshold=0.51,
                minimum_total_match_score=0.35,
                geometry_weight=0.4,
                embedding_weight=0.6,
                max_missed_detections=3,
                confidence_margin=0.07,
                representative_embeddings_per_track=6,
                top_crops_per_track=5,
                quality_improvement_margin=0.08,
            ),
            enhancement=EnhancementSettings(
                enable_preprocessing=True,
                minimum_brightness_to_enhance=0.31,
                clahe_clip_limit=2.4,
                clahe_tile_grid_size=10,
                gamma=1.15,
                denoise_strength=6,
            ),
            search=SearchSettings(
                enabled=True,
                prefer_faiss=False,
                coarse_top_k=10,
                refine_top_k=16,
            ),
            likelihood_ratio=LikelihoodRatioSettings(
                max_scores_per_distribution=12345,
                minimum_identities_with_faces=3,
                minimum_same_source_scores=7,
                minimum_different_source_scores=9,
                minimum_unique_scores_per_distribution=4,
                kde_bandwidth_scale=1.25,
                kde_uniform_floor_weight=0.0025,
                kde_min_density=1e-10,
            ),
            distributed=DistributedSettings(
                enabled=True,
                execution_label="lote_compartilhado",
                node_name="estacao-a",
                heartbeat_interval_seconds=20,
                stale_lock_timeout_minutes=180,
                auto_finalize=False,
                validate_partial_integrity=True,
                auto_reprocess_invalid_partials=False,
            ),
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "config.yaml"
            save_app_config(config, output_path)
            loaded = load_app_config(output_path)

        self.assertEqual(config.app.output_directory_name, loaded.app.output_directory_name)
        self.assertEqual(config.app.log_level, loaded.app.log_level)
        self.assertEqual(config.app.use_local_temp_copy, loaded.app.use_local_temp_copy)
        self.assertEqual(config.video.sampling_interval_seconds, loaded.video.sampling_interval_seconds)
        self.assertEqual(config.video.keyframe_interval_seconds, loaded.video.keyframe_interval_seconds)
        self.assertEqual(config.video.significant_change_threshold, loaded.video.significant_change_threshold)
        self.assertEqual(config.face_model.det_size, loaded.face_model.det_size)
        self.assertEqual(config.face_model.minimum_face_quality, loaded.face_model.minimum_face_quality)
        self.assertEqual(config.face_model.minimum_face_size_pixels, loaded.face_model.minimum_face_size_pixels)
        self.assertEqual(config.face_model.providers, loaded.face_model.providers)
        self.assertEqual(config.clustering.min_cluster_size, loaded.clustering.min_cluster_size)
        self.assertEqual(config.clustering.min_track_size, loaded.clustering.min_track_size)
        self.assertEqual(config.reporting.compile_pdf, loaded.reporting.compile_pdf)
        self.assertEqual(config.reporting.max_tracks_per_group, loaded.reporting.max_tracks_per_group)
        self.assertEqual(config.tracking.max_missed_detections, loaded.tracking.max_missed_detections)
        self.assertEqual(config.enhancement.gamma, loaded.enhancement.gamma)
        self.assertEqual(config.search.refine_top_k, loaded.search.refine_top_k)
        self.assertEqual(
            config.likelihood_ratio.max_scores_per_distribution,
            loaded.likelihood_ratio.max_scores_per_distribution,
        )
        self.assertEqual(
            config.likelihood_ratio.minimum_identities_with_faces,
            loaded.likelihood_ratio.minimum_identities_with_faces,
        )
        self.assertEqual(
            config.likelihood_ratio.minimum_same_source_scores,
            loaded.likelihood_ratio.minimum_same_source_scores,
        )
        self.assertEqual(
            config.likelihood_ratio.minimum_different_source_scores,
            loaded.likelihood_ratio.minimum_different_source_scores,
        )
        self.assertEqual(
            config.likelihood_ratio.minimum_unique_scores_per_distribution,
            loaded.likelihood_ratio.minimum_unique_scores_per_distribution,
        )
        self.assertEqual(config.likelihood_ratio.kde_bandwidth_scale, loaded.likelihood_ratio.kde_bandwidth_scale)
        self.assertEqual(
            config.likelihood_ratio.kde_uniform_floor_weight,
            loaded.likelihood_ratio.kde_uniform_floor_weight,
        )
        self.assertEqual(config.likelihood_ratio.kde_min_density, loaded.likelihood_ratio.kde_min_density)
        self.assertEqual(config.distributed.execution_label, loaded.distributed.execution_label)
        self.assertEqual(config.distributed.node_name, loaded.distributed.node_name)
        self.assertEqual(config.distributed.auto_finalize, loaded.distributed.auto_finalize)
        self.assertEqual(
            config.distributed.validate_partial_integrity,
            loaded.distributed.validate_partial_integrity,
        )
        self.assertEqual(
            config.distributed.auto_reprocess_invalid_partials,
            loaded.distributed.auto_reprocess_invalid_partials,
        )
        self.assertEqual(config.forensics.chain_of_custody_note, loaded.forensics.chain_of_custody_note)

    def test_save_and_load_app_config_with_original_resolution(self) -> None:
        config = AppConfig(
            app=AppSettings(
                name="Inventario Faces",
                output_directory_name="saida_original",
                report_title="Relatorio Persistido",
                organization="Laboratorio Teste",
                log_level="INFO",
                use_local_temp_copy=False,
            ),
            media=MediaSettings(
                image_extensions=(".jpg", ".png"),
                video_extensions=(".mp4", ".avi"),
            ),
            video=VideoSettings(
                sampling_interval_seconds=2.0,
                max_frames_per_video=None,
                keyframe_interval_seconds=4.0,
                significant_change_threshold=0.18,
            ),
            face_model=FaceModelSettings(
                backend="insightface",
                model_name="buffalo_l",
                det_size=None,
                minimum_face_quality=0.60,
                minimum_face_size_pixels=40,
                ctx_id=0,
                providers=(),
            ),
            clustering=ClusteringSettings(
                assignment_similarity=0.52,
                candidate_similarity=0.44,
                min_cluster_size=1,
            ),
            reporting=ReportingSettings(
                compile_pdf=True,
            ),
            forensics=ForensicsSettings(
                chain_of_custody_note="Nota de teste"
            ),
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "config.yaml"
            save_app_config(config, output_path)
            loaded = load_app_config(output_path)

        self.assertIsNone(loaded.face_model.det_size)

    def test_app_settings_rejects_output_directory_with_path_segments(self) -> None:
        with self.assertRaisesRegex(ValueError, "Diretorio de saida"):
            AppSettings(
                name="Inventario Faces",
                output_directory_name="logs/saida",
                report_title="Relatorio",
                organization="Laboratorio Teste",
            )

    def test_distributed_settings_require_validation_for_auto_reprocess(self) -> None:
        with self.assertRaisesRegex(ValueError, "Recuperacao automatica de parciais"):
            DistributedSettings(
                enabled=True,
                execution_label="lote_compartilhado",
                validate_partial_integrity=False,
                auto_reprocess_invalid_partials=True,
            )

    def test_load_app_config_parses_boolean_strings_safely(self) -> None:
        yaml_content = """
app:
  name: "Inventario Faces"
  output_directory_name: "saida_personalizada"
  report_title: "Relatorio Persistido"
  organization: "Laboratorio Teste"
  log_level: "debug"
  use_local_temp_copy: "false"
media:
  image_extensions: [".jpg"]
  video_extensions: [".mp4"]
video:
  sampling_interval_seconds: 2.0
face_model:
  backend: "insightface"
  model_name: "buffalo_l"
  det_size: [640, 640]
clustering:
  assignment_similarity: 0.52
  candidate_similarity: 0.44
reporting:
  compile_pdf: "true"
forensics:
  chain_of_custody_note: "Nota de teste"
search:
  enabled: "true"
  prefer_faiss: "false"
distributed:
  enabled: "false"
  auto_finalize: "true"
  validate_partial_integrity: "true"
  auto_reprocess_invalid_partials: "false"
"""

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "config.yaml"
            output_path.write_text(yaml_content, encoding="utf-8")
            loaded = load_app_config(output_path)

        self.assertFalse(loaded.app.use_local_temp_copy)
        self.assertEqual("DEBUG", loaded.app.log_level)
        self.assertTrue(loaded.reporting.compile_pdf)
        self.assertTrue(loaded.search.enabled)
        self.assertFalse(loaded.search.prefer_faiss)
        self.assertFalse(loaded.distributed.enabled)

    def test_load_app_config_reports_source_path_for_invalid_values(self) -> None:
        yaml_content = """
app:
  name: "Inventario Faces"
  output_directory_name: "saida_personalizada"
  report_title: "Relatorio Persistido"
  organization: "Laboratorio Teste"
media:
  image_extensions: [".jpg"]
  video_extensions: [".mp4"]
video:
  sampling_interval_seconds: 2.0
face_model:
  backend: "insightface"
  model_name: "buffalo_l"
  det_size: [640, 640]
clustering:
  assignment_similarity: 0.40
  candidate_similarity: 0.60
reporting:
  compile_pdf: true
forensics:
  chain_of_custody_note: "Nota de teste"
"""

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "config.yaml"
            output_path.write_text(yaml_content, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, re.escape(str(output_path))):
                load_app_config(output_path)

    def test_default_config_uses_rigorous_forensic_defaults(self) -> None:
        loaded = load_app_config(locate_default_config())

        self.assertEqual(1.0, loaded.video.sampling_interval_seconds)
        self.assertIsNone(loaded.video.max_frames_per_video)
        self.assertEqual((800, 800), loaded.face_model.det_size)
        self.assertEqual(0.68, loaded.face_model.minimum_face_quality)
        self.assertEqual(48, loaded.face_model.minimum_face_size_pixels)
        self.assertEqual(0, loaded.face_model.ctx_id)
        self.assertEqual((), loaded.face_model.providers)
        self.assertEqual(0.62, loaded.clustering.assignment_similarity)
        self.assertEqual(0.56, loaded.clustering.candidate_similarity)
        self.assertEqual(0.58, loaded.tracking.embedding_similarity_threshold)
        self.assertEqual(0.45, loaded.tracking.minimum_total_match_score)
        self.assertEqual(12, loaded.search.coarse_top_k)
        self.assertEqual(20, loaded.search.refine_top_k)
        self.assertEqual(20000, loaded.likelihood_ratio.max_scores_per_distribution)
        self.assertEqual(2, loaded.likelihood_ratio.minimum_identities_with_faces)
        self.assertEqual(5, loaded.likelihood_ratio.minimum_same_source_scores)
        self.assertEqual(5, loaded.likelihood_ratio.minimum_different_source_scores)
        self.assertEqual(2, loaded.likelihood_ratio.minimum_unique_scores_per_distribution)
        self.assertEqual(1.0, loaded.likelihood_ratio.kde_bandwidth_scale)
        self.assertEqual(0.001, loaded.likelihood_ratio.kde_uniform_floor_weight)
        self.assertEqual(1e-12, loaded.likelihood_ratio.kde_min_density)
        self.assertIn("probabilisticos", loaded.forensics.chain_of_custody_note)

    def test_packaged_default_config_matches_project_default_config(self) -> None:
        project_defaults = (PROJECT_ROOT / "config" / "defaults.yaml").read_text(encoding="utf-8")
        packaged_defaults = (
            PROJECT_ROOT / "src" / "inventario_faces" / "config" / "defaults.yaml"
        ).read_text(encoding="utf-8")

        self.assertEqual(project_defaults, packaged_defaults)

    def test_load_default_app_config_ignores_persistent_override(self) -> None:
        from inventario_faces.infrastructure.config_loader import load_default_app_config

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "config.yaml"
            output_path.write_text(
                """
app:
  name: "Inventario Faces"
  output_directory_name: "saida_personalizada"
  report_title: "Relatorio Persistido"
  organization: "Laboratorio Teste"
media:
  image_extensions: [".jpg"]
  video_extensions: [".mp4"]
video:
  sampling_interval_seconds: 5.0
face_model:
  backend: "insightface"
  model_name: "buffalo_l"
  det_size: [640, 640]
clustering:
  assignment_similarity: 0.52
  candidate_similarity: 0.44
reporting:
  compile_pdf: true
forensics:
  chain_of_custody_note: "Nota de teste"
""",
                encoding="utf-8",
            )

            overridden = load_app_config(output_path)
            defaults_only = load_default_app_config()

        self.assertEqual(5.0, overridden.video.sampling_interval_seconds)
        self.assertEqual(1.0, defaults_only.video.sampling_interval_seconds)


if __name__ == "__main__":
    unittest.main()
