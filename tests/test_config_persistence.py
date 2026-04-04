from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tests._bootstrap import PROJECT_ROOT  # noqa: F401
from inventario_faces.domain.config import (
    AppConfig,
    AppSettings,
    ClusteringSettings,
    EnhancementSettings,
    FaceModelSettings,
    ForensicsSettings,
    MediaSettings,
    ReportingSettings,
    SearchSettings,
    TrackingSettings,
    VideoSettings,
)
from inventario_faces.infrastructure.config_loader import load_app_config, save_app_config


class ConfigPersistenceTests(unittest.TestCase):
    def test_save_and_load_app_config_round_trip(self) -> None:
        config = AppConfig(
            app=AppSettings(
                name="Inventario Faces",
                output_directory_name="saida_personalizada",
                report_title="Relatorio Persistido",
                organization="Laboratorio Teste",
                log_level="DEBUG",
                mediainfo_directory=r"C:\Ferramentas\MediaInfo",
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
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "config.yaml"
            save_app_config(config, output_path)
            loaded = load_app_config(output_path)

        self.assertEqual(config.app.output_directory_name, loaded.app.output_directory_name)
        self.assertEqual(config.app.log_level, loaded.app.log_level)
        self.assertEqual(config.app.mediainfo_directory, loaded.app.mediainfo_directory)
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
        self.assertEqual(config.forensics.chain_of_custody_note, loaded.forensics.chain_of_custody_note)

    def test_save_and_load_app_config_with_original_resolution(self) -> None:
        config = AppConfig(
            app=AppSettings(
                name="Inventario Faces",
                output_directory_name="saida_original",
                report_title="Relatorio Persistido",
                organization="Laboratorio Teste",
                log_level="INFO",
                mediainfo_directory=None,
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


if __name__ == "__main__":
    unittest.main()
