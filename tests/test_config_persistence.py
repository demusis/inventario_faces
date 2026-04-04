from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tests._bootstrap import PROJECT_ROOT  # noqa: F401
from inventario_faces.domain.config import (
    AppConfig,
    AppSettings,
    ClusteringSettings,
    FaceModelSettings,
    ForensicsSettings,
    MediaSettings,
    ReportingSettings,
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
            ),
            reporting=ReportingSettings(
                max_gallery_faces_per_group=8,
                compile_pdf=False,
            ),
            forensics=ForensicsSettings(
                chain_of_custody_note="Nota de teste"
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
        self.assertEqual(config.face_model.det_size, loaded.face_model.det_size)
        self.assertEqual(config.face_model.minimum_face_quality, loaded.face_model.minimum_face_quality)
        self.assertEqual(config.face_model.minimum_face_size_pixels, loaded.face_model.minimum_face_size_pixels)
        self.assertEqual(config.face_model.providers, loaded.face_model.providers)
        self.assertEqual(config.clustering.min_cluster_size, loaded.clustering.min_cluster_size)
        self.assertEqual(config.reporting.compile_pdf, loaded.reporting.compile_pdf)
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
                max_gallery_faces_per_group=6,
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
