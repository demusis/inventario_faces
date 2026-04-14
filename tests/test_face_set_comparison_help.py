from __future__ import annotations

import unittest

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
from inventario_faces.gui.face_set_comparison_help import build_face_set_comparison_help_html


class FaceSetComparisonHelpTests(unittest.TestCase):
    def test_help_html_reflects_runtime_configuration(self) -> None:
        config = AppConfig(
            app=AppSettings(
                name="Inventario Faces",
                output_directory_name="inventario_faces_output",
                report_title="Relatório",
                organization="Laboratório",
            ),
            media=MediaSettings(
                image_extensions=(".jpg", ".png"),
                video_extensions=(".mp4", ".avi"),
            ),
            video=VideoSettings(sampling_interval_seconds=1.0, max_frames_per_video=10),
            face_model=FaceModelSettings(
                backend="insightface",
                model_name="buffalo_l",
                det_size=(800, 800),
                minimum_face_quality=0.68,
                minimum_face_size_pixels=48,
                ctx_id=0,
                providers=(),
            ),
            clustering=ClusteringSettings(
                assignment_similarity=0.52,
                candidate_similarity=0.44,
            ),
            reporting=ReportingSettings(),
            forensics=ForensicsSettings(chain_of_custody_note="Arquivos originais preservados."),
        )

        html = build_face_set_comparison_help_html(config)

        self.assertIn("Ajuda da comparação entre grupos faciais", html)
        self.assertIn("seleção automática com preferência por GPU e fallback para CPU", html)
        self.assertIn("0.52", html)
        self.assertIn("0.44", html)
        self.assertIn(".jpg, .png", html)


if __name__ == "__main__":
    unittest.main()
