from __future__ import annotations

import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path

from docx import Document

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
from inventario_faces.domain.entities import (
    BoundingBox,
    FaceCluster,
    FaceOccurrence,
    FaceSizeStatistics,
    FileRecord,
    InventoryResult,
    MediaInfoAttribute,
    MediaInfoTrack,
    MediaType,
    ProcessingSummary,
    ReportArtifacts,
)
from inventario_faces.reporting.docx_renderer import DocxReportGenerator


class DocxReportGeneratorTests(unittest.TestCase):
    def test_generate_docx_report_with_main_sections(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            logs_dir = root / "logs"
            logs_dir.mkdir(parents=True, exist_ok=True)
            (logs_dir / "run.log").write_text("linha de log", encoding="utf-8")

            generator = DocxReportGenerator(self._config())
            result = self._result(root, logs_dir)

            artifacts = generator.generate(result)

            self.assertIsNotNone(artifacts.docx_path)
            self.assertTrue(artifacts.docx_path.exists())

            document = Document(str(artifacts.docx_path))
            full_text = "\n".join(paragraph.text for paragraph in document.paragraphs)
            self.assertIn("Resumo Executivo", full_text)
            self.assertIn("Metodologia", full_text)
            self.assertIn("Anexo técnico", full_text)
            self.assertIn("aplicação de código aberto", full_text)
            self.assertIn("https://github.com/demusis/inventario_faces", full_text)

    def _config(self) -> AppConfig:
        return AppConfig(
            app=AppSettings(
                name="Inventario Faces",
                output_directory_name="inventario_faces_output",
                report_title="Relatório Forense de Inventário Facial",
                organization="Laboratório Teste",
                log_level="INFO",
            ),
            media=MediaSettings(
                image_extensions=(".jpg", ".png"),
                video_extensions=(".mp4", ".avi"),
            ),
            video=VideoSettings(
                sampling_interval_seconds=1.0,
                max_frames_per_video=10,
            ),
            face_model=FaceModelSettings(
                backend="insightface",
                model_name="buffalo_l",
                det_size=(640, 640),
                minimum_face_quality=0.6,
                minimum_face_size_pixels=40,
                ctx_id=0,
                providers=("CPUExecutionProvider",),
            ),
            clustering=ClusteringSettings(
                assignment_similarity=0.52,
                candidate_similarity=0.44,
                min_cluster_size=1,
            ),
            reporting=ReportingSettings(
                max_gallery_faces_per_group=4,
                compile_pdf=False,
            ),
            forensics=ForensicsSettings(
                chain_of_custody_note="Arquivos originais preservados."
            ),
        )

    def _result(self, root: Path, logs_dir: Path) -> InventoryResult:
        occurrence = FaceOccurrence(
            occurrence_id="O000001",
            source_path=Path("evidencia.mp4"),
            sha512="a" * 128,
            media_type=MediaType.VIDEO,
            analysis_timestamp_utc=datetime.now(UTC),
            frame_index=10,
            frame_timestamp_seconds=10.0,
            bbox=BoundingBox(10, 20, 60, 90),
            detection_score=0.83,
            embedding=[1.0, 0.0, 0.0],
            crop_path=None,
            context_image_path=None,
            cluster_id="I001",
        )
        cluster = FaceCluster(
            cluster_id="I001",
            occurrence_ids=["O000001"],
            centroid_embedding=[1.0, 0.0, 0.0],
        )
        file_record = FileRecord(
            path=Path("evidencia.mp4"),
            media_type=MediaType.VIDEO,
            sha512="a" * 128,
            size_bytes=123456,
            discovered_at_utc=datetime.now(UTC),
            modified_at_utc=datetime.now(UTC),
            media_info_tracks=(
                MediaInfoTrack(
                    track_type="Geral",
                    attributes=(
                        MediaInfoAttribute(label="Formato", value="MPEG-4"),
                        MediaInfoAttribute(label="Duração", value="00:00:10.000"),
                    ),
                ),
            ),
        )
        summary = ProcessingSummary(
            total_files=1,
            media_files=1,
            image_files=0,
            video_files=1,
            total_occurrences=1,
            total_clusters=1,
            probable_match_pairs=0,
            total_detected_face_sizes=FaceSizeStatistics(count=1, min_pixels=50, max_pixels=50, mean_pixels=50, stddev_pixels=0),
            selected_face_sizes=FaceSizeStatistics(count=1, min_pixels=50, max_pixels=50, mean_pixels=50, stddev_pixels=0),
        )
        return InventoryResult(
            run_directory=root,
            started_at_utc=datetime.now(UTC),
            finished_at_utc=datetime.now(UTC),
            root_directory=root,
            files=[file_record],
            occurrences=[occurrence],
            clusters=[cluster],
            report=ReportArtifacts(tex_path=root / "report" / "relatorio_forense.tex", pdf_path=None),
            summary=summary,
            logs_directory=logs_dir,
            manifest_path=root / "inventory" / "manifest.json",
        )


if __name__ == "__main__":
    unittest.main()
