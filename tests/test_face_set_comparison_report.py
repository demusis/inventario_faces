from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from docx import Document

from tests._bootstrap import PROJECT_ROOT  # noqa: F401
from inventario_faces.domain.config import FaceModelSettings
from inventario_faces.domain.entities import (
    BoundingBox,
    FaceSetComparisonEntry,
    FaceSetComparisonInput,
    FaceSetComparisonMatch,
    FaceSetComparisonResult,
    FaceSetComparisonSummary,
)
from inventario_faces.infrastructure.config_loader import load_app_config, save_app_config
from inventario_faces.reporting.face_set_comparison_docx_renderer import (
    REPORT_FILE_NAME,
    FaceSetComparisonDocxReportGenerator,
)


def _entry(entry_id: str, set_label: str, source: str) -> FaceSetComparisonEntry:
    return FaceSetComparisonEntry(
        entry_id=entry_id,
        set_label=set_label,
        source_path=Path(source),
        sha512="ab" * 64,
        track_id=f"T{entry_id}",
        occurrence_id=f"O{entry_id}",
        keyframe_id=None,
        bbox=BoundingBox(10.0, 10.0, 110.0, 130.0),
        detection_score=0.88,
        quality_score=0.74,
        sharpness=0.4,
        brightness=0.5,
        illumination=0.8,
        frontality=0.9,
        embedding_dimension=512,
        embedding_source="keyframe",
        crop_path=None,
        context_image_path=None,
    )


def _document_text(path: Path) -> str:
    document = Document(str(path))
    parts = [paragraph.text for paragraph in document.paragraphs]
    for table in document.tables:
        for row in table.rows:
            parts.extend(cell.text for cell in row.cells)
    return "\n".join(parts)


class FaceSetComparisonReportTests(unittest.TestCase):
    def test_report_contains_verdict_inputs_margin_and_ranking(self) -> None:
        config = load_app_config()
        timestamp = datetime(2026, 11, 9, 12, 0, tzinfo=timezone.utc)
        result = FaceSetComparisonResult(
            run_directory=Path("execucao"),
            started_at_utc=timestamp,
            finished_at_utc=timestamp,
            logs_directory=Path("execucao/logs"),
            export_directory=Path("execucao/comparison"),
            manifest_path=Path("execucao/comparison/face_set_comparison.json"),
            set_a_inputs=[
                FaceSetComparisonInput("A", Path("padrão.png"), "a" * 128, 1, 1, 1, 1, still_image_margin_px=158)
            ],
            set_b_inputs=[FaceSetComparisonInput("B", Path("questionada.png"), "b" * 128, 1, 1, 1, 1)],
            set_a_faces=[_entry("CA_000001", "A", "padrão.png")],
            set_b_faces=[_entry("CB_000002", "B", "questionada.png")],
            matches=[
                FaceSetComparisonMatch(
                    rank=1,
                    left_entry_id="CA_000001",
                    right_entry_id="CB_000002",
                    left_track_id="TCA_000001",
                    right_track_id="TCB_000002",
                    similarity=0.1805,
                    classification="below_threshold",
                )
            ],
            summary=FaceSetComparisonSummary(
                set_a_images=1,
                set_b_images=1,
                set_a_selected_faces=1,
                set_b_selected_faces=1,
                total_pair_comparisons=1,
                best_similarity=0.1805,
                candidate_threshold=0.56,
                assignment_threshold=0.62,
            ),
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            path = FaceSetComparisonDocxReportGenerator(config).generate(
                result,
                {"questionada.png": "below_threshold"},
                Path(temp_dir),
            )
            self.assertEqual(path.name, REPORT_FILE_NAME)
            text = _document_text(path)

        self.assertIn("Comparação entre Grupos Faciais", text)
        self.assertIn("Abaixo do limiar", text)
        self.assertIn("0.1805", text)
        self.assertIn("a" * 128, text)
        self.assertIn("margem neutra de 158px", text)
        self.assertIn("CA_000001", text)
        self.assertIn("Advertência pericial", text)
        self.assertIn("margem para imagens estáticas=0.50", text)


class StillImageMarginConfigTests(unittest.TestCase):
    def test_default_and_round_trip(self) -> None:
        config = load_app_config()
        self.assertEqual(config.face_model.still_image_margin_ratio, 0.5)
        changed = replace(config, face_model=replace(config.face_model, still_image_margin_ratio=0.25))
        with tempfile.TemporaryDirectory() as temp_dir:
            path = save_app_config(changed, Path(temp_dir) / "config.yaml")
            self.assertEqual(load_app_config(path).face_model.still_image_margin_ratio, 0.25)

    def test_rejects_out_of_range(self) -> None:
        for value in (-0.1, 2.5):
            with self.assertRaises(ValueError):
                FaceModelSettings(backend="onnx", model_name="buffalo_l", det_size=(640, 640), still_image_margin_ratio=value)


if __name__ == "__main__":
    unittest.main()
