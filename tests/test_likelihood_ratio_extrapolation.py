from __future__ import annotations

import unittest
from pathlib import Path

from tests._bootstrap import PROJECT_ROOT  # noqa: F401
from inventario_faces.domain.entities import (
    FaceSetComparisonCalibration,
    FaceSetComparisonCalibrationSummary,
    FaceSetComparisonMatch,
)
from inventario_faces.infrastructure.config_loader import load_app_config
from inventario_faces.services.clustering_service import ClusteringService
from inventario_faces.services.hashing_service import HashingService
from inventario_faces.services.inventory_service import InventoryService
from inventario_faces.services.scanner_service import ScannerService
from inventario_faces.services.video_service import VideoService


def _match(rank: int, similarity: float) -> FaceSetComparisonMatch:
    return FaceSetComparisonMatch(
        rank=rank,
        left_entry_id=f"A{rank}",
        right_entry_id=f"B{rank}",
        left_track_id=f"TA{rank}",
        right_track_id=f"TB{rank}",
        similarity=similarity,
        classification="candidate",
    )


class LikelihoodRatioExtrapolationTests(unittest.TestCase):
    def setUp(self) -> None:
        config = load_app_config()
        self.service = InventoryService(
            config=config,
            scanner_service=ScannerService(config.media),
            hashing_service=HashingService(),
            media_service=VideoService(config.video),
            clustering_service=ClusteringService(config.clustering),
            report_generator=None,
            face_analyzer_factory=lambda: None,
        )
        self.calibration = FaceSetComparisonCalibration(
            summary=FaceSetComparisonCalibrationSummary(
                dataset_root=Path("dataset"),
                support_ready=True,
            ),
            genuine_scores=[0.62, 0.66, 0.70, 0.74, 0.78, 0.82, 0.86, 0.90],
            impostor_scores=[0.10, 0.14, 0.18, 0.22, 0.26, 0.30, 0.34, 0.38],
        )

    def test_match_outside_calibration_support_is_flagged(self) -> None:
        matches = [
            _match(1, 0.97),  # acima do maior score de calibracao (0.90)
            _match(2, 0.55),  # dentro do suporte [0.10, 0.90]
            _match(3, 0.02),  # abaixo do menor score de calibracao (0.10)
        ]

        calibrated = self.service._lr_calibrator._apply_face_set_likelihood_ratio_calibration(
            matches,
            self.calibration,
        )

        by_rank = {item.rank: item for item in calibrated}
        self.assertTrue(by_rank[1].outside_calibration_support)
        self.assertFalse(by_rank[2].outside_calibration_support)
        self.assertTrue(by_rank[3].outside_calibration_support)
        self.assertIn("fora do intervalo de calibracao", by_rank[1].evidence_label)
        self.assertNotIn("fora do intervalo de calibracao", by_rank[2].evidence_label)

    def test_all_matches_keep_likelihood_ratio_values(self) -> None:
        matches = [_match(1, 0.97), _match(2, 0.55)]

        calibrated = self.service._lr_calibrator._apply_face_set_likelihood_ratio_calibration(
            matches,
            self.calibration,
        )

        for item in calibrated:
            self.assertIsNotNone(item.likelihood_ratio)
            self.assertIsNotNone(item.log10_likelihood_ratio)
            self.assertIsNotNone(item.evidence_label)


if __name__ == "__main__":
    unittest.main()
