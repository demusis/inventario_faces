from __future__ import annotations

import unittest
from pathlib import Path

from tests._bootstrap import PROJECT_ROOT  # noqa: F401
from inventario_faces.domain.entities import BoundingBox, FaceSetComparisonEntry
from inventario_faces.infrastructure.config_loader import load_app_config
from inventario_faces.services.clustering_service import ClusteringService
from inventario_faces.services.face_set_comparison_service import FaceSetComparisonService
from inventario_faces.services.hashing_service import HashingService
from inventario_faces.services.lr_calibration import LikelihoodRatioCalibrator
from inventario_faces.services.scanner_service import ScannerService
from inventario_faces.services.video_service import VideoService


def _entry(entry_id: str, set_label: str, embedding: list[float], source: str) -> FaceSetComparisonEntry:
    return FaceSetComparisonEntry(
        entry_id=entry_id,
        set_label=set_label,
        source_path=Path(source),
        sha512="0" * 8,
        track_id=f"T{entry_id}",
        occurrence_id=f"O{entry_id}",
        keyframe_id=None,
        bbox=BoundingBox(0.0, 0.0, 10.0, 10.0),
        detection_score=0.9,
        quality_score=0.8,
        sharpness=None,
        brightness=None,
        illumination=None,
        frontality=None,
        embedding_dimension=len(embedding),
        embedding_source="test",
        crop_path=None,
        context_image_path=None,
        embedding=list(embedding),
    )


class ClusterEmbeddingsTests(unittest.TestCase):
    def setUp(self) -> None:
        self._clustering = ClusteringService(load_app_config().clustering)

    def test_groups_similar_and_separates_dissimilar(self) -> None:
        items = [
            ("a", [1.0, 0.0, 0.0], 1.0),
            ("b", [0.99, 0.01, 0.0], 1.0),  # quase identico a 'a'
            ("c", [0.0, 1.0, 0.0], 1.0),  # ortogonal
        ]
        groups = self._clustering.cluster_embeddings(items)
        flattened = {frozenset(group) for group in groups}
        self.assertIn(frozenset({"a", "b"}), flattened)
        self.assertIn(frozenset({"c"}), flattened)

    def test_singletons_are_preserved(self) -> None:
        items = [
            ("x", [1.0, 0.0, 0.0], 1.0),
            ("y", [0.0, 1.0, 0.0], 1.0),
            ("z", [0.0, 0.0, 1.0], 1.0),
        ]
        groups = self._clustering.cluster_embeddings(items)
        self.assertEqual(len(groups), 3)

    def test_is_deterministic_regardless_of_input_order(self) -> None:
        base = [
            ("a", [1.0, 0.0, 0.0], 1.0),
            ("b", [0.98, 0.02, 0.0], 1.0),
            ("c", [0.0, 1.0, 0.0], 1.0),
            ("d", [0.0, 0.97, 0.03], 1.0),
        ]
        forward = self._clustering.cluster_embeddings(base)
        backward = self._clustering.cluster_embeddings(list(reversed(base)))
        self.assertEqual(
            {frozenset(group) for group in forward},
            {frozenset(group) for group in backward},
        )

    def test_ignores_items_without_embedding(self) -> None:
        items = [("a", [1.0, 0.0], 1.0), ("b", [], 1.0)]
        groups = self._clustering.cluster_embeddings(items)
        self.assertEqual([sorted(g) for g in groups], [["a"]])


class AssignIndividualsTests(unittest.TestCase):
    def setUp(self) -> None:
        config = load_app_config()
        self._service = FaceSetComparisonService(
            config=config,
            scanner_service=ScannerService(config.media),
            hashing_service=HashingService(),
            media_service=VideoService(config.video),
            tracking_service=object(),
            face_analyzer_factory=lambda: None,
            lr_calibrator=LikelihoodRatioCalibrator(config),
            clustering_service=ClusteringService(config.clustering),
        )

    def test_repeated_individual_shares_id_across_media(self) -> None:
        entries = [
            _entry("A1", "A", [1.0, 0.0, 0.0], "foto1.jpg"),
            _entry("A2", "A", [0.99, 0.01, 0.0], "foto2.jpg"),  # mesma pessoa, outra midia
            _entry("A3", "A", [0.0, 1.0, 0.0], "foto3.jpg"),  # outra pessoa
        ]
        assigned = self._service._assign_individuals(entries, "A")
        by_id = {item.entry_id: item.individual_id for item in assigned}
        self.assertEqual(by_id["A1"], by_id["A2"])
        self.assertNotEqual(by_id["A1"], by_id["A3"])
        for individual_id in by_id.values():
            self.assertTrue(individual_id.startswith("A-IND-"))

    def test_count_individuals_reports_distinct_and_repeated(self) -> None:
        entries = [
            _entry("A1", "A", [1.0, 0.0, 0.0], "foto1.jpg"),
            _entry("A2", "A", [0.99, 0.01, 0.0], "foto2.jpg"),
            _entry("A3", "A", [0.0, 1.0, 0.0], "foto3.jpg"),
        ]
        assigned = self._service._assign_individuals(entries, "A")
        distinct, repeated = self._service._count_individuals(assigned)
        self.assertEqual(distinct, 2)
        self.assertEqual(repeated, 1)

    def test_entry_without_embedding_gets_no_individual(self) -> None:
        entries = [
            _entry("A1", "A", [1.0, 0.0, 0.0], "foto1.jpg"),
            _entry("A2", "A", [], "foto2.jpg"),
        ]
        assigned = self._service._assign_individuals(entries, "A")
        by_id = {item.entry_id: item.individual_id for item in assigned}
        self.assertIsNotNone(by_id["A1"])
        self.assertIsNone(by_id["A2"])


if __name__ == "__main__":
    unittest.main()
