from __future__ import annotations

import unittest
from datetime import UTC, datetime
from pathlib import Path

from tests._bootstrap import PROJECT_ROOT  # noqa: F401
from inventario_faces.domain.config import ClusteringSettings
from inventario_faces.domain.entities import BoundingBox, FaceOccurrence, MediaType
from inventario_faces.services.clustering_service import ClusteringService


class ClusteringServiceTests(unittest.TestCase):
    def test_cluster_groups_similar_embeddings(self) -> None:
        service = ClusteringService(
            ClusteringSettings(
                assignment_similarity=0.8,
                candidate_similarity=0.5,
                min_cluster_size=1,
            )
        )

        occurrences = [
            self._occurrence("O1", [1.0, 0.0, 0.0]),
            self._occurrence("O2", [0.98, 0.02, 0.0]),
            self._occurrence("O3", [0.0, 1.0, 0.0]),
        ]

        clusters = service.cluster(occurrences)

        self.assertEqual(2, len(clusters))
        self.assertEqual("I001", occurrences[0].cluster_id)
        self.assertEqual("I001", occurrences[1].cluster_id)
        self.assertEqual("I002", occurrences[2].cluster_id)

    def _occurrence(self, occurrence_id: str, embedding: list[float]) -> FaceOccurrence:
        return FaceOccurrence(
            occurrence_id=occurrence_id,
            source_path=Path("sample.jpg"),
            sha512="hash",
            media_type=MediaType.IMAGE,
            analysis_timestamp_utc=datetime.now(tz=UTC),
            frame_index=None,
            frame_timestamp_seconds=None,
            bbox=BoundingBox(0, 0, 10, 10),
            detection_score=0.99,
            embedding=embedding,
            crop_path=None,
        )


if __name__ == "__main__":
    unittest.main()
