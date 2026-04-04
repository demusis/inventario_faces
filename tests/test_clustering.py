from __future__ import annotations

import unittest
from datetime import UTC, datetime
from pathlib import Path

from tests._bootstrap import PROJECT_ROOT  # noqa: F401
from inventario_faces.domain.config import ClusteringSettings
from inventario_faces.domain.entities import BoundingBox, FaceOccurrence, FaceTrack, MediaType
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

    def test_cluster_groups_tracks_as_primary_unit(self) -> None:
        service = ClusteringService(
            ClusteringSettings(
                assignment_similarity=0.8,
                candidate_similarity=0.5,
                min_cluster_size=1,
            )
        )

        tracks = [
            self._track("T1", ["O1", "O2"], [1.0, 0.0, 0.0]),
            self._track("T2", ["O3"], [0.99, 0.01, 0.0]),
            self._track("T3", ["O4"], [0.0, 1.0, 0.0]),
        ]

        clusters = service.cluster(tracks)

        self.assertEqual(2, len(clusters))
        self.assertEqual("I001", tracks[0].cluster_id)
        self.assertEqual("I001", tracks[1].cluster_id)
        self.assertEqual("I002", tracks[2].cluster_id)
        self.assertEqual(["T1", "T2"], clusters[0].track_ids)

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

    def _track(self, track_id: str, occurrence_ids: list[str], embedding: list[float]) -> FaceTrack:
        return FaceTrack(
            track_id=track_id,
            source_path=Path("sample.mp4"),
            video_path=Path("sample.mp4"),
            media_type=MediaType.VIDEO,
            sha512="hash",
            start_frame=0,
            end_frame=10,
            start_time=0.0,
            end_time=10.0,
            occurrence_ids=occurrence_ids,
            representative_embeddings=[embedding],
            average_embedding=embedding,
        )


if __name__ == "__main__":
    unittest.main()
