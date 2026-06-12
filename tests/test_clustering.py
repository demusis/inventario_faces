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

    def test_cluster_assignment_does_not_depend_on_input_order(self) -> None:
        service = ClusteringService(
            ClusteringSettings(
                assignment_similarity=0.8,
                candidate_similarity=0.5,
                min_cluster_size=1,
            )
        )

        def build_tracks() -> list:
            return [
                self._track("T1", ["O1", "O2", "O3"], [1.0, 0.0, 0.0]),
                self._track("T2", ["O4"], [0.97, 0.05, 0.0]),
                self._track("T3", ["O5"], [0.0, 1.0, 0.0]),
                self._track("T4", ["O6", "O7"], [0.02, 0.99, 0.0]),
            ]

        forward = build_tracks()
        service.cluster(forward)
        forward_groups = {
            frozenset(track.track_id for track in forward if track.cluster_id == cluster_id)
            for cluster_id in {track.cluster_id for track in forward}
        }

        reversed_tracks = build_tracks()[::-1]
        service.cluster(reversed_tracks)
        reversed_groups = {
            frozenset(track.track_id for track in reversed_tracks if track.cluster_id == cluster_id)
            for cluster_id in {track.cluster_id for track in reversed_tracks}
        }

        self.assertEqual(forward_groups, reversed_groups)
        self.assertEqual({frozenset({"T1", "T2"}), frozenset({"T3", "T4"})}, forward_groups)

    def test_centroid_weights_tracks_by_occurrence_count(self) -> None:
        service = ClusteringService(
            ClusteringSettings(
                assignment_similarity=0.9,
                candidate_similarity=0.5,
                min_cluster_size=1,
            )
        )

        heavy = self._track("T1", [f"O{i}" for i in range(10)], [1.0, 0.0, 0.0])
        light = self._track("T2", ["O90"], [0.92, 0.39, 0.0])

        clusters = service.cluster([heavy, light])

        self.assertEqual(1, len(clusters))
        centroid = clusters[0].centroid_embedding
        # Com peso 10:1, o centroide deve permanecer muito proximo do track forte.
        self.assertGreater(centroid[0], 0.99)
        self.assertLess(centroid[1], 0.06)

    def test_refinement_reassigns_track_to_closer_cluster(self) -> None:
        service = ClusteringService(
            ClusteringSettings(
                assignment_similarity=0.7,
                candidate_similarity=0.5,
                min_cluster_size=1,
            )
        )

        # Em um passe guloso puro na ordem abaixo, o track intermediario seria
        # capturado pelo primeiro cluster; o refinamento deve move-lo para o
        # cluster cujo centroide final e mais proximo.
        tracks = [
            self._track("T1", ["O1", "O2", "O3"], [1.0, 0.0, 0.0]),
            self._track("T2", ["O4", "O5"], [0.76, 0.65, 0.0]),
            self._track("T3", ["O6", "O7"], [0.62, 0.78, 0.0]),
            self._track("T4", ["O8"], [0.0, 1.0, 0.0]),
        ]

        service.cluster(tracks)

        t2_cluster = next(track.cluster_id for track in tracks if track.track_id == "T2")
        t3_cluster = next(track.cluster_id for track in tracks if track.track_id == "T3")
        self.assertEqual(t2_cluster, t3_cluster)

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
