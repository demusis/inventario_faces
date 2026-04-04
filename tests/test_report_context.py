from __future__ import annotations

import unittest
from pathlib import Path

from tests._bootstrap import PROJECT_ROOT  # noqa: F401
from inventario_faces.domain.entities import FaceTrack, MediaType, TrackQualityStatistics
from inventario_faces.reporting.report_context import mean_pairwise_track_similarity
from inventario_faces.reporting.report_support import format_group_similarity


class ReportContextTests(unittest.TestCase):
    def test_mean_pairwise_track_similarity_returns_average_of_all_pairs(self) -> None:
        tracks = [
            self._track("T1", [1.0, 0.0, 0.0]),
            self._track("T2", [1.0, 0.0, 0.0]),
            self._track("T3", [0.0, 1.0, 0.0]),
        ]

        similarity = mean_pairwise_track_similarity(tracks)

        self.assertIsNotNone(similarity)
        self.assertAlmostEqual(1.0 / 3.0, similarity or 0.0, places=6)
        self.assertEqual("0.333", format_group_similarity(similarity, len(tracks)))

    def test_mean_pairwise_track_similarity_handles_unitary_group(self) -> None:
        similarity = mean_pairwise_track_similarity([self._track("T1", [1.0, 0.0, 0.0])])

        self.assertIsNone(similarity)
        self.assertEqual("n/a (grupo unitário)", format_group_similarity(similarity, 1))

    def _track(self, track_id: str, average_embedding: list[float]) -> FaceTrack:
        return FaceTrack(
            track_id=track_id,
            source_path=Path(f"{track_id}.mp4"),
            video_path=Path(f"{track_id}.mp4"),
            media_type=MediaType.VIDEO,
            sha512="a" * 128,
            start_frame=0,
            end_frame=0,
            start_time=0.0,
            end_time=0.0,
            average_embedding=average_embedding,
            quality_statistics=TrackQualityStatistics(),
        )


if __name__ == "__main__":
    unittest.main()
