from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from inventario_faces.domain.config import ClusteringSettings
from inventario_faces.domain.entities import FaceCluster, FaceOccurrence
from inventario_faces.utils.math_utils import average_embeddings, cosine_similarity


@dataclass
class _WorkingCluster:
    cluster_id: str
    occurrence_ids: list[str] = field(default_factory=list)
    embeddings: list[list[float]] = field(default_factory=list)
    representative_crop_path: Path | None = None

    @property
    def centroid(self) -> list[float]:
        return average_embeddings(self.embeddings)


class ClusteringService:
    def __init__(self, settings: ClusteringSettings) -> None:
        self._settings = settings

    def cluster(self, occurrences: list[FaceOccurrence]) -> list[FaceCluster]:
        working_clusters: list[_WorkingCluster] = []

        for occurrence in occurrences:
            best_cluster: _WorkingCluster | None = None
            best_score = -1.0

            for cluster in working_clusters:
                similarity = cosine_similarity(occurrence.embedding, cluster.centroid)
                if similarity > best_score:
                    best_cluster = cluster
                    best_score = similarity

            if best_cluster is not None and best_score >= self._settings.assignment_similarity:
                assigned_cluster = best_cluster
            else:
                assigned_cluster = _WorkingCluster(cluster_id=f"I{len(working_clusters) + 1:03d}")
                working_clusters.append(assigned_cluster)

            assigned_cluster.occurrence_ids.append(occurrence.occurrence_id)
            assigned_cluster.embeddings.append(occurrence.embedding)
            if assigned_cluster.representative_crop_path is None and occurrence.crop_path is not None:
                assigned_cluster.representative_crop_path = occurrence.crop_path
            occurrence.cluster_id = assigned_cluster.cluster_id

        clusters: list[FaceCluster] = []
        for cluster in working_clusters:
            if len(cluster.occurrence_ids) < self._settings.min_cluster_size:
                continue
            clusters.append(
                FaceCluster(
                    cluster_id=cluster.cluster_id,
                    occurrence_ids=list(cluster.occurrence_ids),
                    centroid_embedding=cluster.centroid,
                    representative_crop_path=cluster.representative_crop_path,
                )
            )

        self._attach_candidate_matches(clusters, occurrences)
        return clusters

    def _attach_candidate_matches(
        self,
        clusters: list[FaceCluster],
        occurrences: list[FaceOccurrence],
    ) -> None:
        for index, cluster in enumerate(clusters):
            candidates: list[str] = []
            for other_index, other_cluster in enumerate(clusters):
                if index == other_index:
                    continue
                similarity = cosine_similarity(cluster.centroid_embedding, other_cluster.centroid_embedding)
                if similarity >= self._settings.candidate_similarity:
                    candidates.append(other_cluster.cluster_id)
            cluster.candidate_cluster_ids = sorted(candidates)

        by_cluster = {cluster.cluster_id: cluster for cluster in clusters}
        for occurrence in occurrences:
            if occurrence.cluster_id is None:
                continue
            cluster = by_cluster.get(occurrence.cluster_id)
            occurrence.suggested_cluster_ids = list(cluster.candidate_cluster_ids if cluster else [])
