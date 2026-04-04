from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from inventario_faces.domain.config import ClusteringSettings
from inventario_faces.domain.entities import FaceCluster, FaceOccurrence, FaceTrack, MediaType
from inventario_faces.utils.math_utils import average_embeddings, cosine_similarity


@dataclass
class _WorkingCluster:
    cluster_id: str
    track_ids: list[str] = field(default_factory=list)
    occurrence_ids: list[str] = field(default_factory=list)
    embeddings: list[list[float]] = field(default_factory=list)
    representative_crop_path: Path | None = None
    representative_track_id: str | None = None
    preview_paths: list[Path] = field(default_factory=list)

    @property
    def centroid(self) -> list[float]:
        return average_embeddings(self.embeddings)


class ClusteringService:
    def __init__(self, settings: ClusteringSettings) -> None:
        self._settings = settings

    def cluster(self, items: Iterable[FaceTrack | FaceOccurrence]) -> list[FaceCluster]:
        original_items = list(items)
        tracks = self._coerce_tracks(original_items)
        working_clusters: list[_WorkingCluster] = []

        for track in tracks:
            if len(track.occurrence_ids) < self._settings.min_track_size:
                continue
            embedding = track.average_embedding or average_embeddings(track.representative_embeddings)
            if not embedding:
                continue
            best_cluster: _WorkingCluster | None = None
            best_score = -1.0

            for cluster in working_clusters:
                similarity = cosine_similarity(embedding, cluster.centroid)
                if similarity > best_score:
                    best_cluster = cluster
                    best_score = similarity

            if best_cluster is not None and best_score >= self._settings.assignment_similarity:
                assigned_cluster = best_cluster
            else:
                assigned_cluster = _WorkingCluster(cluster_id=f"I{len(working_clusters) + 1:03d}")
                working_clusters.append(assigned_cluster)

            assigned_cluster.track_ids.append(track.track_id)
            assigned_cluster.occurrence_ids.extend(track.occurrence_ids)
            assigned_cluster.embeddings.append(embedding)
            if assigned_cluster.representative_crop_path is None and track.preview_path is not None:
                assigned_cluster.representative_crop_path = track.preview_path
                assigned_cluster.representative_track_id = track.track_id
            if track.preview_path is not None:
                assigned_cluster.preview_paths.append(track.preview_path)
            track.cluster_id = assigned_cluster.cluster_id

        clusters: list[FaceCluster] = []
        for cluster in working_clusters:
            if len(cluster.track_ids) < self._settings.min_cluster_size:
                continue
            clusters.append(
                FaceCluster(
                    cluster_id=cluster.cluster_id,
                    track_ids=list(cluster.track_ids),
                    occurrence_ids=list(cluster.occurrence_ids),
                    centroid_embedding=cluster.centroid,
                    representative_crop_path=cluster.representative_crop_path,
                    representative_track_id=cluster.representative_track_id,
                    preview_paths=list(cluster.preview_paths),
                )
            )

        self._attach_candidate_matches(clusters, tracks)
        self._propagate_occurrence_assignments(original_items, tracks)
        return clusters

    def _attach_candidate_matches(
        self,
        clusters: list[FaceCluster],
        tracks: list[FaceTrack],
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
        for track in tracks:
            if track.cluster_id is None:
                continue
            cluster = by_cluster.get(track.cluster_id)
            track.candidate_cluster_ids = list(cluster.candidate_cluster_ids if cluster else [])

    def _coerce_tracks(self, items: Iterable[FaceTrack | FaceOccurrence]) -> list[FaceTrack]:
        tracks: list[FaceTrack] = []
        for index, item in enumerate(items, start=1):
            if isinstance(item, FaceTrack):
                tracks.append(item)
                continue
            tracks.append(
                FaceTrack(
                    track_id=item.track_id or f"T{index:06d}",
                    source_path=item.source_path,
                    video_path=item.source_path if item.media_type == MediaType.VIDEO else None,
                    media_type=item.media_type,
                    sha512=item.sha512,
                    start_frame=item.frame_index,
                    end_frame=item.frame_index,
                    start_time=item.frame_timestamp_seconds,
                    end_time=item.frame_timestamp_seconds,
                    occurrence_ids=[item.occurrence_id],
                    representative_embeddings=[item.embedding] if item.embedding else [],
                    average_embedding=list(item.embedding),
                    best_occurrence_id=item.occurrence_id,
                    preview_path=item.crop_path,
                    top_crop_paths=[item.crop_path] if item.crop_path is not None else [],
                )
            )
            if item.track_id is None:
                item.track_id = tracks[-1].track_id
        return tracks

    def _propagate_occurrence_assignments(
        self,
        original_items: list[FaceTrack | FaceOccurrence],
        tracks: list[FaceTrack],
    ) -> None:
        track_map = {track.track_id: track for track in tracks}
        for item in original_items:
            if not isinstance(item, FaceOccurrence):
                continue
            track = track_map.get(item.track_id or "")
            if track is None:
                continue
            item.cluster_id = track.cluster_id
            item.suggested_cluster_ids = list(track.candidate_cluster_ids)
