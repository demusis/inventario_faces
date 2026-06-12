from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from inventario_faces.domain.config import ClusteringSettings
from inventario_faces.domain.entities import FaceCluster, FaceOccurrence, FaceTrack, MediaType
from inventario_faces.utils.math_utils import (
    average_embeddings,
    cosine_similarity,
    weighted_average_embeddings,
)


@dataclass(frozen=True)
class _EligibleTrack:
    track: FaceTrack
    embedding: list[float]
    weight: float


class ClusteringService:
    # Limite de seguranca: o refinamento converge muito antes disso na pratica.
    _MAX_REFINEMENT_ITERATIONS = 10

    def __init__(self, settings: ClusteringSettings) -> None:
        self._settings = settings

    def cluster(self, items: Iterable[FaceTrack | FaceOccurrence]) -> list[FaceCluster]:
        original_items = list(items)
        tracks = self._coerce_tracks(original_items)
        eligible = self._eligible_tracks(tracks)
        memberships = self._greedy_assign(eligible)
        memberships = self._refine_memberships(eligible, memberships)
        clusters = self._build_clusters(eligible, memberships)
        self._attach_candidate_matches(clusters, tracks)
        self._propagate_occurrence_assignments(original_items, tracks)
        return clusters

    def _eligible_tracks(self, tracks: list[FaceTrack]) -> list[_EligibleTrack]:
        eligible: list[_EligibleTrack] = []
        for track in tracks:
            if len(track.occurrence_ids) < self._settings.min_track_size:
                continue
            embedding = track.average_embedding or average_embeddings(track.representative_embeddings)
            if not embedding:
                continue
            eligible.append(
                _EligibleTrack(
                    track=track,
                    embedding=list(embedding),
                    weight=float(max(1, len(track.occurrence_ids))),
                )
            )
        # Ordena por forca evidencial para que o passe guloso nao dependa da
        # ordem de chegada dos arquivos; o desempate por track_id mantem o
        # resultado deterministico entre execucoes.
        eligible.sort(key=lambda item: (-item.weight, item.track.track_id))
        return eligible

    def _greedy_assign(self, eligible: list[_EligibleTrack]) -> list[list[int]]:
        memberships: list[list[int]] = []
        centroids: list[list[float]] = []
        for index, item in enumerate(eligible):
            best_cluster = -1
            best_score = -1.0
            for cluster_index, centroid in enumerate(centroids):
                similarity = cosine_similarity(item.embedding, centroid)
                if similarity > best_score:
                    best_cluster = cluster_index
                    best_score = similarity
            if best_cluster >= 0 and best_score >= self._settings.assignment_similarity:
                memberships[best_cluster].append(index)
                centroids[best_cluster] = self._centroid(eligible, memberships[best_cluster])
            else:
                memberships.append([index])
                centroids.append(list(item.embedding))
        return memberships

    def _refine_memberships(
        self,
        eligible: list[_EligibleTrack],
        memberships: list[list[int]],
    ) -> list[list[int]]:
        for _ in range(self._MAX_REFINEMENT_ITERATIONS):
            centroids = [self._centroid(eligible, members) for members in memberships]
            assignment_of = {
                member: cluster_index
                for cluster_index, members in enumerate(memberships)
                for member in members
            }
            moved = False
            for index in range(len(eligible)):
                current = assignment_of[index]
                current_score = cosine_similarity(eligible[index].embedding, centroids[current])
                best_cluster = current
                best_score = current_score
                for cluster_index, centroid in enumerate(centroids):
                    if cluster_index == current or not memberships[cluster_index]:
                        continue
                    similarity = cosine_similarity(eligible[index].embedding, centroid)
                    if similarity >= self._settings.assignment_similarity and similarity > best_score + 1e-6:
                        best_cluster = cluster_index
                        best_score = similarity
                if best_cluster != current:
                    memberships[current].remove(index)
                    memberships[best_cluster].append(index)
                    assignment_of[index] = best_cluster
                    moved = True
            memberships = [sorted(members) for members in memberships if members]
            if not moved:
                break
        return memberships

    def _centroid(self, eligible: list[_EligibleTrack], members: list[int]) -> list[float]:
        return weighted_average_embeddings(
            [eligible[member].embedding for member in members],
            [eligible[member].weight for member in members],
        )

    def _build_clusters(
        self,
        eligible: list[_EligibleTrack],
        memberships: list[list[int]],
    ) -> list[FaceCluster]:
        # Numeracao por ordem do track mais forte de cada grupo, mantendo IDs
        # estaveis e independentes da ordem interna do refinamento.
        ordered_memberships = sorted(
            (members for members in memberships if members),
            key=lambda members: min(members),
        )
        clusters: list[FaceCluster] = []
        for members in ordered_memberships:
            if len(members) < self._settings.min_cluster_size:
                continue
            cluster_id = f"I{len(clusters) + 1:03d}"
            track_ids: list[str] = []
            occurrence_ids: list[str] = []
            preview_paths = []
            representative_crop_path = None
            representative_track_id = None
            for member in members:
                track = eligible[member].track
                track.cluster_id = cluster_id
                track_ids.append(track.track_id)
                occurrence_ids.extend(track.occurrence_ids)
                if track.preview_path is not None:
                    preview_paths.append(track.preview_path)
                    if representative_crop_path is None:
                        representative_crop_path = track.preview_path
                        representative_track_id = track.track_id
            clusters.append(
                FaceCluster(
                    cluster_id=cluster_id,
                    track_ids=track_ids,
                    occurrence_ids=occurrence_ids,
                    centroid_embedding=self._centroid(eligible, members),
                    representative_crop_path=representative_crop_path,
                    representative_track_id=representative_track_id,
                    preview_paths=preview_paths,
                )
            )
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
