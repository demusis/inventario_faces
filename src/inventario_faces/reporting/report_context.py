from __future__ import annotations

from collections import defaultdict

from inventario_faces.domain.entities import FaceCluster, FaceOccurrence, FaceTrack, InventoryResult, KeyFrame, MediaType


def occurrences_by_track(result: InventoryResult) -> dict[str, list[FaceOccurrence]]:
    grouped: dict[str, list[FaceOccurrence]] = defaultdict(list)
    for occurrence in result.occurrences:
        if occurrence.track_id is not None:
            grouped[occurrence.track_id].append(occurrence)
    return grouped


def keyframes_by_track(result: InventoryResult) -> dict[str, list[KeyFrame]]:
    grouped: dict[str, list[KeyFrame]] = defaultdict(list)
    for keyframe in result.keyframes:
        grouped[keyframe.track_id].append(keyframe)
    return grouped


def tracks_by_cluster(result: InventoryResult) -> dict[str, list[FaceTrack]]:
    grouped: dict[str, list[FaceTrack]] = defaultdict(list)
    for track in result.tracks:
        if track.cluster_id is not None:
            grouped[track.cluster_id].append(track)
    return grouped


def tracks_by_video(result: InventoryResult) -> dict[str, list[FaceTrack]]:
    grouped: dict[str, list[FaceTrack]] = defaultdict(list)
    for track in result.tracks:
        if track.media_type != MediaType.VIDEO:
            continue
        grouped[str(track.source_path)].append(track)
    return grouped


def candidate_cluster_pairs(clusters: list[FaceCluster]) -> list[tuple[str, str]]:
    pairs = {
        tuple(sorted((cluster.cluster_id, candidate)))
        for cluster in clusters
        for candidate in cluster.candidate_cluster_ids
    }
    return sorted(pairs)
