from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from inventario_faces.domain.config import SearchSettings
from inventario_faces.domain.entities import FaceCluster, FaceOccurrence, FaceTrack, SearchArtifacts
from inventario_faces.utils.math_utils import cosine_similarity
from inventario_faces.utils.path_utils import ensure_directory


@dataclass(frozen=True)
class SearchHit:
    entity_id: str
    score: float
    stage: str


class SearchIndexService:
    def __init__(self, settings: SearchSettings) -> None:
        self._settings = settings

    def build(
        self,
        run_directory: Path,
        tracks: list[FaceTrack],
        clusters: list[FaceCluster],
    ) -> SearchArtifacts | None:
        if not self._settings.enabled:
            return None

        search_directory = ensure_directory(run_directory / "search")
        engine = "numpy"
        faiss = None
        if self._settings.prefer_faiss:
            try:
                import faiss  # type: ignore
            except ImportError:
                faiss = None
            else:
                engine = "faiss"

        track_payload = [(track.track_id, track.average_embedding) for track in tracks if track.average_embedding]
        cluster_payload = [
            (cluster.cluster_id, cluster.centroid_embedding)
            for cluster in clusters
            if cluster.centroid_embedding
        ]

        track_index_path, track_metadata_path = self._write_index(
            search_directory=search_directory,
            prefix="tracks",
            payload=track_payload,
            engine=engine,
            faiss_module=faiss,
        )
        cluster_index_path, cluster_metadata_path = self._write_index(
            search_directory=search_directory,
            prefix="clusters",
            payload=cluster_payload,
            engine=engine,
            faiss_module=faiss,
        )
        return SearchArtifacts(
            engine=engine,
            track_index_path=track_index_path,
            track_metadata_path=track_metadata_path,
            cluster_index_path=cluster_index_path,
            cluster_metadata_path=cluster_metadata_path,
            track_vector_count=len(track_payload),
            cluster_vector_count=len(cluster_payload),
        )

    def search(
        self,
        query_embedding: Iterable[float],
        tracks: list[FaceTrack],
        clusters: list[FaceCluster],
        occurrences: list[FaceOccurrence],
    ) -> dict[str, list[SearchHit]]:
        query = list(query_embedding)
        cluster_hits = sorted(
            [
                SearchHit(cluster.cluster_id, cosine_similarity(query, cluster.centroid_embedding), "coarse_cluster")
                for cluster in clusters
                if cluster.centroid_embedding
            ],
            key=lambda item: item.score,
            reverse=True,
        )[: self._settings.coarse_top_k]

        cluster_ids = {hit.entity_id for hit in cluster_hits}
        candidate_tracks = [
            track for track in tracks if track.cluster_id in cluster_ids or not cluster_ids
        ]
        track_hits = sorted(
            [
                SearchHit(track.track_id, cosine_similarity(query, track.average_embedding), "refined_track")
                for track in candidate_tracks
                if track.average_embedding
            ],
            key=lambda item: item.score,
            reverse=True,
        )[: self._settings.refine_top_k]

        track_ids = {hit.entity_id for hit in track_hits}
        occurrence_hits = sorted(
            [
                SearchHit(occurrence.occurrence_id, cosine_similarity(query, occurrence.embedding), "track_occurrence")
                for occurrence in occurrences
                if occurrence.track_id in track_ids and occurrence.embedding
            ],
            key=lambda item: item.score,
            reverse=True,
        )[: self._settings.refine_top_k]
        return {
            "clusters": cluster_hits,
            "tracks": track_hits,
            "occurrences": occurrence_hits,
        }

    def _write_index(
        self,
        search_directory: Path,
        prefix: str,
        payload: list[tuple[str, list[float]]],
        engine: str,
        faiss_module,
    ) -> tuple[Path | None, Path | None]:
        metadata_path = search_directory / f"{prefix}_metadata.json"
        metadata_path.write_text(
            json.dumps(
                {
                    "engine": engine,
                    "ids": [item_id for item_id, _ in payload],
                    "dimension": len(payload[0][1]) if payload else 0,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        if not payload:
            return None, metadata_path

        vectors = np.asarray([embedding for _, embedding in payload], dtype=np.float32)
        if engine == "faiss" and faiss_module is not None:
            index = faiss_module.IndexFlatIP(vectors.shape[1])
            index.add(vectors)
            index_path = search_directory / f"{prefix}.faiss"
            faiss_module.write_index(index, str(index_path))
            return index_path, metadata_path

        index_path = search_directory / f"{prefix}.npy"
        np.save(index_path, vectors)
        return index_path, metadata_path
