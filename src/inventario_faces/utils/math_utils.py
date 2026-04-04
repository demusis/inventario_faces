from __future__ import annotations

from typing import Iterable

import numpy as np


def l2_normalize(vector: Iterable[float]) -> list[float]:
    array = np.asarray(list(vector), dtype=np.float32)
    norm = float(np.linalg.norm(array))
    if norm == 0.0:
        return array.tolist()
    return (array / norm).tolist()


def cosine_similarity(left: Iterable[float], right: Iterable[float]) -> float:
    left_array = np.asarray(list(left), dtype=np.float32)
    right_array = np.asarray(list(right), dtype=np.float32)
    left_norm = np.linalg.norm(left_array)
    right_norm = np.linalg.norm(right_array)
    if float(left_norm) == 0.0 or float(right_norm) == 0.0:
        return 0.0
    return float(np.dot(left_array, right_array) / (left_norm * right_norm))


def average_embeddings(embeddings: Iterable[Iterable[float]]) -> list[float]:
    rows = [np.asarray(list(embedding), dtype=np.float32) for embedding in embeddings]
    if not rows:
        return []
    matrix = np.vstack(rows)
    centroid = matrix.mean(axis=0)
    return l2_normalize(centroid.tolist())
