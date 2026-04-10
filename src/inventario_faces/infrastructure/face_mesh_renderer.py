from __future__ import annotations

from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from inventario_faces.domain.entities import BoundingBox
from inventario_faces.utils.path_utils import file_io_path


def load_bgr_image(path: Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(file_io_path(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Nao foi possivel ler a imagem derivada: {path}")
    return image


def save_bgr_image(path: Path, bgr_pixels: object) -> None:
    ok, encoded = cv2.imencode(".jpg", np.asarray(bgr_pixels))
    if not ok:
        raise RuntimeError(f"Nao foi possivel gravar a imagem derivada: {path}")
    encoded.tofile(file_io_path(path))


def draw_face_mesh(
    bgr_pixels: object,
    landmarks: Iterable[tuple[float, float]],
    *,
    bbox: BoundingBox | None = None,
    translate: tuple[float, float] = (0.0, 0.0),
    draw_bbox: bool = True,
) -> np.ndarray:
    canvas = np.asarray(bgr_pixels).copy()
    if canvas.ndim != 3 or canvas.shape[2] < 3:
        raise RuntimeError("Imagem invalida para renderizacao da malha facial.")

    height, width = canvas.shape[:2]
    points = _normalize_landmarks(landmarks, width=width, height=height, translate=translate)

    if draw_bbox and bbox is not None:
        top_left = (
            max(0, int(round(bbox.x1 + translate[0]))),
            max(0, int(round(bbox.y1 + translate[1]))),
        )
        bottom_right = (
            min(width - 1, int(round(bbox.x2 + translate[0]))),
            min(height - 1, int(round(bbox.y2 + translate[1]))),
        )
        cv2.rectangle(canvas, top_left, bottom_right, (0, 0, 255), 2)

    if len(points) >= 3:
        subdiv = cv2.Subdiv2D((0, 0, width, height))
        for point in points:
            subdiv.insert((float(point[0]), float(point[1])))
        for triangle in subdiv.getTriangleList():
            p1 = (int(round(triangle[0])), int(round(triangle[1])))
            p2 = (int(round(triangle[2])), int(round(triangle[3])))
            p3 = (int(round(triangle[4])), int(round(triangle[5])))
            if _is_inside_canvas(p1, width, height) and _is_inside_canvas(p2, width, height) and _is_inside_canvas(p3, width, height):
                cv2.line(canvas, p1, p2, (0, 200, 0), 1, cv2.LINE_AA)
                cv2.line(canvas, p2, p3, (0, 200, 0), 1, cv2.LINE_AA)
                cv2.line(canvas, p3, p1, (0, 200, 0), 1, cv2.LINE_AA)

    point_radius = max(1, int(round(min(width, height) / 160)))
    for point in points:
        cv2.circle(canvas, (int(point[0]), int(point[1])), point_radius, (0, 220, 255), -1, cv2.LINE_AA)
    return canvas


def _normalize_landmarks(
    landmarks: Iterable[tuple[float, float]],
    *,
    width: int,
    height: int,
    translate: tuple[float, float],
) -> list[tuple[int, int]]:
    normalized: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    translate_x, translate_y = translate
    for raw_x, raw_y in landmarks:
        x = int(round(float(raw_x) + translate_x))
        y = int(round(float(raw_y) + translate_y))
        if x < 0 or y < 0 or x >= width or y >= height:
            continue
        point = (x, y)
        if point in seen:
            continue
        seen.add(point)
        normalized.append(point)
    return normalized


def _is_inside_canvas(point: tuple[int, int], width: int, height: int) -> bool:
    return 0 <= point[0] < width and 0 <= point[1] < height
