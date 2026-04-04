from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from inventario_faces.domain.entities import BoundingBox
from inventario_faces.utils.path_utils import ensure_directory
from inventario_faces.utils.path_utils import file_io_path, safe_stem


class ArtifactStore:
    def __init__(self, run_directory: Path) -> None:
        self.frames_directory = ensure_directory(run_directory / "artifacts" / "frames")
        self.crops_directory = ensure_directory(run_directory / "artifacts" / "crops")
        self.contexts_directory = ensure_directory(run_directory / "artifacts" / "contexts")

    def save_frame(self, image_name: str, bgr_pixels: object) -> Path:
        path = self.frames_directory / f"{image_name}.jpg"
        self._write_image(path, bgr_pixels)
        return path

    def save_crop(self, occurrence_id: str, crop_bgr: object) -> Path:
        path = self.crops_directory / f"{occurrence_id}.jpg"
        self._write_image(path, crop_bgr)
        return path

    def save_context(
        self,
        occurrence_id: str,
        image_name: str,
        bgr_pixels: object,
        bbox: BoundingBox,
    ) -> Path:
        path = self.contexts_directory / f"{occurrence_id}_{safe_stem(image_name)}.jpg"
        context = np.asarray(bgr_pixels).copy()
        top_left = (max(0, int(bbox.x1)), max(0, int(bbox.y1)))
        bottom_right = (max(0, int(bbox.x2)), max(0, int(bbox.y2)))
        cv2.rectangle(context, top_left, bottom_right, (0, 0, 255), 2)
        self._write_image(path, context)
        return path

    def _write_image(self, path: Path, bgr_pixels: object) -> None:
        ok, encoded = cv2.imencode(".jpg", np.asarray(bgr_pixels))
        if not ok:
            raise RuntimeError(f"Nao foi possivel gravar a imagem derivada: {path}")
        encoded.tofile(file_io_path(path))
