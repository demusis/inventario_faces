from __future__ import annotations

from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

from inventario_faces.domain.config import VideoSettings
from inventario_faces.domain.entities import SampledFrame
from inventario_faces.utils.path_utils import file_io_path, safe_stem


class MediaDecodeError(RuntimeError):
    """Erro ao decodificar arquivo de midia."""


class VideoService:
    def __init__(self, settings: VideoSettings) -> None:
        self._settings = settings

    def load_image(self, path: Path) -> SampledFrame:
        image = cv2.imdecode(np.fromfile(file_io_path(path), dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise MediaDecodeError(f"Nao foi possivel ler a imagem: {path}")
        return SampledFrame(
            source_path=path,
            image_name=safe_stem(path.stem),
            frame_index=None,
            timestamp_seconds=None,
            bgr_pixels=image,
        )

    def sample_video(self, path: Path) -> Iterator[SampledFrame]:
        capture = self._open_capture(path)
        if not capture.isOpened():
            raise MediaDecodeError(f"Nao foi possivel abrir o video: {path}")

        try:
            fps = float(capture.get(cv2.CAP_PROP_FPS)) or 0.0
            total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
            frame_step = self._calculate_frame_step(fps)
            frame_indexes = range(0, total_frames or 1, frame_step)
            if self._settings.max_frames_per_video is not None:
                frame_indexes = range(0, total_frames or 1, frame_step)

            yielded = 0
            for frame_index in frame_indexes:
                if self._settings.max_frames_per_video is not None and yielded >= self._settings.max_frames_per_video:
                    break
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ok, frame = capture.read()
                if not ok or frame is None:
                    continue
                timestamp = float(frame_index / fps) if fps > 0 else None
                yielded += 1
                yield SampledFrame(
                    source_path=path,
                    image_name=f"{safe_stem(path.stem)}_frame_{frame_index:06d}",
                    frame_index=frame_index,
                    timestamp_seconds=timestamp,
                    bgr_pixels=frame,
                )
        finally:
            capture.release()

    def _calculate_frame_step(self, fps: float) -> int:
        if fps <= 0:
            return 1
        return max(1, int(round(fps * self._settings.sampling_interval_seconds)))

    def _open_capture(self, path: Path) -> cv2.VideoCapture:
        capture = cv2.VideoCapture(str(path))
        if capture.isOpened():
            return capture
        capture.release()
        return cv2.VideoCapture(file_io_path(path))
