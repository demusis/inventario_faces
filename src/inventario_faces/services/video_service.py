from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

import cv2
import numpy as np

from inventario_faces.domain.config import VideoSettings
from inventario_faces.domain.entities import SampledFrame
from inventario_faces.utils.path_utils import file_io_path, safe_stem


class MediaDecodeError(RuntimeError):
    """Erro ao decodificar arquivo de midia."""


@dataclass(frozen=True)
class VideoSamplingInfo:
    """Metadados operacionais da amostragem aplicada a um vídeo."""

    fps: float
    total_frames: int | None
    duration_seconds: float | None
    frame_step: int
    actual_sampling_interval_seconds: float | None
    planned_sample_count: int | None
    max_sample_count: int | None


class VideoService:
    """Decodifica imagens e produz amostras de vídeo sem alterar os originais."""

    def __init__(self, settings: VideoSettings) -> None:
        self._settings = settings
        os.environ.setdefault("OPENCV_FFMPEG_READ_ATTEMPTS", "16384")

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
            original_bgr_pixels=image,
        )

    def describe_video(self, path: Path) -> VideoSamplingInfo:
        """Lê apenas os metadados necessários para planejar a amostragem do vídeo."""

        capture = self._open_capture(path)
        if not capture.isOpened():
            raise MediaDecodeError(f"Nao foi possivel abrir o video: {path}")
        try:
            return self._read_sampling_info(capture)
        finally:
            capture.release()

    def sample_video(
        self,
        path: Path,
        metadata_callback: Callable[[VideoSamplingInfo], None] | None = None,
    ) -> Iterator[SampledFrame]:
        """Produz quadros amostrados com seus índices reais e metadados de amostragem."""

        capture = self._open_capture(path)
        if not capture.isOpened():
            raise MediaDecodeError(f"Nao foi possivel abrir o video: {path}")

        try:
            sampling_info = self._read_sampling_info(capture)
            if metadata_callback is not None:
                metadata_callback(sampling_info)

            yielded = 0
            current_frame_index = 0
            next_target_index = 0

            while True:
                if (
                    sampling_info.max_sample_count is not None
                    and yielded >= sampling_info.max_sample_count
                ):
                    break

                if current_frame_index < next_target_index:
                    if not capture.grab():
                        break
                    current_frame_index += 1
                    continue

                ok, frame = capture.read()
                if not ok or frame is None:
                    break

                timestamp = (
                    float(current_frame_index / sampling_info.fps)
                    if sampling_info.fps > 0
                    else None
                )
                yielded += 1
                yield SampledFrame(
                    source_path=path,
                    image_name=f"{safe_stem(path.stem)}_frame_{current_frame_index:06d}",
                    frame_index=current_frame_index,
                    timestamp_seconds=timestamp,
                    bgr_pixels=frame,
                    original_bgr_pixels=frame,
                )
                current_frame_index += 1
                next_target_index += sampling_info.frame_step
        finally:
            capture.release()

    def _read_sampling_info(self, capture: cv2.VideoCapture) -> VideoSamplingInfo:
        fps = float(capture.get(cv2.CAP_PROP_FPS)) or 0.0
        raw_total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        total_frames = raw_total_frames if raw_total_frames > 0 else None
        frame_step = self._calculate_frame_step(fps)
        duration_seconds = (
            float(raw_total_frames / fps)
            if raw_total_frames > 0 and fps > 0
            else None
        )
        actual_sampling_interval_seconds = (
            float(frame_step / fps)
            if fps > 0
            else None
        )
        planned_sample_count = (
            len(range(0, raw_total_frames, frame_step))
            if raw_total_frames > 0
            else None
        )
        if planned_sample_count is not None and self._settings.max_frames_per_video is not None:
            planned_sample_count = min(planned_sample_count, self._settings.max_frames_per_video)
        return VideoSamplingInfo(
            fps=fps,
            total_frames=total_frames,
            duration_seconds=duration_seconds,
            frame_step=frame_step,
            actual_sampling_interval_seconds=actual_sampling_interval_seconds,
            planned_sample_count=planned_sample_count,
            max_sample_count=self._settings.max_frames_per_video,
        )

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
