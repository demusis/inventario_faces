from __future__ import annotations

import os
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

from tests._bootstrap import PROJECT_ROOT  # noqa: F401
from inventario_faces.domain.config import VideoSettings
from inventario_faces.services.video_service import VideoService


class _FakeCapture:
    def __init__(self, frames: list[np.ndarray], fps: float) -> None:
        self._frames = [frame.copy() for frame in frames]
        self._fps = fps
        self._cursor = 0
        self._released = False

    def isOpened(self) -> bool:
        return True

    def get(self, prop_id: int) -> float:
        if prop_id == cv2.CAP_PROP_FPS:
            return self._fps
        if prop_id == cv2.CAP_PROP_FRAME_COUNT:
            return float(len(self._frames))
        return 0.0

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self._cursor >= len(self._frames):
            return False, None
        frame = self._frames[self._cursor]
        self._cursor += 1
        return True, frame.copy()

    def grab(self) -> bool:
        if self._cursor >= len(self._frames):
            return False
        self._cursor += 1
        return True

    def release(self) -> None:
        self._released = True


class VideoServiceTests(unittest.TestCase):
    def test_video_service_sets_default_ffmpeg_read_attempts(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("OPENCV_FFMPEG_READ_ATTEMPTS", None)
            VideoService(
                VideoSettings(
                    sampling_interval_seconds=1.0,
                    max_frames_per_video=None,
                )
            )

            self.assertEqual("16384", os.environ.get("OPENCV_FFMPEG_READ_ATTEMPTS"))

    def test_sample_video_returns_real_frame_indexes_for_sampled_frames(self) -> None:
        frames = [np.full((8, 8, 3), fill_value=index, dtype=np.uint8) for index in range(5)]
        capture = _FakeCapture(frames, fps=2.0)
        service = VideoService(
            VideoSettings(
                sampling_interval_seconds=1.0,
                max_frames_per_video=None,
            )
        )
        service._open_capture = lambda path: capture  # type: ignore[method-assign]
        metadata: list[object] = []

        sampled_frames = list(
            service.sample_video(Path("sample.mp4"), metadata_callback=metadata.append)
        )

        self.assertEqual([0, 2, 4], [frame.frame_index for frame in sampled_frames])
        self.assertEqual([0.0, 1.0, 2.0], [frame.timestamp_seconds for frame in sampled_frames])
        self.assertEqual(1, len(metadata))
        sampling_info = metadata[0]
        self.assertEqual(2, sampling_info.frame_step)
        self.assertEqual(3, sampling_info.planned_sample_count)
        self.assertEqual(1.0, sampling_info.actual_sampling_interval_seconds)
        self.assertEqual(2.5, sampling_info.duration_seconds)
        self.assertTrue(capture._released)


if __name__ == "__main__":
    unittest.main()
