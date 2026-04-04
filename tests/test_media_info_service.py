from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from tests._bootstrap import PROJECT_ROOT  # noqa: F401
from inventario_faces.infrastructure.media_info_service import MediaInfoService


class MediaInfoServiceTests(unittest.TestCase):
    def test_extracts_image_metadata_without_external_executable(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "sample.jpg"
            Image.new("RGB", (80, 60), color=(10, 20, 30)).save(image_path, format="JPEG")

            service = MediaInfoService()
            tracks, error = service.extract(image_path)

            self.assertIsNone(service.executable_path)
            self.assertIsNone(error)
            self.assertGreaterEqual(len(tracks), 2)
            self.assertEqual("Geral", tracks[0].track_type)
            self.assertEqual("Imagem", tracks[1].track_type)
            labels = {attribute.label for track in tracks for attribute in track.attributes}
            self.assertIn("Formato", labels)
            self.assertIn("Largura", labels)
            self.assertIn("Altura", labels)

    def test_extracts_video_metadata_with_internal_extractor(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            video_path = Path(temp_dir) / "sample.avi"
            writer = cv2.VideoWriter(
                str(video_path),
                cv2.VideoWriter_fourcc(*"MJPG"),
                5.0,
                (64, 48),
            )
            if not writer.isOpened():
                self.skipTest("VideoWriter não disponível no ambiente de teste.")
            frame = np.zeros((48, 64, 3), dtype=np.uint8)
            for _ in range(10):
                writer.write(frame)
            writer.release()

            service = MediaInfoService()
            tracks, error = service.extract(video_path)

            self.assertIsNone(error)
            self.assertGreaterEqual(len(tracks), 2)
            labels = {attribute.label for track in tracks for attribute in track.attributes}
            self.assertIn("Duração", labels)
            self.assertIn("Taxa de quadros", labels)
            self.assertIn("Número de quadros", labels)


if __name__ == "__main__":
    unittest.main()
