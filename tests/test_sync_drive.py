from __future__ import annotations

import unittest
from pathlib import Path
from unittest import mock

from tests._bootstrap import PROJECT_ROOT  # noqa: F401
from inventario_faces.infrastructure.sync_drive import (
    detect_sync_provider,
    sync_drive_warning_lines,
)


class SyncDriveDetectionTests(unittest.TestCase):
    def test_detects_google_drive_by_path_marker(self) -> None:
        with mock.patch(
            "inventario_faces.infrastructure.sync_drive._volume_label",
            return_value=None,
        ):
            self.assertEqual(
                "Google Drive",
                detect_sync_provider(Path("G:/Meu Drive/Em processamento/evidencias")),
            )
            self.assertEqual(
                "Google Drive",
                detect_sync_provider(Path("H:/Shared drives/pericia/lote1")),
            )

    def test_detects_google_drive_by_volume_label(self) -> None:
        with mock.patch(
            "inventario_faces.infrastructure.sync_drive._volume_label",
            return_value="Google Drive",
        ):
            self.assertEqual(
                "Google Drive",
                detect_sync_provider(Path("X:/evidencias/lote")),
            )

    def test_detects_onedrive_by_environment(self) -> None:
        onedrive_root = Path("C:/Users/perito/OneDrive - Org")
        with (
            mock.patch(
                "inventario_faces.infrastructure.sync_drive._volume_label",
                return_value=None,
            ),
            mock.patch.dict(
                "os.environ",
                {"OneDrive": str(onedrive_root)},
                clear=False,
            ),
        ):
            self.assertEqual(
                "OneDrive",
                detect_sync_provider(onedrive_root / "casos" / "lote2"),
            )

    def test_detects_dropbox_by_path_component(self) -> None:
        with mock.patch(
            "inventario_faces.infrastructure.sync_drive._volume_label",
            return_value=None,
        ):
            self.assertEqual(
                "Dropbox",
                detect_sync_provider(Path("D:/Dropbox/evidencias")),
            )

    def test_returns_none_for_regular_paths(self) -> None:
        with (
            mock.patch(
                "inventario_faces.infrastructure.sync_drive._volume_label",
                return_value=None,
            ),
            mock.patch.dict(
                "os.environ",
                {"OneDrive": "", "OneDriveConsumer": "", "OneDriveCommercial": ""},
                clear=False,
            ),
        ):
            self.assertIsNone(detect_sync_provider(Path("D:/casos/lote_local")))
            self.assertIsNone(detect_sync_provider(Path("//servidor/evidencias/lote")))

    def test_warning_lines_mention_provider_and_path(self) -> None:
        lines = sync_drive_warning_lines(
            "Diretorio de evidencias",
            Path("G:/Meu Drive/lote"),
            "Google Drive",
        )
        self.assertEqual(3, len(lines))
        self.assertIn("Google Drive", lines[0])
        self.assertIn("Diretorio de evidencias", lines[0])
        self.assertTrue(all(line.startswith("[Distribuicao][ALERTA]") for line in lines))


if __name__ == "__main__":
    unittest.main()
