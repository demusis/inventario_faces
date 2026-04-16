from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tests._bootstrap import PROJECT_ROOT  # noqa: F401
from inventario_faces.versioning import (
    bump_semver,
    read_current_version,
    sync_project_version,
)


class VersioningTests(unittest.TestCase):
    def test_bump_semver_advances_requested_part(self) -> None:
        self.assertEqual("1.4.10", bump_semver("1.4.9", part="patch"))
        self.assertEqual("1.5.0", bump_semver("1.4.9", part="minor"))
        self.assertEqual("2.0.0", bump_semver("1.4.9", part="major"))

    def test_sync_project_version_updates_package_and_installer_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_root = Path(tmp_dir)
            package_path = project_root / "src" / "inventario_faces"
            package_path.mkdir(parents=True)
            build_path = project_root / "build"
            build_path.mkdir(parents=True)

            (package_path / "__init__.py").write_text(
                '"""Inventario Faces."""\n\n__all__ = ["__version__"]\n\n__version__ = "0.2.0"\n',
                encoding="utf-8",
            )
            (build_path / "inventario_faces.iss").write_text(
                '#define MyAppName "Inventario Faces"\n#ifndef MyAppVersion\n  #define MyAppVersion "0.2.0"\n#endif\n',
                encoding="utf-8",
            )

            changed = sync_project_version(project_root, version="0.2.1", write=True)

            self.assertEqual(2, len(changed))
            self.assertIn('__version__ = "0.2.1"', (package_path / "__init__.py").read_text(encoding="utf-8"))
            self.assertIn('#define MyAppVersion "0.2.1"', (build_path / "inventario_faces.iss").read_text(encoding="utf-8"))

    def test_read_current_version_matches_package_metadata(self) -> None:
        self.assertRegex(read_current_version(Path(PROJECT_ROOT)), r"^\d+\.\d+\.\d+$")


if __name__ == "__main__":
    unittest.main()
