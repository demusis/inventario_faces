from __future__ import annotations

import tomllib
import unittest
from pathlib import Path

from tests._bootstrap import PROJECT_ROOT  # noqa: F401
from inventario_faces import __version__
from inventario_faces.app import resolve_app_icon_path
from inventario_faces.versioning import INSTALLER_VERSION_PATH


class VersionMetadataTests(unittest.TestCase):
    def test_pyproject_uses_package_version_attribute(self) -> None:
        pyproject_path = Path(PROJECT_ROOT) / "pyproject.toml"
        payload = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))

        self.assertIn("version", payload["project"]["dynamic"])
        self.assertEqual(
            "inventario_faces.__version__",
            payload["tool"]["setuptools"]["dynamic"]["version"]["attr"],
        )
        self.assertRegex(__version__, r"^\d+\.\d+\.\d+$")

    def test_installer_fallback_version_matches_package_version(self) -> None:
        installer_path = Path(PROJECT_ROOT) / INSTALLER_VERSION_PATH
        installer_text = installer_path.read_text(encoding="utf-8")

        self.assertIn(f'#define MyAppVersion "{__version__}"', installer_text)
        self.assertTrue((Path(PROJECT_ROOT) / "tools" / "bump_version.py").exists())

    def test_icon_assets_are_packaged_and_resolvable(self) -> None:
        pyproject_path = Path(PROJECT_ROOT) / "pyproject.toml"
        payload = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
        package_data = payload["tool"]["setuptools"]["package-data"]["inventario_faces"]

        self.assertIn("assets/*.png", package_data)
        self.assertIn("assets/*.ico", package_data)

        icon_path = resolve_app_icon_path()
        self.assertIsNotNone(icon_path)
        assert icon_path is not None
        self.assertTrue(icon_path.exists())
        self.assertEqual(".ico", icon_path.suffix.lower())


if __name__ == "__main__":
    unittest.main()
