from __future__ import annotations

import tomllib
import unittest
from pathlib import Path

from tests._bootstrap import PROJECT_ROOT  # noqa: F401
from inventario_faces import __version__


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


if __name__ == "__main__":
    unittest.main()
