from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path

from tests._bootstrap import PROJECT_ROOT  # noqa: F401


class StartupImportTests(unittest.TestCase):
    def _module_presence(self, module_name: str, tracked_modules: list[str]) -> dict[str, bool]:
        source_root = Path(PROJECT_ROOT) / "src"
        script = (
            "import json, sys; "
            f"sys.path.insert(0, {source_root.as_posix()!r}); "
            f"import {module_name}; "
            f"print(json.dumps({{name: (name in sys.modules) for name in {tracked_modules!r}}}))"
        )
        completed = subprocess.run(
            [sys.executable, "-c", script],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        return json.loads(completed.stdout.strip())

    def test_importing_app_avoids_loading_inventory_pipeline(self) -> None:
        presence = self._module_presence(
            "inventario_faces.app",
            [
                "docx",
                "inventario_faces.reporting.docx_renderer",
                "inventario_faces.services.inventory_service",
                "inventario_faces.infrastructure.face_analyzer_insight",
                "scipy",
            ],
        )

        self.assertFalse(presence["docx"])
        self.assertFalse(presence["inventario_faces.reporting.docx_renderer"])
        self.assertFalse(presence["inventario_faces.services.inventory_service"])
        self.assertFalse(presence["inventario_faces.infrastructure.face_analyzer_insight"])
        self.assertFalse(presence["scipy"])

    def test_importing_main_window_keeps_dialogs_and_workers_lazy(self) -> None:
        presence = self._module_presence(
            "inventario_faces.gui.main_window",
            [
                "inventario_faces.gui.face_set_comparison_dialog",
                "inventario_faces.gui.distributed_monitor_dialog",
                "inventario_faces.gui.worker",
                "inventario_faces.services.inventory_service",
            ],
        )

        self.assertFalse(presence["inventario_faces.gui.face_set_comparison_dialog"])
        self.assertFalse(presence["inventario_faces.gui.distributed_monitor_dialog"])
        self.assertFalse(presence["inventario_faces.gui.worker"])
        self.assertFalse(presence["inventario_faces.services.inventory_service"])


if __name__ == "__main__":
    unittest.main()
