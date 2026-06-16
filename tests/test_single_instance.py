from __future__ import annotations

import unittest
from uuid import uuid4

from PySide6.QtCore import QCoreApplication, QTimer

from tests._bootstrap import PROJECT_ROOT  # noqa: F401
from inventario_faces.gui.single_instance import (
    SingleInstanceCoordinator,
    build_single_instance_server_name,
)


class SingleInstanceTests(unittest.TestCase):
    def test_server_name_is_stable_for_same_input(self) -> None:
        first = build_single_instance_server_name("Inventario Faces", "C:/apps/inventario_faces.exe")
        second = build_single_instance_server_name("Inventario Faces", "C:/apps/inventario_faces.exe")

        self.assertEqual(first, second)

    def test_server_name_changes_with_runtime_anchor(self) -> None:
        first = build_single_instance_server_name("Inventario Faces", "C:/apps/inventario_faces.exe")
        second = build_single_instance_server_name("Inventario Faces", "C:/apps/outra_instancia.exe")

        self.assertNotEqual(first, second)
        self.assertTrue(first.startswith("inventario_faces_"))
        self.assertTrue(second.startswith("inventario_faces_"))

    def test_secondary_instance_notifies_primary_instance(self) -> None:
        app = QCoreApplication.instance() or QCoreApplication([])
        server_name = build_single_instance_server_name("Inventario Faces", f"test-{uuid4()}")
        primary = SingleInstanceCoordinator(server_name)
        secondary = SingleInstanceCoordinator(server_name)
        activations: list[str] = []

        self.assertTrue(primary.start())
        primary.activation_requested.connect(lambda: activations.append("activate"))

        self.assertTrue(secondary.notify_existing_instance())
        QTimer.singleShot(500, app.quit)
        app.exec()

        self.assertEqual(["activate"], activations)
        primary.close()
        secondary.close()


if __name__ == "__main__":
    unittest.main()
