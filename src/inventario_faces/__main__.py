from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from inventario_faces.app import build_inventory_service, load_runtime_config
from inventario_faces.gui.main_window import MainWindow


def main() -> int:
    application = QApplication(sys.argv)
    window = MainWindow(
        service_factory=build_inventory_service,
        initial_config=load_runtime_config(),
    )
    window.show()
    return application.exec()


if __name__ == "__main__":
    raise SystemExit(main())
