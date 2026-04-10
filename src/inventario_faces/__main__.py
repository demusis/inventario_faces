from __future__ import annotations

import sys

from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication

from inventario_faces.app import build_inventory_service, load_runtime_config, resolve_app_icon_path
from inventario_faces.gui.main_window import MainWindow


def main() -> int:
    application = QApplication(sys.argv)
    icon_path = resolve_app_icon_path()
    if icon_path is not None:
        icon = QIcon(str(icon_path))
        if not icon.isNull():
            application.setWindowIcon(icon)
    window = MainWindow(
        service_factory=build_inventory_service,
        initial_config=load_runtime_config(),
    )
    window.show()
    return application.exec()


if __name__ == "__main__":
    raise SystemExit(main())
