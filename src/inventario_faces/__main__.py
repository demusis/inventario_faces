from __future__ import annotations

import sys

from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QTimer

from inventario_faces.app import (
    build_inventory_service,
    load_runtime_config,
    preload_shared_face_analyzer_async,
    resolve_app_icon_path,
)
from inventario_faces.gui.main_window import MainWindow
from inventario_faces.gui.single_instance import (
    SingleInstanceCoordinator,
    present_window,
    runtime_single_instance_server_name,
)


def main() -> int:
    application = QApplication(sys.argv)
    coordinator = SingleInstanceCoordinator(runtime_single_instance_server_name(), application)
    if coordinator.notify_existing_instance():
        return 0
    if not coordinator.start():
        raise RuntimeError("Nao foi possivel inicializar o coordenador de instancia unica do aplicativo.")
    icon_path = resolve_app_icon_path()
    if icon_path is not None:
        icon = QIcon(str(icon_path))
        if not icon.isNull():
            application.setWindowIcon(icon)
    initial_config = load_runtime_config()
    window = MainWindow(
        service_factory=build_inventory_service,
        initial_config=initial_config,
    )
    coordinator.activation_requested.connect(lambda: QTimer.singleShot(0, lambda: present_window(window)))
    application.aboutToQuit.connect(coordinator.close)
    window.show()
    present_window(window)
    QTimer.singleShot(600, lambda: preload_shared_face_analyzer_async(initial_config.face_model))
    return application.exec()


if __name__ == "__main__":
    raise SystemExit(main())
