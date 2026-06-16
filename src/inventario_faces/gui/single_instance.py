from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import sys

from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtNetwork import QLocalServer, QLocalSocket
from PySide6.QtWidgets import QWidget

_ACTIVATE_PAYLOAD = b"ACTIVATE\n"


def build_single_instance_server_name(app_id: str, anchor: str) -> str:
    normalized_app_id = "".join(character if character.isalnum() else "_" for character in app_id.lower()).strip("_")
    digest = sha256(f"{normalized_app_id}|{anchor}".encode("utf-8")).hexdigest()[:20]
    return f"{normalized_app_id}_{digest}" if normalized_app_id else f"inventario_faces_{digest}"


def runtime_single_instance_server_name(app_id: str = "inventario_faces") -> str:
    runtime_anchor_parts = [str(Path(sys.executable).resolve())]
    if not getattr(sys, "frozen", False):
        runtime_anchor_parts.append(str(Path(__file__).resolve().parents[2]))
    return build_single_instance_server_name(app_id, "|".join(runtime_anchor_parts))


def present_window(window: QWidget) -> None:
    window_state = window.windowState()
    if window_state & Qt.WindowMinimized:
        window.setWindowState((window_state & ~Qt.WindowMinimized) | Qt.WindowActive)
    else:
        window.setWindowState(window_state | Qt.WindowActive)
    if not window.isVisible():
        window.show()
    window.raise_()
    window.activateWindow()


class SingleInstanceCoordinator(QObject):
    activation_requested = Signal()

    def __init__(self, server_name: str, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._server_name = server_name
        self._server: QLocalServer | None = None
        self._connections: set[QLocalSocket] = set()

    def notify_existing_instance(self, timeout_ms: int = 800) -> bool:
        socket = QLocalSocket(self)
        socket.connectToServer(self._server_name)
        if not socket.waitForConnected(timeout_ms):
            return False
        socket.write(_ACTIVATE_PAYLOAD)
        socket.flush()
        socket.waitForBytesWritten(timeout_ms)
        socket.disconnectFromServer()
        return True

    def start(self) -> bool:
        if self._server is not None:
            return True

        server = QLocalServer(self)
        if not server.listen(self._server_name):
            QLocalServer.removeServer(self._server_name)
            if not server.listen(self._server_name):
                return False
        server.newConnection.connect(self._on_new_connection)
        self._server = server
        return True

    def close(self) -> None:
        for connection in list(self._connections):
            connection.disconnectFromServer()
            connection.deleteLater()
        self._connections.clear()
        if self._server is not None:
            self._server.close()
            self._server.deleteLater()
            self._server = None
        QLocalServer.removeServer(self._server_name)

    def _on_new_connection(self) -> None:
        if self._server is None:
            return
        while self._server.hasPendingConnections():
            connection = self._server.nextPendingConnection()
            if connection is None:
                return
            self._connections.add(connection)
            connection.readyRead.connect(lambda socket=connection: self._on_ready_read(socket))
            connection.disconnected.connect(lambda socket=connection: self._dispose_connection(socket))

    def _on_ready_read(self, connection: QLocalSocket) -> None:
        payload = bytes(connection.readAll())
        if _ACTIVATE_PAYLOAD.strip() in payload:
            self.activation_requested.emit()
        connection.disconnectFromServer()

    def _dispose_connection(self, connection: QLocalSocket) -> None:
        self._connections.discard(connection)
        connection.deleteLater()
