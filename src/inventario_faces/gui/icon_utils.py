from __future__ import annotations

from PySide6.QtWidgets import QAbstractButton, QStyle, QWidget


def apply_standard_icon(
    owner: QWidget,
    button: QAbstractButton,
    pixmap: QStyle.StandardPixmap,
) -> None:
    """Apply a native Qt standard icon to a button-like widget."""

    button.setIcon(owner.style().standardIcon(pixmap))
