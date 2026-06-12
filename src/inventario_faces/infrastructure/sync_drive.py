"""Deteccao heuristica de diretorios hospedados em drives de sincronizacao.

A coordenacao distribuida depende de semantica atomica de filesystem
(criacao exclusiva de arquivos e renomeacao atomica). Provedores de
sincronizacao em nuvem (Google Drive, OneDrive, Dropbox) nao oferecem
essas garantias entre maquinas, entao o aplicativo apenas alerta o
usuario quando detecta esse cenario, sem bloquear a execucao.
"""

from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path

_GOOGLE_DRIVE_MARKERS = (
    "meu drive",
    "my drive",
    "shared drives",
    "drives compartilhados",
    ".shortcut-targets-by-id",
)


def _volume_label(path: Path) -> str | None:
    if sys.platform != "win32":
        return None
    drive = Path(path).resolve().drive
    if not drive:
        return None
    buffer = ctypes.create_unicode_buffer(261)
    filesystem_buffer = ctypes.create_unicode_buffer(261)
    result = ctypes.windll.kernel32.GetVolumeInformationW(
        ctypes.c_wchar_p(f"{drive}\\"),
        buffer,
        len(buffer),
        None,
        None,
        None,
        filesystem_buffer,
        len(filesystem_buffer),
    )
    if not result:
        return None
    return buffer.value or None


def detect_sync_provider(path: Path) -> str | None:
    """Retorna o nome do provedor de sincronizacao detectado, ou None."""
    resolved = Path(path).expanduser()
    try:
        resolved = resolved.resolve()
    except OSError:
        pass
    lowered_parts = [part.lower() for part in resolved.parts]
    lowered_text = str(resolved).lower()

    if any(marker in lowered_parts for marker in _GOOGLE_DRIVE_MARKERS):
        return "Google Drive"
    label = _volume_label(resolved)
    if label and "google drive" in label.lower():
        return "Google Drive"

    onedrive_roots = [
        os.environ.get("OneDrive"),
        os.environ.get("OneDriveConsumer"),
        os.environ.get("OneDriveCommercial"),
    ]
    for root in onedrive_roots:
        if not root:
            continue
        root_text = str(Path(root)).lower().rstrip("\\/")
        if lowered_text == root_text or lowered_text.startswith(root_text + os.sep):
            return "OneDrive"
    if any(part.startswith("onedrive") for part in lowered_parts):
        return "OneDrive"

    if "dropbox" in lowered_parts:
        return "Dropbox"

    return None


def sync_drive_warning_lines(label: str, path: Path, provider: str) -> list[str]:
    """Linhas de alerta para log quando um caminho distribuido usa drive sincronizado."""
    return [
        (
            f"[Distribuicao][ALERTA] {label} ({path}) parece estar em um drive de "
            f"sincronizacao ({provider})."
        ),
        (
            "[Distribuicao][ALERTA] Esse tipo de armazenamento nao garante locks "
            "atomicos entre estacoes: duas instancias podem assumir o mesmo arquivo "
            "ou corromper o manifesto compartilhado."
        ),
        (
            "[Distribuicao][ALERTA] Para coordenacao multi-instancia, prefira um "
            "compartilhamento SMB/UNC (ex.: \\\\servidor\\evidencias) ou NAS montado "
            "de forma identica em todas as maquinas."
        ),
    ]
