# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_submodules

from pathlib import Path

project_root = Path.cwd()
hiddenimports = ["cv2", "numpy", "onnxruntime"]

try:
    hiddenimports += collect_submodules("insightface")
except Exception:
    pass

datas = [
    (str(project_root / "config" / "defaults.yaml"), "config"),
    (str(project_root / "src" / "inventario_faces" / "config" / "defaults.yaml"), "inventario_faces/config"),
    (
        str(project_root / "src" / "inventario_faces" / "reporting" / "templates" / "forensic_report_template.tex"),
        "inventario_faces/reporting/templates",
    ),
    (str(project_root / "models"), "models"),
]

a = Analysis(
    [str(project_root / "src" / "inventario_faces" / "__main__.py")],
    pathex=[str(project_root / "src")],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="InventarioFaces",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    name="InventarioFaces",
)
