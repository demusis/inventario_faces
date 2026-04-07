# -*- mode: python ; coding: utf-8 -*-
from pathlib import Path

import yaml
from PyInstaller.utils.hooks import collect_submodules

project_root = Path.cwd()
hiddenimports = ["cv2", "numpy", "onnxruntime"]

try:
    hiddenimports += collect_submodules("insightface")
except Exception:
    pass


def _collect_tree(source_root: Path, target_root: str) -> list[tuple[str, str]]:
    if not source_root.exists():
        return []

    collected: list[tuple[str, str]] = []
    for path in source_root.rglob("*"):
        if not path.is_file():
            continue
        relative_parent = path.parent.relative_to(source_root)
        destination = Path(target_root)
        if relative_parent != Path("."):
            destination /= relative_parent
        collected.append((str(path), destination.as_posix()))
    return collected


defaults_path = project_root / "config" / "defaults.yaml"
default_model_name = "buffalo_l"
try:
    with defaults_path.open("r", encoding="utf-8") as stream:
        defaults_payload = yaml.safe_load(stream) or {}
    default_model_name = str(defaults_payload.get("face_model", {}).get("model_name") or default_model_name)
except Exception:
    pass

datas = [
    (str(project_root / "src" / "inventario_faces" / "config" / "defaults.yaml"), "inventario_faces/config"),
    (str(project_root / "src" / "inventario_faces" / "assets" / "app_icon.png"), "inventario_faces/assets"),
    (str(project_root / "src" / "inventario_faces" / "assets" / "app_icon.ico"), "inventario_faces/assets"),
    (
        str(project_root / "src" / "inventario_faces" / "reporting" / "templates" / "forensic_report_template.tex"),
        "inventario_faces/reporting/templates",
    ),
    (str(project_root / "models" / "README.txt"), "models"),
]
datas += _collect_tree(project_root / "models" / default_model_name, f"models/{default_model_name}")

a = Analysis(
    [str(project_root / "src" / "inventario_faces" / "__main__.py")],
    pathex=[str(project_root / "src")],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "tkinter",
        "matplotlib.tests",
        "numpy.tests",
        "scipy.tests",
        "sklearn.tests",
        "PIL.tests",
    ],
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
    icon=str(project_root / "src" / "inventario_faces" / "assets" / "app_icon.ico"),
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
