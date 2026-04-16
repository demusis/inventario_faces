from __future__ import annotations

from pathlib import Path
import re

__all__ = [
    "PACKAGE_VERSION_PATH",
    "INSTALLER_VERSION_PATH",
    "bump_semver",
    "read_current_version",
    "sync_project_version",
    "validate_semver",
]


PACKAGE_VERSION_PATH = Path("src") / "inventario_faces" / "__init__.py"
INSTALLER_VERSION_PATH = Path("build") / "inventario_faces.iss"

_SEMVER_RE = re.compile(r"^(?P<major>\d+)\.(?P<minor>\d+)\.(?P<patch>\d+)$")
_PACKAGE_VERSION_RE = re.compile(r'(?m)^__version__\s*=\s*"(?P<version>\d+\.\d+\.\d+)"\s*$')
_INSTALLER_VERSION_RE = re.compile(r'(?m)^\s*#define MyAppVersion "(?P<version>\d+\.\d+\.\d+)"\s*$')


def validate_semver(version: str) -> str:
    if not _SEMVER_RE.fullmatch(version):
        raise ValueError(f"Versao invalida: {version!r}. Use o formato X.Y.Z.")
    return version


def bump_semver(version: str, *, part: str = "patch") -> str:
    match = _SEMVER_RE.fullmatch(validate_semver(version))
    assert match is not None
    major = int(match.group("major"))
    minor = int(match.group("minor"))
    patch = int(match.group("patch"))

    if part == "major":
        return f"{major + 1}.0.0"
    if part == "minor":
        return f"{major}.{minor + 1}.0"
    if part == "patch":
        return f"{major}.{minor}.{patch + 1}"
    raise ValueError(f"Parte de versao nao suportada: {part!r}.")


def _replace_version(content: str, pattern: re.Pattern[str], version: str, *, label: str) -> str:
    validate_semver(version)
    if pattern.search(content) is None:
        raise ValueError(f"Nao foi possivel localizar o marcador de versao em {label}.")
    return pattern.sub(lambda match: match.group(0).replace(match.group("version"), version), content, count=1)


def read_current_version(project_root: Path) -> str:
    package_path = project_root / PACKAGE_VERSION_PATH
    content = package_path.read_text(encoding="utf-8")
    match = _PACKAGE_VERSION_RE.search(content)
    if match is None:
        raise ValueError(f"Nao foi possivel ler __version__ em {package_path}.")
    return validate_semver(match.group("version"))


def sync_project_version(
    project_root: Path,
    *,
    version: str,
    write: bool = True,
) -> list[Path]:
    version = validate_semver(version)
    targets = {
        project_root / PACKAGE_VERSION_PATH: (_PACKAGE_VERSION_RE, "src/inventario_faces/__init__.py"),
        project_root / INSTALLER_VERSION_PATH: (_INSTALLER_VERSION_RE, "build/inventario_faces.iss"),
    }
    changed_files: list[Path] = []

    for target_path, (pattern, label) in targets.items():
        content = target_path.read_text(encoding="utf-8")
        updated = _replace_version(content, pattern, version, label=label)
        if updated != content:
            changed_files.append(target_path)
            if write:
                target_path.write_text(updated, encoding="utf-8")

    return changed_files
