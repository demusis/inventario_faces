from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from inventario_faces.versioning import bump_semver, read_current_version, sync_project_version, validate_semver


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Atualiza a versao do app e mantem em sincronia os arquivos versionados do projeto.",
    )
    parser.add_argument(
        "--part",
        choices=("major", "minor", "patch"),
        default="patch",
        help="Parte da versao semantica a ser incrementada quando --set nao for informado.",
    )
    parser.add_argument(
        "--set",
        dest="target_version",
        help="Define explicitamente a versao desejada no formato X.Y.Z.",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Grava as alteracoes nos arquivos. Sem esta flag, executa apenas simulacao.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    current_version = read_current_version(PROJECT_ROOT)
    target_version = validate_semver(args.target_version) if args.target_version else bump_semver(current_version, part=args.part)
    changed_files = sync_project_version(PROJECT_ROOT, version=target_version, write=args.write)

    action = "gravada" if args.write else "simulada"
    if changed_files:
        print(f"Atualizacao {action}: {current_version} -> {target_version}")
        for file_path in changed_files:
            print(file_path.relative_to(PROJECT_ROOT).as_posix())
    else:
        print(f"Versao mantida em {target_version}; nenhum arquivo exigiu atualizacao.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
