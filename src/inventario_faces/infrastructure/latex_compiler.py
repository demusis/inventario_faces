from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


class LatexCompilationError(RuntimeError):
    """Erro ao compilar o relatorio LaTeX."""


class LatexCompiler:
    def __init__(self, executable: str = "pdflatex") -> None:
        self._executable = executable

    def is_available(self) -> bool:
        return shutil.which(self._executable) is not None

    def compile(self, tex_path: Path) -> Path:
        if not self.is_available():
            raise LatexCompilationError("pdflatex nao encontrado no PATH.")

        working_directory = tex_path.parent
        for _ in range(2):
            completed = subprocess.run(
                [
                    self._executable,
                    "-interaction=nonstopmode",
                    "-halt-on-error",
                    "-disable-installer",
                    tex_path.name,
                ],
                cwd=working_directory,
                capture_output=True,
                text=True,
                check=False,
            )
            if completed.returncode != 0:
                raise LatexCompilationError(
                    f"Falha ao compilar {tex_path.name}: {completed.stdout}\n{completed.stderr}"
                )

        pdf_path = tex_path.with_suffix(".pdf")
        if not pdf_path.exists():
            raise LatexCompilationError("Compilacao concluida sem gerar PDF.")
        return pdf_path
