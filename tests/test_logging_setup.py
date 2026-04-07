from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tests._bootstrap import PROJECT_ROOT  # noqa: F401
from inventario_faces.infrastructure.logging_setup import (
    build_file_logger,
    close_file_logger,
    format_exception_traceback,
    summarize_exception,
)


class LoggingSetupTests(unittest.TestCase):
    def test_summarize_exception_includes_causal_chain(self) -> None:
        try:
            try:
                raise ValueError("entrada invalida")
            except ValueError as exc:
                raise RuntimeError("falha ao processar") from exc
        except RuntimeError as exc:
            summary = summarize_exception(exc)
            traceback_text = format_exception_traceback(exc)

        self.assertIn("RuntimeError: falha ao processar", summary)
        self.assertIn("ValueError: entrada invalida", summary)
        self.assertIn("Traceback", traceback_text)
        self.assertIn("ValueError: entrada invalida", traceback_text)

    def test_file_logger_writes_utc_timestamp_with_milliseconds(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            log_directory = Path(temp_dir)
            logger = build_file_logger(log_directory, "INFO")
            logger.info("mensagem de teste")
            close_file_logger(logger)

            log_text = (log_directory / "run.log").read_text(encoding="utf-8")

        self.assertRegex(
            log_text,
            r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z \| INFO \| mensagem de teste",
        )


if __name__ == "__main__":
    unittest.main()
