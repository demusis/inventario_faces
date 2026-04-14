from __future__ import annotations

import unittest

from tests._bootstrap import PROJECT_ROOT  # noqa: F401
from inventario_faces.gui.config_help import (
    FAISS_URL,
    abnt_reference_html,
    build_config_help_html,
)


class ConfigHelpTests(unittest.TestCase):
    def test_build_config_help_html_includes_caveat_and_reference(self) -> None:
        html = build_config_help_html(
            definition="Campo de teste.",
            operational_effect="Impacto operacional de teste.",
            recommendation="Recomendação de teste.",
            caveat="Observação complementar.",
            references=[("FAISS", FAISS_URL)],
        )

        self.assertIn("Definição.", html)
        self.assertIn("Observação.", html)
        self.assertIn("FAISS", html)
        self.assertIn(FAISS_URL, html)

    def test_abnt_reference_html_falls_back_for_unknown_reference(self) -> None:
        reference = abnt_reference_html("Referencia de teste", "https://example.com/referencia")

        self.assertIn("Referencia de teste.", reference)
        self.assertIn("https://example.com/referencia", reference)


if __name__ == "__main__":
    unittest.main()
