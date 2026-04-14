from __future__ import annotations

import unittest

from tests._bootstrap import PROJECT_ROOT  # noqa: F401
from inventario_faces.domain.entities import FaceSetComparisonMatch
from inventario_faces.gui.face_set_comparison_dialog import (
    _likelihood_ratio_selection_html,
    _mann_whitney_group_comparison,
)


class FaceSetComparisonStatisticsTests(unittest.TestCase):
    def test_likelihood_ratio_selection_html_explains_density_ratio(self) -> None:
        html = _likelihood_ratio_selection_html(
            FaceSetComparisonMatch(
                rank=1,
                left_entry_id="A1",
                right_entry_id="B1",
                left_track_id="TA1",
                right_track_id="TB1",
                similarity=-0.0242,
                classification="below_threshold",
                likelihood_ratio=0.1564,
                log10_likelihood_ratio=-0.8058,
                same_source_density=0.021,
                different_source_density=0.134271,
            ),
            left_name="Abbas_Kiarostami_0001.jpg",
            right_name="Adam_Scott_0002.jpg",
        )

        self.assertIn("LR = f(score|H1) / f(score|H2)", html)
        self.assertIn("0.021", html)
        self.assertIn("0.134271", html)
        self.assertIn("0.1564", html)
        self.assertIn("H2 (origem distinta)", html)

    def test_mann_whitney_group_comparison_reports_significant_difference(self) -> None:
        result = _mann_whitney_group_comparison(
            [0.81, 0.83, 0.86, 0.89, 0.91],
            [0.42, 0.45, 0.47, 0.5, 0.54],
            alpha=0.05,
            metric_label="qualidade facial",
        )

        self.assertTrue(result.available)
        self.assertIsNotNone(result.u_statistic)
        self.assertIsNotNone(result.p_value)
        self.assertTrue(result.significant)
        self.assertGreater(result.rank_biserial or 0.0, 0.0)
        self.assertGreater(result.common_language_effect or 0.0, 0.5)

    def test_mann_whitney_group_comparison_requires_two_values_per_group(self) -> None:
        result = _mann_whitney_group_comparison(
            [0.75],
            [0.61, 0.64],
            alpha=0.05,
            metric_label="qualidade facial",
        )

        self.assertFalse(result.available)
        self.assertIn("ao menos 2 observações válidas", result.note or "")


if __name__ == "__main__":
    unittest.main()
