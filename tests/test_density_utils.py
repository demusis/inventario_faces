from __future__ import annotations

import unittest

from inventario_faces.utils.density_utils import fit_score_density_model, score_density_method_label


class ScoreDensityUtilsTests(unittest.TestCase):
    def test_bounded_logit_kde_generates_curve_within_support(self) -> None:
        model = fit_score_density_model(
            [0.02, 0.18, 0.41, 0.77, 0.81],
            method="bounded_logit_kde",
            bandwidth_scale=1.0,
        )

        grid, density = model.curve(lower=0.0, upper=0.9, points=128)

        self.assertGreaterEqual(min(grid), 0.0)
        self.assertLessEqual(max(grid), 0.9)
        self.assertTrue(all(value >= 0.0 for value in density))

    def test_bounded_logit_kde_prefers_positive_region_for_positive_scores(self) -> None:
        model = fit_score_density_model(
            [0.12, 0.17, 0.23, 0.35, 0.44, 0.59, 0.73],
            method="bounded_logit_kde",
            bandwidth_scale=1.0,
        )

        negative_density = float(model.evaluate_raw([-0.15])[0])
        central_density = float(model.evaluate_raw([0.35])[0])

        self.assertLess(negative_density, central_density)

    def test_density_method_label_returns_user_facing_name(self) -> None:
        self.assertEqual("KDE limitada por logito", score_density_method_label("bounded_logit_kde"))
        self.assertEqual("KDE gaussiana direta", score_density_method_label("gaussian_kde"))


if __name__ == "__main__":
    unittest.main()
