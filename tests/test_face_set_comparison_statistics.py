from __future__ import annotations

import unittest

from tests._bootstrap import PROJECT_ROOT  # noqa: F401
from inventario_faces.domain.entities import FaceSetComparisonMatch, FaceSetComparisonSummary
from inventario_faces.gui.face_set_comparison_dialog import (
    _DistributionSeries,
    _distribution_card_title,
    _distribution_help_html,
    _distribution_popup_html,
    _distribution_zone_short_label,
    _likelihood_ratio_selection_html,
    _mann_whitney_group_comparison,
    _summary_help_html,
    _summary_popup_html,
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

    def test_distribution_popup_html_explains_dashed_means_and_thresholds(self) -> None:
        html = _distribution_popup_html(
            FaceSetComparisonSummary(
                total_pair_comparisons=98,
                best_similarity=0.7302,
                candidate_threshold=0.5600,
                assignment_threshold=0.6200,
            ),
            [
                _DistributionSeries(
                    label="Pares Padrão x Questionado (atribuição)",
                    classification="assignment",
                    color="#0f766e",
                    values=(0.61, 0.64, 0.68, 0.71, 0.73),
                    sufficient=True,
                    mean=0.6740,
                    median=0.6800,
                    q1=0.6400,
                    q3=0.7100,
                ),
                _DistributionSeries(
                    label="Pares Padrão x Questionado (candidata)",
                    classification="candidate",
                    color="#b45309",
                    values=(0.55, 0.57, 0.58, 0.59, 0.60),
                    sufficient=True,
                    mean=0.5780,
                    median=0.5800,
                    q1=0.5700,
                    q3=0.5900,
                ),
                _DistributionSeries(
                    label="Pares Padrão x Questionado (abaixo do limiar)",
                    classification="below_threshold",
                    color="#7c3aed",
                    values=(0.14, 0.18),
                    sufficient=False,
                    note="Amostra insuficiente.",
                ),
            ],
            {
                "mean": 0.4785,
                "median": 0.5772,
                "q1": 0.4038,
                "q3": 0.6426,
                "ci_low": 0.4303,
                "ci_high": 0.5233,
            },
            significance_percent=5.0,
            bootstrap_resamples=5000,
        )

        self.assertIn("linhas tracejadas coloridas", html)
        self.assertIn("centro de gravidade", html)
        self.assertIn("limiares de decisão", html)
        self.assertIn("Linhas sólidas", html)
        self.assertIn("linha azul contínua", html)
        self.assertIn("IC bootstrap", html)
        self.assertIn("par Padrão x Questionado", html)
        self.assertIn("não traça uma curva isolada para Padrão e outra para Questionado", html)
        self.assertIn("PxQ", html)
        self.assertIn("Leitura rápida", html)
        self.assertIn("Melhor Escore no Contexto", html)
        self.assertIn("Resumo por faixa decisória", html)
        self.assertIn("Síntese Geral", html)

    def test_distribution_help_html_documents_visual_elements(self) -> None:
        html = _distribution_help_html()

        self.assertIn("Linhas tracejadas coloridas", html)
        self.assertIn("centro de gravidade", html)
        self.assertIn("Linhas sólidas e faixas de fundo", html)
        self.assertIn("Linha azul contínua", html)
        self.assertIn("Faixa vertical cinza", html)
        self.assertIn("PxQ", html)
        self.assertIn('não há uma curva "Padrão" e outra "Questionado"', html)
        self.assertIn("Pares Padrão x Questionado em atribuição", html)
        self.assertIn("Pares Padrão x Questionado em faixa candidata", html)
        self.assertIn("Pares Padrão x Questionado abaixo do limiar", html)
        self.assertIn("sobrepõem bastante", html)
        self.assertIn("curva mais estreita e alta", html)

    def test_summary_popup_html_explains_bootstrap_and_quality_test(self) -> None:
        html = _summary_popup_html(
            FaceSetComparisonSummary(
                total_pair_comparisons=84,
                set_a_selected_faces=7,
                set_b_selected_faces=12,
                assignment_matches=23,
                candidate_matches=19,
                best_similarity=0.7302,
                candidate_threshold=0.5600,
                assignment_threshold=0.6200,
            ),
            {
                "mean": 0.4734,
                "median": 0.5772,
                "q1": 0.4038,
                "q3": 0.6426,
                "ci_low": 0.4227,
                "ci_high": 0.5220,
            },
            _mann_whitney_group_comparison(
                [0.81, 0.83, 0.86, 0.88, 0.9],
                [0.58, 0.61, 0.63, 0.65, 0.68],
                alpha=0.05,
                metric_label="qualidade facial",
                left_label="PadrÃ£o",
                right_label="Questionado",
            ),
            [
                _DistributionSeries(
                    label="Pares PadrÃ£o x Questionado (atribuiÃ§Ã£o)",
                    classification="assignment",
                    color="#0f766e",
                    values=(0.63, 0.66, 0.68, 0.71),
                    sufficient=True,
                    mean=0.6700,
                    median=0.6700,
                    q1=0.6450,
                    q3=0.6950,
                ),
                _DistributionSeries(
                    label="Pares PadrÃ£o x Questionado (candidata)",
                    classification="candidate",
                    color="#b45309",
                    values=(0.56, 0.58, 0.59, 0.60),
                    sufficient=True,
                    mean=0.5825,
                    median=0.5850,
                    q1=0.5750,
                    q3=0.5950,
                ),
                _DistributionSeries(
                    label="Pares PadrÃ£o x Questionado (abaixo do limiar)",
                    classification="below_threshold",
                    color="#7c3aed",
                    values=(0.19, 0.23),
                    sufficient=False,
                    note="Amostra insuficiente.",
                ),
            ],
            significance_percent=5.0,
            bootstrap_resamples=5000,
            procedure_details=(
                "Embeddings faciais comparados por similaridade cosseno.",
                "Faces selecionadas pela maior qualidade por entrada.",
            ),
        )

        self.assertIn("Leitura técnica do resumo estatístico", html)
        self.assertIn("Panorama inferencial", html)
        self.assertIn("IC bootstrap (95.00%)", html)
        self.assertIn("Comparação entre Padrão e Questionado", html)
        self.assertIn("não compara diretamente os escores de similaridade", html)
        self.assertIn("Correlação bisserial de postos", html)
        self.assertIn("Probabilidade de superioridade comum", html)
        self.assertIn("Suporte por faixa decisória", html)
        self.assertIn("Procedimento e configuração usada", html)

    def test_summary_help_html_documents_metrics_and_test_scope(self) -> None:
        html = _summary_help_html()

        self.assertIn("Ajuda do resumo estatístico", html)
        self.assertIn("Como ler os cards do topo", html)
        self.assertIn("IC bootstrap", html)
        self.assertIn("U de Mann-Whitney bilateral", html)
        self.assertIn("p-valor bilateral", html)
        self.assertIn("Correlação bisserial de postos", html)
        self.assertIn("Probabilidade de superioridade comum", html)
        self.assertIn("não testa identidade", html)
        self.assertIn("Padrão x Questionado", html)
        self.assertIn("Suporte por faixa decisória", html)

    def test_distribution_card_titles_use_full_case_labels(self) -> None:
        self.assertEqual(
            "Média Padrão x Questionado em atribuição",
            _distribution_card_title("assignment"),
        )
        self.assertEqual(
            "Média Padrão x Questionado em faixa candidata",
            _distribution_card_title("candidate"),
        )
        self.assertEqual(
            "Média Padrão x Questionado abaixo do limiar",
            _distribution_card_title("below_threshold"),
        )

    def test_distribution_best_score_card_label_uses_full_case_labels(self) -> None:
        self.assertEqual(
            "Padrão x Questionado em atribuição",
            _distribution_zone_short_label(
                0.73,
                candidate_threshold=0.56,
                assignment_threshold=0.62,
            ),
        )


if __name__ == "__main__":
    unittest.main()
