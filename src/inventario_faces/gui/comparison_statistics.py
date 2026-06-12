from __future__ import annotations

from dataclasses import dataclass
from html import escape

import numpy as np
from scipy.stats import mannwhitneyu

from inventario_faces.domain.entities import FaceSetComparisonMatch, FaceSetComparisonSummary


@dataclass(frozen=True)
class _DistributionSeries:
    label: str
    classification: str
    color: str
    values: tuple[float, ...]
    sufficient: bool
    note: str | None = None
    kde_x: tuple[float, ...] = ()
    kde_y: tuple[float, ...] = ()
    mean: float | None = None
    median: float | None = None
    q1: float | None = None
    q3: float | None = None
    ci_low: float | None = None
    ci_high: float | None = None


@dataclass(frozen=True)
class _GroupComparisonTestResult:
    metric_label: str
    left_label: str
    right_label: str
    left_count: int
    right_count: int
    left_median: float | None = None
    right_median: float | None = None
    u_statistic: float | None = None
    p_value: float | None = None
    rank_biserial: float | None = None
    common_language_effect: float | None = None
    significant: bool | None = None
    available: bool = False
    note: str | None = None


def _expanded_score_range(
    values: list[float] | tuple[float, ...],
    *,
    observed_score: float | None = None,
    minimum_span: float = 0.2,
) -> tuple[float, float]:
    numeric_values = [float(value) for value in values]
    if observed_score is not None:
        numeric_values.append(float(observed_score))
    if not numeric_values:
        return 0.0, 1.0

    min_value = min(numeric_values)
    max_value = max(numeric_values)
    lower = min_value
    upper = max_value
    span = upper - lower
    padding = max(span * 0.08, 0.02)
    lower -= padding
    upper += padding

    if (upper - lower) < minimum_span:
        center = (lower + upper) / 2.0
        lower = center - (minimum_span / 2.0)
        upper = center + (minimum_span / 2.0)

    if min_value >= 0.0:
        lower = max(0.0, lower)
        upper = max(upper, min(1.0, lower + minimum_span))
    if max_value <= 0.0:
        upper = min(0.0, upper)
        lower = min(lower, max(-1.0, upper - minimum_span))

    lower = max(-1.0, lower)
    upper = min(1.0, upper)

    if upper <= lower:
        if min_value >= 0.0:
            lower = 0.0
            upper = min(1.0, max(minimum_span, max_value + 0.02))
        elif max_value <= 0.0:
            upper = 0.0
            lower = max(-1.0, min(-minimum_span, min_value - 0.02))
        else:
            lower = max(-1.0, min_value - 0.1)
            upper = min(1.0, max_value + 0.1)

    return float(lower), float(upper)


def _histogram_density(
    values: list[float] | tuple[float, ...],
    *,
    lower: float,
    upper: float,
    minimum_bins: int = 16,
    maximum_bins: int = 48,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    if not values or upper <= lower:
        return (), ()

    array = np.asarray(values, dtype=np.float64)
    sample_size = int(array.size)
    if sample_size <= 1:
        return (), ()

    q1, q3 = np.quantile(array, [0.25, 0.75], method="linear")
    iqr = float(q3 - q1)
    if iqr > 1e-12:
        bin_width = 2.0 * iqr * (sample_size ** (-1.0 / 3.0))
        estimated_bins = int(np.ceil((upper - lower) / bin_width)) if bin_width > 1e-12 else minimum_bins
    else:
        estimated_bins = int(np.ceil(np.sqrt(sample_size)))
    bin_count = max(minimum_bins, min(maximum_bins, estimated_bins))

    histogram, edges = np.histogram(array, bins=bin_count, range=(lower, upper), density=True)
    return (
        tuple(float(value) for value in edges),
        tuple(float(value) for value in histogram),
    )


def _mann_whitney_group_comparison(
    left_values: list[float] | tuple[float, ...],
    right_values: list[float] | tuple[float, ...],
    *,
    alpha: float,
    metric_label: str,
    left_label: str = "Padrão",
    right_label: str = "Questionado",
) -> _GroupComparisonTestResult:
    left_sample = [float(value) for value in left_values]
    right_sample = [float(value) for value in right_values]
    left_count = len(left_sample)
    right_count = len(right_sample)
    if left_count < 2 or right_count < 2:
        return _GroupComparisonTestResult(
            metric_label=metric_label,
            left_label=left_label,
            right_label=right_label,
            left_count=left_count,
            right_count=right_count,
            available=False,
            note=(
                "Teste U de Mann-Whitney indisponível: são necessárias ao menos 2 observações válidas "
                "em Padrão e Questionado."
            ),
        )

    left_array = np.asarray(left_sample, dtype=np.float64)
    right_array = np.asarray(right_sample, dtype=np.float64)
    try:
        test_result = mannwhitneyu(left_array, right_array, alternative="two-sided", method="auto")
    except TypeError:
        test_result = mannwhitneyu(left_array, right_array, alternative="two-sided")
    except ValueError as exc:
        return _GroupComparisonTestResult(
            metric_label=metric_label,
            left_label=left_label,
            right_label=right_label,
            left_count=left_count,
            right_count=right_count,
            available=False,
            note=f"Teste U de Mann-Whitney indisponível: {exc}",
        )

    pair_count = left_count * right_count
    u_statistic = float(test_result.statistic)
    p_value = float(test_result.pvalue)
    common_language_effect = (u_statistic / pair_count) if pair_count > 0 else None
    rank_biserial = (
        (2.0 * common_language_effect) - 1.0 if common_language_effect is not None else None
    )
    return _GroupComparisonTestResult(
        metric_label=metric_label,
        left_label=left_label,
        right_label=right_label,
        left_count=left_count,
        right_count=right_count,
        left_median=float(np.median(left_array)),
        right_median=float(np.median(right_array)),
        u_statistic=u_statistic,
        p_value=p_value,
        rank_biserial=rank_biserial,
        common_language_effect=common_language_effect,
        significant=(p_value <= alpha),
        available=True,
    )


def _format_density_value(value: float | None) -> str:
    if value is None or not np.isfinite(value):
        return "-"
    absolute = abs(float(value))
    if absolute == 0.0:
        return "0"
    if 1e-3 <= absolute < 1e3:
        return f"{value:.6f}".rstrip("0").rstrip(".")
    return f"{value:.3e}"


def _likelihood_ratio_selection_html(
    match: FaceSetComparisonMatch | None,
    *,
    left_name: str = "-",
    right_name: str = "-",
) -> str:
    if match is None:
        return "Linha tracejada azul: nenhum confronto selecionado."

    left = escape(left_name)
    right = escape(right_name)
    score_text = f"{match.similarity:.4f}"
    header = (
        "Linha tracejada azul: confronto selecionado na tabela | "
        f"rank {match.rank} | similaridade {score_text} | {left} x {right}"
    )
    if (
        match.same_source_density is None
        or match.different_source_density is None
        or match.likelihood_ratio is None
        or match.log10_likelihood_ratio is None
    ):
        return (
            f"{header}<br>"
            "<b>Leitura do gráfico</b>: o LR é obtido pela razão entre as alturas "
            "das curvas H1 e H2 exatamente no ponto da linha tracejada."
        )

    h1_density = _format_density_value(match.same_source_density)
    h2_density = _format_density_value(match.different_source_density)
    lr_value = _format_density_value(match.likelihood_ratio)
    log10_lr = f"{match.log10_likelihood_ratio:.4f}"
    favored_hypothesis = "H1 (mesma origem)" if match.likelihood_ratio >= 1.0 else "H2 (origem distinta)"
    return (
        f"{header}<br>"
        f"<b>Leitura do gráfico</b>: no score selecionado <code>x={score_text}</code>, "
        f"a curva verde fornece <code>f(score|H1)={h1_density}</code> e a curva vermelha "
        f"<code>f(score|H2)={h2_density}</code>.<br>"
        f"<b>Cálculo</b>: <code>LR = f(score|H1) / f(score|H2) = {h1_density} / {h2_density} = {lr_value}</code><br>"
        f"<b>Escala log</b>: <code>log10(LR) = {log10_lr}</code> | "
        f"<b>Leitura</b>: neste ponto, a evidência favorece <b>{escape(favored_hypothesis)}</b>."
    )


def _distribution_zone_label(
    score: float | None,
    *,
    candidate_threshold: float,
    assignment_threshold: float,
) -> str:
    if score is None:
        return "sem escore observado"
    if score >= assignment_threshold:
        return "faixa de atribuição dos pares Padrão x Questionado"
    if score >= candidate_threshold:
        return "faixa candidata dos pares Padrão x Questionado"
    return "faixa abaixo do limiar candidato nos pares Padrão x Questionado"


def _distribution_zone_short_label(
    score: float | None,
    *,
    candidate_threshold: float,
    assignment_threshold: float,
) -> str:
    if score is None:
        return "Padrão x Questionado sem escore"
    if score >= assignment_threshold:
        return "Padrão x Questionado em atribuição"
    if score >= candidate_threshold:
        return "Padrão x Questionado em faixa candidata"
    return "Padrão x Questionado abaixo do limiar"


def _distribution_format_float(value: float | None) -> str:
    return "-" if value is None else f"{value:.4f}"


def _distribution_series_label(classification: str) -> str:
    mapping = {
        "assignment": "PxQ atribuição",
        "candidate": "PxQ candidata",
        "below_threshold": "PxQ abaixo do limiar",
    }
    return mapping.get(classification, classification)


def _distribution_series_text_label(classification: str) -> str:
    mapping = {
        "assignment": "pares Padrão x Questionado em atribuição",
        "candidate": "pares Padrão x Questionado em faixa candidata",
        "below_threshold": "pares Padrão x Questionado abaixo do limiar",
    }
    return mapping.get(classification, classification)


def _distribution_card_title(classification: str) -> str:
    mapping = {
        "assignment": "Média Padrão x Questionado em atribuição",
        "candidate": "Média Padrão x Questionado em faixa candidata",
        "below_threshold": "Média Padrão x Questionado abaixo do limiar",
    }
    return mapping.get(classification, classification)


def _distribution_popup_html(
    summary: FaceSetComparisonSummary,
    series_list: list[_DistributionSeries],
    overall_stats: dict[str, float | None],
    *,
    significance_percent: float,
    bootstrap_resamples: int,
) -> str:
    confidence_level = max(0.0, 100.0 - significance_percent)
    best_zone = _distribution_zone_label(
        summary.best_similarity,
        candidate_threshold=summary.candidate_threshold,
        assignment_threshold=summary.assignment_threshold,
    )
    supported_series = [series for series in series_list if series.sufficient]
    center_items = "".join(
        (
            "<li>"
            f"<b>{escape(_distribution_series_text_label(series.classification))}</b>: "
            f"média <code>{_distribution_format_float(series.mean)}</code>"
            "</li>"
        )
        for series in supported_series
        if series.mean is not None
    )
    class_items = "".join(
        (
            "<li>"
            f"<b>{escape(_distribution_series_text_label(series.classification))}</b>: "
            f"n={len(series.values)} | "
            f"média <code>{_distribution_format_float(series.mean)}</code> | "
            f"mediana <code>{_distribution_format_float(series.median)}</code> | "
            "intervalo interquartil "
            f"<code>{_distribution_format_float(series.q1)} a {_distribution_format_float(series.q3)}</code>"
            "</li>"
        )
        if series.sufficient
        else (
            "<li>"
            f"<b>{escape(_distribution_series_text_label(series.classification))}</b>: n={len(series.values)} | "
            f"{escape(series.note or 'Curva não exibida por falta de suporte estatístico.')}"
            "</li>"
        )
        for series in series_list
    )
    return (
        "<style>"
        "body { font-family: 'Segoe UI'; color: #0f172a; line-height: 1.42; }"
        "h3 { margin: 0 0 8px 0; color: #0f172a; }"
        "p { margin: 0 0 10px 0; }"
        "ul { margin: 0 0 12px 18px; }"
        "li { margin: 0 0 6px 0; }"
        "code { background: #f8fafc; padding: 1px 4px; border-radius: 4px; }"
        ".lede { margin: 0 0 12px 0; color: #334155; }"
        ".panel { background: #f8fafc; border: 1px solid #d9e3ee; border-radius: 10px; padding: 10px 12px; margin: 0 0 12px 0; }"
        ".panel strong { color: #0f172a; }"
        ".small { color: #475569; }"
        ".kicker { color: #0f766e; font-size: 11px; font-weight: 700; letter-spacing: 0.04em; text-transform: uppercase; margin: 0 0 4px 0; }"
        "</style>"
        "<h3>Leitura técnica do gráfico</h3>"
        "<p class='lede'>"
        "Este gráfico não traça uma curva isolada para Padrão e outra para Questionado. "
        "Como a unidade analisada aqui é o <b>par Padrão x Questionado</b>, "
        "todas as curvas representam distribuições de pares <b>Padrão x Questionado</b>; "
        "o que muda entre elas é a <b>faixa decisória</b> em que cada par caiu."
        "</p>"
        "<div class='panel'>"
        "<div class='kicker'>Orientação</div>"
        "<p>"
        "Nas legendas curtas do gráfico, <b>PxQ</b> significa <b>Padrão x Questionado</b>."
        "</p>"
        "<p>"
        "As curvas coloridas mostram a densidade suavizada dos escores dos pares Padrão x Questionado "
        "em cada faixa decisória. As <b>linhas tracejadas coloridas</b> marcam a <b>média de cada distribuição</b>, "
        "isto é, o <b>centro de gravidade</b> de cada curva. A linha tracejada escura marca a média geral de "
        f"todos os <b>{summary.total_pair_comparisons}</b> pares Padrão x Questionado comparados."
        "</p>"
        "</div>"
        "<div class='panel'>"
        "<div class='kicker'>Leitura rápida</div>"
        "<ul>"
        "<li><b>Linhas sólidas</b> = limiares de decisão.</li>"
        "<li><b>Linhas tracejadas</b> = médias / centros de gravidade.</li>"
        "<li><b>Linha azul contínua</b> = melhor escore observado.</li>"
        "<li><b>Faixa cinza</b> = IC bootstrap da média geral.</li>"
        "</ul>"
        "<p>"
        "Nesta execução, o limiar de candidata é "
        f"<code>{summary.candidate_threshold:.4f}</code> e o limiar de atribuição é "
        f"<code>{summary.assignment_threshold:.4f}</code>."
        "</p>"
        "</div>"
        "<div class='panel'>"
        "<div class='kicker'>Melhor Escore no Contexto</div>"
        "<p>"
        "A <b>linha azul contínua</b> marca o melhor escore observado "
        f"<code>{_distribution_format_float(summary.best_similarity)}</code>. "
        "Neste momento, ele está situado na "
        f"<b>{escape(best_zone)}</b>."
        "</p>"
        "<p>"
        f"A faixa vertical acinzentada mostra o <b>IC bootstrap ({confidence_level:.2f}%) da média geral</b>, "
        f"calculado com <b>{bootstrap_resamples}</b> reamostragens."
        "</p>"
        "<p>"
        "Em termos práticos: a curva verde reúne os pares Padrão x Questionado que alcançaram a faixa de atribuição; "
        "a curva laranja reúne os pares que ficaram na faixa candidata; "
        "a curva roxa reúne os pares que permaneceram abaixo do limiar candidato."
        "</p>"
        "</div>"
        "<h3>Centros de gravidade</h3>"
        f"<ul>{center_items or '<li>Não houve classe com suporte suficiente para estimar a média da curva.</li>'}</ul>"
        "<h3>Resumo por faixa decisória</h3>"
        f"<ul>{class_items}</ul>"
        "<div class='panel small'>"
        "<div class='kicker'>Síntese Geral</div>"
        "<p>"
        f"média <code>{_distribution_format_float(overall_stats.get('mean'))}</code> | "
        f"mediana <code>{_distribution_format_float(overall_stats.get('median'))}</code> | "
        f"Q1 <code>{_distribution_format_float(overall_stats.get('q1'))}</code> | "
        f"Q3 <code>{_distribution_format_float(overall_stats.get('q3'))}</code> | "
        f"IC bootstrap <code>{_distribution_format_float(overall_stats.get('ci_low'))} a "
        f"{_distribution_format_float(overall_stats.get('ci_high'))}</code>."
        "</p>"
        "</div>"
    )


def _distribution_help_html() -> str:
    return (
        "<style>"
        "body { font-family: 'Segoe UI'; color: #0f172a; line-height: 1.4; }"
        "h2, h3 { color: #0f172a; }"
        "p { margin: 0 0 10px 0; }"
        "ul { margin: 0 0 12px 18px; }"
        "li { margin: 0 0 6px 0; }"
        "</style>"
        "<h2>Ajuda da distribuição de similaridades</h2>"
        "<p>"
        "Este painel resume como os escores de similaridade dos pares Padrão x Questionado "
        "se distribuem entre as faixas decisórias abaixo do limiar, candidata e atribuição."
        "</p>"
        "<p>"
        "Nas legendas curtas do gráfico, <b>PxQ</b> significa <b>Padrão x Questionado</b>."
        "</p>"
        "<h3>Por que não há uma curva \"Padrão\" e outra \"Questionado\"?</h3>"
        "<p>"
        "Porque a similaridade não pertence a Padrão ou a Questionado isoladamente: ela nasce do "
        "<b>par Padrão x Questionado</b>. Por isso, cada curva do gráfico representa um subconjunto "
        "de pares Padrão x Questionado, separado conforme a faixa decisória em que o score caiu."
        "</p>"
        "<h3>O que representa cada curva</h3>"
        "<ul>"
        "<li><b>Pares Padrão x Questionado em atribuição</b>: reúne os confrontos cujo escore ficou igual ou acima do limiar de atribuição. É a região dos vínculos mais fortes dentro da regra de decisão adotada.</li>"
        "<li><b>Pares Padrão x Questionado em faixa candidata</b>: reúne os confrontos entre o limiar de candidata e o limiar de atribuição. Corresponde à zona intermediária, em que o escore merece atenção, mas ainda não alcança o patamar de atribuição.</li>"
        "<li><b>Pares Padrão x Questionado abaixo do limiar</b>: reúne os confrontos que ficaram abaixo do limiar de candidata. Em regra, representa a faixa de escores mais fracos nesta execução.</li>"
        "</ul>"
        "<h3>Como ler o gráfico</h3>"
        "<ul>"
        "<li><b>Curvas coloridas</b>: densidade suavizada (KDE) dos escores dos pares Padrão x Questionado em cada faixa decisória.</li>"
        "<li><b>Linhas tracejadas coloridas</b>: média de cada faixa decisória, isto é, o centro de gravidade da curva correspondente.</li>"
        "<li><b>Linha tracejada escura</b>: média geral de todos os pares Padrão x Questionado comparados.</li>"
        "<li><b>Linhas sólidas e faixas de fundo</b>: limiares de decisão e as regiões abaixo do limiar, candidata e atribuição.</li>"
        "<li><b>Linha azul contínua</b>: melhor escore observado no ranking atual.</li>"
        "<li><b>Faixa vertical cinza</b>: intervalo de confiança bootstrap da média geral.</li>"
        "</ul>"
        "<h3>Como interpretar</h3>"
        "<p>"
        "Quando a média da curva dos pares Padrão x Questionado em atribuição aparece deslocada à direita da média da curva candidata, "
        "os pares classificados como atribuição tendem a ter escores mais altos. Quando a curva abaixo do limiar ocupa a região "
        "à esquerda, isso indica concentração de escores mais fracos."
        "</p>"
        "<p>"
        "Uma curva mais deslocada para a direita indica tendência a escores mais altos. "
        "Uma curva mais estreita e alta sugere menor dispersão e maior homogeneidade dos escores daquela faixa decisória; "
        "uma curva mais larga e baixa sugere maior variabilidade interna."
        "</p>"
        "<p>"
        "Quando duas curvas se sobrepõem bastante, a separação entre as faixas decisórias fica menos nítida naquela faixa de score. "
        "Quando os centros de gravidade ficam bem afastados, a diferença entre as faixas decisórias se torna mais evidente."
        "</p>"
        "<p>"
        "As alturas das curvas representam <b>densidade relativa</b>, e não probabilidade direta de identificação. "
        "Em outras palavras: o gráfico mostra onde os escores dos pares Padrão x Questionado tendem a se concentrar, "
        "não uma chance direta de autoria. O texto abaixo do gráfico detalha médias, medianas, quartis, suporte amostral "
        "e a posição do melhor escore em relação aos limiares."
        "</p>"
    )


def _summary_format_p_value(value: float | None) -> str:
    if value is None:
        return "-"
    if value < 1e-4:
        return f"{value:.2e}"
    return f"{value:.4f}"


def _group_comparison_direction_text(test_result: _GroupComparisonTestResult) -> str:
    if test_result.rank_biserial is None:
        return "sem tendência direcional relevante entre Padrão e Questionado"
    if test_result.rank_biserial > 0.05:
        return f"{test_result.left_label} tende a apresentar {test_result.metric_label} maior"
    if test_result.rank_biserial < -0.05:
        return f"{test_result.right_label} tende a apresentar {test_result.metric_label} maior"
    return "sem tendência direcional relevante entre Padrão e Questionado"


def _group_comparison_significance_text(
    test_result: _GroupComparisonTestResult,
    *,
    significance_percent: float,
) -> str:
    return (
        f"diferença estatisticamente significativa ao nível de {significance_percent:.2f}%"
        if test_result.significant
        else f"diferença não significativa ao nível de {significance_percent:.2f}%"
    )


def _summary_popup_html(
    summary: FaceSetComparisonSummary,
    overall_stats: dict[str, float | None],
    group_test: _GroupComparisonTestResult,
    series_list: list[_DistributionSeries],
    *,
    significance_percent: float,
    bootstrap_resamples: int,
    procedure_details: tuple[str, ...] = (),
    support_note: str | None = None,
) -> str:
    confidence_level = max(0.0, 100.0 - significance_percent)
    best_zone = _distribution_zone_label(
        summary.best_similarity,
        candidate_threshold=summary.candidate_threshold,
        assignment_threshold=summary.assignment_threshold,
    )
    support_items = "".join(
        (
            "<li>"
            f"<b>{escape(_distribution_series_text_label(series.classification))}</b>: "
            f"n={len(series.values)} | "
            f"média <code>{_distribution_format_float(series.mean)}</code> | "
            f"mediana <code>{_distribution_format_float(series.median)}</code> | "
            "intervalo interquartil "
            f"<code>{_distribution_format_float(series.q1)} a {_distribution_format_float(series.q3)}</code>"
            "</li>"
        )
        if series.sufficient
        else (
            "<li>"
            f"<b>{escape(_distribution_series_text_label(series.classification))}</b>: "
            f"n={len(series.values)} | "
            f"{escape(series.note or 'Amostra insuficiente para estimar a distribuição desta faixa.')}"
            "</li>"
        )
        for series in series_list
    )
    procedure_items = "".join(
        f"<li>{escape(line)}</li>"
        for line in procedure_details
        if line.strip()
    )
    if overall_stats.get("mean") is None:
        inference_panel = (
            "<div class='panel warning'>"
            "<div class='kicker'>Panorama inferencial</div>"
            "<p><b>As estatísticas globais inferenciais não puderam ser estimadas com suporte suficiente.</b></p>"
            f"<p>{escape(support_note or 'Amostra insuficiente para bootstrap e sumarização robusta.')}</p>"
            "<p>"
            f"O nível de significância adotado nesta leitura foi de <b>{significance_percent:.2f}%</b>, "
            f"equivalente a uma confiança nominal bootstrap de <b>{confidence_level:.2f}%</b>."
            "</p>"
            "</div>"
        )
    else:
        inference_panel = (
            "<div class='panel'>"
            "<div class='kicker'>Panorama inferencial</div>"
            "<p>"
            "As medidas abaixo resumem a distribuição global dos escores dos pares <b>Padrão x Questionado</b>. "
            f"A média geral foi <code>{_distribution_format_float(overall_stats.get('mean'))}</code>, "
            f"a mediana foi <code>{_distribution_format_float(overall_stats.get('median'))}</code> "
            "e a metade central dos escores ficou entre "
            f"<code>{_distribution_format_float(overall_stats.get('q1'))}</code> e "
            f"<code>{_distribution_format_float(overall_stats.get('q3'))}</code>."
            "</p>"
            "<p>"
            f"O <b>IC bootstrap ({confidence_level:.2f}%) da média geral</b> vai de "
            f"<code>{_distribution_format_float(overall_stats.get('ci_low'))}</code> a "
            f"<code>{_distribution_format_float(overall_stats.get('ci_high'))}</code>, "
            f"calculado com <b>{bootstrap_resamples}</b> reamostragens com reposição, "
            f"adotando <b>nível de significância de {significance_percent:.2f}%</b>."
            "</p>"
            "<p>"
            f"O melhor escore observado foi <code>{_distribution_format_float(summary.best_similarity)}</code>, "
            f"atualmente situado na <b>{escape(best_zone)}</b>."
            "</p>"
            "</div>"
        )
    if not group_test.available:
        group_panel = (
            "<div class='panel warning'>"
            "<div class='kicker'>Comparação entre Padrão e Questionado</div>"
            "<p><b>O teste de comparação da qualidade facial não pôde ser estimado.</b></p>"
            f"<p>{escape(group_test.note or 'Teste U de Mann-Whitney indisponível.')}</p>"
            "</div>"
        )
    else:
        significance_label = _group_comparison_significance_text(
            group_test,
            significance_percent=significance_percent,
        )
        direction = _group_comparison_direction_text(group_test)
        group_panel = (
            "<div class='panel'>"
            "<div class='kicker'>Comparação entre Padrão e Questionado</div>"
            "<p>"
            "O teste <b>U de Mann-Whitney bilateral</b> compara a <b>qualidade facial</b> das faces "
            "selecionadas em <b>Padrão</b> e <b>Questionado</b>. "
            "Ele <b>não compara diretamente os escores de similaridade entre pares</b>; "
            "ele verifica se um dos grupos entrou na comparação com faces sistematicamente melhores."
            "</p>"
            "<ul>"
            f"<li><b>Tamanho amostral</b>: {group_test.left_label} n={group_test.left_count} | "
            f"{group_test.right_label} n={group_test.right_count}</li>"
            f"<li><b>Medianas de qualidade</b>: {group_test.left_label} "
            f"<code>{_distribution_format_float(group_test.left_median)}</code> | "
            f"{group_test.right_label} <code>{_distribution_format_float(group_test.right_median)}</code></li>"
            f"<li><b>U</b>: <code>{_distribution_format_float(group_test.u_statistic)}</code> | "
            f"<b>p-valor bilateral</b>: <code>{_summary_format_p_value(group_test.p_value)}</code></li>"
            f"<li><b>Correlação bisserial de postos</b>: "
            f"<code>{_distribution_format_float(group_test.rank_biserial)}</code></li>"
            f"<li><b>Probabilidade de superioridade comum</b> "
            f"({group_test.left_label} &gt; {group_test.right_label}): "
            f"<code>{_distribution_format_float(group_test.common_language_effect)}</code></li>"
            "</ul>"
            "<p>"
            f"<b>Interpretação</b>: {escape(significance_label)}; {escape(direction)}."
            "</p>"
            "</div>"
        )
    return (
        "<style>"
        "body { font-family: 'Segoe UI'; color: #0f172a; line-height: 1.42; }"
        "h3 { margin: 0 0 8px 0; color: #0f172a; }"
        "p { margin: 0 0 10px 0; }"
        "ul { margin: 0 0 12px 18px; }"
        "li { margin: 0 0 6px 0; }"
        "code { background: #f8fafc; padding: 1px 4px; border-radius: 4px; }"
        ".lede { margin: 0 0 12px 0; color: #334155; }"
        ".panel { background: #f8fafc; border: 1px solid #d9e3ee; border-radius: 10px; padding: 10px 12px; margin: 0 0 12px 0; }"
        ".panel.warning { background: #fff7ed; border-color: #f3d6b3; }"
        ".small { color: #475569; }"
        ".kicker { color: #0f766e; font-size: 11px; font-weight: 700; letter-spacing: 0.04em; text-transform: uppercase; margin: 0 0 4px 0; }"
        "</style>"
        "<h3>Leitura técnica do resumo estatístico</h3>"
        "<p class='lede'>"
        "Este painel combina duas leituras complementares: "
        "a distribuição global dos escores dos pares <b>Padrão x Questionado</b> "
        "e a comparação da <b>qualidade facial</b> das faces selecionadas em cada grupo."
        "</p>"
        "<div class='panel'>"
        "<div class='kicker'>Orientação</div>"
        "<p>"
        "Os cards do topo condensam o panorama numérico principal. "
        "Eles devem ser lidos em conjunto com o texto abaixo: "
        "os cards mostram os valores; a leitura técnica explica o que esses valores significam."
        "</p>"
        "<p>"
        f"Nesta execução foram comparados <b>{summary.total_pair_comparisons}</b> pares Padrão x Questionado, "
        f"a partir de <b>{summary.set_a_selected_faces}</b> faces selecionadas no Padrão e "
        f"<b>{summary.set_b_selected_faces}</b> no Questionado."
        "</p>"
        "<p>"
        f"O nível de significância adotado nesta leitura foi de <b>{significance_percent:.2f}%</b>; "
        f"por isso, o bootstrap foi resumido como um intervalo com confiança nominal aproximada de <b>{confidence_level:.2f}%</b>."
        "</p>"
        "</div>"
        f"{inference_panel}"
        f"{group_panel}"
        "<h3>Suporte por faixa decisória</h3>"
        "<p>"
        "A relação abaixo mostra quantos pares Padrão x Questionado sustentaram cada faixa decisória "
        "e como esses subconjuntos se comportaram em termos de média, mediana e intervalo interquartil."
        "</p>"
        f"<ul>{support_items or '<li>Não houve suporte suficiente para detalhar as faixas decisórias.</li>'}</ul>"
        "<div class='panel small'>"
        "<div class='kicker'>Procedimento e configuração usada</div>"
        "<ul>"
        f"{procedure_items or '<li>Sem detalhes adicionais de procedimento registrados nesta execução.</li>'}"
        f"<li>IC bootstrap da média com {bootstrap_resamples} reamostragens e significância de {significance_percent:.2f}% "
        f"(confiança nominal aproximada de {confidence_level:.2f}%).</li>"
        "<li>Intervalo de confiança obtido por bootstrap percentílico não paramétrico, com reamostragem com reposição dos scores observados.</li>"
        "<li>Comparação entre Padrão e Questionado realizada por U de Mann-Whitney bilateral sobre a qualidade facial das faces selecionadas.</li>"
        "</ul>"
        "</div>"
    )


def _summary_help_html() -> str:
    return (
        "<style>"
        "body { font-family: 'Segoe UI'; color: #0f172a; line-height: 1.4; }"
        "h2, h3 { color: #0f172a; }"
        "p { margin: 0 0 10px 0; }"
        "ul { margin: 0 0 12px 18px; }"
        "li { margin: 0 0 6px 0; }"
        "</style>"
        "<h2>Ajuda do resumo estatístico</h2>"
        "<p>"
        "Este painel resume a execução por dois eixos: "
        "os escores dos pares <b>Padrão x Questionado</b> e a qualidade facial das faces selecionadas "
        "em cada grupo."
        "</p>"
        "<h3>Como ler os cards do topo</h3>"
        "<ul>"
        "<li><b>Pares comparados</b>: total de pares Padrão x Questionado efetivamente avaliados.</li>"
        "<li><b>Faces aproveitadas</b>: quantidade de faces escolhidas no Padrão e no Questionado para gerar os pares e sustentar o teste de qualidade.</li>"
        "<li><b>Maior similaridade observada</b>: melhor escore desta execução e a faixa decisória em que ele caiu.</li>"
        "<li><b>Pares por faixa decisória</b>: quantos pares chegaram à atribuição, quantos ficaram na faixa candidata e quantos permaneceram abaixo do limiar.</li>"
        "<li><b>Limiar candidata / atribuição</b>: limiares decisórios usados para separar faixa candidata e faixa de atribuição.</li>"
        "<li><b>Média, mediana e faixa interquartil</b>: descrevem o centro e a dispersão dos escores dos pares Padrão x Questionado.</li>"
        "<li><b>IC bootstrap da média</b>: intervalo de confiança da média geral, estimado por reamostragem com reposição.</li>"
        "</ul>"
        "<h3>O que o IC bootstrap responde</h3>"
        "<p>"
        "O IC bootstrap quantifica a estabilidade da média dos escores observados. "
        "Intervalos mais estreitos sugerem estimativa mais estável; intervalos mais largos sugerem maior incerteza."
        "</p>"
        "<p>"
        "O nível de significância adotado não aparece como card isolado; ele é informado no texto técnico "
        "porque serve como parâmetro de leitura do IC bootstrap e da interpretação inferencial."
        "</p>"
        "<h3>O que o teste entre Padrão e Questionado responde</h3>"
        "<p>"
        "O teste <b>U de Mann-Whitney bilateral</b> compara a <b>qualidade facial</b> das faces selecionadas "
        "em Padrão e Questionado. Ele não testa identidade, nem compara diretamente os escores de similaridade "
        "dos pares. O objetivo é indicar se um dos grupos entrou na comparação com qualidade facial "
        "sistematicamente maior."
        "</p>"
        "<ul>"
        "<li><b>p-valor bilateral</b>: informa se a diferença observada é compatível com a hipótese de distribuições semelhantes.</li>"
        "<li><b>Correlação bisserial de postos</b>: resume direção e magnitude do deslocamento entre Padrão e Questionado.</li>"
        "<li><b>Probabilidade de superioridade comum</b>: chance estimada de uma face escolhida ao acaso em Padrão superar uma face escolhida ao acaso em Questionado na métrica de qualidade facial.</li>"
        "</ul>"
        "<h3>Suporte por faixa decisória</h3>"
        "<p>"
        "A seção por faixa decisória mostra como os pares Padrão x Questionado se distribuíram entre "
        "atribuição, faixa candidata e abaixo do limiar. Quando uma faixa tem suporte insuficiente, "
        "o painel informa isso explicitamente para evitar leitura exagerada."
        "</p>"
    )
