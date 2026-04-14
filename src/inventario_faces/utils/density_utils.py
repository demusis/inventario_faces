from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.stats import gaussian_kde

DEFAULT_SCORE_DENSITY_METHOD = "bounded_logit_kde"
SCORE_DENSITY_METHOD_CHOICES: tuple[tuple[str, str], ...] = (
    ("bounded_logit_kde", "KDE limitada por logito"),
    ("gaussian_kde", "KDE gaussiana direta"),
)
_SCORE_DENSITY_METHOD_LABELS = dict(SCORE_DENSITY_METHOD_CHOICES)
_SUPPORT_LOWER = -1.0
_SUPPORT_UPPER = 1.0
_SUPPORT_WIDTH = _SUPPORT_UPPER - _SUPPORT_LOWER
_EPSILON = 1e-6


def normalize_score_density_method(value: str | None) -> str:
    normalized = str(value or DEFAULT_SCORE_DENSITY_METHOD).strip().lower()
    if normalized not in _SCORE_DENSITY_METHOD_LABELS:
        allowed = ", ".join(method for method, _ in SCORE_DENSITY_METHOD_CHOICES)
        raise ValueError(f"Metodo de densidade invalido: {value!r}. Opcoes aceitas: {allowed}.")
    return normalized


def score_density_method_label(value: str | None) -> str:
    method = normalize_score_density_method(value)
    return _SCORE_DENSITY_METHOD_LABELS[method]


def _clip_score_array(values: np.ndarray) -> np.ndarray:
    return np.clip(values, _SUPPORT_LOWER + _EPSILON, _SUPPORT_UPPER - _EPSILON)


def _score_to_unit_interval(values: np.ndarray) -> np.ndarray:
    return (values - _SUPPORT_LOWER) / _SUPPORT_WIDTH


def _score_to_logit(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    clipped = _clip_score_array(values)
    unit_values = np.clip(_score_to_unit_interval(clipped), _EPSILON, 1.0 - _EPSILON)
    transformed = np.log(unit_values / (1.0 - unit_values))
    jacobian = 1.0 / (_SUPPORT_WIDTH * unit_values * (1.0 - unit_values))
    return transformed, jacobian


@dataclass(frozen=True)
class ScoreDensityModel:
    method: str
    kernel: gaussian_kde
    support_lower: float = _SUPPORT_LOWER
    support_upper: float = _SUPPORT_UPPER

    def evaluate_raw(self, values: float | list[float] | tuple[float, ...] | np.ndarray) -> np.ndarray:
        scores = np.asarray(values, dtype=np.float64)
        if scores.ndim == 0:
            scores = scores.reshape(1)
        if self.method == "bounded_logit_kde":
            transformed, jacobian = _score_to_logit(scores)
            density = np.asarray(self.kernel(transformed), dtype=np.float64) * jacobian
        else:
            density = np.asarray(self.kernel(_clip_score_array(scores)), dtype=np.float64)
        return np.maximum(0.0, density)

    def evaluate(
        self,
        values: float | list[float] | tuple[float, ...] | np.ndarray,
        *,
        uniform_floor_weight: float = 0.0,
        min_density: float = 0.0,
    ) -> np.ndarray:
        density = self.evaluate_raw(values)
        return stabilize_score_density(
            density,
            uniform_floor_weight=uniform_floor_weight,
            min_density=min_density,
            support_lower=self.support_lower,
            support_upper=self.support_upper,
        )

    def curve(
        self,
        *,
        lower: float,
        upper: float,
        points: int = 256,
        uniform_floor_weight: float = 0.0,
        min_density: float = 0.0,
    ) -> tuple[tuple[float, ...], tuple[float, ...]]:
        grid_lower = max(self.support_lower + _EPSILON, lower)
        grid_upper = min(self.support_upper - _EPSILON, upper)
        if grid_upper <= grid_lower:
            center = max(self.support_lower + _EPSILON, min(self.support_upper - _EPSILON, (lower + upper) / 2.0))
            half_span = min(0.05, max(_EPSILON, (self.support_upper - self.support_lower) / 40.0))
            grid_lower = max(self.support_lower + _EPSILON, center - half_span)
            grid_upper = min(self.support_upper - _EPSILON, center + half_span)
        grid = np.linspace(grid_lower, grid_upper, max(16, points))
        density = self.evaluate(
            grid,
            uniform_floor_weight=uniform_floor_weight,
            min_density=min_density,
        )
        return tuple(float(value) for value in grid), tuple(float(value) for value in density)


def fit_score_density_model(
    values: list[float] | tuple[float, ...] | np.ndarray,
    *,
    method: str | None = None,
    bandwidth_scale: float = 1.0,
) -> ScoreDensityModel:
    normalized_method = normalize_score_density_method(method)
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise ValueError("Os valores para ajuste da densidade devem formar um vetor unidimensional nao vazio.")
    if normalized_method == "bounded_logit_kde":
        transformed, _ = _score_to_logit(array)
    else:
        transformed = _clip_score_array(array)
    if math.isclose(bandwidth_scale, 1.0, rel_tol=1e-12, abs_tol=1e-12):
        kernel = gaussian_kde(transformed)
    else:
        kernel = gaussian_kde(transformed, bw_method=lambda model: model.scotts_factor() * bandwidth_scale)
    return ScoreDensityModel(method=normalized_method, kernel=kernel)


def stabilize_score_density(
    density: float | list[float] | tuple[float, ...] | np.ndarray,
    *,
    uniform_floor_weight: float,
    min_density: float,
    support_lower: float = _SUPPORT_LOWER,
    support_upper: float = _SUPPORT_UPPER,
) -> np.ndarray:
    values = np.asarray(density, dtype=np.float64)
    if values.ndim == 0:
        values = values.reshape(1)
    support_width = max(1e-12, support_upper - support_lower)
    uniform_density = 1.0 / support_width
    mixed = ((1.0 - uniform_floor_weight) * np.maximum(0.0, values)) + (uniform_floor_weight * uniform_density)
    return np.maximum(min_density, mixed)
