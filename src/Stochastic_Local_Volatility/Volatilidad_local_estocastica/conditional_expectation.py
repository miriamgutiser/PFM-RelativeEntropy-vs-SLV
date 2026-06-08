"""
Estimadores de la esperanza condicional E[V_t | S_t = K].

El estimador por bins sigue la idea no paramétrica usada en esquemas Monte Carlo
Heston-SLV: ordenar trayectorias por spot, dividirlas en bins y usar la varianza
media de cada bin como estimación de la esperanza condicional.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ConditionalExpectationResult:
    bin_centers: np.ndarray
    bin_values: np.ndarray
    estimates: np.ndarray


def estimate_ev_given_s_binning(
    spots: np.ndarray,
    variances: np.ndarray,
    query_spots: np.ndarray | None = None,
    n_bins: int = 20,
    min_value: float = 1e-12,
) -> ConditionalExpectationResult:
    """
    Estima E[V | S] mediante bins con el mismo número aproximado de caminos.

    Parámetros
    ----------
    spots, variances
        Muestras Monte Carlo en el mismo instante temporal.
    query_spots
        Puntos donde se evalúa la esperanza condicional. Si se omite, el
        estimador se evalúa en ``spots``.
    n_bins
        Número de bins con tamaño similar.
    """
    spots = np.asarray(spots, dtype=float).reshape(-1)
    variances = np.asarray(variances, dtype=float).reshape(-1)
    if spots.shape != variances.shape:
        raise ValueError("spots y variances deben tener la misma forma.")
    if spots.size == 0:
        raise ValueError("Se necesita al menos una trayectoria.")
    if query_spots is None:
        query_spots = spots
    query_spots = np.asarray(query_spots, dtype=float)

    valid = np.isfinite(spots) & np.isfinite(variances)
    if not np.any(valid):
        fallback = np.full(query_spots.shape, min_value, dtype=float)
        return ConditionalExpectationResult(
            bin_centers=np.array([0.0]),
            bin_values=np.array([min_value]),
            estimates=fallback,
        )

    spots_valid = spots[valid]
    variances_valid = np.maximum(variances[valid], min_value)
    order = np.argsort(spots_valid)
    spots_sorted = spots_valid[order]
    variances_sorted = variances_valid[order]

    n_bins = max(1, min(int(n_bins), spots_sorted.size))
    index_bins = np.array_split(np.arange(spots_sorted.size), n_bins)

    centers = np.empty(len(index_bins), dtype=float)
    values = np.empty(len(index_bins), dtype=float)
    for idx, bin_indices in enumerate(index_bins):
        centers[idx] = float(np.mean(spots_sorted[bin_indices]))
        values[idx] = float(np.mean(variances_sorted[bin_indices]))

    # np.interp requiere valores de x crecientes. Colapsamos centros repetidos.
    unique_centers, inverse = np.unique(centers, return_inverse=True)
    unique_values = np.zeros_like(unique_centers)
    counts = np.zeros_like(unique_centers)
    for idx, group_idx in enumerate(inverse):
        unique_values[group_idx] += values[idx]
        counts[group_idx] += 1.0
    unique_values = unique_values / np.maximum(counts, 1.0)

    estimates = np.interp(
        query_spots.reshape(-1),
        unique_centers,
        unique_values,
        left=unique_values[0],
        right=unique_values[-1],
    ).reshape(query_spots.shape)
    estimates = np.maximum(estimates, min_value)

    return ConditionalExpectationResult(
        bin_centers=unique_centers,
        bin_values=unique_values,
        estimates=estimates,
    )


def estimate_ev_for_paths(
    spots: np.ndarray,
    variances: np.ndarray,
    n_bins: int = 20,
    min_value: float = 1e-12,
) -> np.ndarray:
    return estimate_ev_given_s_binning(
        spots=spots,
        variances=variances,
        query_spots=spots,
        n_bins=n_bins,
        min_value=min_value,
    ).estimates


def estimate_ev_given_s_regression(
    spots: np.ndarray,
    variances: np.ndarray,
    query_spots: np.ndarray | None = None,
    degree: int = 2,
    min_value: float = 1e-12,
) -> np.ndarray:
    """Alternativa sencilla por regresión polinómica para diagnósticos."""
    spots = np.asarray(spots, dtype=float).reshape(-1)
    variances = np.asarray(variances, dtype=float).reshape(-1)
    if query_spots is None:
        query_spots = spots
    query_spots = np.asarray(query_spots, dtype=float)

    valid = np.isfinite(spots) & np.isfinite(variances)
    if np.sum(valid) <= degree:
        return np.full(query_spots.shape, max(float(np.nanmean(variances)), min_value), dtype=float)

    coefficients = np.polyfit(spots[valid], np.maximum(variances[valid], min_value), degree)
    estimates = np.polyval(coefficients, query_spots)
    return np.maximum(estimates, min_value)
