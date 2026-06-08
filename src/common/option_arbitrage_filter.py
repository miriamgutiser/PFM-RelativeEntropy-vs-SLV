"""
Filtro comun de arbitraje estatico para superficies finitas de opciones.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np

from common.finite_option_set_arbitrage import (
    DEFAULT_TOLERANCE,
    FiniteOptionSetNAReport,
    finite_option_set_na_report,
)
from common.types import OptionType


@dataclass(frozen=True)
class RemovedOption:
    """
    Opcion descartada por el filtro.
    """

    maturity_label: str
    maturity: float
    strike: float
    price: float
    reason: str


@dataclass(frozen=True)
class FilteredOptionSurface:
    """
    Resultado del filtrado de una superficie de precios.
    """

    filtered_prices: np.ndarray
    valid_mask: np.ndarray
    removed_options: tuple[RemovedOption, ...]
    diagnostics: tuple[FiniteOptionSetNAReport, ...]


def _as_price_surface(prices, n_maturities: int, n_strikes: int) -> np.ndarray:
    surface = np.asarray(prices, dtype=float)
    expected_shape = (n_maturities, n_strikes)
    if surface.shape != expected_shape:
        raise ValueError(
            f"prices debe tener shape {expected_shape}, pero tiene {surface.shape}."
        )
    return surface


def _subset_score(indices: tuple[int, ...], strikes: np.ndarray, forward: float) -> tuple:
    """
    Criterio determinista para desempatar subconjuntos validos.
    """
    selected = strikes[list(indices)]
    span = float(selected[-1] - selected[0]) if selected.size > 1 else 0.0
    center_distance = float(abs(np.mean(selected) - forward))
    gaps = np.diff(selected)
    max_gap = float(np.max(gaps)) if gaps.size else 0.0
    return (len(indices), span, -center_distance, -max_gap, tuple(-int(i) for i in indices))


def largest_no_arbitrage_subset(
    factor: float,
    forward: float,
    strikes,
    prices,
    option_type: OptionType = OptionType.CALL,
    tolerance: float = DEFAULT_TOLERANCE,
    min_points: int = 2,
    min_abs_price: float = 0.0,
) -> tuple[tuple[int, ...], FiniteOptionSetNAReport | None]:
    """
    Busca el subconjunto mas grande de strikes que cumple no arbitraje.
    """
    strikes_array = np.asarray(strikes, dtype=float).reshape(-1)
    prices_array = np.asarray(prices, dtype=float).reshape(-1)
    if strikes_array.size != prices_array.size:
        raise ValueError("strikes y prices deben tener la misma longitud.")

    candidate_indices = tuple(
        int(index)
        for index, price in enumerate(prices_array)
        if np.isfinite(price) and abs(float(price)) > min_abs_price
    )
    if len(candidate_indices) < min_points:
        return tuple(), None

    best_indices: tuple[int, ...] = tuple()
    best_report: FiniteOptionSetNAReport | None = None
    best_score: tuple | None = None

    for subset_size in range(len(candidate_indices), min_points - 1, -1):
        for subset in combinations(candidate_indices, subset_size):
            report = finite_option_set_na_report(
                option_type=option_type,
                factor=factor,
                forward=forward,
                strikes=strikes_array[list(subset)],
                option_prices=prices_array[list(subset)],
                tolerance=tolerance,
            )
            if not report.no_arbitrage:
                continue

            score = _subset_score(subset, strikes_array, forward)
            if best_score is None or score > best_score:
                best_indices = tuple(int(index) for index in subset)
                best_report = report
                best_score = score

        if best_report is not None:
            break

    return best_indices, best_report


def filter_option_price_surface(
    option_type: OptionType,
    S0: float,
    r: float,
    q: float,
    strikes,
    maturities,
    prices,
    maturity_labels: list[str] | tuple[str, ...] | None = None,
    tolerance: float = DEFAULT_TOLERANCE,
    min_points_per_maturity: int = 2,
    min_abs_price: float = 0.0,
) -> FilteredOptionSurface:
    """
    Filtra una superficie de precios vencimiento a vencimiento.

    Devuelve una matriz de precios con NaN en los puntos descartados, una mascara
    booleana con los puntos conservados y diagnosticos por vencimiento.
    """
    strikes_array = np.asarray(strikes, dtype=float).reshape(-1)
    maturities_array = np.asarray(maturities, dtype=float).reshape(-1)
    surface = _as_price_surface(prices, maturities_array.size, strikes_array.size)

    if maturity_labels is None:
        labels = tuple(f"T={float(maturity):.6f}" for maturity in maturities_array)
    else:
        labels = tuple(str(label) for label in maturity_labels)
        if len(labels) != maturities_array.size:
            raise ValueError("maturity_labels y maturities deben tener la misma longitud.")

    filtered_prices = np.array(surface, copy=True, dtype=float)
    valid_mask = np.isfinite(filtered_prices)
    removed_options: list[RemovedOption] = []
    diagnostics: list[FiniteOptionSetNAReport] = []

    if min_abs_price > 0.0:
        degenerate_mask = np.abs(filtered_prices) <= min_abs_price
        valid_mask &= ~degenerate_mask
        for maturity_index, strike_index in zip(*np.where(degenerate_mask)):
            removed_options.append(
                RemovedOption(
                    maturity_label=labels[maturity_index],
                    maturity=float(maturities_array[maturity_index]),
                    strike=float(strikes_array[strike_index]),
                    price=float(surface[maturity_index, strike_index]),
                    reason=f"precio nulo o degenerado (|precio| <= {min_abs_price:g})",
                )
            )

    for maturity_index, maturity in enumerate(maturities_array):
        row_mask = valid_mask[maturity_index]
        row_indices = tuple(int(index) for index in np.where(row_mask)[0])
        if len(row_indices) < min_points_per_maturity:
            continue

        factor = float(np.exp(r * maturity))
        forward = float(S0 * np.exp((r - q) * maturity))
        row_prices = filtered_prices[maturity_index]

        report = finite_option_set_na_report(
            option_type=option_type,
            factor=factor,
            forward=forward,
            strikes=strikes_array[list(row_indices)],
            option_prices=row_prices[list(row_indices)],
            maturity=float(maturity),
            tolerance=tolerance,
        )
        diagnostics.append(report)
        if report.no_arbitrage:
            continue

        kept_indices, kept_report = largest_no_arbitrage_subset(
            factor=factor,
            forward=forward,
            strikes=strikes_array[list(row_indices)],
            prices=row_prices[list(row_indices)],
            option_type=option_type,
            tolerance=tolerance,
            min_points=min_points_per_maturity,
            min_abs_price=min_abs_price,
        )
        kept_global_indices = {row_indices[index] for index in kept_indices}
        if kept_report is not None:
            diagnostics.append(kept_report)

        for strike_index in row_indices:
            if strike_index in kept_global_indices:
                continue
            valid_mask[maturity_index, strike_index] = False
            removed_options.append(
                RemovedOption(
                    maturity_label=labels[maturity_index],
                    maturity=float(maturity),
                    strike=float(strikes_array[strike_index]),
                    price=float(surface[maturity_index, strike_index]),
                    reason=(
                        "; ".join(report.diagnostics)
                        + "; descartada para conservar el subconjunto maximo sin arbitraje"
                    ),
                )
            )

    filtered_prices[~valid_mask] = np.nan
    return FilteredOptionSurface(
        filtered_prices=filtered_prices,
        valid_mask=valid_mask,
        removed_options=tuple(removed_options),
        diagnostics=tuple(diagnostics),
    )


def filter_surface_like_prices(values, valid_mask) -> np.ndarray:
    """
    Aplica una mascara de arbitraje a otra superficie con la misma forma.
    """
    values_array = np.asarray(values, dtype=float)
    mask_array = np.asarray(valid_mask, dtype=bool)
    if values_array.shape != mask_array.shape:
        raise ValueError("values y valid_mask deben tener la misma forma.")

    filtered = np.array(values_array, copy=True, dtype=float)
    filtered[~mask_array] = np.nan
    return filtered
