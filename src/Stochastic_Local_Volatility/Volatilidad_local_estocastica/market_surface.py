"""
Superficie de volatilidad implícita de mercado para los ejemplos Heston-SLV.

Los valores se guardan en porcentaje en la tabla original y se exponen en
formato decimal con forma [n_vencimientos, n_strikes].
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


AVAILABLE_STRIKES = np.array(
    [40.0, 60.0, 80.0, 100.0, 120.0, 140.0, 160.0, 180.0],
    dtype=float,
)
DEFAULT_STRIKES = np.array([40.0, 60.0, 80.0, 100.0, 120.0, 140.0], dtype=float)

MATURITY_LABELS = ["1w", "1m", "6m", "1y", "2y", "5y", "8y", "10y"]
MATURITIES = np.array([1.0 / 52.0, 1.0 / 12.0, 0.5, 1.0, 2.0, 5.0, 8.0, 10.0], dtype=float)

# Filas: strikes. Columnas: vencimientos. Valores: volatilidades implícitas en porcentaje.
IMPLIED_VOL_PERCENT = np.array(
    [
        [22.622, 22.371, 20.706, 18.537, 18.041, 16.395, 14.491, 13.136],
        [15.491, 15.350, 14.415, 13.195, 12.916, 11.988, 10.917, 10.177],
        [9.930, 9.882, 9.562, 9.154, 9.070, 8.786, 8.459, 8.251],
        [5.001, 5.070, 5.515, 6.041, 6.196, 6.639, 7.015, 7.221],
        [4.542, 4.598, 4.962, 5.430, 5.654, 6.277, 6.782, 7.041],
        [6.278, 6.300, 6.405, 6.460, 6.607, 6.982, 7.232, 7.328],
        [7.748, 7.752, 7.707, 7.517, 7.615, 7.806, 7.824, 7.751],
        [8.988, 8.976, 8.818, 8.445, 8.508, 8.563, 8.399, 8.182],
    ],
    dtype=float,
)


@dataclass(frozen=True)
class MarketSurface:
    """Contenedor de la superficie de mercado usada por Heston y Heston-SLV."""

    S0: float
    r: float
    strikes: np.ndarray
    maturities: np.ndarray
    maturity_labels: tuple[str, ...]
    implied_vols: np.ndarray

    @property
    def shape(self) -> tuple[int, int]:
        return self.implied_vols.shape


def _indices_for_values(values: list[float] | tuple[float, ...] | np.ndarray, available: np.ndarray) -> list[int]:
    indices: list[int] = []
    for value in values:
        matches = np.where(np.isclose(available, float(value)))[0]
        if len(matches) == 0:
            raise ValueError(f"El valor {value} no está disponible en la superficie de mercado.")
        indices.append(int(matches[0]))
    return indices


def select_strike_indices(selected_strikes: list[float] | tuple[float, ...] | np.ndarray) -> list[int]:
    return _indices_for_values(selected_strikes, AVAILABLE_STRIKES)


def select_maturity_indices(selected_labels: list[str] | tuple[str, ...]) -> list[int]:
    indices: list[int] = []
    for label in selected_labels:
        if label not in MATURITY_LABELS:
            raise ValueError(f"El vencimiento {label!r} no está disponible. Usa uno de {MATURITY_LABELS}.")
        indices.append(MATURITY_LABELS.index(label))
    return indices


def load_market_surface(
    selected_strikes: list[float] | tuple[float, ...] | np.ndarray | None = None,
    selected_maturity_labels: list[str] | tuple[str, ...] | None = None,
    S0: float = 100.0,
    r: float = 0.0,
) -> MarketSurface:
    """
    Carga la superficie de volatilidad implícita de mercado.

    Devuelve
    --------
    MarketSurface
        Superficie con volatilidades implícitas en decimal y forma [T, K].
    """
    if selected_strikes is None:
        selected_strikes = DEFAULT_STRIKES
    if selected_maturity_labels is None:
        selected_maturity_labels = tuple(MATURITY_LABELS)

    strike_idx = select_strike_indices(selected_strikes)
    maturity_idx = select_maturity_indices(tuple(selected_maturity_labels))

    vols = IMPLIED_VOL_PERCENT[np.ix_(strike_idx, maturity_idx)].T / 100.0
    maturities = MATURITIES[maturity_idx]
    strikes = np.asarray(selected_strikes, dtype=float)
    labels = tuple(MATURITY_LABELS[i] for i in maturity_idx)

    return MarketSurface(
        S0=float(S0),
        r=float(r),
        strikes=strikes,
        maturities=maturities,
        maturity_labels=labels,
        implied_vols=vols,
    )
