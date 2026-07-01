"""
Superficie de volatilidad local de Dupire para el flujo Heston-SLV.

Dada una superficie de precios call de mercado, la varianza local se aproxima por

    sigma_LV(t, K)^2 = (dC/dt + r K dC/dK) / (0.5 K^2 d2C/dK2).

La superficie resultante se combina después con E[V_t | S_t = K] para construir
la función de apalancamiento de Heston-SLV.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator, RegularGridInterpolator

from common.types import OptionType
from Stochastic_Local_Volatility.Volatilidad_local_estocastica.heston_calibration import bs_price_surface
from Stochastic_Local_Volatility.Volatilidad_local_estocastica.market_surface import (
    AVAILABLE_STRIKES,
    MarketSurface,
    load_market_surface,
)


@dataclass
class LocalVolSurface:
    market_surface: MarketSurface
    option_type: OptionType
    price_surface: np.ndarray
    local_variance_grid: np.ndarray

    def __post_init__(self) -> None:
        self._interpolator = RegularGridInterpolator(
            (self.market_surface.maturities, self.market_surface.strikes),
            self.local_variance_grid,
            bounds_error=False,
            fill_value=None,
        )

    @property
    def local_vol_grid(self) -> np.ndarray:
        return np.sqrt(np.maximum(self.local_variance_grid, 0.0))

    def variance(self, t: float | np.ndarray, K: float | np.ndarray) -> float | np.ndarray:
        t_arr, k_arr = np.broadcast_arrays(np.asarray(t, dtype=float), np.asarray(K, dtype=float))
        t_clip = np.clip(t_arr, self.market_surface.maturities[0], self.market_surface.maturities[-1])
        k_clip = np.clip(k_arr, self.market_surface.strikes[0], self.market_surface.strikes[-1])
        points = np.column_stack([t_clip.ravel(), k_clip.ravel()])
        values = np.asarray(self._interpolator(points), dtype=float).reshape(t_arr.shape)
        values = np.maximum(values, 0.0)
        if values.shape == ():
            return float(values)
        return values

    def sigma(self, t: float | np.ndarray, K: float | np.ndarray) -> float | np.ndarray:
        return np.sqrt(self.variance(t, K))


def _gradient(values: np.ndarray, grid: np.ndarray, axis: int) -> np.ndarray:
    edge_order = 2 if len(grid) >= 3 else 1
    return np.gradient(values, grid, axis=axis, edge_order=edge_order)


def _densify_strike_grid(market_surface: MarketSurface, strike_grid_step: float | None) -> MarketSurface:
    """
    Interpola la superficie de IV en una malla de strikes mas fina.

    Dupire usa derivadas primera y segunda respecto a K. Con la malla original
    de strikes separados 20 unidades, esas derivadas son muy ruidosas cerca de
    ATM. La interpolacion PCHIP conserva la forma del smile sin introducir
    oscilaciones fuertes como podria hacer una spline cubica global.
    """
    if strike_grid_step is None or strike_grid_step <= 0.0:
        return market_surface

    min_strike = float(np.min(market_surface.strikes))
    max_strike = float(np.max(market_surface.strikes))
    dense_strikes = np.arange(min_strike, max_strike + 0.5 * strike_grid_step, strike_grid_step, dtype=float)
    dense_strikes = np.unique(np.concatenate([dense_strikes, market_surface.strikes]))
    dense_strikes = dense_strikes[(dense_strikes >= min_strike) & (dense_strikes <= max_strike)]
    dense_strikes.sort()

    if dense_strikes.size <= market_surface.strikes.size:
        return market_surface

    dense_vols = np.empty((len(market_surface.maturities), len(dense_strikes)), dtype=float)
    for i_t in range(len(market_surface.maturities)):
        interpolator = PchipInterpolator(
            market_surface.strikes,
            market_surface.implied_vols[i_t, :],
            extrapolate=False,
        )
        dense_vols[i_t, :] = interpolator(dense_strikes)

    return MarketSurface(
        S0=market_surface.S0,
        r=market_surface.r,
        strikes=dense_strikes,
        maturities=market_surface.maturities,
        maturity_labels=market_surface.maturity_labels,
        implied_vols=dense_vols,
    )


def build_local_vol_surface(
    market_surface: MarketSurface | None = None,
    option_type: OptionType = OptionType.CALL,
    min_variance: float = 1e-8,
    max_variance: float = 25.0,
    strike_grid_step: float | None = 2.5,
) -> LocalVolSurface:
    """
    Construye una superficie de volatilidad local de Dupire desde la superficie
    de volatilidad implícita de mercado.

    El cálculo usa diferencias finitas y está pensado como implementación de
    investigación/docencia. En producción, la calibración SLV suele partir de
    una superficie de precios call suavizada y libre de arbitraje antes de
    aplicar Dupire.
    """
    if market_surface is None:
        market_surface = load_market_surface()
    market_surface = _densify_strike_grid(market_surface, strike_grid_step)

    prices = bs_price_surface(
        option_type,
        market_surface.S0,
        market_surface.strikes,
        market_surface.maturities,
        market_surface.r,
        market_surface.implied_vols,
    )

    dC_dT = _gradient(prices, market_surface.maturities, axis=0)
    dC_dK = _gradient(prices, market_surface.strikes, axis=1)
    d2C_dK2 = _gradient(dC_dK, market_surface.strikes, axis=1)

    K = market_surface.strikes.reshape(1, -1)
    numerator = dC_dT + market_surface.r * K * dC_dK
    denominator = 0.5 * K**2 * d2C_dK2

    with np.errstate(divide="ignore", invalid="ignore"):
        local_variance = numerator / denominator

    local_variance = np.real(local_variance)
    local_variance = np.where(np.isfinite(local_variance), local_variance, np.nan)
    fallback = np.nanmedian(market_surface.implied_vols**2)
    if not np.isfinite(fallback) or fallback <= 0.0:
        fallback = 0.04
    local_variance = np.nan_to_num(local_variance, nan=fallback, posinf=max_variance, neginf=min_variance)
    local_variance = np.clip(local_variance, min_variance, max_variance)

    return LocalVolSurface(
        market_surface=market_surface,
        option_type=option_type,
        price_surface=prices,
        local_variance_grid=local_variance,
    )


def dupire_sigma(t: float | np.ndarray, K: float | np.ndarray, surface: LocalVolSurface | None = None):
    if surface is None:
        surface = build_local_vol_surface()
    return surface.sigma(t, K)


def main() -> None:
    market_surface = load_market_surface(
        selected_strikes=tuple(float(strike) for strike in AVAILABLE_STRIKES)
    )
    local_vol_surface = build_local_vol_surface(
        market_surface=market_surface,
        option_type=OptionType.CALL,
    )

    local_variance_table = pd.DataFrame(
        local_vol_surface.local_variance_grid.T,
        index=[float(strike) for strike in market_surface.strikes],
        columns=list(market_surface.maturity_labels),
    )
    local_variance_table.index.name = "Strike"

    print("Matriz de varianzas locales sigma_LV^2(T,K)")
    print(local_variance_table.to_string(float_format=lambda value: f"{value:.8f}"))


if __name__ == "__main__":
    main()
