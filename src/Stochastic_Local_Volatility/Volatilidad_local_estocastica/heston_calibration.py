"""
Utilidades de calibración para el modelo de Heston puro.

Este módulo calibra deliberadamente solo los parámetros de volatilidad
estocástica de Heston: ``kappa, theta, gamma, rho, v0``. 

La componente de apalancamiento/volatilidad local del SLV se trata después en
``dupire_local_vol.py`` y ``slv_heston_mc.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import brentq, minimize

from common.types import OptionType
from Stochastic_Local_Volatility.core.COS_method import CallPutOptionPriceCOSMthd
from Stochastic_Local_Volatility.core.pricing_Black_Scholes import BS_Call_Put_Option_Price
from Stochastic_Local_Volatility.volatilidad_estocastica.Heston_model import chf_heston
from Stochastic_Local_Volatility.Volatilidad_local_estocastica.market_surface import (
    MarketSurface,
    load_market_surface,
)


@dataclass(frozen=True)
class HestonParams:
    """Parámetros de Heston bajo la medida neutral al riesgo."""

    kappa: float
    theta: float
    gamma: float
    rho: float
    v0: float

    @property
    def sigma(self) -> float:
        """Alias compatible con código antiguo para la volatilidad de la varianza."""
        return self.gamma

    @property
    def feller_value(self) -> float:
        return 2.0 * self.kappa * self.theta - self.gamma**2


@dataclass(frozen=True)
class CalibrationConfig:
    selected_strikes: tuple[float, ...] | None = None
    selected_maturity_labels: tuple[str, ...] | None = None
    option_type: OptionType = OptionType.CALL
    calibration_target: str = "prices"
    cos_n: int = 5000
    cos_l: float = 5.0
    enforce_feller: bool = True


@dataclass(frozen=True)
class HestonCalibrationResult:
    params: HestonParams
    optimizer_result: Any
    market_surface: MarketSurface
    market_prices: np.ndarray
    model_prices: np.ndarray
    model_implied_vols: np.ndarray


def option_type_label(cp: OptionType) -> str:
    return "Call" if cp == OptionType.CALL else "Put"


def price_option_bs(CP: OptionType, S0: float, K: float, T: float, r: float, sigma: float) -> float:
    price = BS_Call_Put_Option_Price(CP, S0, K, sigma, 0.0, T, r)
    return float(np.atleast_1d(price).flatten()[0])


def price_option_heston_cos(
    CP: OptionType,
    S0: float,
    K: float,
    T: float,
    r: float,
    params: HestonParams,
    N: int = 5000,
    L: float = 5.0,
) -> float:
    cf = lambda u: chf_heston(
        u,
        T,
        params.kappa,
        params.theta,
        params.gamma,
        params.rho,
        params.v0,
        r,
    )

    def direct_price(cp: OptionType) -> float:
        price = CallPutOptionPriceCOSMthd(cf, cp, S0, r, T, np.array([K]), N, L)
        return float(np.real(np.atleast_1d(price).flatten()[0]))

    discounted_strike = K * np.exp(-r * T)
    if CP == OptionType.CALL and K < S0:
        price = direct_price(OptionType.PUT) + S0 - discounted_strike
    elif CP == OptionType.PUT and K > S0:
        price = direct_price(OptionType.CALL) - S0 + discounted_strike
    else:
        price = direct_price(CP)
    return max(float(price), 0.0)


def implied_vol_silent(CP: OptionType, price: float, S0: float, K: float, T: float, r: float) -> float:
    if T <= 0.0 or price <= 0.0:
        return np.nan

    if CP == OptionType.CALL:
        intrinsic = max(S0 - K * np.exp(-r * T), 0.0)
        upper_bound = S0
    else:
        intrinsic = max(K * np.exp(-r * T) - S0, 0.0)
        upper_bound = K * np.exp(-r * T)

    safe_price = float(np.clip(price, intrinsic + 1e-12, upper_bound - 1e-12))

    def objective(sigma: float) -> float:
        return price_option_bs(CP, S0, K, T, r, sigma) - safe_price

    try:
        return float(brentq(objective, 1e-6, 10.0, maxiter=300))
    except (ValueError, RuntimeError):
        return np.nan


def bs_price_surface(CP: OptionType, S0: float, K: np.ndarray, T: np.ndarray, r: float, sigma: np.ndarray) -> np.ndarray:
    out = np.zeros((len(T), len(K)), dtype=float)
    for i_t, maturity in enumerate(T):
        for i_k, strike in enumerate(K):
            out[i_t, i_k] = price_option_bs(CP, S0, float(strike), float(maturity), r, float(sigma[i_t, i_k]))
    return out


def heston_price_surface(
    CP: OptionType,
    S0: float,
    K: np.ndarray,
    T: np.ndarray,
    r: float,
    params: HestonParams,
    N: int = 5000,
    L: float = 5.0,
) -> np.ndarray:
    out = np.zeros((len(T), len(K)), dtype=float)
    for i_t, maturity in enumerate(T):
        for i_k, strike in enumerate(K):
            out[i_t, i_k] = price_option_heston_cos(CP, S0, float(strike), float(maturity), r, params, N=N, L=L)
    return out


def heston_surface_iv(
    CP: OptionType,
    S0: float,
    K: np.ndarray,
    T: np.ndarray,
    r: float,
    params: HestonParams,
    N: int = 5000,
    L: float = 5.0,
) -> np.ndarray:
    prices = heston_price_surface(CP, S0, K, T, r, params, N=N, L=L)
    out = np.zeros_like(prices)
    for i_t, maturity in enumerate(T):
        for i_k, strike in enumerate(K):
            out[i_t, i_k] = implied_vol_silent(CP, prices[i_t, i_k], S0, float(strike), float(maturity), r)
    return out


def unpack_params(x: np.ndarray) -> HestonParams:
    kappa, theta, gamma, rho, v0 = x
    return HestonParams(float(kappa), float(theta), float(gamma), float(rho), float(v0))


def weighted_rmse(err: np.ndarray, weights: np.ndarray | None = None) -> float:
    err = np.asarray(err, dtype=float)
    valid = np.isfinite(err)
    if not np.any(valid):
        return 1e6
    if weights is None:
        weights = np.ones_like(err)
    weights = np.asarray(weights, dtype=float)
    w = weights[valid]
    e = err[valid]
    weight_sum = float(np.sum(w))
    if weight_sum <= 0.0:
        return 1e6
    return float(np.sqrt(np.sum(w * e * e) / weight_sum))


def initial_guess_from_market(surface: MarketSurface) -> tuple[np.ndarray, list[tuple[float, float]]]:
    idx_k = int(np.argmin(np.abs(surface.strikes - surface.S0)))
    idx_t_short = int(np.argmin(np.abs(surface.maturities - 1.0 / 52.0)))
    idx_t_long = int(np.argmin(np.abs(surface.maturities - 10.0)))

    v0_guess = float(surface.implied_vols[idx_t_short, idx_k] ** 2)
    theta_guess = float(surface.implied_vols[idx_t_long, idx_k] ** 2)
    x0 = np.array([2.0, theta_guess, 0.3, -0.7, v0_guess], dtype=float)
    bounds = [
        (1e-4, 10.0),
        (1e-6, 1.0),
        (1e-4, 5.0),
        (-0.999, 0.999),
        (1e-6, 1.0),
    ]
    return x0, bounds


def calibrate_heston(
    CP: OptionType,
    S0: float,
    K: np.ndarray,
    T: np.ndarray,
    r: float,
    sigma: np.ndarray,
    calibration_target: str = "prices",
    weights: np.ndarray | None = None,
    cos_n: int = 5000,
    cos_l: float = 5.0,
    enforce_feller: bool = True,
):
    surface = MarketSurface(S0=S0, r=r, strikes=K, maturities=T, maturity_labels=tuple(str(x) for x in T), implied_vols=sigma)
    x0, bounds = initial_guess_from_market(surface)
    prices_market = bs_price_surface(CP, S0, K, T, r, sigma)

    def objective_prices(x: np.ndarray) -> float:
        params = unpack_params(x)
        prices_model = heston_price_surface(CP, S0, K, T, r, params, N=cos_n, L=cos_l)
        return weighted_rmse(prices_model - prices_market, weights=weights)

    def objective_iv(x: np.ndarray) -> float:
        params = unpack_params(x)
        iv_model = heston_surface_iv(CP, S0, K, T, r, params, N=cos_n, L=cos_l)
        return weighted_rmse(iv_model - sigma, weights=weights)

    if calibration_target == "prices":
        objective = objective_prices
    elif calibration_target == "iv":
        objective = objective_iv
    else:
        raise ValueError("calibration_target debe ser 'prices' o 'iv'.")

    constraints = []
    if enforce_feller:
        constraints.append({"type": "ineq", "fun": lambda x: 2.0 * x[0] * x[1] - x[2] ** 2})

    result = minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 500, "ftol": 1e-9},
    )
    return unpack_params(result.x), result, prices_market


def calibrate_heston_from_market(
    market_surface: MarketSurface | None = None,
    config: CalibrationConfig | None = None,
    weights: np.ndarray | None = None,
) -> HestonCalibrationResult:
    if config is None:
        config = CalibrationConfig()
    if market_surface is None:
        market_surface = load_market_surface(
            selected_strikes=config.selected_strikes,
            selected_maturity_labels=config.selected_maturity_labels,
        )

    params, optimizer_result, market_prices = calibrate_heston(
        CP=config.option_type,
        S0=market_surface.S0,
        K=market_surface.strikes,
        T=market_surface.maturities,
        r=market_surface.r,
        sigma=market_surface.implied_vols,
        calibration_target=config.calibration_target,
        weights=weights,
        cos_n=config.cos_n,
        cos_l=config.cos_l,
        enforce_feller=config.enforce_feller,
    )
    model_prices = heston_price_surface(
        config.option_type,
        market_surface.S0,
        market_surface.strikes,
        market_surface.maturities,
        market_surface.r,
        params,
        N=config.cos_n,
        L=config.cos_l,
    )
    model_ivs = heston_surface_iv(
        config.option_type,
        market_surface.S0,
        market_surface.strikes,
        market_surface.maturities,
        market_surface.r,
        params,
        N=config.cos_n,
        L=config.cos_l,
    )
    return HestonCalibrationResult(
        params=params,
        optimizer_result=optimizer_result,
        market_surface=market_surface,
        market_prices=market_prices,
        model_prices=model_prices,
        model_implied_vols=model_ivs,
    )


def main() -> None:
    config = CalibrationConfig()
    result = calibrate_heston_from_market(config=config)
    params = result.params

    print("Calibración Heston")
    print(f"option_type = {option_type_label(config.option_type)}")
    print(f"target      = {config.calibration_target}")
    print(f"kappa       = {params.kappa:.7f}")
    print(f"theta       = {params.theta:.7f}")
    print(f"gamma       = {params.gamma:.7f}")
    print(f"rho         = {params.rho:.7f}")
    print(f"v0          = {params.v0:.7f}")
    print(f"objective   = {result.optimizer_result.fun:.7f}")
    print(f"feller      = {params.feller_value:.7f}")


if __name__ == "__main__":
    main()
