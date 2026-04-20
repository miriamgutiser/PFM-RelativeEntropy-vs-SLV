"""
==========================================================
Heston_Calibration_NoPlots.py
==========================================================

Calibración del modelo de Heston a una superficie de volatilidad
implícita de mercado.
El proceso de calibración sigue estos pasos:
1. Fijamos manualmente los strikes que se quieren usar (valores absolutos),
2. Convertimos las volatilidades implícitas de mercado a precios Black-Scholes,
3. Calibramos Heston sobre esa malla (5 parámetros libres, condición de Feller impuesta),
4. Mostramos la tabla final de precios y errores.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq, minimize

# ---- project imports ----
from common.types import OptionType
from Stochastic_Local_Volatility.core.pricing_Black_Scholes import BS_Call_Put_Option_Price
from Stochastic_Local_Volatility.core.COS_method import CallPutOptionPriceCOSMthd
from Stochastic_Local_Volatility.volatilidad_estocastica.Heston_model import chf_heston


# ---------------------------------------------------------------------------
# Datos de mercado
# ---------------------------------------------------------------------------
AVAILABLE_K = np.array([40.0, 60.0, 80.0, 100.0, 120.0, 140.0, 160.0, 180.0], dtype=float)
K_STRIKES   = np.array([40.0, 60.0, 80.0, 100.0, 120.0, 140.0], dtype=float)
FULL_STRIKES = K_STRIKES

MATURITY_LABELS = ["1w", "1m", "6m", "1y", "2y", "5y", "8y", "10y"]
T_GRID = np.array([1 / 52, 1 / 12, 0.5, 1.0, 2.0, 5.0, 8.0, 10.0], dtype=float)
MATURITIES = T_GRID

# Matriz de volatilidades implícitas de mercado [strikes × vencimientos], en %.
# Filas: strikes crecientes. Columnas: vencimientos crecientes.
SIGMA_AVAILABLE = np.array(
    [
        [22.622, 22.371, 20.706, 18.537, 18.041, 16.395, 14.491, 13.136],  # K=40
        [15.491, 15.350, 14.415, 13.195, 12.916, 11.988, 10.917, 10.177],  # K=60
        [9.930,  9.882,  9.562,  9.154,  9.070,  8.786,  8.459,  8.251],   # K=80
        [5.001,  5.070,  5.515,  6.041,  6.196,  6.639,  7.015,  7.221],   # K=100
        [4.542,  4.598,  4.962,  5.430,  5.654,  6.277,  6.782,  7.041],   # K=120
        [6.278,  6.300,  6.405,  6.460,  6.607,  6.982,  7.232,  7.328],   # K=140
        [7.748,  7.752,  7.707,  7.517,  7.615,  7.806,  7.824,  7.751],   # K=160
        [8.988,  8.976,  8.818,  8.445,  8.508,  8.563,  8.399,  8.182],   # K=180
    ],
    dtype=float,
)

# Pre-indexado para la submalla por defecto
_K_IDX: list[int] = []
for _strike in K_STRIKES:
    _matches = np.where(np.isclose(AVAILABLE_K, _strike))[0]
    if len(_matches) == 0:
        raise ValueError(f"El strike {_strike} no está disponible en la tabla de mercado.")
    _K_IDX.append(int(_matches[0]))

SIGMA_MARKET = SIGMA_AVAILABLE[_K_IDX, :]
VOL_FULL     = SIGMA_MARKET

DISPLAY_DECIMALS = 7


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class HestonParams:
    """
    Parámetros del modelo de Heston.
    """
    kappa: float
    theta: float
    sigma: float   # vol de vol (σ_v)
    rho: float
    v0: float


@dataclass
class CalibrationConfig:
    """
    Configuración principal del script.

    `selected_strikes` permite elegir la submalla de strikes (valores absolutos).
    `CP` define si toda la malla se trata como calls o como puts.
    """
    selected_strikes: tuple[float, ...] = tuple(float(x) for x in K_STRIKES)
    CP: OptionType = OptionType.CALL
    calibration_target: str = "prices"
    export_csv: bool = True
    export_filename: str = "heston_calibration_option_values.csv"


# ---------------------------------------------------------------------------
# Utilidades de presentación
# ---------------------------------------------------------------------------

def option_type_label(cp: OptionType) -> str:
    """
    Convierte un OptionType a su representación de texto para la tabla de resultados.
    Args:
        cp (OptionType): Tipo de opción (CALL o PUT).

    Returns:
        str: cadena "Call" o "Put" según el tipo de opción.
    """
    return "Call" if cp == OptionType.CALL else "Put"


def sanitize_display_value(x: float, decimals: int = DISPLAY_DECIMALS) -> float:
    """
    Redondea un valor numérico para presentación.
    """
    if not np.isfinite(x):
        return x
    return round(float(x), decimals)


def display_float_format(x: float) -> str:
    """
    Formatea flotantes en notación decimal fija para evitar notación científica.
    """
    if not np.isfinite(x):
        return str(x)
    return f"{float(x):.{DISPLAY_DECIMALS}f}"


def print_dataframe_fixed(df: pd.DataFrame, index: bool = True) -> None:
    """
    Imprime un DataFrame con flotantes en formato fijo.
    """
    print(df.to_string(index=index, float_format=display_float_format))


# ---------------------------------------------------------------------------
# Carga de superficie de mercado
# ---------------------------------------------------------------------------

def select_strike_indices(selected_strikes: tuple[float, ...] | list[float] | np.ndarray) -> list[int]:
    """
    Devuelve los índices de los strikes seleccionados dentro de la tabla completa.
    Los strikes se interpretan como valores absolutos.
    """
    idx = []
    for strike in selected_strikes:
        matches = np.where(np.isclose(AVAILABLE_K, strike))[0]
        if len(matches) == 0:
            raise ValueError(f"El strike {strike} no está disponible en la tabla de mercado.")
        idx.append(int(matches[0]))
    return idx


def load_market_surface(
    selected_strikes: tuple[float, ...] | list[float] | np.ndarray | None = None,
):
    """
    Carga la malla de mercado para la subselección de strikes indicada.

    Returns:
        tuple: (S0, K, T, maturity_labels, sigma)
            - S0           : precio del subyacente
            - K            : array de strikes (valores absolutos)
            - T            : array de vencimientos en años
            - maturity_labels: etiquetas de vencimientos
            - sigma        : superficie de IVs en decimal (forma: [T, K])
    """
    if selected_strikes is None:
        selected_strikes = tuple(float(x) for x in K_STRIKES)

    idx = select_strike_indices(selected_strikes)
    S0 = 100.0
    K = np.asarray(selected_strikes, dtype=float)
    sigma = SIGMA_AVAILABLE[idx, :].T / 100.0     # [T × K], en decimal
    return S0, K, T_GRID.copy(), MATURITY_LABELS.copy(), sigma


# ---------------------------------------------------------------------------
# Pricing: Black-Scholes y Heston-COS
# ---------------------------------------------------------------------------

def price_option_bs(
    CP: OptionType, S0: float, K: float, T: float, r: float, sigma: float
) -> float:
    """
    Calcula el precio de una opción europea con la fórmula de Black-Scholes.
    Args:
        CP (OptionType): Tipo de opción (CALL o PUT).
        S0 (float): Precio del activo subyacente.
        K (float): Strike de la opción.
        T (float): Tiempo hasta el vencimiento.
        r (float): Tasa de interés sin riesgo.
        sigma (float): Volatilidad implícita o asumida.

    Returns:
        float: Precio de la opción bajo Black-Scholes.
    """
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
    """
    Calcula el precio de una opción europea bajo el modelo de Heston usando el método COS.
    Args:
        CP (OptionType): Tipo de opción (CALL o PUT).
        S0 (float): Precio del activo subyacente.
        K (float): Strike de la opción.
        T (float): Tiempo hasta el vencimiento.
        r (float): Tasa de interés sin riesgo.
        params (HestonParams): Parámetros del modelo de Heston.
        N (int, optional): Número de términos en la serie COS. Defaults to 5000.
        L (float, optional): Rango de integración para el método COS. Defaults to 5.0.

    Returns:
        float: Precio de la opción bajo el modelo de Heston.
    """
    cf = lambda u: chf_heston(
        u, T, params.kappa, params.theta, params.sigma, params.rho, params.v0, r
    )

    def price_direct(cp: OptionType) -> float:
        price = CallPutOptionPriceCOSMthd(cf, cp, S0, r, T, np.array([K]), N, L)
        return float(np.real(np.atleast_1d(price).flatten()[0]))

    discount_strike = K * np.exp(-r * T)
    if CP == OptionType.CALL and K < S0:
        put_price = price_direct(OptionType.PUT)
        price_value = put_price + S0 - discount_strike
    elif CP == OptionType.PUT and K > S0:
        call_price = price_direct(OptionType.CALL)
        price_value = call_price - S0 + discount_strike
    else:
        price_value = price_direct(CP)

    return max(price_value, 0.0)


def implied_vol_silent(
    CP: OptionType, price: float, S0: float, K: float, T: float, r: float
) -> float:
    """
    Calcula la volatilidad implícita usando el método de Brent.
    Devuelve NaN en caso de error en lugar de lanzar una excepción.
    Args:
        CP (OptionType): Tipo de opción (CALL o PUT).
        price (float): Precio de la opción.
        S0 (float): Precio del activo subyacente.
        K (float): Strike de la opción.
        T (float): Tiempo hasta el vencimiento.
        r (float): Tasa de interés sin riesgo.

    Returns:
        float: Volatilidad implícita. Si el precio es inválido o no se encuentra una raíz, devuelve NaN.
    """
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
        """
        Función objetivo para encontrar la volatilidad implícita.
        Args:
            sigma (float): Volatilidad a evaluar.

        Returns:
            float: Diferencia entre el precio de la opción y el precio observado.
        """
        return price_option_bs(CP, S0, K, T, r, sigma) - safe_price

    try:
        return brentq(objective, 1e-6, 10.0, maxiter=300)
    except (ValueError, RuntimeError):
        return np.nan


def bs_price_surface(
    CP: OptionType,
    S0: float,
    K: np.ndarray,
    T: np.ndarray,
    r: float,
    sigma: np.ndarray,
) -> np.ndarray:
    """
    Calcula la superficie de precios de opciones usando Black-Scholes con las volatilidades de mercado.
    Args:
        CP (OptionType): Tipo de opción (CALL o PUT).
        S0 (float): Precio del activo subyacente.
        K (np.ndarray): Strikes de las opciones.
        T (np.ndarray): Tiempos hasta el vencimiento.
        r (float): Tasa de interés sin riesgo.
        sigma (np.ndarray): Volatilidades de mercado [T × K], en decimal.

    Returns:
        np.ndarray: Superficie de precios de opciones [T × K].
    """
    out = np.zeros((len(T), len(K)))
    for iT, t in enumerate(T):
        for iK, k in enumerate(K):
            out[iT, iK] = price_option_bs(CP, S0, k, t, r, sigma[iT, iK])
    return out


def heston_price_surface(
    CP: OptionType,
    S0: float,
    K: np.ndarray,
    T: np.ndarray,
    r: float,
    params: HestonParams,
) -> np.ndarray:
    """
    Calcula la superficie de precios de opciones usando el modelo de Heston.
    Args:
        CP (OptionType): Tipo de opción (CALL o PUT).
        S0 (float): Precio del activo subyacente.
        K (np.ndarray): Strikes de las opciones.
        T (np.ndarray): Tiempos hasta el vencimiento.
        r (float): Tasa de interés libre de riesgo.
        params (HestonParams): Parámetros del modelo de Heston.

    Returns:
        np.ndarray: Superficie de precios de opciones [T × K].
    """
    out = np.zeros((len(T), len(K)))
    for iT, t in enumerate(T):
        for iK, k in enumerate(K):
            out[iT, iK] = price_option_heston_cos(CP, S0, k, t, r, params)
    return out


def heston_surface_sigma(
    CP: OptionType,
    S0: float,
    K: np.ndarray,
    T: np.ndarray,
    r: float,
    params: HestonParams,
) -> np.ndarray:
    """
    Calcula la superficie de volatilidades implícitas del modelo de Heston.
    Args:
        CP (OptionType): Tipo de opción (CALL o PUT).
        S0 (float): Precio del activo subyacente.
        K (np.ndarray): Strikes de las opciones.
        T (np.ndarray): Tiempos hasta el vencimiento.
        r (float): Tasa de interés sin riesgo.
        params (HestonParams): Parámetros del modelo de Heston.

    Returns:
        np.ndarray: Superficie de volatilidades implícitas [T × K], en decimal.
    """
    out = np.zeros((len(T), len(K)))
    for iT, t in enumerate(T):
        for iK, k in enumerate(K):
            price = price_option_heston_cos(CP, S0, k, t, r, params)
            out[iT, iK] = implied_vol_silent(CP, price, S0, k, t, r)
    return out


heston_surface_iv = heston_surface_sigma


# ---------------------------------------------------------------------------
# Calibración
# ---------------------------------------------------------------------------

def unpack_params(x: np.ndarray) -> HestonParams:
    """
    Convierte un vector de 5 parámetros optimizados a una instancia de HestonParams.

    Args:
        x (np.ndarray): Vector [kappa, theta, sigma_v, rho, v0].

    Returns:
        HestonParams: Instancia con los parámetros convertidos.
    """
    kappa, theta, sigma, rho, v0 = x
    return HestonParams(
        kappa=float(kappa),
        theta=float(theta),
        sigma=float(sigma),
        rho=float(rho),
        v0=float(v0),
    )


def weighted_rmse(err: np.ndarray, weights: np.ndarray | None = None) -> float:
    """
    Calcula la raíz del error cuadrático medio (RMSE) ponderado, ignorando los valores no finitos.
    Args:
        err (np.ndarray): Vector de errores entre el modelo y el mercado.
        weights (np.ndarray | None, optional): Vector de pesos. Defaults to None (pesos uniformes).

    Returns:
        float: El RMSE ponderado.
    """
    valid_mask = np.isfinite(err)
    if not np.any(valid_mask):
        return 1e6

    if weights is None:
        weights = np.ones_like(err)

    w = np.asarray(weights, dtype=float)[valid_mask]
    e = np.asarray(err, dtype=float)[valid_mask]
    weight_sum = np.sum(w)
    if weight_sum <= 0.0:
        return 1e6
    return float(np.sqrt(np.sum(w * e * e) / weight_sum))


def objective_surface_prices(
    x: np.ndarray,
    CP: OptionType,
    S0: float,
    K: np.ndarray,
    T: np.ndarray,
    r: float,
    prices_market: np.ndarray,
    weights: np.ndarray | None = None,
) -> float:
    """
    Función objetivo para la calibración de Heston basada en precios.
    Calcula el RMSE ponderado entre los precios del modelo y los precios de mercado.
    Args:
        x (np.ndarray): Vector de parámetros [kappa, theta, sigma_v, rho, v0].
        CP (OptionType): Tipo de opción (CALL o PUT).
        S0 (float): Precio del activo subyacente.
        K (np.ndarray): Strikes de las opciones.
        T (np.ndarray): Tiempos hasta el vencimiento.
        r (float): Tasa de interés sin riesgo.
        prices_market (np.ndarray): Precios de mercado de las opciones [T × K].
        weights (np.ndarray | None, optional): Vector de pesos. Defaults to None.

    Returns:
        float: El RMSE ponderado entre precios Heston y precios de mercado.
    """
    params = unpack_params(x)
    prices_model = heston_price_surface(CP, S0, K, T, r, params)
    return weighted_rmse(prices_model - prices_market, weights=weights)


def objective_surface_iv(
    x: np.ndarray,
    CP: OptionType,
    S0: float,
    K: np.ndarray,
    T: np.ndarray,
    r: float,
    sigma: np.ndarray,
    weights: np.ndarray | None = None,
) -> float:
    """
    Función objetivo para la calibración de Heston basada en volatilidades implícitas.
    Calcula el RMSE ponderado entre las IVs del modelo y las de mercado.
    Args:
        x (np.ndarray): Vector de parámetros [kappa, theta, sigma_v, rho, v0].
        CP (OptionType): Tipo de opción (CALL o PUT).
        S0 (float): Precio del activo subyacente.
        K (np.ndarray): Strikes de las opciones.
        T (np.ndarray): Tiempos hasta el vencimiento.
        r (float): Tasa de interés sin riesgo.
        sigma (np.ndarray): Volatilidades implícitas de mercado [T × K], en decimal.
        weights (np.ndarray | None, optional): Vector de pesos. Defaults to None.

    Returns:
        float: El RMSE ponderado entre IVs Heston y IVs de mercado.
    """
    params = unpack_params(x)
    sigma_model = heston_surface_sigma(CP, S0, K, T, r, params)
    return weighted_rmse(sigma_model - sigma, weights=weights)


def initial_guess_from_book(
    S0: float,
    K: np.ndarray,
    T: np.ndarray,
    sigma: np.ndarray,
) -> tuple[np.ndarray, list[tuple[float, float]]]:
    """
    Construye una conjetura inicial y los límites de optimización para calibrar los
    5 parámetros de Heston (kappa, theta, sigma_v, rho, v0).

    La conjetura se basa en:
    - v0   : varianza implícita ATM a corto plazo (1 semana).
    - theta : varianza implícita ATM a largo plazo (10 años).
    - kappa : velocidad de reversión a la media, valor inicial 2.0.
    - sigma_v: vol de vol, valor inicial 0.3.
    - rho  : correlación spot-vol, valor inicial -0.7 (típico en equity).

    Args:
        S0 (float): Precio del activo subyacente.
        K (np.ndarray): Strikes de las opciones (valores absolutos).
        T (np.ndarray): Tiempos hasta el vencimiento.
        sigma (np.ndarray): Superficie de IVs de mercado [T × K], en decimal.

    Returns:
        tuple: (x0, bounds) — vector inicial y límites de optimización.
    """
    idx_K    = int(np.argmin(np.abs(K - S0)))
    idx_T_1w = int(np.argmin(np.abs(T - 1 / 52)))
    idx_T_10y= int(np.argmin(np.abs(T - 10.0)))

    v0_guess    = sigma[idx_T_1w, idx_K] ** 2
    theta_guess = sigma[idx_T_10y, idx_K] ** 2
    kappa_guess = 2.0
    sigma_guess = 0.3
    rho_guess   = -0.7

    x0 = np.array([kappa_guess, theta_guess, sigma_guess, rho_guess, v0_guess], dtype=float)
    bounds = [
        (1e-4, 10.0),   # kappa
        (1e-6,  1.0),   # theta
        (1e-4,  5.0),   # sigma_v
        (-0.999, 0.999),# rho
        (1e-6,  1.0),   # v0
    ]
    return x0, bounds


def calibrate_heston(
    CP: OptionType,
    S0: float,
    K: np.ndarray | None = None,
    T: np.ndarray | None = None,
    r: float = 0.0,
    sigma: np.ndarray | None = None,
    calibration_target: str = "prices",
    weights: np.ndarray | None = None,
    **legacy_kwargs,
):
    """
    Calibra el modelo de Heston sobre la superficie de mercado seleccionada.

    Calibra los 5 parámetros (kappa, theta, sigma_v, rho, v0) simultáneamente
    e impone la condición de Feller (2κθ ≥ σ_v²) como restricción de desigualdad
    mediante el optimizador SLSQP.

    Args:
        CP (OptionType): Tipo de opción (CALL o PUT).
        S0 (float): Precio del activo subyacente.
        K (np.ndarray): Strikes de las opciones (valores absolutos).
        T (np.ndarray): Tiempos hasta el vencimiento.
        r (float): Tasa de interés sin riesgo.
        sigma (np.ndarray): Superficie de IVs de mercado [T × K], en decimal.
        calibration_target (str, optional): `"prices"` o `"iv"`. Defaults to "prices".
        weights (np.ndarray | None, optional): Pesos para el RMSE. Defaults to None.

    Returns:
        tuple: (params, res, prices_market)
            - params       : HestonParams calibrados.
            - res          : resultado completo de scipy.optimize.minimize.
            - prices_market: superficie de precios BS calculada a partir de las IVs de mercado.
    """
    if K is None:
        K = legacy_kwargs.pop("strikes", None)
    if T is None:
        T = legacy_kwargs.pop("maturities", None)
    if sigma is None:
        sigma = legacy_kwargs.pop("vol_market", None)
    if legacy_kwargs:
        raise TypeError(f"Argumentos no esperados: {', '.join(sorted(legacy_kwargs))}")
    if K is None or T is None or sigma is None:
        raise ValueError("Hay que proporcionar K, T y sigma para calibrar Heston.")

    x0, bounds = initial_guess_from_book(S0, K, T, sigma)
    prices_market = bs_price_surface(CP, S0, K, T, r, sigma)

    # Condición de Feller: 2·κ·θ ≥ σ_v²  →  g(x) = 2·x[0]·x[1] − x[2]² ≥ 0
    feller_constraint = {
        "type": "ineq",
        "fun": lambda x: 2.0 * x[0] * x[1] - x[2] ** 2,
    }

    if calibration_target == "prices":
        objective = lambda x: objective_surface_prices(
            x, CP, S0, K, T, r, prices_market, weights=weights,
        )
    elif calibration_target == "iv":
        objective = lambda x: objective_surface_iv(
            x, CP, S0, K, T, r, sigma, weights=weights,
        )
    else:
        raise ValueError("calibration_target debe ser 'prices' o 'iv'.")

    res = minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=[feller_constraint],
        options={"maxiter": 500, "ftol": 1e-9},
    )
    params = unpack_params(res.x)
    return params, res, prices_market


# ---------------------------------------------------------------------------
# Construcción de tablas de resultados
# ---------------------------------------------------------------------------

def surface_table(values: np.ndarray, maturity_labels: list[str], K: np.ndarray) -> pd.DataFrame:
    """
    Convierte una superficie numérica en una tabla de pandas con etiquetas legibles.
    Args:
        values (np.ndarray): Matriz de valores [T × K].
        maturity_labels (list[str]): Etiquetas de los vencimientos para las filas.
        K (np.ndarray): Strikes que se usarán como columnas.

    Returns:
        pd.DataFrame: Tabla con índices por vencimiento y columnas por strike.
    """
    return pd.DataFrame(values, index=maturity_labels, columns=[f"K={int(k)}" for k in K])


def build_output_table(
    CP: OptionType,
    S0: float,
    K: np.ndarray,
    T: np.ndarray,
    maturity_labels: list[str],
    r: float,
    sigma: np.ndarray,
    prices_market: np.ndarray,
    params: HestonParams,
) -> pd.DataFrame:
    """
    Construye una tabla plana con precios, volatilidades y errores para cada punto de la malla.
    Args:
        CP (OptionType): Tipo de opción (CALL o PUT).
        S0 (float): Precio del activo subyacente.
        K (np.ndarray): Strikes de las opciones.
        T (np.ndarray): Tiempos hasta el vencimiento.
        maturity_labels (list[str]): Etiquetas de los vencimientos.
        r (float): Tasa de interés sin riesgo.
        sigma (np.ndarray): Superficie de IVs de mercado [T × K], en decimal.
        prices_market (np.ndarray): Superficie de precios BS [T × K].
        params (HestonParams): Parámetros calibrados del modelo de Heston.

    Returns:
        pd.DataFrame: Tabla plana con valores de mercado, valores del modelo y errores.
    """
    rows = []
    model_prices = heston_price_surface(CP, S0, K, T, r, params)
    sigma_model  = heston_surface_sigma(CP, S0, K, T, r, params)

    for iT, t in enumerate(T):
        for iK, k in enumerate(K):
            rows.append(
                {
                    "maturity_label":  maturity_labels[iT],
                    "maturity":        t,
                    "strike":          k,
                    "option_type":     option_type_label(CP),
                    "market_iv":       sigma[iT, iK],
                    "market_price_bs": prices_market[iT, iK],
                    "heston_price":    model_prices[iT, iK],
                    "heston_iv":       sigma_model[iT, iK],
                    "abs_price_error": abs(model_prices[iT, iK] - prices_market[iT, iK]),
                    "abs_iv_error":    abs(sigma_model[iT, iK] - sigma[iT, iK]),
                }
            )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Funciones de impresión
# ---------------------------------------------------------------------------

def print_params(
    label: str, params: HestonParams, objective_name: str, objective_value: float
) -> None:
    """
    Imprime por pantalla los parámetros calibrados y el valor de la función objetivo.
    Args:
        label (str): Título descriptivo del bloque a imprimir.
        params (HestonParams): Parámetros del modelo de Heston.
        objective_name (str): Nombre de la función objetivo evaluada.
        objective_value (float): Valor alcanzado por la función objetivo.

    Returns:
        None: Esta función solo imprime.
    """
    sep = "=" * 60
    print(sep)
    print(f"  {label}")
    print(sep)
    print(f"  kappa = {params.kappa:.{DISPLAY_DECIMALS}f}")
    print(f"  theta = {params.theta:.{DISPLAY_DECIMALS}f}")
    print(f"  sigma = {params.sigma:.{DISPLAY_DECIMALS}f}")
    print(f"  rho   = {params.rho:.{DISPLAY_DECIMALS}f}")
    print(f"  v0    = {params.v0:.{DISPLAY_DECIMALS}f}")
    print(f"  --- {objective_name}: {objective_value:.{DISPLAY_DECIMALS}f} ---")
    print()


def print_configuration(config: CalibrationConfig, K: np.ndarray) -> None:
    """
    Imprime la configuración principal de la calibración que se va a ejecutar.
    Args:
        config (CalibrationConfig): Configuración general del script.
        K (np.ndarray): Strikes seleccionados para la calibración.

    Returns:
        None: Esta función solo imprime.
    """
    strike_text = ", ".join(f"{k:.1f}" for k in K)
    print("=" * 60)
    print("  Configuración de calibración")
    print("=" * 60)
    print(f"  Strikes seleccionados   : {strike_text}")
    print(f"  Tipo de opción (CP)     : {option_type_label(config.CP)}")
    print(f"  Objetivo calibración    : {config.calibration_target}")
    print()


# ---------------------------------------------------------------------------
# Script principal
# ---------------------------------------------------------------------------

def main() -> None:

    config = CalibrationConfig(
        CP=OptionType.CALL,
        calibration_target="prices",
        export_csv=True,
    )

    S0, K, T, maturity_labels, sigma = load_market_surface(config.selected_strikes)
    r = 0.0

    print_configuration(config, K)

    print("=" * 60)
    print("  Superficie de volatilidad implícita de mercado [%]")
    print("=" * 60)
    print_dataframe_fixed(surface_table(100.0 * sigma, maturity_labels, K))
    print()

    print(">>> Calibrando Heston (5 parámetros libres, condición de Feller impuesta) ...")
    params, result, prices_market = calibrate_heston(
        CP=config.CP,
        S0=S0,
        K=K,
        T=T,
        r=r,
        sigma=sigma,
        calibration_target=config.calibration_target,
    )

    # Verificación post-calibración de la condición de Feller
    feller_value = 2.0 * params.kappa * params.theta - params.sigma ** 2
    feller_ok = feller_value >= 0.0

    objective_name = "RMSE precios" if config.calibration_target == "prices" else "RMSE IV"
    print_params("Calibración Heston", params, objective_name, result.fun)
    print(
        f"  Condición de Feller  "
        f"(2κθ − σ² = {feller_value:.{DISPLAY_DECIMALS}f}): "
        f"{'✓ satisfecha' if feller_ok else '✗ NO satisfecha'}"
    )
    print()

    prices_heston = heston_price_surface(config.CP, S0, K, T, r, params)

    print("--- Precios de mercado (BS con volatilidad implícita observada) ---")
    print_dataframe_fixed(surface_table(prices_market, maturity_labels, K))
    print()

    print("--- Precios Heston calibrado (COS) ---")
    print_dataframe_fixed(surface_table(prices_heston, maturity_labels, K))
    print()

    output_table = build_output_table(
        CP=config.CP,
        S0=S0,
        K=K,
        T=T,
        maturity_labels=maturity_labels,
        r=r,
        sigma=sigma,
        prices_market=prices_market,
        params=params,
    )

    print("--- Tabla plana de resultados ---")
    print_dataframe_fixed(output_table, index=False)
    print()

    if config.export_csv:
        export_path = Path(__file__).resolve().with_name(config.export_filename)
        output_table_to_export = output_table.copy()
        numeric_cols = output_table_to_export.select_dtypes(include=[np.number]).columns
        output_table_to_export[numeric_cols] = output_table_to_export[numeric_cols].apply(
            lambda col: col.map(
                lambda x: round(x, DISPLAY_DECIMALS) if pd.notna(x) else x
            )
        )
        output_table_to_export.to_csv(
            export_path,
            index=False,
            float_format=f"%.{DISPLAY_DECIMALS}f",
        )
        print(f"CSV exportado en: {export_path}")
        print()


if __name__ == "__main__":
    main()
