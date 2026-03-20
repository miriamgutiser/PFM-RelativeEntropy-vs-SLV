"""
==========================================================
Heston_Calibration_Complete.py
==========================================================

Calibración del modelo de Heston a la superficie de volatilidad
implícita del mercado.

Tres modos de calibración:
    A) ATM-only            — un solo punto (K=S0, T=1y)
    B) ATM term structure  — toda la fila ATM
    C) Superficie completa — toda la rejilla (T, K)

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

from dataclasses import dataclass
from scipy.optimize import minimize, brentq

# ---- path setup (mismo patrón que HestonGoodGuessOfInitial.py) ----
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

# ---- project imports ----
from core.COS_method import CallPutOptionPriceCOSMthd
from core.pricing_Black_Scholes import BS_Call_Put_Option_Price
from core.types import OptionType
from volatilidad_estocastica.Heston_model import chf_heston


# ============================================================
# 1. DATOS DE MERCADO (Tabla 8.1 del libro)
#    Las volatilidades del libro están en porcentajes.
# ============================================================

def load_market_surface():
    """
    Carga la superficie de volatilidad implícita del mercado.

    Returns:
        S0: precio spot
        strikes_abs: strikes absolutos
        maturities: vencimientos en años
        maturity_labels: etiquetas de vencimientos
        vol_market: matriz de volatilidades implícitas
    """
    S0 = 100.0

    strike_pct = np.array([40, 60, 80, 100, 120, 140, 160, 180], dtype=float)
    strikes_abs = S0 * strike_pct / 100.0

    maturity_labels = ["1w", "1m", "6m", "1y", "2y", "5y", "8y", "10y"]
    maturities = np.array([1/52, 1/12, 0.5, 1.0, 2.0, 5.0, 8.0, 10.0], dtype=float)

    # Tabla 8.1, en %
    vol_pct = np.array([
        [22.622, 22.371, 20.706, 18.537, 18.041, 16.395, 14.491, 13.136],  # K=40
        [15.491, 15.350, 14.415, 13.195, 12.916, 11.988, 10.917, 10.177],  # K=60
        [ 9.930,  9.882,  9.562,  9.154,  9.070,  8.786,  8.459,  8.251],  # K=80
        [ 5.001,  5.070,  5.515,  6.041,  6.196,  6.639,  7.015,  7.221],  # K=100
        [ 4.542,  4.598,  4.962,  5.430,  5.654,  6.277,  6.782,  7.041],  # K=120
        [ 6.278,  6.300,  6.405,  6.460,  6.607,  6.982,  7.232,  7.328],  # K=140
        [ 7.748,  7.752,  7.707,  7.517,  7.615,  7.806,  7.824,  7.751],  # K=160
        [ 8.988,  8.976,  8.818,  8.445,  8.508,  8.563,  8.399,  8.182],  # K=180
    ], dtype=float)

    # Queremos forma (nT, nK)
    vol_market = (vol_pct.T) / 100.0
    return S0, strikes_abs, maturities, maturity_labels, vol_market


# ============================================================
# 2. MODELO DE HESTON — contenedor de parámetros
# ============================================================

@dataclass
class HestonParams:
    """
    Parámetros del modelo de Heston.
    
    kappa  = velocidad de reversión
    theta  = varianza media de largo plazo  (v̄ / vBar)
    sigma  = vol de la varianza             (γ / gamma)
    rho    = correlación
    v0     = varianza inicial
    """
    kappa: float
    theta: float
    sigma: float   # gamma en la notación del proyecto
    rho: float
    v0: float


# ============================================================
# 3. PRICING HESTON — usando COS method
# ============================================================

def _heston_call_price_COS(S0, K, T, r, p: HestonParams, N=5000, L=5):
    """
    Precio call europeo bajo Heston usando el método COS.
    Mapea los parámetros del dataclass HestonParams a la interfaz
    de chf_heston y CallPutOptionPriceCOSMthd.

    Args:
        S0 (float) : precio spot
        K (float) : strike (escalar)
        T (float) : vencimiento
        r (float) : tasa libre de riesgo
        p (HestonParams) : parámetros del modelo de Heston
        N (int) : número de términos COS
        L  : tamaño dominio truncamiento

    Returns:
        float : precio de la call
    """
    # Función característica del proyecto:
    #   chf_heston(u, tau, kappa, vBar, gamma, rho, v0, r)
    cf = lambda u: chf_heston(u, T, p.kappa, p.theta, p.sigma, p.rho, p.v0, r)

    K_arr = np.array([K])
    price = CallPutOptionPriceCOSMthd(cf, OptionType.CALL, S0, r, T, K_arr, N, L)

    # Extraer escalar del resultado
    if isinstance(price, np.ndarray):
        return float(price.flatten()[0])
    return float(price)


def _implied_vol_silent(price, S0, K, T, r):
    """
    Calcula la volatilidad implícita BS usando brentq (root-finding robusto)
    con BS_Call_Put_Option_Price del proyecto.
    
    Args:
        price (float): precio de la call
        S0 (float): precio spot
        K (float): strike
        T (float): vencimiento
        r (float): tasa libre de riesgo

    Returns:
        float : IV (en decimal), o np.nan si falla
    """
    if T <= 0 or price <= 0:
        return np.nan

    # Clamp: el precio de la call debe estar entre el valor intrínseco y S0
    intrinsic = max(S0 - K * np.exp(-r * T), 0.0)
    price = np.clip(price, intrinsic + 1e-12, S0 - 1e-12)

    def f(sigma):
        return float(BS_Call_Put_Option_Price(OptionType.CALL, S0, K, sigma, 0, T, r)) - price

    try:
        return brentq(f, 1e-6, 10.0, maxiter=300)
    except (ValueError, RuntimeError):
        return np.nan


def heston_surface_iv(S0, strikes, maturities, r, params: HestonParams):
    """
    Devuelve matriz (nT, nK) de volatilidades implícitas Heston
    usando el método COS.
    
    Args:
        S0 (float): precio spot
        strikes (np.ndarray): strikes
        maturities (np.ndarray): vencimientos
        r (float): tasa libre de riesgo
        params (HestonParams): parámetros del modelo de Heston

    Returns:
        np.ndarray: matriz (nT, nK) de volatilidades implícitas Heston
    """
    out = np.zeros((len(maturities), len(strikes)))
    for iT, T in enumerate(maturities):
        for iK, K in enumerate(strikes):
            c = _heston_call_price_COS(S0, K, T, r, params)
            out[iT, iK] = _implied_vol_silent(c, S0, K, T, r)
    return out


# ============================================================
# 4. FUNCIONES OBJETIVO DE CALIBRACIÓN
# ============================================================

def unpack_params(x, fix_kappa=None):
    """
    Si fix_kappa es None: x = [kappa, theta, sigma, rho, v0]
    Si fix_kappa es número : x = [theta, sigma, rho, v0]
    
    Args:
        x (np.ndarray): parámetros del modelo de Heston
        fix_kappa (float, optional): valor fijo de kappa. Defaults to None.

    Returns:
        HestonParams: parámetros del modelo de Heston
    """
    if fix_kappa is None:
        kappa, theta, sigma, rho, v0 = x
    else:
        kappa = fix_kappa
        theta, sigma, rho, v0 = x

    return HestonParams(
        kappa=float(kappa),
        theta=float(theta),
        sigma=float(sigma),
        rho=float(rho),
        v0=float(v0),
    )


def objective_surface(x, S0, strikes, maturities, r, vol_market, fix_kappa=None, weights=None):
    """
    Función objetivo para calibración de la superficie de volatilidades implícitas.
    
    Args:
        x (np.ndarray): parámetros del modelo de Heston
        S0 (float): precio spot
        strikes (np.ndarray): strikes
        maturities (np.ndarray): vencimientos
        r (float): tasa libre de riesgo
        vol_market (np.ndarray): volatilidades implícitas del mercado
        fix_kappa (float, optional): valor fijo de kappa. Defaults to None.
        weights (np.ndarray, optional): pesos para la función objetivo. Defaults to None.

    Returns:
        float: valor de la función objetivo
    """
    p = unpack_params(x, fix_kappa=fix_kappa)
    vol_model = heston_surface_iv(S0, strikes, maturities, r, p)

    err = vol_model - vol_market
    if weights is None:
        weights = np.ones_like(err)

    # Identificamos qué celdas son números válidos (ignorando los NaNs)
    valid_mask = np.isfinite(err)
    
    # Si literalmente todos fallan, entonces sí abortamos
    if not np.any(valid_mask):
        return 1e6

    # Calculamos el error solo sobre los puntos que no han dado NaN
    return np.sqrt(np.sum((weights[valid_mask] * err[valid_mask])**2))


def objective_atm_only(x, S0, strikes, maturities, r, vol_market, fix_kappa=None):
    """
    Calibración a un único punto ATM, por ejemplo ATM a 1 año.
    
    Args:
        x (np.ndarray): parámetros del modelo de Heston
        S0 (float): precio spot
        strikes (np.ndarray): strikes
        maturities (np.ndarray): vencimientos
        r (float): tasa libre de riesgo
        vol_market (np.ndarray): volatilidades implícitas del mercado
        fix_kappa (float, optional): valor fijo de kappa. Defaults to None.

    Returns:
        float: valor de la función objetivo
    """
    p = unpack_params(x, fix_kappa=fix_kappa)

    # Elegimos ATM y 1y
    idx_K = np.argmin(np.abs(strikes - S0))         # Busca el strike más cercano a ATM
    idx_T = np.argmin(np.abs(maturities - 1.0))     # Busca el vencimiento más cercano a 1 año

    # print(f"Strike ATM: {strikes[idx_K]}, Vencimiento 1y: {maturities[idx_T]}")
    # print(f"Volatilidad del mercado ATM: {vol_market[idx_T, idx_K]}")
    
    # Calcula la volatilidad del modelo ATM usando la fórmula de Heston(CF)
    vol_model = heston_surface_iv(S0, [strikes[idx_K]], [maturities[idx_T]], r, p)[0, 0]
    # print(f"Volatilidad del modelo ATM: {vol_model}")
    #  Busca en la tabla real de mercado cuál era la volatilidad observada ahí.
    vol_mkt = vol_market[idx_T, idx_K]
    # print(f"Volatilidad del mercado ATM: {vol_mkt}")
    return (vol_model - vol_mkt)**2


def objective_term_structure_atm(x, S0, strikes, maturities, r, vol_market, fix_kappa=None):
    """
    Calibración usando solo la fila ATM a todos los vencimientos.
    
    Args:
        x (np.ndarray): parámetros del modelo de Heston
        S0 (float): precio spot
        strikes (np.ndarray): strikes
        maturities (np.ndarray): vencimientos
        r (float): tasa libre de riesgo
        vol_market (np.ndarray): volatilidades implícitas del mercado
        fix_kappa (float, optional): valor fijo de kappa. Defaults to None.

    Returns:
        float: valor de la función objetivo
    """
    p = unpack_params(x, fix_kappa=fix_kappa)       # Desempaqueta los parámetros
    idx_K = np.argmin(np.abs(strikes - S0))         # Busca el strike más cercano a ATM

    print(f"Strike ATM seleccionado: {strikes[idx_K]}")
    # Calcula las volatilidades del modelo ATM usando la fórmula de Heston(CF)
    model = heston_surface_iv(S0, [strikes[idx_K]], maturities, r, p).reshape(-1)
    print(f"Volatilidad del modelo ATM (8 vencimientos): {model}")
    # Extrae de la tabla real del mercado toda la columna correspondiente a ese strike ATM
    market = vol_market[:, idx_K]
    print(f"Volatilidad del mercado ATM (8 vencimientos): {market}")
    return np.sqrt(np.sum((model - market)**2))


# ============================================================
# 5. HEURÍSTICAS INICIALES DEL LIBRO
# ============================================================

def initial_guess_from_book(S0, strikes, maturities, vol_market, fix_kappa=True):
    """
    Genera una estimación inicial de los parámetros de Heston basándose en heurísticas del libro.
    
    Args:
        S0 (float): precio spot
        strikes (np.ndarray): strikes
        maturities (np.ndarray): vencimientos
        vol_market (np.ndarray): volatilidades implícitas del mercado
        fix_kappa (bool, optional): si True, fija kappa en 0.5. Defaults to True.

    Returns:
        np.ndarray: parámetros del modelo de Heston
    """
    idx_K = np.argmin(np.abs(strikes - S0))             #Busca el strike más cercano a ATM
    idx_T_1w = np.argmin(np.abs(maturities - 1/52))     #Busca el vencimiento más cercano a 1 semana
    idx_T_10y = np.argmin(np.abs(maturities - 10.0))    #Busca el vencimiento más cercano a 10 años

    v0_guess = vol_market[idx_T_1w, idx_K]**2          #v0 es la varianza inicial, se aproxima con la volatilidad del mercado a 1 semana
    theta_guess = vol_market[idx_T_10y, idx_K]**2      #theta es la varianza a largo plazo, se aproxima con la volatilidad del mercado a 10 años

    # Semillas razonables
    kappa_guess = 0.5
    sigma_guess = 0.3
    rho_guess = -0.7

    # Si fix_kappa es True, fija kappa en 0.5, si no, lo deja como una variable libre.
    if fix_kappa:
        x0 = np.array([theta_guess, sigma_guess, rho_guess, v0_guess], dtype=float)
        bounds = [
            (1e-6, 1.0),   # theta
            (1e-4, 5.0),   # sigma
            (-0.999, 0.999), # rho
            (1e-6, 1.0),   # v0
        ]
        return x0, bounds, kappa_guess
    else:
        x0 = np.array([kappa_guess, theta_guess, sigma_guess, rho_guess, v0_guess], dtype=float)
        bounds = [
            (1e-4, 10.0),  # kappa
            (1e-6, 1.0),   # theta
            (1e-4, 5.0),   # sigma
            (-0.999, 0.999), # rho
            (1e-6, 1.0),   # v0
        ]
        return x0, bounds, None


def calibrate_heston(S0, strikes, maturities, r, vol_market, mode="surface", fix_kappa=True):
    """
    Calibra los parámetros del modelo de Heston a la superficie de volatilidades implícitas.
    
    Args:
        S0 (float): precio spot
        strikes (np.ndarray): strikes
        maturities (np.ndarray): vencimientos
        r (float): tasa libre de riesgo
        vol_market (np.ndarray): volatilidades implícitas del mercado
        mode (str, optional): modo de calibración. Puede ser 'atm_only', 'atm_term_structure' o 'surface'. Defaults to "surface".
        fix_kappa (bool, optional): si True, fija kappa en 0.5. Defaults to True.

    Returns:
        dict: diccionario con los parámetros calibrados y resultados
    """

    # Genera una estimación inicial de los parámetros de Heston.
    x0, bounds, fixed_kappa = initial_guess_from_book(S0, strikes, maturities, vol_market, fix_kappa=fix_kappa)

    if mode == "atm_only":
        fun = lambda x: objective_atm_only(x, S0, strikes, maturities, r, vol_market, fix_kappa=fixed_kappa)
    elif mode == "atm_term_structure":
        fun = lambda x: objective_term_structure_atm(x, S0, strikes, maturities, r, vol_market, fix_kappa=fixed_kappa)
    elif mode == "surface":
        fun = lambda x: objective_surface(x, S0, strikes, maturities, r, vol_market, fix_kappa=fixed_kappa)
    else:
        raise ValueError("mode debe ser 'atm_only', 'atm_term_structure' o 'surface'.")

    # Minimiza la función objetivo usando el método L-BFGS-B.
    res = minimize(
        fun,
        x0,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 200}
    )

    # Parámetros calibrados
    p = unpack_params(res.x, fix_kappa=fixed_kappa)
    return p, res


# ============================================================
# 6. GRÁFICAS
# ============================================================

def plot_surface(strikes, maturities, vol_surface, title):
    """
    Grafica la superficie de volatilidades implícitas.
    
    Args:
        strikes (np.ndarray): strikes
        maturities (np.ndarray): vencimientos
        vol_surface (np.ndarray): superficie de volatilidades implícitas
        title (str): título de la gráfica
    """ 
    K_grid, T_grid = np.meshgrid(strikes / 100.0, maturities)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_wireframe(K_grid, T_grid, 100.0 * vol_surface, rstride=1, cstride=1)
    ax.set_xlabel("Strike / S0")
    ax.set_ylabel("Time to maturity")
    ax.set_zlabel("Implied volatility [%]")
    ax.set_title(title)
    plt.tight_layout()
    plt.show()


def compare_market_vs_model(strikes, maturities, vol_market, vol_model, title):
    """
    Compara las volatilidades implícitas del mercado con el modelo de Heston.
    
    Args:
        strikes (np.ndarray): strikes
        maturities (np.ndarray): vencimientos
        vol_market (np.ndarray): volatilidades implícitas del mercado
        vol_model (np.ndarray): volatilidades implícitas del modelo de Heston
        title (str): título de la gráfica
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Smile para algunos vencimientos
    chosen_T_idx = [0, 2, 3, 7]  # 1w, 6m, 1y, 10y
    for idx in chosen_T_idx:
        axes[0].plot(strikes, 100 * vol_market[idx], "o-", alpha=0.7, label=f"Mkt T={maturities[idx]:.3g}")
        axes[0].plot(strikes, 100 * vol_model[idx], "--", alpha=0.9, label=f"Heston T={maturities[idx]:.3g}")
    axes[0].set_title("Smiles: mercado vs Heston")
    axes[0].set_xlabel("Strike")
    axes[0].set_ylabel("IV [%]")
    axes[0].legend(fontsize=8)
  
    # ATM term structure
    idx_atm = np.argmin(np.abs(strikes - 100.0))
    axes[1].plot(maturities, 100 * vol_market[:, idx_atm], "o-", label="Mercado ATM")
    axes[1].plot(maturities, 100 * vol_model[:, idx_atm], "--", label="Heston ATM")
    axes[1].set_title("Estructura temporal ATM")
    axes[1].set_xlabel("Maturity")
    axes[1].set_ylabel("IV [%]")
    axes[1].legend()

    fig.suptitle(title)
    plt.tight_layout()
    plt.show()
    plt.tight_layout()
    plt.show()

def plot_calibrated_surface(strikes, maturities, vol_surface, S0, mode, title):
    """
    Grafica la superficie de volatilidades implícitas.

    Args:
        strikes (np.ndarray): strikes
        maturities (np.ndarray): vencimientos
        vol_surface (np.ndarray): superficie de volatilidades implícitas
        S0 (float): precio inicial del activo subyacente
        mode (str): modo de calibración
        title (str): título de la gráfica
    """
    K_grid, T_grid = np.meshgrid(strikes / 100.0, maturities)
    
    idx_K = np.argmin(np.abs(strikes - S0))
    idx_T = np.argmin(np.abs(maturities - 1.0))
    
    fig = plt.figure(figsize=(8, 6))
    fig.patch.set_facecolor('white')
    ax = fig.add_subplot(111, projection="3d")
    
    ax.plot_wireframe(K_grid, T_grid, 100.0 * vol_surface, color='#cc7722', rstride=1, cstride=1)
    
    if mode == "atm_only":
        ax.scatter(strikes[idx_K]/100.0, maturities[idx_T], 100.0 * vol_surface[idx_T, idx_K], color='#8b4513', s=40, depthshade=False)
    elif mode == "atm_term_structure":
        ax.scatter(np.repeat(strikes[idx_K]/100.0, len(maturities)), maturities, 100.0 * vol_surface[:, idx_K], color='#8b4513', s=40, depthshade=False)
    elif mode == "surface":
        ax.scatter(K_grid, T_grid, 100.0 * vol_surface, color='#8b4513', s=15, depthshade=False)

    ax.set_xlabel("Strike")
    ax.set_ylabel("Vencimiento")
    ax.set_zlabel("volatilidad implícita [%]")
    ax.set_title(title)
    ax.grid(True)

    plt.tight_layout()
    plt.show()


# ============================================================
# 7. MAIN
# ============================================================

def main():
    """
    Función principal para calibrar el modelo de Heston a la superficie de volatilidades implícitas.
    """
    S0, strikes, maturities, maturity_labels, vol_market = load_market_surface()

    # Tipo libre de riesgo fijo; aquí lo tomo 0 por simplicidad, porque en la tabla
    # del libro solo se da la superficie de IV, no una curva de tipos asociada.
    r = 0.0

    print("=== Datos del libro (Tabla 8.1) ===")
    df = pd.DataFrame(100 * vol_market, index=maturity_labels, columns=[f"K={int(k)}" for k in strikes])
    print(df.round(3))
    print()

    # --------------------------------------------------------
    # A) Calibración a un solo punto ATM
    # --------------------------------------------------------
    p_atm, res_atm = calibrate_heston(
        S0, strikes, maturities, r, vol_market,
        mode="atm_only", fix_kappa=True
    )
    vol_atm = heston_surface_iv(S0, strikes, maturities, r, p_atm)

    print("=== Calibración ATM-only ===")
    print(p_atm)
    print("objective =", res_atm.fun)
    plot_calibrated_surface(strikes, maturities, vol_atm, S0, "atm_only", "Heston calibrado solo a un punto ATM")

    # --------------------------------------------------------
    # B) Calibración a la estructura temporal ATM
    # --------------------------------------------------------
    p_term, res_term = calibrate_heston(
        S0, strikes, maturities, r, vol_market,
        mode="atm_term_structure", fix_kappa=True
    )
    vol_term = heston_surface_iv(S0, strikes, maturities, r, p_term)

    print("=== Calibración ATM term structure ===")
    print(p_term)
    print("objective =", res_term.fun)
    plot_calibrated_surface(strikes, maturities, vol_term, S0, "atm_term_structure", "Heston calibrado a la estructura temporal ATM")

    # --------------------------------------------------------
    # C) Calibración a toda la superficie
    # --------------------------------------------------------
    p_surf, res_surf = calibrate_heston(
        S0, strikes, maturities, r, vol_market,
        mode="surface", fix_kappa=True
    )
    vol_surf = heston_surface_iv(S0, strikes, maturities, r, p_surf)

    print("=== Calibración superficie completa ===")
    print(p_surf)
    print("objective =", res_surf.fun)
    plot_calibrated_surface(strikes, maturities, vol_surf, S0, "surface", "Heston calibrado a toda la superficie")

    # Comparaciones
    plot_surface(strikes, maturities, vol_market, "Superficie de volatilidad del mercado (Tabla 8.1)")
    compare_market_vs_model(strikes, maturities, vol_market, vol_surf, "Mercado vs Heston (calibración superficie completa)")


if __name__ == "__main__":
    main()