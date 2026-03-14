"""
===============================================================================
Interpolación y Volatilidad Implícita (Black–Scholes)
===============================================================================
Este módulo contiene utilidades para:

    - Interpolar superficies definidas en una rejilla regular (t, S)
      a lo largo de una trayectoria simulada (t_path, S_path).
    - Calcular la volatilidad implícita de Black–Scholes a partir de un
      precio de mercado (call/put) mediante:
          (i) estimación inicial por interpolación en una malla de volatilidades
         (ii) refinamiento con Newton–Raphson.

Incluye:
    - interpolate_along_path
    - ImpliedVolatility

Dependencias:
    - NumPy
    - SciPy (RegularGridInterpolator, optimize.newton)
    - Funciones de pricing Black–Scholes y el enum OptionType
"""


import numpy as np
from scipy.interpolate import RegularGridInterpolator
import scipy.optimize as optimize
from scipy.optimize import brentq, fsolve
from scipy.stats import norm

from core.pricing_Black_Scholes import BS_Call_Put_Option_Price,BS_Call_Put_Option_Price_tau
from core.types import OptionType

def interpolate_along_path(time_grid, s_grid, surface, t_path, s_path, fill_value=np.nan):
    """
    Interpola una superficie definida en una rejilla (time_grid, s_grid)
    a lo largo de una trayectoria (t_path, s_path).
    """
    F = RegularGridInterpolator(
        (time_grid, s_grid),
        surface,
        bounds_error=False,
        fill_value=fill_value
    )

    V = np.empty(len(t_path), dtype=float)

    for j, (tj, sj) in enumerate(zip(t_path, s_path)):
        V[j] = float(F((tj, sj)))

    return V

# Implied volatility method

def ImpliedVolatility(CP,marketPrice,K,T,S_0,r):
    """
    Calcula la volatilidad implícita de Black-Scholes (sigma_impl) a partir del precio de mercado
    de una opción europea (call/put), resolviendo numéricamente:

        BS_price(CP, S_0, K, sigma, T, r) = marketPrice

    Se construye una malla de volatilidades y se evalúa el precio BS en dicha malla para 
    obtener valor inicial de sigma por interpolación. 
    Se refina la solución aplicando el método de Newton-Raphson sobre la función:

            f(sigma) = BS_price(sigma) -  marketPrice

    Args:
        CP (OptionType): Tipo de opción (CALL o PUT).
        marketPrice (float): Precio observado en el mercado para la opción
        K (float): Strike de la opción
        T (float): Tiempo a vencimiento (maturity)
        S_0 (float): precio spot del subyacente
        r (float): tipo de interés libre de riesgo

    Returns:
        float: Volatilidad implícita que reproduce el precio de mercado bajo Black-Scholes
    """

    # To determine initial volatility we define a grid for sigma
    # and interpolate on the inverse function

    sigmaGrid = np.linspace(0,2,200)
    optPriceGrid = BS_Call_Put_Option_Price_tau(CP,S_0,K,sigmaGrid,T,r)
    sigmaInitial = np.interp(marketPrice,optPriceGrid,sigmaGrid)
    print("Initial volatility = {0}".format(sigmaInitial))
    
    # Use already determined input for the local-search (final tuning)

    func = lambda sigma: float(np.power(BS_Call_Put_Option_Price(CP,S_0,K,sigma,0,T,r) - marketPrice, 1.0))
    try:
        impliedVol = optimize.newton(func, float(sigmaInitial), tol=1e-15, maxiter=100)
    except RuntimeError:
        impliedVol = sigmaInitial
    
    print("Final volatility = {0}".format(impliedVol))
    return float(impliedVol)


def ImpliedVolatility_Vectorized(CP, marketPrice, K, T, S_0, r):
    """Calcula volatilidades implícitas para un vector de strikes (vectorizado).
    
    Extiende la función ImpliedVolatility para aplicarla a múltiples strikes simultáneamente.
    Itera sobre cada strike y calcula su IV correspondiente, suprimiendo los prints internos.
    
    Args:
        CP (OptionType): Tipo de opción (CALL o PUT).
        marketPrice (ndarray): Vector de precios de mercado, uno por strike.
        K (ndarray): Vector de strikes.
        T (float): Tiempo hasta vencimiento (maturity).
        S_0 (float): Precio inicial del activo subyacente.
        r (float): Tasa de interés libre de riesgo.
    
    Returns:
        ndarray: Vector de volatilidades implícitas, una por strike.
    """
    import io
    from contextlib import redirect_stdout
    
    impliedVol = np.zeros(len(K))
    for idx in range(len(K)):
        try:
            # Suprime prints internos de ImpliedVolatility
            with redirect_stdout(io.StringIO()):
                impliedVol[idx] = ImpliedVolatility(CP, marketPrice[idx], K[idx], T, S_0, r)
        except:
            # Si falla, retorna NaN
            impliedVol[idx] = np.nan
    return impliedVol
