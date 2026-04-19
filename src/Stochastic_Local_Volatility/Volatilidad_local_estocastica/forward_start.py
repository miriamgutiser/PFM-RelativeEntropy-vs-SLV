"""
Funciones auxiliares para opciones forward-start bajo el modelo de Heston SLV.

Incluye:
    - ChFHestonForward                 : función característica forward-starting
    - CallPutOptionPriceCOSMthd_FrwdStart : precio vía método COS
    - BS_Call_Option_Price_FrwdStart   : precio Black-Scholes forward-start
    - ImpliedVolatilityFrwdStart       : volatilidad implícita forward-start
"""

import numpy as np
from scipy.stats import norm
from scipy.optimize import fsolve

from Stochastic_Local_Volatility.core.COS_method import CallPutCoefficients
from common.types import OptionType


def ChFHestonForward(u, T_1, T_2, kappa, vBar, gamma, rho, v0, r):
    """
    Función característica forward-starting del modelo de Heston.

    Utiliza la función generadora de momentos (MGF) del proceso CIR evaluada en
    T_1 para marginalizar v_{T_1}:

        c_bar(t)     = gamma^2 / (4*kappa) * (1 - exp(-kappa*t))
        d_bar        = 4*kappa*vBar / gamma^2
        kappa_bar(t) = 4*kappa*v0*exp(-kappa*t) / (gamma^2*(1-exp(-kappa*t)))
        a_1 = C * c_bar(T1) * kappa_bar(T1) / (1 - 2*C*c_bar(T1))
        a_2 = (1 / (1 - 2*C*c_bar(T1)))^(0.5*d_bar)
        ChF = exp(A + a_1) * a_2

    Args:
        u (float o ndarray): argumento complejo de la función característica
        T_1 (float): tiempo de inicio del período forward
        T_2 (float): vencimiento de la opción
        kappa (float): velocidad de reversión a la media
        vBar (float): varianza a largo plazo
        gamma (float): volatilidad de la varianza (vol-vol)
        rho (float): correlación entre BM del activo y de la varianza
        v0 (float): varianza inicial en t=0
        r (float): tipo de interés libre de riesgo

    Returns:
        ndarray: valores de la función característica
    """
    i   = 1j
    tau = T_2 - T_1

    # Coeficientes MGF del proceso CIR evaluados en T_1
    c_bar     = gamma**2 / (4.0 * kappa) * (1.0 - np.exp(-kappa * T_1))
    d_bar     = 4.0 * kappa * vBar / gamma**2
    kappa_bar = (4.0 * kappa * v0 * np.exp(-kappa * T_1)
                 / (gamma**2 * (1.0 - np.exp(-kappa * T_1))))

    # Componentes del modelo Heston en el período [T_1, T_2]
    D_1 = np.sqrt((kappa - i * rho * gamma * u)**2 + (u**2 + i * u) * gamma**2)
    g   = (kappa - i * rho * gamma * u - D_1) / (kappa - i * rho * gamma * u + D_1)

    C = ((1.0 / gamma**2) * (1.0 - np.exp(-D_1 * tau))
         / (1.0 - g * np.exp(-D_1 * tau))
         * (kappa - gamma * rho * i * u - D_1))
    A = (i * u * r * tau
         + kappa * vBar * tau / gamma**2 * (kappa - gamma * rho * i * u - D_1)
         - 2.0 * kappa * vBar / gamma**2
         * np.log((1.0 - g * np.exp(-D_1 * tau)) / (1.0 - g)))

    # Ajuste forward-start vía MGF
    a_1 = C * c_bar * kappa_bar / (1.0 - 2.0 * C * c_bar)
    a_2 = (1.0 / (1.0 - 2.0 * C * c_bar))**(0.5 * d_bar)

    return np.exp(A + a_1) * a_2


def CallPutOptionPriceCOSMthd_FrwdStart(cf, CP, r, T1, T2, K, N, L):
    """
    Precio de una opción forward-start (Call o Put) utilizando el método COS.

    Calcula el valor descontado de una opción cuyo strike se fija en $T_1$ como un
    porcentaje $K$ del precio del activo $S_{T_1}$. El payoff en $T_2$ es
    $(S_{T_2} - K \cdot S_{T_1})^+$ para un Call y $(K \cdot S_{T_1} - S_{T_2})^+$ para un Put.

    Args:
        cf (callable): función característica del log-retorno forward $\ln(S_{T_2}/S_{T_1})$
        CP (OptionType): tipo de opción (Call o Put)
        r (float): tipo de interés libre de riesgo
        T1 (float): tiempo de inicio del periodo forward
        T2 (float): tiempo de vencimiento de la opción
        K (float o ndarray): strike relativo (moneyness)
        N (int): número de términos en la expansión de Fourier
        L (float): parámetro para definir el rango de truncamiento [a, b]

    Returns:
        ndarray: precio de la opción en t=0 evaluado para cada strike relativo K
    """
    K   = np.atleast_1d(K) + 1.0
    tau = T2 - T1
    x0  = np.log(1.0 / K)

    a = -L * np.sqrt(tau)
    b =  L * np.sqrt(tau)

    k_idx = np.linspace(0, N - 1, N).reshape(N, 1)
    u     = k_idx * np.pi / (b - a)

    H_k  = CallPutCoefficients(CP, a, b, k_idx)
    temp = cf(u) * H_k
    temp[0] = 0.5 * temp[0]

    mat   = np.exp(1j * np.outer(x0 - a, u.flatten()))
    value = np.exp(-r * T2) * K * np.real(mat @ temp.flatten())
    return value


def BS_Call_Option_Price_FrwdStart(K, sigma, T1, T2, r):
    """
    Precio Black-Scholes de una opción call forward-start.

    K se pasa como (moneyness - 1); internamente se desplaza al strike real.
    Fórmula (coincide con MATLAB):
        value = exp(-r*T1)*N(d1) - N(d2)*K_adj*exp(-r*T2)

    Args:
        K (float o ndarray): strike desplazado (k - 1)
        sigma (float): volatilidad
        T1 (float): tiempo de inicio del período forward
        T2 (float): vencimiento
        r (float): tipo de interés libre de riesgo

    Returns:
        ndarray: precios de la opción
    """
    K_adj = np.atleast_1d(K) + 1.0
    tau   = T2 - T1
    with np.errstate(divide='ignore', invalid='ignore'):
        d1 = (np.log(1.0 / K_adj) + (r + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))
        d2 = d1 - sigma * np.sqrt(tau)
    return np.exp(-r * T1) * norm.cdf(d1) - norm.cdf(d2) * K_adj * np.exp(-r * T2)


def ImpliedVolatilityFrwdStart(marketPrice, K, T1, T2, r, initialVol=0.3):
    """
    Volatilidad implícita para opciones forward-start (vectorizado sobre K).

    Traduce el `fzero` de MATLAB: resuelve element-wise para cada strike.

    Args:
        marketPrice (float o ndarray): precio(s) de mercado de la opción
        K (float o ndarray): vector de strikes desplazados (k - 1)
        T1 (float): tiempo de inicio del período forward
        T2 (float): vencimiento
        r (float): tipo de interés libre de riesgo
        initialVol (float): volatilidad inicial para el solver

    Returns:
        ndarray: volatilidades implícitas
    """
    marketPrice = np.atleast_1d(marketPrice).flatten()
    K           = np.atleast_1d(K).flatten()
    impliedVol  = np.zeros(len(K))

    for i in range(len(K)):
        mp = marketPrice[i]
        ki = K[i]
        func = lambda sigma, _mp=mp, _ki=ki: (
            BS_Call_Option_Price_FrwdStart(_ki, sigma, T1, T2, r) - _mp
        )
        try:
            impliedVol[i] = fsolve(func, initialVol)[0]
        except Exception:
            impliedVol[i] = np.nan

    return impliedVol
