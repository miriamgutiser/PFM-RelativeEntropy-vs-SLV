"""
===============================================================================
MONTE CARLO PRICING
===============================================================================
Este módulo contiene funciones auxiliares para la valoración de instrumentos
financieros mediante simulación Monte Carlo.

Incluye:
    - EUOptionPriceFromMCPaths
    - MCprice

Notas:
    - Se asume medida neutral al riesgo.
    - El descuento se realiza con tipo continuo r.
    - La precisión depende del número de trayectorias simuladas.
"""


import numpy as np
from core.types import OptionType


def EUOptionPriceFromMCPaths(CP,S,K,T,r):
    """
    Calcula el precio de una opción europea (Call o Put) mediante
    simulación Monte Carlo a partir de trayectorias simuladas del
    subyacente en el tiempo de vencimiento T.
    Args:
        CP (OptionType): Tipo de opción. Debe ser CALL o PUT
        S (np.ndarray): Vector de simulaciones Monte Carlo del precio
            del subyacente en el vencimiento T (S_T).
        K (float): Strike de la opción.
        T (float): Tiempo hasta vencimiento (en años).
        r (float): Tipo de interés libre de riesgo.

    Returns:
        float: Estimación Monte Carlo del precio actual de la opción.
    """

    # S is a vector of Monte Carlo samples at T

    if CP == OptionType.CALL:
        return np.exp(-r*T)*np.mean(np.maximum(S-K,0.0))
    elif CP == OptionType.PUT:
        return np.exp(-r*T)*np.mean(np.maximum(K-S,0.0))
    else:
        raise ValueError("OptionType debe ser CALL o PUT")


def MCprice(Kvec, Smat, T, r):
    """
    Calcula el precio de opciones europeas tipo call mediante Monte Carlo a partir de una matriz de precios 
    simulados (Smat) y un vector de strikes (Kvec).
    
    Args:
        Kvec (array_like): Vector de strikes (precios de ejercicio) para las opciones.
        Smat (array_like): Vector o matriz de precios finales del activo subyacente simulados por Monte Carlo.
        T (float): Tiempo hasta vencimiento de las opciones (en años).
        r (float): Tasa de interés libre de riesgo (continua).
    
    Returns:
        ndarray: Vector de precios de opciones call, uno por cada strike en Kvec.
                 Cada precio se calcula como E[max(S_T - K, 0)] * exp(-r*T).
    """
    Kvec = np.asarray(Kvec).flatten()
    val = np.zeros(len(Kvec))
    for i in range(len(Kvec)):
        payoff = np.maximum(Smat - Kvec[i], 0)
        val[i] = np.exp(-r*T) * np.mean(payoff)
    return val