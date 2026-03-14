"""
===============================================================================
Black–Scholes (precios y griegas)
===============================================================================
Este módulo contiene funciones auxiliares para valoración y sensibilidades (griegas)
de opciones europeas bajo el modelo de Black–Scholes con volatilidad constante.

Incluye:
    - Cálculo de d1 y d2
    - Precio Black–Scholes para Call/Put (con entrada en (t,T) o en tau=T-t)
    - Delta, Gamma y Vega de Black–Scholes
Notas:
    - Las funciones aceptan escalares o arrays (vía NumPy).
    - El tipo de opción se especifica con `OptionType.CALL` o `OptionType.PUT`.
"""


import enum
import numpy as np
import scipy.stats as st
from core.types import OptionType

def _d1_d2(S_0,K,sigma,t,T,r):
    """
    Cálculo de d_1 y d_2 en Black_Scholes.
    Args:
        S_0 (float): precio actual del subyacente
        K (float): Strike
        sigma (float): Volatilidad constante
        t (float): tiempo actual
        T (float): tiempo hasta vencimiento
        r (float): tipo de interés libre de riesgo

    Returns:
        float: Precio teórico de la opción
    """
    S_0 = np.asarray(S_0, dtype=float)
    K  = np.asarray(K, dtype=float)
    tau = np.asarray(T-t, dtype=float)

    d1 = (np.log(S_0 / K) + (r + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))
    d2 = d1 - sigma * np.sqrt(tau)
    return d1, d2


# Black-Scholes call option price




def BS_Call_Put_Option_Price(CP,S_0,K,sigma,t,T,r):
    """
    Cálculo del precio de una Opcion ya sea Call o Put bajo el modelo de Black-Scholes con volatilidad constante.
    Args:
        CP (String): "c" para Call, "p" para Put
        S_0 (float): precio actual del subyacente
        K (float): Strike
        sigma (float): Volatilidad constante
        t (float): tiempo actual
        T (float): tiempo hasta vencimiento
        r (float): tipo de interés libre de riesgo

    Returns:
       float: Precio teórico de la opción.
    """
    #print('Maturity T={0} and t={1}'.format(T,t))
    #print(float(sigma * np.sqrt(T-t)))
    #print('strike K ={0}'.format(K))

    if K is list:
        K = np.array(K).reshape([len(K),1])
    d1, d2 = _d1_d2 (S_0,K,sigma,t,T,r)
    if CP == OptionType.CALL:
        value = st.norm.cdf(d1) * S_0 - st.norm.cdf(d2) * K * np.exp(-r * (T-t))
    elif CP == OptionType.PUT :
        value = st.norm.cdf(-d2) * K * np.exp(-r * (T-t)) - st.norm.cdf(-d1)*S_0
    else:
        raise ValueError("OptionType debe ser CALL o PUT")
    
    return value

def BS_Call_Put_Option_Price_tau(CP,S_0,K,sigma,tau,r):
    """
    Cálculo del precio de una Opcion ya sea Call o Put bajo el modelo de Black-Scholes con volatilidad constante.
    Se toma tau como 
        tau = T-t
    en la función anterior. 
    Args:
        CP (String): "c" para Call, "p" para Put
        S_0 (float): precio actual del subyacente
        K (float): Strike
        sigma (float): Volatilidad constante
        tau (float): tiempo hasta vencimiento
        r (float): tipo de interés libre de riesgo

    Returns:
       float: Precio teórico de la opción.
    """
    if K is list:
        K = np.array(K).reshape([len(K),1])
    d1, d2 = _d1_d2 (S_0,K,sigma,0,tau,r)
    if CP == OptionType.CALL:
        value = st.norm.cdf(d1) * S_0 - st.norm.cdf(d2) * K * np.exp(-r * tau)
    elif CP == OptionType.PUT :
        value = st.norm.cdf(-d2) * K * np.exp(-r * tau) - st.norm.cdf(-d1)*S_0
    else:
        raise ValueError("OptionType debe ser CALL o PUT")
    
    return value

def BS_Delta(CP,S_0,K,sigma,t,T,r):
    """
    Cálculo de la sensibilidad del precio de un derivado ante cambios en el precio de su activo subyacente.
    Args:
        CP (String): "c" para Call, "p" para Put
        S_0 (float): Precio actual del subyacente
        K (float): Strike
        sigma (float): volatilidad constante
        t (float): tiempo actual
        T (float): tiempo hasta vencimiento
        r (float): tipo de interés libre de riesgo

    Returns:
        float: Delta
    """
    if K is list:
        K = np.array(K).reshape([len(K),1])
    d1, _ = _d1_d2(S_0,K,sigma,t,T,r)
    if CP == OptionType.CALL:
        value = st.norm.cdf(d1)
    elif CP == OptionType.PUT:
       value = st.norm.cdf(d1)-1
    else:
        raise ValueError("OptionType debe ser CALL o PUT")
    
    return value

def BS_Gamma(S_0,K,sigma,t,T,r):
    """
    Cálculo de la sensibilidad y tasa de cambio del Delta ante variaciones en el precio del activo subyacente.
    Args:
        S_0 (float): Precio actual del subyacente
        K (float): Strike
        sigma (float): volatilidad constante
        t (float): tiempo actual
        T (float): tiempo hasta vencimiento
        r (float): tipo de interés libre de riesgo

    Returns:
        float: Gamma
    """
    if K is list:
        K = np.array(K).reshape([len(K),1])
    d1, _ = _d1_d2(S_0,K,sigma,t,T,r)
    return st.norm.pdf(d1) / (S_0 * sigma * np.sqrt(T-t))

def BS_Vega(S_0,K,sigma,t,T,r):
    """
    Cálculo de la sensibilidad del precio de la opción respecto a la volatilidad (Vega = ∂V/∂σ)
    Args:
         S_0 (float): Precio actual del subyacente
        K (float): Strike
        sigma (float): volatilidad constante
        t (float): tiempo actual
        T (float): tiempo hasta vencimiento
        r (float): tipo de interés libre de riesgo

    Returns:
        float: Vega
    """
    d1, _ = _d1_d2(S_0,K,sigma,t,T,r)
    return S_0*st.norm.pdf(d1)*np.sqrt(T-t)

def BS_Vega_tau(S_0,K,sigma,tau,r):
    """
    Cálculo de la sensibilidad del precio de la opción respecto a la volatilidad (Vega = ∂V/∂σ)
    Args:
         S_0 (float): Precio actual del subyacente
        K (float): Strike
        sigma (float): volatilidad constante
        t (float): tiempo actual
        T (float): tiempo hasta vencimiento
        r (float): tipo de interés libre de riesgo

    Returns:
        float: Vega
    """
    d1, _ = _d1_d2(S_0,K,sigma,0,tau,r)
    return S_0*st.norm.pdf(d1)*np.sqrt(tau)
