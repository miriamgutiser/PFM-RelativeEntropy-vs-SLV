#%%
"""
Created on Wed Oct 24 20:37:32 2018
Black- Scholes implied volatility function
@author: Lech A. Grzelak
"""



import numpy as np
import scipy.stats as st
from common.types import OptionType
from Stochastic_Local_Volatility.core.pricing_Black_Scholes import BS_Call_Put_Option_Price, BS_Vega, BS_Call_Put_Option_Price_tau, BS_Vega_tau

def ImpliedVolatility(CP,S_0,K,sigma,t,T,r,V_market):
    """
    Cálculo de la volatilidad implícita mediante el método de Newton–Raphson.

    Esta función determina la volatilidad implícita sigma_impl tal que el precio
    teórico de Black–Scholes coincida con el precio observado en mercado
    (V_market).
    Args:
        CP (String): Call o Put.
        S_0 (float): precio actual del subyacente
        K (float): Strike
        sigma (float): Volatilidad constante, valor inicial para la iteración
        t (float): tiempo actual
        T (float): tiempo hasta vencimiento
        r (float): tipo de interés libre de riesgo
        V_market (float): precio de mercado de la opción

    Returns:
        float: Volatilidad implícita que iguala el precio del modelo
               con el precio de mercado.

    Observaciones:
        - El algoritmo itera hasta que la diferencia entre dos iteraciones
          consecutivas es menor que 1e-10.
        - Requiere que Vega no sea cercana a cero para garantizar estabilidad.
    """

    error    = 1e10; # initial error

    # Convenient lambda expressions

    optPrice = lambda sigma: BS_Call_Put_Option_Price(CP,S_0,K,sigma,t,T,r)
    vega= lambda sigma: BS_Vega(S_0,K,sigma,t,T,r)
    
    # When the difference between the model and market price is large
    # perform the following iteration

    while error>10e-10:
        f         = V_market - optPrice(sigma);
        f_prim    = -vega(sigma);
        sigma_new = sigma - f / f_prim;
    
        error=abs(sigma_new-sigma);
        sigma=sigma_new;
    return sigma



def ImpliedVolatility_tau(CP,S_0,K,sigma,tau,r,V_market):
    """
    Cálculo de la volatilidad implícita mediante el método de Newton–Raphson.
    
    Esta función determina la volatilidad implícita sigma_impl tal que el precio
    teórico de Black–Scholes coincida con el precio observado en mercado
    (V_market).
    Args:
        CP (String): Call o Put.
        S_0 (float): precio actual del subyacente
        K (float): Strike
        sigma (float): Volatilidad constante, valor inicial para la iteración
        t (float): tiempo actual
        T (float): tiempo hasta vencimiento
        r (float): tipo de interés libre de riesgo
        V_market (float): precio de mercado de la opción

    Returns:
        float: Volatilidad implícita que iguala el precio del modelo
               con el precio de mercado.

    Observaciones:
        - El algoritmo itera hasta que la diferencia entre dos iteraciones
          consecutivas es menor que 1e-10.
        - Requiere que Vega no sea cercana a cero para garantizar estabilidad.
    """

    error    = 1e10; # initial error

    # Convenient lambda expressions

    optPrice = lambda sigma: BS_Call_Put_Option_Price_tau(CP,S_0,K,sigma,tau,r)
    vega= lambda sigma: BS_Vega_tau(S_0,K,sigma,tau,r)
    
    # When the difference between the model and market price is large
    # perform the following iteration

    while error>10e-10:
        f         = V_market - optPrice(sigma);
        f_prim    = -vega(sigma);
        sigma_new = sigma - f / f_prim;
    
        error=abs(sigma_new-sigma);
        sigma=sigma_new;
    return sigma
