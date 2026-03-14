"""
===============================================================================
SABR – HAGAN IMPLIED VOLATILITY (2002)
===============================================================================
Este módulo implementa la aproximación asintótica de Hagan et al. (2002)
para la volatilidad implícita de Black–Scholes bajo el modelo SABR.

Modelo SABR bajo medida neutral al riesgo:

    dF_t = α_t F_t^β dW_t^1
    dα_t = γ α_t dW_t^2
    d⟨W^1, W^2⟩_t = ρ dt

Hagan et al. derivan una expansión asintótica en tiempo corto que
proporciona una expresión cerrada aproximada para:

    σ_impl(K, T)

Incluye:
    - HaganImpliedVolatility

Notas:
    - Aproximación válida para vencimientos no muy largos.
    - Muy utilizada en calibración de smiles de tipos de interés.
"""


import numpy as np

def HaganImpliedVolatility(K,T,f,alpha,beta,rho,gamma):
    """
    Aproximación asintótica de Hagan et al. (2002) para la volatilidad
    implícita Black–Scholes bajo el modelo SABR.   

    Hagan et al. derivan una expansión asintótica que proporciona una
    expresión cerrada aproximada para la volatilidad implícita
    Black–Scholes σ_impl(K,T).

    Interpretación de los parámetros en el modelo.
    ----------------------------------------------
        - alpha: Controla el nivel general de la volatilidad (magnitud de sonrisa)
        - beta:  Controla la forma estructural (lognormal si beta = 1, normal si beta = 0)
        - rho:   Deterimna la asimetría (skew) de la sonrisa
        - gamma: Controla la curvatura (smile)

    Args:
        K (float np.array): Strike
        T (float): Tiempo a vencimiento
        f (float): Forward F(t_0,T)
        alpha (float): Nivel inivial de volatilidad 
        beta (float): Elasticidad del modelo
        rho (float): Correlación
        gamma (float): Volatilidad de la volatilidad 

    Returns:
        np.darray: Volatilidad implícita de Black-Scholes sigma_impl(K,T)
    """

    # We make sure that the input is of array type

    if type(K) == float:
        K = np.array([K])
    if K is not np.array:
        K = np.array(K).reshape([len(K),1])
    
    # The strike prices cannot be too close to 0

    K[K<1e-10] = 1e-10
    
    z        = gamma/alpha*np.power(f*K,(1.0-beta)/2.0)*np.log(f/K)
    x_z      = np.log((np.sqrt(1.0-2.0*rho*z+z*z)+z-rho)/(1.0-rho))
    A        = alpha/(np.power(f*K,((1.0-beta)/2.0))*(1.0+np.power(1.0-beta,2.0)/24.0*
                               np.power(np.log(f/K),2.0)+np.power((1.0-beta),4.0)/1920.0*
                               np.power(np.log(f/K),4.0)))
    B1       = 1.0 + (np.power((1.0-beta),2.0)/24.0*alpha*alpha/(np.power((f*K),
                1-beta))+1/4*(rho*beta*gamma*alpha)/(np.power((f*K),
                             ((1.0-beta)/2.0)))+(2.0-3.0*rho*rho)/24.0*gamma*gamma)*T
    impVol   = A*(z/x_z) * B1
    B2 = 1.0 + (np.power(1.0-beta,2.0)/24.0*alpha*alpha/
                (np.power(f,2.0-2.0*beta))+1.0/4.0*(rho*beta*gamma*
                alpha)/np.power(f,(1.0-beta))+(2.0-3.0*rho*rho)/24.0*gamma*gamma)*T

    # Special treatment of ATM strike value

    impVol[np.where(K==f)] = alpha / np.power(f,(1-beta)) * B2
    return impVol

