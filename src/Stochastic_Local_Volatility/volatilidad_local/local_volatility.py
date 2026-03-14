"""
Herramientas para la construcción de la volatilidad local a partir de la
superficie de volatilidad implícita generada por el modelo SABR (Hagan et al.).

Implementa la fórmula de Dupire expresada en términos de la varianza total

    ω(x,t) = σ_impl²(x,t)·t,

calculando sus derivadas parciales mediante diferencias finitas centrales.

Funciones incluidas:
    - LocalVarianceBasedOnSABR : calcula σ_LV²(x,t) a partir de parámetros SABR
"""

import numpy as np
from core.diferencias_finitas import central_diff, central_diff2
from core.hagan_implied_volatility import HaganImpliedVolatility

def LocalVarianceBasedOnSABR(s0,frwd,r,alpha,beta,rho,volvol):
    """
    Construye una aproximación de la varianza local σ_LV^2(x,t) a partir de la
    volatilidad implícita σ_impl(x,t) generada por SABR (Hagan).
    Args:
        s0 (float): spot S0 (para el término log(x/(s0 e^{rt})))
        frwd (float):  forward F (entrada de Hagan)
        r (float): Tipo interés libre de riesgo
        alpha (float): Nivel inivial de volatilidad (SABR)
        beta (float): Elasticidad del modelo (SABR)
        rho (float): Correlación (SABR)
        volvol (float): gamma en SABR, volatilidad de volatilidad

    Returns:
        callable[[float, float], float]: Función sigmalv2(x,t) que devuelve la varianza local sigma_LV^2(x,t)
                  aproximada asociada a la superficie implícita SABR, donde x representa el strike (K)

    Notes:
        - dt, dx: pasos de diferencias finitas
     """

    # Define shock size for approximating derivatives

    dt =0.001
    dx =0.001

    # Function for Hagan's implied volatility approximation 

    sigma =lambda x,t: HaganImpliedVolatility(x,t,frwd,alpha,beta,rho,volvol)

    # Derivatives

    dsigmadt   = lambda x,t: central_diff(lambda tau: sigma(x,tau), t, dt)
    dsigmadx   = lambda x,t: central_diff(lambda xx: sigma(xx,t), x, dx)
    d2sigmadx2 = lambda x,t: central_diff2(lambda xx: sigma(xx,t), x, dx)
    omega      = lambda x,t: sigma(x,t)*sigma(x,t)*t
    domegadt   = lambda x,t: sigma(x,t)**2.0 + 2.0*t*sigma(x,t)*dsigmadt(x,t)
    domegadx   = lambda x,t: 2.0*t*sigma(x,t)*dsigmadx(x,t)
    #d2omegadx2 = lambda x,t: 2.0*t*(dsigmadx(x,t))**2.0 + 2.0*t*sigma(x,t)*d2sigmadx2(x,t)
    d2omegadx2 = lambda x,t: 2.0*t*np.power(dsigmadx(x,t),2.0) + 2.0*t*sigma(x,t)*d2sigmadx2(x,t)

    term1    = lambda x,t:  1.0+x*domegadx(x,t)*(0.5 - np.log(x/(s0*np.exp(r*t))) / omega(x,t))    
    term2    = lambda x,t:  0.5*np.power(x,2.0)*d2omegadx2(x,t)
    term3    = lambda x,t:  0.5*np.power(x,2.0)*np.power(domegadx(x,t),2.0)*(-1.0/8.0-1.0/(2.0*omega(x,t))\
            +np.log(x/(s0*np.exp(r*t)))*np.log(x/(s0*np.exp(r*t)))/(2*omega(x,t)*omega(x,t)))

    # Final expression for local variance

    sigmalv2 = lambda x,t:(domegadt(x,t)+r*x*domegadx(x,t))/(term1(x,t)+term2(x,t)+term3(x,t))
    return sigmalv2