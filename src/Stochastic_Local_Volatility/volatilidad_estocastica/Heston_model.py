"""
==========================================================
Heston_model.py
==========================================================

Este módulo implementa la función característica del modelo de
volatilidad estocástica de Heston.

El modelo de Heston está definido por el sistema:

    dS_t = r S_t dt + sqrt(V_t) S_t dW_t^S
    dV_t = kappa (vbar - V_t) dt + gamma sqrt(V_t) dW_t^V

con:

    corr(dW_t^S, dW_t^V) = rho

La función característica de log(S_T) bajo la medida neutral al riesgo
admite una expresión cerrada, lo que permite valorar opciones europeas
mediante métodos de transformadas (COS, Fourier inversion, etc.).

Devuelve la función característica φ(u) que puede ser
utilizada por métodos de pricing como el método COS.
"""

import numpy as np

def ChFHestonModel(r, tau, kappa, gamma, vbar, v0, rho):
    """
    Función característica del modelo de Heston bajo la medida neutral de riesgo. 
    Args:
        r (float): Tipo de interés libre de riesgo.
        tau (float): Tiempo hasta vencimiento (tau = T - t)
        kappa (float): Velocidad de reversión a la media del proceso de varianza.
        gamma (float): Volatilidad de la varianza (Volatilidad de la volatilidad)
        vbar (float): Media a largo plazo de la varianza.
        v0 (float): Varianza inicial V_0
        rho (float): Correlación entre el movimiento browniano del activo 
                     y el de la varianza. 

    Returns:
        callable: Función característica φ(u), evaluable para valores complejos u.
                  Esta función puede emplearse en métodos de pricing como el
                  método COS o transformadas de Fourier.
    """
    i = 1j

    D1 = lambda u: np.sqrt(np.power(kappa-gamma*rho*i*u,2)+(u*u+i*u)*gamma*gamma)
    g  = lambda u: (kappa-gamma*rho*i*u-D1(u))/(kappa-gamma*rho*i*u+D1(u))
    C  = lambda u: (1.0-np.exp(-D1(u)*tau))/(gamma*gamma*(1.0-g(u)*np.exp(-D1(u)*tau)))\
        *(kappa-gamma*rho*i*u-D1(u))

    # Note that we exclude the term -r*tau, as the discounting is performed in the COS method

    A  = lambda u: r * i*u *tau + kappa*vbar*tau/gamma/gamma *(kappa-gamma*rho*i*u-D1(u))\
        - 2*kappa*vbar/gamma/gamma*np.log((1.0-g(u)*np.exp(-D1(u)*tau))/(1.0-g(u)))

    # Characteristic function for the Heston model    

    cf = lambda u: np.exp(A(u) + C(u)*v0)
    return cf 

def chf_heston(u, tau, kappa, vBar, gamma, rho, v0, r):
    """
    Calculate the characteristic function for the Heston model.
    Args:
        u (float): Complex argument for the characteristic function.
        tau (float): Time to maturity (T - t)
        kappa (float):  Speed of mean reversion of the variance process.
        vBar (float): Long-term variance.
        gamma (float): Volatility of variance.
        rho (float): Correlation between the Brownian motions.
        v0 (float): Initial variance.
        r (float): Risk-free rate.

    Returns:
        float: The characteristic function value.
    """
    i = 1j
    
    # Functions D_1 and g
    D_1 = np.sqrt((kappa - i * rho * gamma * u)**2 + (u**2 + i * u) * gamma**2)
    g = (kappa - i * rho * gamma * u - D_1) / (kappa - i * rho * gamma * u + D_1)
    
    # Complex-valued functions A and C
    C = (1 / gamma**2) * (1 - np.exp(-D_1 * tau)) / (1 - g * np.exp(-D_1 * tau)) * (kappa - gamma * rho * i * u - D_1)
    A = (i * u * r * tau + kappa * vBar * tau / gamma**2 * (kappa - gamma * rho * i * u - D_1) - 
         2 * kappa * vBar / gamma**2 * np.log((1 - g * np.exp(-D_1 * tau)) / (1 - g)))
    
    # ChF for the Heston model
    cf = np.exp(A + C * v0)
    return cf

def CIR_Sample(NoOfPaths, kappa, gamma, vbar, s, t, v_s):
    """Exact sampler for the Cox-Ingersoll-Ross variance process.

    Parameters are the same as in the Matlab version, with `s` and `t`
    representing the previous and current times, respectively, and `v_s`
    the variance at time `s`.
    
    Args:
        NoOfPaths (int): Number of paths to generate.
        kappa (float): Speed of mean reversion.
        gamma (float): Volatility of variance.
        vbar (float): Long-term variance.
        s (float): Previous time.
        t (float): Current time.
        v_s (ndarray): Variance at time s.
    
    Returns:
        ndarray: Sampled variance values.
    """
    # See the Matlab code at the bottom of the original script for the
    # corresponding expressions.  The Python `noncentral_chisquare`
    # interface matches the Matlab `ncx2` distribution.
    delta = 4.0 * kappa * vbar / gamma / gamma
    c = 1.0 / (4.0 * kappa) * gamma * gamma * (1.0 - np.exp(-kappa * (t - s)))
    kappaBar = (
        4.0 * kappa * v_s * np.exp(-kappa * (t - s))
        / (gamma * gamma * (1.0 - np.exp(-kappa * (t - s))))
    )
    sample = c * np.random.noncentral_chisquare(delta, kappaBar, NoOfPaths)
    return sample

def GeneratePathsHestonAES(NoOfPaths, NoOfSteps, T, r, S_0, kappa, gamma, rho, vbar, v0):
    """Generate Heston paths using the Almost-Exact Scheme (AES).

    Args:
        NoOfPaths (int): Number of Monte Carlo paths.
        NoOfSteps (int): Number of time steps.
        T (float): Time to maturity.
        r (float): Risk-free rate.
        S_0 (float): Initial stock price.
        kappa (float): Speed of mean reversion of variance.
        gamma (float): Volatility of variance.
        rho (float): Correlation between Brownian motions.
        vbar (float): Long-term variance.
        v0 (float): Initial variance.

    Returns
    -------
    S : ndarray
        Asset paths (NoOfPaths x (NoOfSteps+1)).
    time : ndarray
        Time grid (length NoOfSteps+1).
    V : ndarray
        Variance paths (NoOfPaths x (NoOfSteps+1)).
    """
    Z1 = np.random.normal(0.0, 1.0, [NoOfPaths, NoOfSteps])
    W1 = np.zeros([NoOfPaths, NoOfSteps+1])
    V = np.zeros([NoOfPaths, NoOfSteps+1])
    X = np.zeros([NoOfPaths, NoOfSteps+1])
    V[:, 0] = v0
    X[:, 0] = np.log(S_0)
    
    time = np.zeros([NoOfSteps+1])
    dt = T / float(NoOfSteps)
    for i in range(0, NoOfSteps):
        if NoOfPaths > 1:
            Z1[:, i] = (Z1[:, i] - np.mean(Z1[:, i])) / np.std(Z1[:, i])
        W1[:, i+1] = W1[:, i] + np.sqrt(dt) * Z1[:, i]
        # Exact samples for the variance process
        V[:, i+1] = CIR_Sample(NoOfPaths, kappa, gamma, vbar, 0, dt, V[:, i])
        k0 = (r - rho/gamma*kappa*vbar)*dt
        k1 = (rho*kappa/gamma - 0.5)*dt - rho/gamma
        k2 = rho / gamma
        X[:, i+1] = (
            X[:, i] + k0 + k1*V[:, i] + k2 * V[:, i+1]
            + np.sqrt((1.0 - rho**2)*V[:, i]) * (W1[:, i+1] - W1[:, i])
        )
        time[i+1] = time[i] + dt
    S = np.exp(X)
    return S, time, V