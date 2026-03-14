"""
==========================================================
Heston_model2D.py
==========================================================

Este módulo implementa el método COS 2D para el modelo de Heston.
Contiene las funciones características 2D y métodos numéricos asociados
para valorar opciones usando el método COS bidimensional.

El método COS 2D permite:
- Calcular la función característica 2D del modelo Heston
- Computar densidades marginales y condicionales
- Estimar expectativas condicionales E[V|S]

Referencia:
    Fang, F., & Oosterlee, C. W. (2008). Un novedoso método de valoración 
    de opciones europeas basado en series de coseno de Fourier.
"""

import numpy as np


def ChFHeston2D(u1, u2, v0, ubar, lmbda, rho, eta, r, t, N1, N2):
    """Función característica bidimensional del modelo de Heston.
    
    Calcula la función característica 2D para el precio logarítmico del activo S y 
    la varianza V en el modelo de Heston usando operaciones matriciales.
    
    Args:
        u1 (ndarray): Grid de frecuencias para el precio logarítmico (forma: N1)
        u2 (ndarray): Grid de frecuencias para la varianza (forma: N2)
        v0 (float): Varianza inicial.
        ubar (float): Varianza a largo plazo.
        lmbda (float): Velocidad de reversión a la media (kappa).
        rho (float): Correlación entre los movimientos brownianos.
        eta (float): Volatilidad de la varianza (gamma).
        r (float): Tasa de interés libre de riesgo.
        t (float): Tiempo hasta vencimiento.
        N1 (int): Número de frecuencias en la dimensión del precio logarítmico.
        N2 (int): Número de frecuencias en la dimensión de la varianza.
    
    Returns:
        ndarray: Valores de la función característica 2D (forma: N1 x N2)
    """
    # ensure correct shapes for broadcasting
    u1 = np.reshape(u1, (N1, 1))
    u2 = np.reshape(u2, (1, N2))
    beta = lmbda - 1j * u1 * rho * eta              # N1x1
    D = np.sqrt(beta ** 2 + u1 * (1j + u1) * eta ** 2)  # N1x1
    beta_mat = np.tile(beta, (1, N2))               # N1xN2
    D_mat = np.tile(D, (1, N2))                     # N1xN2
    u2_mat = np.tile(u2, (N1, 1))                   # N1xN2
    Gtilde = (beta_mat - D_mat - 1j * u2_mat * eta ** 2) / (
        beta_mat + D_mat - 1j * u2_mat * eta ** 2
    )
    A = (
        lmbda * ubar / eta ** 2 * ((beta_mat - D_mat) * t
        - 2 * np.log((Gtilde * np.exp(-D_mat * t) - 1) / (Gtilde - 1)))
    )
    B = (1 / eta ** 2) * (
        beta_mat - D_mat - Gtilde * (np.exp(-D_mat * t) * (beta_mat + D_mat))
    ) / (1 - Gtilde * np.exp(-D_mat * t))
    A = A + (1j * u1) * r * t
    phi = np.exp(A) * np.exp(v0 * B)
    return phi


def ksi(z1, z2, N, a, b):
    """Función auxiliar similar a psi utilizada en la fórmula COS.
    
    Calcula los términos de coseno integrados utilizados en el método COS
    para calcular expectativas condicionales.
    
    Args:
        z1 (float): Límite inferior de integración.
        z2 (float): Límite superior de integración.
        N (int): Número de términos de frecuencia.
        a (float): Límite inferior del dominio de integración.
        b (float): Límite superior del dominio de integración.
    
    Returns:
        ndarray: Vector de términos de coseno integrados (longitud: N)
    """
    k = np.arange(1, N)
    ksitemp = -(
        (-b + a) * np.cos(np.pi * k * (-z1 + a) / (-b + a))
        + (b - a) * np.cos(np.pi * k * (-z2 + a) / (-b + a))
        + np.pi * k * (
            -z1 * np.sin(np.pi * k * (-z1 + a) / (-b + a))
            + np.sin(np.pi * k * (-z2 + a) / (-b + a)) * z2
        )
    ) * (-b + a) / (k ** 2) / np.pi ** 2
    ksi0 = z2 ** 2 / 2 - z1 ** 2 / 2
    ksivector = np.concatenate(([ksi0], ksitemp))
    return ksivector


def COSMethod2DHeston(x0, v0, rho_xv, kappa, vBar, gamma_c, T, r):
    """Retorna grids y expectativas condicionales por el método COS 2D.
    
    Implementa el método de Fourier-Coseno bidimensional para el modelo de Heston 
    para calcular:
    - La densidad conjunta de (S, V) en el tiempo T
    - La expectativa condicional E[V|S]
    
    Este método es más eficiente y preciso que Monte Carlo para 
    calcular expectativas condicionales.
    
    Args:
        x0 (float): Precio logarítmico inicial del activo (log(S_0))
        v0 (float): Varianza inicial.
        rho_xv (float): Correlación entre el precio logarítmico y la varianza.
        kappa (float): Velocidad de reversión a la media.
        vBar (float): Varianza a largo plazo.
        gamma_c (float): Volatilidad de la varianza.
        T (float): Tiempo hasta vencimiento.
        r (float): Tasa de interés libre de riesgo.
    
    Returns:
        tuple: (S, v, EVS, f) donde:
            - S (ndarray): Grid de precios del activo
            - v (ndarray): Grid de varianza
            - EVS (ndarray): Expectativa condicional E[V|S] en el grid S
            - f (ndarray): Densidad conjunta f(S, V) en el grid 2D
    """
    L = 12
    c1 = (
        r * T
        + (1 - np.exp(-kappa * T)) * (vBar - v0) / (2 * kappa)
        - 0.5 * vBar * T
    )
    c2alt = vBar * (1 + gamma_c) * T
    a1 = x0 + c1 - L * abs(np.sqrt(c2alt))
    b1 = x0 + c1 + L * abs(np.sqrt(c2alt))
    a2 = 0
    b2 = 0.5
    N1 = int(1e3)
    N2 = int(1e3)
    k1 = np.arange(N1)
    k2 = np.arange(N2)
    omega1 = k1 * np.pi / (b1 - a1)
    omega2 = k2 * np.pi / (b2 - a2)
    phiplus = ChFHeston2D(
        omega1[:, None], omega2[None, :], v0, vBar, kappa, rho_xv, gamma_c, r, T, N1, N2
    ) * np.exp(1j * omega1[:, None] * x0)
    Fplus = np.real(
        phiplus
        * np.exp(-1j * a1 * omega1[:, None])
        * np.exp(-1j * a2 * omega2[None, :])
    )
    phimin = ChFHeston2D(
        omega1[:, None], -omega2[None, :], v0, vBar, kappa, rho_xv, gamma_c, r, T, N1, N2
    ) * np.exp(1j * omega1[:, None] * x0)
    Fmin = np.real(
        phimin
        * np.exp(-1j * a1 * omega1[:, None])
        * np.exp(-1j * a2 * (-omega2)[None, :])
    )
    Fplus[0, 0] = 1
    Fmin[0, 0] = 1
    F = 0.5 * (Fplus + Fmin)
    F[:, 0] = 0.5 * F[:, 0]
    F[0, :] = 0.5 * F[0, :]
    xgrid = np.linspace(a1, b1, 151)
    ygrid = np.linspace(a2, b2, 101)
    cosx = np.cos(np.outer(omega1, xgrid - a1))
    cosy = np.cos(np.outer(ygrid - a2, omega2)).T
    S = np.exp(xgrid)
    v = ygrid
    Fx = Fplus[:, 0]
    Fx[0] = 0.5 * Fx[0]
    fx = 2 / (b1 - a1) * np.dot(np.cos(np.outer(xgrid - a1, omega1)), Fx)
    # compute 2D density (note the orientation of `cosy`)
    f = cosx.T.dot(F).dot(cosy)
    f = 2 / (b1 - a1) * 2 / (b2 - a2) * f
    # conditional expectation vector
    coef = 2 / (b1 - a1) * 2 / (b2 - a2)
    ksi_vec = ksi(a2, b2, N2, a2, b2).reshape(-1, 1)  # N2 x 1
    numerator = cosx.T.dot(F).dot(ksi_vec)            # len(xgrid) x 1
    EVS = coef * numerator / fx.reshape(-1, 1)        # ensure fx is column
    return S, v, EVS.flatten(), f
