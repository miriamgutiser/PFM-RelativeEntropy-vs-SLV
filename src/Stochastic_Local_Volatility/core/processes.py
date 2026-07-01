"""
===============================================================================
Procesos
===============================================================================
Este módulo contiene funciones para la simulación numérica de procesos estocásticos utilizados 
en modelización financiera.
Incluye:
    - Movimiento browniano geométrico (Euler)
    - Condición de Feller
    - Proceso CIR
"""

import numpy as np
import scipy.stats as st

def GeneratePathsGBMEuler(NoOfPaths, NoOfSteps, T, r, sigma, S0):
    """
    Genera trayectorias simuladas del precio de un activo bajo un modelo
    de Movimiento Browniano Geométrico (GBM) mediante el esquema de Euler.
    El proceso continuo es:
        dS_t = r S_t dt + sigma S_t dW_t

    y su discretización por Euler usada aquí es:
        S_{i+1} = S_i + r S_i dt + sigma S_i (W_{i+1} - W_i),
    con:
        W_{i+1} - W_i = sqrt(dt) * Z_i,  Z_i ~ N(0,1)
    Args:
        NoOfPaths (int): Número de trayectorias Monte Carlo a simular.
        NoOfSteps (int): Número de pasos temporales por trayectoria.
        T (float): Tiempo a vencimiento.
        r (float): Tipo de interés (deriva bajo medida de valoración).
        sigma (float): Volatilidad constante del GBM.
        S0 (float): Precio inicial del activo, S(0).

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - S: Array de dimensiones (NoOfPaths, NoOfSteps+1) con las trayectorias simuladas.
                 La columna 0 corresponde a S0.
            - time: Array de longitud (NoOfSteps+1,) con la malla temporal desde 0 hasta T.
    """
    S = np.zeros((NoOfPaths, NoOfSteps + 1), dtype=float)
    S[:, 0] = S0

    Z = np.random.normal(0.0, 1.0, size=(NoOfPaths, NoOfSteps))
    W = np.zeros((NoOfPaths, NoOfSteps + 1), dtype=float)

    dt = T / float(NoOfSteps)
    time = np.zeros(NoOfSteps + 1, dtype=float)

    for i in range(NoOfSteps):
        if NoOfPaths > 1:
            Z[:, i] = (Z[:, i] - np.mean(Z[:, i])) / np.std(Z[:, i])

        W[:, i + 1] = W[:, i] + np.sqrt(dt) * Z[:, i]

        # Euler: S_{i+1} = S_i + r S_i dt + sigma S_i (W_{i+1}-W_i)
        S[:, i + 1] = S[:, i] + r * S[:, i] * dt + sigma * S[:, i] * (W[:, i + 1] - W[:, i])
        time[i + 1] = time[i] + dt

    return S, time

def FellerCondition(kappa,v_bar,gamma):
    """
    Calcula el parámetro de la condición de Feller para un proceso
    de Cos-Ingersoll-Ross (CIR). La condición de Feller se cumple si q_F >= 0.
    Args:
        kappa (float): Velocidad de reversión a la media del proceso.
        v_bar (float): Nivel medio a largo plazo del proceso.
        gamma (float): Coeficiente de difusión (volatilidad de la varianza).

    Returns:
        float: Valor de q_F.
    """
    q_F = 2.0*kappa*v_bar/np.power(gamma,2.0)-1.0
    print("Feller condition, q_F is equal to %.2f" % q_F)
    return q_F
        
# PDF and CDF for CIR process

def CIR_PDF_CDF(x,gamma,kappa,t,s,v_bar,v_s):    
    """
    Calcula la densidad (PDF) y la función de distribución acumulada (CDF/FDA)
    de un proceso CIR en el instante t, condicionada al valor inicial V_s = v_s.
    Args:
        x (array_like): Puntos (x > 0) donde se evalúan la PDF y la CDF de V_t.
        gamma (float): Coeficiente de difusión del CIR (volatilidad de la varianza).
        kappa (float): Velocidad de reversión a la media.
        t (float): Instante final (t > s) en el que se evalúa la distribución.
        s (float):  Instante inicial/condicionante.
        v_bar (float): Media a largo plazo del proceso (nivel de reversión).
        v_s (float): Valor inicial/condición del proceso en el instante s (V_s = v_s).

    Returns:
        dict: Diccionario con:
                - PDF: array con los valores de la densidad f_{V_t}(x)
                - CDF: array con los valores de la acumulada F_{V_t}(x)
    """
    # Escala
    c_s_t      = np.power(gamma,2.0) / (4.0*kappa) * (1.0-np.exp(-kappa*(t-s)));
    # Grados de libertad
    d          = 4.0 * kappa * v_bar / np.power(gamma,2.0)     

    lambda_s_t = 4.0 * kappa * np.exp(-kappa * (t-s)) / (np.power(gamma,2.0) * (1.0 - np.exp(-kappa * (t-s)))) * v_s;
    f_X        = 1.0 / c_s_t * st.ncx2.pdf(x/c_s_t,d,lambda_s_t);
    F_X        = st.ncx2.cdf(x/c_s_t,d,lambda_s_t);
    output = {"pdf":f_X, "cdf":F_X}    
    return output
