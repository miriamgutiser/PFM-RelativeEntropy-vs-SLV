from core.types import OptionType

import numpy as np
from scipy.interpolate import interp1d
from scipy.fftpack import fft

def CallPutOptionPriceCOSMthd(cf,CP,S0,r,tau,K,N,L):
    """
    Calcula el precio de opciones europeas de compra y venta usando el método COS 
    (Characteristic Option Function Series).
    
    Este método utiliza la serie de Fourier del payoff de la opción combinada con la 
    función característica para computar los precios de opciones de forma eficiente.

    Args:
        cf (callable): Función característica de la distribución de log-retornos
        CP (OptionType): Tipo de opción - OptionType.CALL para opciones de compra, 
                        OptionType.PUT para opciones de venta
        S0 (float): Precio inicial del activo subyacente
        r (float): Tasa de interés libre de riesgo (constante)
        tau (float): Tiempo hasta el vencimiento en años
        K (float o np.ndarray): Precio(s) de ejercicio - puede ser un escalar o array de precios
        N (int): Número de términos en la expansión de la serie de Fourier
        L (float): Tamaño del dominio de truncación (típicamente 8 o 10)

    Returns:
        np.ndarray: Precio(s) de la opción correspondiente(s) al (los) precio(s) de ejercicio
    """

    if not isinstance(K, np.ndarray):
        K = np.array(K).reshape([len(K),1])
    
    # Assigning i=sqrt(-1)

    i = 1j 
    x0 = np.log(S0 / K)   
    
    # Truncation domain

    a = 0.0 - L * np.sqrt(tau)
    b = 0.0 + L * np.sqrt(tau)
    
    # Summation from k = 0 to k=N-1

    k = np.linspace(0,N-1,N).reshape([N,1])  
    u = k * np.pi / (b - a);  

    # Determine coefficients for put prices  

    H_k = CallPutCoefficients(CP,a,b,k)   
    mat = np.exp(i * np.outer((x0 - a) , u))
    temp = cf(u) * H_k 
    temp[0] = 0.5 * temp[0]    
    value = np.exp(-r * tau) * K * np.real(mat.dot(temp))     
    return value

# Determine coefficients for put prices 

def CallPutCoefficients(CP,a,b,k):
    if CP==OptionType.CALL:                  
        c = 0.0
        d = b
        coef = Chi_Psi(a,b,c,d,k)
        Chi_k = coef["chi"]
        Psi_k = coef["psi"]
        if a < b and b < 0.0:
            H_k = np.zeros([len(k),1])
        else:
            H_k      = 2.0 / (b - a) * (Chi_k - Psi_k)  
    elif CP==OptionType.PUT:
        c = a
        d = 0.0
        coef = Chi_Psi(a,b,c,d,k)
        Chi_k = coef["chi"]
        Psi_k = coef["psi"]
        H_k      = 2.0 / (b - a) * (- Chi_k + Psi_k)               
    
    return H_k    

def Chi_Psi(a, b, c, d, k):
    psi = np.sin(k * np.pi * (d - a) / (b - a)) - np.sin(k * np.pi * (c - a)/(b - a))
    psi[1:] = psi[1:] * (b - a) / (k[1:] * np.pi)
    psi[0] = d - c
    
    chi = 1.0 / (1.0 + np.power((k * np.pi / (b - a)) , 2.0)) 
    expr1 = np.cos(k * np.pi * (d - a)/(b - a)) * np.exp(d)  - np.cos(k * np.pi 
                  * (c - a) / (b - a)) * np.exp(c)
    expr2 = k * np.pi / (b - a) * np.sin(k * np.pi * 
                        (d - a) / (b - a))   - k * np.pi / (b - a) * np.sin(k 
                        * np.pi * (c - a) / (b - a)) * np.exp(c)
    chi = chi * (expr1 + expr2)
    
    value = {"chi":chi,"psi":psi }
    return value


def Carr_Madan_Call(ChF, T, K, x0):
    """
    Calcula el precio de una opción call europea utilizando el método de Carr-Madan.
    Este método se basa en la transformada de Fourier y la función característica del modelo.
    
    Args:
        ChF (callable): Función característica del modelo de precios (ej. Heston, Black-Scholes).
                        Debe aceptar un array de argumentos complejos u y retornar valores complejos.
        T (float): Tiempo hasta vencimiento de la opción (en años).
        K (array_like): Vector de strikes (precios de ejercicio). Se convierten automáticamente a array.
        x0 (float): Logaritmo del precio inicial del activo (x0 = log(S0)).
    
    Returns:
        ndarray: Vector de precios de opciones call europeas, uno por cada strike en K.
                 Utiliza la transformada rápida de Fourier (FFT) para evaluar la integral de Fourier.
    """
    K = np.asarray(K).flatten()
    K[K < 1e-5] = 1e-5
    
    alpha = 0.75
    c = 300.0
    N_CM = int(2**12)
    
    eta = c / N_CM
    b = np.pi / eta
    u = np.arange(N_CM) * eta
    lambda_param = 2 * np.pi / (N_CM * eta)
    i = 1j
    
    u_new = u - (alpha + 1) * i
    cf_vals = np.exp(1j * u_new * x0) * ChF(u_new, T)
    psi = cf_vals / (alpha**2 + alpha - u**2 + 1j * (2 * alpha + 1) * u)
    
    SimpsonW = 3 + (-1) ** np.arange(1, N_CM + 1) - np.concatenate([[1], np.zeros(N_CM - 1)])
    SimpsonW[-1] = 0
    
    FFTFun = np.exp(1j * b * u) * psi * SimpsonW
    payoff = np.real(eta * np.fft.fft(FFTFun) / 3)
    strike = np.exp(-b + lambda_param * np.arange(N_CM))
    
    f_interp = interp1d(strike, payoff, kind='cubic', fill_value='extrapolate')
    payoff_specific = f_interp(K)
    
    value = np.exp(-np.log(K) * alpha) * payoff_specific / np.pi
    return value