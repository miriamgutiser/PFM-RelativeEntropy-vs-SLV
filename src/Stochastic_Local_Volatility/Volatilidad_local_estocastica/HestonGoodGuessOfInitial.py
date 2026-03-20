"""
==========================================================
HestonGoodGuessOfInitial.py
==========================================================

Compara las volatilidades implícitas del mercado con el modelo de Heston
utilizando suposiciones iniciales de parámetros.

Suposiciones iniciales heurísticas para los parámetros de Heston:
- v0 (varianza inicial): Volatilidad ATM a 1 semana al cuadrado
- vBar (theta, varianza media anual): Volatilidad ATM a 10 años al cuadrado
- kappa (velocidad de reversión a la media): 0.7
- gamma (sigma, volatilidad de la varianza): 0.01
- rho (correlación entre el activo y su volatilidad): -0.1

"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Project imports
from core.COS_method import CallPutOptionPriceCOSMthd
from core.pricing_Black_Scholes import BS_Call_Put_Option_Price
from core.types import OptionType
from volatilidad_estocastica.Heston_model import chf_heston
from scipy.optimize import brentq

def heston_good_guess_initial():
    """
    Compara las volatilidades implícitas del mercado con el modelo de Heston
    utilizando suposiciones iniciales de parámetros.
    """
    # Parameters
    CP = OptionType.CALL
    S0 = 100.0
    r = 0.05
    K = np.array([S0])

    # ATM implied volatilities from the market (Table 8.1)
    tauV = np.array([1.0/48.0, 1.0/12.0, 1.0/2.0, 1.0, 2.0, 5.0, 8.0, 10.0]) 
    ATMVolMrkt = np.array([5.001, 5.070, 5.515, 6.041, 6.196, 6.639, 7.015, 7.221]) / 100.0
        
    # COS method settings
    N = 5000
    L = 5
            
    # Heston model parameters (Initial Guesses)
    kappa = 0.7
    gamma = 0.01
    rho = -0.1
        
    # Heston model parameters based on the ATM implied volatilities
    v0 = ATMVolMrkt[0]**2 
    vBar = ATMVolMrkt[-1]**2

    print(f"Using Guesses: v0={v0:.6f}, vBar={vBar:.6f}, kappa={kappa}, gamma={gamma}, rho={rho}")

    IV = np.zeros(len(tauV))
    for i in range(len(tauV)):
        T = tauV[i]

        # Characteristic function of the Heston model
        cf = lambda u: chf_heston(u, T, kappa, vBar, gamma, rho, v0, r)
        
        # CallPutOptionPriceCOSMthd returns a 2D array if K is a list/array
        optionPrice = CallPutOptionPriceCOSMthd(cf, CP, S0, r, T, K, N, L)
        
        # Extract scalar value from result
        if hasattr(optionPrice, 'item'):
            price = optionPrice.item()
        elif isinstance(optionPrice, np.ndarray):
            price = optionPrice.flatten()[0]
        else:
            price = optionPrice

        # Calculate implied volatility
        intrinsic = max(S0 - S0 * np.exp(-r * T), 0.0) # K = S0 al ser ATM
        safe_price = np.clip(price, intrinsic + 1e-12, S0 - 1e-12)
        
        def f(sigma):
            return float(BS_Call_Put_Option_Price(OptionType.CALL, S0, S0, sigma, 0, T, r)) - safe_price
            
        try:
            IV[i] = brentq(f, 1e-6, 10.0, maxiter=300) * 100
        except (ValueError, RuntimeError):
            IV[i] = np.nan

    # Visualization
    plt.figure(figsize=(10, 6))
    plt.plot(tauV, ATMVolMrkt * 100, '--r', marker='o', label='Market ATM Vol')
    plt.plot(tauV, IV, 'b', marker='s', label='Heston Model (Guess)')
    
    plt.grid(True)
    plt.xlabel('maturity, T')
    plt.ylabel('implied volatility [%]')
    plt.title('Heston Model ATM Implied Vol vs Market (Initial Guess)')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    heston_good_guess_initial()
