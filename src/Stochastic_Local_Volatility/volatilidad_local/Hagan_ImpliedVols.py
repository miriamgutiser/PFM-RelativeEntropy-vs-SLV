#%%
"""
Ilustra el efecto de los parámetros del modelo SABR
sobre la sonrisa de volatilidad implícita Black–Scholes utilizando
la aproximación asintótica de Hagan et al. (2002).

Se generan curvas σ_impl(K,T) variando individualmente:

    1) β  → Elasticidad del modelo (controla la naturaleza CEV/lognormal)
    2) α  → Nivel general de volatilidad
    3) ρ  → Controla la asimetría (skew)
    4) γ  → Controla la curvatura (smile)

"""


import numpy as np
import matplotlib.pyplot as plt
from Stochastic_Local_Volatility.core.hagan_implied_volatility import HaganImpliedVolatility
from common.types import OptionType
# Parameters for Hagan's SABR approximation formula

CP    = OptionType.CALL
beta  = 0.5
alpha = 0.3
rho   = -0.5
gamma = 0.4
f_0   = 1.0
T     = 1.0
K    = np.linspace(0.1,3,100)

# Figure 1, effect of parameter beta on the implied volatility

Beta = [0.1, 0.3, 0.5, 0.75, 1]
legendL  = []
plt.figure(1)
for betaTemp in Beta:
    iv = HaganImpliedVolatility(K,T,f_0,alpha,betaTemp,rho,gamma)*100.0
    plt.plot(K,iv)
    legendL.append(('Beta= {0:.1f}').format(betaTemp))
  
plt.xlabel('K')
plt.ylabel('Volatilidades implícitas [%]')
plt.grid()
plt.legend(legendL)

# Figure 2, effect of alpha on the implied volatility

Alpha = [0.1, 0.2, 0.3, 0.4, 0.5]
legendL  = []
plt.figure(2)
for alphaTemp in Alpha:
    iv = HaganImpliedVolatility(K,T,f_0,alphaTemp,beta,rho,gamma)*100.0
    plt.plot(K,iv)
    legendL.append(('Alpha= {0:.1f}').format(alphaTemp))
  
plt.xlabel('K')
plt.ylabel('Volatilidades implícitas [%]')
plt.grid()
plt.legend(legendL)

# Figure 3, effect of rho on the implied volatility

Rho = [-0.9, -0.45, 0.0, 0.45, 0.9]
legendL  = []
plt.figure(3)
for rhoTemp in Rho:
    iv = HaganImpliedVolatility(K,T,f_0,alpha,beta,rhoTemp,gamma)*100.0
    plt.plot(K,iv)
    legendL.append(('Rho= {0:.1f}').format(rhoTemp))
  
plt.xlabel('K')
plt.ylabel('Volatilidades implícitas [%]')
plt.grid()
plt.legend(legendL)

# Figure 4, effect of gamma on the implied volatility

Gamma = [0.1, 0.3, 0.5, 0.7, 0.9]
legendL  = []
plt.figure(4)
for gammaTemp in Gamma:
    iv = HaganImpliedVolatility(K,T,f_0,alpha,beta,rho,gammaTemp)*100.0
    plt.plot(K,iv)
    legendL.append(('Gamma= {0:.1f}').format(gammaTemp))
  
plt.xlabel('K')
plt.ylabel('Volatilidades implícitas [%]')
plt.grid()
plt.legend(legendL)

plt.show()
