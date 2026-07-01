#%%
"""
Comparación entre la volatilidad local derivada de la fórmula de Hagan (SABR)
y los precios obtenidos mediante simulación de Monte Carlo.

A partir de la superficie de volatilidad implícita de SABR se construye la
varianza local σ_LV²(x,t) (fórmula de Dupire) y se simulan trayectorias del
subyacente con dicha volatilidad. Finalmente se recuperan las volatilidades
implícitas de los precios Monte Carlo y se comparan con las de Hagan.

Funciones incluidas:
    - LocalVarianceBasedOnSABR : construye σ_LV²(x,t) a partir de parámetros SABR
    - mainCalculation           : ejecuta la simulación y genera los gráficos
"""


import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
import scipy.optimize as optimize
from Stochastic_Local_Volatility.core.interpolation import ImpliedVolatility
from Stochastic_Local_Volatility.core.hagan_implied_volatility import HaganImpliedVolatility
from Stochastic_Local_Volatility.core.pricing_Black_Scholes import BS_Call_Put_Option_Price
from Stochastic_Local_Volatility.core.monteCarlo import EUOptionPriceFromMCPaths
from common.types import OptionType
from Stochastic_Local_Volatility.volatilidad_local.local_volatility import LocalVarianceBasedOnSABR

def mainCalculation():

    # For the SABR model we take beta =1 and rho =0 (as simplification)

    beta   = 1.0
    rho    = 0.0

    # Other model parameters

    volvol = 0.2
    s0     = 1.0
    T      = 10.0
    r      = 0.05
    alpha  = 0.2
    f_0    = s0*np.exp(r*T)
    CP     = OptionType.CALL

    # Monte Carlo settings

    NoOfPaths = 25000
    NoOfSteps = (int)(100*T)
    
    # We define the market to be driven by Hagan's SABR formula
    # Based on this formula we derive the local volatility/variance 

    sigma     = lambda x,t: HaganImpliedVolatility(x,t,f_0,alpha,beta,rho,volvol)
    
    # Local variance based on the Hagan's SABR formula

    sigmalv2  = LocalVarianceBasedOnSABR(s0,f_0,r,alpha,beta,rho,volvol)
           
    # Monte Carlo simulation

    dt        = T/NoOfSteps
    np.random.seed(4)
    Z         = np.random.normal(0.0,1.0,[NoOfPaths,NoOfSteps])
    S         = np.zeros([NoOfPaths,NoOfSteps+1])
    
    S[:,0]    = s0;
    time = np.zeros([NoOfSteps+1,1])
    
    for i in range(0,NoOfSteps):

        # This condition is necessary as for t=0 we cannot compute implied
        # volatilities

        if time[i]==0.0:
            time[i]=0.0001
        
        print("current time is {0:.2f}".format(float(time[i, 0])))

        # Standarize Normal(0,1)

        Z[:,i]=(Z[:,i]-np.mean(Z[:,i]))/np.std(Z[:,i])
        
        # Compute local volatility

        S_i = np.array(S[:,i]).reshape([NoOfPaths,1])
        temp = sigmalv2(S_i,time[i])
        sig = np.real(temp)
        np.nan_to_num(sig)
        
        # Because of discretizations we may encouter negative variance which
        # is set to 0 here.

        sig=np.maximum(sig,1e-14)
        sigmaLV = np.sqrt(sig)  
        
        # Stock path

        S[:,i+1]=S[:,i] * (1.0 + r*dt + sigmaLV.transpose()*Z[:,i]*np.sqrt(dt))
                       
        # We force that at each time S(t)/M(t) is a martingale

        S[:,i+1]= S[:,i+1] - np.mean(S[:,i+1]) + s0*np.exp(r*time[i])
        
        # Make sure that after moment matching we don't encounter negative stock values

        S[:,i+1]=np.maximum(S[:,i+1],1e-14)
        
        # Adjust time

        time[i+1] = time[i] + dt

    # Plot some results

    K = np.linspace(0.2,5.0,25)
    #c_n = np.array([-1.5, -1.0, -0.5,0.0, 0.5, 1.0, 1.5])
    #K= s0*np.exp(r*T) * np.exp(0.1 * c_n * np.sqrt(T))
    OptPrice = np.zeros([len(K),1])
    IV_Hagan = np.zeros([len(K),1])
    IV_MC = np.zeros([len(K),1])
    for (idx,k) in enumerate(K):
        OptPrice[idx] = EUOptionPriceFromMCPaths(CP,S[:,-1],k,T,r)
        IV_Hagan[idx] = sigma([k],T)*100.0
        IV_MC[idx]    = ImpliedVolatility(CP,OptPrice[idx],k,T,s0,r)*100.0
        
    # Plot the option prices

    plt.figure(1)
    plt.plot(K,OptPrice)
    plt.grid()
    plt.xlabel('strike')
    plt.ylabel('option price')
    
    # Plot the implied volatilities

    plt.figure(2)
    plt.plot(K,IV_Hagan)
    plt.plot(K,IV_MC,'-r')
    plt.grid()
    plt.xlabel('strike')
    plt.ylabel('implied volatility')
    plt.legend(['Hagan','Monte Carlo'])
    plt.axis((float(np.min(K)), float(np.max(K)), 0.0, 40.0))

    plt.show()
mainCalculation()
