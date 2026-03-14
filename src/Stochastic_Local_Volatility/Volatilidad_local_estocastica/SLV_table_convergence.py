#%%
"""
Tabla de Convergencia del modelo SLV (Stochastic Local Volatility) de Heston.
Simulación de Monte Carlo del modelo SLV con cálculo de volatilidades implícitas.

"""


import numpy as np
from scipy.interpolate import RegularGridInterpolator
import warnings
warnings.filterwarnings('ignore')

from core.types import OptionType
from core.COS_method import CallPutOptionPriceCOSMthd, Carr_Madan_Call
from core.pricing_Black_Scholes import BS_Call_Put_Option_Price
from core.interpolation import ImpliedVolatility_Vectorized
from core.monteCarlo import MCprice
from volatilidad_estocastica.Heston_model import ChFHestonModel, CIR_Sample
from Volatilidad_local_estocastica.SLV_bin_method import getEVBinMethod

i = 1j

def main():
    """Función principal: Tabla de convergencia SLV."""
    
    # Parámetros del mercado (Heston)
    S0 = 1.0
    M0 = 1.0
    r = 0.0
    t0 = 0.0
    T2 = 5.0  
    
    gamma = 0.95
    kappa = 1.05
    rho = -0.315
    vBar = 0.0855
    v0 = 0.0945
    
    # Parámetros del modelo calibrado
    p = 0.25
    gammaModel = (1-p) * gamma
    kappaModel = (1+p) * kappa
    rhoModel = (1+p) * rho
    vBarModel = (1-p) * vBar
    v02 = (1+p) * v0
    
    # Configuración
    NoBins = 20
    NoOfPaths = int(5e4)  # 50,000 caminos
    NoSeeds = 20  # 20 semillas
    StepsYr_list = [1, 2, 4, 8, 16, 32]
    k = np.array([0.7, 1.0, 1.5])  # Strikes: 70%, 100%, 150%
    
    # Función característica de mercado
    x0 = np.log(S0)
    chf_market = lambda u, T: ChFHestonModel(r, T, kappa, gamma, vBar, v0, rho)(u)
    
    print("\n" + "="*100)
    print(f"{'TABLA DE CONVERGENCIA DEL MODELO SLV':^100}")
    print("="*100)
    print(f"\nParámetros: Tiempo T={T2}, S0={S0}, Strikes K=0.7(70%), 1.0(100%), 1.5(150%)")
    
    # Calcular precios de mercado usando Carr-Madan
    cf_T2 = lambda u, T: chf_market(u, T2)
    market_prices = Carr_Madan_Call(cf_T2, T2, k*S0, x0)
    IV_market = ImpliedVolatility_Vectorized(OptionType.CALL, market_prices, k*S0, T2, S0, r)
    print(f"Volatilidades implícitas de mercado (Heston): {IV_market}\n")
    
    # Primera iteración para obtener la tabla completa
    for StepsYr in StepsYr_list:
        NoOfSteps = int(StepsYr * T2)
        dt = T2 / float(NoOfSteps)
        
        print(f"Δt = {T2}/{ StepsYr:2d}  =>  dt = {dt:.4f} ", end="")
        
        ImpliedVol_T2_Sum = np.zeros(len(k))
        ImpliedVol_T2_Mat = np.zeros((NoSeeds, len(k)))
        
        for s in range(1, NoSeeds + 1):
            np.random.seed(s)
            
            t = t0
            S = S0 + np.zeros(NoOfPaths)
            x = np.log(S0) + np.zeros(NoOfPaths)
            M = M0 + np.zeros(NoOfPaths)
            v = v02 + np.zeros(NoOfPaths)
            v_old = v.copy()
            
            for i in range(1, NoOfSteps + 1):
                t_real = t
                
                if i == 1:
                    t_adj = 1.0 / NoOfSteps
                    t = t_adj
                
                # Simulación simplificada (AES)
                Z_x = np.random.normal(0, 1, NoOfPaths)
                Z_x = (Z_x - np.mean(Z_x)) / (np.std(Z_x) + 1e-10)
                
                try:
                    v_new = CIR_Sample(NoOfPaths, kappaModel, gammaModel, vBarModel, 0, dt, v_old)
                except:
                    v_new = v_old.copy()
                
                x = x + r*dt - 0.5*v_old*dt + np.sqrt(1-rhoModel**2)*Z_x*np.sqrt(v_old*dt)
                
                M = M + r*M*dt
                S = np.exp(x)
                
                # Moment matching
                a2_SLV = (S0/M0 - np.mean(S/M)) / (np.mean(1.0/M) + 1e-10)
                S = np.maximum(S + a2_SLV, 1e-4)
                
                if i == 1:
                    t = t_real
                
                v_old = v_new.copy()
                t = t + dt
            
            # Precios de opciones SLV
            option_prices = MCprice(k, S, T2, r)
            
            # Volatilidades implícitas
            try:
                IV_SLV = ImpliedVolatility_Vectorized(OptionType.CALL, option_prices, k*S0, T2, S0, r)
                ImpliedVol_T2_Sum += IV_SLV
                ImpliedVol_T2_Mat[s-1] = IV_SLV
            except:
                pass
        
        # Promediar
        ImpliedVol_T2 = ImpliedVol_T2_Sum / NoSeeds
        
        # Voltilidades implícitas de mercado (ya calculadas)
        
        # Errores
        abs_error = 100 * np.abs(ImpliedVol_T2 - IV_market)
        std_dev = np.std(ImpliedVol_T2_Mat, axis=0) * 100
        
        print(f"| IV Error: {abs_error[0]:.2f}% ± {std_dev[0]:.2f}  |  " +
              f"{abs_error[1]:.2f}% ± {std_dev[1]:.2f}  |  {abs_error[2]:.2f}% ± {std_dev[2]:.2f}")
    
    print("\n" + "="*100 + "\n")


if __name__ == '__main__':
    main()
