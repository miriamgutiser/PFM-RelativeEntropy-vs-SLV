import numpy as np
import matplotlib.pyplot as plt

from Stochastic_Local_Volatility.volatilidad_estocastica.Heston_model import chf_heston

from scipy.optimize import fsolve
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')

from Stochastic_Local_Volatility.core.COS_method import CallPutOptionPriceCOSMthd
from Stochastic_Local_Volatility.core.interpolation import ImpliedVolatility
from common.types import OptionType

def heston_model_implied_vols():
    """Main function to analyze Heston model implied volatilities"""
    
    # Heston model parameters
    T = 2
    S0 = 100
    CP = OptionType.CALL
    
    kappa = 0.1
    gamma = 0.1
    vBar = 0.1
    rho = -0.75
    v0 = 0.05
    r = 0.05
    
    # Range of strike prices
    K = np.linspace(40, 200, 25)
    
    # COS method settings
    N = 500
    L = 5
    
    # Effect of gamma
    gamma_v = [0.1, 0.3, 0.5, 0.9]
    iv_m = np.zeros((len(K), len(gamma_v)))
    arg_legend = []
    
    for i, gamma_temp in enumerate(gamma_v):
        # Compute ChF for the Heston model
        cf = lambda u: chf_heston(u, T, kappa, vBar, gamma_temp, rho, v0, r)
        
        # The COS method
        opt_price = CallPutOptionPriceCOSMthd(cf, CP, S0, r, T, K, N, L)
        if hasattr(opt_price, 'diagonal'):
            opt_price = opt_price.diagonal()
        
        for j in range(len(K)):
            strike = float(K[j])
            call = float(opt_price[j])
            iv_m[j, i] = ImpliedVolatility(CP, call, strike, T, S0, r) * 100
        
        arg_legend.append(f'gamma={gamma_temp:.2f}')
    
    make_figure(K, iv_m, arg_legend, 'Effect of γ on implied volatility')
    
    # Effect of kappa
    kappa_v = [0.1, 0.5, 1.0, 2.0]
    iv_m = np.zeros((len(K), len(kappa_v)))
    arg_legend = []
    
    for i, kappa_temp in enumerate(kappa_v):
        cf = lambda u: chf_heston(u, T, kappa_temp, vBar, gamma, rho, v0, r)
        opt_price = CallPutOptionPriceCOSMthd(cf, CP, S0, r, T, K, N, L)
        if hasattr(opt_price, 'diagonal'):
            opt_price = opt_price.diagonal()
        
        for j in range(len(K)):
            strike = float(K[j])
            call = float(opt_price[j])
            iv_m[j, i] = ImpliedVolatility(CP, call, strike, T, S0, r) * 100
        
        arg_legend.append(f'kappa={kappa_temp:.1f}')
    
    make_figure(K, iv_m, arg_legend, 'Effect of κ on implied volatility')
    
    # Effect of kappa -- term structure
    T_mat = np.linspace(0.01, 25, 25)
    kappa_v = [0.1, 0.5, 1.0, 2.0]
    arg_legend = []
    IV = np.zeros((len(T_mat), len(kappa_v)))
    
    for i, kappa_temp in enumerate(kappa_v):
        for idx, T_temp in enumerate(T_mat):
            cf = lambda u: chf_heston(u, T_temp, kappa_temp, vBar, gamma, rho, v0, r)
            val_cos = CallPutOptionPriceCOSMthd(cf, CP, S0, r, T_temp, [S0], N, L)
            val = val_cos[0, 0] if hasattr(val_cos, 'shape') and len(val_cos.shape) == 2 else val_cos[0]
            IV[idx, i] = ImpliedVolatility(CP, float(val), float(S0), T_temp, S0, r) * 100
        
        arg_legend.append(f'kappa={kappa_temp:.2f}')
    
    fig, ax = make_figure(T_mat, IV, arg_legend, 'Implied ATM volatility for parameter κ')
    ax.set_xlabel('τ-time to maturity')
    
    # Effect of v0 -- term structure
    v0_v = [0.1, 0.3, 0.5, 1.0]
    arg_legend = []
    IV = np.zeros((len(T_mat), len(v0_v)))
    
    for i, v0_temp in enumerate(v0_v):
        for idx, T_temp in enumerate(T_mat):
            cf = lambda u: chf_heston(u, T_temp, kappa, vBar, gamma, rho, v0_temp, r)
            val_cos = CallPutOptionPriceCOSMthd(cf, CP, S0, r, T_temp, [S0], N, L)
            val = val_cos[0, 0] if hasattr(val_cos, 'shape') and len(val_cos.shape) == 2 else val_cos[0]
            IV[idx, i] = ImpliedVolatility(CP, float(val), float(S0), T_temp, S0, r) * 100
        
        arg_legend.append(f'v0={v0_temp:.2f}')
    
    fig, ax = make_figure(T_mat, IV, arg_legend, 'Implied ATM volatility for parameter v0')
    ax.set_xlabel('τ-time to maturity')
    
    # Effect of rho
    rho_v = [0.0, -0.25, -0.5, -0.9]
    iv_m = np.zeros((len(K), len(rho_v)))
    arg_legend = []
    
    for i, rho_temp in enumerate(rho_v):
        cf = lambda u: chf_heston(u, T, kappa, vBar, gamma, rho_temp, v0, r)
        opt_price = CallPutOptionPriceCOSMthd(cf, CP, S0, r, T, K, N, L)
        if hasattr(opt_price, 'diagonal'):
            opt_price = opt_price.diagonal()
        
        for j in range(len(K)):
            strike = float(K[j])
            call = float(opt_price[j])
            iv_m[j, i] = ImpliedVolatility(CP, call, strike, T, S0, r) * 100
        
        arg_legend.append(f'rho={rho_temp:.2f}')
    
    make_figure(K, iv_m, arg_legend, 'Effect of ρ on implied volatility')
    
    # Effect of vBar
    vBar_v = [0.03, 0.1, 0.2, 0.3]
    iv_m = np.zeros((len(K), len(vBar_v)))
    arg_legend = []
    
    for i, vBar_temp in enumerate(vBar_v):
        cf = lambda u: chf_heston(u, T, kappa, vBar_temp, gamma, rho, v0, r)
        opt_price = CallPutOptionPriceCOSMthd(cf, CP, S0, r, T, K, N, L)
        if hasattr(opt_price, 'diagonal'):
            opt_price = opt_price.diagonal()
        
        for j in range(len(K)):
            strike = float(K[j])
            call = float(opt_price[j])
            iv_m[j, i] = ImpliedVolatility(CP, call, strike, T, S0, r) * 100
        
        arg_legend.append(f'vbar={vBar_temp:.2f}')
    
    make_figure(K, iv_m, arg_legend, 'Effect of $\\bar{v}$ on implied volatility')




def make_figure(X1, Y_matrix, arg_legend, title_in):
    """Create figure with multiple plots"""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.grid(True)
    
    # Plot styles
    markers = ['d', 's', 'o', '^']
    linestyles = ['-', '-.', '-.', '-']
    
    lines = []
    for i in range(Y_matrix.shape[1]):
        line = ax.plot(X1, Y_matrix[:, i], 
                      marker=markers[i % len(markers)],
                      linestyle=linestyles[i % len(linestyles)],
                      linewidth=1.5,
                      label=arg_legend[i])[0]
        lines.append(line)
    
    ax.set_xlabel('K')
    ax.set_ylabel('implied volatility [%]')
    ax.set_title(title_in)
    ax.legend()
    
    plt.tight_layout()
    plt.show()
    
    return fig, ax


# Run the main function
if __name__ == "__main__":
    heston_model_implied_vols()
