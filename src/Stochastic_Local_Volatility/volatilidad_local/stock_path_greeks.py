#%%
"""
Created on Thu Dec 12 2018
GBM path and corresponding Greeks
@author: Lech A. Grzelak
"""



import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
from common.types import OptionType

from mpl_toolkits import mplot3d
from scipy.interpolate import RegularGridInterpolator
from Stochastic_Local_Volatility.core.pricing_Black_Scholes import  BS_Call_Put_Option_Price, BS_Delta, BS_Gamma, BS_Vega
from Stochastic_Local_Volatility.core.interpolation import interpolate_along_path
from Stochastic_Local_Volatility.core.processes import GeneratePathsGBMEuler


def StockPathAndGreeks():
    """
    Simula las trayectorias de un activo bajo GBM (Euler) y genera superficies de precios y griegas Black-Scholes, 
    superponiendo una trayectoria simulada sobre cada superficie mediante interpolación en una malla (t, S).

    Observaciones:
        - Black-Scholes con sigma constante en la valoración y griegas, no dependen de la simulación excepto
          para la trayectoria que se superpone como referencia.
    Returns:
        None
    """
    np.random.seed(43)

    # Monte Carlo settings
    s0 = 10
    noOfPaths = 50
    noOfSteps = 500
    r = 0.05
    sigma = 0.4
    T = 1.001

    # Strike price
    K = 10

    # Path to plot 
    pathId = 6
    pathId_py = pathId - 1

    S, x = GeneratePathsGBMEuler(noOfPaths, noOfSteps, T, r, sigma, s0)

    # Function pointers 
    CallOpt = lambda t, s0_: BS_Call_Put_Option_Price(OptionType.CALL, s0_, [K], sigma, t, T, r)
    PutOpt = lambda t, s0_: BS_Call_Put_Option_Price(OptionType.PUT, s0_, [K], sigma, t, T, r)
    DeltaCall = lambda t, s0_: BS_Delta(OptionType.CALL, s0_, [K], sigma, t, T, r)
    Gamma = lambda t, s0_: BS_Gamma(s0_, [K], sigma, t, T, r)
    Vega = lambda t, s0_: BS_Vega(s0_, [K], sigma, t, T, r)

    # Grid for the stock values
    s0Grid = np.linspace(s0 / 100.0, 1.5 * s0, 50)

    timeGrid = np.arange(0.02, T + 1e-12, 0.025)

    # Prepare empty matrices for storing the results
    callOptM = np.zeros((len(timeGrid), len(s0Grid)), dtype=float)
    putOptM = np.zeros((len(timeGrid), len(s0Grid)), dtype=float)
    deltaCallM = np.zeros((len(timeGrid), len(s0Grid)), dtype=float)
    gammaM = np.zeros((len(timeGrid), len(s0Grid)), dtype=float)
    vegaM = np.zeros((len(timeGrid), len(s0Grid)), dtype=float)
    TM = np.zeros((len(timeGrid), len(s0Grid)), dtype=float)
    s0M = np.zeros((len(timeGrid), len(s0Grid)), dtype=float)

    # Rellenar superficies
    for i in range(len(timeGrid)):
        TM[i, :] = timeGrid[i]
        s0M[i, :] = s0Grid

        callOptM[i, :] = np.squeeze(CallOpt(timeGrid[i], s0Grid))
        putOptM[i, :] = np.squeeze(PutOpt(timeGrid[i], s0Grid))
        deltaCallM[i, :] = np.squeeze(DeltaCall(timeGrid[i], s0Grid))
        gammaM[i, :] = np.squeeze(Gamma(timeGrid[i], s0Grid))
        vegaM[i, :] = np.squeeze(Vega(timeGrid[i], s0Grid))

    # 1) Stock path
    plt.figure(1)
    plt.plot(x, S[pathId_py, :], linewidth=1.5)
    plt.xlabel("t")
    plt.ylabel("S(t)")
    plt.axis([0, T, float(np.min(S[pathId_py, :])), float(np.max(S[pathId_py, :]))])
    plt.grid(True)

    # Helper: interpolación tipo griddedInterpolant(TM,s0M,Surface)
    # RegularGridInterpolator usa (timeGrid, s0Grid) y valores (len(timeGrid), len(s0Grid))
    def plot_surface_with_path(fig_id, Z, title, zlabel):
        fig = plt.figure(fig_id)
        ax = fig.add_subplot(111, projection="3d")
            
        ax.plot_wireframe(
        TM, s0M, Z,
        rstride=1, cstride=1,
        color=(0.75, 0.45, 0.0),   
        linewidth=0.8
    )

        # ax.plot_surface(TM, s0M, Z, linewidth=0, antialiased=True)
        ax.set_xlabel("t")
        ax.set_ylabel("S")
        ax.set_zlabel(zlabel)
        ax.set_title(title)

        V = interpolate_along_path(timeGrid, s0Grid, Z, x, S[pathId_py, :])

        mask = ~np.isnan(V)
        ax.plot3D(x[mask], S[pathId_py, :][mask], V[mask], "r", linewidth=2)
        ax.set_xlim(0.0, float(T))
        ax.set_ylim(0.0, float(np.max(s0Grid)))
        ax.set_zlim(0.0, float(np.nanmax(Z)))

        ax.view_init(elev=25, azim=-120)

        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False

        return ax

    # 2) Call option surface with Monte Carlo path
    plot_surface_with_path(2, callOptM, "Call option", "V")

    # 3) Put option surface with Monte Carlo path
    plot_surface_with_path(3, putOptM, "Put option", "V")

    # 4) Call option delta profile
    plot_surface_with_path(4, deltaCallM, "Delta for a call option", "Delta")

    # 5) Option gamma profile
    plot_surface_with_path(5, gammaM, "Gamma", "Gamma")

    # 6) Option vega profile
    plot_surface_with_path(6, vegaM, "Vega", "Vega")

    plt.show()


if __name__ == "__main__":
    StockPathAndGreeks()
