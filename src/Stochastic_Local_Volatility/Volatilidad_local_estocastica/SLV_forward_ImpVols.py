"""
Este script compara el modelo de Volatilidad Local Estocástica (SLV) con
Volatilidad Local (LV) y con Heston en la valoración de opciones vanilla
europeas (vencimiento T2) y opciones forward-start (fechas T1 y T2).

A partir de un “mercado” de referencia (Heston), se calcula la volatilidad
local vía derivadas numéricas (tipo Dupire) y se simulan trayectorias por
Monte Carlo (Euler o AES), estimando E[V|S] mediante binning para construir
la función de apalancamiento SLV. Finalmente se obtienen precios y volatilidades implícitas
y se representan las curvas comparativas.
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import fsolve
import warnings

warnings.filterwarnings('ignore')

# ---- Core project imports ----
from core.monteCarlo import MCprice
from core.COS_method import Carr_Madan_Call
from core.interpolation import ImpliedVolatility_Vectorized
from core.types import OptionType

from volatilidad_estocastica.Heston_model import chf_heston, CIR_Sample
from Volatilidad_local_estocastica.SLV_bin_method import getEVBinMethod
from Volatilidad_local_estocastica.forward_start import (
    ChFHestonForward,
    CallPutOptionPriceCOSMthd_FrwdStart,
    BS_Call_Option_Price_FrwdStart,
    ImpliedVolatilityFrwdStart,
)


def SLV_forward_ImpVols():
    """
    Ejecuta la simulación y comparación entre los modelos SLV, LV y Heston
    para opciones europeas y forward-start.

    La función:
    - define los parámetros del mercado de referencia y del modelo calibrado,
    - construye las funciones características necesarias,
    - calcula precios analíticos de referencia,
    - obtiene una volatilidad local mediante diferencias finitas,
    - simula trayectorias bajo SLV y LV por Monte Carlo,
    - calcula precios e implied vols,
    - y genera las figuras comparativas finales.

    Returns
    -------
    None
        La función no devuelve valores explícitos; muestra gráficas con las
        volatilidades implícitas obtenidas.
    """
    StepsYr = 10
    method  = 'Euler'   # 'AES' or 'Euler'

    # --- Input parameters ---
    S0 = 1.0
    M0 = 1.0
    r  = 0.0
    t0 = 0.0
    T1 = 2.0
    T2 = 5.0     

    # Market ("true") Heston parameters
    gamma = 0.5
    kappa = 0.3
    rho   = -0.6
    vBar  = 0.05
    v0    = 0.04

    # Poorly-calibrated model parameters (p = 0.25 — Moderate)
    p          = 0.25
    gammaModel = (1 - p) * gamma    # 0.375
    kappaModel = (1 + p) * kappa    # 0.375
    rhoModel   = (1 + p) * rho      # -0.75
    vBarModel  = (1 - p) * vBar     # 0.0375
    v0Model    = (1 + p) * v0       # 0.05

    # --- Settings ---
    NoBins    = 20
    NoOfPaths = int(5e4)
    NoOfSteps = int(StepsYr * T2)
    dt        = 1.0 / StepsYr

    delta = np.arange(-3.5, 4.0, 0.5)
    k     = S0 * np.exp(0.1 * np.sqrt(T2) * delta)
    doneFlag = 0

    # --- Characteristic functions ---
    def cf(u, T):
        return chf_heston(u, T, kappa, vBar, gamma, rho, v0, r)

    def cfHeston(u, T):
        return chf_heston(u, T, kappaModel, vBarModel, gammaModel, rhoModel, v0Model, r)

    def cfHestonFrwd(u):
        return ChFHestonForward(u, T1, T2, kappaModel, vBarModel, gammaModel, rhoModel, v0Model, r)

    # Market call price via Carr-Madan
    def Vc(t, x):
        return Carr_Madan_Call(cf, t, x, np.log(S0))

    # --- Forward-start price (Heston): COS method ---
    optPriceHestonFrwd = CallPutOptionPriceCOSMthd_FrwdStart(
        cfHestonFrwd, OptionType.CALL, r, T1, T2, k - 1.0, 2000, 5
    )

    # --- Local-vol derivatives (diferencias finitas) ---
    bump_T = 1e-4
    bump_K = 1e-4

    def dC_dT(T, S):
        return (Vc(T + bump_T, S) - Vc(T, S)) / bump_T

    def dC_dK(T, S):
        return (Vc(T, S + bump_K) - Vc(T, S - bump_K)) / (2.0 * bump_K)

    def d2C_dK2(T, S):
        return (Vc(T, S + bump_K) + Vc(T, S - bump_K) - 2.0 * Vc(T, S)) / bump_K**2

    # --- Monte Carlo ---
    NoSeeds = 1

    ImpliedVol_T2Sum    = 0.0
    ImpliedVol_T2Sum_LV = 0.0
    EuropPriceSum       = 0.0
    EuropPriceSum_LV    = 0.0

    fig1, ax1 = plt.subplots(figsize=(7, 5))   # forward-start figure

    for s in range(1, NoSeeds + 1):
        np.random.seed(s)
        t = t0

        S      = np.ones(NoOfPaths) * S0
        S_LV   = S.copy()
        x      = np.ones(NoOfPaths) * np.log(S0)
        M      = np.ones(NoOfPaths) * M0
        v      = np.ones(NoOfPaths) * v0Model
        v_old  = v.copy()
        v_new  = v.copy()   # initialise (used in getEVBinMethod at step 2+)
        sigma_old = np.ones(NoOfPaths)

        S_T1    = S.copy()
        S_T1_LV = S_LV.copy()
        doneFlag = 0
        Tvec    = np.zeros(2)

        for i in range(1, int(NoOfSteps) + 1):
            print(f'seed number = {s:.0f} and time= {t:2.2f} of t_max = {T2:2.2f}')

            t_real = t

            # Flat extrapolation at t=0
            if i == 1:
                t_adj = 1.0 / NoOfSteps
                t     = t_adj

            # ---- Local vol: SLV ----
            nominator  = dC_dT(t, S) + r * S * dC_dK(t, S)
            d2CdK2     = np.maximum(np.abs(d2C_dK2(t, S)), 1e-7)
            sigma_det  = np.sqrt(np.abs(nominator) / (0.5 * S**2 * d2CdK2))

            # ---- Local vol: pure LV — MATLAB uses S (not S_LV) in r*S term ----
            nominator_LV  = dC_dT(t, S_LV) + r * S * dC_dK(t, S_LV)
            d2CdK2_LV     = np.maximum(np.abs(d2C_dK2(t, S_LV)), 1e-7)
            sigma_det_LV  = np.sqrt(np.abs(nominator_LV) / (0.5 * S_LV**2 * d2CdK2_LV))

            # ---- E[V|S] via bin method ----
            if t_real != 0:
                EV        = getEVBinMethod(S, v_new, NoBins)
                sigma_new = sigma_det / np.sqrt(EV)
            else:
                # v0 (market) at t=0
                sigma_new = sigma_det / np.sqrt(v0)
                sigma_old = sigma_new.copy()

            # ---- Simulate one dt ----
            if method == 'AES':
                Z_x   = np.random.randn(NoOfPaths)
                Z_x   = (Z_x - Z_x.mean()) / Z_x.std()
                v_new = CIR_Sample(NoOfPaths, kappaModel, gammaModel, vBarModel, 0, dt, v_old)

                x = (x
                     + r * dt
                     - 0.5 * sigma_old**2 * v_old * dt
                     + (rhoModel * sigma_old / gammaModel)
                       * (v_new - v_old - kappaModel * vBarModel * dt + kappaModel * v_old * dt)
                     + np.sqrt(1.0 - rhoModel**2) * Z_x * np.sqrt(sigma_old**2 * v_old * dt))

                S_LV = S_LV * np.exp(
                    (r - 0.5 * sigma_det_LV**2) * dt + sigma_det_LV * np.sqrt(dt) * Z_x
                )

            else:   # Euler
                Z_x = np.random.randn(NoOfPaths)
                Z_v = np.random.randn(NoOfPaths)
                Z_x = (Z_x - Z_x.mean()) / Z_x.std()
                Z_v = (Z_v - Z_v.mean()) / Z_v.std()
                Z_v = rhoModel * Z_x + np.sqrt(1.0 - rhoModel**2) * Z_v

                # MATLAB Euler uses v (current) in the drift, then updates v
                x     = x + (r - 0.5 * sigma_old**2 * v) * dt + sigma_old * np.sqrt(v * dt) * Z_x
                v     = np.maximum(
                    v + kappaModel * (vBarModel - v) * dt + gammaModel * np.sqrt(v * dt) * Z_v, 0.0
                )
                v_new = v.copy()

                S_LV = S_LV * np.exp(
                    (r - 0.5 * sigma_det_LV**2) * dt + sigma_det_LV * np.sqrt(dt) * Z_x
                )

            M   = M + r * M * dt
            S   = np.exp(x)

            # Moment matching
            a2_SLV = (S0 / M0 - np.mean(S / M)) / np.mean(1.0 / M)
            S      = np.maximum(S + a2_SLV, 1e-4)

            a2_LV  = (S0 / M0 - np.mean(S_LV / M)) / np.mean(1.0 / M)
            S_LV   = np.maximum(S_LV + a2_LV, 1e-4)

            if i == 1:
                t = t_real

            sigma_old = sigma_new.copy()
            v_old     = v_new.copy()
            t         = t + dt

            # Snapshot at T1
            if abs(t - T1 - dt) < (1 + dt) * dt and doneFlag == 0:
                S_T1    = S.copy()
                S_T1_LV = S_LV.copy()
                doneFlag = 1
                Tvec[0] = t

        Tvec[1] = t

        # --- Option prices ---
        EuropPriceT2    = MCprice(k, S,    Tvec[1], r)
        EuropPriceT2_LV = MCprice(k, S_LV, Tvec[1], r)

        FrwdStartOpt    = MCprice(k, S    / S_T1,    Tvec[1], r)
        FrwdStartOpt_LV = MCprice(k, S_LV / S_T1_LV, Tvec[1], r)

        # --- Implied vols ---
        ImpliedVol_T2    = ImpliedVolatility_Vectorized(OptionType.CALL, EuropPriceT2,    k, T2, S0, r)
        ImpliedVol_T2_LV = ImpliedVolatility_Vectorized(OptionType.CALL, EuropPriceT2_LV, k, T2, S0, r)

        ImpliedVol_FrwdStart        = ImpliedVolatilityFrwdStart(FrwdStartOpt,          k - 1.0, T1, T2, r, 0.3)
        ImpliedVol_FrwdStart_LV     = ImpliedVolatilityFrwdStart(FrwdStartOpt_LV,       k - 1.0, T1, T2, r, 0.3)
        ImpliedVol_FrwdStart_Heston = ImpliedVolatilityFrwdStart(optPriceHestonFrwd,    k - 1.0, T1, T2, r, 0.3)

        # Figure 1 — forward-start (plotted per seed, with hold-on)
        ax1.plot(k, ImpliedVol_FrwdStart        * 100, label='SLV'   if s == 1 else '_')
        ax1.plot(k, ImpliedVol_FrwdStart_LV     * 100, label='LV'    if s == 1 else '_')
        ax1.plot(k, ImpliedVol_FrwdStart_Heston * 100, label='Heston'if s == 1 else '_')

        # Accumulate
        ImpliedVol_T2Sum    += ImpliedVol_T2
        ImpliedVol_T2Sum_LV += ImpliedVol_T2_LV
        EuropPriceSum       += EuropPriceT2
        EuropPriceSum_LV    += EuropPriceT2_LV

    # --- Averages ---
    ImpliedVol_T2    = ImpliedVol_T2Sum    / NoSeeds
    ImpliedVol_T2_LV = ImpliedVol_T2Sum_LV / NoSeeds

    # --- Analytic benchmarks ---
    EuropPriceT2_Market = Carr_Madan_Call(cf,      T2, k, np.log(S0))
    EuropPriceT2_Heston = Carr_Madan_Call(cfHeston, T2, k, np.log(S0))

    ImpliedVol_T2_Market = ImpliedVolatility_Vectorized(OptionType.CALL, EuropPriceT2_Market, k, T2, S0, r)
    ImpliedVol_T2_Heston = ImpliedVolatility_Vectorized(OptionType.CALL, EuropPriceT2_Heston, k, T2, S0, r)

    # ================================================================
    # Figure 1 — forward-start 
    # ================================================================
    ax1.set_xlabel('K')
    ax1.set_ylabel('volatilidad implícita [%]')
    ax1.set_title(f'Forward-start IV — T₁={T1:.0f}, T₂={T2:.0f}')
    ax1.legend()
    ax1.grid(True)
    ax1.set_facecolor('white')
    fig1.patch.set_facecolor('white')

    # ================================================================
    # Figure 2 — European call 
    # ================================================================
    YMatrix = np.column_stack([
        ImpliedVol_T2_Market * 100,
        ImpliedVol_T2_LV    * 100,
        ImpliedVol_T2       * 100,
        ImpliedVol_T2_Heston* 100,
    ])
    MakeFigure(k, YMatrix, ['Market', 'LV', 'SLV', 'Heston'],
               f'Implied volatility at T2={T2:.0f}')

    plt.show()


def MakeFigure(X1, YMatrix1, argLegend, titleIn):
    """
    Genera una figura comparativa de volatilidades implícitas.

    Args:
        X1 (array_like): Vector de strikes.
        YMatrix1 (ndarray): Matriz cuyas columnas contienen las curvas a representar.
        argLegend (list[str]): Lista de etiquetas para la leyenda.
        titleIn (str): Título de la figura.

    Returns:
        fig, ax: Figura y ejes de la representación.
    """
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    ax.grid(True)
    ax.box = True   # equivalent to box(axes1,'on')

    markers    = ['D', 's', 'o', None]
    linestyles = ['-', '-.', '-.', '-']

    for i in range(YMatrix1.shape[1]):
        lbl = argLegend[i] if i < len(argLegend) else ''
        m   = markers[i]    if i < len(markers)    else None
        ls  = linestyles[i] if i < len(linestyles) else '-'
        ax.plot(X1, YMatrix1[:, i], marker=m, linestyle=ls,
                linewidth=1.5, label=lbl)

    ax.set_xlabel('K')
    ax.set_ylabel('implied volatility [%]')
    ax.set_title(titleIn)
    ax.legend()
    return fig, ax


if __name__ == "__main__":
    SLV_forward_ImpVols()