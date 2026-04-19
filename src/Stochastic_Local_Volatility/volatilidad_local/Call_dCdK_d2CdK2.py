#%%
"""
Valores de opción call en función del tiempo t y el strike K, con su primera y segundas derivadas 
con respecto al strike K, es decir,  dV_c/dK, d^2V_c/dK^2.

- Construye superficies (t, K) de:
    1) Precio Call Black–Scholes
    2) dC/dK (derivada numérica central)
    3) d²C/dK² (segunda derivada numérica central)
"""




import numpy as np
import matplotlib.pyplot as plt
from common.types import OptionType
from Stochastic_Local_Volatility.core.pricing_Black_Scholes import BS_Call_Put_Option_Price
from Stochastic_Local_Volatility.core.diferencias_finitas import central_diff, central_diff2

def dCdK(S_0, K, sigma, t, T, r, h = 1e-4):
    """
    Aproximación numérica de la primera derivada del precio de una call
    respecto al strike (∂C/∂K) mediante diferencias finitas centrales.   
    Args:
        S_0 (float): Precio spot del activo subyacente en t=0.
        K (float or np.ndarray): Strike(s) sobre los que se evalúa la derivada.
        sigma (float): Volatilidad constante del modelo Black–Scholes.
        t (float): Tiempo actual.
        T (float): Tiempo a vencimiento.
        r (float): Tipo de interés libre de riesgo.
        h (float, optional): Paso de discretización sobre K. Defaults to 1e-4.

    Returns:
        np.ndarray: Aproximación numérica de ∂C/∂K evaluada en K.
                    Devuelve array incluso si K es escalar.
    """
    K = np.array(K).reshape([len(K),1])
    CP = OptionType
    C = lambda k: BS_Call_Put_Option_Price(CP,S_0,k,sigma,t,T,r) 
    return central_diff2(C, K, h)


def d2CdK2(S_0, K, sigma, t, T, r, h = 1e-4):
    """
    Aproximación numérica de la segunda derivada del precio de una call
    respecto al strike (∂²C/∂K²) mediante diferencias finitas centrales.

    Args:
        S_0 (float): Precio spot del activo subyacente en t=0.
        K (float or np.ndarray): Strike(s) sobre los que se evalúa la derivada.
        sigma (float): Volatilidad constante del modelo Black–Scholes.
        t (float): Tiempo actual.
        T (float): Tiempo a vencimiento.
        r (float): Tipo de interés libre de riesgo.
        h (float, optional): Paso de discretización en la variable K. Defaults to 1e-4.

    Returns:
        np.ndarray: Aproximación numérica de ∂²C/∂K² evaluada en K.
                    Devuelve array incluso si K es escalar.
    """
    K = np.array(K).reshape([len(K),1])
    CP = OptionType.CALL
    C = lambda k: BS_Call_Put_Option_Price(CP, S_0, k, sigma, t, T, r) 
    return central_diff2(C, K, h)


def Call_dCdK_d2CdK2():
    """
    Construye y visualiza las siguientes superficies bajo el modelo de Black-Scholes con volatilidad constante:

        1) Precio de la Call C(t,K)
        2) Primera derivada respecto al strike K
        3) Segunda derivada respecto al strike K
    """
    # Model settings
    s0 = 10
    r = 0.05
    sigma = 0.4
    T = 1.001
    CP = OptionType.CALL

    # Function pointers
    CallOpt = lambda t, K: BS_Call_Put_Option_Price(CP, s0, K, sigma, t, T, r)
    dCdKOpt = lambda t, K: dCdK(s0, K, sigma, t, T, r)
    d2CdK2Opt = lambda t, K: d2CdK2(s0, K, sigma, t, T, r)

    # Grid for strike values
    KGrid = np.linspace(s0 / 100.0, 1.5 * s0, 50)

    # MATLAB: timeGrid = 0.02:0.025:T
    timeGrid = np.arange(0.02, T + 1e-12, 0.025)

    # Prepare empty matrices
    callOptM = np.zeros((len(timeGrid), len(KGrid)), dtype=float)
    dCdKM = np.zeros((len(timeGrid), len(KGrid)), dtype=float)
    d2CdK2M = np.zeros((len(timeGrid), len(KGrid)), dtype=float)
    TM = np.zeros((len(timeGrid), len(KGrid)), dtype=float)
    KM = np.zeros((len(timeGrid), len(KGrid)), dtype=float)

    for i in range(len(timeGrid)):
        TM[i, :] = timeGrid[i]
        KM[i, :] = KGrid

        # Squeeze por si pricing devuelve arrays columna
        callOptM[i, :] = np.squeeze(CallOpt(timeGrid[i], KGrid))
        dCdKM[i, :] = np.squeeze(dCdKOpt(timeGrid[i], KGrid))
        d2CdK2M[i, :] = np.squeeze(d2CdK2Opt(timeGrid[i], KGrid))

    # --- Plot 1: Call option surface (mesh) ---
    fig1 = plt.figure(1)
    ax1 = fig1.add_subplot(111, projection="3d")
    ax1.plot_wireframe(
        TM, KM, callOptM,
        rstride=1, cstride=1,
        color=(0.75, 0.45, 0.0),
        linewidth=0.8
    )
    ax1.grid(True)
    ax1.set_xlim(0.0, float(T))
    ax1.set_ylim(0.0, float(np.max(KGrid)))
    ax1.set_zlim(0.0, float(np.max(callOptM)))
    ax1.set_xlabel("t")
    ax1.set_ylabel("K")
    ax1.set_zlabel("Call")
    # ax1.view_init(elev=25, azim=120)
    ax1.view_init(elev=18.8, azim=-60)  # MATLAB: view([-117.5 18.8])

    # --- Plot 2: dC/dK surface (mesh) ---
    fig2 = plt.figure(2)
    ax2 = fig2.add_subplot(111, projection="3d")
    ax2.plot_wireframe(
        TM, KM, dCdKM,
        rstride=1, cstride=1,
        color=(0.75, 0.45, 0.0),
        linewidth=0.8
    )
    ax2.grid(True)
    ax2.set_xlim(0.0, float(T))
    ax2.set_ylim(0.0, float(np.max(KGrid)))
    ax2.set_zlim(float(np.min(dCdKM)), 0.0)     
    ax2.set_xlabel("t")
    ax2.set_ylabel("K")
    ax2.set_zlabel("dC / dK")
    #ax2.view_init(elev=25, azim=)
    ax2.view_init(elev=37.2, azim=-40)  

    # --- Plot 3: d2C/dK2 surface (mesh) ---
    fig3 = plt.figure(3)
    ax3 = fig3.add_subplot(111, projection="3d")
    ax3.plot_wireframe(
        TM, KM, d2CdK2M,
        rstride=1, cstride=1,
        color=(0.75, 0.45, 0.0),
        linewidth=0.8
    )
    ax3.grid(True)
    ax3.set_xlim(0.0, float(T))
    ax3.set_ylim(0.0, float(np.max(KGrid)))
    ax3.set_zlim(0.0, float(np.max(d2CdK2M)))
    ax3.set_xlabel("t")
    ax3.set_ylabel("K")
    ax3.set_zlabel("d2C / dK2")
    #ax3.view_init(elev=25, azim=120)
    ax3.view_init(elev=18.8, azim=-140.5)  # MATLAB: view([-117.5 18.8])

    plt.show()


if __name__ == "__main__":
    Call_dCdK_d2CdK2()
