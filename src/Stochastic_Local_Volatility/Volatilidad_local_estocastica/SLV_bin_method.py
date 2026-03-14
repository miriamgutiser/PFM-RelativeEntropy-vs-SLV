#%%
"""
Este código implementa el método de binning para estimar la función de densidad condicional E[V|S=si] en
un modelo de volatilidad local estocástica. 

Se generan trayectorias de precios y varianzas utilizando el modelo de Heston, y luego se comparan 
las estimaciones obtenidas mediante el método de binning con las obtenidas mediante un método COS 2D 
y un enfoque de regresión.

Finalmente, se visualizan los resultados mediante gráficos de dispersión y contornos.
"""


import numpy as np
import matplotlib.pyplot as plt

from core.types import OptionType
from volatilidad_estocastica.Heston_model import ChFHestonModel, CIR_Sample, GeneratePathsHestonAES
from volatilidad_estocastica.Heston_model2D import ChFHeston2D, ksi, COSMethod2DHeston


def getEVBinMethod(S, v, NoOfBins):
    """
    Calcula la estimación no paramétrica de E[V|S=si] usando el método de binning.
    
    Args:
        S (float): Grid de precios del activo subyacente.
        v (float): Grid de varianza.
        NoOfBins (int): Número de bins para el método de binning.

    Returns:
        ndarray: Estimación no paramétrica de E[V|S=si].
    """
    S = np.asarray(S).flatten()
    v = np.asarray(v).flatten()
    n = len(S)
    if NoOfBins != 1:
        mat = np.column_stack([S, v])
        mat = mat[np.argsort(mat[:, 0])]
        BinSize = n // NoOfBins
        # first bin: rows 0..BinSize-1  (MATLAB 1:BinSize)
        first_mean = np.mean(mat[0:BinSize, 1])
        first_lo   = mat[0, 0]
        first_hi   = mat[BinSize - 1, 0]          # mat(BinSize,1) in MATLAB
        # last bin: rows (NoBins-1)*BinSize..NoBins*BinSize  (inclusive, MATLAB)
        last_mean  = np.mean(mat[(NoOfBins - 1) * BinSize : NoOfBins * BinSize + 1, 1])
        last_lo    = mat[(NoOfBins - 1) * BinSize, 0]
        last_hi    = mat[min(NoOfBins * BinSize, n - 1), 0]
        val = first_mean * ((S >= first_lo) & (S < first_hi))
        for i in range(2, NoOfBins):
            start   = (i - 1) * BinSize            # Python 0-based = MATLAB (i-1)*BinSize
            end     = i * BinSize                  # exclusive in Python
            # MATLAB: mat((i-1)*BinSize:i*BinSize,2) → include boundary element
            meanval = np.mean(mat[start : end + 1, 1])
            lo      = mat[start, 0]                # mat((i-1)*BinSize,1)
            hi      = mat[end, 0]                  # mat(i*BinSize,1)
            val     = val + meanval * ((S >= lo) & (S < hi))
        val = val + last_mean * ((S >= last_lo) & (S <= last_hi))
        val[val == 0] = last_mean
        return val
    else:
        return np.mean(v) * np.ones(n)




def mainCalculation():
    
    S0 = 1.0
    r = 0.0
    x0 = np.log(S0)

    kappa = 0.5
    gamma_c = 0.1
    vBar = 0.07
    v0 = 0.07
    rho_xv = -0.7

    qparameter = 2 * kappa * vBar / (gamma_c ** 2) - 1
    print(f"qparameter = {qparameter}")  # Feller condition

    NoBins = 10
    T = 1.0
    NoOfPaths = int(1e5)
    NoOfSteps = int(1e2 * T)

    # -- 2DCOS 2D METHOD --
    Sgrid, _, EVS, _ = COSMethod2DHeston(x0, v0, rho_xv, kappa, vBar, gamma_c, T, r)

    # -- Monte Carlo simulation --
    S_paths, time, V_paths = GeneratePathsHestonAES(
        NoOfPaths, NoOfSteps, T, r, S0, kappa, gamma_c, rho_xv, vBar, v0
    )
    S = S_paths[:, -1]
    V = V_paths[:, -1]

    # -- Regression-based approach --
    X1 = np.column_stack([np.ones(NoOfPaths), S, S ** 2])
    beta1 = np.linalg.lstsq(X1, V, rcond=None)[0]
    EV2 = X1 @ beta1
    matr = np.column_stack([S, EV2])
    matr = matr[np.argsort(matr[:, 0])]

    # -- Non-parametric approach --
    mat1_vals = getEVBinMethod(S, V, NoBins)
    mat1 = np.column_stack([S, mat1_vals])
    mat1 = mat1[np.argsort(mat1[:, 0])]

    # -- Plot --
    plt.figure()
    plt.grid(True)
    p1 = plt.scatter(S[: int(NoOfPaths / 10)], V[: int(NoOfPaths / 10)], c='g', s=1)
    plt.xlabel('S')
    plt.ylabel('V')
    plt.xlim([0, np.max(S)])
    plt.ylim([0, np.max(V)])  # restrict variance axis as requested
    p2, = plt.plot(Sgrid, EVS, 'k--x', markersize=5, linewidth=2)
    p3, = plt.plot(mat1[:, 0], mat1[:, 1], 'r', linewidth=2)
    p4, = plt.plot(matr[:, 0], matr[:, 1], 'b--', linewidth=2)
    # explicit legend for the first figure
    plt.legend(
        [p1, p2, p3, p4],
        ['Observaciones', 'COS 2D', 'Non-parametric', 'Regression-based'],
        loc='best',
    )

    # contour plot
    plt.figure()
    n, xedges, yedges = np.histogram2d(S, V, bins=[30, 30])
    Xc = 0.5 * (xedges[:-1] + xedges[1:])
    Yc = 0.5 * (yedges[:-1] + yedges[1:])
    contour = plt.contour(Xc, Yc, n.T, levels=6, colors='b')
    p1, = plt.plot(Sgrid, EVS, 'k--x', markersize=5, linewidth=2)
    p2, = plt.plot(mat1[:, 0], mat1[:, 1], 'r', linewidth=2)
    plt.xlim([0, np.max(S)])
    plt.ylim([0, np.max(V)])  # restrict variance axis as requested
    plt.grid(True)
    plt.xlabel('S')
    plt.ylabel('V')
    # Create proxy artist for contour legend
    contour_proxy = plt.Line2D([0], [0], color='b', linewidth=2)
    plt.legend(
        [contour_proxy, p1, p2],
        ['Densidad', 'COS 2D', 'No paramétrico'],
        loc='best',
    )

    plt.show()


if __name__ == "__main__":
    mainCalculation()