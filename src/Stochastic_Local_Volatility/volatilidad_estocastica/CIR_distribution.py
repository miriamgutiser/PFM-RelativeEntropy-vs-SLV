#%%
"""
Distribución CIR. 
------------------

Estudiamos la distribución de transición del proceso
de Cox–Ingersoll–Ross (CIR) en un instante futuro t, dada una
condición inicial V_s = v_s. 

El proceso CIR viene definido por la ecuación diferencial estocástica:

    dV_u = kappa (v_bar - V_u) du + gamma sqrt(V_u) dW_u

Se analiza la ley exacta de V_t | V_s = v_s, que puede expresarse
mediante una distribución chi-cuadrado no central.

Comparar el comportamiento de la densidad (PDF) y de la función
de distribución acumulada (CDF/FDA) del proceso CIR en dos casos:

1) Cuando se cumple la condición de Feller:
       2 * kappa * v_bar >= gamma^2

2) Cuando no se cumple la condición de Feller.

La condición de Feller garantiza la positividad estricta del proceso.
Cuando no se satisface, el proceso puede aproximarse a cero con mayor
probabilidad, lo que se refleja en la forma de la densidad.

utilizamos funciones definidas en core.processes:
    - FellerCondition (o feller_q)
    - CIR_PDF_CDF (o cir_transition_pdf_cdf)
"""



import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st

from Stochastic_Local_Volatility.core.processes import FellerCondition,CIR_PDF_CDF

# Global variables 

# s = 0
# t = 5
# x = np.linspace(0.001,0.501,100)

s = 0
t = 2
x = np.linspace(0.0005, 0.30, 200)


# Feller condition


def Analysis():

    # Feller condition satisfied

    # gamma = 0.316, kappa = 0.5, v_s   = 0.2
    # v_bar = 0.05
    kappa = 1.5
    v_s   = 0.04
    v_bar = 0.04
    gamma = 0.2
    q_F_satisfied = FellerCondition(kappa,v_bar,gamma)
    print(q_F_satisfied)
    output_case_satisfied = CIR_PDF_CDF(x,gamma,kappa,t,s,v_bar,v_s)
    
    # Feller condition not satisfied

    # gamma    = 0.129, kappa    = 0.5, v_s      = 0.2
    # v_bar    = 0.05
    gamma = 0.5
    output_case_not_satisfied = CIR_PDF_CDF(x,gamma,kappa,t,s,v_bar,v_s)
    q_F_not_satisfied = FellerCondition(kappa,v_bar, gamma)
    
    # Generate output plots

    plt.figure(1)
    plt.plot(x, output_case_satisfied["pdf"],
         label=f"Feller satisfecha (q_F = {q_F_satisfied:.4f})")
    plt.plot(x, output_case_not_satisfied["pdf"], linestyle="--",
         label=f"Feller no satisfecha (q_F = {q_F_not_satisfied:.4f})")
    plt.grid()
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.ylim([0.0,25.0])
    plt.legend(loc = "upper right", frameon = True)

    plt.figure(2)
    plt.plot(x, output_case_satisfied["cdf"],
         label=f"Feller satisfecha (q_F = {q_F_satisfied:.4f})")
    plt.plot(x, output_case_not_satisfied["cdf"], linestyle="--",
         label=f"Feller no satisfecha (q_F = {q_F_not_satisfied:.4f})")
    plt.grid()
    plt.xlabel("x")
    plt.ylabel("F(x)")
    plt.legend(loc="lower right", frameon=True)

    plt.show()

Analysis()
