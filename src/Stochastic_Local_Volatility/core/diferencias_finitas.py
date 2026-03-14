"""
===============================================================================
Diferencias Finitas (Derivación Numérica)
===============================================================================
Este módulo contiene funciones para la aproximación numérica de derivadas
de funciones unidimensionales mediante esquemas de diferencias finitas
centrales.

Incluye:
    - Primera derivada (orden O(h²))
    - Segunda derivada (orden O(h²))

Notas:
    - Las funciones están vectorizadas (compatibles con NumPy).
    - El error de truncamiento es cuadrático en el paso h.
    - La precisión depende del equilibrio entre error de truncamiento
      y error de redondeo (h no debe ser ni demasiado grande ni demasiado pequeño).
"""


import numpy as np

def central_diff(f, x, h):
    """
    Aproximación numérica de la primera derivada f'(x)
    mediante el esquema de diferencias finitas centrales.

    Fórmula utilizada:
        f'(x) ≈ [ f(x + h) - f(x - h) ] / (2h)

    Este esquema es de orden O(h²), es decir, el error de truncamiento
    decrece cuadráticamente con el tamaño del paso h.

    Args:
        f (callable): Función unidimensional a derivar. Debe aceptar
                      arrays NumPy como argumento.
        x (float or np.ndarray): Punto(s) donde se evalúa la derivada.
        h (float): Paso de discretización.

    Returns:
        np.ndarray: Aproximación numérica de f'(x).
                    Devuelve siempre un array NumPy.
    """
    x = np.asarray(x, dtype=float)
    return (f(x + h) - f(x - h)) / (2.0 * h)


def central_diff2(f, x, h):
    """
    Aproximación numérica de la segunda derivada f''(x)
    mediante el esquema de diferencias finitas centrales.

    Fórmula utilizada:
        f''(x) ≈ [ f(x + h) - 2 f(x) + f(x - h) ] / h²

    Este esquema es de orden O(h²) en el error de truncamiento.

    Args:
        f (callable): Función unidimensional a derivar. Debe aceptar
                      arrays NumPy como argumento.
        x (float or np.ndarray): Punto(s) donde se evalúa la segunda derivada.
        h (float): Paso de discretización.

    Returns:
        np.ndarray: Aproximación numérica de f''(x).
                    Devuelve siempre un array NumPy.
    """
    x = np.asarray(x, dtype=float)
    return (f(x + h) - 2.0 * f(x) + f(x - h)) / (h * h)

