"""
Calibración de pesos q_i mediante mínima entropía relativa.

Este script toma como entrada la simulación previa generada en
`gbm_prior_paths.py` y calibra unos pesos posteriores `q_i` sobre las
trayectorias para reproducir los precios benchmark de mercado.

Funciones principales utilizadas en este script:
- `load_prior_simulation`: carga la matriz de payoffs y los pesos previos.
- `precios_benchmark_market`: construye el vector de precios objetivo.
- `calibrate_relative_entropy_weights`: ejecuta la calibración principal.
- `calcular_pesos_posteriores`: recupera los pesos posteriores a partir de lambda.
- `calcular_precios_modelo_y_gradiente`: calcula precios modelo y gradiente.
- `calcular_jacobiano`: construye el Jacobiano para el paso de Newton.
- `resolver_paso_newton`: resuelve la dirección de actualización.
- `relative_entropy_distance`: calcula la entropía relativa final.
- `main`: coordina la carga, calibración, exportación e impresión de resultados.

"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from Entropia_Relativa.data_loader import BENCHMARK_OPTIONS
from Entropia_Relativa.python_variable_io import (
    export_python_variables_module,
    load_python_variables_module,
)


DISPLAY_DECIMALS = 7


@dataclass(frozen=True)
class RelativeEntropyCalibrationConfig:
    """
    Configuración de la calibración de pesos.
    """

    prior_filename: str = "gbm_prior_simulation_results.py"
    export_results: bool = True
    export_filename: str = "relative_entropy_calibration_results.py"
    maxiter: int = 2000
    gradient_tolerance: float = 1e-10
    pricing_tolerance: float = 1e-6
    alpha_inicial: float = 0.01
    alpha_maximo: float = 1.0
    factor_aumento_alpha: float = 2.0
    factor_reduccion_alpha: float = 5.0
    limite_factor_condicion: float = 10.0
    max_ajustes_alpha: int = 10


@dataclass(frozen=True)
class RelativeEntropyCalibrationResult:
    """
    Resultado de la calibración por entropía relativa.
    """

    lambdas: np.ndarray
    pesos_posteriores: np.ndarray
    precios_modelo: np.ndarray
    precios_replicados_gtq: np.ndarray
    errores_precio: np.ndarray
    error_consistencia_gtq: float
    entropia_relativa: float
    valor_objetivo: float
    n_paths: int
    n_benchmarks: int
    optimizador_exito: bool
    optimizador_mensaje: str
    aceptado: bool
    iteraciones: int
    norma_gradiente: float


def load_prior_simulation(prior_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Carga la simulación previa desde un archivo .py con variables.
    """
    try:
        data = load_python_variables_module(prior_path)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"No se encuentra la simulación previa: {prior_path}"
        ) from exc
    matriz_payoffs = np.asarray(data.discounted_payoff_matrix, dtype=float)
    tiempos = np.asarray(data.times, dtype=float)

    if hasattr(data, "prior_weights"):
        pesos_prior = np.asarray(data.prior_weights, dtype=float)
    else:
        n_paths = matriz_payoffs.shape[0]
        pesos_prior = np.full(n_paths, 1.0 / n_paths, dtype=float)

    return matriz_payoffs, pesos_prior, tiempos


def precios_benchmark_market() -> np.ndarray:
    """
    Devuelve el vector C de precios benchmark de mercado.
    """
    return np.array([benchmark.market_price for benchmark in BENCHMARK_OPTIONS], dtype=float)


def dual_objective_and_gradient(
    lambdas: np.ndarray,
    matriz_payoffs: np.ndarray,
    pesos_prior: np.ndarray,
    precios_market: np.ndarray,
) -> tuple[float, np.ndarray]:
    """
    Evalúa la función dual y su gradiente.

    Se minimiza:
        W(lambda) = log(sum_i p_i exp(lambda · g_i)) - lambda · C
    """
    n_paths, n_benchmarks = matriz_payoffs.shape

    puntuaciones_trayectoria = np.zeros(n_paths, dtype=float)
    for i in range(n_paths):
        puntuacion_i = 0.0
        for j in range(n_benchmarks):
            puntuacion_i += matriz_payoffs[i, j] * lambdas[j]
        puntuaciones_trayectoria[i] = puntuacion_i

    max_puntuacion = float(np.max(puntuaciones_trayectoria))
    suma_particion = 0.0
    for i in range(n_paths):
        suma_particion += pesos_prior[i] * np.exp(puntuaciones_trayectoria[i] - max_puntuacion)

    log_particion = max_puntuacion + np.log(suma_particion)

    pesos_posteriores = np.zeros(n_paths, dtype=float)
    for i in range(n_paths):
        pesos_posteriores[i] = pesos_prior[i] * np.exp(puntuaciones_trayectoria[i] - log_particion)

    gradiente = np.zeros(n_benchmarks, dtype=float)
    for j in range(n_benchmarks):
        suma_ponderada = 0.0
        for i in range(n_paths):
            suma_ponderada += matriz_payoffs[i, j] * pesos_posteriores[i]
        gradiente[j] = suma_ponderada - precios_market[j]

    lambda_por_market = 0.0
    for j in range(n_benchmarks):
        lambda_por_market += lambdas[j] * precios_market[j]

    valor_objetivo = log_particion - lambda_por_market
    return float(valor_objetivo), gradiente


def calcular_pesos_posteriores(
    lambdas: np.ndarray,
    matriz_payoffs: np.ndarray,
    pesos_prior: np.ndarray,
) -> tuple[np.ndarray, float]:
    """
    Recupera los pesos q_i a partir de los multiplicadores óptimos.
    """
    n_paths, n_benchmarks = matriz_payoffs.shape

    puntuaciones_trayectoria = np.zeros(n_paths, dtype=float)
    for i in range(n_paths):
        puntuacion_i = 0.0
        for j in range(n_benchmarks):
            puntuacion_i += matriz_payoffs[i, j] * lambdas[j]
        puntuaciones_trayectoria[i] = puntuacion_i

    max_puntuacion = float(np.max(puntuaciones_trayectoria))
    suma_particion = 0.0
    for i in range(n_paths):
        suma_particion += pesos_prior[i] * np.exp(puntuaciones_trayectoria[i] - max_puntuacion)

    log_particion = max_puntuacion + np.log(suma_particion)

    pesos_posteriores = np.zeros(n_paths, dtype=float)
    for i in range(n_paths):
        pesos_posteriores[i] = pesos_prior[i] * np.exp(puntuaciones_trayectoria[i] - log_particion)

    return pesos_posteriores, float(log_particion)


def calcular_precios_modelo_y_gradiente(
    matriz_payoffs: np.ndarray,
    pesos_posteriores: np.ndarray,
    precios_market: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calcula los precios modelo bajo q y el gradiente de W(lambda).
    """
    n_paths, n_benchmarks = matriz_payoffs.shape
    precios_modelo = np.zeros(n_benchmarks, dtype=float)

    for j in range(n_benchmarks):
        suma_ponderada = 0.0
        for i in range(n_paths):
            suma_ponderada += matriz_payoffs[i, j] * pesos_posteriores[i]
        precios_modelo[j] = suma_ponderada

    gradiente = np.zeros(n_benchmarks, dtype=float)
    for j in range(n_benchmarks):
        gradiente[j] = precios_modelo[j] - precios_market[j]

    return precios_modelo, gradiente


def calcular_precios_por_gtq(
    matriz_payoffs: np.ndarray,
    pesos_posteriores: np.ndarray,
) -> np.ndarray:
    """
    Calcula G^T q de forma explícita para verificar la replicación benchmark.
    """
    return np.asarray(matriz_payoffs, dtype=float).T @ np.asarray(
        pesos_posteriores, dtype=float
    )


def calcular_jacobiano(
    matriz_payoffs: np.ndarray,
    pesos_posteriores: np.ndarray,
    precios_modelo: np.ndarray,
) -> np.ndarray:
    """
    Calcula el Jacobiano J_ab según la ecuación de covarianzas.
    """
    n_paths, n_benchmarks = matriz_payoffs.shape
    jacobiano = np.zeros((n_benchmarks, n_benchmarks), dtype=float)

    for a in range(n_benchmarks):
        for b in range(a, n_benchmarks):
            suma_productos = 0.0
            for i in range(n_paths):
                suma_productos += (
                    matriz_payoffs[i, a]
                    * matriz_payoffs[i, b]
                    * pesos_posteriores[i]
                )
            valor_ab = suma_productos - precios_modelo[a] * precios_modelo[b]
            jacobiano[a, b] = valor_ab
            jacobiano[b, a] = valor_ab

    return jacobiano


def resolver_paso_newton(
    jacobiano: np.ndarray,
    gradiente: np.ndarray,
) -> np.ndarray:
    """
    Resuelve J * paso = gradiente con una regularización suave si hace falta.
    """
    n_benchmarks = len(gradiente)
    identidad = np.eye(n_benchmarks, dtype=float)
    regularizacion = 0.0

    for _ in range(8):
        try:
            if regularizacion == 0.0:
                return np.linalg.solve(jacobiano, gradiente)
            return np.linalg.solve(jacobiano + regularizacion * identidad, gradiente)
        except np.linalg.LinAlgError:
            if regularizacion == 0.0:
                regularizacion = 1e-10
            else:
                regularizacion *= 10.0

    return np.linalg.lstsq(jacobiano + regularizacion * identidad, gradiente, rcond=None)[0]


def norma_infinito(vector: np.ndarray) -> float:
    """
    Devuelve la norma infinito de un vector.
    """
    maximo = 0.0
    for valor in vector:
        abs_valor = abs(float(valor))
        if abs_valor > maximo:
            maximo = abs_valor
    return maximo


def producto_escalar(x: np.ndarray, y: np.ndarray) -> float:
    """
    Calcula un producto escalar usando sumas explícitas.
    """
    suma = 0.0
    for i in range(len(x)):
        suma += float(x[i]) * float(y[i])
    return suma


def objetivo_dual(
    lambdas: np.ndarray,
    matriz_payoffs: np.ndarray,
    pesos_prior: np.ndarray,
    precios_market: np.ndarray,
) -> float:
    """
    Evalúa W(lambda) usando pesos posteriores obtenidos con lambda.
    """
    _, log_particion = calcular_pesos_posteriores(
        lambdas=lambdas,
        matriz_payoffs=matriz_payoffs,
        pesos_prior=pesos_prior,
    )
    return float(log_particion - producto_escalar(lambdas, precios_market))


def relative_entropy_distance(
    pesos_posteriores: np.ndarray,
    pesos_prior: np.ndarray,
) -> float:
    """
    Calcula D(q|p) = sum_i q_i log(q_i / p_i).
    """
    positive_mask = (pesos_posteriores > 0.0) & (pesos_prior > 0.0)
    q = pesos_posteriores[positive_mask]
    p = pesos_prior[positive_mask]
    return float(np.sum(q * np.log(q / p)))


def calibrate_relative_entropy_weights(
    matriz_payoffs: np.ndarray,
    pesos_prior: np.ndarray,
    precios_market: np.ndarray,
    config: RelativeEntropyCalibrationConfig,
) -> RelativeEntropyCalibrationResult:
    """
    Calibra los pesos q_i minimizando la entropía relativa al prior.
    """
    n_paths, n_benchmarks = matriz_payoffs.shape
    lambdas = np.zeros(n_benchmarks, dtype=float)
    alpha = config.alpha_inicial
    iteracion_final = 0
    mensaje = "Número máximo de iteraciones alcanzado."
    optimizador_exito = False

    for iteracion in range(1, config.maxiter + 1):
        iteracion_final = iteracion

        pesos_posteriores, _ = calcular_pesos_posteriores(
            lambdas=lambdas,
            matriz_payoffs=matriz_payoffs,
            pesos_prior=pesos_prior,
        )
        precios_modelo, gradiente = calcular_precios_modelo_y_gradiente(
            matriz_payoffs=matriz_payoffs,
            pesos_posteriores=pesos_posteriores,
            precios_market=precios_market,
        )
        jacobiano = calcular_jacobiano(
            matriz_payoffs=matriz_payoffs,
            pesos_posteriores=pesos_posteriores,
            precios_modelo=precios_modelo,
        )

        norma_gradiente = norma_infinito(gradiente)
        max_abs_error = norma_infinito(precios_modelo - precios_market)
        if (
            norma_gradiente <= config.gradient_tolerance
            or max_abs_error <= config.pricing_tolerance
        ):
            mensaje = "Gradiente y precios dentro de tolerancia."
            optimizador_exito = True
            break

        paso_newton = resolver_paso_newton(jacobiano, gradiente)
        condicion_actual = float(np.linalg.cond(jacobiano))
        alpha_local = alpha
        aceptado_paso = False

        for _ in range(config.max_ajustes_alpha):
            lambdas_candidatas = np.zeros(n_benchmarks, dtype=float)
            for j in range(n_benchmarks):
                lambdas_candidatas[j] = lambdas[j] - alpha_local * paso_newton[j]

            pesos_candidatos, _ = calcular_pesos_posteriores(
                lambdas=lambdas_candidatas,
                matriz_payoffs=matriz_payoffs,
                pesos_prior=pesos_prior,
            )
            precios_modelo_candidatos, _ = calcular_precios_modelo_y_gradiente(
                matriz_payoffs=matriz_payoffs,
                pesos_posteriores=pesos_candidatos,
                precios_market=precios_market,
            )
            jacobiano_candidato = calcular_jacobiano(
                matriz_payoffs=matriz_payoffs,
                pesos_posteriores=pesos_candidatos,
                precios_modelo=precios_modelo_candidatos,
            )
            condicion_candidata = float(np.linalg.cond(jacobiano_candidato))

            if (
                np.isfinite(condicion_actual)
                and np.isfinite(condicion_candidata)
                and condicion_actual > 0.0
                and condicion_candidata > config.limite_factor_condicion * condicion_actual
            ):
                alpha_local /= config.factor_reduccion_alpha
                continue

            lambdas = lambdas_candidatas
            aceptado_paso = True
            break

        if not aceptado_paso:
            mensaje = "No se pudo aceptar un paso de Newton estable."
            optimizador_exito = False
            break

        alpha = min(config.alpha_maximo, alpha_local * config.factor_aumento_alpha)

    pesos_posteriores, _ = calcular_pesos_posteriores(
        lambdas=lambdas,
        matriz_payoffs=matriz_payoffs,
        pesos_prior=pesos_prior,
    )
    precios_modelo, gradiente = calcular_precios_modelo_y_gradiente(
        matriz_payoffs=matriz_payoffs,
        pesos_posteriores=pesos_posteriores,
        precios_market=precios_market,
    )
    errores_precio = np.zeros(n_benchmarks, dtype=float)
    for j in range(n_benchmarks):
        errores_precio[j] = precios_modelo[j] - precios_market[j]
    precios_replicados_gtq = calcular_precios_por_gtq(
        matriz_payoffs=matriz_payoffs,
        pesos_posteriores=pesos_posteriores,
    )
    error_consistencia_gtq = float(
        np.max(np.abs(precios_replicados_gtq - precios_modelo))
    )

    max_abs_error = norma_infinito(errores_precio)
    norma_gradiente = norma_infinito(gradiente)
    aceptado = bool(optimizador_exito or max_abs_error <= config.pricing_tolerance)
    entropia_relativa = relative_entropy_distance(
        pesos_posteriores=pesos_posteriores,
        pesos_prior=pesos_prior,
    )
    valor_objetivo = objetivo_dual(
        lambdas=lambdas,
        matriz_payoffs=matriz_payoffs,
        pesos_prior=pesos_prior,
        precios_market=precios_market,
    )

    return RelativeEntropyCalibrationResult(
        lambdas=lambdas,
        pesos_posteriores=pesos_posteriores,
        precios_modelo=precios_modelo,
        precios_replicados_gtq=precios_replicados_gtq,
        errores_precio=errores_precio,
        error_consistencia_gtq=error_consistencia_gtq,
        entropia_relativa=entropia_relativa,
        valor_objetivo=valor_objetivo,
        n_paths=n_paths,
        n_benchmarks=n_benchmarks,
        optimizador_exito=optimizador_exito,
        optimizador_mensaje=mensaje,
        aceptado=aceptado,
        iteraciones=iteracion_final,
        norma_gradiente=norma_gradiente,
    )


def main() -> None:
    config = RelativeEntropyCalibrationConfig()
    prior_path = Path(__file__).resolve().with_name(config.prior_filename)

    matriz_payoffs, pesos_prior, _ = load_prior_simulation(prior_path)
    precios_market = precios_benchmark_market()

    calibracion = calibrate_relative_entropy_weights(
        matriz_payoffs=matriz_payoffs,
        pesos_prior=pesos_prior,
        precios_market=precios_market,
        config=config,
    )

    if config.export_results:
        export_path = Path(__file__).resolve().with_name(config.export_filename)
        export_python_variables_module(
            module_path=export_path,
            variables={
                "lambdas": calibracion.lambdas,
                "pesos_posteriores": calibracion.pesos_posteriores,
                "precios_modelo": calibracion.precios_modelo,
                "precios_replicados_gtq": calibracion.precios_replicados_gtq,
                "precios_market": precios_market,
                "errores_precio": calibracion.errores_precio,
                "error_consistencia_gtq": calibracion.error_consistencia_gtq,
                "entropia_relativa": calibracion.entropia_relativa,
                "valor_objetivo": calibracion.valor_objetivo,
                "iteraciones": calibracion.iteraciones,
                "norma_gradiente": calibracion.norma_gradiente,
            },
        )
    else:
        export_path = None

    max_abs_error = float(np.max(np.abs(calibracion.errores_precio)))
    mae = float(np.mean(np.abs(calibracion.errores_precio)))

    print(
        f"Calibración completada con n = {calibracion.n_paths} trayectorias "
        f"y N = {calibracion.n_benchmarks} benchmarks."
    )
    print(
        f"Max abs error = {max_abs_error:.{DISPLAY_DECIMALS}f}, "
        f"MAE = {mae:.{DISPLAY_DECIMALS}f}, "
        f"D(q|p) = {calibracion.entropia_relativa:.{DISPLAY_DECIMALS}f}"
    )
    print(
        f"Iteraciones = {calibracion.iteraciones}, "
        f"norma gradiente = {calibracion.norma_gradiente:.{DISPLAY_DECIMALS}f}"
    )
    print(
        f"Estado optimizador: {'aceptado' if calibracion.aceptado else 'no aceptado'} "
        f"({calibracion.optimizador_mensaje})"
    )
    print(
        f"Chequeo G^T q: max |G^T q - C| = {max_abs_error:.{DISPLAY_DECIMALS}f}, "
        f"max |G^T q - precios_modelo| = {calibracion.error_consistencia_gtq:.{DISPLAY_DECIMALS}f}"
    )
    if export_path is not None:
        print(f"Pesos q_i y resultados guardados como variables en {export_path.name}.")


if __name__ == "__main__":
    main()
