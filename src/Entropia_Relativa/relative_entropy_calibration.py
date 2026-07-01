"""
Calibración de pesos q_i mediante mínima entropía relativa.

El módulo sigue el esquema del paper:
`Weighted Monte Carlo: Calibrating the Smile and Preserving
Martingale Condition` (Elices, Gimenez), incorporando:

1. calibración exacta por mínima entropía relativa,
2. versión de mínimos cuadrados ponderados (WLS),
3. restricciones adicionales de forwards,
4. restricciones sintéticas de martingala por ventanas.

La calibración se formula sobre una matriz general de restricciones
`G`, donde cada columna puede representar un payoff benchmark,
un forward descontado o una restricción sintética de martingala.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from Entropia_Relativa.data_loader import BENCHMARK_OPTIONS, K, S0, q, r
from Entropia_Relativa.data_loader import (
    calibration_filename,
    calibration_mode,
    filter_degenerate_option_constraints,
    include_forward_constraints,
    include_martingale_constraints,
    min_option_positive_payoff_count,
    min_option_variance,
    prior_filename,
    wls_weight_forwards,
    wls_weight_martingale,
    wls_weight_options,
)
from Entropia_Relativa.python_variable_io import (
    export_python_variables_module,
    load_python_variables_module,
)


DISPLAY_DECIMALS = 2
CALIBRATION_MODE_EXACT = "exact"
CALIBRATION_MODE_WLS = "weighted_least_squares"
CONSTRAINT_KIND_OPTION = "option"
CONSTRAINT_KIND_FORWARD = "forward"
CONSTRAINT_KIND_MARTINGALE = "martingale"


@dataclass(frozen=True)
class PriorSimulationData:
    """
    Datos cargados de la simulación previa.
    """

    path_matrix: np.ndarray
    discounted_payoff_matrix: np.ndarray
    prior_weights: np.ndarray
    times: np.ndarray


@dataclass(frozen=True)
class CalibrationConstraint:
    """
    Restricción individual de la calibración.
    """

    name: str
    kind: str
    payoffs: np.ndarray
    target_value: float
    least_squares_weight: float


@dataclass(frozen=True)
class ConstraintSet:
    """
    Problema de calibración ya ensamblado.
    """

    matrix: np.ndarray
    targets: np.ndarray
    least_squares_weights: np.ndarray
    names: tuple[str, ...]
    kinds: tuple[str, ...]
    n_option_constraints: int
    n_forward_constraints: int
    n_martingale_constraints: int


@dataclass(frozen=True)
class RelativeEntropyCalibrationConfig:
    """
    Configuración de la calibración de pesos.
    """

    prior_filename: str = prior_filename
    export_results: bool = True
    export_filename: str = calibration_filename
    calibration_mode: str = calibration_mode
    include_forward_constraints: bool = include_forward_constraints
    include_martingale_constraints: bool = include_martingale_constraints
    wls_weight_options: float = wls_weight_options
    wls_weight_forwards: float = wls_weight_forwards
    wls_weight_martingale: float = wls_weight_martingale
    filter_degenerate_option_constraints: bool = filter_degenerate_option_constraints
    min_option_positive_payoff_count: int = min_option_positive_payoff_count
    min_option_variance: float = min_option_variance
    martingale_std_multiplier: float = 2.0
    martingale_min_paths_per_window: int = 25
    martingale_min_variance: float = 1e-12
    maxiter: int = 500
    gradient_tolerance: float = 1e-8
    pricing_tolerance: float = 5e-4
    alpha_inicial: float = 0.01
    alpha_maximo: float = 1.0
    factor_aumento_alpha: float = 2.0
    factor_reduccion_alpha: float = 5.0
    limite_factor_condicion: float = 10.0
    max_ajustes_alpha: int = 10
    filter_itm_option_constraints_in_exact: bool = True
    exact_min_option_moneyness: float = 1.0
    exact_newton_eigenvalue_tolerance: float = 1e-10


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
    n_constraints: int
    n_option_constraints: int
    n_forward_constraints: int
    n_martingale_constraints: int
    calibration_mode: str
    optimizador_exito: bool
    optimizador_mensaje: str
    aceptado: bool
    iteraciones: int
    norma_gradiente: float
    max_abs_error_total: float
    max_abs_error_options: float
    max_abs_error_forwards: float
    max_abs_error_martingale: float
    least_squares_weights: np.ndarray
    constraint_targets: np.ndarray
    constraint_kinds: tuple[str, ...]
    constraint_names: tuple[str, ...]


def _validate_calibration_mode(calibration_mode: str) -> None:
    if calibration_mode not in {CALIBRATION_MODE_EXACT, CALIBRATION_MODE_WLS}:
        raise ValueError(
            "calibration_mode debe ser "
            f"'{CALIBRATION_MODE_EXACT}' o '{CALIBRATION_MODE_WLS}'."
        )


def load_prior_simulation(prior_path: Path) -> PriorSimulationData:
    """
    Carga la simulación previa desde un archivo .py con variables.
    """
    try:
        data = load_python_variables_module(prior_path)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"No se encuentra la simulación previa: {prior_path}"
        ) from exc

    path_matrix = np.asarray(data.path_matrix, dtype=float)
    matriz_payoffs = np.asarray(data.discounted_payoff_matrix, dtype=float)
    tiempos = np.asarray(data.times, dtype=float)

    if hasattr(data, "prior_weights"):
        pesos_prior = np.asarray(data.prior_weights, dtype=float)
    else:
        n_paths = matriz_payoffs.shape[0]
        pesos_prior = np.full(n_paths, 1.0 / n_paths, dtype=float)

    return PriorSimulationData(
        path_matrix=path_matrix,
        discounted_payoff_matrix=matriz_payoffs,
        prior_weights=pesos_prior,
        times=tiempos,
    )


def precios_benchmark_market() -> np.ndarray:
    """
    Devuelve el vector de precios benchmark de mercado.
    """
    return np.array([benchmark.market_price for benchmark in BENCHMARK_OPTIONS], dtype=float)


def discount_factor(maturity: float) -> float:
    """
    Factor de descuento hasta t0.
    """
    return float(np.exp(-r * maturity))


def forward_price(maturity: float) -> float:
    """
    Forward del subyacente para el vencimiento dado.
    """
    return float(S0 * np.exp((r - q) * maturity))


def discounted_forward_target(maturity: float) -> float:
    """
    Valor presente del subyacente bajo la medida risk-neutral.
    """
    return float(S0 * np.exp(-q * maturity))


def _build_option_constraints(
    prior_data: PriorSimulationData,
    config: RelativeEntropyCalibrationConfig,
) -> list[CalibrationConstraint]:
    option_prices = precios_benchmark_market()
    constraints: list[CalibrationConstraint] = []
    exact_filter_threshold = float(config.exact_min_option_moneyness * S0)

    for index, benchmark in enumerate(BENCHMARK_OPTIONS):
        if (
            config.calibration_mode == CALIBRATION_MODE_EXACT
            and config.filter_itm_option_constraints_in_exact
            and float(benchmark.strike) < exact_filter_threshold
        ):
            continue

        payoffs = np.asarray(
            prior_data.discounted_payoff_matrix[:, index],
            dtype=float,
        )
        if config.filter_degenerate_option_constraints:
            n_positive_payoffs = int(np.count_nonzero(payoffs > 0.0))
            variance = _constraint_variance(
                payoffs=payoffs,
                pesos_prior=prior_data.prior_weights,
            )
            if n_positive_payoffs < config.min_option_positive_payoff_count:
                continue
            if variance < config.min_option_variance:
                continue

        constraints.append(
            CalibrationConstraint(
                name=(
                    f"option_{benchmark.maturity_label}_"
                    f"K{benchmark.strike:.4f}"
                ),
                kind=CONSTRAINT_KIND_OPTION,
                payoffs=payoffs,
                target_value=float(option_prices[index]),
                least_squares_weight=float(config.wls_weight_options),
            )
        )

    return constraints


def _build_forward_constraints(
    prior_data: PriorSimulationData,
    config: RelativeEntropyCalibrationConfig,
) -> list[CalibrationConstraint]:
    constraints: list[CalibrationConstraint] = []
    for time_index in range(1, prior_data.times.size):
        maturity = float(prior_data.times[time_index])
        discounted_spot = discount_factor(maturity) * prior_data.path_matrix[:, time_index]
        constraints.append(
            CalibrationConstraint(
                name=f"forward_T{maturity:.6f}",
                kind=CONSTRAINT_KIND_FORWARD,
                payoffs=np.asarray(discounted_spot, dtype=float),
                target_value=discounted_forward_target(maturity),
                least_squares_weight=float(config.wls_weight_forwards),
            )
        )
    return constraints


def _base_window_edges(strikes: np.ndarray) -> tuple[np.ndarray, float, float]:
    strikes_sorted = np.unique(np.sort(np.asarray(strikes, dtype=float)))
    if strikes_sorted.size == 0:
        raise ValueError("La rejilla de strikes no puede estar vacia.")

    if strikes_sorted.size == 1:
        half_width = max(1.0, 0.1 * abs(float(strikes_sorted[0])))
        edges = np.array(
            [max(0.0, float(strikes_sorted[0]) - half_width), float(strikes_sorted[0]) + half_width],
            dtype=float,
        )
        return edges, half_width, half_width

    midpoints = 0.5 * (strikes_sorted[:-1] + strikes_sorted[1:])
    left_step = float(midpoints[0] - strikes_sorted[0])
    right_step = float(strikes_sorted[-1] - midpoints[-1])
    edges = np.concatenate(
        (
            [float(strikes_sorted[0] - left_step)],
            midpoints,
            [float(strikes_sorted[-1] + right_step)],
        )
    )
    return edges.astype(float), left_step, right_step


def _build_martingale_window_edges(
    lower_bound: float,
    upper_bound: float,
    strikes: np.ndarray,
) -> np.ndarray:
    edges, left_step, right_step = _base_window_edges(strikes=strikes)
    edges_list = edges.tolist()

    while edges_list[0] > lower_bound:
        edges_list.insert(0, edges_list[0] - left_step)

    while edges_list[-1] < upper_bound:
        edges_list.append(edges_list[-1] + right_step)

    edges_array = np.maximum(np.asarray(edges_list, dtype=float), 0.0)
    edges_array = np.unique(edges_array)
    if edges_array.size < 2:
        raise ValueError("No se pudieron construir ventanas de martingala validas.")
    return edges_array


def _constraint_variance(
    payoffs: np.ndarray,
    pesos_prior: np.ndarray,
) -> float:
    mean_value = float(np.dot(payoffs, pesos_prior))
    centered = payoffs - mean_value
    return float(np.dot(centered * centered, pesos_prior))


def _build_martingale_constraints(
    prior_data: PriorSimulationData,
    config: RelativeEntropyCalibrationConfig,
) -> list[CalibrationConstraint]:
    constraints: list[CalibrationConstraint] = []
    if prior_data.times.size < 3:
        return constraints

    for left_index in range(1, prior_data.times.size - 1):
        right_index = left_index + 1
        t_left = float(prior_data.times[left_index])
        t_right = float(prior_data.times[right_index])

        spots_left = np.asarray(prior_data.path_matrix[:, left_index], dtype=float)
        spots_right = np.asarray(prior_data.path_matrix[:, right_index], dtype=float)
        forward_left = forward_price(t_left)
        forward_right = forward_price(t_right)

        std_left = float(np.std(spots_left))
        lower_bound = max(
            0.0,
            min(float(K[0]), forward_left - config.martingale_std_multiplier * std_left),
        )
        upper_bound = max(
            float(K[-1]),
            forward_left + config.martingale_std_multiplier * std_left,
        )
        window_edges = _build_martingale_window_edges(
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            strikes=K,
        )

        for window_index in range(window_edges.size - 1):
            left_edge = float(window_edges[window_index])
            right_edge = float(window_edges[window_index + 1])

            if window_index == window_edges.size - 2:
                mask = (spots_left >= left_edge) & (spots_left <= right_edge)
            else:
                mask = (spots_left >= left_edge) & (spots_left < right_edge)

            n_paths_window = int(np.count_nonzero(mask))
            if n_paths_window < config.martingale_min_paths_per_window:
                continue

            payoff = (
                (spots_right * (forward_left / forward_right)) - spots_left
            ) * mask.astype(float)
            variance = _constraint_variance(
                payoffs=payoff,
                pesos_prior=prior_data.prior_weights,
            )
            if variance < config.martingale_min_variance:
                continue

            constraints.append(
                CalibrationConstraint(
                    name=(
                        f"martingale_T{t_left:.6f}_T{t_right:.6f}_"
                        f"[{left_edge:.4f},{right_edge:.4f}]"
                    ),
                    kind=CONSTRAINT_KIND_MARTINGALE,
                    payoffs=np.asarray(payoff, dtype=float),
                    target_value=0.0,
                    least_squares_weight=float(config.wls_weight_martingale),
                )
            )

    return constraints


def build_constraint_set(
    prior_data: PriorSimulationData,
    config: RelativeEntropyCalibrationConfig,
) -> ConstraintSet:
    """
    Construye el sistema completo de restricciones G, C y omega.
    """
    _validate_calibration_mode(config.calibration_mode)

    constraints: list[CalibrationConstraint] = []
    option_constraints = _build_option_constraints(prior_data=prior_data, config=config)
    constraints.extend(option_constraints)

    forward_constraints: list[CalibrationConstraint] = []
    if config.include_forward_constraints:
        forward_constraints = _build_forward_constraints(
            prior_data=prior_data,
            config=config,
        )
        constraints.extend(forward_constraints)

    martingale_constraints: list[CalibrationConstraint] = []
    if config.include_martingale_constraints:
        martingale_constraints = _build_martingale_constraints(
            prior_data=prior_data,
            config=config,
        )
        constraints.extend(martingale_constraints)

    if not constraints:
        raise ValueError("No hay restricciones disponibles para calibrar.")

    matrix = np.column_stack([constraint.payoffs for constraint in constraints]).astype(float)
    targets = np.array([constraint.target_value for constraint in constraints], dtype=float)
    least_squares_weights = np.array(
        [constraint.least_squares_weight for constraint in constraints],
        dtype=float,
    )
    if config.calibration_mode == CALIBRATION_MODE_EXACT:
        least_squares_weights = np.zeros_like(least_squares_weights)

    return ConstraintSet(
        matrix=matrix,
        targets=targets,
        least_squares_weights=least_squares_weights,
        names=tuple(constraint.name for constraint in constraints),
        kinds=tuple(constraint.kind for constraint in constraints),
        n_option_constraints=len(option_constraints),
        n_forward_constraints=len(forward_constraints),
        n_martingale_constraints=len(martingale_constraints),
    )


def dual_objective_and_gradient(
    lambdas: np.ndarray,
    matriz_payoffs: np.ndarray,
    pesos_prior: np.ndarray,
    precios_market: np.ndarray,
    least_squares_weights: np.ndarray | None = None,
) -> tuple[float, np.ndarray]:
    """
    Evalúa la función dual generalizada y su gradiente.

    Exacta:
        W(lambda) = log Z(lambda) - lambda · C

    WLS:
        H(lambda) = W(lambda) + 0.5 * sum_j omega_j * lambda_j^2
    """
    pesos_posteriores, log_particion = calcular_pesos_posteriores(
        lambdas=lambdas,
        matriz_payoffs=matriz_payoffs,
        pesos_prior=pesos_prior,
    )
    _, gradiente = calcular_precios_modelo_y_gradiente(
        matriz_payoffs=matriz_payoffs,
        pesos_posteriores=pesos_posteriores,
        precios_market=precios_market,
        lambdas=lambdas,
        least_squares_weights=least_squares_weights,
    )
    valor_objetivo = objetivo_dual(
        lambdas=lambdas,
        matriz_payoffs=matriz_payoffs,
        pesos_prior=pesos_prior,
        precios_market=precios_market,
        least_squares_weights=least_squares_weights,
        log_particion=log_particion,
    )
    return valor_objetivo, gradiente


def calcular_pesos_posteriores(
    lambdas: np.ndarray,
    matriz_payoffs: np.ndarray,
    pesos_prior: np.ndarray,
) -> tuple[np.ndarray, float]:
    """
    Recupera los pesos q_i a partir de los multiplicadores óptimos.
    """
    puntuaciones_trayectoria = np.asarray(matriz_payoffs, dtype=float) @ np.asarray(
        lambdas,
        dtype=float,
    )

    max_puntuacion = float(np.max(puntuaciones_trayectoria))
    suma_particion = float(
        np.sum(
            pesos_prior * np.exp(puntuaciones_trayectoria - max_puntuacion)
        )
    )
    log_particion = max_puntuacion + np.log(suma_particion)
    pesos_posteriores = pesos_prior * np.exp(puntuaciones_trayectoria - log_particion)
    return np.asarray(pesos_posteriores, dtype=float), float(log_particion)


def calcular_precios_modelo_y_gradiente(
    matriz_payoffs: np.ndarray,
    pesos_posteriores: np.ndarray,
    precios_market: np.ndarray,
    lambdas: np.ndarray | None = None,
    least_squares_weights: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calcula precios modelo bajo q y el gradiente de la función dual.
    """
    precios_modelo = np.asarray(matriz_payoffs, dtype=float).T @ np.asarray(
        pesos_posteriores,
        dtype=float,
    )

    gradiente = precios_modelo - np.asarray(precios_market, dtype=float)
    if least_squares_weights is not None and lambdas is not None:
        gradiente = gradiente + np.asarray(least_squares_weights, dtype=float) * np.asarray(
            lambdas,
            dtype=float,
        )

    return np.asarray(precios_modelo, dtype=float), np.asarray(gradiente, dtype=float)


def calcular_precios_por_gtq(
    matriz_payoffs: np.ndarray,
    pesos_posteriores: np.ndarray,
) -> np.ndarray:
    """
    Calcula G^T q de forma explícita.
    """
    return np.asarray(matriz_payoffs, dtype=float).T @ np.asarray(
        pesos_posteriores,
        dtype=float,
    )


def calcular_jacobiano(
    matriz_payoffs: np.ndarray,
    pesos_posteriores: np.ndarray,
    precios_modelo: np.ndarray,
    least_squares_weights: np.ndarray | None = None,
) -> np.ndarray:
    """
    Calcula el Jacobiano:

    exacto:
        J_ab = Cov_q(g_a, g_b)

    WLS:
        J_ab = Cov_q(g_a, g_b) + omega_a * 1{a=b}
    """
    centered = np.asarray(matriz_payoffs, dtype=float) - np.asarray(
        precios_modelo,
        dtype=float,
    )[None, :]
    jacobiano = centered.T @ (centered * np.asarray(pesos_posteriores, dtype=float)[:, None])

    if least_squares_weights is not None:
        jacobiano = jacobiano + np.diag(np.asarray(least_squares_weights, dtype=float))

    return np.asarray(jacobiano, dtype=float)


def resolver_paso_newton(
    jacobiano: np.ndarray,
    gradiente: np.ndarray,
) -> np.ndarray:
    """
    Resuelve J * paso = gradiente con una regularización suave si hace falta.
    """
    n_constraints = len(gradiente)
    identidad = np.eye(n_constraints, dtype=float)
    regularizacion = 0.0

    for _ in range(8):
        try:
            if regularizacion == 0.0:
                return np.linalg.solve(jacobiano, gradiente)
            return np.linalg.solve(jacobiano + regularizacion * identidad, gradiente)
        except np.linalg.LinAlgError:
            regularizacion = 1e-10 if regularizacion == 0.0 else 10.0 * regularizacion

    return np.linalg.lstsq(
        jacobiano + regularizacion * identidad,
        gradiente,
        rcond=None,
    )[0]


def resolver_paso_newton_exacto(
    jacobiano: np.ndarray,
    gradiente: np.ndarray,
    eigenvalue_tolerance: float,
) -> np.ndarray:
    """
    Resuelve el paso exacto con una pseudo-inversa espectral.

    En modo exacto no existe la diagonal WLS que regulariza el Jacobiano. Las
    opciones ITM y las restricciones de forward pueden dejar covarianzas casi
    singulares, por lo que se proyecta el paso sobre los autovalores positivos.
    """
    matriz = np.asarray(jacobiano, dtype=float)
    matriz = 0.5 * (matriz + matriz.T)
    gradiente_array = np.asarray(gradiente, dtype=float)

    try:
        autovalores, autovectores = np.linalg.eigh(matriz)
    except np.linalg.LinAlgError:
        return np.asarray(gradiente_array, dtype=float)

    max_autovalor = max(float(np.max(autovalores)), 0.0)
    tolerancia = max(1e-12, float(eigenvalue_tolerance) * max_autovalor)
    mascara = autovalores > tolerancia

    if np.any(mascara):
        base = autovectores[:, mascara]
        paso = base @ ((base.T @ gradiente_array) / autovalores[mascara])
    else:
        paso = gradiente_array.copy()

    if not np.all(np.isfinite(paso)):
        paso = gradiente_array.copy()
    if producto_escalar(gradiente_array, paso) <= 0.0:
        paso = gradiente_array.copy()

    return np.asarray(paso, dtype=float)


def norma_infinito(vector: np.ndarray) -> float:
    """
    Devuelve la norma infinito de un vector.
    """
    return float(np.max(np.abs(np.asarray(vector, dtype=float))))


def producto_escalar(x: np.ndarray, y: np.ndarray) -> float:
    """
    Calcula un producto escalar.
    """
    return float(np.dot(np.asarray(x, dtype=float), np.asarray(y, dtype=float)))


def objetivo_dual(
    lambdas: np.ndarray,
    matriz_payoffs: np.ndarray,
    pesos_prior: np.ndarray,
    precios_market: np.ndarray,
    least_squares_weights: np.ndarray | None = None,
    log_particion: float | None = None,
) -> float:
    """
    Evalúa la función dual exacta o WLS.
    """
    if log_particion is None:
        _, log_particion = calcular_pesos_posteriores(
            lambdas=lambdas,
            matriz_payoffs=matriz_payoffs,
            pesos_prior=pesos_prior,
        )

    valor = float(log_particion - producto_escalar(lambdas, precios_market))
    if least_squares_weights is not None:
        valor += 0.5 * producto_escalar(
            np.asarray(least_squares_weights, dtype=float),
            np.asarray(lambdas, dtype=float) ** 2,
        )
    return float(valor)


def relative_entropy_distance(
    pesos_posteriores: np.ndarray,
    pesos_prior: np.ndarray,
) -> float:
    """
    Calcula D(q|p) = sum_i q_i log(q_i / p_i).
    """
    positive_mask = (pesos_posteriores > 0.0) & (pesos_prior > 0.0)
    q_post = pesos_posteriores[positive_mask]
    p_prior = pesos_prior[positive_mask]
    return float(np.sum(q_post * np.log(q_post / p_prior)))


def _max_error_for_kind(
    errores: np.ndarray,
    kinds: tuple[str, ...],
    target_kind: str,
) -> float:
    indices = [index for index, kind in enumerate(kinds) if kind == target_kind]
    if not indices:
        return 0.0
    errores_kind = np.asarray(errores, dtype=float)[indices]
    return float(np.max(np.abs(errores_kind)))


def calibrate_relative_entropy_weights(
    matriz_payoffs: np.ndarray,
    pesos_prior: np.ndarray,
    precios_market: np.ndarray,
    config: RelativeEntropyCalibrationConfig,
    least_squares_weights: np.ndarray | None = None,
    constraint_kinds: tuple[str, ...] | None = None,
    constraint_names: tuple[str, ...] | None = None,
) -> RelativeEntropyCalibrationResult:
    """
    Calibra los pesos q_i minimizando la entropía relativa al prior.
    """
    _validate_calibration_mode(config.calibration_mode)

    n_paths, n_constraints = matriz_payoffs.shape
    lambdas = np.zeros(n_constraints, dtype=float)
    alpha = config.alpha_inicial
    iteracion_final = 0
    mensaje = "Numero maximo de iteraciones alcanzado."
    optimizador_exito = False

    if least_squares_weights is None:
        least_squares_weights = np.zeros(n_constraints, dtype=float)
    else:
        least_squares_weights = np.asarray(least_squares_weights, dtype=float)

    if constraint_kinds is None:
        constraint_kinds = tuple(CONSTRAINT_KIND_OPTION for _ in range(n_constraints))
    if constraint_names is None:
        constraint_names = tuple(f"constraint_{index}" for index in range(n_constraints))

    usa_wls = config.calibration_mode == CALIBRATION_MODE_WLS

    for iteracion in range(1, config.maxiter + 1):
        iteracion_final = iteracion

        pesos_posteriores, log_particion = calcular_pesos_posteriores(
            lambdas=lambdas,
            matriz_payoffs=matriz_payoffs,
            pesos_prior=pesos_prior,
        )
        precios_modelo, gradiente = calcular_precios_modelo_y_gradiente(
            matriz_payoffs=matriz_payoffs,
            pesos_posteriores=pesos_posteriores,
            precios_market=precios_market,
            lambdas=lambdas,
            least_squares_weights=least_squares_weights,
        )
        jacobiano = calcular_jacobiano(
            matriz_payoffs=matriz_payoffs,
            pesos_posteriores=pesos_posteriores,
            precios_modelo=precios_modelo,
            least_squares_weights=least_squares_weights,
        )

        norma_gradiente = norma_infinito(gradiente)
        max_abs_error = norma_infinito(precios_modelo - precios_market)
        if norma_gradiente <= config.gradient_tolerance or (
            not usa_wls and max_abs_error <= config.pricing_tolerance
        ):
            mensaje = "Gradiente y restricciones dentro de tolerancia."
            optimizador_exito = True
            break

        if usa_wls:
            paso_newton = resolver_paso_newton(
                jacobiano=jacobiano,
                gradiente=gradiente,
            )
        else:
            paso_newton = resolver_paso_newton_exacto(
                jacobiano=jacobiano,
                gradiente=gradiente,
                eigenvalue_tolerance=config.exact_newton_eigenvalue_tolerance,
            )
        condicion_actual = float(np.linalg.cond(jacobiano))
        valor_objetivo_actual = objetivo_dual(
            lambdas=lambdas,
            matriz_payoffs=matriz_payoffs,
            pesos_prior=pesos_prior,
            precios_market=precios_market,
            least_squares_weights=least_squares_weights,
            log_particion=log_particion,
        )
        alpha_local = alpha
        aceptado_paso = False

        for _ in range(config.max_ajustes_alpha):
            lambdas_candidatas = lambdas - alpha_local * paso_newton
            pesos_candidatos, log_particion_candidato = calcular_pesos_posteriores(
                lambdas=lambdas_candidatas,
                matriz_payoffs=matriz_payoffs,
                pesos_prior=pesos_prior,
            )
            precios_modelo_candidatos, _ = calcular_precios_modelo_y_gradiente(
                matriz_payoffs=matriz_payoffs,
                pesos_posteriores=pesos_candidatos,
                precios_market=precios_market,
                lambdas=lambdas_candidatas,
                least_squares_weights=least_squares_weights,
            )
            jacobiano_candidato = calcular_jacobiano(
                matriz_payoffs=matriz_payoffs,
                pesos_posteriores=pesos_candidatos,
                precios_modelo=precios_modelo_candidatos,
                least_squares_weights=least_squares_weights,
            )
            condicion_candidata = float(np.linalg.cond(jacobiano_candidato))
            valor_objetivo_candidato = objetivo_dual(
                lambdas=lambdas_candidatas,
                matriz_payoffs=matriz_payoffs,
                pesos_prior=pesos_prior,
                precios_market=precios_market,
                least_squares_weights=least_squares_weights,
                log_particion=log_particion_candidato,
            )

            if not np.isfinite(valor_objetivo_candidato):
                alpha_local /= config.factor_reduccion_alpha
                continue

            if valor_objetivo_candidato > valor_objetivo_actual + 1e-12:
                alpha_local /= config.factor_reduccion_alpha
                continue

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

    pesos_posteriores, log_particion = calcular_pesos_posteriores(
        lambdas=lambdas,
        matriz_payoffs=matriz_payoffs,
        pesos_prior=pesos_prior,
    )
    precios_modelo, gradiente = calcular_precios_modelo_y_gradiente(
        matriz_payoffs=matriz_payoffs,
        pesos_posteriores=pesos_posteriores,
        precios_market=precios_market,
        lambdas=lambdas,
        least_squares_weights=least_squares_weights,
    )
    errores_precio = np.asarray(precios_modelo - precios_market, dtype=float)
    precios_replicados_gtq = calcular_precios_por_gtq(
        matriz_payoffs=matriz_payoffs,
        pesos_posteriores=pesos_posteriores,
    )
    error_consistencia_gtq = float(
        np.max(np.abs(precios_replicados_gtq - precios_modelo))
    )

    max_abs_error_total = norma_infinito(errores_precio)
    norma_gradiente = norma_infinito(gradiente)
    aceptado = bool(
        optimizador_exito
        or (not usa_wls and max_abs_error_total <= config.pricing_tolerance)
        or (usa_wls and norma_gradiente <= config.gradient_tolerance)
    )
    entropia_relativa = relative_entropy_distance(
        pesos_posteriores=pesos_posteriores,
        pesos_prior=pesos_prior,
    )
    valor_objetivo = objetivo_dual(
        lambdas=lambdas,
        matriz_payoffs=matriz_payoffs,
        pesos_prior=pesos_prior,
        precios_market=precios_market,
        least_squares_weights=least_squares_weights,
        log_particion=log_particion,
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
        n_constraints=n_constraints,
        n_option_constraints=sum(
            1 for kind in constraint_kinds if kind == CONSTRAINT_KIND_OPTION
        ),
        n_forward_constraints=sum(
            1 for kind in constraint_kinds if kind == CONSTRAINT_KIND_FORWARD
        ),
        n_martingale_constraints=sum(
            1 for kind in constraint_kinds if kind == CONSTRAINT_KIND_MARTINGALE
        ),
        calibration_mode=config.calibration_mode,
        optimizador_exito=optimizador_exito,
        optimizador_mensaje=mensaje,
        aceptado=aceptado,
        iteraciones=iteracion_final,
        norma_gradiente=norma_gradiente,
        max_abs_error_total=max_abs_error_total,
        max_abs_error_options=_max_error_for_kind(
            errores=errores_precio,
            kinds=constraint_kinds,
            target_kind=CONSTRAINT_KIND_OPTION,
        ),
        max_abs_error_forwards=_max_error_for_kind(
            errores=errores_precio,
            kinds=constraint_kinds,
            target_kind=CONSTRAINT_KIND_FORWARD,
        ),
        max_abs_error_martingale=_max_error_for_kind(
            errores=errores_precio,
            kinds=constraint_kinds,
            target_kind=CONSTRAINT_KIND_MARTINGALE,
        ),
        least_squares_weights=least_squares_weights,
        constraint_targets=np.asarray(precios_market, dtype=float),
        constraint_kinds=constraint_kinds,
        constraint_names=constraint_names,
    )


def main() -> None:
    config = RelativeEntropyCalibrationConfig()
    prior_path = Path(__file__).resolve().with_name(config.prior_filename)

    prior_data = load_prior_simulation(prior_path)
    constraint_set = build_constraint_set(prior_data=prior_data, config=config)

    calibracion = calibrate_relative_entropy_weights(
        matriz_payoffs=constraint_set.matrix,
        pesos_prior=prior_data.prior_weights,
        precios_market=constraint_set.targets,
        config=config,
        least_squares_weights=constraint_set.least_squares_weights,
        constraint_kinds=constraint_set.kinds,
        constraint_names=constraint_set.names,
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
                "precios_market": calibracion.constraint_targets,
                "errores_precio": calibracion.errores_precio,
                "error_consistencia_gtq": calibracion.error_consistencia_gtq,
                "entropia_relativa": calibracion.entropia_relativa,
                "valor_objetivo": calibracion.valor_objetivo,
                "iteraciones": calibracion.iteraciones,
                "norma_gradiente": calibracion.norma_gradiente,
                "modo_calibracion": calibracion.calibration_mode,
                "n_constraints": calibracion.n_constraints,
                "n_option_constraints": calibracion.n_option_constraints,
                "n_forward_constraints": calibracion.n_forward_constraints,
                "n_martingale_constraints": calibracion.n_martingale_constraints,
                "max_abs_error_total": calibracion.max_abs_error_total,
                "max_abs_error_options": calibracion.max_abs_error_options,
                "max_abs_error_forwards": calibracion.max_abs_error_forwards,
                "max_abs_error_martingale": calibracion.max_abs_error_martingale,
            },
        )
    else:
        export_path = None

    print(
        f"Calibracion completada con n = {calibracion.n_paths} trayectorias "
        f"y N = {calibracion.n_constraints} restricciones."
    )
    print(
        f"Modo = {calibracion.calibration_mode}, "
        f"opciones = {calibracion.n_option_constraints}, "
        f"forwards = {calibracion.n_forward_constraints}, "
        f"martingala = {calibracion.n_martingale_constraints}"
    )
    print(
        f"Max abs total = {calibracion.max_abs_error_total:.{DISPLAY_DECIMALS}f}, "
        f"opciones = {calibracion.max_abs_error_options:.{DISPLAY_DECIMALS}f}, "
        f"forwards = {calibracion.max_abs_error_forwards:.{DISPLAY_DECIMALS}f}, "
        f"martingala = {calibracion.max_abs_error_martingale:.{DISPLAY_DECIMALS}f}"
    )
    print(
        f"D(q|p) = {calibracion.entropia_relativa:.{DISPLAY_DECIMALS}f}, "
        f"iteraciones = {calibracion.iteraciones}, "
        f"norma gradiente = {calibracion.norma_gradiente:.{DISPLAY_DECIMALS}f}"
    )
    print(
        f"Estado optimizador: {'aceptado' if calibracion.aceptado else 'no aceptado'} "
        f"({calibracion.optimizador_mensaje})"
    )
    print(
        f"Chequeo G^T q: max |G^T q - precios_modelo| = "
        f"{calibracion.error_consistencia_gtq:.{DISPLAY_DECIMALS}f}"
    )
    if export_path is not None:
        print(f"Pesos q_i y resultados guardados como variables en {export_path.name}.")


if __name__ == "__main__":
    main()
