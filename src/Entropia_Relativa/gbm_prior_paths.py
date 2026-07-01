"""
Simulación de trayectorias previas GBM y matriz de payoffs descontados.

Este script sigue el primer paso del enfoque de Weighted Monte Carlo:
1. simular trayectorias del subyacente bajo una medida previa,
2. calcular los payoffs descontados de los instrumentos benchmark
   sobre cada trayectoria simulada.

La medida previa elegida aquí es un GBM risk-neutral:

    dS_t = (r - q) S_t dt + sigma S_t dW_t

Las trayectorias se simulan exactamente en las fechas de vencimiento de
los benchmarks definidas en ``data_loader.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
from scipy.optimize import brentq

from common.types import OptionType
from Entropia_Relativa.data_loader import (
    BENCHMARK_OPTIONS,
    S0,
    T,
    BenchmarkOption,
    include_spot_column,
    n_paths,
    q,
    r,
    seed,
    sigma_prior,
    use_antithetic,
    use_atmf_term_structure_prior,
)
from Entropia_Relativa.python_variable_io import export_python_variables_module
from Stochastic_Local_Volatility.core.pricing_Black_Scholes import (
    BS_Call_Put_Option_Price_tau,
)


DISPLAY_DECIMALS = 2


@dataclass(frozen=True)
class PriorGBMConfig:
    """
    Configuración de la simulación bajo la medida previa.
    """

    n_paths: int = n_paths
    sigma: float = sigma_prior
    use_atmf_term_structure: bool = use_atmf_term_structure_prior
    seed: int = seed
    use_antithetic: bool = use_antithetic
    include_spot_column: bool = include_spot_column
    export_results: bool = True
    export_filename: str = "gbm_prior_simulation_results.py"


@dataclass(frozen=True)
class PriorSimulationResult:
    """
    Resultado completo de la simulación bajo la medida previa.
    """

    times: np.ndarray
    path_matrix: np.ndarray
    discounted_payoff_matrix: np.ndarray
    prior_weights: np.ndarray
    benchmark_prices_prior: np.ndarray


def _validate_time_grid(times: np.ndarray) -> np.ndarray:
    """
    Valida y devuelve una malla temporal estrictamente creciente.
    """
    time_grid = np.asarray(times, dtype=float).reshape(-1)
    if time_grid.size == 0:
        raise ValueError("La malla temporal no puede estar vacía.")
    if np.any(time_grid <= 0.0):
        raise ValueError("Todas las fechas de simulación deben ser positivas.")
    if np.any(np.diff(time_grid) <= 0.0):
        raise ValueError("La malla temporal debe ser estrictamente creciente.")
    return time_grid


def _normal_draws(
    n_paths: int,
    n_steps: int,
    seed: int,
    use_antithetic: bool,
) -> np.ndarray:
    """
    Genera números aleatorios estándar para la simulación de trayectorias.
    Args:
        n_paths (int): Número total de trayectorias a simular.
        n_steps (int): Número de pasos de simulación.
        seed (int): Semilla para la generación de números aleatorios.
        use_antithetic (bool): Indica si se deben usar variables antitéticas.

    Returns:
        np.ndarray: Matriz de números aleatorios estándar.
    """
    rng = np.random.default_rng(seed)       # Números aleatorios para Monte Carlo

    if use_antithetic:
        n_base = (n_paths + 1) // 2
        z_base = rng.standard_normal((n_base, n_steps))
        z = np.vstack((z_base, -z_base))
        return z[:n_paths, :]

    return rng.standard_normal((n_paths, n_steps))


def _implied_vol_from_market_price(
    CP: OptionType,
    market_price: float,
    strike: float,
    maturity: float,
    fallback_sigma: float,
) -> float:
    intrinsic = max(
        S0 * np.exp(-q * maturity) - strike * np.exp(-r * maturity),
        0.0,
    ) if CP == OptionType.CALL else max(
        strike * np.exp(-r * maturity) - S0 * np.exp(-q * maturity),
        0.0,
    )
    safe_price = max(float(market_price), float(intrinsic) + 1e-12)

    def objective(vol: float) -> float:
        price = BS_Call_Put_Option_Price_tau(
            CP=CP,
            S_0=float(S0),
            K=float(strike),
            sigma=float(vol),
            tau=float(maturity),
            r=float(r),
        )
        return float(price - safe_price)

    try:
        return float(brentq(objective, 1e-8, 5.0, maxiter=500))
    except ValueError:
        return float(fallback_sigma)


def _infer_total_variance_curve_from_benchmarks(
    benchmarks: list[BenchmarkOption],
    fallback_sigma: float,
) -> tuple[np.ndarray, np.ndarray]:
    benchmark_times = np.array(sorted({float(b.maturity) for b in benchmarks}), dtype=float)
    total_variances = np.empty_like(benchmark_times)

    for idx, maturity in enumerate(benchmark_times):
        benchmarks_at_maturity = [
            benchmark for benchmark in benchmarks if np.isclose(benchmark.maturity, maturity)
        ]
        forward = float(S0 * np.exp((r - q) * maturity))
        atmf_benchmark = min(
            benchmarks_at_maturity,
            key=lambda benchmark: abs(float(benchmark.strike) - forward),
        )
        implied_vol = _implied_vol_from_market_price(
            CP=atmf_benchmark.CP,
            market_price=atmf_benchmark.market_price,
            strike=atmf_benchmark.strike,
            maturity=maturity,
            fallback_sigma=fallback_sigma,
        )
        total_variances[idx] = float(implied_vol * implied_vol * maturity)

    total_variances = np.maximum.accumulate(total_variances)
    return benchmark_times, total_variances


def _interval_sigmas_from_atmf_curve(
    observation_times: np.ndarray,
    benchmarks: list[BenchmarkOption],
    fallback_sigma: float,
) -> np.ndarray:
    benchmark_times, total_variances = _infer_total_variance_curve_from_benchmarks(
        benchmarks=benchmarks,
        fallback_sigma=fallback_sigma,
    )
    total_variance_observations = np.interp(
        observation_times,
        benchmark_times,
        total_variances,
    )
    dt = np.diff(np.concatenate(([0.0], observation_times)))
    interval_variances = np.diff(
        np.concatenate(([0.0], total_variance_observations))
    ) / dt
    interval_variances = np.maximum(interval_variances, 0.0)
    return np.sqrt(interval_variances)


def simulate_gbm_prior_paths(
    config: PriorGBMConfig,
    times: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Simula trayectorias del subyacente bajo un GBM risk-neutral.

    Args:
        config: Configuración de la simulación.
        times: Fechas donde se quiere observar el subyacente. Si no se
            indica, se usan los vencimientos benchmark ``T``.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - vector de tiempos de la matriz devuelta,
            - matriz ``S_paths`` de forma ``(n_paths, n_times)``.
    """
    observation_times = _validate_time_grid(T if times is None else times)  # Fechas de observación para la simulación
    dt = np.diff(np.concatenate(([0.0], observation_times)))                # Incrementos de tiempo entre observaciones

    z = _normal_draws(
        n_paths=config.n_paths,
        n_steps=observation_times.size,
        seed=config.seed,
        use_antithetic=config.use_antithetic,
    )                                                           # Números aleatorios estándar para la simulación    

    if config.use_atmf_term_structure:
        interval_sigmas = _interval_sigmas_from_atmf_curve(
            observation_times=observation_times,
            benchmarks=BENCHMARK_OPTIONS,
            fallback_sigma=config.sigma,
        )
    else:
        interval_sigmas = np.full(observation_times.size, config.sigma, dtype=float)

    drift = (r - q - 0.5 * interval_sigmas * interval_sigmas) * dt
    diffusion = interval_sigmas * np.sqrt(dt)
    log_increments = drift[None, :] + diffusion[None, :] * z    # Incrementos logarítmicos para cada trayectoria y paso de tiempo
    log_paths = np.cumsum(log_increments, axis=1)               # Trayectorias logarítmicas acumuladas (log(S_t / S0))
    simulated_prices = S0 * np.exp(log_paths)          # Trayectorias simuladas de precios (S_t = S0 * exp(log(S_t / S0)))      

    if config.include_spot_column:
        times_out = np.concatenate(([0.0], observation_times))
        path_matrix = np.column_stack((np.full(config.n_paths, S0, dtype=float), simulated_prices))
    else:
        times_out = observation_times
        path_matrix = simulated_prices

    return times_out, path_matrix


def pathwise_discounted_payoff(
    CP: OptionType,
    terminal_prices: np.ndarray,
    strike: float,
    maturity: float,
) -> np.ndarray:
    """
    Calcula el payoff descontado de una opción sobre cada trayectoria simulada.
    Args:
        CP (OptionType): Tipo de opción (CALL o PUT).
        terminal_prices (np.ndarray): Precios terminales de las trayectorias.
        strike (float): Precio de ejercicio.
        maturity (float): Tiempo de vencimiento.

    Raises:
        ValueError: _description_

    Returns:
        np.ndarray: _description_
    """
    if CP == OptionType.CALL:
        intrinsic = np.maximum(terminal_prices - strike, 0.0)
    elif CP == OptionType.PUT:
        intrinsic = np.maximum(strike - terminal_prices, 0.0)
    else:
        raise ValueError("OptionType debe ser CALL o PUT.")

    return np.exp(-r * maturity) * intrinsic


def build_discounted_payoff_matrix(
    path_matrix: np.ndarray,
    times: np.ndarray,
    benchmarks: list[BenchmarkOption] | None = None,
) -> np.ndarray:
    """
    Construye la matriz de payoffs descontados para cada trayectoria y cada benchmark.
    Args:
        path_matrix (np.ndarray): Matriz de trayectorias simuladas con forma (n_paths, n_times).
        times (np.ndarray): Vector de tiempos correspondiente a las columnas de path_matrix.
        benchmarks (list[BenchmarkOption] | None, optional): Lista de opciones benchmark. Defaults to None.

    Raises:
        ValueError: Si algún vencimiento de los benchmarks no está presente en la malla temporal de las trayectorias.

    Returns:
        np.ndarray: Matriz de payoffs descontados con forma (n_paths, n_benchmarks).
    """
    if benchmarks is None:
        benchmarks = BENCHMARK_OPTIONS

    time_grid = np.asarray(times, dtype=float).reshape(-1)
    n_paths = path_matrix.shape[0]
    payoff_matrix = np.empty((n_paths, len(benchmarks)), dtype=float)

    for j, benchmark in enumerate(benchmarks):
        maturity = float(benchmark.maturity)
        matches = np.where(np.isclose(time_grid, maturity))[0]
        if matches.size == 0:
            raise ValueError(
                f"El vencimiento {maturity} no está presente en la matriz de trayectorias."
            )
        terminal_prices = path_matrix[:, int(matches[0])]
        payoff_matrix[:, j] = pathwise_discounted_payoff(
            CP=benchmark.CP,
            terminal_prices=terminal_prices,
            strike=benchmark.strike,
            maturity=maturity,
        )

    return payoff_matrix


def simulate_prior_and_benchmarks(
    config: PriorGBMConfig,
    benchmarks: list[BenchmarkOption] | None = None,
) -> PriorSimulationResult:
    """
    Simula trayectorias bajo la medida previa y calcula los payoffs descontados de los benchmarks.
    Args:
        config (PriorGBMConfig): Configuración de la simulación previa.
        benchmarks (list[BenchmarkOption] | None, optional): Lista de opciones benchmark. Defaults to None.

    Returns:
        PriorSimulationResult: Objeto con los resultados de la simulación, incluyendo:
            - times (np.ndarray): Vector de tiempos.
            - path_matrix (np.ndarray): Matriz de trayectorias simuladas.
            - discounted_payoff_matrix (np.ndarray): Matriz de payoffs descontados.
            - prior_weights (np.ndarray): Pesos previos para cada trayectoria.
            - benchmark_prices_prior (np.ndarray): Precios benchmark bajo la medida previa.

    """
    if benchmarks is None:
        benchmarks = BENCHMARK_OPTIONS

    times, path_matrix = simulate_gbm_prior_paths(config=config)
    payoff_matrix = build_discounted_payoff_matrix(
        path_matrix=path_matrix,
        times=times,
        benchmarks=benchmarks,
    )
    prior_weights = np.full(config.n_paths, 1.0 / config.n_paths, dtype=float)
    benchmark_prices_prior = payoff_matrix.mean(axis=0)

    return PriorSimulationResult(
        times=times,
        path_matrix=path_matrix,
        discounted_payoff_matrix=payoff_matrix,
        prior_weights=prior_weights,
        benchmark_prices_prior=benchmark_prices_prior,
    )


def _format_array_preview(x: np.ndarray, n_cols: int = 4) -> str:
    """
    Formatea una vista previa de un array para impresión.
    Args:
        x (np.ndarray): Array a formatear.
        n_cols (int, optional): Número de columnas a mostrar. Defaults to 4.

    Returns:
        str: Vista previa del array formateado.
    """
    preview = np.asarray(x, dtype=float).reshape(-1)[:n_cols]
    return ", ".join(f"{value:.{DISPLAY_DECIMALS}f}" for value in preview)


def main() -> None:
    config = PriorGBMConfig(
        n_paths=500,
        sigma=0.20,
        seed=12345,
        use_antithetic=True,
        include_spot_column=True,
        export_results=True,
    )

    result = simulate_prior_and_benchmarks(config)
    n = result.path_matrix.shape[0]
    m = len(T)

    export_path = None
    if config.export_results:
        export_path = Path(__file__).resolve().with_name(config.export_filename)
        export_python_variables_module(
            module_path=export_path,
            variables={
                "times": result.times,
                "path_matrix": result.path_matrix,
                "discounted_payoff_matrix": result.discounted_payoff_matrix,
                "prior_weights": result.prior_weights,
                "benchmark_prices_prior": result.benchmark_prices_prior,
            },
        )

    mae = np.mean(
        np.abs(
            result.benchmark_prices_prior
            - np.array([benchmark.market_price for benchmark in BENCHMARK_OPTIONS], dtype=float)
        )
    ) # Error absoluto medio entre los precios benchmark bajo la medida previa y los precios de mercado.
    print(f"Usando n = {n} trayectorias y m = {m} steps de simulación.")
    if export_path is not None:
        print(
            f"Simulación completada. Variables guardadas en {export_path.name}."
        )
    else:
        print("Simulación completada. Las matrices quedan disponibles en memoria.")
    print(f"Diagnóstico previo: Error absoluto medio benchmark = {mae:.{DISPLAY_DECIMALS}f}")


if __name__ == "__main__":
    main()
