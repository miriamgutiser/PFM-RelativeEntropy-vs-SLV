"""
Pricing de opciones europeas con los pesos calibrados por entropía relativa.

Una vez calibrados los pesos q_i sobre las trayectorias simuladas, este
script los utiliza para calcular precios de opciones con strikes arbitrarios
(no necesariamente los benchmark). La fórmula es:

    Precio = sum_{i=1}^{N} q_i * g_i

donde g_i es el payoff descontado de la opción en la trayectoria i.

Se comparan los precios obtenidos con:
- pesos calibrados q_i  (WMC),
- pesos uniformes 1/N   (Monte Carlo clásico).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from common.types import OptionType
from Entropia_Relativa.data_loader import (
    BENCHMARK_OPTIONS,
    S0,
    T,
    K,
    r,
)
from Entropia_Relativa.gbm_prior_paths import pathwise_discounted_payoff
from Entropia_Relativa.python_variable_io import load_python_variables_module


DISPLAY_DECIMALS = 7


@dataclass(frozen=True)
class PricingConfig:
    """
    Configuración del pricing de opciones.
    """

    prior_filename: str = "gbm_prior_simulation_results.py"
    calibration_filename: str = "relative_entropy_calibration_results.py"
    pricing_strikes: np.ndarray | None = None
    pricing_maturities: np.ndarray | None = None
    option_type: OptionType = OptionType.CALL


@dataclass(frozen=True)
class PricingResult:
    """
    Resultado del pricing de una opción.
    """

    CP: OptionType
    strike: float
    maturity: float
    price_wmc: float
    price_mc: float
    is_benchmark: bool


def load_simulation_and_weights(
    prior_path: Path,
    calibration_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Carga las trayectorias simuladas y los pesos calibrados.

    Returns:
        tuple con:
            - path_matrix (N x n_times): precios simulados en cada fecha,
            - times: vector de tiempos,
            - pesos_posteriores: pesos calibrados q_i,
            - pesos_prior: pesos uniformes 1/N.
    """
    prior_data = load_python_variables_module(prior_path)
    path_matrix = np.asarray(prior_data.path_matrix, dtype=float)
    times = np.asarray(prior_data.times, dtype=float)

    calibration_data = load_python_variables_module(calibration_path)
    pesos_posteriores = np.asarray(calibration_data.pesos_posteriores, dtype=float)

    n_paths = path_matrix.shape[0]
    pesos_prior = np.full(n_paths, 1.0 / n_paths, dtype=float)

    return path_matrix, times, pesos_posteriores, pesos_prior


def price_option_with_weights(
    CP: OptionType,
    strike: float,
    maturity: float,
    path_matrix: np.ndarray,
    times: np.ndarray,
    weights: np.ndarray,
) -> float:
    """
    Calcula el precio de una opción europea usando pesos dados.

    Precio = sum_i w_i * payoff_descontado_i
    """
    matches = np.where(np.isclose(times, maturity))[0]
    if matches.size == 0:
        raise ValueError(
            f"El vencimiento {maturity} no está en la malla temporal de las trayectorias."
        )
    terminal_prices = path_matrix[:, int(matches[0])]

    discounted_payoffs = pathwise_discounted_payoff(
        CP=CP,
        terminal_prices=terminal_prices,
        strike=strike,
        maturity=maturity,
    )

    return float(np.sum(weights * discounted_payoffs))


def price_options_grid(
    config: PricingConfig,
    path_matrix: np.ndarray,
    times: np.ndarray,
    pesos_posteriores: np.ndarray,
    pesos_prior: np.ndarray,
) -> list[PricingResult]:
    """
    Calcula precios para una malla de strikes y vencimientos.
    """
    strikes = config.pricing_strikes if config.pricing_strikes is not None else K
    maturities = config.pricing_maturities if config.pricing_maturities is not None else T

    benchmark_set = {
        (b.strike, b.maturity) for b in BENCHMARK_OPTIONS
    }

    results: list[PricingResult] = []
    for maturity in maturities:
        for strike in strikes:
            price_wmc = price_option_with_weights(
                CP=config.option_type,
                strike=strike,
                maturity=maturity,
                path_matrix=path_matrix,
                times=times,
                weights=pesos_posteriores,
            )
            price_mc = price_option_with_weights(
                CP=config.option_type,
                strike=strike,
                maturity=maturity,
                path_matrix=path_matrix,
                times=times,
                weights=pesos_prior,
            )
            is_benchmark = (strike, maturity) in benchmark_set

            results.append(
                PricingResult(
                    CP=config.option_type,
                    strike=strike,
                    maturity=maturity,
                    price_wmc=price_wmc,
                    price_mc=price_mc,
                    is_benchmark=is_benchmark,
                )
            )

    return results


def _format_pricing_table(
    results: list[PricingResult],
    benchmark_prices_market: dict[tuple[float, float], float] | None = None,
) -> str:
    """
    Formatea los resultados de pricing como una tabla legible.
    Incluye precio de mercado si está disponible.
    """
    lines: list[str] = []
    show_market = benchmark_prices_market is not None
    header = (
        f"{'Tipo':<6} {'Strike':>8} {'Vencimiento':>12} "
        f"{'Precio WMC':>14} {'Precio MC':>14} "
    )
    if show_market:
        header += f"{'Precio Mkt':>14} {'WMC-Mkt':>12} "
    header += f"{'WMC-MC':>12}"
    lines.append(header)
    lines.append("-" * len(header))

    current_maturity = None
    for res in results:
        if current_maturity is not None and res.maturity != current_maturity:
            lines.append("")
        current_maturity = res.maturity

        diff_mc = res.price_wmc - res.price_mc
        tipo = "CALL" if res.CP == OptionType.CALL else "PUT"

        line = (
            f"{tipo:<6} {res.strike:>8.1f} {res.maturity:>12.4f} "
            f"{res.price_wmc:>14.{DISPLAY_DECIMALS}f} {res.price_mc:>14.{DISPLAY_DECIMALS}f} "
        )
        if show_market:
            market_price = benchmark_prices_market.get(
                (res.strike, res.maturity), None
            )
            if market_price is not None:
                diff_mkt = res.price_wmc - market_price
                line += f"{market_price:>14.{DISPLAY_DECIMALS}f} {diff_mkt:>12.2e} "
            else:
                line += f"{'---':>14} {'---':>12} "
        line += f"{diff_mc:>12.{DISPLAY_DECIMALS}f}"
        lines.append(line)

    return "\n".join(lines)


# Strikes fuera de muestra: interpolados entre los benchmarks, no usados en calibración
OOS_STRIKES = np.array([90.0, 110.0], dtype=float)


def main() -> None:
    # Malla completa: strikes benchmark + strikes fuera de muestra, ordenados
    all_strikes = np.sort(np.unique(np.concatenate([K, OOS_STRIKES])))

    config = PricingConfig(
        pricing_strikes=all_strikes,
        pricing_maturities=T,
        option_type=OptionType.CALL,
    )

    base_dir = Path(__file__).resolve().parent
    prior_path = base_dir / config.prior_filename
    calibration_path = base_dir / config.calibration_filename

    path_matrix, times, pesos_posteriores, pesos_prior = load_simulation_and_weights(
        prior_path=prior_path,
        calibration_path=calibration_path,
    )

    n_paths = path_matrix.shape[0]
    print(f"Trayectorias cargadas: {n_paths}")
    print(f"Suma pesos calibrados: {np.sum(pesos_posteriores):.10f}")
    print(f"Suma pesos uniformes:  {np.sum(pesos_prior):.10f}")
    print(f"Strikes fuera de muestra (OOS): {OOS_STRIKES.tolist()}")
    print()

    results = price_options_grid(
        config=config,
        path_matrix=path_matrix,
        times=times,
        pesos_posteriores=pesos_posteriores,
        pesos_prior=pesos_prior,
    )

    benchmark_prices_market = {
        (b.strike, b.maturity): b.market_price for b in BENCHMARK_OPTIONS
    }

    print(_format_pricing_table(results, benchmark_prices_market))

    # --- Resumen benchmarks ---
    benchmark_results = [res for res in results if res.is_benchmark]
    if benchmark_results:
        max_error = 0.0
        for res in benchmark_results:
            market_price = benchmark_prices_market.get((res.strike, res.maturity), None)
            if market_price is not None:
                error = abs(res.price_wmc - market_price)
                max_error = max(max_error, error)
        print(f"\nMax |error| WMC vs Market (benchmarks)    = {max_error:.2e}")

    # --- Resumen fuera de muestra ---
    oos_results = [
        res for res in results if res.strike in OOS_STRIKES
    ]
    if oos_results:
        print("\n--- Fuera de muestra (OOS): WMC vs MC clásico ---")
        print(f"{'Strike':>8} {'Vencim.':>10} {'WMC':>14} {'MC':>14} {'WMC-MC':>12}")
        print("-" * 62)
        for res in oos_results:
            diff = res.price_wmc - res.price_mc
            print(
                f"{res.strike:>8.1f} {res.maturity:>10.4f} "
                f"{res.price_wmc:>14.{DISPLAY_DECIMALS}f} "
                f"{res.price_mc:>14.{DISPLAY_DECIMALS}f} "
                f"{diff:>12.{DISPLAY_DECIMALS}f}"
            )


if __name__ == "__main__":
    main()
