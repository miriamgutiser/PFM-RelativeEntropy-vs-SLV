"""
Carga de los datos benchmark de opciones para Entropia Relativa.

Este archivo actua como punto de entrada editable del modulo:
1. ajustas aqui strikes, vencimientos y precios benchmark,
2. lo ejecutas,
3. se lanza toda la simulacion, calibracion y pricing final.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


from common.types import OptionType


DISPLAY_DECIMALS = 7


# Parametros globales editables del metodo
n_paths = 20000
sigma_prior = 0.20
#True: prior con curva de volatilidad por vencimiento, False: Prior GBM con volatilidad constante
use_atmf_term_structure_prior = True
seed = 12345
use_antithetic = True
include_spot_column = True
pricing_option_type = OptionType.CALL

calibration_mode = "weighted_least_squares"  # weighted_least_squares o exact
include_forward_constraints = True
include_martingale_constraints = True
wls_weight_options = 1e-7
wls_weight_forwards = 1e-7
wls_weight_martingale = 1e-7
filter_degenerate_option_constraints = True
min_option_positive_payoff_count = 5
min_option_variance = 1e-12
prior_filename = "gbm_prior_simulation_results.py"
calibration_filename = "relative_entropy_calibration_results.py"
weights_txt_filename = "relative_entropy_weights.txt"


@dataclass(frozen=True)
class BenchmarkOption:
    """
    Datos de una opcion benchmark.
    """

    CP: OptionType
    strike: float
    maturity: float
    market_price: float
    maturity_label: str


@dataclass(frozen=True)
class FullPipelineConfig:
    """
    Configuracion global del pipeline completo.
    """

    n_paths: int = n_paths
    sigma: float = sigma_prior
    use_atmf_term_structure_prior: bool = use_atmf_term_structure_prior
    seed: int = seed
    use_antithetic: bool = use_antithetic
    include_spot_column: bool = include_spot_column
    pricing_option_type: OptionType = pricing_option_type
    calibration_mode: str = calibration_mode
    include_forward_constraints: bool = include_forward_constraints
    include_martingale_constraints: bool = include_martingale_constraints
    wls_weight_options: float = wls_weight_options
    wls_weight_forwards: float = wls_weight_forwards
    wls_weight_martingale: float = wls_weight_martingale
    filter_degenerate_option_constraints: bool = filter_degenerate_option_constraints
    min_option_positive_payoff_count: int = min_option_positive_payoff_count
    min_option_variance: float = min_option_variance
    prior_filename: str = prior_filename
    calibration_filename: str = calibration_filename
    weights_txt_filename: str = weights_txt_filename


S0 = 100.0
r = 0.0
q = 0.0
CP = OptionType.CALL

# MATURITY_LABELS = ["1w", "1m", "6m", "1y", "2y", "5y", "8y", "10y"]
# T = np.array([1.0 / 52.0, 1.0 / 12.0, 0.5, 1.0, 2.0, 5.0, 8.0, 10.0], dtype=float)
# K = np.array([40.0, 60.0, 80.0, 100.0, 120.0, 140.0], dtype=float)
MATURITY_LABELS = ["1w", "1m", "6m", "1y", "2y", "5y"]
T = np.array([1.0 / 52.0, 1.0 / 12.0, 0.5, 1.0, 2.0, 5.0], dtype=float)
K = np.array([100.0, 120.0, 140.0], dtype=float)

# Filas: vencimientos en el orden de T.
# Columnas: strikes en el orden de K.
TARGET_OPTION_PRICES = np.array(
    [
        [0.3688735, 0.0000000, 0.0000000],
        [0.7372565, 0.0000000, 0.0000000],
        [1.5783534, 0.0003602, 0.0000004],
        [2.1976425, 0.0092500, 0.0002104],
        [3.2306143, 0.0607294, 0.0052822],
        [5.7249901, 0.3915454, 0.0590638],

    ],
    dtype=float,
)


def _validate_benchmark_data() -> None:
    if len(MATURITY_LABELS) != len(T):
        raise ValueError(
            "MATURITY_LABELS y T deben tener la misma longitud. "
            f"Recibido: {len(MATURITY_LABELS)} etiquetas y {len(T)} vencimientos."
        )

    expected_shape = (len(T), len(K))
    if TARGET_OPTION_PRICES.shape != expected_shape:
        raise ValueError(
            "TARGET_OPTION_PRICES debe tener shape "
            f"{expected_shape}, pero tiene {TARGET_OPTION_PRICES.shape}."
        )

    if len(np.unique(T)) != len(T):
        raise ValueError("Los vencimientos T no deben repetirse.")

    if len(np.unique(K)) != len(K):
        raise ValueError("Los strikes K no deben repetirse.")


def build_benchmark_options() -> list[BenchmarkOption]:
    """
    Devuelve la lista plana de opciones benchmark.
    """
    benchmarks: list[BenchmarkOption] = []
    for maturity_label, maturity, row_prices in zip(
        MATURITY_LABELS, T, TARGET_OPTION_PRICES
    ):
        for strike, market_price in zip(K, row_prices):
            benchmarks.append(
                BenchmarkOption(
                    CP=CP,
                    strike=float(strike),
                    maturity=float(maturity),
                    market_price=float(market_price),
                    maturity_label=maturity_label,
                )
            )
    return benchmarks


def format_benchmark_summary() -> str:
    lines = [
        "=== Benchmarks cargados ===",
        f"Tipo opcion: {'CALL' if CP == OptionType.CALL else 'PUT'}",
        f"Spot S0: {S0:.4f}",
        f"Numero de vencimientos: {len(T)}",
        f"Numero de strikes: {len(K)}",
        f"Total opciones benchmark: {len(BENCHMARK_OPTIONS)}",
        f"Vencimientos: {MATURITY_LABELS}",
        f"Strikes: {K.tolist()}",
        f"Numero de trayectorias: {n_paths}",
        f"Sigma prior fallback: {sigma_prior:.4f}",
        f"Semilla: {seed}",
        f"Prior ATMF por vencimiento: {use_atmf_term_structure_prior}",
        "Calibracion paper:",
        f"  modo={calibration_mode}",
        f"  forwards={include_forward_constraints}",
        f"  martingala={include_martingale_constraints}",
        f"  filtrar OTM degeneradas={filter_degenerate_option_constraints}",
    ]
    return "\n".join(lines)


_validate_benchmark_data()
BENCHMARK_OPTIONS = build_benchmark_options()


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Ejecuta simulacion prior + calibracion + pricing en una sola corrida."
    )
    parser.add_argument("--n-paths", type=int, default=n_paths, help="Numero de trayectorias.")
    parser.add_argument("--sigma", type=float, default=sigma_prior, help="Volatilidad prior GBM.")
    parser.add_argument("--seed", type=int, default=seed, help="Semilla aleatoria.")
    parser.add_argument(
        "--mode",
        type=str,
        default=calibration_mode,
        help="Modo de calibracion: exact o weighted_least_squares.",
    )
    return parser


def _build_pipeline_config_from_args(args: argparse.Namespace) -> FullPipelineConfig:
    return FullPipelineConfig(
        n_paths=args.n_paths,
        sigma=args.sigma,
        seed=args.seed,
        calibration_mode=args.mode,
    )


def _export_prior_results(
    pipeline_config: FullPipelineConfig,
    base_dir: Path,
) -> Path:
    from Entropia_Relativa.gbm_prior_paths import (
        PriorGBMConfig,
        simulate_prior_and_benchmarks,
    )
    from Entropia_Relativa.python_variable_io import export_python_variables_module

    prior_config = PriorGBMConfig(
        n_paths=pipeline_config.n_paths,
        sigma=pipeline_config.sigma,
        use_atmf_term_structure=pipeline_config.use_atmf_term_structure_prior,
        seed=pipeline_config.seed,
        use_antithetic=pipeline_config.use_antithetic,
        include_spot_column=pipeline_config.include_spot_column,
        export_results=True,
        export_filename=pipeline_config.prior_filename,
    )
    prior_result = simulate_prior_and_benchmarks(config=prior_config)
    export_path = base_dir / pipeline_config.prior_filename
    export_python_variables_module(
        module_path=export_path,
        variables={
            "times": prior_result.times,
            "path_matrix": prior_result.path_matrix,
            "discounted_payoff_matrix": prior_result.discounted_payoff_matrix,
            "prior_weights": prior_result.prior_weights,
            "benchmark_prices_prior": prior_result.benchmark_prices_prior,
        },
    )

    benchmark_prices_market = np.array(
        [benchmark.market_price for benchmark in BENCHMARK_OPTIONS],
        dtype=float,
    )
    mae = float(
        np.mean(np.abs(prior_result.benchmark_prices_prior - benchmark_prices_market))
    )

    print("=== Simulacion prior ===")
    print(
        f"Trayectorias = {pipeline_config.n_paths}, "
        f"sigma = {pipeline_config.sigma:.4f}, seed = {pipeline_config.seed}"
    )
    print(f"Prior guardado en {export_path.name}")
    print(f"MAE prior en benchmarks = {mae:.{DISPLAY_DECIMALS}f}")
    print()

    return export_path


def _export_calibration_results(
    pipeline_config: FullPipelineConfig,
    base_dir: Path,
    prior_path: Path,
) -> Path:
    from Entropia_Relativa.python_variable_io import export_python_variables_module
    from Entropia_Relativa.relative_entropy_calibration import (
        RelativeEntropyCalibrationConfig,
        build_constraint_set,
        calibrate_relative_entropy_weights,
        load_prior_simulation,
    )

    calibration_config = RelativeEntropyCalibrationConfig(
        prior_filename=prior_path.name,
        export_results=True,
        export_filename=pipeline_config.calibration_filename,
        calibration_mode=pipeline_config.calibration_mode,
        include_forward_constraints=pipeline_config.include_forward_constraints,
        include_martingale_constraints=pipeline_config.include_martingale_constraints,
        wls_weight_options=pipeline_config.wls_weight_options,
        wls_weight_forwards=pipeline_config.wls_weight_forwards,
        wls_weight_martingale=pipeline_config.wls_weight_martingale,
        filter_degenerate_option_constraints=pipeline_config.filter_degenerate_option_constraints,
        min_option_positive_payoff_count=pipeline_config.min_option_positive_payoff_count,
        min_option_variance=pipeline_config.min_option_variance,
    )
    prior_data = load_prior_simulation(prior_path)
    constraint_set = build_constraint_set(
        prior_data=prior_data,
        config=calibration_config,
    )
    calibration_result = calibrate_relative_entropy_weights(
        matriz_payoffs=constraint_set.matrix,
        pesos_prior=prior_data.prior_weights,
        precios_market=constraint_set.targets,
        config=calibration_config,
        least_squares_weights=constraint_set.least_squares_weights,
        constraint_kinds=constraint_set.kinds,
        constraint_names=constraint_set.names,
    )

    export_path = base_dir / pipeline_config.calibration_filename
    export_python_variables_module(
        module_path=export_path,
        variables={
            "lambdas": calibration_result.lambdas,
            "pesos_posteriores": calibration_result.pesos_posteriores,
            "precios_modelo": calibration_result.precios_modelo,
            "precios_replicados_gtq": calibration_result.precios_replicados_gtq,
            "precios_market": calibration_result.constraint_targets,
            "errores_precio": calibration_result.errores_precio,
            "error_consistencia_gtq": calibration_result.error_consistencia_gtq,
            "entropia_relativa": calibration_result.entropia_relativa,
            "valor_objetivo": calibration_result.valor_objetivo,
            "iteraciones": calibration_result.iteraciones,
            "norma_gradiente": calibration_result.norma_gradiente,
            "modo_calibracion": calibration_result.calibration_mode,
            "n_constraints": calibration_result.n_constraints,
            "n_option_constraints": calibration_result.n_option_constraints,
            "n_forward_constraints": calibration_result.n_forward_constraints,
            "n_martingale_constraints": calibration_result.n_martingale_constraints,
            "max_abs_error_total": calibration_result.max_abs_error_total,
            "max_abs_error_options": calibration_result.max_abs_error_options,
            "max_abs_error_forwards": calibration_result.max_abs_error_forwards,
            "max_abs_error_martingale": calibration_result.max_abs_error_martingale,
        },
    )
    weights_txt_path = _export_weights_txt(
        pipeline_config=pipeline_config,
        prior_data=prior_data,
        pesos_posteriores=calibration_result.pesos_posteriores,
        base_dir=base_dir,
    )

    print("=== Calibracion ===")
    print(
        f"Modo = {calibration_result.calibration_mode}, "
        f"restricciones = {calibration_result.n_constraints}"
    )
    print(
        f"Opciones = {calibration_result.n_option_constraints}, "
        f"forwards = {calibration_result.n_forward_constraints}, "
        f"martingala = {calibration_result.n_martingale_constraints}"
    )
    print(
        f"Max abs total = {calibration_result.max_abs_error_total:.2e}, "
        f"grad = {calibration_result.norma_gradiente:.2e}, "
        f"iteraciones = {calibration_result.iteraciones}"
    )
    print(f"Calibracion guardada en {export_path.name}")
    print(f"Pesos guardados en {weights_txt_path}")
    print()

    return export_path


def _export_weights_txt(
    pipeline_config: FullPipelineConfig,
    prior_data,
    pesos_posteriores: np.ndarray,
    base_dir: Path,
) -> Path:
    results_dir = base_dir.parents[1] / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    txt_path = results_dir / pipeline_config.weights_txt_filename

    pesos_prior = np.asarray(prior_data.prior_weights, dtype=float)
    pesos_post = np.asarray(pesos_posteriores, dtype=float)
    ratio = np.divide(
        pesos_post,
        pesos_prior,
        out=np.full_like(pesos_post, np.nan),
        where=pesos_prior > 0.0,
    )
    log_ratio = np.where(
        (pesos_post > 0.0) & (pesos_prior > 0.0),
        np.log(pesos_post / pesos_prior),
        np.nan,
    )

    lines = [
        "Pesos de la calibracion por entropia relativa",
        "",
        (
            f"{'Path':>8} {'Peso prior':>18} {'Peso posterior':>18} "
            f"{'Posterior/Prior':>18} {'log(Posterior/Prior)':>24}"
        ),
        "-" * 92,
    ]

    for path_id, (peso_prior, peso_post, weight_ratio, log_weight_ratio) in enumerate(
        zip(pesos_prior, pesos_post, ratio, log_ratio),
        start=1,
    ):
        ratio_str = f"{weight_ratio:>18.10e}" if np.isfinite(weight_ratio) else f"{'':>18}"
        log_ratio_str = (
            f"{log_weight_ratio:>24.10e}" if np.isfinite(log_weight_ratio) else f"{'':>24}"
        )
        lines.append(
            f"{path_id:>8d} {peso_prior:>18.10e} {peso_post:>18.10e} "
            f"{ratio_str} {log_ratio_str}"
        )

    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return txt_path


def _run_pricing(
    pipeline_config: FullPipelineConfig,
    prior_path: Path,
    calibration_path: Path,
) -> None:
    from Entropia_Relativa.option_pricing import (
        PricingConfig,
        _format_calibration_summary,
        _format_pricing_table,
        load_simulation_and_weights,
        price_options_grid,
    )

    pricing_config = PricingConfig(
        prior_filename=prior_path.name,
        calibration_filename=calibration_path.name,
        pricing_strikes=K,
        pricing_maturities=T,
        option_type=pipeline_config.pricing_option_type,
    )

    (
        path_matrix,
        times,
        pesos_posteriores,
        pesos_prior,
        calibration_metadata,
    ) = load_simulation_and_weights(
        prior_path=prior_path,
        calibration_path=calibration_path,
    )

    results = price_options_grid(
        config=pricing_config,
        path_matrix=path_matrix,
        times=times,
        pesos_posteriores=pesos_posteriores,
        pesos_prior=pesos_prior,
    )
    benchmark_prices_market = {
        (benchmark.strike, benchmark.maturity): benchmark.market_price
        for benchmark in BENCHMARK_OPTIONS
    }
    benchmark_results = [result for result in results if result.is_benchmark]
    max_error = 0.0
    for result in benchmark_results:
        market_price = benchmark_prices_market[(result.strike, result.maturity)]
        max_error = max(max_error, abs(result.price_wmc - market_price))

    print("=== Pricing ===")
    print(f"Trayectorias cargadas: {path_matrix.shape[0]}")
    print(f"Suma pesos calibrados: {np.sum(pesos_posteriores):.10f}")
    print(f"Suma pesos previos:    {np.sum(pesos_prior):.10f}")
    print(_format_calibration_summary(calibration_metadata))
    print()
    print(_format_pricing_table(results, benchmark_prices_market))
    print()
    print(f"Max |error| WMC vs Market (benchmarks) = {max_error:.2e}")


def run_full_pipeline(
    pipeline_config: FullPipelineConfig | None = None,
    base_dir: Path | None = None,
) -> tuple[Path, Path]:
    if pipeline_config is None:
        pipeline_config = FullPipelineConfig()
    if base_dir is None:
        base_dir = Path(__file__).resolve().parent

    prior_path = _export_prior_results(
        pipeline_config=pipeline_config,
        base_dir=base_dir,
    )
    calibration_path = _export_calibration_results(
        pipeline_config=pipeline_config,
        base_dir=base_dir,
        prior_path=prior_path,
    )
    _run_pricing(
        pipeline_config=pipeline_config,
        prior_path=prior_path,
        calibration_path=calibration_path,
    )
    return prior_path, calibration_path


def run_from_cli(argv: list[str] | None = None) -> tuple[Path, Path]:
    args = _build_arg_parser().parse_args(argv)
    pipeline_config = _build_pipeline_config_from_args(args)
    return run_full_pipeline(pipeline_config=pipeline_config)


def main() -> None:

    print(format_benchmark_summary())
    print()
    run_from_cli()


if __name__ == "__main__":
    main()
