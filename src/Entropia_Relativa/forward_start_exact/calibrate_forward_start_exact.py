"""
Calibracion WMC exacta anadiendo restricciones forward-start.

Este script no modifica el flujo WMC existente. Construye una calibracion
alternativa en esta carpeta, incorporando payoffs forward-start a la matriz de
restricciones y forzando siempre el modo exact.

Por defecto usa como targets los precios forward-start SLV para la misma malla
T1/T2/strikes definida en wmc_forward_start.py.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(os.path.dirname(CURRENT_DIR))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from common.types import OptionType
from Entropia_Relativa import wmc_forward_start
from Entropia_Relativa.data_loader import S0, calibration_filename, prior_filename, r
from Entropia_Relativa.python_variable_io import export_python_variables_module
from Entropia_Relativa.relative_entropy_calibration import (
    CALIBRATION_MODE_EXACT,
    CalibrationConstraint,
    RelativeEntropyCalibrationConfig,
    build_constraint_set,
    calibrate_relative_entropy_weights,
    load_prior_simulation,
)
from Stochastic_Local_Volatility.Volatilidad_local_estocastica import (
    slv_forward_start_precios,
)


DISPLAY_DECIMALS = 6
CONSTRAINT_KIND_FORWARD_START = "forward_start"
DEFAULT_T1_VALUES =( tuple(float(value) for value in wmc_forward_start._as_float_list(wmc_forward_start.T1)))
DEFAULT_T2_VALUES = tuple(float(value) for value in wmc_forward_start._as_float_list(wmc_forward_start.T2))
DEFAULT_STRIKES = tuple(float(value) for value in wmc_forward_start.STRIKES)
DEFAULT_OUTPUT_FILE = "relative_entropy_forward_start_exact_results.py"


@dataclass(frozen=True)
class ForwardStartExactCalibrationConfig:
    S0: float = float(S0)
    T1_values: tuple[float, ...] = DEFAULT_T1_VALUES
    T2_values: tuple[float, ...] = DEFAULT_T2_VALUES
    strikes: tuple[float, ...] = DEFAULT_STRIKES
    option_type: OptionType = wmc_forward_start._option_type(wmc_forward_start.OPTION_TYPE)
    prior_file: str = prior_filename
    output_file: str = DEFAULT_OUTPUT_FILE
    target_source: str = "slv"
    target_prices: tuple[float, ...] = ()
    slv_n_paths: int = slv_forward_start_precios.N_PATHS
    slv_steps_per_year: int = slv_forward_start_precios.STEPS_PER_YEAR
    slv_n_bins: int = slv_forward_start_precios.N_BINS
    slv_scheme: str = slv_forward_start_precios.SCHEME
    slv_seed: int | None = slv_forward_start_precios.SEED
    include_forward_constraints: bool = True
    include_martingale_constraints: bool = True
    export_results: bool = True


def _option_type_label(option_type: OptionType) -> str:
    if option_type == OptionType.CALL:
        return "call"
    if option_type == OptionType.PUT:
        return "put"
    raise ValueError("OptionType debe ser CALL o PUT.")


def _valid_forward_pairs(T1_values: tuple[float, ...], T2_values: tuple[float, ...]) -> list[tuple[float, float]]:
    pairs: list[tuple[float, float]] = []
    seen: set[tuple[float, float]] = set()
    for T1_value in T1_values:
        for T2_value in T2_values:
            T1_float = float(T1_value)
            T2_float = float(T2_value)
            if T2_float <= T1_float:
                continue
            key = (round(T1_float, 12), round(T2_float, 12))
            if key in seen:
                continue
            seen.add(key)
            pairs.append((T1_float, T2_float))
    if not pairs:
        raise ValueError("Debe existir al menos una combinacion con T2 > T1.")
    return pairs


def _forward_start_grid(
    config: ForwardStartExactCalibrationConfig,
) -> list[tuple[float, float, float]]:
    strikes = np.asarray(config.strikes, dtype=float).reshape(-1)
    if strikes.size == 0:
        raise ValueError("Debes proporcionar al menos un strike.")
    return [
        (T1_value, T2_value, float(strike))
        for T1_value, T2_value in _valid_forward_pairs(config.T1_values, config.T2_values)
        for strike in strikes
    ]


def _module_path(filename: str) -> Path:
    path = Path(filename)
    if path.is_absolute():
        return path
    return Path(__file__).resolve().parents[1] / path


def _output_path(filename: str) -> Path:
    path = Path(filename)
    if path.is_absolute():
        return path
    return Path(__file__).resolve().parent / path


def _find_time_index(times: np.ndarray, target: float) -> int:
    matches = np.where(np.isclose(times, float(target), rtol=1e-10, atol=1e-10))[0]
    if matches.size == 0:
        raise ValueError(
            f"El tiempo {target} no esta en la malla del prior. "
            f"Disponibles: {np.asarray(times, dtype=float).tolist()}."
        )
    return int(matches[0])


def _forward_start_discounted_payoff(
    S_T1: np.ndarray,
    S_T2: np.ndarray,
    strike: float,
    S0_value: float,
    T2: float,
    option_type: OptionType,
) -> np.ndarray:
    payoff = wmc_forward_start.pathwise_forward_start_payoff(
        S_T1=S_T1,
        S_T2=S_T2,
        strike=float(strike),
        S0_value=float(S0_value),
        CP=option_type,
    )
    return np.exp(-r * float(T2)) * payoff


def build_forward_start_constraints(
    path_matrix: np.ndarray,
    times: np.ndarray,
    config: ForwardStartExactCalibrationConfig,
    target_prices: np.ndarray,
) -> list[CalibrationConstraint]:
    """
    Construye restricciones de precio forward-start con payoffs descontados.
    """
    grid = _forward_start_grid(config)
    targets = np.asarray(target_prices, dtype=float).reshape(-1)
    if targets.size != len(grid):
        raise ValueError(
            "target_prices debe tener un precio por punto (T1, T2, strike). "
            f"Recibido {targets.size}; esperado {len(grid)}."
        )

    spot_cache: dict[float, np.ndarray] = {}

    def spots_at(time_value: float) -> np.ndarray:
        key = round(float(time_value), 12)
        if key not in spot_cache:
            spot_cache[key] = np.asarray(path_matrix[:, _find_time_index(times, float(time_value))], dtype=float)
        return spot_cache[key]

    constraints: list[CalibrationConstraint] = []
    for (T1_value, T2_value, strike), target_price in zip(grid, targets):
        S_T1 = spots_at(T1_value)
        S_T2 = spots_at(T2_value)
        constraints.append(
            CalibrationConstraint(
                name=(
                    f"forward_start_{_option_type_label(config.option_type)}_"
                    f"T1_{T1_value:.6f}_T2_{T2_value:.6f}_K_{float(strike):.4f}"
                ),
                kind=CONSTRAINT_KIND_FORWARD_START,
                payoffs=_forward_start_discounted_payoff(
                    S_T1=S_T1,
                    S_T2=S_T2,
                    strike=float(strike),
                    S0_value=float(config.S0),
                    T2=T2_value,
                    option_type=config.option_type,
                ),
                target_value=float(target_price),
                least_squares_weight=0.0,
            )
        )
    return constraints


def slv_forward_start_targets(config: ForwardStartExactCalibrationConfig) -> pd.DataFrame:
    """
    Genera targets forward-start desde el flujo SLV existente.
    """
    targets = slv_forward_start_precios.run_slv_forward_start(
        S0_value=float(config.S0),
        T1_list=[float(value) for value in config.T1_values],
        T2_list=[float(value) for value in config.T2_values],
        strikes=list(config.strikes),
        option_type=config.option_type,
        n_paths=int(config.slv_n_paths),
        steps_per_year=int(config.slv_steps_per_year),
        n_bins=int(config.slv_n_bins),
        scheme=str(config.slv_scheme),
        seed=config.slv_seed if config.slv_seed is not None else 12345,
    )
    targets = targets.rename(columns={"slv_price": "target_price"})
    return targets[["option_type", "T1", "T2", "strike", "target_price"]].copy()


def manual_forward_start_targets(config: ForwardStartExactCalibrationConfig) -> pd.DataFrame:
    targets = np.asarray(config.target_prices, dtype=float).reshape(-1)
    grid = _forward_start_grid(config)
    if targets.size != len(grid):
        raise ValueError(
            "--target-prices debe tener un precio por punto (T1, T2, strike). "
            f"Recibido {targets.size}; esperado {len(grid)}."
        )
    return pd.DataFrame(
        {
            "option_type": _option_type_label(config.option_type),
            "T1": [T1_value for T1_value, _, _ in grid],
            "T2": [T2_value for _, T2_value, _ in grid],
            "strike": [strike for _, _, strike in grid],
            "target_price": targets,
        }
    )


def load_forward_start_targets(config: ForwardStartExactCalibrationConfig) -> pd.DataFrame:
    source = config.target_source.strip().lower()
    if source == "slv":
        return slv_forward_start_targets(config)
    if source == "manual":
        return manual_forward_start_targets(config)
    raise ValueError("target_source debe ser 'slv' o 'manual'.")


def _extended_constraint_arrays(
    base_matrix: np.ndarray,
    base_targets: np.ndarray,
    base_weights: np.ndarray,
    base_names: tuple[str, ...],
    base_kinds: tuple[str, ...],
    extra_constraints: list[CalibrationConstraint],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[str, ...], tuple[str, ...]]:
    extra_matrix = np.column_stack([constraint.payoffs for constraint in extra_constraints]).astype(float)
    matrix = np.column_stack((base_matrix, extra_matrix)).astype(float)
    targets = np.concatenate(
        (np.asarray(base_targets, dtype=float), np.array([c.target_value for c in extra_constraints], dtype=float))
    )
    least_squares_weights = np.concatenate(
        (
            np.asarray(base_weights, dtype=float),
            np.array([c.least_squares_weight for c in extra_constraints], dtype=float),
        )
    )
    names = tuple(base_names) + tuple(c.name for c in extra_constraints)
    kinds = tuple(base_kinds) + tuple(c.kind for c in extra_constraints)
    return matrix, targets, least_squares_weights, names, kinds


def calibrate_forward_start_exact(
    config: ForwardStartExactCalibrationConfig = ForwardStartExactCalibrationConfig(),
) -> tuple[pd.DataFrame, dict[str, object]]:
    """
    Calibra pesos WMC exactos incluyendo restricciones forward-start.
    """
    prior_path = _module_path(config.prior_file)
    prior_data = load_prior_simulation(prior_path)
    target_table = load_forward_start_targets(config)
    target_prices = target_table["target_price"].to_numpy(dtype=float)

    calibration_config = RelativeEntropyCalibrationConfig(
        prior_filename=str(prior_path),
        export_results=False,
        export_filename=calibration_filename,
        calibration_mode=CALIBRATION_MODE_EXACT,
        include_forward_constraints=bool(config.include_forward_constraints),
        include_martingale_constraints=bool(config.include_martingale_constraints),
    )
    base_constraints = build_constraint_set(prior_data=prior_data, config=calibration_config)
    forward_start_constraints = build_forward_start_constraints(
        path_matrix=prior_data.path_matrix,
        times=prior_data.times,
        config=config,
        target_prices=target_prices,
    )
    matrix, targets, least_squares_weights, names, kinds = _extended_constraint_arrays(
        base_matrix=base_constraints.matrix,
        base_targets=base_constraints.targets,
        base_weights=base_constraints.least_squares_weights,
        base_names=base_constraints.names,
        base_kinds=base_constraints.kinds,
        extra_constraints=forward_start_constraints,
    )

    calibration = calibrate_relative_entropy_weights(
        matriz_payoffs=matrix,
        pesos_prior=prior_data.prior_weights,
        precios_market=targets,
        config=calibration_config,
        least_squares_weights=least_squares_weights,
        constraint_kinds=kinds,
        constraint_names=names,
    )

    fs_matrix = np.column_stack([constraint.payoffs for constraint in forward_start_constraints]).astype(float)
    wmc_forward_prices = fs_matrix.T @ calibration.pesos_posteriores
    comparison = target_table.copy()
    comparison["wmc_exact_price"] = wmc_forward_prices
    comparison["abs_error"] = np.abs(comparison["wmc_exact_price"] - comparison["target_price"])

    errors = np.asarray(calibration.errores_precio, dtype=float)
    kinds_array = np.asarray(kinds, dtype=object)
    fs_mask = kinds_array == CONSTRAINT_KIND_FORWARD_START
    option_mask = kinds_array == "option"
    metadata: dict[str, object] = {
        "prior_path": prior_path,
        "output_path": _output_path(config.output_file),
        "calibration": calibration,
        "constraint_names": names,
        "constraint_kinds": kinds,
        "constraint_targets": targets,
        "constraint_errors": errors,
        "n_base_constraints": int(base_constraints.matrix.shape[1]),
        "n_forward_start_constraints": int(len(forward_start_constraints)),
        "max_abs_error_forward_start": float(np.max(np.abs(errors[fs_mask]))) if np.any(fs_mask) else 0.0,
        "max_abs_error_options": float(np.max(np.abs(errors[option_mask]))) if np.any(option_mask) else 0.0,
    }
    return comparison, metadata


def export_calibration_result(
    comparison: pd.DataFrame,
    metadata: dict[str, object],
    config: ForwardStartExactCalibrationConfig,
) -> Path:
    calibration = metadata["calibration"]
    output_path = Path(metadata["output_path"])
    variables = {
        "modo_calibracion": CALIBRATION_MODE_EXACT,
        "target_source": config.target_source,
        "option_type": _option_type_label(config.option_type),
        "S0": float(config.S0),
        "T1_values": np.asarray(config.T1_values, dtype=float),
        "T2_values": np.asarray(config.T2_values, dtype=float),
        "forward_start_pairs": comparison[["T1", "T2"]].drop_duplicates().to_numpy(dtype=float),
        "strikes": comparison["strike"].to_numpy(dtype=float),
        "target_forward_start_prices": comparison["target_price"].to_numpy(dtype=float),
        "wmc_forward_start_prices": comparison["wmc_exact_price"].to_numpy(dtype=float),
        "forward_start_errors": comparison["abs_error"].to_numpy(dtype=float),
        "pesos_posteriores": calibration.pesos_posteriores,
        "lambdas": calibration.lambdas,
        "constraint_names": np.array(metadata["constraint_names"], dtype=str),
        "constraint_kinds": np.array(metadata["constraint_kinds"], dtype=str),
        "constraint_targets": np.asarray(metadata["constraint_targets"], dtype=float),
        "constraint_errors": np.asarray(metadata["constraint_errors"], dtype=float),
        "n_constraints": int(calibration.n_constraints),
        "n_forward_start_constraints": int(metadata["n_forward_start_constraints"]),
        "max_abs_error_total": float(calibration.max_abs_error_total),
        "max_abs_error_options": float(metadata["max_abs_error_options"]),
        "max_abs_error_forward_start": float(metadata["max_abs_error_forward_start"]),
        "entropia_relativa": float(calibration.entropia_relativa),
        "iteraciones": int(calibration.iteraciones),
        "norma_gradiente": float(calibration.norma_gradiente),
        "aceptado": bool(calibration.aceptado),
        "optimizador_mensaje": str(calibration.optimizador_mensaje),
    }
    return export_python_variables_module(output_path, variables)


def _print_result(
    comparison: pd.DataFrame,
    metadata: dict[str, object],
    config: ForwardStartExactCalibrationConfig,
) -> None:
    calibration = metadata["calibration"]
    print("=== Calibracion WMC exacta con restricciones forward-start ===")
    print(f"Prior: {metadata['prior_path']}")
    print(f"Modo: {CALIBRATION_MODE_EXACT}")
    print(f"Target forward-start: {config.target_source}")
    print(f"Tipo opcion: {_option_type_label(config.option_type).upper()}")
    print(f"T1={tuple(float(value) for value in config.T1_values)}")
    print(f"T2={tuple(float(value) for value in config.T2_values)}")
    print(f"Strikes: {tuple(float(x) for x in comparison['strike'])}")
    print(
        f"Restricciones base={metadata['n_base_constraints']}, "
        f"forward-start={metadata['n_forward_start_constraints']}, "
        f"total={calibration.n_constraints}"
    )
    print(
        f"Estado: {'aceptado' if calibration.aceptado else 'no aceptado'} "
        f"({calibration.optimizador_mensaje})"
    )
    print(
        f"Max abs total={calibration.max_abs_error_total:.{DISPLAY_DECIMALS}f}, "
        f"vanilla={metadata['max_abs_error_options']:.{DISPLAY_DECIMALS}f}, "
        f"forward-start={metadata['max_abs_error_forward_start']:.{DISPLAY_DECIMALS}f}"
    )
    print()

    print("=== Forward-start calibrado ===")
    display_cols = ["option_type", "T1", "T2", "strike", "target_price", "wmc_exact_price", "abs_error"]
    print(comparison[display_cols].to_string(index=False, float_format=lambda value: f"{value:.{DISPLAY_DECIMALS}f}"))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Calibra WMC exacto anadiendo restricciones forward-start sin tocar los resultados existentes.",
    )
    parser.add_argument("--S0", type=float, default=ForwardStartExactCalibrationConfig.S0)
    parser.add_argument(
        "--T1",
        nargs="+",
        type=wmc_forward_start._maturity_value,
        default=list(ForwardStartExactCalibrationConfig.T1_values),
    )
    parser.add_argument(
        "--T2",
        nargs="+",
        type=wmc_forward_start._maturity_value,
        default=list(ForwardStartExactCalibrationConfig.T2_values),
    )
    parser.add_argument("--strikes", nargs="+", type=float, default=list(ForwardStartExactCalibrationConfig.strikes))
    parser.add_argument(
        "--option-type",
        type=wmc_forward_start._option_type,
        default=ForwardStartExactCalibrationConfig.option_type,
        help="Tipo de opcion: call o put.",
    )
    parser.add_argument("--prior-file", default=ForwardStartExactCalibrationConfig.prior_file)
    parser.add_argument("--output-file", default=ForwardStartExactCalibrationConfig.output_file)
    parser.add_argument("--target-source", choices=["slv", "manual"], default=ForwardStartExactCalibrationConfig.target_source)
    parser.add_argument(
        "--target-prices",
        nargs="*",
        type=float,
        default=list(ForwardStartExactCalibrationConfig.target_prices),
        help="Precios forward-start manuales, uno por punto (T1, T2, strike). Requiere --target-source manual.",
    )
    parser.add_argument("--slv-n-paths", type=int, default=ForwardStartExactCalibrationConfig.slv_n_paths)
    parser.add_argument("--slv-steps-per-year", type=int, default=ForwardStartExactCalibrationConfig.slv_steps_per_year)
    parser.add_argument("--slv-n-bins", type=int, default=ForwardStartExactCalibrationConfig.slv_n_bins)
    parser.add_argument("--slv-scheme", choices=["Euler", "AES"], default=ForwardStartExactCalibrationConfig.slv_scheme)
    parser.add_argument("--slv-seed", type=int, default=ForwardStartExactCalibrationConfig.slv_seed)
    parser.add_argument("--no-forward-constraints", action="store_false", dest="include_forward_constraints", default=True)
    parser.add_argument("--no-martingale-constraints", action="store_false", dest="include_martingale_constraints", default=True)
    parser.add_argument("--no-export", action="store_false", dest="export_results", default=True)
    return parser


def main() -> None:
    args, _ = _build_parser().parse_known_args()
    config = ForwardStartExactCalibrationConfig(
        S0=float(args.S0),
        T1_values=tuple(float(x) for x in args.T1),
        T2_values=tuple(float(x) for x in args.T2),
        strikes=tuple(float(x) for x in args.strikes),
        option_type=args.option_type,
        prior_file=str(args.prior_file),
        output_file=str(args.output_file),
        target_source=str(args.target_source),
        target_prices=tuple(float(x) for x in args.target_prices),
        slv_n_paths=int(args.slv_n_paths),
        slv_steps_per_year=int(args.slv_steps_per_year),
        slv_n_bins=int(args.slv_n_bins),
        slv_scheme=str(args.slv_scheme),
        slv_seed=args.slv_seed,
        include_forward_constraints=bool(args.include_forward_constraints),
        include_martingale_constraints=bool(args.include_martingale_constraints),
        export_results=bool(args.export_results),
    )

    comparison, metadata = calibrate_forward_start_exact(config)
    _print_result(comparison=comparison, metadata=metadata, config=config)
    if config.export_results:
        export_path = export_calibration_result(comparison=comparison, metadata=metadata, config=config)
        print()
        print(f"Resultado alternativo guardado en: {export_path}")
        print("Puedes usarlo despues con --calibration-file apuntando a este archivo.")


if __name__ == "__main__":
    main()
