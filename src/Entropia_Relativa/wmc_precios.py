#%%
# Weighted Monte Carlo: precios con los pesos posteriores ya calibrados.

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from common.types import OptionType
from Entropia_Relativa.data_loader import (
    MATURITY_LABELS,
    T,
    calibration_filename,
    prior_filename,
)
from Entropia_Relativa.gbm_prior_paths import pathwise_discounted_payoff
from Entropia_Relativa.python_variable_io import load_python_variables_module


# Cambia estos valores si ejecutas el archivo por celdas.
# STRIKES = (70.0, 90.0, 110.0)
STRIKES = (40, 60, 80, 100, 120, 140, 160)
MATURITIES = (1.0 / 52.0, 1.0 / 12.0, 0.5, 1.0, 2.0, 5.0, 8.0, 10.0)
OPTION_TYPE = "call"  # "call" o "put"
DEFAULT_PRIOR_FILE = prior_filename
DEFAULT_CALIBRATION_FILE = calibration_filename


def _option_type(value: str) -> OptionType:
    normalized = value.strip().lower()
    if normalized == "call":
        return OptionType.CALL
    if normalized == "put":
        return OptionType.PUT
    raise argparse.ArgumentTypeError("option-type debe ser 'call' o 'put'.")


def _maturity_value(value: str) -> float:
    normalized = value.strip().lower()
    label_to_maturity = {
        label.lower(): float(maturity)
        for label, maturity in zip(MATURITY_LABELS, T)
    }
    if normalized in label_to_maturity:
        return label_to_maturity[normalized]

    if "/" in normalized:
        numerator, denominator = normalized.split("/", maxsplit=1)
        try:
            return float(numerator) / float(denominator)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"Vencimiento no valido: {value!r}.") from exc

    try:
        return float(normalized)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Vencimiento no valido: {value!r}.") from exc


def _module_path(filename: str) -> Path:
    path = Path(filename)
    if path.is_absolute():
        return path
    return Path(__file__).resolve().parent / path


def _load_paths_and_weights(
    prior_file: str,
    calibration_file: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    prior_data = load_python_variables_module(_module_path(prior_file))
    calibration_data = load_python_variables_module(_module_path(calibration_file))

    path_matrix = np.asarray(prior_data.path_matrix, dtype=float)
    times = np.asarray(prior_data.times, dtype=float)
    if not hasattr(calibration_data, "pesos_posteriores"):
        raise AttributeError(
            "El archivo de calibracion no contiene la variable "
            "'pesos_posteriores'. Recalibra o revisa el fichero usado."
        )
    weights = np.asarray(calibration_data.pesos_posteriores, dtype=float).reshape(-1)

    if path_matrix.shape[0] != weights.size:
        raise ValueError(
            "El numero de trayectorias no coincide con el numero de pesos: "
            f"{path_matrix.shape[0]} trayectorias y {weights.size} pesos."
        )

    return path_matrix, times, weights


def wmc_price(
    option_type: OptionType,
    strike: float,
    maturity: float,
    path_matrix: np.ndarray,
    times: np.ndarray,
    weights: np.ndarray,
) -> float:
    matches = np.where(np.isclose(times, maturity))[0]
    if matches.size == 0:
        raise ValueError(
            f"El vencimiento {maturity} no esta en la malla de trayectorias. "
            f"Disponibles: {times.tolist()}."
        )

    terminal_prices = path_matrix[:, int(matches[0])]
    discounted_payoffs = pathwise_discounted_payoff(
        CP=option_type,
        terminal_prices=terminal_prices,
        strike=float(strike),
        maturity=float(maturity),
    )
    return float(np.dot(weights, discounted_payoffs))


def _build_parser() -> argparse.ArgumentParser:
    available = ", ".join(
        f"{label}={float(maturity):.2f}"
        for label, maturity in zip(MATURITY_LABELS, T)
    )
    parser = argparse.ArgumentParser(
        description="Valora opciones vanilla por WMC usando pesos posteriores.",
        epilog=f"Vencimientos disponibles: {available}.",
    )
    parser.add_argument(
        "--option-type",
        type=_option_type,
        default=_option_type(OPTION_TYPE),
    )
    parser.add_argument("--strikes", nargs="+", type=float, default=list(STRIKES))
    parser.add_argument(
        "--maturities",
        nargs="+",
        type=_maturity_value,
        default=list(MATURITIES),
    )
    parser.add_argument("--prior-file", default=DEFAULT_PRIOR_FILE)
    parser.add_argument(
        "--calibration-file",
        default=DEFAULT_CALIBRATION_FILE,
        help="Archivo .py exportado por la calibracion con pesos_posteriores.",
    )
    parser.add_argument(
        "--weights-file",
        dest="calibration_file",
        help=argparse.SUPPRESS,
    )
    return parser


def main() -> None:
    args, _ = _build_parser().parse_known_args()
    path_matrix, times, weights = _load_paths_and_weights(
        prior_file=args.prior_file,
        calibration_file=args.calibration_file,
    )

    print(f"{'Tipo':<6} {'Strike':>8} {'Vencimiento':>12} {'Precio WMC':>14}")
    print("-" * 45)
    for maturity in args.maturities:
        for strike in args.strikes:
            price = wmc_price(
                option_type=args.option_type,
                strike=float(strike),
                maturity=float(maturity),
                path_matrix=path_matrix,
                times=times,
                weights=weights,
            )
            tipo = "CALL" if args.option_type == OptionType.CALL else "PUT"
            print(
                f"{tipo:<6} {float(strike):>8.2f} "
                f"{float(maturity):>12.2f} {price:>14.2f}"
            )


if __name__ == "__main__":
    main()
