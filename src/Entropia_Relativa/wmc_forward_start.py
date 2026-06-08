#%%
# Weighted Monte Carlo: precios de opciones forward-start con pesos posteriores.
#
# El usuario introduce datos de contrato en unidades monetarias:
#   - S0: spot actual del activo
#   - K: precio de ejercicio actual/equivalente del contrato
#
# El forward-start fija en T1 un strike proporcional al spot de T1:
#   K_T1 = (K / S0) * S(T1)
#
# Por tanto, el payoff monetario real es:
#   Call = max(S(T2) - (K/S0) * S(T1), 0)
#   Put  = max((K/S0) * S(T1) - S(T2), 0)
#
# Si T1~0, entonces S(T1)~S0 y el payoff converge a una vanilla con strike K.

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from common.types import OptionType
from Entropia_Relativa import data_loader
from Entropia_Relativa.data_loader import (
    MATURITY_LABELS,
    T,
    calibration_filename,
    prior_filename,
    r,
)
from Entropia_Relativa.python_variable_io import load_python_variables_module


DISPLAY_DECIMALS = 7
DEFAULT_PRIOR_FILE = prior_filename
DEFAULT_CALIBRATION_FILE = calibration_filename

# Valores editables si ejecutas el archivo por celdas.
S0 = 100
T1 = 0.0
T2 = 10
STRIKES = (70.0, 100.0, 120.0)
OPTION_TYPE = "call"


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


def _as_float_list(value) -> list[float]:
    if isinstance(value, (str, bytes)):
        return [float(value)]
    try:
        return [float(item) for item in value]
    except TypeError:
        return [float(value)]


def _module_path(filename: str) -> Path:
    path = Path(filename)
    if path.is_absolute():
        return path
    return Path(__file__).resolve().parent / path


def _validate_spot_strikes(strikes: list[float] | tuple[float, ...] | np.ndarray) -> np.ndarray:
    strike_arr = np.asarray(strikes, dtype=float).reshape(-1)
    if strike_arr.size == 0:
        raise ValueError("Debes proporcionar al menos un strike.")
    if not np.all(np.isfinite(strike_arr)):
        raise ValueError("Todos los strikes deben ser finitos.")
    if np.any(strike_arr <= 0.0):
        raise ValueError("Los strikes monetarios deben ser positivos.")
    return strike_arr


def _load_paths_and_weights(
    prior_file: str,
    calibration_file: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """
    Carga la matriz de trayectorias, los tiempos y los pesos calibrados.
    """
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

    calibration_mode = str(getattr(calibration_data, "modo_calibracion", "desconocido"))
    return path_matrix, times, weights, calibration_mode


def _resolve_pricing_s0(path_matrix: np.ndarray, times: np.ndarray, requested_s0: float) -> float:
    """
    Valida que S0 sea consistente con las trayectorias cargadas.
    """
    pricing_s0 = float(requested_s0)
    if pricing_s0 <= 0.0:
        raise ValueError("S0 debe ser positivo.")

    zero_matches = np.where(np.isclose(times, 0.0, rtol=0.0, atol=1e-12))[0]
    if zero_matches.size == 0:
        return pricing_s0

    path_s0 = float(np.mean(path_matrix[:, int(zero_matches[0])]))
    if not np.isclose(pricing_s0, path_s0, rtol=1e-6, atol=1e-8):
        raise ValueError(
            f"S0={pricing_s0} no coincide con el spot inicial de las "
            f"trayectorias cargadas ({path_s0:.10f}). Regenera las trayectorias "
            "con ese S0 antes de usarlo para pricing."
        )
    return pricing_s0


def _find_time_index(times: np.ndarray, target: float) -> int:
    """
    Devuelve el indice en ``times`` correspondiente a ``target``.
    """
    matches = np.where(np.isclose(times, target, rtol=1e-10, atol=1e-10))[0]
    if matches.size == 0:
        raise ValueError(
            f"El tiempo {target} no esta en la malla de trayectorias. "
            f"Disponibles: {times.tolist()}. Este script solo usa pesos ya "
            "calibrados; para una fecha nueva hay que regenerar antes el prior "
            "y los pesos con esa fecha en la malla."
        )
    return int(matches[0])


def pathwise_forward_start_payoff(
    S_T1: np.ndarray,
    S_T2: np.ndarray,
    strike: float,
    S0_value: float,
    CP: OptionType,
) -> np.ndarray:
    """
    Calcula el payoff forward-start monetario para cada trayectoria.

    El input ``strike`` es el precio de ejercicio monetario K. El contrato
    forward-start fija en T1 el strike proporcional (K/S0)*S(T1).
    """
    if S0_value <= 0.0:
        raise ValueError("S0 debe ser positivo.")
    strike_T1 = (float(strike) / float(S0_value)) * S_T1

    if CP == OptionType.CALL:
        return np.maximum(S_T2 - strike_T1, 0.0)
    if CP == OptionType.PUT:
        return np.maximum(strike_T1 - S_T2, 0.0)
    raise ValueError("OptionType debe ser CALL o PUT.")


def wmc_forward_start_price(
    option_type: OptionType,
    S0_value: float,
    T1: float,
    T2: float,
    strike: float,
    path_matrix: np.ndarray,
    times: np.ndarray,
    weights: np.ndarray,
) -> float:
    """
    Precio de una opcion forward-start bajo WMC.

        Precio = e^{-r T2} * sum_i q_i * Payoff_i
    """
    idx_1 = _find_time_index(times, T1)
    idx_2 = _find_time_index(times, T2)

    S_T1 = path_matrix[:, idx_1]
    S_T2 = path_matrix[:, idx_2]

    payoffs = pathwise_forward_start_payoff(
        S_T1=S_T1,
        S_T2=S_T2,
        strike=strike,
        S0_value=S0_value,
        CP=option_type,
    )

    discount_factor = float(np.exp(-r * T2))
    return float(discount_factor * np.dot(weights, payoffs))


def _build_parser() -> argparse.ArgumentParser:
    available = ", ".join(
        f"{label}={float(maturity):.6g}"
        for label, maturity in zip(MATURITY_LABELS, T)
    )
    parser = argparse.ArgumentParser(
        description=(
            "Valora opciones forward-start por WMC usando pesos ya calibrados.\n"
            "K es el precio de ejercicio monetario introducido en el contrato.\n"
            "El contrato forward-start usa K_T1=(K/S0)*S(T1).\n"
            "Call=max(S(T2)-K_T1,0); Put=max(K_T1-S(T2),0).\n"
            "Con T1 cercano a 0, el precio converge a una vanilla con strike K."
        ),
        epilog=f"Vencimientos benchmark disponibles: {available}.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--S0",
        type=float,
        default=S0,
        help="Spot inicial solo para validar las trayectorias cargadas.",
    )
    parser.add_argument(
        "--option-type",
        type=_option_type,
        default=_option_type(OPTION_TYPE),
        help="Tipo de opcion: 'call' o 'put'.",
    )
    parser.add_argument(
        "--T1",
        nargs="+",
        type=_maturity_value,
        default=_as_float_list(T1),
        help="Fecha(s) de inicio del periodo forward (T1).",
    )
    parser.add_argument(
        "--T2",
        nargs="+",
        type=_maturity_value,
        default=_as_float_list(T2),
        help="Fecha(s) de vencimiento de la opcion (T2 > T1).",
    )
    parser.add_argument(
        "--strikes",
        nargs="+",
        type=float,
        default=list(STRIKES),
        help="Strike(s) monetarios/precios de ejercicio. ATM forward: K=S0.",
    )
    parser.add_argument("--prior-file", default=DEFAULT_PRIOR_FILE)
    parser.add_argument(
        "--calibration-file",
        default=DEFAULT_CALIBRATION_FILE,
        help="Archivo .py exportado por la calibracion con pesos_posteriores.",
    )
    return parser


def _format_results_table(results: list[dict]) -> str:
    """
    Formatea los resultados de pricing como una tabla legible.
    """
    lines: list[str] = []
    header = (
        f"{'Tipo':<6} {'S0':>10} {'T1':>10} {'T2':>10} "
        f"{'K':>12} {'Precio WMC':>14}"
    )
    lines.append(header)
    lines.append("-" * len(header))

    current_pair = None
    for row in results:
        pair = (row["T1"], row["T2"])
        if current_pair is not None and pair != current_pair:
            lines.append("")
        current_pair = pair

        tipo = "CALL" if row["option_type"] == OptionType.CALL else "PUT"
        lines.append(
            f"{tipo:<6} {row['S0']:>10.4f} {row['T1']:>10.4f} {row['T2']:>10.4f} "
            f"{row['strike']:>12.4f} "
            f"{row['price']:>14.{DISPLAY_DECIMALS}f}"
        )

    return "\n".join(lines)


def main() -> None:
    args, _ = _build_parser().parse_known_args()

    T1_list = [float(t) for t in args.T1]
    T2_list = [float(t) for t in args.T2]
    strikes = _validate_spot_strikes(args.strikes)

    prior_path = _module_path(args.prior_file)
    calibration_path = _module_path(args.calibration_file)

    path_matrix, times, weights, weights_mode = _load_paths_and_weights(
        prior_file=str(prior_path),
        calibration_file=str(calibration_path),
    )
    pricing_s0 = _resolve_pricing_s0(
        path_matrix=path_matrix,
        times=times,
        requested_s0=float(args.S0),
    )

    n_paths = path_matrix.shape[0]
    print(f"Trayectorias: {n_paths}")
    print(f"Suma pesos calibrados: {np.sum(weights):.10f}")
    print(f"Modo pesos: {weights_mode}")
    print(f"S0 pricing: {pricing_s0:.4f}")
    print(f"Tiempos en la malla: {times.tolist()}")
    print()

    results: list[dict] = []
    for T1_value in T1_list:
        for T2_value in T2_list:
            if T2_value <= T1_value:
                print(f"Aviso: T2={T2_value} <= T1={T1_value}. Se omite esta combinacion.")
                continue
            for strike in strikes:
                price = wmc_forward_start_price(
                    option_type=args.option_type,
                    S0_value=pricing_s0,
                    T1=float(T1_value),
                    T2=float(T2_value),
                    strike=float(strike),
                    path_matrix=path_matrix,
                    times=times,
                    weights=weights,
                )
                results.append(
                    {
                        "option_type": args.option_type,
                        "S0": pricing_s0,
                        "T1": float(T1_value),
                        "T2": float(T2_value),
                        "strike": float(strike),
                        "price": price,
                    }
                )

    if results:
        print(_format_results_table(results))
    else:
        print("No se han generado resultados. Revisa que T2 > T1.")


if __name__ == "__main__":
    main()
