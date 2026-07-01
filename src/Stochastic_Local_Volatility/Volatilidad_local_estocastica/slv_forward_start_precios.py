#%%
# SLV-Heston: precios de opciones forward-start con volatilidad local estocástica.
#
# Las opciones forward-start dependen del rendimiento porcentual del subyacente
# entre dos fechas futuras T1 (inicio) y T2 (vencimiento):
#
#     Payoff_call = max( (S(T2) - S(T1)) / S(T1) - K, 0 )
#     Payoff_put  = max( K - (S(T2) - S(T1)) / S(T1), 0 )
#
# En esa formula, K es un rendimiento. Para poder introducir strikes en
# unidades de precio y obtener precios monetarios comparables con una vainilla,
# este script usa K_payoff = K_spot / S0 - 1 y multiplica el payoff por S(T1).
# Ejemplo: con T1~0, S0=100 y K_spot=70, el payoff converge a (S(T2)-70)^+.
#
# El proceso SLV simula la dinámica conjunta calibrada:
#     dS/S = r dt + L(t, S) sqrt(V) dW^S
#     dV   = kappa(theta - V) dt + gamma sqrt(V) dW^V
# con L(t, S) = sigma_LV(t, S) / sqrt(E[V | S]).
#
# Los precios se calculan como la media Monte Carlo de los payoffs descontados
# sobre las trayectorias simuladas (pesos uniformes 1/N).
#
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from common.types import OptionType
from Stochastic_Local_Volatility.Volatilidad_local_estocastica.dupire_local_vol import (
    build_local_vol_surface,
)
from Stochastic_Local_Volatility.Volatilidad_local_estocastica.heston_calibration import (
    CalibrationConfig,
    HestonParams,
    calibrate_heston_from_market,
)
from Stochastic_Local_Volatility.Volatilidad_local_estocastica.market_surface import (
    AVAILABLE_STRIKES,
    MATURITIES as MARKET_MATURITIES,
    MATURITY_LABELS as MARKET_MATURITY_LABELS,
    load_market_surface,
)
from Stochastic_Local_Volatility.Volatilidad_local_estocastica.slv_heston_mc import (
    SLVSimulationConfig,
    simulate_heston_slv,
)


DISPLAY_DECIMALS = 2

S0 = 100
# T1 = (0.5,1.0,2.0,5.0)
# T2 = (1.0,2.0,5.0,10.0)            # Fecha(s) de vencimiento (T2 > T1)
# STRIKES = (70,90, 110)   # Strikes en unidades de precio
T1 = (2.0,)
T2 = (10,)
STRIKES = (60.0, 70.0, 80.0, 90.0, 100.0, 110.0, 120.0, 130.0, 140.0)


OPTION_TYPE = "call"           # "call" o "put"
N_PATHS = 50_000
STEPS_PER_YEAR = 50
N_BINS = 20
SCHEME = "Euler"
SEED = 12345


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
        for label, maturity in zip(MARKET_MATURITY_LABELS, MARKET_MATURITIES)
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
    """
    Permite configurar T1/T2 como escalar o como iterable en la cabecera.
    """
    if isinstance(value, (str, bytes)):
        return [float(value)]
    try:
        return [float(item) for item in value]
    except TypeError:
        return [float(value)]


def _validate_forward_strikes(strikes: list[float] | tuple[float, ...] | np.ndarray) -> np.ndarray:
    """
    Valida y devuelve strikes en unidades de precio.
    """
    strike_arr = np.asarray(strikes, dtype=float).reshape(-1)
    if strike_arr.size == 0:
        raise ValueError("Debes proporcionar al menos un strike forward.")
    if not np.all(np.isfinite(strike_arr)):
        raise ValueError("Todos los strikes deben ser finitos.")
    if np.any(strike_arr <= 0.0):
        raise ValueError("Los strikes en unidades de precio deben ser positivos.")
    return strike_arr


def _return_strikes_from_spot_strikes(strikes: np.ndarray, S0_value: float) -> np.ndarray:
    """
    Convierte strikes de precio a la K de rendimiento de la formula forward-start.
    """
    if S0_value <= 0.0:
        raise ValueError("S0 debe ser positivo.")
    return np.asarray(strikes, dtype=float) / float(S0_value) - 1.0


def price_forward_start_mc(
    spots_T1: np.ndarray,
    spots_T2: np.ndarray,
    strike: float,
    T2: float,
    r: float,
    CP: OptionType,
) -> float:
    """
    Precio Monte Carlo de una opcion forward-start.

        Precio = e^{-r T2} * (1/N) * sum_i S_T1 * max((S_T2-S_T1)/S_T1 - K, 0)

    Con K = K_spot/S0 - 1, el payoff equivale a
        max(S_T2 - (K_spot/S0) * S_T1, 0)
    y para T1 cercano a 0 converge a una vanilla con strike K_spot.
    """
    rendimiento = (spots_T2 - spots_T1) / np.maximum(spots_T1, 1e-12)

    if CP == OptionType.CALL:
        payoffs = spots_T1 * np.maximum(rendimiento - strike, 0.0)
    elif CP == OptionType.PUT:
        payoffs = spots_T1 * np.maximum(strike - rendimiento, 0.0)
    else:
        raise ValueError("OptionType debe ser CALL o PUT.")

    return float(np.exp(-r * T2) * np.mean(payoffs))


def run_slv_forward_start(
    S0_value: float,
    T1_list: list[float],
    T2_list: list[float],
    strikes: list[float],
    option_type: OptionType,
    n_paths: int = N_PATHS,
    steps_per_year: int = STEPS_PER_YEAR,
    n_bins: int = N_BINS,
    scheme: str = SCHEME,
    seed: int = SEED,
) -> pd.DataFrame:
    """
    Ejecuta la valoracion de opciones forward-start bajo SLV completo.
    """
    T1_list = [float(t) for t in T1_list]
    T2_list = [float(t) for t in T2_list]
    strikes_arr = _validate_forward_strikes(strikes)
    valid_pairs = [(t1, t2) for t1 in T1_list for t2 in T2_list if t2 > t1]
    if not valid_pairs:
        raise ValueError("Debe existir al menos una combinacion con T2 > T1.")

    # 1. Superficie de mercado
    market_surface = load_market_surface(
        selected_strikes=tuple(float(x) for x in AVAILABLE_STRIKES),
        S0=float(S0_value),
    )
    payoff_strikes = _return_strikes_from_spot_strikes(strikes_arr, market_surface.S0)

    # 2. Calibrar Heston
    calibracion = calibrate_heston_from_market(
        market_surface=market_surface,
        config=CalibrationConfig(option_type=OptionType.CALL),
    )
    params = calibracion.params

    # 3. Superficie de volatilidad local de Dupire
    local_vol_surface = build_local_vol_surface(
        market_surface=market_surface,
        option_type=OptionType.CALL,
    )

    # 4. Recopilar todos los vencimientos unicos para la simulación
    all_maturities = sorted({t for pair in valid_pairs for t in pair})

    # 5. Simular SLV
    sim_config = SLVSimulationConfig(
        maturities=tuple(float(t) for t in all_maturities),
        strikes=(float(S0_value),),  # placeholder, no relevante para forward-start
        n_paths=n_paths,
        steps_per_year=steps_per_year,
        n_bins=n_bins,
        seed=seed,
        option_type=option_type,
        scheme=scheme,
    )
    slv_result = simulate_heston_slv(
        config=sim_config,
        heston_params=params,
        market_surface=market_surface,
        local_vol_surface=local_vol_surface,
    )

    # 6. Calcular precios forward-start
    rows: list[dict] = []
    for T1, T2 in valid_pairs:
        key_1 = round(float(T1), 12)
        key_2 = round(float(T2), 12)
        if key_1 not in slv_result.snapshots or key_2 not in slv_result.snapshots:
            print(
                f"Aviso: No hay snapshot para T1={T1:.2f} o T2={T2:.2f}. "
                "Se omite esta combinacion."
            )
            continue

        spots_T1 = slv_result.snapshots[key_1]
        spots_T2 = slv_result.snapshots[key_2]

        for K_spot, K_payoff in zip(strikes_arr, payoff_strikes):
            slv_price = price_forward_start_mc(
                spots_T1=spots_T1,
                spots_T2=spots_T2,
                strike=float(K_payoff),
                T2=T2,
                r=market_surface.r,
                CP=option_type,
            )

            rows.append({
                "option_type": "call" if option_type == OptionType.CALL else "put",
                "S0": float(market_surface.S0),
                "T1": float(T1),
                "T2": float(T2),
                "strike": float(K_spot),
                "slv_price": slv_price,
            })

    df = pd.DataFrame(rows)
    df.attrs["heston_params"] = params
    return df


def _build_parser() -> argparse.ArgumentParser:
    available = ", ".join(
        f"{label}={float(maturity):.2f}"
        for label, maturity in zip(MARKET_MATURITY_LABELS, MARKET_MATURITIES)
    )
    parser = argparse.ArgumentParser(
        description=(
            "Valora opciones forward-start bajo el modelo SLV completo calibrado.\n"
            "El payoff forward-start monetario es S(T1)*max((S(T2)-S(T1))/S(T1) - K, 0)\n"
            "para calls y S(T1)*max(K - (S(T2)-S(T1))/S(T1), 0) para puts.\n"
            "Los strikes de entrada se dan en unidades de precio y se convierten a\n"
            "K = strike/S0 - 1 dentro del payoff."
        ),
        epilog=f"Vencimientos disponibles: {available}.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--S0",
        type=float,
        default=S0,
        help="Spot inicial usado para construir la superficie/simulacion. Default: 100.",
    )
    parser.add_argument(
        "--option-type",
        type=_option_type,
        default=_option_type(OPTION_TYPE),
        help="Tipo de opcion: 'call' o 'put'. Default: call.",
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
        help=(
            "Strike(s) en unidades de precio. Con S0=100, K=90 equivale "
            "a -10%% en el payoff y K=120 equivale a +20%%. ATM: K=S0."
        ),
    )
    parser.add_argument("--n-paths", type=int, default=N_PATHS)
    parser.add_argument("--steps-per-year", type=int, default=STEPS_PER_YEAR)
    parser.add_argument("--n-bins", type=int, default=N_BINS)
    parser.add_argument("--scheme", choices=["Euler", "AES"], default=SCHEME)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser


def _print_results(df: pd.DataFrame, params: HestonParams | None = None) -> None:
    if params is not None:
        print("=== Parametros de la componente estocastica Heston usados en SLV ===")
        print(f"  kappa = {params.kappa:.2f}")
        print(f"  theta = {params.theta:.2f}")
        print(f"  gamma = {params.gamma:.2f}")
        print(f"  rho   = {params.rho:.2f}")
        print(f"  v0    = {params.v0:.2f}")
        print()

    print(
        f"{'Tipo':<6} {'T1':>10} {'T2':>10} "
        f"{'K':>12} "
        f"{'Precio SLV':>14}"
    )
    print("-" * 68)

    current_pair = None
    for _, row in df.iterrows():
        pair = (row["T1"], row["T2"])
        if current_pair is not None and pair != current_pair:
            print()
        current_pair = pair

        tipo = str(row["option_type"]).upper()
        print(
            f"{tipo:<6} {row['T1']:>10.2f} {row['T2']:>10.2f} "
            f"{row['strike']:>12.2f} "
            f"{row['slv_price']:>14.{DISPLAY_DECIMALS}f}"
        )


def main() -> None:
    args, _ = _build_parser().parse_known_args()

    df = run_slv_forward_start(
        S0_value=float(args.S0),
        T1_list=[float(t) for t in args.T1],
        T2_list=[float(t) for t in args.T2],
        strikes=[float(k) for k in args.strikes],
        option_type=args.option_type,
        n_paths=args.n_paths,
        steps_per_year=args.steps_per_year,
        n_bins=args.n_bins,
        scheme=args.scheme,
        seed=args.seed,
    )

    _print_results(df, params=df.attrs.get("heston_params"))


if __name__ == "__main__":
    main()
