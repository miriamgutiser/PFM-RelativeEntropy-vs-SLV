#%%
"""
SLV-Heston final: precios vanilla con parámetros Heston fijos.

Este script refleja slv_precios.py, pero no recalibra Heston en tiempo de
ejecución. Los parámetros Heston, S0 y r están fijos a los valores reportados
en el capítulo de implementación numérica.
"""

from __future__ import annotations

import argparse

import pandas as pd

from common.types import OptionType
from Stochastic_Local_Volatility.Volatilidad_local_estocastica.dupire_local_vol import (
    LocalVolSurface,
    build_local_vol_surface,
)
from Stochastic_Local_Volatility.Volatilidad_local_estocastica.heston_calibration import (
    HestonParams,
)
from Stochastic_Local_Volatility.Volatilidad_local_estocastica.market_surface import (
    AVAILABLE_STRIKES,
    MATURITIES as MARKET_MATURITIES,
    MATURITY_LABELS as MARKET_MATURITY_LABELS,
    MarketSurface,
    load_market_surface,
)
from Stochastic_Local_Volatility.Volatilidad_local_estocastica.slv_heston_mc import (
    SLVSimulationConfig,
    simulate_heston_slv,
)

# Valores fijos en el informe.
S0 = 100.0
RISK_FREE_RATE = 0.0
FIXED_HESTON_PARAMS = HestonParams(
    kappa=2.12,
    theta=0.06,
    gamma=0.16,
    rho=-0.71,
    v0=0.001,
)

# Valores por defecto para la sección de comparación vanilla.
STRIKES = (70.0, 90.0, 110.0)
MATURITIES = (1.0 / 52.0, 1.0 / 12.0, 0.5, 1.0, 2.0, 5.0, 8.0, 10.0)
OPTION_TYPE = "call"  # "call" o "put"
N_PATHS = 50_000
STEPS_PER_YEAR = 50
N_BINS = 20
SCHEME = "Euler"  # "Euler" o "AES"
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


def _build_parser() -> argparse.ArgumentParser:
    available = ", ".join(
        f"{label}={float(maturity):.2f}"
        for label, maturity in zip(MARKET_MATURITY_LABELS, MARKET_MATURITIES)
    )
    parser = argparse.ArgumentParser(
        description="Valora opciones vanilla bajo Heston-SLV con parámetros fijos.",
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
    parser.add_argument("--n-paths", type=int, default=N_PATHS)
    parser.add_argument("--steps-per-year", type=int, default=STEPS_PER_YEAR)
    parser.add_argument("--n-bins", type=int, default=N_BINS)
    parser.add_argument("--scheme", choices=["Euler", "AES"], default=SCHEME)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--hide-local-variance",
        action="store_true",
        help="No imprime la tabla de varianza local de Dupire.",
    )
    return parser


def _print_price_table(prices: pd.DataFrame) -> None:
    print("=== Precios Heston-SLV finales ===")
    print(f"{'Tipo':<6} {'Strike':>8} {'Vencimiento':>12} {'Precio SLV':>14}")
    print("-" * 45)
    for row in prices.itertuples(index=False):
        tipo = str(row.option_type).upper()
        print(
            f"{tipo:<6} {float(row.strike):>8.2f} "
            f"{float(row.maturity):>12.2f} {float(row.slv_price):>14.2f}"
        )


def _print_fixed_heston_parameters(params: HestonParams) -> None:
    print("=== Parámetros Heston fijos ===")
    print(f"S0        = {S0:.6f}")
    print(f"r         = {RISK_FREE_RATE:.6f}")
    print(f"kappa     = {params.kappa:.6f}")
    print(f"theta     = {params.theta:.6f}")
    print(f"gamma     = {params.gamma:.6f}")
    print(f"rho       = {params.rho:.6f}")
    print(f"v0        = {params.v0:.6f}")
    print(f"feller    = {params.feller_value:.6f}")


def _print_dupire_local_variance(local_vol_surface: LocalVolSurface) -> None:
    variance_table = pd.DataFrame(
        local_vol_surface.local_variance_grid.T,
        index=[float(strike) for strike in local_vol_surface.market_surface.strikes],
        columns=list(local_vol_surface.market_surface.maturity_labels),
    )
    variance_table.index.name = "Strike"

    print("=== Varianza local de Dupire sigma_LV^2(T, K) fija ===")
    print(variance_table.to_string(float_format=lambda value: f"{value:.6f}"))


def prepare_fixed_slv_pricing_inputs() -> tuple[MarketSurface, HestonParams, LocalVolSurface]:
    """
    Construye los objetos fijos usados por el script final de valoración SLV.

    La superficie de volatilidad local determinista se reutiliza desde dupire_local_vol.
    El factor de apalancamiento completo L(t,S) se evalúa después dentro de
    simulate_heston_slv con el mismo estimador de esperanza condicional usado en
    el resto del código.
    """
    market_surface = load_market_surface(
        selected_strikes=tuple(float(x) for x in AVAILABLE_STRIKES),
        S0=S0,
        r=RISK_FREE_RATE,
    )

    # Dupire se construye a partir de precios de calls. Esta superficie es la
    # entrada determinista fija usada por el factor de apalancamiento SLV durante
    # la simulación.
    local_vol_surface = build_local_vol_surface(
        market_surface=market_surface,
        option_type=OptionType.CALL,
    )
    return market_surface, FIXED_HESTON_PARAMS, local_vol_surface


def run_slv_prices_final(
    maturities: tuple[float, ...],
    strikes: tuple[float, ...],
    option_type: OptionType,
    n_paths: int = N_PATHS,
    steps_per_year: int = STEPS_PER_YEAR,
    n_bins: int = N_BINS,
    scheme: str = SCHEME,
    seed: int | None = SEED,
    market_surface: MarketSurface | None = None,
    heston_params: HestonParams | None = None,
    local_vol_surface: LocalVolSurface | None = None,
) -> pd.DataFrame:
    """
    Devuelve precios vanilla Heston-SLV con parámetros Heston fijos.
    """
    if market_surface is None or heston_params is None or local_vol_surface is None:
        market_surface, heston_params, local_vol_surface = prepare_fixed_slv_pricing_inputs()

    sim_config = SLVSimulationConfig(
        maturities=tuple(float(x) for x in maturities),
        strikes=tuple(float(x) for x in strikes),
        n_paths=n_paths,
        steps_per_year=steps_per_year,
        n_bins=n_bins,
        seed=seed,
        option_type=option_type,
        scheme=scheme,
    )
    result = simulate_heston_slv(
        config=sim_config,
        heston_params=heston_params,
        market_surface=market_surface,
        local_vol_surface=local_vol_surface,
    )
    return result.prices.copy()


def main() -> None:
    args, _ = _build_parser().parse_known_args()

    market_surface, heston_params, local_vol_surface = prepare_fixed_slv_pricing_inputs()

    _print_fixed_heston_parameters(heston_params)
    print()
    if not args.hide_local_variance:
        _print_dupire_local_variance(local_vol_surface)
        print()

    prices = run_slv_prices_final(
        maturities=tuple(float(x) for x in args.maturities),
        strikes=tuple(float(x) for x in args.strikes),
        option_type=args.option_type,
        n_paths=args.n_paths,
        steps_per_year=args.steps_per_year,
        n_bins=args.n_bins,
        scheme=args.scheme,
        seed=args.seed,
        market_surface=market_surface,
        heston_params=heston_params,
        local_vol_surface=local_vol_surface,
    )

    _print_price_table(prices)


if __name__ == "__main__":
    main()
