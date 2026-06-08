#%%
# SLV-Heston: precios de opciones con volatilidad local estocástica calibrada.
#
# La calibración Heston y la construcción de Dupire se hacen con la superficie
# call de market_surface.py. Después se puede elegir si el payoff valorado por
# Monte Carlo es call o put.

from __future__ import annotations

import argparse

from common.types import OptionType
from Stochastic_Local_Volatility.Volatilidad_local_estocastica.dupire_local_vol import (
    build_local_vol_surface,
)
from Stochastic_Local_Volatility.Volatilidad_local_estocastica.heston_calibration import (
    CalibrationConfig,
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

# Cambia estos valores si ejecutas el archivo por celdas.
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
        f"{label}={float(maturity):.6g}"
        for label, maturity in zip(MARKET_MATURITY_LABELS, MARKET_MATURITIES)
    )
    parser = argparse.ArgumentParser(
        description="Valora opciones vanilla bajo Heston-SLV calibrado.",
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
    return parser


def _print_price_table(prices) -> None:
    print(f"{'Tipo':<6} {'Strike':>8} {'Vencimiento':>12} {'Precio SLV':>14}")
    print("-" * 45)
    for row in prices.itertuples(index=False):
        tipo = str(row.option_type).upper()
        print(
            f"{tipo:<6} {float(row.strike):>8.1f} "
            f"{float(row.maturity):>12.4f} {float(row.slv_price):>14.7f}"
        )


def main() -> None:
    args, _ = _build_parser().parse_known_args()

    market_surface = load_market_surface(
        selected_strikes=tuple(float(x) for x in AVAILABLE_STRIKES)
    )

    # La calibración se realiza con calls, que son también la entrada natural
    # de la fórmula de Dupire usada en dupire_local_vol.py.
    calibracion = calibrate_heston_from_market(
        market_surface=market_surface,
        config=CalibrationConfig(option_type=OptionType.CALL),
    )
    params = calibracion.params
    local_vol_surface = build_local_vol_surface(
        market_surface=market_surface,
        option_type=OptionType.CALL,
    )

    sim_config = SLVSimulationConfig(
        maturities=tuple(float(x) for x in args.maturities),
        strikes=tuple(float(x) for x in args.strikes),
        n_paths=args.n_paths,
        steps_per_year=args.steps_per_year,
        n_bins=args.n_bins,
        seed=args.seed,
        option_type=args.option_type,
        scheme=args.scheme,
    )
    resultado = simulate_heston_slv(
        config=sim_config,
        heston_params=params,
        market_surface=market_surface,
        local_vol_surface=local_vol_surface,
    )

    _print_price_table(resultado.prices)


if __name__ == "__main__":
    main()
