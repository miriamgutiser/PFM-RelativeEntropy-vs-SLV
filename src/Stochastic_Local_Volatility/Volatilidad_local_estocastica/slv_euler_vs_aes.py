"""
Smile de volatilidad implicita para comparar Euler y AES en Heston-SLV.

Este script reutiliza los precios calculados con el flujo de slv_precios.py:
los strikes, vencimientos, numero de caminos, pasos temporales, bins, seed y
tipo de opcion se leen desde ese archivo. Aqui solo se eligen los vencimientos
que se quieren mostrar en la grafica.
"""

from __future__ import annotations

import argparse
import io
import warnings
from contextlib import redirect_stdout
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common.types import OptionType
from Stochastic_Local_Volatility.core.interpolation import ImpliedVolatility
from Stochastic_Local_Volatility.Volatilidad_local_estocastica import slv_precios
from Stochastic_Local_Volatility.Volatilidad_local_estocastica.heston_calibration import price_option_bs
from Stochastic_Local_Volatility.Volatilidad_local_estocastica.market_surface import (
    MATURITIES as MARKET_MATURITIES,
    MATURITY_LABELS as MARKET_MATURITY_LABELS,
)


# Cambia solo esto para decidir que vencimientos dibujar.
# Los precios se calculan para los MATURITIES y STRIKES definidos en slv_precios.py.
# Si quieres dibujar todos los vencimientos calculados, usa PLOT_MATURITIES = ().
PLOT_MATURITIES = (5.0,)


@dataclass(frozen=True)
class SmileComparisonConfig:
    price_maturities: tuple[float, ...] = tuple(float(x) for x in slv_precios.MATURITIES)
    price_strikes: tuple[float, ...] = tuple(float(x) for x in slv_precios.STRIKES)
    plot_maturities: tuple[float, ...] = PLOT_MATURITIES
    n_paths: int = slv_precios.N_PATHS
    steps_per_year: int = slv_precios.STEPS_PER_YEAR
    n_bins: int = slv_precios.N_BINS
    seed: int | None = slv_precios.SEED
    option_type: OptionType = slv_precios._option_type(slv_precios.OPTION_TYPE)


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


def _validate_plot_maturities(config: SmileComparisonConfig) -> None:
    if not config.plot_maturities:
        return
    missing = [
        maturity
        for maturity in config.plot_maturities
        if not any(np.isclose(float(maturity), float(base_t)) for base_t in config.price_maturities)
    ]
    if missing:
        raise ValueError(
            "PLOT_MATURITIES debe estar incluido en MATURITIES de slv_precios.py. "
            f"No encontrados: {missing}"
        )


def _market_iv_lookup(market_surface) -> dict[tuple[float, float], float]:
    lookup: dict[tuple[float, float], float] = {}
    for i_t, maturity in enumerate(market_surface.maturities):
        for i_k, strike in enumerate(market_surface.strikes):
            lookup[(round(float(maturity), 12), round(float(strike), 12))] = float(
                market_surface.implied_vols[i_t, i_k]
            )
    return lookup


def _market_price_lookup(option_type: OptionType, market_surface) -> dict[tuple[float, float], float]:
    lookup: dict[tuple[float, float], float] = {}
    for i_t, maturity in enumerate(market_surface.maturities):
        for i_k, strike in enumerate(market_surface.strikes):
            lookup[(round(float(maturity), 12), round(float(strike), 12))] = price_option_bs(
                option_type,
                market_surface.S0,
                float(strike),
                float(maturity),
                market_surface.r,
                float(market_surface.implied_vols[i_t, i_k]),
            )
    return lookup


def _implied_vol_from_interpolation(
    option_type: OptionType,
    price: float,
    strike: float,
    maturity: float,
    market_surface,
) -> float:
    """
    Invierte el precio usando ImpliedVolatility de core/interpolation.py.
    """
    try:
        with redirect_stdout(io.StringIO()):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                return ImpliedVolatility(
                    option_type,
                    float(price),
                    float(strike),
                    float(maturity),
                    market_surface.S0,
                    market_surface.r,
                )
    except Exception:
        return np.nan


def _run_scheme_prices(
    scheme: str,
    config: SmileComparisonConfig,
    market_surface,
    calibracion,
    local_vol_surface,
) -> pd.DataFrame:
    """
    Llama a slv_precios.py para obtener los precios del esquema indicado.
    """
    if config.seed is not None:
        np.random.seed(config.seed)

    prices = slv_precios.run_slv_prices(
        maturities=config.price_maturities,
        strikes=config.price_strikes,
        option_type=config.option_type,
        n_paths=config.n_paths,
        steps_per_year=config.steps_per_year,
        n_bins=config.n_bins,
        scheme=scheme,
        seed=config.seed,
        market_surface=market_surface,
        calibracion=calibracion,
        local_vol_surface=local_vol_surface,
    )
    prices["scheme"] = scheme
    return prices


def _filter_plot_maturities(comparison: pd.DataFrame, config: SmileComparisonConfig) -> pd.DataFrame:
    if not config.plot_maturities:
        return comparison.copy()
    wanted = {round(float(maturity), 12) for maturity in config.plot_maturities}
    mask = comparison["maturity"].map(lambda value: round(float(value), 12) in wanted)
    return comparison.loc[mask].copy()


def compare_euler_aes(config: SmileComparisonConfig = SmileComparisonConfig()) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Obtiene precios desde slv_precios.py, calcula IVs y devuelve tabla/resumen.
    """
    _validate_plot_maturities(config)

    market_surface, calibracion, local_vol_surface = slv_precios.prepare_slv_pricing_inputs(
        option_type=config.option_type,
    )
    market_iv_lookup = _market_iv_lookup(market_surface)
    market_price_lookup = _market_price_lookup(config.option_type, market_surface)

    scheme_frames = []
    for scheme in ("Euler", "AES"):
        prices = _run_scheme_prices(
            scheme=scheme,
            config=config,
            market_surface=market_surface,
            calibracion=calibracion,
            local_vol_surface=local_vol_surface,
        )
        prices["implied_vol"] = [
            _implied_vol_from_interpolation(
                config.option_type,
                float(row.slv_price),
                float(row.strike),
                float(row.maturity),
                market_surface,
            )
            for row in prices.itertuples(index=False)
        ]
        prices["market_iv"] = [
            market_iv_lookup.get((round(float(row.maturity), 12), round(float(row.strike), 12)), np.nan)
            for row in prices.itertuples(index=False)
        ]
        prices["market_price"] = [
            market_price_lookup.get((round(float(row.maturity), 12), round(float(row.strike), 12)), np.nan)
            for row in prices.itertuples(index=False)
        ]
        prices["abs_iv_error"] = np.abs(prices["implied_vol"] - prices["market_iv"])
        scheme_frames.append(prices)

    long_table = pd.concat(scheme_frames, ignore_index=True)
    pivot = long_table.pivot_table(
        index=["maturity", "strike", "option_type", "market_price", "market_iv"],
        columns="scheme",
        values=["slv_price", "implied_vol", "abs_iv_error"],
        aggfunc="first",
    )
    pivot.columns = [f"{scheme.lower()}_{metric}" for metric, scheme in pivot.columns]
    comparison = pivot.reset_index()
    comparison = _filter_plot_maturities(comparison, config)

    comparison["aes_minus_euler"] = comparison["aes_slv_price"] - comparison["euler_slv_price"]
    comparison["aes_iv_minus_euler_iv"] = comparison["aes_implied_vol"] - comparison["euler_implied_vol"]
    comparison["euler_price_error"] = comparison["euler_slv_price"] - comparison["market_price"]
    comparison["aes_price_error"] = comparison["aes_slv_price"] - comparison["market_price"]
    comparison["market_iv_pct"] = 100.0 * comparison["market_iv"]
    comparison["euler_iv_pct"] = 100.0 * comparison["euler_implied_vol"]
    comparison["aes_iv_pct"] = 100.0 * comparison["aes_implied_vol"]
    comparison["better_scheme"] = np.where(
        comparison["aes_abs_iv_error"] < comparison["euler_abs_iv_error"],
        "AES",
        np.where(comparison["euler_abs_iv_error"] < comparison["aes_abs_iv_error"], "Euler", "Empate"),
    )

    summary_rows = []
    for scheme, iv_col in (("Euler", "euler_implied_vol"), ("AES", "aes_implied_vol")):
        errors = np.abs(comparison[iv_col].to_numpy(dtype=float) - comparison["market_iv"].to_numpy(dtype=float))
        errors = errors[np.isfinite(errors)]
        summary_rows.append(
            {
                "scheme": scheme,
                "n_observations": int(errors.size),
                "mae_vol_points": float(100.0 * np.mean(errors)) if errors.size else np.nan,
                "rmse_vol_points": float(100.0 * np.sqrt(np.mean(errors**2))) if errors.size else np.nan,
                "max_abs_vol_points": float(100.0 * np.max(errors)) if errors.size else np.nan,
                "mean_implied_vol_pct": float(100.0 * comparison[iv_col].mean()),
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values("rmse_vol_points", kind="stable").reset_index(drop=True)
    return comparison, summary


def _subplot_grid(n_items: int) -> tuple[int, int]:
    n_cols = min(4, max(1, int(np.ceil(np.sqrt(n_items)))))
    n_rows = int(np.ceil(n_items / n_cols))
    return n_rows, n_cols


def _label_maturity(maturity: float) -> str:
    for label, value in zip(MARKET_MATURITY_LABELS, MARKET_MATURITIES):
        if np.isclose(float(maturity), float(value)):
            return label
    return f"T={maturity:.2f}"


def plot_implied_vol_smiles(comparison: pd.DataFrame):
    maturities = sorted(comparison["maturity"].unique())
    n_rows, n_cols = _subplot_grid(len(maturities))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.6 * n_cols, 3.4 * n_rows), squeeze=False)

    for ax, maturity in zip(axes.ravel(), maturities):
        data = comparison[comparison["maturity"] == maturity].sort_values("strike")
        strikes = data["strike"].to_numpy(dtype=float)
        ax.plot(strikes, data["market_iv_pct"], color="black", linestyle="--", linewidth=1.8, label="Mercado")
        ax.plot(strikes, data["euler_iv_pct"], marker="o", linewidth=1.6, label="Euler")
        ax.plot(strikes, data["aes_iv_pct"], marker="s", linewidth=1.6, label="AES")
        ax.set_title(_label_maturity(float(maturity)))
        ax.set_xlabel("Strike")
        ax.set_ylabel("Volatilidad implicita [%]")
        ax.grid(True, alpha=0.25)

    for ax in axes.ravel()[len(maturities):]:
        ax.axis("off")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.suptitle("Smile de volatilidad implicita", y=0.99)
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.94), ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.87))
    return fig, axes


def _build_parser() -> argparse.ArgumentParser:
    available = ", ".join(
        f"{label}={float(maturity):.2f}"
        for label, maturity in zip(MARKET_MATURITY_LABELS, MARKET_MATURITIES)
    )
    parser = argparse.ArgumentParser(
        description="Grafica smiles de IV de precios SLV calculados desde slv_precios.py.",
        epilog=f"Vencimientos disponibles: {available}.",
    )
    parser.add_argument(
        "--plot-maturities",
        nargs="*",
        type=_maturity_value,
        default=list(PLOT_MATURITIES),
        help="Vencimientos a dibujar. Si se omite el valor tras el flag, se dibujan todos.",
    )
    return parser


def _print_summary(summary: pd.DataFrame, comparison: pd.DataFrame, config: SmileComparisonConfig) -> None:
    winner = str(summary.iloc[0]["scheme"])
    print("=== Configuracion tomada de slv_precios.py ===")
    print(f"STRIKES      = {config.price_strikes}")
    print(f"MATURITIES   = {config.price_maturities}")
    print(f"PLOT         = {config.plot_maturities if config.plot_maturities else 'todos'}")
    print(f"N_PATHS      = {config.n_paths}")
    print(f"STEPS/YEAR   = {config.steps_per_year}")
    print()
    print("=== Comparacion Euler vs AES en Heston-SLV ===")
    print("Criterio: menor RMSE de volatilidad implicita frente al smile de mercado.")
    print("Las metricas estan en puntos porcentuales de volatilidad.")
    print()
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print()
    print(f"Mejor esquema por RMSE de volatilidad implicita: {winner}")
    print()
    print("=== Precios e IV por punto ===")
    display_cols = [
        "maturity",
        "strike",
        "market_price",
        "euler_slv_price",
        "aes_slv_price",
        "market_iv_pct",
        "euler_iv_pct",
        "aes_iv_pct",
    ]
    print(comparison[display_cols].to_string(index=False, float_format=lambda value: f"{value:.4f}"))

    n_euler_nan = int(comparison["euler_implied_vol"].isna().sum())
    n_aes_nan = int(comparison["aes_implied_vol"].isna().sum())
    if n_euler_nan or n_aes_nan:
        print()
        print(
            "Aviso: hay puntos sin IV finita "
            f"(Euler={n_euler_nan}, AES={n_aes_nan}). "
            "Suele ocurrir cuando el precio Monte Carlo queda fuera de cotas."
        )


def main() -> None:
    args, _ = _build_parser().parse_known_args()
    plot_maturities = tuple(float(x) for x in args.plot_maturities)
    config = SmileComparisonConfig(plot_maturities=plot_maturities)
    comparison, summary = compare_euler_aes(config)
    _print_summary(summary, comparison, config)
    plot_implied_vol_smiles(comparison)
    plt.show()


if __name__ == "__main__":
    main()
