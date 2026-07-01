"""
Compara precios forward-start WMC y SLV en volatilidad implicita.

Por defecto recalcula precios WMC con los pesos guardados, obtiene precios SLV
con el flujo existente y convierte ambos a volatilidad implicita
Black-Scholes forward-start. Si solo se quiere WMC, usa ``--wmc-only``.
"""

from __future__ import annotations

import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(CURRENT_DIR)
if CURRENT_DIR in sys.path:
    sys.path.remove(CURRENT_DIR)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import brentq

from common.types import OptionType
from Entropia_Relativa import wmc_forward_start
from Stochastic_Local_Volatility.Volatilidad_local_estocastica import (
    slv_forward_start_precios,
)
from Stochastic_Local_Volatility.Volatilidad_local_estocastica.forward_start import (
    BS_Call_Option_Price_FrwdStart,
)


DISPLAY_DECIMALS = 4
DEFAULT_PLOT_FILENAME = "forward_start_iv_wmc_vs_slv.png"
DEFAULT_T1_VALUES = (2.0,)
DEFAULT_T2_VALUES = (10.0,)
DEFAULT_STRIKES = tuple(float(value) for value in wmc_forward_start.STRIKES)
IMPLIED_VOL_MIN = 1e-8
IMPLIED_VOL_MAX = 5.0
BOUND_TOL = 1e-7
PRICE_TOL = 1e-12


@dataclass(frozen=True)
class ForwardStartIVComparisonConfig:
    S0: float = float(wmc_forward_start.S0)
    T1_values: tuple[float, ...] = DEFAULT_T1_VALUES
    T2_values: tuple[float, ...] = DEFAULT_T2_VALUES
    strikes: tuple[float, ...] = DEFAULT_STRIKES
    option_type: OptionType = wmc_forward_start._option_type(wmc_forward_start.OPTION_TYPE)
    n_paths: int = slv_forward_start_precios.N_PATHS
    steps_per_year: int = slv_forward_start_precios.STEPS_PER_YEAR
    n_bins: int = slv_forward_start_precios.N_BINS
    scheme: str = slv_forward_start_precios.SCHEME
    seed: int | None = slv_forward_start_precios.SEED
    prior_file: str = wmc_forward_start.DEFAULT_PRIOR_FILE
    calibration_file: str = wmc_forward_start.DEFAULT_CALIBRATION_FILE
    compare_slv: bool = True
    initial_vol: float = 0.3


def _default_output_path() -> Path:
    return Path(__file__).resolve().parents[2] / "results" / DEFAULT_PLOT_FILENAME


def _option_type_label(option_type: OptionType) -> str:
    if option_type == OptionType.CALL:
        return "call"
    if option_type == OptionType.PUT:
        return "put"
    raise ValueError("OptionType debe ser CALL o PUT.")


def _option_type_from_label(label: str) -> OptionType:
    normalized = label.strip().lower()
    if normalized == "call":
        return OptionType.CALL
    if normalized == "put":
        return OptionType.PUT
    raise ValueError(f"Tipo de opcion no reconocido: {label!r}.")


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


def run_wmc_forward_start_prices(config: ForwardStartIVComparisonConfig) -> pd.DataFrame:
    """
    Calcula precios forward-start WMC reutilizando pesos y trayectorias guardadas.
    """
    strikes = wmc_forward_start._validate_spot_strikes(config.strikes)
    prior_path = wmc_forward_start._module_path(config.prior_file)
    calibration_path = wmc_forward_start._module_path(config.calibration_file)

    path_matrix, times, weights, weights_mode = wmc_forward_start._load_paths_and_weights(
        prior_file=str(prior_path),
        calibration_file=str(calibration_path),
    )
    pricing_s0 = wmc_forward_start._resolve_pricing_s0(
        path_matrix=path_matrix,
        times=times,
        requested_s0=float(config.S0),
    )

    rows: list[dict[str, float | str]] = []
    for T1_value, T2_value in _valid_forward_pairs(config.T1_values, config.T2_values):
        for strike in strikes:
            price = wmc_forward_start.wmc_forward_start_price(
                option_type=config.option_type,
                S0_value=pricing_s0,
                T1=T1_value,
                T2=T2_value,
                strike=float(strike),
                path_matrix=path_matrix,
                times=times,
                weights=weights,
            )
            rows.append(
                {
                    "option_type": _option_type_label(config.option_type),
                    "S0": float(pricing_s0),
                    "T1": T1_value,
                    "T2": T2_value,
                    "strike": float(strike),
                    "wmc_price": float(price),
                }
            )

    df = pd.DataFrame(rows)
    df.attrs["weights_mode"] = weights_mode
    df.attrs["n_paths_wmc"] = int(path_matrix.shape[0])
    return df


def run_slv_forward_start_prices(config: ForwardStartIVComparisonConfig) -> pd.DataFrame:
    """
    Calcula precios forward-start SLV con el flujo existente.
    """
    slv_prices = slv_forward_start_precios.run_slv_forward_start(
        S0_value=float(config.S0),
        T1_list=[float(value) for value in config.T1_values],
        T2_list=[float(value) for value in config.T2_values],
        strikes=list(config.strikes),
        option_type=config.option_type,
        n_paths=config.n_paths,
        steps_per_year=config.steps_per_year,
        n_bins=config.n_bins,
        scheme=config.scheme,
        seed=config.seed if config.seed is not None else 12345,
    )
    return slv_prices[["option_type", "T1", "T2", "strike", "slv_price"]].copy()


def _add_merge_keys(df: pd.DataFrame) -> pd.DataFrame:
    keyed = df.copy()
    for column in ("T1", "T2", "strike"):
        keyed[f"_{column}_key"] = keyed[column].map(lambda value: round(float(value), 12))
    return keyed


def _merge_price_tables(wmc_prices: pd.DataFrame, slv_prices: pd.DataFrame) -> pd.DataFrame:
    wmc_keyed = _add_merge_keys(wmc_prices)
    slv_keyed = _add_merge_keys(slv_prices)
    merged = wmc_keyed.merge(
        slv_keyed[["option_type", "_T1_key", "_T2_key", "_strike_key", "slv_price"]],
        on=["option_type", "_T1_key", "_T2_key", "_strike_key"],
        how="inner",
    )
    if merged.empty:
        raise ValueError("No hay puntos comunes entre los resultados WMC y SLV.")

    return merged[["option_type", "S0", "T1", "T2", "strike", "wmc_price", "slv_price"]].copy()


def _call_equivalent_forward_prices(
    normalized_prices: np.ndarray,
    option_type: OptionType,
    strike_ratios: np.ndarray,
    T1: float,
    T2: float,
    r: float,
) -> np.ndarray:
    """
    Convierte precios put  normalizados a calls por paridad.
    """
    tau = float(T2) - float(T1)
    if option_type == OptionType.CALL:
        return normalized_prices
    if option_type == OptionType.PUT:
        return normalized_prices + 1.0 - strike_ratios * np.exp(-r * tau)
    raise ValueError("OptionType debe ser CALL o PUT.")


def _monetary_forward_start_call_price(
    strike_ratio: float,
    sigma: float,
    T1: float,
    T2: float,
    r: float,
) -> float:
    """
    Precio BS normalizado por S0 para el payoff monetario forward-start.

    La funcion existente BS_Call_Option_Price_FrwdStart valora el payoff de
    ratio. El payoff usado por WMC/SLV multiplica por S(T1), por lo que el
    precio normalizado compatible es exp(r*T1) veces ese precio de ratio.
    """
    ratio_price = BS_Call_Option_Price_FrwdStart(
        float(strike_ratio) - 1.0,
        float(sigma),
        float(T1),
        float(T2),
        float(r),
    )
    return float(np.exp(float(r) * float(T1)) * np.asarray(ratio_price, dtype=float).reshape(-1)[0])


def _implied_vol_from_monetary_call_price(
    call_price: float,
    strike_ratio: float,
    T1: float,
    T2: float,
    r: float,
    initial_vol: float,
) -> float:
    if not np.isfinite(call_price) or not np.isfinite(strike_ratio) or strike_ratio <= 0.0:
        return np.nan

    tau = float(T2) - float(T1)
    if tau <= 0.0:
        return np.nan

    lower_bound = max(0.0, 1.0 - float(strike_ratio) * np.exp(-float(r) * tau))
    upper_bound = 1.0
    if call_price < lower_bound - BOUND_TOL or call_price > upper_bound + BOUND_TOL:
        return np.nan

    clipped_price = min(max(float(call_price), lower_bound), upper_bound)
    if clipped_price <= lower_bound + PRICE_TOL:
        return 0.0

    def objective(sigma: float) -> float:
        return _monetary_forward_start_call_price(
            strike_ratio=float(strike_ratio),
            sigma=float(sigma),
            T1=float(T1),
            T2=float(T2),
            r=float(r),
        ) - clipped_price

    high = max(float(initial_vol), 0.25)
    high = min(max(high, IMPLIED_VOL_MIN * 10.0), IMPLIED_VOL_MAX)
    while objective(high) < 0.0 and high < IMPLIED_VOL_MAX:
        high = min(2.0 * high, IMPLIED_VOL_MAX)

    if objective(high) < 0.0:
        return np.nan

    try:
        return float(brentq(objective, IMPLIED_VOL_MIN, high, xtol=1e-12, rtol=1e-12, maxiter=100))
    except ValueError:
        return np.nan


def _forward_start_implied_vols(
    prices: np.ndarray,
    S0_values: np.ndarray,
    option_type: OptionType,
    strikes: np.ndarray,
    T1: float,
    T2: float,
    r: float,
    initial_vol: float,
) -> np.ndarray:
    strike_ratios = np.asarray(strikes, dtype=float) / np.asarray(S0_values, dtype=float)
    normalized_prices = np.asarray(prices, dtype=float) / np.asarray(S0_values, dtype=float)
    call_prices = _call_equivalent_forward_prices(
        normalized_prices=normalized_prices,
        option_type=option_type,
        strike_ratios=strike_ratios,
        T1=float(T1),
        T2=float(T2),
        r=float(r),
    )
    valid = np.isfinite(call_prices) & np.isfinite(strike_ratios) & (strike_ratios > 0.0)
    implied_vols = np.full_like(call_prices, np.nan, dtype=float)
    for idx in np.where(valid)[0]:
        implied_vols[idx] = _implied_vol_from_monetary_call_price(
            call_price=float(call_prices[idx]),
            strike_ratio=float(strike_ratios[idx]),
            T1=float(T1),
            T2=float(T2),
            r=float(r),
            initial_vol=float(initial_vol),
        )
    implied_vols[~np.isfinite(implied_vols) | (implied_vols < 0.0)] = np.nan
    return implied_vols


def add_forward_start_implied_vols(
    comparison: pd.DataFrame,
    config: ForwardStartIVComparisonConfig,
) -> pd.DataFrame:
    """
    Anade volatilidades implicitas forward-start a la tabla de precios.
    """
    enriched = comparison.copy()
    enriched["wmc_forward_iv"] = np.nan
    enriched["slv_forward_iv"] = np.nan

    r = float(wmc_forward_start.r)
    for (option_type_label, T1_value, T2_value), group in enriched.groupby(["option_type", "T1", "T2"]):
        option_type = _option_type_from_label(str(option_type_label))
        strikes = group["strike"].to_numpy(dtype=float)
        S0_values = group["S0"].to_numpy(dtype=float)

        for price_col, iv_col in (("wmc_price", "wmc_forward_iv"), ("slv_price", "slv_forward_iv")):
            enriched.loc[group.index, iv_col] = _forward_start_implied_vols(
                prices=group[price_col].to_numpy(dtype=float),
                S0_values=S0_values,
                option_type=option_type,
                strikes=strikes,
                T1=float(T1_value),
                T2=float(T2_value),
                r=r,
                initial_vol=config.initial_vol,
            )

    enriched["wmc_iv_pct"] = 100.0 * enriched["wmc_forward_iv"]
    enriched["slv_iv_pct"] = 100.0 * enriched["slv_forward_iv"]
    enriched["slv_minus_wmc_iv_pct"] = enriched["slv_iv_pct"] - enriched["wmc_iv_pct"]
    return enriched.sort_values(["T1", "T2", "strike"], kind="stable").reset_index(drop=True)


def add_wmc_forward_start_implied_vols(
    wmc_prices: pd.DataFrame,
    config: ForwardStartIVComparisonConfig,
) -> pd.DataFrame:
    """
    Anade volatilidades implicitas solo a los precios WMC forward-start.
    """
    enriched = wmc_prices.copy()
    enriched["wmc_forward_iv"] = np.nan

    r = float(wmc_forward_start.r)
    for (option_type_label, T1_value, T2_value), group in enriched.groupby(["option_type", "T1", "T2"]):
        option_type = _option_type_from_label(str(option_type_label))
        enriched.loc[group.index, "wmc_forward_iv"] = _forward_start_implied_vols(
            prices=group["wmc_price"].to_numpy(dtype=float),
            S0_values=group["S0"].to_numpy(dtype=float),
            option_type=option_type,
            strikes=group["strike"].to_numpy(dtype=float),
            T1=float(T1_value),
            T2=float(T2_value),
            r=r,
            initial_vol=config.initial_vol,
        )

    enriched["wmc_iv_pct"] = 100.0 * enriched["wmc_forward_iv"]
    return enriched.sort_values(["T1", "T2", "strike"], kind="stable").reset_index(drop=True)


def compare_forward_start_implied_vols(
    config: ForwardStartIVComparisonConfig = ForwardStartIVComparisonConfig(),
) -> pd.DataFrame:
    wmc_prices = run_wmc_forward_start_prices(config)
    if not config.compare_slv:
        return add_wmc_forward_start_implied_vols(wmc_prices=wmc_prices, config=config)

    slv_prices = run_slv_forward_start_prices(config)
    comparison = _merge_price_tables(wmc_prices=wmc_prices, slv_prices=slv_prices)
    return add_forward_start_implied_vols(comparison=comparison, config=config)


def _subplot_grid(n_items: int) -> tuple[int, int]:
    n_cols = min(4, max(1, int(np.ceil(np.sqrt(n_items)))))
    n_rows = int(np.ceil(n_items / n_cols))
    return n_rows, n_cols


def plot_forward_start_iv_comparison(
    comparison: pd.DataFrame,
    output_path: Path | None = None,
):
    pairs = sorted({(float(row.T1), float(row.T2)) for row in comparison[["T1", "T2"]].itertuples(index=False)})
    if len(pairs) == 1:
        fig, ax = plt.subplots(figsize=(7.2, 4.8))
        axes = np.asarray([[ax]])
    else:
        n_rows, n_cols = _subplot_grid(len(pairs))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.8 * n_cols, 3.4 * n_rows), squeeze=False)

    for ax, (T1_value, T2_value) in zip(axes.ravel(), pairs):
        data = comparison[
            np.isclose(comparison["T1"], T1_value) & np.isclose(comparison["T2"], T2_value)
        ].sort_values("strike")
        x = data["strike"].to_numpy(dtype=float)
        ax.plot(x, data["wmc_iv_pct"], marker="o", linewidth=1.8, label="WMC")
        if "slv_iv_pct" in data.columns:
            ax.plot(x, data["slv_iv_pct"], marker="s", linewidth=1.8, label="SLV")
        ax.axvline(float(data["S0"].iloc[0]), color="gray", linestyle=":", linewidth=1.0)
        ax.set_title(f"T1={T1_value:.2f}, T2={T2_value:.2f}")
        ax.set_xlabel("Strike")
        ax.set_ylabel("Volatilidad implicita [%]")
        ax.grid(True, alpha=0.25)

    for ax in axes.ravel()[len(pairs):]:
        ax.axis("off")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.suptitle("Smile forward-start de volatilidad implicita", y=0.99)
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.93), ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.86))

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
    return fig, axes


def summarize_iv_differences(comparison: pd.DataFrame) -> pd.DataFrame:
    if "slv_forward_iv" not in comparison.columns:
        return pd.DataFrame()

    rows = []
    for (T1_value, T2_value), group in comparison.groupby(["T1", "T2"], sort=True):
        diff = group["slv_forward_iv"].to_numpy(dtype=float) - group["wmc_forward_iv"].to_numpy(dtype=float)
        diff = diff[np.isfinite(diff)]
        rows.append(
            {
                "T1": float(T1_value),
                "T2": float(T2_value),
                "n_points": int(diff.size),
                "mean_abs_iv_diff_pct": float(100.0 * np.mean(np.abs(diff))) if diff.size else np.nan,
                "rmse_iv_diff_pct": float(100.0 * np.sqrt(np.mean(diff**2))) if diff.size else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _print_results(comparison: pd.DataFrame, summary: pd.DataFrame, config: ForwardStartIVComparisonConfig) -> None:
    print("=== Configuracion forward-start ===")
    print(f"Tipo opcion: {_option_type_label(config.option_type).upper()}")
    print(f"S0: {config.S0:.2f}")
    print(f"T1: {tuple(float(value) for value in config.T1_values)}")
    print(f"T2: {tuple(float(value) for value in config.T2_values)}")
    print(f"Strikes: {config.strikes}")
    print(f"Fuente WMC: {config.calibration_file}")
    if config.compare_slv:
        print(f"SLV: scheme={config.scheme}, n_paths={config.n_paths}, steps/year={config.steps_per_year}")
    print()

    title_suffix = "WMC vs SLV" if config.compare_slv and "slv_price" in comparison.columns else "WMC"
    print(f"=== Volatilidades implicitas forward-start {title_suffix} ===")
    display_cols = [
        "option_type",
        "T1",
        "T2",
        "strike",
        "wmc_price",
        "wmc_iv_pct",
    ]
    if config.compare_slv and "slv_price" in comparison.columns:
        display_cols.extend(["slv_price", "slv_iv_pct", "slv_minus_wmc_iv_pct"])
    print(comparison[display_cols].to_string(index=False, float_format=lambda value: f"{value:.{DISPLAY_DECIMALS}f}"))

    if not summary.empty:
        print()
        print("=== Distancia SLV vs WMC en volatilidad implicita ===")
        print(summary.to_string(index=False, float_format=lambda value: f"{value:.{DISPLAY_DECIMALS}f}"))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convierte precios forward-start WMC/SLV en IVs y grafica ambos smiles.",
    )
    parser.add_argument("--S0", type=float, default=ForwardStartIVComparisonConfig.S0)
    parser.add_argument(
        "--option-type",
        type=wmc_forward_start._option_type,
        default=ForwardStartIVComparisonConfig.option_type,
        help="Tipo de opcion: call o put.",
    )
    parser.add_argument(
        "--T1",
        nargs="+",
        type=wmc_forward_start._maturity_value,
        default=list(ForwardStartIVComparisonConfig.T1_values),
        help="Fecha(s) de inicio forward. Acepta etiquetas como 6m, 1y, 2y.",
    )
    parser.add_argument(
        "--T2",
        nargs="+",
        type=wmc_forward_start._maturity_value,
        default=list(ForwardStartIVComparisonConfig.T2_values),
        help="Fecha(s) de vencimiento forward. Debe cumplirse T2 > T1.",
    )
    parser.add_argument("--strikes", nargs="+", type=float, default=list(ForwardStartIVComparisonConfig.strikes))
    parser.add_argument("--n-paths", type=int, default=ForwardStartIVComparisonConfig.n_paths)
    parser.add_argument("--steps-per-year", type=int, default=ForwardStartIVComparisonConfig.steps_per_year)
    parser.add_argument("--n-bins", type=int, default=ForwardStartIVComparisonConfig.n_bins)
    parser.add_argument("--scheme", choices=["Euler", "AES"], default=ForwardStartIVComparisonConfig.scheme)
    parser.add_argument("--seed", type=int, default=ForwardStartIVComparisonConfig.seed)
    parser.add_argument("--prior-file", default=ForwardStartIVComparisonConfig.prior_file)
    parser.add_argument(
        "--wmc-only",
        action="store_false",
        dest="compare_slv",
        default=ForwardStartIVComparisonConfig.compare_slv,
        help="Calcula solo WMC y no ejecuta el flujo SLV.",
    )
    parser.add_argument("--calibration-file", default=ForwardStartIVComparisonConfig.calibration_file)
    parser.add_argument("--initial-vol", type=float, default=ForwardStartIVComparisonConfig.initial_vol)
    parser.add_argument("--output", type=Path, default=_default_output_path())
    parser.add_argument(
        "--save-plot",
        action="store_true",
        dest="export_plot",
        default=False,
        help="Guarda tambien el grafico en --output. Por defecto solo se muestra en pantalla.",
    )
    parser.add_argument("--no-export-plot", action="store_false", dest="export_plot")
    parser.add_argument("--no-show", action="store_false", dest="show_plot", default=True)
    return parser


def main() -> None:
    args, _ = _build_parser().parse_known_args()
    config = ForwardStartIVComparisonConfig(
        S0=float(args.S0),
        T1_values=tuple(float(x) for x in args.T1),
        T2_values=tuple(float(x) for x in args.T2),
        strikes=tuple(float(x) for x in args.strikes),
        option_type=args.option_type,
        n_paths=int(args.n_paths),
        steps_per_year=int(args.steps_per_year),
        n_bins=int(args.n_bins),
        scheme=str(args.scheme),
        seed=args.seed,
        prior_file=str(args.prior_file),
        calibration_file=str(args.calibration_file),
        compare_slv=bool(args.compare_slv),
        initial_vol=float(args.initial_vol),
    )

    comparison = compare_forward_start_implied_vols(config)
    summary = summarize_iv_differences(comparison)
    _print_results(comparison=comparison, summary=summary, config=config)

    output_path = Path(args.output) if args.export_plot else None
    plot_forward_start_iv_comparison(comparison=comparison, output_path=output_path)
    if args.export_plot:
        print()
        print(f"Grafico guardado en: {output_path}")

    if args.show_plot:
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    main()
