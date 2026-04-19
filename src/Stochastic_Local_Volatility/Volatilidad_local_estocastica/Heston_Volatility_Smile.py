"""
==========================================================
Heston_Volatility_Smile.py
==========================================================

Calcula la sonrisa de volatilidad implícita ("volatility smile")
usando la calibración de Heston ya implementada en
`Heston_Calibration_NoPlots.py`.

En este script, el smile se entiende como:

    IV(K, T fijo)

es decir, la volatilidad implícita como función del strike para un
vencimiento concreto.

El flujo es:
1. calibrar Heston con una malla de strikes elegida,
2. reevaluar la volatilidad implícita del modelo sobre una malla de strikes para smile,
3. comparar `market_iv` frente a `heston_iv`,
4. imprimir una tabla por vencimiento y exportar un CSV plano.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


from common.types import OptionType
from Stochastic_Local_Volatility.Volatilidad_local_estocastica.Heston_Calibration_NoPlots import (
    CalibrationConfig,
    DISPLAY_DECIMALS,
    FULL_STRIKE_PCT,
    calibrate_heston,
    heston_surface_iv,
    load_market_surface,
    option_type_label,
    print_params,
    sanitize_display_value,
)


@dataclass
class SmileConfig:
    calibration_strike_pct: tuple[float, ...] = (60.0, 80.0, 100.0, 120.0, 140.0)
    smile_strike_pct: tuple[float, ...] = tuple(float(x) for x in FULL_STRIKE_PCT)
    selected_maturity_labels: tuple[str, ...] = ("1w", "1m", "6m", "1y", "2y", "5y", "8y", "10y")
    CP: OptionType = OptionType.CALL
    calibration_target: str = "prices"
    fix_kappa: bool = True
    show_plot: bool = True
    export_plot: bool = True
    plot_filename: str = "heston_volatility_smile.png"
    export_csv: bool = True
    export_filename: str = "heston_volatility_smile.csv"


def find_maturity_indices(all_labels: list[str], selected_labels: tuple[str, ...]) -> list[int]:
    idx = []
    for label in selected_labels:
        if label not in all_labels:
            raise ValueError(f"El vencimiento '{label}' no está disponible. Usa uno de: {all_labels}")
        idx.append(all_labels.index(label))
    return idx


def build_smile_table(
    maturity_label: str,
    maturity: float,
    strikes: np.ndarray,
    S0: float,
    market_iv: np.ndarray,
    heston_iv: np.ndarray,
) -> pd.DataFrame:
    market_iv_pct = 100.0 * market_iv
    heston_iv_pct = 100.0 * heston_iv

    return pd.DataFrame(
        {
            "maturity_label": maturity_label,
            "maturity": maturity,
            "strike": strikes,
            "strike_pct": 100.0 * strikes / S0,
            "moneyness": strikes / S0,
            "market_iv_pct": market_iv_pct,
            "heston_iv_pct": heston_iv_pct,
            "abs_iv_error_pct": np.abs(heston_iv_pct - market_iv_pct),
        }
    )


def sanitize_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    numeric_cols = out.select_dtypes(include=[np.number]).columns
    out[numeric_cols] = out[numeric_cols].apply(
        lambda col: col.map(
            lambda x: sanitize_display_value(float(x), decimals=DISPLAY_DECIMALS) if pd.notna(x) else x
        )
    )
    return out


def plot_smiles(smile_tables: list[pd.DataFrame], cp: OptionType, output_path: Path | None = None):
    n_plots = len(smile_tables)
    n_cols = 2
    n_rows = int(np.ceil(n_plots / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 3.8 * n_rows), squeeze=False)
    axes_flat = axes.flatten()

    for ax, smile_table in zip(axes_flat, smile_tables):
        maturity_label = str(smile_table["maturity_label"].iloc[0])
        x = smile_table["strike_pct"].to_numpy(dtype=float)
        market_iv = smile_table["market_iv_pct"].to_numpy(dtype=float)
        heston_iv = smile_table["heston_iv_pct"].to_numpy(dtype=float)

        market_mask = np.isfinite(market_iv)
        heston_mask = np.isfinite(heston_iv)

        ax.plot(
            x[market_mask],
            market_iv[market_mask],
            color="#1f77b4",
            marker="o",
            linewidth=2.0,
            label="Market IV",
        )
        ax.plot(
            x[heston_mask],
            heston_iv[heston_mask],
            color="#d62728",
            marker="s",
            linewidth=2.0,
            linestyle="--",
            label="Heston IV",
        )
        ax.axvline(100.0, color="gray", linestyle=":", linewidth=1.0)
        ax.set_title(f"Smile T = {maturity_label}")
        ax.set_xlabel("Strike [% de S0]")
        ax.set_ylabel("IV [%]")
        ax.grid(True, alpha=0.3)
        ax.legend()

    for ax in axes_flat[n_plots:]:
        ax.axis("off")

    fig.suptitle(f"Volatility Smile - {option_type_label(cp)}", fontsize=14)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.97])

    if output_path is not None:
        fig.savefig(output_path, dpi=200, bbox_inches="tight")

    return fig


def plot_single_smile(smile_table: pd.DataFrame, cp: OptionType):
    maturity_label = str(smile_table["maturity_label"].iloc[0])
    x = smile_table["strike_pct"].to_numpy(dtype=float)
    market_iv = smile_table["market_iv_pct"].to_numpy(dtype=float)
    heston_iv = smile_table["heston_iv_pct"].to_numpy(dtype=float)

    market_mask = np.isfinite(market_iv)
    heston_mask = np.isfinite(heston_iv)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        x[market_mask],
        market_iv[market_mask],
        color="#1f77b4",
        marker="o",
        linewidth=2.0,
        label="Market IV",
    )
    ax.plot(
        x[heston_mask],
        heston_iv[heston_mask],
        color="#d62728",
        marker="s",
        linewidth=2.0,
        linestyle="--",
        label="Heston IV",
    )
    ax.axvline(100.0, color="gray", linestyle=":", linewidth=1.0)
    ax.set_title(f"Volatility Smile - {option_type_label(cp)} - T = {maturity_label}")
    ax.set_xlabel("Strike [% de S0]")
    ax.set_ylabel("IV [%]")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


def main() -> None:
    config = SmileConfig()

    calibration_config = CalibrationConfig(
        selected_strike_pct=config.calibration_strike_pct,
        CP=config.CP,
        calibration_target=config.calibration_target,
        fix_kappa=config.fix_kappa,
        export_csv=False,
    )

    S0, calib_strikes, calib_maturities, calib_labels, calib_vol_market = load_market_surface(
        calibration_config.selected_strike_pct
    )
    _, smile_strikes, smile_maturities, smile_labels, smile_vol_market = load_market_surface(
        config.smile_strike_pct
    )

    if calib_labels != smile_labels or not np.allclose(calib_maturities, smile_maturities):
        raise RuntimeError("Las mallas de vencimientos de calibración y smile no coinciden.")

    r = 0.0

    print("=" * 60)
    print("  Cálculo de volatility smile")
    print("=" * 60)
    print(f"  Tipo de opción (CP)     : {option_type_label(config.CP)}")
    print(f"  Strikes calibración [%] : {', '.join(str(int(x)) for x in config.calibration_strike_pct)}")
    print(f"  Strikes smile [%]       : {', '.join(str(int(x)) for x in config.smile_strike_pct)}")
    print(f"  Objetivo calibración    : {config.calibration_target}")
    print(f"  Mostrar gráfica         : {'Sí' if config.show_plot else 'No'}")
    print(f"  Exportar gráfica        : {'Sí' if config.export_plot else 'No'}")
    print()

    print(">>> Calibrando Heston sobre la malla base ...")
    params, result, _ = calibrate_heston(
        CP=calibration_config.CP,
        S0=S0,
        strikes=calib_strikes,
        maturities=calib_maturities,
        r=r,
        vol_market=calib_vol_market,
        calibration_target=calibration_config.calibration_target,
        fix_kappa=calibration_config.fix_kappa,
    )

    objective_name = "RMSE precios" if config.calibration_target == "prices" else "RMSE IV"
    print_params("Parámetros calibrados para el smile", params, objective_name, result.fun)

    heston_iv_smile = heston_surface_iv(config.CP, S0, smile_strikes, smile_maturities, r, params)
    selected_idx = find_maturity_indices(smile_labels, config.selected_maturity_labels)

    all_smiles = []
    smile_tables = []
    for idx in selected_idx:
        smile_table = build_smile_table(
            maturity_label=smile_labels[idx],
            maturity=smile_maturities[idx],
            strikes=smile_strikes,
            S0=S0,
            market_iv=smile_vol_market[idx, :],
            heston_iv=heston_iv_smile[idx, :],
        )
        all_smiles.append(smile_table)
        smile_tables.append(smile_table)

    smile_output = pd.concat(all_smiles, ignore_index=True)

    plot_path = Path(__file__).resolve().with_name(config.plot_filename) if config.export_plot else None
    combined_fig = plot_smiles(smile_tables, config.CP, output_path=plot_path)

    if plot_path is not None:
        print(f"Gráfica exportada en: {plot_path}")
        print()

    if config.export_csv:
        export_path = Path(__file__).resolve().with_name(config.export_filename)
        sanitize_numeric_columns(smile_output).to_csv(
            export_path,
            index=False,
            float_format=f"%.{DISPLAY_DECIMALS}f",
        )
        print(f"CSV exportado en: {export_path}")
        print()

    if config.show_plot:
        for smile_table in smile_tables:
            plot_single_smile(smile_table, config.CP)
        plt.close(combined_fig)
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    main()
