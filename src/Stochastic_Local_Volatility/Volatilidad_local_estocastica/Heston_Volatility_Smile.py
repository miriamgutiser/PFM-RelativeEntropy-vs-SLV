"""
Grafica sonrisas de volatilidad implícita de Heston tras calibrar Heston puro.

Este script es de diagnóstico: no calibra la función de apalancamiento SLV.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common.types import OptionType
from Stochastic_Local_Volatility.Volatilidad_local_estocastica.heston_calibration import (
    CalibrationConfig,
    calibrate_heston_from_market,
    heston_surface_iv,
    option_type_label,
)
from Stochastic_Local_Volatility.Volatilidad_local_estocastica.market_surface import (
    AVAILABLE_STRIKES,
    load_market_surface,
)


DISPLAY_DECIMALS = 7


@dataclass
class SmileConfig:
    calibration_strikes: tuple[float, ...] = (60.0, 80.0, 100.0, 120.0, 140.0)
    smile_strikes: tuple[float, ...] = tuple(float(x) for x in AVAILABLE_STRIKES)
    selected_maturity_labels: tuple[str, ...] = ("1w", "1m", "6m", "1y", "2y", "5y", "8y", "10y")
    CP: OptionType = OptionType.CALL
    calibration_target: str = "prices"
    show_plot: bool = True
    export_plot: bool = True
    plot_filename: str = "heston_volatility_smile.png"
    export_csv: bool = True
    export_filename: str = "heston_volatility_smile.csv"


def find_maturity_indices(all_labels: tuple[str, ...], selected_labels: tuple[str, ...]) -> list[int]:
    indices = []
    for label in selected_labels:
        if label not in all_labels:
            raise ValueError(f"El vencimiento {label!r} no está disponible. Usa uno de: {all_labels}")
        indices.append(all_labels.index(label))
    return indices


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
    out[numeric_cols] = out[numeric_cols].round(DISPLAY_DECIMALS)
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

        ax.plot(x, market_iv, color="#1f77b4", marker="o", linewidth=2.0, label="IV mercado")
        ax.plot(x, heston_iv, color="#d62728", marker="s", linewidth=2.0, linestyle="--", label="Heston IV")
        ax.axvline(100.0, color="gray", linestyle=":", linewidth=1.0)
        ax.set_title(f"Sonrisa T = {maturity_label}")
        ax.set_xlabel("Strike [% de S0]")
        ax.set_ylabel("IV [%]")
        ax.grid(True, alpha=0.3)
        ax.legend()

    for ax in axes_flat[n_plots:]:
        ax.axis("off")

    fig.suptitle(f"Sonrisa de volatilidad - {option_type_label(cp)}", fontsize=14)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.97])
    if output_path is not None:
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
    return fig


def main() -> None:
    config = SmileConfig()
    calibration_surface = load_market_surface(selected_strikes=config.calibration_strikes)
    smile_surface = load_market_surface(selected_strikes=config.smile_strikes)

    calibration_config = CalibrationConfig(
        selected_strikes=config.calibration_strikes,
        option_type=config.CP,
        calibration_target=config.calibration_target,
    )
    calibration = calibrate_heston_from_market(
        market_surface=calibration_surface,
        config=calibration_config,
    )
    params = calibration.params

    print("Calibración Heston para smile")
    print(f"kappa={params.kappa:.7f}, theta={params.theta:.7f}, gamma={params.gamma:.7f}, rho={params.rho:.7f}, v0={params.v0:.7f}")
    print(f"objective={calibration.optimizer_result.fun:.7f}")

    heston_iv_smile = heston_surface_iv(
        config.CP,
        smile_surface.S0,
        smile_surface.strikes,
        smile_surface.maturities,
        smile_surface.r,
        params,
    )
    selected_idx = find_maturity_indices(smile_surface.maturity_labels, config.selected_maturity_labels)

    smile_tables = []
    for idx in selected_idx:
        smile_tables.append(
            build_smile_table(
                maturity_label=smile_surface.maturity_labels[idx],
                maturity=smile_surface.maturities[idx],
                strikes=smile_surface.strikes,
                S0=smile_surface.S0,
                market_iv=smile_surface.implied_vols[idx, :],
                heston_iv=heston_iv_smile[idx, :],
            )
        )

    smile_output = pd.concat(smile_tables, ignore_index=True)
    plot_path = Path(__file__).resolve().with_name(config.plot_filename) if config.export_plot else None
    combined_fig = plot_smiles(smile_tables, config.CP, output_path=plot_path)

    if config.export_csv:
        export_path = Path(__file__).resolve().with_name(config.export_filename)
        sanitize_numeric_columns(smile_output).to_csv(export_path, index=False)
        print(f"CSV exportado en: {export_path}")

    if config.show_plot:
        plt.show()
    else:
        plt.close(combined_fig)


if __name__ == "__main__":
    main()
