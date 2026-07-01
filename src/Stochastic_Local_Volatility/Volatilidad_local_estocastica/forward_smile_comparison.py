"""
Comparación gráfica de sonrisa forward-start entre SV, LV y SLV.

La curva SV corresponde a Heston puro y se valora de forma semianalítica con
la función característica forward-start de Heston. Las curvas LV y SLV se
valoran por Monte Carlo a partir del cociente S(T2) / S(T1).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common.types import OptionType
from Stochastic_Local_Volatility.core.monteCarlo import MCprice
from Stochastic_Local_Volatility.Volatilidad_local_estocastica.dupire_local_vol import (
    LocalVolSurface,
    build_local_vol_surface,
)
from Stochastic_Local_Volatility.Volatilidad_local_estocastica.forward_start import (
    CallPutOptionPriceCOSMthd_FrwdStart,
    ChFHestonForward,
    ImpliedVolatilityFrwdStart,
)
from Stochastic_Local_Volatility.Volatilidad_local_estocastica.heston_calibration import (
    CalibrationConfig,
    HestonParams,
    calibrate_heston_from_market,
)
from Stochastic_Local_Volatility.Volatilidad_local_estocastica.market_surface import load_market_surface
from Stochastic_Local_Volatility.Volatilidad_local_estocastica.slv_heston_mc import (
    SLVSimulationConfig,
    _build_time_grid,
    _standard_normals,
    price_forward_start_from_snapshots,
    simulate_heston_slv,
)


@dataclass(frozen=True)
class ForwardSmileConfig:
    T1: float = 2.0
    T2: float = 5.0
    relative_strikes: tuple[float, ...] = (0.70, 0.80, 0.90, 1.00, 1.10, 1.20, 1.30)
    n_paths: int = 50_000
    steps_per_year: int = 50
    n_bins: int = 20
    seed: int | None = 12345
    option_type: OptionType = OptionType.CALL
    scheme: str = "Euler"
    show_plot: bool = True
    export_plot: bool = False
    plot_filename: str = "forward_smile_slv_lv_sv.png"


def heston_forward_start_prices(
    params: HestonParams,
    config: ForwardSmileConfig,
    r: float,
) -> np.ndarray:
    cf_forward = lambda u: ChFHestonForward(
        u,
        config.T1,
        config.T2,
        params.kappa,
        params.theta,
        params.gamma,
        params.rho,
        params.v0,
        r,
    )
    shifted_strikes = np.asarray(config.relative_strikes, dtype=float) - 1.0
    return CallPutOptionPriceCOSMthd_FrwdStart(
        cf_forward,
        config.option_type,
        r,
        config.T1,
        config.T2,
        shifted_strikes,
        2000,
        5.0,
    )


def simulate_local_vol_snapshots(
    config: ForwardSmileConfig,
    local_vol_surface: LocalVolSurface,
) -> dict[float, np.ndarray]:
    """
    Simula el modelo de volatilidad local puro y guarda S(T1) y S(T2).
    """
    rng = np.random.default_rng(config.seed)
    market_surface = local_vol_surface.market_surface
    time_grid = _build_time_grid((config.T1, config.T2), config.steps_per_year)
    snapshot_times = {round(config.T1, 12), round(config.T2, 12)}

    S = np.full(config.n_paths, market_surface.S0, dtype=float)
    snapshots: dict[float, np.ndarray] = {}

    for i in range(len(time_grid) - 1):
        t = float(time_grid[i])
        t_next = float(time_grid[i + 1])
        dt = t_next - t
        if dt <= 0.0:
            continue

        z = _standard_normals(rng, config.n_paths)
        sigma_lv = np.asarray(local_vol_surface.sigma(max(t, market_surface.maturities[0]), S), dtype=float)
        S = S * np.exp((market_surface.r - 0.5 * sigma_lv**2) * dt + sigma_lv * np.sqrt(dt) * z)

        target_forward = market_surface.S0 * np.exp(market_surface.r * t_next)
        mean_spot = float(np.mean(S))
        if mean_spot > 0.0:
            S = S * (target_forward / mean_spot)
        S = np.maximum(S, 1e-12)

        rounded_t = round(t_next, 12)
        if rounded_t in snapshot_times:
            snapshots[rounded_t] = S.copy()

    return snapshots


def local_vol_forward_start_prices(
    config: ForwardSmileConfig,
    local_vol_surface: LocalVolSurface,
) -> np.ndarray:
    """
    Calcula precios forward-start bajo LV puro usando S(T2)/S(T1).
    """
    snapshots = simulate_local_vol_snapshots(config, local_vol_surface)
    key_1 = round(config.T1, 12)
    key_2 = round(config.T2, 12)
    if key_1 not in snapshots or key_2 not in snapshots:
        raise RuntimeError("No se han guardado las muestras LV necesarias para T1 y T2.")

    forward_ratio = snapshots[key_2] / np.maximum(snapshots[key_1], 1e-12)
    return MCprice(
        np.asarray(config.relative_strikes, dtype=float),
        forward_ratio,
        config.T2,
        local_vol_surface.market_surface.r,
        CP=config.option_type,
    )


def plot_forward_smile(comparison: pd.DataFrame, output_path: Path | None = None):
    """
    Dibuja las volatilidades implícitas forward de SV, LV y SLV.
    """
    fig, ax = plt.subplots(figsize=(9, 6))
    x = comparison["relative_strike"].to_numpy(dtype=float)
    ax.plot(x, 100.0 * comparison["sv_forward_iv"], marker="s", linewidth=2.0, label="SV - Heston")
    ax.plot(x, 100.0 * comparison["lv_forward_iv"], marker="^", linewidth=2.0, label="LV")
    ax.plot(x, 100.0 * comparison["slv_forward_iv"], marker="o", linewidth=2.0, label="SLV")
    ax.axvline(1.0, color="gray", linestyle=":", linewidth=1.0)
    ax.set_xlabel("Strike relativo K/S(T1)")
    ax.set_ylabel("Volatilidad implícita forward [%]")
    ax.set_title(f"Forward smile T1={comparison['T1'].iloc[0]:.2f}, T2={comparison['T2'].iloc[0]:.2f}")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    if output_path is not None:
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
    return fig, ax


def compare_forward_smiles(
    config: ForwardSmileConfig = ForwardSmileConfig(),
    heston_params: HestonParams | None = None,
) -> pd.DataFrame:
    market_surface = load_market_surface()
    if heston_params is None:
        heston_params = calibrate_heston_from_market(
            market_surface=market_surface,
            config=CalibrationConfig(option_type=config.option_type),
        ).params
    local_vol_surface = build_local_vol_surface(market_surface, option_type=config.option_type)

    sim_config = SLVSimulationConfig(
        maturities=(config.T1, config.T2),
        strikes=tuple(market_surface.strikes),
        n_paths=config.n_paths,
        steps_per_year=config.steps_per_year,
        n_bins=config.n_bins,
        seed=config.seed,
        option_type=config.option_type,
        scheme=config.scheme,
    )
    slv_result = simulate_heston_slv(
        config=sim_config,
        heston_params=heston_params,
        market_surface=market_surface,
    )
    slv_forward = price_forward_start_from_snapshots(
        slv_result,
        T1=config.T1,
        T2=config.T2,
        relative_strikes=config.relative_strikes,
    )

    sv_prices = heston_forward_start_prices(heston_params, config, market_surface.r)
    lv_prices = local_vol_forward_start_prices(config, local_vol_surface)
    shifted_strikes = np.asarray(config.relative_strikes, dtype=float) - 1.0
    sv_iv = ImpliedVolatilityFrwdStart(
        sv_prices,
        shifted_strikes,
        config.T1,
        config.T2,
        market_surface.r,
        0.3,
    )
    lv_iv = ImpliedVolatilityFrwdStart(
        lv_prices,
        shifted_strikes,
        config.T1,
        config.T2,
        market_surface.r,
        0.3,
    )
    slv_iv = ImpliedVolatilityFrwdStart(
        slv_forward["slv_forward_start_price"].to_numpy(dtype=float),
        shifted_strikes,
        config.T1,
        config.T2,
        market_surface.r,
        0.3,
    )

    return pd.DataFrame(
        {
            "T1": config.T1,
            "T2": config.T2,
            "relative_strike": np.asarray(config.relative_strikes, dtype=float),
            "sv_forward_price": sv_prices,
            "lv_forward_price": lv_prices,
            "slv_forward_price": slv_forward["slv_forward_start_price"].to_numpy(dtype=float),
            "sv_forward_iv": sv_iv,
            "lv_forward_iv": lv_iv,
            "slv_forward_iv": slv_iv,
        }
    )


def main() -> None:
    config = ForwardSmileConfig()
    comparison = compare_forward_smiles(config)
    print(comparison.to_string(index=False, float_format=lambda value: f"{value:.2f}"))
    plot_path = Path(__file__).resolve().with_name(config.plot_filename) if config.export_plot else None
    plot_forward_smile(comparison, output_path=plot_path)
    if config.show_plot:
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    main()
