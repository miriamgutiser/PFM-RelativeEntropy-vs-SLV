"""
Tabla de convergencia para precios Monte Carlo bajo Heston-SLV.

Edita ``ConvergenceConfig`` para elegir vencimientos, strikes, caminos y pasos
temporales.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from common.types import OptionType
from Stochastic_Local_Volatility.Volatilidad_local_estocastica.dupire_local_vol import build_local_vol_surface
from Stochastic_Local_Volatility.Volatilidad_local_estocastica.heston_calibration import (
    CalibrationConfig,
    calibrate_heston_from_market,
)
from Stochastic_Local_Volatility.Volatilidad_local_estocastica.market_surface import (
    AVAILABLE_STRIKES,
    load_market_surface,
)
from Stochastic_Local_Volatility.Volatilidad_local_estocastica.heston_calibration import bs_price_surface
from Stochastic_Local_Volatility.Volatilidad_local_estocastica.slv_heston_mc import (
    SLVSimulationConfig,
    simulate_heston_slv,
)


@dataclass(frozen=True)
class ConvergenceConfig:
    maturities: tuple[float, ...] = (1.0 / 52.0, 1.0 / 12.0, 0.5, 1.0, 2.0, 5.0, 8.0, 10.0)
    strikes: tuple[float, ...] = (40.0, 60.0, 80.0, 100.0, 120.0, 140.0)
    # steps_per_year son los pasos temporales por año usados en la simulación.
    # Por ejemplo, 40 implica dt = 1/40 y 10 años se simulan con 400 pasos.
    steps_per_year_values: tuple[int, ...] = (5, 10, 20, 40)
    n_paths: int = 25_000
    n_bins: int = 20
    seed: int | None = 12345
    option_type: OptionType = OptionType.CALL
    scheme: str = "Euler"


def _market_price_lookup(config: ConvergenceConfig, market_surface) -> dict[tuple[float, float], float]:
    market_prices = bs_price_surface(
        config.option_type,
        market_surface.S0,
        market_surface.strikes,
        market_surface.maturities,
        market_surface.r,
        market_surface.implied_vols,
    )
    lookup: dict[tuple[float, float], float] = {}
    for i_t, maturity in enumerate(market_surface.maturities):
        for i_k, strike in enumerate(market_surface.strikes):
            lookup[(round(float(maturity), 12), round(float(strike), 12))] = float(market_prices[i_t, i_k])
    return lookup


def build_convergence_table(config: ConvergenceConfig = ConvergenceConfig()) -> pd.DataFrame:
    market_surface = load_market_surface(selected_strikes=tuple(float(x) for x in AVAILABLE_STRIKES))
    calibration = calibrate_heston_from_market(
        market_surface=market_surface,
        config=CalibrationConfig(option_type=config.option_type),
    )
    local_vol_surface = build_local_vol_surface(market_surface, option_type=config.option_type)
    market_lookup = _market_price_lookup(config, market_surface)

    rows = []
    for steps_per_year in config.steps_per_year_values:
        sim_config = SLVSimulationConfig(
            maturities=config.maturities,
            strikes=config.strikes,
            n_paths=config.n_paths,
            steps_per_year=steps_per_year,
            n_bins=config.n_bins,
            seed=config.seed,
            option_type=config.option_type,
            scheme=config.scheme,
        )
        result = simulate_heston_slv(
            config=sim_config,
            heston_params=calibration.params,
            market_surface=market_surface,
            local_vol_surface=local_vol_surface,
        )
        prices = result.prices.copy()
        prices["steps_per_year"] = steps_per_year
        for _, row in prices.iterrows():
            key = (round(float(row["maturity"]), 12), round(float(row["strike"]), 12))
            market_price = market_lookup.get(key, np.nan)
            rows.append(
                {
                    "steps_per_year": steps_per_year,
                    "maturity": float(row["maturity"]),
                    "strike": float(row["strike"]),
                    "slv_price": float(row["slv_price"]),
                    "market_price": market_price,
                    "abs_error": abs(float(row["slv_price"]) - market_price) if np.isfinite(market_price) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    table = build_convergence_table()
    table = table.drop(columns=["steps_per_year"])
    print(table.to_string(index=False, float_format=lambda value: f"{value:.2f}"))


if __name__ == "__main__":
    main()
