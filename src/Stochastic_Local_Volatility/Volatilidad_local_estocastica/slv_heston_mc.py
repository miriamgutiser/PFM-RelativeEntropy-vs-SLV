"""
Simulación Monte Carlo del modelo Heston-SLV.

La dinámica simulada del spot es

    dS_t / S_t = r dt + L(t, S_t) sqrt(V_t) dW_t^S

con

    L(t, K) = sigma_LV(t, K) / sqrt(E[V_t | S_t = K]).
Se pasan vencimientos y strikes arbitrarios; el módulo devuelve
precios de opciones vanilla en esos vencimientos.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from common.types import OptionType
from Stochastic_Local_Volatility.core.monteCarlo import MCprice
from Stochastic_Local_Volatility.Volatilidad_local_estocastica.conditional_expectation import estimate_ev_for_paths
from Stochastic_Local_Volatility.Volatilidad_local_estocastica.dupire_local_vol import (
    LocalVolSurface,
    build_local_vol_surface,
)
from Stochastic_Local_Volatility.Volatilidad_local_estocastica.heston_calibration import (
    CalibrationConfig,
    HestonParams,
    calibrate_heston_from_market,
)
from Stochastic_Local_Volatility.Volatilidad_local_estocastica.market_surface import (
    AVAILABLE_STRIKES,
    MarketSurface,
    load_market_surface,
)
from Stochastic_Local_Volatility.volatilidad_estocastica.Heston_model import CIR_Sample


@dataclass(frozen=True)
class SLVSimulationConfig:
    maturities: tuple[float, ...]
    strikes: tuple[float, ...]
    n_paths: int = 50_000
    steps_per_year: int = 50
    n_bins: int = 20
    seed: int | None = 12345
    option_type: OptionType = OptionType.CALL
    scheme: str = "Euler"
    moment_match: bool = True
    min_variance: float = 1e-12


@dataclass(frozen=True)
class SLVSimulationResult:
    config: SLVSimulationConfig
    heston_params: HestonParams
    market_surface: MarketSurface
    snapshots: dict[float, np.ndarray]
    variance_snapshots: dict[float, np.ndarray]
    prices: pd.DataFrame


def _build_time_grid(maturities: tuple[float, ...], steps_per_year: int) -> np.ndarray:
    maturities_arr = np.asarray(maturities, dtype=float)
    if np.any(maturities_arr <= 0.0):
        raise ValueError("Todos los vencimientos deben ser positivos.")
    max_t = float(np.max(maturities_arr))
    n_steps = max(1, int(np.ceil(max_t * steps_per_year)))
    base_grid = np.linspace(0.0, max_t, n_steps + 1)
    time_grid = np.unique(np.concatenate([base_grid, maturities_arr]))
    return np.sort(time_grid)


def _standard_normals(rng: np.random.Generator, n_paths: int) -> np.ndarray:
    z = rng.standard_normal(n_paths)
    if n_paths > 1:
        std = float(np.std(z))
        if std > 0.0:
            z = (z - float(np.mean(z))) / std
    return z


def _price_snapshot(
    spots: np.ndarray,
    maturity: float,
    strikes: tuple[float, ...],
    r: float,
    option_type: OptionType,
) -> np.ndarray:
    return MCprice(np.asarray(strikes, dtype=float), spots, maturity, r, CP=option_type)


def simulate_heston_slv(
    config: SLVSimulationConfig,
    heston_params: HestonParams | None = None,
    market_surface: MarketSurface | None = None,
    local_vol_surface: LocalVolSurface | None = None,
) -> SLVSimulationResult:
    """
    Simula trayectorias Heston-SLV y valora opciones vanilla en los vencimientos
    solicitados.
    """
    if market_surface is None:
        market_surface = load_market_surface(selected_strikes=tuple(float(x) for x in AVAILABLE_STRIKES))
    if heston_params is None:
        calibration = calibrate_heston_from_market(
            market_surface=market_surface,
            config=CalibrationConfig(option_type=config.option_type),
        )
        heston_params = calibration.params
    if local_vol_surface is None:
        local_vol_surface = build_local_vol_surface(market_surface, option_type=config.option_type)

    scheme = config.scheme.strip().upper()
    if scheme not in {"EULER", "AES"}:
        raise ValueError("config.scheme debe ser 'Euler' o 'AES'.")

    rng = np.random.default_rng(config.seed)
    time_grid = _build_time_grid(config.maturities, config.steps_per_year)
    maturity_set = {round(float(t), 12) for t in config.maturities}

    S = np.full(config.n_paths, market_surface.S0, dtype=float)
    V = np.full(config.n_paths, heston_params.v0, dtype=float)
    snapshots: dict[float, np.ndarray] = {}
    variance_snapshots: dict[float, np.ndarray] = {}

    for i in range(len(time_grid) - 1):
        t = float(time_grid[i])
        t_next = float(time_grid[i + 1])
        dt = t_next - t
        if dt <= 0.0:
            continue

        v_pos = np.maximum(V, config.min_variance)
        ev_given_s = estimate_ev_for_paths(S, v_pos, n_bins=config.n_bins, min_value=config.min_variance)
        sigma_lv = np.asarray(local_vol_surface.sigma(max(t, market_surface.maturities[0]), S), dtype=float)
        leverage = sigma_lv / np.sqrt(np.maximum(ev_given_s, config.min_variance))
        leverage = np.maximum(leverage, 0.0)

        z_s = _standard_normals(rng, config.n_paths)
        if scheme == "AES" and heston_params.gamma > config.min_variance:
            v_new = CIR_Sample(
                config.n_paths,
                heston_params.kappa,
                heston_params.gamma,
                heston_params.theta,
                0.0,
                dt,
                v_pos,
            )
            corr_term = (
                heston_params.rho
                * leverage
                / heston_params.gamma
                * (
                    v_new
                    - v_pos
                    - heston_params.kappa * heston_params.theta * dt
                    + heston_params.kappa * v_pos * dt
                )
            )
            independent_term = (
                np.sqrt(max(1.0 - heston_params.rho**2, 0.0))
                * leverage
                * np.sqrt(v_pos * dt)
                * z_s
            )
            S = S * np.exp(
                (market_surface.r - 0.5 * leverage**2 * v_pos) * dt
                + corr_term
                + independent_term
            )
            V = np.maximum(v_new, 0.0)
        else:
            z_aux = _standard_normals(rng, config.n_paths)
            z_v = heston_params.rho * z_s + np.sqrt(max(1.0 - heston_params.rho**2, 0.0)) * z_aux
            inst_vol = leverage * np.sqrt(v_pos)
            S = S * np.exp((market_surface.r - 0.5 * inst_vol**2) * dt + inst_vol * np.sqrt(dt) * z_s)
            V = V + heston_params.kappa * (heston_params.theta - v_pos) * dt
            V = V + heston_params.gamma * np.sqrt(v_pos) * np.sqrt(dt) * z_v
            V = np.maximum(V, 0.0)

        if config.moment_match:
            target_forward = market_surface.S0 * np.exp(market_surface.r * t_next)
            mean_spot = float(np.mean(S))
            if mean_spot > 0.0:
                S = S * (target_forward / mean_spot)
            S = np.maximum(S, 1e-12)

        rounded_t = round(t_next, 12)
        if rounded_t in maturity_set:
            snapshots[rounded_t] = S.copy()
            variance_snapshots[rounded_t] = V.copy()

    rows = []
    for maturity in config.maturities:
        key = round(float(maturity), 12)
        if key not in snapshots:
            raise RuntimeError(f"No se guardó ninguna muestra para el vencimiento {maturity}.")
        prices = _price_snapshot(
            spots=snapshots[key],
            maturity=float(maturity),
            strikes=config.strikes,
            r=market_surface.r,
            option_type=config.option_type,
        )
        for strike, price in zip(config.strikes, prices):
            rows.append(
                {
                    "maturity": float(maturity),
                    "strike": float(strike),
                    "option_type": "call" if config.option_type == OptionType.CALL else "put",
                    "slv_price": float(price),
                }
            )

    return SLVSimulationResult(
        config=config,
        heston_params=heston_params,
        market_surface=market_surface,
        snapshots=snapshots,
        variance_snapshots=variance_snapshots,
        prices=pd.DataFrame(rows),
    )


def price_options_heston_slv(
    maturities: tuple[float, ...],
    strikes: tuple[float, ...],
    heston_params: HestonParams | None = None,
    n_paths: int = 50_000,
    steps_per_year: int = 50,
    n_bins: int = 20,
    seed: int | None = 12345,
    option_type: OptionType = OptionType.CALL,
    scheme: str = "Euler",
) -> pd.DataFrame:
    config = SLVSimulationConfig(
        maturities=tuple(float(t) for t in maturities),
        strikes=tuple(float(k) for k in strikes),
        n_paths=n_paths,
        steps_per_year=steps_per_year,
        n_bins=n_bins,
        seed=seed,
        option_type=option_type,
        scheme=scheme,
    )
    return simulate_heston_slv(config=config, heston_params=heston_params).prices


def price_forward_start_from_snapshots(
    result: SLVSimulationResult,
    T1: float,
    T2: float,
    relative_strikes: tuple[float, ...],
) -> pd.DataFrame:
    key_1 = round(float(T1), 12)
    key_2 = round(float(T2), 12)
    if key_1 not in result.snapshots or key_2 not in result.snapshots:
        raise ValueError("T1 y T2 deben estar incluidos en los vencimientos simulados.")

    forward_ratio = result.snapshots[key_2] / np.maximum(result.snapshots[key_1], 1e-12)
    prices = MCprice(
        np.asarray(relative_strikes, dtype=float),
        forward_ratio,
        float(T2),
        result.market_surface.r,
        CP=result.config.option_type,
    )
    return pd.DataFrame(
        {
            "T1": float(T1),
            "T2": float(T2),
            "relative_strike": np.asarray(relative_strikes, dtype=float),
            "slv_forward_start_price": prices,
        }
    )


def main() -> None:
    params = calibrate_heston_from_market().params
    config = SLVSimulationConfig(
        maturities=(1.0, 2.0, 5.0),
        strikes=(80.0, 100.0, 120.0),
        n_paths=20_000,
        steps_per_year=25,
        seed=12345,
    )
    result = simulate_heston_slv(config=config, heston_params=params)
    print(result.prices.to_string(index=False, float_format=lambda value: f"{value:.2f}"))


if __name__ == "__main__":
    main()
