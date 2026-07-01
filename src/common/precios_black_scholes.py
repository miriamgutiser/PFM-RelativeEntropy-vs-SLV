"""
Calcula precios Black-Scholes a partir de una matriz de volatilidades implicitas.

La matriz de volatilidades esta escrita en porcentaje, con filas=strikes y
columnas=vencimientos. El script la convierte a decimal, calcula el precio de
cada opcion con ``BS_Call_Put_Option_Price`` y muestra la tabla por pantalla
con la misma orientacion que ``heston_calibration.py``: filas=vencimientos y
columnas=strikes.
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

import numpy as np
import pandas as pd

from common.types import OptionType
from Stochastic_Local_Volatility.core.pricing_Black_Scholes import (
    BS_Call_Put_Option_Price,
)


S0 = 100.0
R = 0.0
OPTION_TYPE = "put"  # "call" o "put"

STRIKES = np.array([40.0, 60.0, 80.0, 100.0, 120.0, 140.0, 160.0, 180.0], dtype=float)
MATURITY_LABELS = ("1w", "1m", "6m", "1y", "2y", "5y", "8y", "10y")
MATURITIES = np.array([1.0 / 52.0, 1.0 / 12.0, 0.5, 1.0, 2.0, 5.0, 8.0, 10.0], dtype=float)

# Filas: strikes. Columnas: vencimientos. Valores: volatilidades implicitas en porcentaje.
IMPLIED_VOL_PERCENT = np.array(
    [
        [22.622, 22.371, 20.706, 18.537, 18.041, 16.395, 14.491, 13.136],
        [15.491, 15.350, 14.415, 13.195, 12.916, 11.988, 10.917, 10.177],
        [9.930, 9.882, 9.562, 9.154, 9.070, 8.786, 8.459, 8.251],
        [5.001, 5.070, 5.515, 6.041, 6.196, 6.639, 7.015, 7.221],
        [4.542, 4.598, 4.962, 5.430, 5.654, 6.277, 6.782, 7.041],
        [6.278, 6.300, 6.405, 6.460, 6.607, 6.982, 7.232, 7.328],
        [7.748, 7.752, 7.707, 7.517, 7.615, 7.806, 7.824, 7.751],
        [8.988, 8.976, 8.818, 8.445, 8.508, 8.563, 8.399, 8.182],
    ],
    dtype=float,
)


def parse_option_type(value: str) -> OptionType:
    normalized = value.strip().lower()
    if normalized == "call":
        return OptionType.CALL
    if normalized == "put":
        return OptionType.PUT
    raise argparse.ArgumentTypeError("option-type debe ser 'call' o 'put'.")


def option_type_label(option_type: OptionType) -> str:
    return "Call" if option_type == OptionType.CALL else "Put"


def validate_inputs(strikes: np.ndarray, maturities: np.ndarray, vol_percent: np.ndarray) -> None:
    expected_shape = (len(strikes), len(maturities))
    if vol_percent.shape != expected_shape:
        raise ValueError(
            "La matriz de volatilidades debe tener forma "
            f"{expected_shape}, pero tiene forma {vol_percent.shape}."
        )


def black_scholes_price_matrix(
    option_type: OptionType,
    S_0: float,
    strikes: np.ndarray,
    maturities: np.ndarray,
    r: float,
    implied_vol_percent: np.ndarray,
) -> np.ndarray:
    """
    Devuelve una matriz de precios con la orientacion usada en Heston.

    Filas=vencimientos y columnas=strikes.
    """
    validate_inputs(strikes, maturities, implied_vol_percent)
    implied_vols = implied_vol_percent / 100.0
    prices = np.zeros((len(maturities), len(strikes)), dtype=float)

    for i_k, strike in enumerate(strikes):
        for i_t, maturity in enumerate(maturities):
            price = BS_Call_Put_Option_Price(
                option_type,
                S_0,
                float(strike),
                float(implied_vols[i_k, i_t]),
                0.0,
                float(maturity),
                r,
            )
            prices[i_t, i_k] = float(np.atleast_1d(price).flatten()[0])

    return prices


def build_price_table(prices: np.ndarray, strikes: np.ndarray, maturity_labels: tuple[str, ...]) -> pd.DataFrame:
    table = pd.DataFrame(prices, index=maturity_labels, columns=strikes)
    table.index.name = "maturity"
    return table


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Calcula precios Black-Scholes desde una matriz de volatilidades implicitas.",
    )
    parser.add_argument(
        "--option-type",
        type=parse_option_type,
        default=parse_option_type(OPTION_TYPE),
        help="Tipo de opcion: 'call' o 'put'.",
    )
    parser.add_argument("--S0", type=float, default=S0, help="Spot inicial.")
    parser.add_argument("--r", type=float, default=R, help="Tipo libre de riesgo continuo.")
    parser.add_argument(
        "--decimals",
        type=int,
        default=2,
        help="Numero de decimales que se muestran en pantalla.",
    )
    return parser


def main() -> None:
    args, _ = build_parser().parse_known_args()
    prices = black_scholes_price_matrix(
        option_type=args.option_type,
        S_0=float(args.S0),
        strikes=STRIKES,
        maturities=MATURITIES,
        r=float(args.r),
        implied_vol_percent=IMPLIED_VOL_PERCENT,
    )
    table = build_price_table(prices, STRIKES, MATURITY_LABELS)

    print(f"Precios Black-Scholes - {option_type_label(args.option_type)}")
    print(f"S0 = {float(args.S0):.2f}, r = {float(args.r):.2f}")
    print(table.round(args.decimals).to_string(float_format=lambda value: f"{value:.2f}"))


if __name__ == "__main__":
    main()
