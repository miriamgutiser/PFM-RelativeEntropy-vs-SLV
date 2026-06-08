"""
Static no-arbitrage checks for finite option quote sets.

The routines in this module validate one maturity at a time. Prices are
interpreted as time-0 prices. Put quotes are converted to equivalent call
prices through put-call parity and then checked through the standard finite
strike conditions: bounds, monotonicity and convexity.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from common.types import OptionType


DEFAULT_TOLERANCE = 1e-10


@dataclass(frozen=True)
class FiniteOptionSetNAReport:
    """
    Diagnostic result for a finite option set at one maturity.
    """

    option_type: OptionType
    maturity: float | None
    factor: float
    forward: float
    strikes: np.ndarray
    prices: np.ndarray
    call_prices: np.ndarray
    forward_call_prices: np.ndarray
    state_weights: np.ndarray
    tail_probability: float
    tail_mean: float
    no_arbitrage: bool
    diagnostics: tuple[str, ...]


def _as_1d_float_array(name: str, values) -> np.ndarray:
    array = np.asarray(values, dtype=float).reshape(-1)
    if array.size == 0:
        raise ValueError(f"{name} no puede estar vacio.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contiene valores no finitos.")
    return array


def call_prices_from_option_prices(
    option_type: OptionType,
    factor: float,
    forward: float,
    strikes,
    option_prices,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Sort strikes and return equivalent present-value call prices.

    Args:
        option_type: quoted option type.
        factor: gross compounding factor exp(rT).
        forward: forward price for the maturity.
        strikes: strike vector.
        option_prices: quoted prices at those strikes.
    """
    if not np.isfinite(factor) or factor <= 0.0:
        raise ValueError("factor debe ser positivo y finito.")
    if not np.isfinite(forward) or forward <= 0.0:
        raise ValueError("forward debe ser positivo y finito.")

    strikes_array = _as_1d_float_array("strikes", strikes)
    prices_array = _as_1d_float_array("option_prices", option_prices)
    if strikes_array.size != prices_array.size:
        raise ValueError(
            "strikes y option_prices deben tener la misma longitud. "
            f"Recibido: {strikes_array.size} y {prices_array.size}."
        )
    if np.any(strikes_array <= 0.0):
        raise ValueError("Los strikes deben ser positivos.")
    if np.unique(strikes_array).size != strikes_array.size:
        raise ValueError("Los strikes no deben repetirse.")

    order = np.argsort(strikes_array)
    sorted_strikes = strikes_array[order]
    sorted_prices = prices_array[order]

    if option_type == OptionType.CALL:
        call_prices = sorted_prices
    elif option_type == OptionType.PUT:
        call_prices = sorted_prices + (forward - sorted_strikes) / factor
    else:
        raise ValueError("option_type debe ser OptionType.CALL o OptionType.PUT.")

    return sorted_strikes, sorted_prices, np.asarray(call_prices, dtype=float)


def _format_indices(indices: np.ndarray) -> str:
    return ", ".join(str(int(index)) for index in indices)


def _format_strikes(strikes: np.ndarray) -> str:
    return ", ".join(f"{float(strike):.6g}" for strike in strikes)


def _state_weights_from_slopes(slopes: np.ndarray) -> np.ndarray:
    if slopes.size < 2:
        return np.array([], dtype=float)
    return np.diff(slopes)


def finite_option_set_na_report(
    option_type: OptionType,
    factor: float,
    forward: float,
    strikes,
    option_prices,
    maturity: float | None = None,
    tolerance: float = DEFAULT_TOLERANCE,
) -> FiniteOptionSetNAReport:
    """
    Check static no-arbitrage for a finite set of same-maturity options.

    The check is equivalent to existence of a non-negative terminal
    distribution matching the finite call price set after put-call parity.
    """
    if tolerance < 0.0:
        raise ValueError("tolerance debe ser no negativa.")

    sorted_strikes, sorted_prices, call_prices = call_prices_from_option_prices(
        option_type=option_type,
        factor=factor,
        forward=forward,
        strikes=strikes,
        option_prices=option_prices,
    )
    forward_call_prices = factor * call_prices

    diagnostics: list[str] = []
    lower_bounds = np.maximum(forward - sorted_strikes, 0.0) / factor
    upper_bounds = np.full_like(sorted_strikes, forward / factor)

    lower_violations = np.where(call_prices < lower_bounds - tolerance)[0]
    if lower_violations.size:
        diagnostics.append(
            "precio por debajo de la cota inferior en strikes "
            f"{_format_strikes(sorted_strikes[lower_violations])}"
        )

    upper_violations = np.where(call_prices > upper_bounds + tolerance)[0]
    if upper_violations.size:
        diagnostics.append(
            "precio por encima de la cota superior en strikes "
            f"{_format_strikes(sorted_strikes[upper_violations])}"
        )

    slopes = np.array([], dtype=float)
    state_weights = np.array([], dtype=float)
    tail_probability = 0.0
    tail_mean = float("inf")

    if sorted_strikes.size >= 2:
        strike_steps = np.diff(sorted_strikes)
        slopes = np.diff(forward_call_prices) / strike_steps

        non_decreasing = np.where(slopes > tolerance)[0]
        if non_decreasing.size:
            diagnostics.append(
                "precios no decrecientes entre nodos "
                f"{_format_indices(non_decreasing)}"
            )

        too_steep = np.where(slopes < -1.0 - tolerance)[0]
        if too_steep.size:
            diagnostics.append(
                "pendiente menor que -1 entre nodos "
                f"{_format_indices(too_steep)}"
            )

        convexity_violations = np.where(np.diff(slopes) < -tolerance)[0] + 1
        if convexity_violations.size:
            diagnostics.append(
                "convexidad violada alrededor de strikes "
                f"{_format_strikes(sorted_strikes[convexity_violations])}"
            )

        state_weights = _state_weights_from_slopes(slopes)
        negative_weights = np.where(state_weights < -tolerance)[0]
        if negative_weights.size:
            diagnostics.append(
                "pesos de estado negativos en indices "
                f"{_format_indices(negative_weights)}"
            )

        tail_probability = float(max(0.0, -slopes[-1]))
        if -slopes[-1] < -tolerance:
            diagnostics.append("probabilidad de cola negativa")

    last_forward_call = float(forward_call_prices[-1])
    last_strike = float(sorted_strikes[-1])
    if tail_probability > tolerance:
        tail_mean = last_strike + last_forward_call / tail_probability
        if tail_mean < last_strike - tolerance:
            diagnostics.append("media de cola inferior al ultimo strike")
    elif last_forward_call > tolerance:
        diagnostics.append("precio positivo en el ultimo strike sin probabilidad de cola")

    return FiniteOptionSetNAReport(
        option_type=option_type,
        maturity=maturity,
        factor=float(factor),
        forward=float(forward),
        strikes=sorted_strikes,
        prices=sorted_prices,
        call_prices=call_prices,
        forward_call_prices=forward_call_prices,
        state_weights=state_weights,
        tail_probability=float(tail_probability),
        tail_mean=float(tail_mean),
        no_arbitrage=not diagnostics,
        diagnostics=tuple(diagnostics),
    )


def finite_option_set_na(
    option_type: OptionType,
    factor: float,
    forward: float,
    strikes,
    option_prices,
    tolerance: float = DEFAULT_TOLERANCE,
) -> bool:
    """
    Boolean convenience wrapper for ``finite_option_set_na_report``.
    """
    return finite_option_set_na_report(
        option_type=option_type,
        factor=factor,
        forward=forward,
        strikes=strikes,
        option_prices=option_prices,
        tolerance=tolerance,
    ).no_arbitrage


def finite_call_option_set_na(
    factor: float,
    forward: float,
    strikes,
    call_prices,
    tolerance: float = DEFAULT_TOLERANCE,
) -> bool:
    """
    Compatibility wrapper for finite call quote sets.
    """
    return finite_option_set_na(
        option_type=OptionType.CALL,
        factor=factor,
        forward=forward,
        strikes=strikes,
        option_prices=call_prices,
        tolerance=tolerance,
    )


def assert_no_static_arbitrage(
    option_type: OptionType,
    factor: float,
    forward: float,
    strikes,
    option_prices,
    maturity: float | None = None,
    tolerance: float = DEFAULT_TOLERANCE,
) -> None:
    """
    Validate a quote set and raise ``ValueError`` with diagnostics on failure.
    """
    report = finite_option_set_na_report(
        option_type=option_type,
        factor=factor,
        forward=forward,
        strikes=strikes,
        option_prices=option_prices,
        maturity=maturity,
        tolerance=tolerance,
    )
    if report.no_arbitrage:
        return

    maturity_suffix = "" if maturity is None else f" T={maturity:.8g}"
    raise ValueError(
        "El conjunto finito de opciones tiene arbitraje estatico"
        f"{maturity_suffix}: "
        + "; ".join(report.diagnostics)
    )


def assert_no_static_arbitrage_surface(
    option_type: OptionType,
    spot: float,
    strikes,
    maturities,
    option_price_surface,
    r: float = 0.0,
    q: float = 0.0,
    tolerance: float = DEFAULT_TOLERANCE,
) -> None:
    """
    Validate each maturity row of a price surface.

    ``option_price_surface`` must have shape ``(len(maturities), len(strikes))``.
    """
    strikes_array = _as_1d_float_array("strikes", strikes)
    maturities_array = _as_1d_float_array("maturities", maturities)
    price_surface = np.asarray(option_price_surface, dtype=float)
    expected_shape = (maturities_array.size, strikes_array.size)
    if price_surface.shape != expected_shape:
        raise ValueError(
            "option_price_surface debe tener shape "
            f"{expected_shape}, pero tiene {price_surface.shape}."
        )

    for maturity, row in zip(maturities_array, price_surface):
        factor = float(np.exp(r * maturity))
        forward = float(spot * np.exp((r - q) * maturity))
        assert_no_static_arbitrage(
            option_type=option_type,
            factor=factor,
            forward=forward,
            strikes=strikes_array,
            option_prices=row,
            maturity=float(maturity),
            tolerance=tolerance,
        )
