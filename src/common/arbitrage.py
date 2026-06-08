"""
Compatibilidad historica para chequeos de no arbitraje de calls.
"""

from __future__ import annotations

from common.finite_option_set_arbitrage import (
    DEFAULT_TOLERANCE,
    finite_call_option_set_na,
)


def finite_option_set_na(
    factor: float,
    forward: float,
    strikes,
    call_prices,
    tolerance: float = DEFAULT_TOLERANCE,
) -> bool:
    """
    Decide whether a finite set of European call quotes is arbitrage-free.
    """
    return finite_call_option_set_na(
        factor=factor,
        forward=forward,
        strikes=strikes,
        call_prices=call_prices,
        tolerance=tolerance,
    )


__all__ = ["finite_option_set_na"]
