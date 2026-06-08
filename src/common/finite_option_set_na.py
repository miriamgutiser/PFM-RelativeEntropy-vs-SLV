"""
Compatibilidad para el chequeo de no arbitraje de conjuntos finitos.

El modulo canonico es ``common.finite_option_set_arbitrage``. Este archivo
mantiene la interfaz antigua que usaba ``prices`` o ``call_prices``.
"""

from __future__ import annotations

from common.finite_option_set_arbitrage import (
    DEFAULT_TOLERANCE,
    FiniteOptionSetNAReport,
    assert_no_static_arbitrage,
    assert_no_static_arbitrage_surface,
    call_prices_from_option_prices,
    finite_call_option_set_na,
)
from common.finite_option_set_arbitrage import (
    finite_option_set_na as _finite_option_set_na,
)
from common.finite_option_set_arbitrage import (
    finite_option_set_na_report as _finite_option_set_na_report,
)
from common.types import OptionType


StaticArbitrageReport = FiniteOptionSetNAReport


def finite_option_set_na_report(
    factor: float,
    forward: float,
    strikes,
    prices=None,
    option_type: OptionType = OptionType.CALL,
    call_prices=None,
    tolerance: float = DEFAULT_TOLERANCE,
) -> FiniteOptionSetNAReport:
    """
    Diagnostica no arbitraje estatico en una fila de opciones de igual vencimiento.
    """
    if prices is None:
        if call_prices is None:
            raise ValueError("Hay que proporcionar prices.")
        prices = call_prices
        option_type = OptionType.CALL

    return _finite_option_set_na_report(
        option_type=option_type,
        factor=factor,
        forward=forward,
        strikes=strikes,
        option_prices=prices,
        tolerance=tolerance,
    )


def finite_option_set_na(
    factor: float,
    forward: float,
    strikes,
    prices=None,
    option_type: OptionType = OptionType.CALL,
    call_prices=None,
    tolerance: float = DEFAULT_TOLERANCE,
) -> bool:
    """
    Devuelve True si la fila de opciones no presenta arbitraje estatico.
    """
    if prices is None:
        if call_prices is None:
            raise ValueError("Hay que proporcionar prices.")
        prices = call_prices
        option_type = OptionType.CALL

    return _finite_option_set_na(
        option_type=option_type,
        factor=factor,
        forward=forward,
        strikes=strikes,
        option_prices=prices,
        tolerance=tolerance,
    )


__all__ = [
    "DEFAULT_TOLERANCE",
    "FiniteOptionSetNAReport",
    "StaticArbitrageReport",
    "assert_no_static_arbitrage",
    "assert_no_static_arbitrage_surface",
    "call_prices_from_option_prices",
    "finite_call_option_set_na",
    "finite_option_set_na",
    "finite_option_set_na_report",
]
