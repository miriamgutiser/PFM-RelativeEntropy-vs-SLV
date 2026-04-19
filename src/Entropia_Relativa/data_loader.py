"""
Carga de los datos benchmark de opciones para Entropia Relativa.
"""

from __future__ import annotations

from dataclasses import dataclass
from common.types import OptionType

import numpy as np


@dataclass(frozen=True)
class BenchmarkOption:
    """
    Datos de una opcion benchmark.
    """

    CP: OptionType
    strike: float
    maturity: float
    market_price: float
    maturity_label: str

    @property
    def option_type(self) -> OptionType:
        """
        Alias de compatibilidad con código anterior.
        """
        return self.CP


S0 = 100.0
r = 0.0
q = 0.0
CP = OptionType.CALL

MATURITY_LABELS = ["1w", "1m", "6m", "1y", "2y", "5y", "8y", "10y"]
T = np.array([1.0 / 52.0, 1.0 / 12.0, 0.5, 1.0, 2.0, 5.0, 8.0, 10.0], dtype=float)
K = np.array([40.0, 60.0, 80.0, 100.0, 120.0, 140.0], dtype=float)

# Filas: vencimientos en el orden de T.
# Columnas: strikes en el orden de K.
TARGET_OPTION_PRICES = np.array(
    [
        [60.0000000, 40.0000000, 20.0000000, 0.3688735, 0.0000000, 0.0000000],
        [60.0000000, 40.0000000, 20.0000002, 0.7372565, 0.0000000, 0.0000000],
        [60.0000013, 40.0006095, 20.0460501, 1.5783534, 0.0003602, 0.0000004],
        [60.0005858, 40.0185618, 20.2253192, 2.1976425, 0.0092500, 0.0002104],
        [60.0149915, 40.1260323, 20.6311152, 3.2306143, 0.0607294, 0.0052822],
        [60.1431494, 40.5823083, 21.8115167, 5.7249901, 0.3915454, 0.0590638],
        [60.3040076, 41.0614704, 22.9254663, 7.7445782, 1.0171258, 0.1726500],
        [60.4157276, 41.3813603, 23.6314690, 8.9289235, 1.5947005, 0.2943378],
    ],
    dtype=float,
)


def build_benchmark_options() -> list[BenchmarkOption]:
    """
    Devuelve la lista plana de opciones benchmark.
    """
    benchmarks: list[BenchmarkOption] = []
    for maturity_label, maturity, row_prices in zip(
        MATURITY_LABELS, T, TARGET_OPTION_PRICES
    ):
        for strike, market_price in zip(K, row_prices):
            benchmarks.append(
                BenchmarkOption(
                    CP=CP,
                    strike=float(strike),
                    maturity=float(maturity),
                    market_price=float(market_price),
                    maturity_label=maturity_label,
                )
            )
    return benchmarks


BENCHMARK_OPTIONS = build_benchmark_options()

def main():
    print(f">>> Cargados {len(BENCHMARK_OPTIONS)} opciones benchmark.")
    x = BENCHMARK_OPTIONS[30]
    print(f">>> Ejemplo de opción benchmark: {x}")

if __name__ == "__main__":
    main()
