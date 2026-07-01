"""
Script para debuguear y mostrar las volatilidades implícitas ATMF
que se usan en la simulación prior.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from data_loader import BENCHMARK_OPTIONS, sigma_prior
from gbm_prior_paths import (
    _infer_total_variance_curve_from_benchmarks,
)


def main():
    # Calcular la curva de varianza total implícita
    benchmark_times, total_variances = _infer_total_variance_curve_from_benchmarks(
        benchmarks=BENCHMARK_OPTIONS,
        fallback_sigma=sigma_prior,
    )
    
    print("=" * 80)
    print("VOLATILIDADES IMPLÍCITAS EXTRAÍDAS DE LOS BENCHMARKS")
    print("=" * 80)
    print()
    
    print("Vencimientos benchmark detectados:")
    for t, tv in zip(benchmark_times, total_variances):
        implied_vol = np.sqrt(tv / t) if t > 0 else 0.0
        print(
            f"  T={t:8.2f} (años)  ->  "
            f"Varianza Total={tv:.2f}  ->  Volatilidad={implied_vol:.2f}"
        )
    
    print()
    print("=" * 80)
    print(f"Volatilidad fallback (sigma_prior): {sigma_prior:.2f}")
    print("=" * 80)

    implied_vols = np.sqrt(total_variances / benchmark_times)

    fig, ax = plt.subplots(figsize=(8, 4.8))

    ax.plot(benchmark_times, implied_vols, marker="o", linewidth=2)
    ax.set_title("Volatilidad implicita ATMF")
    ax.set_xlabel("Vencimiento (anos)")
    ax.set_ylabel("Volatilidad")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
