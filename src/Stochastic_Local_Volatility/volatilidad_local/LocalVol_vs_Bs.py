"""
Traducción a Python del script Matlab `LocalVolHagan`.

El script construye una superficie de volatilidad implícita con la fórmula de
Hagan para SABR, obtiene la varianza local asociada mediante Dupire, simula el
modelo de volatilidad local por Monte Carlo y compara:

    1. La sonrisa de Hagan frente a la sonrisa recuperada de Monte Carlo.
    2. La densidad final simulada frente a una densidad lognormal Black-Scholes.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from scipy.stats import gaussian_kde, norm


def hagan_implied_volatility(K, T, f, alpha, beta, rho, gamma):
    """
    Aproximación de Hagan et al. para la volatilidad implícita SABR.
    """
    K_array = np.asarray(K, dtype=float)
    scalar_input = K_array.ndim == 0
    K_array = np.atleast_1d(K_array)
    K_safe = np.maximum(K_array, 1e-12)

    z = gamma / alpha * (f * K_safe) ** ((1.0 - beta) / 2.0) * np.log(f / K_safe)
    x_z = np.log((np.sqrt(1.0 - 2.0 * rho * z + z**2) + z - rho) / (1.0 - rho))

    denominator = (f * K_safe) ** ((1.0 - beta) / 2.0)
    denominator *= (
        1.0
        + (1.0 - beta) ** 2 / 24.0 * np.log(f / K_safe) ** 2
        + (1.0 - beta) ** 4 / 1920.0 * np.log(f / K_safe) ** 4
    )

    A = alpha / denominator
    B1 = 1.0 + (
        (1.0 - beta) ** 2 / 24.0 * alpha**2 / (f * K_safe) ** (1.0 - beta)
        + 0.25 * rho * beta * gamma * alpha / (f * K_safe) ** ((1.0 - beta) / 2.0)
        + (2.0 - 3.0 * rho**2) / 24.0 * gamma**2
    ) * T

    ratio = np.divide(z, x_z, out=np.ones_like(z), where=np.abs(x_z) > 1e-14)
    imp_vol = A * ratio * B1

    B2 = 1.0 + (
        (1.0 - beta) ** 2 / 24.0 * alpha**2 / f ** (2.0 - 2.0 * beta)
        + 0.25 * rho * beta * gamma * alpha / f ** (1.0 - beta)
        + (2.0 - 3.0 * rho**2) / 24.0 * gamma**2
    ) * T
    imp_vol[np.isclose(K_safe, f)] = alpha / f ** (1.0 - beta) * B2

    if scalar_input:
        return float(imp_vol[0])
    return imp_vol


def bs_option_price(option_type, S0, K, sigma, tau, r):
    """
    Precio Black-Scholes de una call o put europea.
    """
    sigma = max(float(sigma), 1e-12)
    tau = max(float(tau), 1e-12)
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))
    d2 = d1 - sigma * np.sqrt(tau)

    if option_type.lower() == "c":
        return norm.cdf(d1) * S0 - norm.cdf(d2) * K * np.exp(-r * tau)
    if option_type.lower() == "p":
        return norm.cdf(-d2) * K * np.exp(-r * tau) - norm.cdf(-d1) * S0
    raise ValueError("option_type debe ser 'c' o 'p'.")


def implied_volatility(option_type, market_price, K, T, S0, r, initial_vol=0.2):
    """
    Volatilidad implícita Black-Scholes obtenida por búsqueda de raíz.
    """
    intrinsic = max(S0 - K * np.exp(-r * T), 0.0) if option_type.lower() == "c" else max(K * np.exp(-r * T) - S0, 0.0)
    if market_price <= intrinsic + 1e-12:
        return 0.0

    def objective(sigma):
        return bs_option_price(option_type, S0, K, sigma, T, r) - market_price

    try:
        return brentq(objective, 1e-8, 5.0, xtol=1e-12, rtol=1e-12, maxiter=100)
    except ValueError:
        return initial_vol


def local_variance_based_on_sabr(s0, frwd, r, alpha, beta, rho, volvol):
    """
    Construye sigma_LV^2(x,t) a partir de la superficie implícita SABR.
    """
    dt = 0.001
    dx = 0.001

    def sigma(x, t):
        return hagan_implied_volatility(x, max(float(t), 1e-8), frwd, alpha, beta, rho, volvol)

    def dsigmadt(x, t):
        return (sigma(x, t + dt) - sigma(x, t)) / dt

    def dsigmadx(x, t):
        x = np.asarray(x, dtype=float)
        return (sigma(x + dx, t) - sigma(np.maximum(x - dx, 1e-12), t)) / (2.0 * dx)

    def d2sigmadx2(x, t):
        x = np.asarray(x, dtype=float)
        return (sigma(x + dx, t) + sigma(np.maximum(x - dx, 1e-12), t) - 2.0 * sigma(x, t)) / dx**2

    def local_variance(x, t):
        x = np.maximum(np.asarray(x, dtype=float), 1e-12)
        t = max(float(t), 1e-8)

        sigma_xt = sigma(x, t)
        omega = sigma_xt**2 * t
        log_moneyness = np.log(x / (s0 * np.exp(r * t)))

        domegadt = sigma_xt**2 + 2.0 * t * sigma_xt * dsigmadt(x, t)
        domegadx = 2.0 * t * sigma_xt * dsigmadx(x, t)
        d2omegadx2 = 2.0 * t * dsigmadx(x, t) ** 2 + 2.0 * t * sigma_xt * d2sigmadx2(x, t)

        term1 = 1.0 + x * domegadx * (0.5 - log_moneyness / omega)
        term2 = 0.5 * x**2 * d2omegadx2
        term3 = (
            0.5
            * x**2
            * domegadx**2
            * (-0.125 - 1.0 / (2.0 * omega) + log_moneyness**2 / (2.0 * omega**2))
        )

        denominator = term1 + term2 + term3
        return (domegadt + r * x * domegadx) / denominator

    return local_variance


def simulate_local_volatility_paths(s0, T, r, sigmalv2, n_paths=25_000, n_steps=None, seed=2):
    """
    Simula el modelo de volatilidad local usando Euler y moment matching.
    """
    n_steps = int(100 * T) if n_steps is None else int(n_steps)
    dt = T / n_steps
    sqrt_dt = np.sqrt(dt)

    rng = np.random.default_rng(seed)
    S = np.full(n_paths, s0, dtype=float)
    times = np.linspace(0.0, T, n_steps + 1)

    for i in range(n_steps):
        current_time = max(times[i], 0.0001)
        print(f"current time is {current_time:.2f}")

        z = rng.standard_normal(n_paths)
        z = (z - np.mean(z)) / np.std(z)

        variance = np.real(sigmalv2(S, current_time))
        variance = np.nan_to_num(variance, nan=0.0, posinf=0.0, neginf=0.0)
        sigma_lv = np.sqrt(np.maximum(variance, 0.0))

        S = S * (1.0 + r * dt + sigma_lv * z * sqrt_dt)
        S = S - np.mean(S) + s0 * np.exp(r * current_time)
        S = np.maximum(S, 1e-14)

    return S


def main():
    """
    Ejecuta la simulación y genera las dos gráficas del script original.
    """
    beta = 1.0
    rho = 0.0

    volvol = 0.2
    s0 = 1.0
    T = 10.0
    r = 0.05
    alpha = 0.2
    f_0 = s0 * np.exp(r * T)

    n_paths = 25_000
    n_steps = int(100 * T)

    sigma = lambda x, t: hagan_implied_volatility(x, t, f_0, alpha, beta, rho, volvol)
    sigma_atm = hagan_implied_volatility(f_0, T, f_0, alpha, beta, rho, volvol)
    sigmalv2 = local_variance_based_on_sabr(s0, f_0, r, alpha, beta, rho, volvol)

    S_T = simulate_local_volatility_paths(
        s0=s0,
        T=T,
        r=r,
        sigmalv2=sigmalv2,
        n_paths=n_paths,
        n_steps=n_steps,
        seed=2,
    )

    K = np.arange(0.2, 5.0 + 1e-12, 0.15)
    iv_hagan = np.zeros_like(K)
    call_prices = np.zeros_like(K)
    iv_mc = np.zeros_like(K)

    for i, strike in enumerate(K):
        iv_hagan[i] = sigma(strike, T)
        call_prices[i] = np.exp(-r * T) * np.mean(np.maximum(S_T - strike, 0.0))
        iv_mc[i] = implied_volatility("c", call_prices[i], strike, T, s0, r, initial_vol=0.2)

    plt.figure(1)
    plt.plot(K, iv_hagan * 100.0, linewidth=1.5)
    plt.plot(K, iv_mc * 100.0, ".r", markersize=10)
    plt.plot(K, np.full_like(K, sigma_atm * 100.0), "--k", linewidth=1.5)
    plt.title("Modelo de volatilidad local para opciones europeas")
    plt.grid(True)
    plt.xlabel("Strike, K")
    plt.ylabel("Volatilidad implícita [%]")
    plt.legend(
        [
            r"$\hat{\sigma}(T,K)$",
            r"$\sigma_{imp}(T,K)$",
            "Vol. implícita ATM Black-Scholes",
        ]
    )

    density_grid = np.arange(0.01, 7.0 + 1e-12, 0.01)
    kde = gaussian_kde(S_T)
    local_vol_density = kde(density_grid)
    lognormal_density = (
        1.0
        / (sigma_atm * density_grid * np.sqrt(2.0 * np.pi * T))
        * np.exp(
            -(
                np.log(density_grid / s0)
                - (r - 0.5 * sigma_atm**2) * T
            )
            ** 2
            / (2.0 * sigma_atm**2 * T)
        )
    )

    plt.figure(2)
    plt.plot(density_grid, local_vol_density, linewidth=1.5)
    plt.plot(density_grid, lognormal_density, "--r", linewidth=1.5)
    plt.xlabel("x")
    plt.ylabel("Función de densidad")
    plt.title("Volatilidad local vs. Black-Scholes")
    plt.grid(True)
    plt.legend(["Volatilidad local", "Black-Scholes"])

    plt.show()


if __name__ == "__main__":
    main()
