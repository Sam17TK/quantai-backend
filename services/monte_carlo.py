"""
services/monte_carlo.py

Monte Carlo risk engine — Section VII of the paper.
  • GARCH(1,1) conditional volatility estimation (Eq. 22–23)
  • GBM price path simulation with LSTM-adjusted drift (Eq. 25)
  • 5% Student-t fat-tail mixing
  • VaR95, CVaR95, profit/loss probabilities (Eq. 26–30)
  • 10,000 paths per stock (Table XIX optimum)
"""
from __future__ import annotations

import math
import numpy as np
from scipy import stats
from typing import Dict, Tuple
from loguru import logger

from config import settings

try:
    from arch import arch_model
    _ARCH_AVAILABLE = True
except ImportError:
    _ARCH_AVAILABLE = False
    logger.warning("arch library not available — using exponential vol estimator")


# ─────────────────────────────────────────────────────────────────────────────
# GARCH(1,1) Volatility Estimation (Eq. 22–23)
# ─────────────────────────────────────────────────────────────────────────────

def estimate_garch_volatility(log_returns: np.ndarray) -> Tuple[float, float]:
    """
    Fit GARCH(1,1) to trailing 252-day log-returns.
    Returns (sigma_daily, sigma_annual).
    Falls back to exponential weighted estimator if arch unavailable.
    """
    returns_pct = log_returns * 100  # arch expects % scale

    if _ARCH_AVAILABLE:
        try:
            am = arch_model(returns_pct, vol="Garch", p=1, q=1, dist="normal",
                            rescale=False)
            res = am.fit(disp="off", show_warning=False)
            # One-step-ahead conditional variance forecast
            fc = res.forecast(horizon=1)
            sigma_daily = float(math.sqrt(fc.variance.iloc[-1, 0])) / 100
        except Exception as exc:
            logger.warning(f"GARCH fit failed ({exc}); using EWMA fallback")
            sigma_daily = _ewma_vol(log_returns)
    else:
        sigma_daily = _ewma_vol(log_returns)

    sigma_annual = sigma_daily * math.sqrt(252)
    return sigma_daily, sigma_annual


def _ewma_vol(log_returns: np.ndarray, lam: float = 0.94) -> float:
    """RiskMetrics EWMA volatility (λ=0.94)."""
    var = float(np.var(log_returns[-20:]))
    for r in reversed(log_returns[-252:]):
        var = lam * var + (1 - lam) * r ** 2
    return math.sqrt(var)


# ─────────────────────────────────────────────────────────────────────────────
# Multi-day Volatility Scaling (Eq. 24)
# ─────────────────────────────────────────────────────────────────────────────

def scale_volatility(sigma_base: float, k: int) -> float:
    """
    Samuelson square-root-of-time with mean-reversion dampening.
    σ̂(T+k) = σ_base · √k · (1 - λ(k-1)/2)   where λ=0.05
    """
    lam = settings.mc_mean_reversion_lambda
    return sigma_base * math.sqrt(k) * (1 - lam * (k - 1) / 2)


# ─────────────────────────────────────────────────────────────────────────────
# Simulation (Eq. 25)
# ─────────────────────────────────────────────────────────────────────────────

def run_monte_carlo(
    price_now: float,
    predicted_daily_returns: list[float],  # LSTM-disaggregated per-day returns (log)
    log_returns_history: np.ndarray,        # trailing 252 days for GARCH
    n_paths: int = 10_000,
) -> Dict:
    """
    Simulate M=10,000 price paths over 7-day horizon using GBM + GARCH + LSTM drift.
    
    Returns a dict with all metrics (Eq. 26–30) plus distribution histogram.
    """
    horizon = settings.lstm_forecast_horizon
    dt      = 1.0 / 252

    sigma_base, sigma_annual = estimate_garch_volatility(log_returns_history)

    # Build per-day volatilities
    sigmas = [scale_volatility(sigma_base, k + 1) for k in range(horizon)]

    # Disaggregate LSTM 7-day cumulative return into daily drift
    mu_daily = np.array(predicted_daily_returns, dtype=np.float64)
    if len(mu_daily) < horizon:
        mu_daily = np.pad(mu_daily, (0, horizon - len(mu_daily)), mode="edge")
    mu_daily = mu_daily[:horizon]

    # Simulate M paths  — shape (M, horizon)
    rng = np.random.default_rng(42)
    normal_innov = rng.standard_normal((n_paths, horizon))

    # Fat-tail mixing: 5% of innovations replaced by Student-t (ν=5)
    t_mask = rng.uniform(size=(n_paths, horizon)) < settings.mc_student_t_fraction
    t_innov = stats.t.rvs(
        df=settings.mc_student_t_df,
        size=(n_paths, horizon),
        random_state=rng.integers(1e6),
    )
    t_scaled = t_innov / math.sqrt(settings.mc_student_t_df / (settings.mc_student_t_df - 2))
    innovations = np.where(t_mask, t_scaled, normal_innov)

    # GBM: S(t+k) = S(t) · exp(Σ [μ_j - σ_j²/2)·dt + σ_j·√dt·ε_j])
    log_price_increments = np.zeros((n_paths, horizon))
    for j in range(horizon):
        drift = (mu_daily[j] - 0.5 * sigmas[j] ** 2) * dt
        diffusion = sigmas[j] * math.sqrt(dt) * innovations[:, j]
        log_price_increments[:, j] = drift + diffusion

    # Cumulative log-price changes → final prices
    log_price_changes = log_price_increments.cumsum(axis=1)
    final_prices = price_now * np.exp(log_price_changes[:, -1])   # S_T+7

    # ── Risk Metrics (Eq. 26–30) ─────────────────────────────────────────────
    p5, p10, p25, p50, p75, p90, p95 = np.percentile(
        final_prices, [5, 10, 25, 50, 75, 90, 95]
    )

    var_95_abs = price_now - p5
    var_95_pct = var_95_abs / price_now * 100

    tail_losses = final_prices[final_prices < p5]
    cvar_95_abs = (price_now - float(tail_losses.mean())) if len(tail_losses) > 0 else var_95_abs
    cvar_95_pct = cvar_95_abs / price_now * 100

    prob_profit   = float(np.mean(final_prices > price_now))
    prob_loss_5pct = float(np.mean(final_prices < 0.95 * price_now))

    # ── Histogram for distribution plot ──────────────────────────────────────
    counts, bin_edges = np.histogram(final_prices, bins=60)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    return {
        "current_price":          round(price_now, 4),
        "paths_simulated":        n_paths,
        "expected_price_median":  round(float(p50), 4),
        "pi_50":                  [round(float(p25), 4), round(float(p75), 4)],
        "pi_80":                  [round(float(p10), 4), round(float(p90), 4)],
        "pi_90":                  [round(float(p5), 4),  round(float(p95), 4)],
        "var_95":                 round(float(var_95_abs), 4),
        "var_95_pct":             round(float(var_95_pct), 4),
        "cvar_95":                round(float(cvar_95_abs), 4),
        "cvar_95_pct":            round(float(cvar_95_pct), 4),
        "prob_profit":            round(prob_profit, 4),
        "prob_loss_5pct":         round(prob_loss_5pct, 4),
        "distribution_bins":      [round(float(x), 2) for x in bin_centers],
        "distribution_counts":    [int(c) for c in counts],
        "garch_volatility_daily": round(sigma_base, 6),
        "garch_volatility_annual":round(sigma_annual, 4),
    }
