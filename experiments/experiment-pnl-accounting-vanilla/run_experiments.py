"""
Mock-data experiment for vanilla-option PnL accounting.

The experiment has two layers:

1. Black-Scholes delta-hedged vanilla options.  This verifies the accounting
   identity that daily PnL is explained by dollar gamma times realized variance
   minus implied variance.
2. A toy implied-volatility-surface portfolio.  The same realized spot and
   volatility-factor path is evaluated under several simplified PnL
   breakdown conventions: Black-Scholes, local volatility, stochastic volatility, Bergomi
   two-factor forward variance, admissible LSV, and a non-admissible LSV
   diagnostic.

The data are deliberately synthetic.  The goal is to make Bergomi's accounting
formulas visible, not to calibrate a production pricing model.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

S0 = 100.0
RATE = 0.0
DIVIDEND = 0.0
BASE_VOL = 0.20
TRADING_DAYS = 252
DT = 1.0 / TRADING_DAYS
BS_INITIAL_MATURITY = 1.25
SEED = 20260611


def norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_d1(spot: float, strike: float, maturity: float, vol: float) -> float:
    vol = max(vol, 1e-10)
    maturity = max(maturity, 1e-10)
    return (math.log(spot / strike) + 0.5 * vol * vol * maturity) / (vol * math.sqrt(maturity))


def bs_price(spot: float, strike: float, maturity: float, vol: float, option_type: str) -> float:
    if maturity <= 0:
        if option_type == "call":
            return max(spot - strike, 0.0)
        if option_type == "put":
            return max(strike - spot, 0.0)
        raise ValueError(f"unsupported option type: {option_type}")
    d1 = bs_d1(spot, strike, maturity, vol)
    d2 = d1 - vol * math.sqrt(maturity)
    if option_type == "call":
        return spot * norm_cdf(d1) - strike * norm_cdf(d2)
    if option_type == "put":
        return strike * norm_cdf(-d2) - spot * norm_cdf(-d1)
    raise ValueError(f"unsupported option type: {option_type}")


def bs_delta(spot: float, strike: float, maturity: float, vol: float, option_type: str) -> float:
    if maturity <= 0:
        if option_type == "call":
            return 1.0 if spot > strike else 0.0
        if option_type == "put":
            return -1.0 if spot < strike else 0.0
    d1 = bs_d1(spot, strike, maturity, vol)
    if option_type == "call":
        return norm_cdf(d1)
    if option_type == "put":
        return norm_cdf(d1) - 1.0
    raise ValueError(f"unsupported option type: {option_type}")


def bs_gamma(spot: float, strike: float, maturity: float, vol: float) -> float:
    if maturity <= 0:
        return 0.0
    d1 = bs_d1(spot, strike, maturity, vol)
    return norm_pdf(d1) / (spot * vol * math.sqrt(maturity))


def bs_vega(spot: float, strike: float, maturity: float, vol: float) -> float:
    if maturity <= 0:
        return 0.0
    d1 = bs_d1(spot, strike, maturity, vol)
    return spot * norm_pdf(d1) * math.sqrt(maturity)


def bs_theta(spot: float, strike: float, maturity: float, vol: float) -> float:
    if maturity <= 0:
        return 0.0
    d1 = bs_d1(spot, strike, maturity, vol)
    return -spot * norm_pdf(d1) * vol / (2.0 * math.sqrt(maturity))


def bs_vanna(spot: float, strike: float, maturity: float, vol: float) -> float:
    if maturity <= 0:
        return 0.0
    d1 = bs_d1(spot, strike, maturity, vol)
    sqrt_t = math.sqrt(maturity)
    return norm_pdf(d1) * sqrt_t * (1.0 - d1 / (vol * sqrt_t))


def bs_volga(spot: float, strike: float, maturity: float, vol: float) -> float:
    if maturity <= 0:
        return 0.0
    d1 = bs_d1(spot, strike, maturity, vol)
    d2 = d1 - vol * math.sqrt(maturity)
    return bs_vega(spot, strike, maturity, vol) * d1 * d2 / vol


@dataclass(frozen=True)
class VanillaLeg:
    name: str
    option_type: str
    strike: float
    maturity: float
    weight: float


def write_csv(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    rows = list(rows)
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def make_standardized_normals(n_steps: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n_steps)
    return (z - z.mean()) / z.std(ddof=0)


def generate_return_path(realized_vol: float, n_steps: int, seed: int) -> np.ndarray:
    z = make_standardized_normals(n_steps, seed)
    returns = realized_vol * math.sqrt(DT) * z
    returns *= realized_vol / math.sqrt(np.sum(returns * returns) / (n_steps * DT))
    return returns


def product_value_delta_gamma_theta(
    product: str,
    spot: float,
    maturity: float,
    vol: float,
) -> Tuple[float, float, float, float]:
    if product == "call_atm":
        legs = [VanillaLeg("call", "call", S0, 1.0, 1.0)]
    elif product == "put_atm":
        legs = [VanillaLeg("put", "put", S0, 1.0, 1.0)]
    elif product == "straddle_atm":
        legs = [
            VanillaLeg("call", "call", S0, 1.0, 1.0),
            VanillaLeg("put", "put", S0, 1.0, 1.0),
        ]
    else:
        raise ValueError(product)

    value = delta = gamma = theta = 0.0
    for leg in legs:
        tau = maturity
        value += leg.weight * bs_price(spot, leg.strike, tau, vol, leg.option_type)
        delta += leg.weight * bs_delta(spot, leg.strike, tau, vol, leg.option_type)
        gamma += leg.weight * bs_gamma(spot, leg.strike, tau, vol)
        theta += leg.weight * bs_theta(spot, leg.strike, tau, vol)
    return value, delta, gamma, theta


def run_bs_delta_experiment() -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    products = ["call_atm", "put_atm", "straddle_atm"]
    scenarios = [
        ("low_realized_vol", 0.15, 11),
        ("matched_realized_vol", 0.20, 17),
        ("high_realized_vol", 0.28, 23),
        ("clustered_same_variance", 0.20, 31),
    ]

    daily_rows: List[Dict[str, object]] = []
    summary_rows: List[Dict[str, object]] = []
    for scenario, realized_vol, seed in scenarios:
        returns = generate_return_path(realized_vol, TRADING_DAYS, seed)
        if scenario == "clustered_same_variance":
            # Same annual realized variance as the matched case, but front-load
            # the variance to make the gamma-weighting effect visible.
            z = make_standardized_normals(TRADING_DAYS, seed)
            scale = np.r_[np.full(TRADING_DAYS // 3, 1.8), np.full(TRADING_DAYS - TRADING_DAYS // 3, 0.6)]
            returns = z * scale
            returns *= 0.20 / math.sqrt(np.sum(returns * returns) / (TRADING_DAYS * DT))

        spots = S0 * np.exp(np.r_[0.0, np.cumsum(returns)])
        realized_var = float(np.sum(returns * returns))

        for product in products:
            cumulative_actual = 0.0
            cumulative_formula = 0.0
            cumulative_residual = 0.0
            gamma_weighted_realized = 0.0
            gamma_weighted_implied = 0.0
            for i in range(TRADING_DAYS):
                tau = max(BS_INITIAL_MATURITY - i * DT, DT)
                tau_next = max(BS_INITIAL_MATURITY - (i + 1) * DT, DT)
                s_t = float(spots[i])
                s_next = float(spots[i + 1])
                ret = returns[i]
                simple_return = (s_next - s_t) / s_t
                value, delta, gamma, theta = product_value_delta_gamma_theta(product, s_t, tau, BASE_VOL)
                value_next, _, _, _ = product_value_delta_gamma_theta(product, s_next, tau_next, BASE_VOL)

                actual_pnl = -(value_next - value) + delta * (s_next - s_t)
                dollar_gamma = 0.5 * s_t * s_t * gamma
                formula_pnl = -dollar_gamma * (simple_return * simple_return - BASE_VOL * BASE_VOL * DT)
                residual = actual_pnl - formula_pnl

                cumulative_actual += actual_pnl
                cumulative_formula += formula_pnl
                cumulative_residual += residual
                gamma_weighted_realized += dollar_gamma * simple_return * simple_return
                gamma_weighted_implied += dollar_gamma * BASE_VOL * BASE_VOL * DT

                daily_rows.append(
                    {
                        "scenario": scenario,
                        "product": product,
                        "day": i,
                        "spot": s_t,
                        "return": simple_return,
                        "value": value,
                        "delta": delta,
                        "gamma": gamma,
                        "theta": theta,
                        "dollar_gamma_half": dollar_gamma,
                        "actual_pnl": actual_pnl,
                        "formula_pnl": formula_pnl,
                        "residual": residual,
                        "cumulative_actual_pnl": cumulative_actual,
                        "cumulative_formula_pnl": cumulative_formula,
                        "cumulative_residual": cumulative_residual,
                    }
                )

            pnl_var = float(np.var([r["actual_pnl"] for r in daily_rows if r["scenario"] == scenario and r["product"] == product]))
            resid_var = float(np.var([r["residual"] for r in daily_rows if r["scenario"] == scenario and r["product"] == product]))
            explanation_ratio = 1.0 - resid_var / pnl_var if pnl_var > 1e-14 else 1.0
            summary_rows.append(
                {
                    "scenario": scenario,
                    "product": product,
                    "annual_realized_variance": realized_var,
                    "annual_realized_vol": math.sqrt(realized_var),
                    "implied_variance": BASE_VOL * BASE_VOL,
                    "total_actual_pnl": cumulative_actual,
                    "total_formula_pnl": cumulative_formula,
                    "total_residual": cumulative_residual,
                    "gamma_weighted_realized_variance": gamma_weighted_realized,
                    "gamma_weighted_implied_variance": gamma_weighted_implied,
                    "explanation_ratio": explanation_ratio,
                }
            )
    return daily_rows, summary_rows


PORTFOLIO = [
    VanillaLeg("long_1y_110_call", "call", 110.0, 1.0, 1.00),
    VanillaLeg("short_1y_90_put", "put", 90.0, 1.0, -0.85),
    VanillaLeg("long_1y_90_put", "put", 90.0, 1.0, 0.35),
    VanillaLeg("long_1y_110_call_extra", "call", 110.0, 1.0, 0.35),
    VanillaLeg("long_18m_atm_call", "call", 100.0, 1.5, 0.65),
    VanillaLeg("short_6m_atm_call", "call", 100.0, 0.5, -0.80),
]


def base_smile_vol(strike: float, spot: float, maturity: float, level: float) -> float:
    tau = max(maturity, 1.0 / 12.0)
    x = math.log(strike / spot)
    skew = -0.070 / math.sqrt(tau)
    curvature = 0.18 / math.sqrt(tau)
    vol = BASE_VOL + level + skew * x + 0.5 * curvature * x * x
    return float(np.clip(vol, 0.05, 0.80))


def portfolio_value(spot: float, time: float, level: float) -> float:
    value = 0.0
    for leg in PORTFOLIO:
        tau = max(leg.maturity - time, 0.0)
        vol = base_smile_vol(leg.strike, spot, tau, level)
        value += leg.weight * bs_price(spot, leg.strike, tau, vol, leg.option_type)
    return value


def model_smile_vol(model: str, strike: float, spot: float, maturity: float, level: float) -> float:
    """Toy daily-calibrated surface used for the recalibration experiment."""
    tau = max(maturity, 1.0 / 12.0)
    x = math.log(strike / spot)
    atm = float(np.clip(BASE_VOL + level, 0.05, 0.80))
    skew = -0.070 / math.sqrt(tau)
    curvature = 0.18 / math.sqrt(tau)

    if model == "black_scholes":
        vol = atm
    elif model == "heston_sv":
        vol = atm + 0.85 * skew * x + 0.5 * 0.35 * curvature * x * x
    else:
        # LV, Bergomi 2F and admissible LSV are treated as exactly calibrated
        # to the current mock smile in this lightweight recalibration module.
        vol = atm + skew * x + 0.5 * curvature * x * x
    return float(np.clip(vol, 0.05, 0.80))


def model_portfolio_value(model: str, spot: float, time: float, level: float) -> float:
    value = 0.0
    for leg in PORTFOLIO:
        tau = max(leg.maturity - time, 0.0)
        vol = model_smile_vol(model, leg.strike, spot, tau, level)
        value += leg.weight * bs_price(spot, leg.strike, tau, vol, leg.option_type)
    return value


def finite_diff_greeks_model(model: str, spot: float, time: float, level: float) -> Dict[str, float]:
    ds = max(0.01 * spot, 0.50)
    dv = 0.0025
    dt_small = min(DT, max(1.5 - time, DT) * 0.25)

    f = lambda s, t, v: model_portfolio_value(model, s, t, v)
    value = f(spot, time, level)
    up_s = f(spot + ds, time, level)
    dn_s = f(spot - ds, time, level)
    up_v = f(spot, time, level + dv)
    dn_v = f(spot, time, level - dv)
    up_s_up_v = f(spot + ds, time, level + dv)
    up_s_dn_v = f(spot + ds, time, level - dv)
    dn_s_up_v = f(spot - ds, time, level + dv)
    dn_s_dn_v = f(spot - ds, time, level - dv)
    later = f(spot, time + dt_small, level)

    return {
        "value": value,
        "delta": (up_s - dn_s) / (2.0 * ds),
        "gamma": (up_s - 2.0 * value + dn_s) / (ds * ds),
        "vega": (up_v - dn_v) / (2.0 * dv),
        "volga": (up_v - 2.0 * value + dn_v) / (dv * dv),
        "vanna": (up_s_up_v - up_s_dn_v - dn_s_up_v + dn_s_dn_v) / (4.0 * ds * dv),
        "theta": (later - value) / dt_small,
    }


def finite_diff_greeks(spot: float, time: float, level: float) -> Dict[str, float]:
    ds = max(0.01 * spot, 0.50)
    dv = 0.0025
    dt_small = min(DT, max(1.5 - time, DT) * 0.25)

    f = portfolio_value
    value = f(spot, time, level)
    up_s = f(spot + ds, time, level)
    dn_s = f(spot - ds, time, level)
    up_v = f(spot, time, level + dv)
    dn_v = f(spot, time, level - dv)
    up_s_up_v = f(spot + ds, time, level + dv)
    up_s_dn_v = f(spot + ds, time, level - dv)
    dn_s_up_v = f(spot - ds, time, level + dv)
    dn_s_dn_v = f(spot - ds, time, level - dv)
    later = f(spot, time + dt_small, level)

    return {
        "value": value,
        "delta": (up_s - dn_s) / (2.0 * ds),
        "gamma": (up_s - 2.0 * value + dn_s) / (ds * ds),
        "vega": (up_v - dn_v) / (2.0 * dv),
        "volga": (up_v - 2.0 * value + dn_v) / (dv * dv),
        "vanna": (up_s_up_v - up_s_dn_v - dn_s_up_v + dn_s_dn_v) / (4.0 * ds * dv),
        "theta": (later - value) / dt_small,
    }


def rolling_stat(values: np.ndarray, i: int, fallback: float, window: int = 40) -> float:
    start = max(0, i - window)
    if i <= start:
        return fallback
    return float(np.sum(values[start:i]) / max(i - start, 1) * TRADING_DAYS)


def daily_recalibrated_breakevens(model: str, i: int, dx: np.ndarray, dlevel: np.ndarray, level_t: float) -> Tuple[float, float]:
    lv_cov = -0.0140 * (1.0 + 0.5 * level_t / BASE_VOL)
    lv_vov = 0.0049 * (1.0 + level_t / BASE_VOL) ** 2
    hist_cov = rolling_stat(dx * dlevel, i, -0.0110)
    hist_vov = rolling_stat(dlevel * dlevel, i, 0.0060)

    if model == "black_scholes":
        return 0.0, 0.0
    if model == "local_vol":
        return lv_cov, lv_vov
    if model == "heston_sv":
        return 0.65 * hist_cov + 0.35 * (-0.0090), 0.65 * hist_vov + 0.35 * 0.0045
    if model == "bergomi_2f":
        return 0.85 * hist_cov + 0.15 * (-0.0105), 0.85 * hist_vov + 0.15 * 0.0063
    if model in {"admissible_lsv", "nonadmissible_lsv"}:
        return 0.70 * hist_cov + 0.30 * lv_cov, 0.70 * hist_vov + 0.30 * lv_vov
    raise ValueError(model)


def theta_consistent_volvol_breakeven(
    greeks: Dict[str, float], spot: float, variance_be: float, cov_be: float, fallback_vov: float
) -> float:
    """Choose the vol-of-vol breakeven that balances the daily theta equation.

    This is a lightweight stand-in for solving the full model pricing PDE after
    daily recalibration.  It keeps the chosen spot-vol covariance fixed and uses
    the volga term to absorb the remaining theta requirement.
    """
    denom = 0.5 * greeks["volga"]
    if abs(denom) < 1e-8:
        return fallback_vov
    required = (
        -greeks["theta"]
        - 0.5 * spot * spot * greeks["gamma"] * variance_be
        - greeks["vanna"] * spot * cov_be
    ) / denom
    if not np.isfinite(required):
        return fallback_vov
    return float(np.clip(required, 0.0, 0.08))


def generate_surface_path() -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(SEED)
    n_steps = TRADING_DAYS
    z_spot = rng.standard_normal(n_steps)
    z_vol_ind = rng.standard_normal(n_steps)
    z_shadow_ind = rng.standard_normal(n_steps)

    spot_vol = 0.20
    vol_of_vol = 0.080
    rho_spot_vol = -0.75
    shadow_vol = 0.10
    rho_spot_shadow = -0.35

    dx = (-0.5 * spot_vol * spot_vol * DT) + spot_vol * math.sqrt(DT) * z_spot
    dlevel = vol_of_vol * math.sqrt(DT) * (
        rho_spot_vol * z_spot + math.sqrt(1.0 - rho_spot_vol * rho_spot_vol) * z_vol_ind
    )
    dshadow = shadow_vol * math.sqrt(DT) * (
        rho_spot_shadow * z_spot + math.sqrt(1.0 - rho_spot_shadow * rho_spot_shadow) * z_shadow_ind
    )

    level = np.r_[0.0, np.cumsum(dlevel)]
    shadow = np.r_[0.0, np.cumsum(dshadow)]
    spot = S0 * np.exp(np.r_[0.0, np.cumsum(dx)])
    return {
        "dx": dx,
        "dlevel": dlevel,
        "dshadow": dshadow,
        "spot": spot,
        "level": level,
        "shadow": shadow,
    }


@dataclass(frozen=True)
class AccountingKernel:
    name: str
    label: str
    variance_be: float
    spot_vol_cov_be: float
    volvol_var_be: float
    leakage_sensitivity: float
    leakage_var_be: float
    note: str


def build_kernels(realized: Dict[str, float]) -> List[AccountingKernel]:
    return [
        AccountingKernel(
            "black_scholes",
            "BS zero-breakeven",
            BASE_VOL * BASE_VOL,
            0.0,
            0.0,
            0.0,
            0.0,
            "constant volatility; no surface-dynamic breakeven",
        ),
        AccountingKernel(
            "local_vol",
            "Local volatility",
            BASE_VOL * BASE_VOL,
            -0.0140,
            0.0049,
            0.0,
            0.0,
            "Vanna and Volga breakevens locked by the initial smile slope",
        ),
        AccountingKernel(
            "heston_sv",
            "Heston-style SV",
            BASE_VOL * BASE_VOL,
            -0.0090,
            0.0045,
            0.0,
            0.0,
            "one stochastic variance factor with lower vol-of-vol breakeven",
        ),
        AccountingKernel(
            "bergomi_2f",
            "Bergomi 2F",
            BASE_VOL * BASE_VOL,
            -0.0105,
            0.0063,
            0.0,
            0.0,
            "two forward-variance factors close to the mock path covariance",
        ),
        AccountingKernel(
            "admissible_lsv",
            "Admissible LSV",
            BASE_VOL * BASE_VOL,
            -0.0100,
            0.0065,
            0.0,
            0.0,
            "local smile plus admissible stochastic volatility; no leakage",
        ),
        AccountingKernel(
            "nonadmissible_lsv",
            "Non-admissible LSV diagnostic",
            BASE_VOL * BASE_VOL,
            -0.0100,
            0.0065,
            5.0,
            realized["shadow_var"],
            "same surface breakevens, plus a non-tradable state leakage diagnostic",
        ),
    ]


def run_surface_accounting_experiment() -> Tuple[
    List[Dict[str, object]], List[Dict[str, object]], List[Dict[str, object]], Dict[str, np.ndarray]
]:
    path = generate_surface_path()
    dx = path["dx"]
    dlevel = path["dlevel"]
    dshadow = path["dshadow"]
    spot = path["spot"]
    level = path["level"]

    realized = {
        "realized_variance": float(np.sum(dx * dx)),
        "spot_vol_cov": float(np.sum(dx * dlevel)),
        "volvol_var": float(np.sum(dlevel * dlevel)),
        "shadow_var": float(np.sum(dshadow * dshadow)),
    }
    kernels = build_kernels(realized)

    daily_rows: List[Dict[str, object]] = []
    summary_rows: List[Dict[str, object]] = []
    kernel_rows: List[Dict[str, object]] = []

    for kernel in kernels:
        cumulative_actual = 0.0
        cumulative_explained = 0.0
        cumulative_residual = 0.0
        cumulative_standard_residual = 0.0
        cumulative_gamma_theta = 0.0
        cumulative_vega = 0.0
        cumulative_vanna = 0.0
        cumulative_volga = 0.0
        cumulative_leakage = 0.0
        pnl_values = []
        residual_values = []
        standard_residual_values = []

        for i in range(TRADING_DAYS):
            time = i * DT
            s_t = float(spot[i])
            s_next = float(spot[i + 1])
            level_t = float(level[i])
            level_next = float(level[i + 1])
            d_s = s_next - s_t
            d_x = float(dx[i])
            d_v = float(dlevel[i])
            d_lam = float(dshadow[i])

            g = finite_diff_greeks(s_t, time, level_t)
            value_next = portfolio_value(s_next, time + DT, level_next)

            gamma_theta = -0.5 * s_t * s_t * g["gamma"] * (d_x * d_x - kernel.variance_be * DT)
            vega_pnl = -g["vega"] * d_v
            vanna_pnl = -g["vanna"] * (d_s * d_v - s_t * kernel.spot_vol_cov_be * DT)
            volga_pnl = -0.5 * g["volga"] * (d_v * d_v - kernel.volvol_var_be * DT)
            leakage_pnl = -kernel.leakage_sensitivity * (
                d_lam - 0.0 * DT
            ) - 0.5 * kernel.leakage_sensitivity * (d_lam * d_lam - kernel.leakage_var_be * DT)

            base_actual_pnl = -(value_next - g["value"]) + g["delta"] * d_s
            actual_pnl = base_actual_pnl + leakage_pnl
            standard_explained = gamma_theta + vega_pnl + vanna_pnl + volga_pnl
            explained = standard_explained + leakage_pnl
            residual = actual_pnl - explained
            standard_residual = actual_pnl - standard_explained

            cumulative_actual += actual_pnl
            cumulative_explained += explained
            cumulative_residual += residual
            cumulative_standard_residual += standard_residual
            cumulative_gamma_theta += gamma_theta
            cumulative_vega += vega_pnl
            cumulative_vanna += vanna_pnl
            cumulative_volga += volga_pnl
            cumulative_leakage += leakage_pnl
            pnl_values.append(actual_pnl)
            residual_values.append(residual)
            standard_residual_values.append(standard_residual)

            daily_rows.append(
                {
                    "model": kernel.name,
                    "model_label": kernel.label,
                    "day": i,
                    "spot": s_t,
                    "level": level_t,
                    "dx": d_x,
                    "dlevel": d_v,
                    "dshadow": d_lam,
                    "portfolio_value": g["value"],
                    "delta": g["delta"],
                    "gamma": g["gamma"],
                    "vega": g["vega"],
                    "vanna": g["vanna"],
                    "volga": g["volga"],
                    "theta": g["theta"],
                    "actual_pnl": actual_pnl,
                    "gamma_theta_pnl": gamma_theta,
                    "vega_pnl": vega_pnl,
                    "vanna_pnl": vanna_pnl,
                    "volga_pnl": volga_pnl,
                    "leakage_pnl": leakage_pnl,
                    "explained_pnl": explained,
                    "standard_explained_pnl": standard_explained,
                    "residual": residual,
                    "standard_residual": standard_residual,
                    "cumulative_actual_pnl": cumulative_actual,
                    "cumulative_explained_pnl": cumulative_explained,
                    "cumulative_gamma_theta_pnl": cumulative_gamma_theta,
                    "cumulative_vega_pnl": cumulative_vega,
                    "cumulative_vanna_pnl": cumulative_vanna,
                    "cumulative_volga_pnl": cumulative_volga,
                    "cumulative_residual": cumulative_residual,
                    "cumulative_standard_residual": cumulative_standard_residual,
                    "cumulative_leakage_pnl": cumulative_leakage,
                }
            )

        pnl_var = float(np.var(pnl_values))
        residual_var = float(np.var(residual_values))
        std_residual_var = float(np.var(standard_residual_values))
        explanation_ratio = 1.0 - residual_var / pnl_var if pnl_var > 1e-14 else 1.0
        standard_explanation_ratio = 1.0 - std_residual_var / pnl_var if pnl_var > 1e-14 else 1.0
        summary_rows.append(
            {
                "model": kernel.name,
                "model_label": kernel.label,
                "total_actual_pnl": cumulative_actual,
                "total_explained_pnl": cumulative_explained,
                "total_standard_explained_pnl": cumulative_actual - cumulative_standard_residual,
                "total_gamma_theta_pnl": sum(
                    r["gamma_theta_pnl"] for r in daily_rows if r["model"] == kernel.name
                ),
                "total_vega_pnl": sum(r["vega_pnl"] for r in daily_rows if r["model"] == kernel.name),
                "total_vanna_pnl": sum(r["vanna_pnl"] for r in daily_rows if r["model"] == kernel.name),
                "total_volga_pnl": sum(r["volga_pnl"] for r in daily_rows if r["model"] == kernel.name),
                "total_leakage_pnl": cumulative_leakage,
                "total_residual": cumulative_residual,
                "total_standard_residual": cumulative_standard_residual,
                "explanation_ratio": explanation_ratio,
                "standard_explanation_ratio": standard_explanation_ratio,
            }
        )

        kernel_rows.append(
            {
                "model": kernel.name,
                "model_label": kernel.label,
                "variance_breakeven": kernel.variance_be,
                "spot_vol_covariance_breakeven": kernel.spot_vol_cov_be,
                "vol_of_vol_variance_breakeven": kernel.volvol_var_be,
                "leakage_sensitivity": kernel.leakage_sensitivity,
                "leakage_variance_breakeven": kernel.leakage_var_be,
                "realized_variance": realized["realized_variance"],
                "realized_spot_vol_covariance": realized["spot_vol_cov"],
                "realized_vol_of_vol_variance": realized["volvol_var"],
                "realized_shadow_variance": realized["shadow_var"],
                "note": kernel.note,
            }
        )

    return daily_rows, summary_rows, kernel_rows, path


def run_daily_recalibrated_experiment(path: Dict[str, np.ndarray]) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    dx = path["dx"]
    dlevel = path["dlevel"]
    dshadow = path["dshadow"]
    spot = path["spot"]
    level = path["level"]

    models = [
        ("black_scholes", "BS ATM-only"),
        ("local_vol", "Daily recalibrated LV"),
        ("heston_sv", "Daily recalibrated Heston-style SV"),
        ("bergomi_2f", "Daily recalibrated Bergomi 2F"),
        ("admissible_lsv", "Daily recalibrated admissible LSV"),
        ("nonadmissible_lsv", "Daily recalibrated non-admissible LSV"),
    ]

    daily_rows: List[Dict[str, object]] = []
    summary_rows: List[Dict[str, object]] = []
    market_values = [portfolio_value(float(spot[i]), i * DT, float(level[i])) for i in range(TRADING_DAYS + 1)]

    for model, label in models:
        cumulative_market_actual = 0.0
        cumulative_model_actual = 0.0
        cumulative_explained = 0.0
        cumulative_model_residual = 0.0
        cumulative_market_residual = 0.0
        cumulative_gamma_theta = 0.0
        cumulative_vega = 0.0
        cumulative_vanna = 0.0
        cumulative_volga = 0.0
        cumulative_leakage = 0.0
        model_residuals = []
        market_residuals = []
        market_pnls = []

        for i in range(TRADING_DAYS):
            time = i * DT
            s_t = float(spot[i])
            s_next = float(spot[i + 1])
            level_t = float(level[i])
            level_next = float(level[i + 1])
            d_s = s_next - s_t
            d_x = float(dx[i])
            d_v = float(dlevel[i])
            d_lam = float(dshadow[i])

            g = finite_diff_greeks_model(model, s_t, time, level_t)
            model_value_next = model_portfolio_value(model, s_next, time + DT, level_next)
            model_base_actual = -(model_value_next - g["value"]) + g["delta"] * d_s
            market_actual = -(market_values[i + 1] - market_values[i]) + g["delta"] * d_s

            cov_be, vov_be = daily_recalibrated_breakevens(model, i, dx, dlevel, level_t)
            if model != "black_scholes":
                vov_be = theta_consistent_volvol_breakeven(g, s_t, BASE_VOL * BASE_VOL, cov_be, vov_be)
            leakage_sensitivity = 5.0 if model == "nonadmissible_lsv" else 0.0
            leakage_var_be = rolling_stat(dshadow * dshadow, i, 0.0100)
            leakage_pnl = -leakage_sensitivity * d_lam - 0.5 * leakage_sensitivity * (
                d_lam * d_lam - leakage_var_be * DT
            )

            model_actual = model_base_actual + leakage_pnl
            gamma_theta = -0.5 * s_t * s_t * g["gamma"] * (d_x * d_x - BASE_VOL * BASE_VOL * DT)
            vega_pnl = -g["vega"] * d_v
            vanna_pnl = -g["vanna"] * (d_s * d_v - s_t * cov_be * DT)
            volga_pnl = -0.5 * g["volga"] * (d_v * d_v - vov_be * DT)
            explained = gamma_theta + vega_pnl + vanna_pnl + volga_pnl + leakage_pnl
            model_residual = model_actual - explained
            market_residual = market_actual - explained

            cumulative_market_actual += market_actual
            cumulative_model_actual += model_actual
            cumulative_explained += explained
            cumulative_model_residual += model_residual
            cumulative_market_residual += market_residual
            cumulative_gamma_theta += gamma_theta
            cumulative_vega += vega_pnl
            cumulative_vanna += vanna_pnl
            cumulative_volga += volga_pnl
            cumulative_leakage += leakage_pnl
            model_residuals.append(model_residual)
            market_residuals.append(market_residual)
            market_pnls.append(market_actual)

            daily_rows.append(
                {
                    "model": model,
                    "model_label": label,
                    "day": i,
                    "spot": s_t,
                    "level": level_t,
                    "cov_breakeven": cov_be,
                    "volvol_breakeven": vov_be,
                    "market_actual_pnl": market_actual,
                    "model_actual_pnl": model_actual,
                    "gamma_theta_pnl": gamma_theta,
                    "vega_pnl": vega_pnl,
                    "vanna_pnl": vanna_pnl,
                    "volga_pnl": volga_pnl,
                    "leakage_pnl": leakage_pnl,
                    "explained_pnl": explained,
                    "model_residual": model_residual,
                    "market_residual": market_residual,
                    "delta": g["delta"],
                    "gamma": g["gamma"],
                    "vega": g["vega"],
                    "vanna": g["vanna"],
                    "volga": g["volga"],
                    "cumulative_market_actual_pnl": cumulative_market_actual,
                    "cumulative_model_actual_pnl": cumulative_model_actual,
                    "cumulative_explained_pnl": cumulative_explained,
                    "cumulative_gamma_theta_pnl": cumulative_gamma_theta,
                    "cumulative_vega_pnl": cumulative_vega,
                    "cumulative_vanna_pnl": cumulative_vanna,
                    "cumulative_volga_pnl": cumulative_volga,
                    "cumulative_leakage_pnl": cumulative_leakage,
                    "cumulative_model_residual": cumulative_model_residual,
                    "cumulative_market_residual": cumulative_market_residual,
                }
            )

        market_var = float(np.var(market_pnls))
        model_resid_var = float(np.var(model_residuals))
        market_resid_var = float(np.var(market_residuals))
        summary_rows.append(
            {
                "model": model,
                "model_label": label,
                "total_market_actual_pnl": cumulative_market_actual,
                "total_model_actual_pnl": cumulative_model_actual,
                "total_explained_pnl": cumulative_explained,
                "total_gamma_theta_pnl": cumulative_gamma_theta,
                "total_vega_pnl": cumulative_vega,
                "total_vanna_pnl": cumulative_vanna,
                "total_volga_pnl": cumulative_volga,
                "total_leakage_pnl": cumulative_leakage,
                "total_model_residual": cumulative_model_residual,
                "total_market_residual": cumulative_market_residual,
                "model_internal_explanation_ratio": 1.0 - model_resid_var / market_var if market_var > 1e-14 else 1.0,
                "market_explanation_ratio": 1.0 - market_resid_var / market_var if market_var > 1e-14 else 1.0,
            }
        )

    return daily_rows, summary_rows


def plot_bs_results(daily_rows: List[Dict[str, object]], summary_rows: List[Dict[str, object]]) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    scenario = "high_realized_vol"
    product = "straddle_atm"
    rows = [r for r in daily_rows if r["scenario"] == scenario and r["product"] == product]
    days = np.array([r["day"] for r in rows])

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    axes[0].plot(days, [r["spot"] for r in rows], color="#1f77b4", lw=1.5)
    axes[0].set_ylabel("Spot")
    axes[0].set_title("BS delta hedge: high realized volatility straddle")

    axes[1].plot(days, [r["dollar_gamma_half"] for r in rows], color="#2ca02c", lw=1.2)
    axes[1].set_ylabel("0.5 S^2 Gamma")

    axes[2].plot(days, [r["cumulative_actual_pnl"] for r in rows], label="actual PnL", lw=1.4)
    axes[2].plot(days, [r["cumulative_formula_pnl"] for r in rows], label="formula PnL", lw=1.4, ls="--")
    axes[2].plot(days, [r["cumulative_residual"] for r in rows], label="residual", lw=1.2, ls=":")
    axes[2].set_ylabel("Cumulative PnL")
    axes[2].set_xlabel("Day")
    axes[2].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES / "bs_delta_accounting_dashboard.png", dpi=180)
    plt.close(fig)

    straddle_summary = [r for r in summary_rows if r["product"] == "straddle_atm"]
    labels = [r["scenario"].replace("_", "\n") for r in straddle_summary]
    x = np.arange(len(labels))
    actual = [r["total_actual_pnl"] for r in straddle_summary]
    formula = [r["total_formula_pnl"] for r in straddle_summary]
    residual = [r["total_residual"] for r in straddle_summary]

    fig, ax = plt.subplots(figsize=(10, 5))
    width = 0.25
    ax.bar(x - width, actual, width, label="actual")
    ax.bar(x, formula, width, label="formula")
    ax.bar(x + width, residual, width, label="residual")
    ax.axhline(0.0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Total PnL")
    ax.set_title("BS accounting totals for the ATM straddle")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES / "bs_delta_summary_bars.png", dpi=180)
    plt.close(fig)


def plot_surface_results(
    daily_rows: List[Dict[str, object]],
    summary_rows: List[Dict[str, object]],
    path: Dict[str, np.ndarray],
) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)

    model = "bergomi_2f"
    rows = [r for r in daily_rows if r["model"] == model]
    days = np.array([r["day"] for r in rows])
    fig, axes = plt.subplots(4, 1, figsize=(11, 11), sharex=True)
    axes[0].plot(days, [r["spot"] for r in rows], lw=1.4)
    axes[0].set_ylabel("Spot")
    axes[0].set_title("Surface-dynamic PnL breakdown dashboard: Bergomi 2F convention")
    axes[1].plot(days, [100.0 * r["level"] for r in rows], color="#9467bd", lw=1.2)
    axes[1].set_ylabel("ATM level move\n(vol pts)")
    axes[2].plot(days, [r["vega"] for r in rows], label="Vega", lw=1.1)
    axes[2].plot(days, [r["vanna"] for r in rows], label="Vanna", lw=1.1)
    axes[2].plot(days, [r["volga"] for r in rows], label="Volga", lw=1.1)
    axes[2].set_ylabel("Greeks")
    axes[2].legend(frameon=False, ncol=3)
    axes[3].plot(days, [r["cumulative_actual_pnl"] for r in rows], label="actual", lw=1.4)
    axes[3].plot(days, [r["cumulative_explained_pnl"] for r in rows], label="explained", lw=1.4, ls="--")
    axes[3].plot(days, [r["cumulative_residual"] for r in rows], label="residual", lw=1.2, ls=":")
    axes[3].set_ylabel("Cumulative PnL")
    axes[3].set_xlabel("Day")
    axes[3].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES / "surface_accounting_dashboard_bergomi2f.png", dpi=180)
    plt.close(fig)

    selected_models = [
        ("black_scholes", "BS zero-breakeven"),
        ("local_vol", "Local volatility"),
        ("heston_sv", "Heston-style SV"),
        ("bergomi_2f", "Bergomi 2F"),
        ("admissible_lsv", "Admissible LSV"),
        ("nonadmissible_lsv", "Non-admissible LSV"),
    ]
    fig, axes = plt.subplots(3, 2, figsize=(14, 12), sharex=True)
    component_specs = [
        ("cumulative_gamma_theta_pnl", "Gamma/Theta", "#4c78a8"),
        ("cumulative_vega_pnl", "Vega", "#f58518"),
        ("cumulative_vanna_pnl", "Vanna", "#54a24b"),
        ("cumulative_volga_pnl", "Volga", "#b279a2"),
        ("cumulative_leakage_pnl", "Leakage", "#e45756"),
        ("cumulative_residual", "Residual", "#79706e"),
    ]
    for ax, (model_name, label) in zip(axes.ravel(), selected_models):
        model_rows = [r for r in daily_rows if r["model"] == model_name]
        model_days = np.array([r["day"] for r in model_rows])
        ax.plot(model_days, [r["cumulative_actual_pnl"] for r in model_rows], color="black", lw=1.5, label="Actual")
        ax.plot(
            model_days,
            [r["cumulative_explained_pnl"] for r in model_rows],
            color="black",
            lw=1.1,
            ls="--",
            label="Explained",
        )
        for key, comp_label, color in component_specs:
            values = np.array([r[key] for r in model_rows], dtype=float)
            if np.max(np.abs(values)) > 1e-10:
                ax.plot(model_days, values, lw=0.95, color=color, label=comp_label)
        ax.axhline(0.0, color="black", lw=0.6)
        ax.set_title(label)
        ax.set_ylabel("Cumulative PnL")
    axes[-1, 0].set_xlabel("Day")
    axes[-1, 1].set_xlabel("Day")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
    fig.suptitle("Cumulative Greek PnL components under different breakdown conventions", y=0.995)
    fig.tight_layout(rect=[0.0, 0.05, 1.0, 0.97])
    fig.savefig(FIGURES / "greek_pnl_components_by_model.png", dpi=180)
    plt.close(fig)

    model_labels = [r["model_label"] for r in summary_rows]
    components = ["total_gamma_theta_pnl", "total_vega_pnl", "total_vanna_pnl", "total_volga_pnl", "total_leakage_pnl", "total_residual"]
    component_labels = ["Gamma/Theta", "Vega", "Vanna", "Volga", "Leakage", "Residual"]
    colors = ["#4c78a8", "#f58518", "#54a24b", "#b279a2", "#e45756", "#79706e"]
    x = np.arange(len(model_labels))
    bottom_pos = np.zeros(len(model_labels))
    bottom_neg = np.zeros(len(model_labels))
    fig, ax = plt.subplots(figsize=(12, 6))
    for comp, label, color in zip(components, component_labels, colors):
        vals = np.array([r[comp] for r in summary_rows], dtype=float)
        bottoms = np.where(vals >= 0.0, bottom_pos, bottom_neg)
        ax.bar(x, vals, bottom=bottoms, label=label, color=color)
        bottom_pos += np.where(vals >= 0.0, vals, 0.0)
        bottom_neg += np.where(vals < 0.0, vals, 0.0)
    ax.axhline(0.0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(model_labels, rotation=20, ha="right")
    ax.set_ylabel("Total PnL contribution")
    ax.set_title("PnL breakdown conventions applied to the same vanilla portfolio path")
    ax.legend(frameon=False, ncol=3)
    fig.tight_layout()
    fig.savefig(FIGURES / "model_accounting_waterfall.png", dpi=180)
    plt.close(fig)

    # Daily local PnL map around the initial state, using finite-difference Greeks.
    g0 = finite_diff_greeks(S0, 0.0, 0.0)
    dx_grid = np.linspace(-0.035, 0.035, 81)
    dv_grid = np.linspace(-0.015, 0.015, 81)
    z = np.zeros((len(dv_grid), len(dx_grid)))
    variance_be = BASE_VOL * BASE_VOL
    cov_be = -0.0105
    volvol_be = 0.0063
    for i, dv in enumerate(dv_grid):
        for j, dx in enumerate(dx_grid):
            d_s = S0 * dx
            z[i, j] = (
                -0.5 * S0 * S0 * g0["gamma"] * (dx * dx - variance_be * DT)
                - g0["vega"] * dv
                - g0["vanna"] * (d_s * dv - S0 * cov_be * DT)
                - 0.5 * g0["volga"] * (dv * dv - volvol_be * DT)
            )
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(
        z,
        origin="lower",
        extent=[dx_grid[0], dx_grid[-1], dv_grid[0] * 100.0, dv_grid[-1] * 100.0],
        aspect="auto",
        cmap="RdBu_r",
    )
    ax.axvline(0.0, color="black", lw=0.7)
    ax.axhline(0.0, color="black", lw=0.7)
    ax.set_xlabel("Daily log return")
    ax.set_ylabel("Daily ATM level move (vol pts)")
    ax.set_title("Local second-order PnL map for the vanilla portfolio")
    fig.colorbar(im, ax=ax, label="Approximate short delta-hedged PnL")
    fig.tight_layout()
    fig.savefig(FIGURES / "daily_pnl_kernel_heatmap.png", dpi=180)
    plt.close(fig)

    nonadmissible = [r for r in daily_rows if r["model"] == "nonadmissible_lsv"]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(days, [r["cumulative_standard_residual"] for r in nonadmissible], label="residual without leakage term", lw=1.3)
    ax.plot(days, [r["cumulative_residual"] for r in nonadmissible], label="residual after leakage term", lw=1.3)
    ax.plot(days, [r["cumulative_leakage_pnl"] for r in nonadmissible], label="cumulative leakage", lw=1.1, ls=":")
    ax.axhline(0.0, color="black", lw=0.8)
    ax.set_xlabel("Day")
    ax.set_ylabel("Cumulative PnL")
    ax.set_title("Non-admissible LSV diagnostic: shadow-state leakage")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES / "lsv_leakage_diagnostic.png", dpi=180)
    plt.close(fig)


def plot_daily_recalibrated_results(
    daily_rows: List[Dict[str, object]], summary_rows: List[Dict[str, object]]
) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    labels = [r["model_label"] for r in summary_rows]
    x = np.arange(len(labels))
    market_residual = np.array([r["total_market_residual"] for r in summary_rows], dtype=float)
    model_residual = np.array([r["total_model_residual"] for r in summary_rows], dtype=float)
    market_r2 = np.array([r["market_explanation_ratio"] for r in summary_rows], dtype=float)
    model_r2 = np.array([r["model_internal_explanation_ratio"] for r in summary_rows], dtype=float)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    width = 0.35
    axes[0].bar(x - width / 2, market_residual, width, label="Residual vs mock market PnL")
    axes[0].bar(x + width / 2, model_residual, width, label="Residual vs model PnL")
    axes[0].axhline(0.0, color="black", lw=0.8)
    axes[0].set_ylabel("Total residual")
    axes[0].set_title("Daily recalibration: residual comparison")
    axes[0].legend(frameon=False)
    axes[1].bar(x - width / 2, market_r2, width, label="Market PnL explanation ratio")
    axes[1].bar(x + width / 2, model_r2, width, label="Model internal explanation ratio")
    axes[1].axhline(0.0, color="black", lw=0.8)
    axes[1].set_ylabel("Explanation ratio")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=20, ha="right")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES / "daily_recalibration_residual_comparison.png", dpi=180)
    plt.close(fig)

    selected_models = [
        ("black_scholes", "BS ATM-only"),
        ("local_vol", "Daily recalibrated LV"),
        ("heston_sv", "Daily recalibrated Heston-style SV"),
        ("bergomi_2f", "Daily recalibrated Bergomi 2F"),
        ("admissible_lsv", "Daily recalibrated admissible LSV"),
        ("nonadmissible_lsv", "Daily recalibrated non-admissible LSV"),
    ]
    fig, axes = plt.subplots(3, 2, figsize=(14, 12), sharex=True)
    component_specs = [
        ("cumulative_gamma_theta_pnl", "Gamma/Theta", "#4c78a8"),
        ("cumulative_vega_pnl", "Vega", "#f58518"),
        ("cumulative_vanna_pnl", "Vanna", "#54a24b"),
        ("cumulative_volga_pnl", "Volga", "#b279a2"),
        ("cumulative_leakage_pnl", "Leakage", "#e45756"),
        ("cumulative_market_residual", "Market residual", "#79706e"),
    ]
    for ax, (model_name, label) in zip(axes.ravel(), selected_models):
        model_rows = [r for r in daily_rows if r["model"] == model_name]
        days = np.array([r["day"] for r in model_rows])
        ax.plot(days, [r["cumulative_market_actual_pnl"] for r in model_rows], color="black", lw=1.5, label="Market actual")
        ax.plot(days, [r["cumulative_explained_pnl"] for r in model_rows], color="black", lw=1.1, ls="--", label="Explained")
        for key, comp_label, color in component_specs:
            values = np.array([r[key] for r in model_rows], dtype=float)
            if np.max(np.abs(values)) > 1e-10:
                ax.plot(days, values, lw=0.95, color=color, label=comp_label)
        ax.axhline(0.0, color="black", lw=0.6)
        ax.set_title(label)
        ax.set_ylabel("Cumulative PnL")
    axes[-1, 0].set_xlabel("Day")
    axes[-1, 1].set_xlabel("Day")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
    fig.suptitle("Daily recalibrated cumulative Greek PnL components", y=0.995)
    fig.tight_layout(rect=[0.0, 0.05, 1.0, 0.97])
    fig.savefig(FIGURES / "daily_recalibrated_greek_pnl_components.png", dpi=180)
    plt.close(fig)


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    bs_daily, bs_summary = run_bs_delta_experiment()
    surface_daily, surface_summary, kernel_rows, surface_path = run_surface_accounting_experiment()
    recal_daily, recal_summary = run_daily_recalibrated_experiment(surface_path)

    write_csv(RESULTS / "bs_delta_pnl_daily.csv", bs_daily)
    write_csv(RESULTS / "bs_delta_pnl_summary.csv", bs_summary)
    write_csv(RESULTS / "surface_accounting_daily.csv", surface_daily)
    write_csv(RESULTS / "surface_accounting_summary.csv", surface_summary)
    write_csv(RESULTS / "model_breakevens.csv", kernel_rows)
    write_csv(RESULTS / "daily_recalibrated_accounting_daily.csv", recal_daily)
    write_csv(RESULTS / "daily_recalibrated_accounting_summary.csv", recal_summary)

    plot_bs_results(bs_daily, bs_summary)
    plot_surface_results(surface_daily, surface_summary, surface_path)
    plot_daily_recalibrated_results(recal_daily, recal_summary)

    print("Generated mock-data vanilla PnL accounting experiment outputs.")
    print(f"Results: {RESULTS}")
    print(f"Figures: {FIGURES}")


if __name__ == "__main__":
    main()
