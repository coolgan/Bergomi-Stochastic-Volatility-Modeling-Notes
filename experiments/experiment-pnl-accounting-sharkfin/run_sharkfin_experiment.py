"""
Mock-data experiment for bullish single shark fin note PnL accounting.

Extends the vanilla PnL accounting experiment to a single shark fin payoff,
with two key additions relative to the vanilla case:

1. Bucket vega: the implied volatility surface is parameterised by two
   independent stochastic factors (level and skew), and vega is computed
   separately for each (strike, maturity) bucket rather than as a single
   ATM-level sensitivity.

2. Barrier event window: the note locks a fixed coupon when spot crosses the
   upper barrier. After knockout the risk is closed; the experiment keeps
   recording zero-risk rows so the 20-day window around the event can be
   plotted.

The data are deliberately synthetic.  The goal is to make Bergomi's accounting
formulas visible for a path-dependent product, not to calibrate a production
pricing model.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

S0 = 100.0
BARRIER = 115.0          # upper knock-out barrier (15% OTM)
STRIKE = 100.0           # ATM call strike
MATURITY = 1.0           # 1-year option
NOTIONAL = 100.0
FLOOR_COUPON = 0.005     # no-touch minimum return on notional
KO_COUPON = 0.020        # coupon locked if the upper barrier is touched
PARTICIPATION = 1.00     # participation in positive terminal return if no touch
FLOOR_PAYOFF = NOTIONAL * FLOOR_COUPON
KO_PAYOFF = NOTIONAL * KO_COUPON
RATE = 0.0
DIVIDEND = 0.0
BASE_VOL = 0.20
TRADING_DAYS = 252
DT = 1.0 / TRADING_DAYS
SEED = 20260614

# Bucket grid: (strike_ratio, maturity) pairs
# strike_ratio is K/S0; bucket vol is computed at these points
STRIKE_RATIOS = [0.80, 0.90, 1.00, 1.10, 1.20]
MATURITIES_BUCKET = [3 / 12, 6 / 12, 1.0]   # 3M, 6M, 1Y


# ---------------------------------------------------------------------------
# Black-Scholes helpers
# ---------------------------------------------------------------------------

def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_price(S: float, K: float, T: float, vol: float, opt: str = "call") -> float:
    if T <= 0.0:
        return max(S - K, 0.0) if opt == "call" else max(K - S, 0.0)
    vol = max(vol, 1e-8)
    d1 = (math.log(S / K) + 0.5 * vol * vol * T) / (vol * math.sqrt(T))
    d2 = d1 - vol * math.sqrt(T)
    if opt == "call":
        return S * _norm_cdf(d1) - K * _norm_cdf(d2)
    return K * _norm_cdf(-d2) - S * _norm_cdf(-d1)


def bs_delta(S: float, K: float, T: float, vol: float, opt: str = "call") -> float:
    if T <= 0.0:
        return (1.0 if S > K else 0.0) if opt == "call" else (-1.0 if S < K else 0.0)
    vol = max(vol, 1e-8)
    d1 = (math.log(S / K) + 0.5 * vol * vol * T) / (vol * math.sqrt(T))
    return _norm_cdf(d1) if opt == "call" else _norm_cdf(d1) - 1.0


def bs_gamma(S: float, K: float, T: float, vol: float) -> float:
    if T <= 0.0:
        return 0.0
    vol = max(vol, 1e-8)
    d1 = (math.log(S / K) + 0.5 * vol * vol * T) / (vol * math.sqrt(T))
    return _norm_pdf(d1) / (S * vol * math.sqrt(T))


# ---------------------------------------------------------------------------
# Bullish single shark fin payoff and PDE pricer
# ---------------------------------------------------------------------------

def sharkfin_terminal_payoff(S: float) -> float:
    """No-touch terminal coupon, excluding principal repayment."""
    return FLOOR_PAYOFF + NOTIONAL * PARTICIPATION * max(S / S0 - 1.0, 0.0)


def _solve_tridiagonal(lower: np.ndarray, diag: np.ndarray, upper: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Thomas algorithm for a tridiagonal linear system."""
    n = len(diag)
    c = np.empty(n - 1, dtype=float)
    d = np.empty(n, dtype=float)
    c[0] = upper[0] / diag[0]
    d[0] = rhs[0] / diag[0]
    for i in range(1, n - 1):
        denom = diag[i] - lower[i - 1] * c[i - 1]
        c[i] = upper[i] / denom
        d[i] = (rhs[i] - lower[i - 1] * d[i - 1]) / denom
    denom = diag[-1] - lower[-1] * c[-1]
    d[-1] = (rhs[-1] - lower[-1] * d[-2]) / denom
    x = np.empty(n, dtype=float)
    x[-1] = d[-1]
    for i in range(n - 2, -1, -1):
        x[i] = d[i] - c[i] * x[i + 1]
    return x


@lru_cache(maxsize=100_000)
def _sharkfin_pde_price_cached(S_key: float, T_key: float, vol_key: float) -> float:
    S = float(S_key)
    T = float(T_key)
    vol = max(float(vol_key), 1e-6)
    if T <= 0.0:
        return KO_PAYOFF if S >= BARRIER else sharkfin_terminal_payoff(S)
    if S >= BARRIER:
        return KO_PAYOFF

    n_space = 180
    n_time = max(60, int(math.ceil(180 * T)))
    s_grid = np.linspace(0.0, BARRIER, n_space + 1)
    ds = s_grid[1] - s_grid[0]
    dt = T / n_time

    values = np.array([sharkfin_terminal_payoff(s) for s in s_grid], dtype=float)
    values[-1] = KO_PAYOFF
    values[0] = FLOOR_PAYOFF

    idx = np.arange(1, n_space, dtype=float)
    s_inner = idx * ds
    drift = (RATE - DIVIDEND) * s_inner
    diffusion = 0.5 * vol * vol * s_inner * s_inner

    a = diffusion / (ds * ds) - drift / (2.0 * ds)
    b = -2.0 * diffusion / (ds * ds) - RATE
    c = diffusion / (ds * ds) + drift / (2.0 * ds)

    lower = -dt * a[1:]
    diag = 1.0 - dt * b
    upper = -dt * c[:-1]
    lower_boundary = FLOOR_PAYOFF
    upper_boundary = KO_PAYOFF

    for _ in range(n_time):
        rhs = values[1:-1].copy()
        rhs[0] += dt * a[0] * lower_boundary
        rhs[-1] += dt * c[-1] * upper_boundary
        values[1:-1] = _solve_tridiagonal(lower, diag, upper, rhs)
        values[0] = lower_boundary
        values[-1] = upper_boundary

    return float(np.interp(S, s_grid, values))


def sharkfin_price(S: float, T: float, vol: float) -> float:
    """PDE price for the coupon leg of a bullish single shark fin note."""
    return _sharkfin_pde_price_cached(round(S, 6), round(max(T, 0.0), 6), round(float(vol), 6))


# ---------------------------------------------------------------------------
# Two-factor mock smile
# level and skew are independent stochastic factors.
# vol(K, T; S, level, skew_factor) = BASE_VOL + level
#                                   + (base_skew_coeff + skew_factor)/sqrt(tau) * x
#                                   + 0.5 * curvature_coeff/sqrt(tau) * x^2
# where x = log(K/S).
# ---------------------------------------------------------------------------

BASE_SKEW_COEFF = -0.070   # fixed component of skew coefficient
CURVATURE_COEFF = 0.18


def smile_vol(
    K: float,
    S: float,
    T: float,
    level: float,
    skew_factor: float,
) -> float:
    """Two-factor mock smile vol."""
    tau = max(T, 1.0 / 12.0)
    x = math.log(K / S)
    skew = (BASE_SKEW_COEFF + skew_factor) / math.sqrt(tau)
    curv = CURVATURE_COEFF / math.sqrt(tau)
    v = BASE_VOL + level + skew * x + 0.5 * curv * x * x
    return float(np.clip(v, 0.05, 0.80))


# ---------------------------------------------------------------------------
# Bucket Greeks: vega/vanna/volga for strike-maturity buckets
# ---------------------------------------------------------------------------
# The shark fin is priced with a scalar effective volatility obtained as a weighted
# average of bucket vols. Bucket Greeks are computed by bumping one bucket at
# a time. The daily bucket PnL is then aggregated into level and skew-factor
# components to keep the report readable.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class VolBucket:
    bucket_id: str
    strike: float
    maturity: float


BUCKETS = tuple(
    VolBucket(f"K{int(round(100 * kr)):03d}_T{int(round(12 * mat)):02d}M", S0 * kr, mat)
    for mat in MATURITIES_BUCKET
    for kr in STRIKE_RATIOS
)


def _bucket_beta(bucket: VolBucket, S: float) -> float:
    tau = max(bucket.maturity, 1.0 / 12.0)
    return math.log(bucket.strike / S) / math.sqrt(tau)


def bucket_vols(S: float, level: float, skew_factor: float) -> Dict[str, float]:
    return {
        b.bucket_id: smile_vol(b.strike, S, b.maturity, level, skew_factor)
        for b in BUCKETS
    }


def bucket_weights(S: float, T: float) -> Dict[str, float]:
    raw: Dict[str, float] = {}
    strike_bandwidth = 0.14
    maturity_bandwidth = 0.65
    anchors = (STRIKE, BARRIER)
    for b in BUCKETS:
        strike_score = 0.0
        for anchor in anchors:
            z_k = math.log(b.strike / anchor) / strike_bandwidth
            strike_score += math.exp(-0.5 * z_k * z_k)
        z_t = math.log(max(b.maturity, 1e-6) / max(T, 1e-6)) / maturity_bandwidth
        raw[b.bucket_id] = strike_score * math.exp(-0.5 * z_t * z_t)
    total = sum(raw.values())
    if total <= 0.0:
        return {b.bucket_id: 1.0 / len(BUCKETS) for b in BUCKETS}
    return {k: v / total for k, v in raw.items()}


def _eff_vol(
    S: float,
    T: float,
    level: float,
    skew_factor: float,
    bucket_bump: Optional[Tuple[str, float]] = None,
) -> float:
    vols = bucket_vols(S, level, skew_factor)
    weights = bucket_weights(S, T)
    if bucket_bump is not None:
        bid, bump = bucket_bump
        vols[bid] = vols[bid] + bump
    return float(np.clip(sum(weights[k] * vols[k] for k in weights), 0.05, 0.80))


def compute_factor_greeks(
    S: float,
    T_remain: float,
    level: float,
    skew_factor: float,
    knocked_out: bool,
    db: float = 0.0025,
    dS: float = 0.50,
) -> Dict[str, float]:
    """
    Return spot Greeks and bucket-vol Greeks.

    The bucket Greeks are stored in the nested ``bucket_greeks`` dictionary.
    Aggregate level/skew Greeks are also returned for compatibility with the
    existing plots. They are computed by propagating bucket Greeks through
    d sigma_bucket = d level + beta_bucket d skew_factor.
    """
    if knocked_out:
        out = {k: 0.0 for k in (
            "delta", "gamma", "dollar_gamma",
            "vega_level", "vanna_level", "volga_level",
            "vega_skew", "vanna_skew", "volga_skew", "theta",
        )}
        out["bucket_greeks"] = {}
        return out

    v0 = _eff_vol(S, T_remain, level, skew_factor)
    p0 = sharkfin_price(S, T_remain, v0)

    # Spot finite differences (for delta, gamma)
    v_su = _eff_vol(S + dS, T_remain, level, skew_factor)
    v_sd = _eff_vol(S - dS, T_remain, level, skew_factor)
    p_su = sharkfin_price(S + dS, T_remain, v_su)
    p_sd = sharkfin_price(S - dS, T_remain, v_sd)
    delta = (p_su - p_sd) / (2.0 * dS)
    gamma = (p_su - 2.0 * p0 + p_sd) / (dS * dS)

    bucket_greeks: Dict[str, Dict[str, float]] = {}
    vega_level = vanna_level = volga_level = 0.0
    vega_skew = vanna_skew = volga_skew = 0.0
    weights = bucket_weights(S, T_remain)

    # PDE Greeks with respect to the scalar effective volatility.
    p_vu = sharkfin_price(S, T_remain, v0 + db)
    p_vd = sharkfin_price(S, T_remain, v0 - db)
    eff_vega = (p_vu - p_vd) / (2.0 * db)
    eff_volga = (p_vu - 2.0 * p0 + p_vd) / (db * db)

    p_su_vu = sharkfin_price(S + dS, T_remain, v_su + db)
    p_su_vd = sharkfin_price(S + dS, T_remain, v_su - db)
    p_sd_vu = sharkfin_price(S - dS, T_remain, v_sd + db)
    p_sd_vd = sharkfin_price(S - dS, T_remain, v_sd - db)
    eff_vanna = (p_su_vu - p_su_vd - p_sd_vu + p_sd_vd) / (4.0 * dS * db)

    for b in BUCKETS:
        bid = b.bucket_id
        beta = _bucket_beta(b, S)
        weight = weights[bid]
        bucket_vega = weight * eff_vega
        bucket_vanna = weight * eff_vanna
        bucket_volga = weight * weight * eff_volga

        bucket_greeks[bid] = {
            "strike": b.strike,
            "maturity": b.maturity,
            "weight": weight,
            "beta": beta,
            "vega": bucket_vega,
            "vanna": bucket_vanna,
            "volga": bucket_volga,
        }

        vega_level += bucket_vega
        vanna_level += bucket_vanna
        volga_level += bucket_volga
        vega_skew += beta * bucket_vega
        vanna_skew += beta * bucket_vanna
        volga_skew += beta * beta * bucket_volga

    # Theta (forward time difference)
    dt_small = min(DT, max(MATURITY - T_remain, DT) * 0.25)
    T_fwd = max(T_remain - dt_small, 0.0)
    v_fwd = _eff_vol(S, T_fwd, level, skew_factor)
    p_fwd = sharkfin_price(S, T_fwd, v_fwd)
    theta = (p_fwd - p0) / dt_small

    return {
        "p0": p0,
        "delta": delta,
        "gamma": gamma,
        "dollar_gamma": 0.5 * S * S * gamma,
        "vega_level": vega_level,
        "vanna_level": vanna_level,
        "volga_level": volga_level,
        "vega_skew": vega_skew,
        "vanna_skew": vanna_skew,
        "volga_skew": volga_skew,
        "theta": theta,
        "bucket_greeks": bucket_greeks,
    }


# ---------------------------------------------------------------------------
# Mock surface path generation (two factors: level + skew_factor)
# ---------------------------------------------------------------------------

def generate_barrier_path() -> Dict[str, np.ndarray]:
    """
    Simulate (S_t, level_t, skew_factor_t, shadow_t) jointly.

    level_t   ~ level factor (parallel shift of ATM vol)
    skew_factor_t ~ incremental skew factor (moves the smile tilt)
    shadow_t  ~ non-admissible LSV diagnostic state (not in pricing)
    """
    rng = np.random.default_rng(SEED)
    n = TRADING_DAYS

    z_s = rng.standard_normal(n)
    z_l = rng.standard_normal(n)   # independent driver for level
    z_k = rng.standard_normal(n)   # independent driver for skew factor
    z_sh = rng.standard_normal(n)  # shadow

    sigma_s = 0.20
    eta_l = 0.080      # vol-of-level
    rho_sl = -0.75
    eta_k = 0.040      # vol-of-skew-factor
    rho_sk = -0.50     # skew factor correlated with spot (leverage effect)
    eta_sh = 0.10
    rho_ssh = -0.35

    dx = (-0.5 * sigma_s ** 2 * DT) + sigma_s * math.sqrt(DT) * z_s
    dlevel = eta_l * math.sqrt(DT) * (
        rho_sl * z_s + math.sqrt(1.0 - rho_sl ** 2) * z_l
    )
    dskew = eta_k * math.sqrt(DT) * (
        rho_sk * z_s + math.sqrt(1.0 - rho_sk ** 2) * z_k
    )
    dshadow = eta_sh * math.sqrt(DT) * (
        rho_ssh * z_s + math.sqrt(1.0 - rho_ssh ** 2) * z_sh
    )

    spot = S0 * np.exp(np.r_[0.0, np.cumsum(dx)])
    level = np.r_[0.0, np.cumsum(dlevel)]
    skew_factor = np.r_[0.0, np.cumsum(dskew)]
    shadow = np.r_[0.0, np.cumsum(dshadow)]

    return {
        "dx": dx, "dlevel": dlevel, "dskew": dskew, "dshadow": dshadow,
        "spot": spot, "level": level, "skew_factor": skew_factor, "shadow": shadow,
    }


# ---------------------------------------------------------------------------
# Breakeven covariance kernels
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BarrierKernel:
    name: str
    label: str
    # Breakevens for the two surface factors
    cov_S_level: float     # hat_c^{S, level}
    var_level: float       # hat_c^{level, level}
    cov_S_skew: float      # hat_c^{S, skew}
    var_skew: float        # hat_c^{skew, skew}
    leakage_sensitivity: float
    leakage_var_be: float
    note: str


def build_barrier_kernels(realized: Dict[str, float]) -> List[BarrierKernel]:
    return [
        BarrierKernel(
            "bs_zero", "BS zero-breakeven",
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            "No surface-dynamic breakeven; all Vanna/Volga enters residual",
        ),
        BarrierKernel(
            "local_vol", "Local volatility",
            -0.0140, 0.0049, -0.0060, 0.0014, 0.0, 0.0,
            "Vanna/Volga breakevens locked by initial smile slope and curvature",
        ),
        BarrierKernel(
            "heston_sv", "Heston-style SV",
            -0.0090, 0.0045, -0.0030, 0.0008, 0.0, 0.0,
            "Single stochastic variance factor; lower skew-factor breakeven",
        ),
        BarrierKernel(
            "bergomi_2f", "Bergomi 2F",
            -0.0105, 0.0063, -0.0050, 0.0016, 0.0, 0.0,
            "Two forward-variance factors; close to mock path realized values",
        ),
        BarrierKernel(
            "admissible_lsv", "Admissible LSV",
            -0.0100, 0.0065, -0.0048, 0.0015, 0.0, 0.0,
            "Local smile + admissible stochastic vol; no leakage",
        ),
        BarrierKernel(
            "nonadmissible_lsv", "Non-admissible LSV",
            -0.0100, 0.0065, -0.0048, 0.0015,
            5.0, realized["shadow_var"],
            "Same surface breakevens plus a shadow-state leakage diagnostic",
        ),
    ]


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_barrier_experiment() -> Tuple[
    List[Dict], List[Dict], List[Dict], List[Dict], Dict[str, np.ndarray]
]:
    path = generate_barrier_path()
    dx = path["dx"]
    dlevel = path["dlevel"]
    dskew = path["dskew"]
    dshadow = path["dshadow"]
    spot = path["spot"]
    level = path["level"]
    skew_factor = path["skew_factor"]
    shadow = path["shadow"]

    realized = {
        "realized_var": float(np.sum(dx * dx)),
        "cov_S_level": float(np.sum(dx * dlevel)),
        "var_level": float(np.sum(dlevel * dlevel)),
        "cov_S_skew": float(np.sum(dx * dskew)),
        "var_skew": float(np.sum(dskew * dskew)),
        "shadow_var": float(np.sum(dshadow * dshadow)),
    }

    kernels = build_barrier_kernels(realized)
    daily_rows: List[Dict] = []
    summary_rows: List[Dict] = []
    kernel_rows: List[Dict] = []
    bucket_rows: List[Dict] = []

    # Detect knockout day (first day spot >= BARRIER)
    knockout_day: Optional[int] = None
    for i in range(TRADING_DAYS + 1):
        if float(spot[i]) >= BARRIER:
            knockout_day = i
            break

    for kernel in kernels:
        cum_actual = 0.0
        cum_explained = 0.0
        cum_residual = 0.0
        cum_std_residual = 0.0
        cum_gamma_theta = 0.0
        cum_vega_level = 0.0
        cum_vega_skew = 0.0
        cum_vanna_level = 0.0
        cum_vanna_skew = 0.0
        cum_volga_level = 0.0
        cum_volga_skew = 0.0
        cum_leakage = 0.0

        knocked_out = False
        locked_coupon = 0.0

        pnl_list: List[float] = []
        resid_list: List[float] = []

        for i in range(TRADING_DAYS):
            t = i * DT
            s_t = float(spot[i])
            s_next = float(spot[i + 1])
            lv_t = float(level[i])
            lv_next = float(level[i + 1])
            sk_t = float(skew_factor[i])
            sk_next = float(skew_factor[i + 1])
            d_x = float(dx[i])
            d_lv = float(dlevel[i])
            d_sk = float(dskew[i])
            d_sh = float(dshadow[i])
            d_s = s_next - s_t
            T_remain = max(MATURITY - t, DT)
            T_remain_next = max(MATURITY - t - DT, 0.0)

            # Check for knockout at next step
            just_knocked_out = (not knocked_out) and (s_next >= BARRIER)
            if knocked_out:
                # Position is closed; no further PnL
                # Still record a zero row so the event window is complete
                row = {
                    "model": kernel.name, "model_label": kernel.label,
                    "day": i, "spot": s_t, "level": lv_t, "skew_factor": sk_t,
                    "barrier_distance": (BARRIER - s_t) / s_t,
                    "knocked_out": True,
                    "actual_pnl": 0.0,
                    "gamma_theta_pnl": 0.0,
                    "vega_level_pnl": 0.0, "vega_skew_pnl": 0.0,
                    "vanna_level_pnl": 0.0, "vanna_skew_pnl": 0.0,
                    "volga_level_pnl": 0.0, "volga_skew_pnl": 0.0,
                    "leakage_pnl": 0.0, "explained_pnl": 0.0,
                    "standard_explained_pnl": 0.0,
                    "residual": 0.0, "standard_residual": 0.0,
                    "locked_coupon": 0.0,
                    "cum_actual_pnl": cum_actual,
                    "cum_explained_pnl": cum_explained,
                    "cum_gamma_theta_pnl": cum_gamma_theta,
                    "cum_vega_level_pnl": cum_vega_level,
                    "cum_vega_skew_pnl": cum_vega_skew,
                    "cum_vanna_level_pnl": cum_vanna_level,
                    "cum_vanna_skew_pnl": cum_vanna_skew,
                    "cum_volga_level_pnl": cum_volga_level,
                    "cum_volga_skew_pnl": cum_volga_skew,
                    "cum_leakage_pnl": cum_leakage,
                    "cum_residual": cum_residual,
                    "cum_std_residual": cum_std_residual,
                }
                daily_rows.append(row)
                pnl_list.append(0.0)
                resid_list.append(0.0)
                continue

            # Factor Greeks (level and skew surface factors) — computed first,
            # delta and p0 are taken from here to avoid redundant pricing calls.
            fg = compute_factor_greeks(s_t, T_remain, lv_t, sk_t, knocked_out)
            p_t = fg.get("p0", sharkfin_price(s_t, T_remain, _eff_vol(s_t, T_remain, lv_t, sk_t)))
            delta_t = fg["delta"]

            # Next-step repricing
            if just_knocked_out:
                # The single shark fin locks its knock-out coupon.
                p_next = KO_PAYOFF
                locked_coupon = KO_PAYOFF
            else:
                v_eff_next = _eff_vol(s_next, T_remain_next, lv_next, sk_next)
                p_next = sharkfin_price(s_next, T_remain_next, v_eff_next)
                locked_coupon = 0.0

            # Actual PnL (short shark fin coupon leg + delta hedge)
            actual_pnl_base = -(p_next - p_t) + delta_t * d_s

            # Leakage term (non-admissible LSV diagnostic)
            leakage_pnl = -kernel.leakage_sensitivity * d_sh - 0.5 * kernel.leakage_sensitivity * (
                d_sh * d_sh - kernel.leakage_var_be * DT
            )
            actual_pnl = actual_pnl_base + leakage_pnl

            # Gamma/Theta component from local Taylor accounting.
            gamma_theta = -fg["theta"] * DT - fg["dollar_gamma"] * (d_x * d_x)

            # Bucket Vega/Vanna/Volga. Bucket moves are induced by the two
            # surface factors: d sigma_j = d level + beta_j d skew.
            vega_level_pnl = vega_skew_pnl = 0.0
            vanna_level_pnl = vanna_skew_pnl = 0.0
            volga_level_pnl = volga_skew_pnl = 0.0
            for bid, bg in fg["bucket_greeks"].items():
                beta = bg["beta"]
                dvol_level = d_lv
                dvol_skew = beta * d_sk
                cov_level = kernel.cov_S_level
                cov_skew = beta * kernel.cov_S_skew
                var_level = kernel.var_level
                var_skew = beta * beta * kernel.var_skew

                bucket_vega_level = -bg["vega"] * dvol_level
                bucket_vega_skew = -bg["vega"] * dvol_skew
                bucket_vanna_level = -bg["vanna"] * (d_s * dvol_level - s_t * cov_level * DT)
                bucket_vanna_skew = -bg["vanna"] * (d_s * dvol_skew - s_t * cov_skew * DT)
                bucket_volga_level = -0.5 * bg["volga"] * (dvol_level * dvol_level - var_level * DT)
                bucket_volga_skew = -0.5 * bg["volga"] * (dvol_skew * dvol_skew - var_skew * DT)

                vega_level_pnl += bucket_vega_level
                vega_skew_pnl += bucket_vega_skew
                vanna_level_pnl += bucket_vanna_level
                vanna_skew_pnl += bucket_vanna_skew
                volga_level_pnl += bucket_volga_level
                volga_skew_pnl += bucket_volga_skew

                bucket_rows.append({
                    "model": kernel.name,
                    "model_label": kernel.label,
                    "day": i,
                    "bucket_id": bid,
                    "bucket_strike": bg["strike"],
                    "bucket_maturity": bg["maturity"],
                    "bucket_weight": bg["weight"],
                    "bucket_beta": beta,
                    "bucket_vega": bg["vega"],
                    "bucket_vanna": bg["vanna"],
                    "bucket_volga": bg["volga"],
                    "dvol_level": dvol_level,
                    "dvol_skew": dvol_skew,
                    "vega_level_pnl": bucket_vega_level,
                    "vega_skew_pnl": bucket_vega_skew,
                    "vanna_level_pnl": bucket_vanna_level,
                    "vanna_skew_pnl": bucket_vanna_skew,
                    "volga_level_pnl": bucket_volga_level,
                    "volga_skew_pnl": bucket_volga_skew,
                })

            standard_explained = (
                gamma_theta
                + vega_level_pnl + vanna_level_pnl + volga_level_pnl
                + vega_skew_pnl + vanna_skew_pnl + volga_skew_pnl
            )
            explained = standard_explained + leakage_pnl
            residual = actual_pnl - explained
            standard_residual = actual_pnl - standard_explained

            cum_actual += actual_pnl
            cum_explained += explained
            cum_residual += residual
            cum_std_residual += standard_residual
            cum_gamma_theta += gamma_theta
            cum_vega_level += vega_level_pnl
            cum_vega_skew += vega_skew_pnl
            cum_vanna_level += vanna_level_pnl
            cum_vanna_skew += vanna_skew_pnl
            cum_volga_level += volga_level_pnl
            cum_volga_skew += volga_skew_pnl
            cum_leakage += leakage_pnl

            pnl_list.append(actual_pnl)
            resid_list.append(residual)

            daily_rows.append({
                "model": kernel.name, "model_label": kernel.label,
                "day": i, "spot": s_t, "level": lv_t, "skew_factor": sk_t,
                "barrier_distance": (BARRIER - s_t) / s_t,
                "knocked_out": knocked_out,
                "actual_pnl": actual_pnl,
                "gamma_theta_pnl": gamma_theta,
                "vega_level_pnl": vega_level_pnl,
                "vega_skew_pnl": vega_skew_pnl,
                "vanna_level_pnl": vanna_level_pnl,
                "vanna_skew_pnl": vanna_skew_pnl,
                "volga_level_pnl": volga_level_pnl,
                "volga_skew_pnl": volga_skew_pnl,
                "leakage_pnl": leakage_pnl,
                "explained_pnl": explained,
                "standard_explained_pnl": standard_explained,
                "residual": residual,
                "standard_residual": standard_residual,
                "locked_coupon": locked_coupon,
                "cum_actual_pnl": cum_actual,
                "cum_explained_pnl": cum_explained,
                "cum_gamma_theta_pnl": cum_gamma_theta,
                "cum_vega_level_pnl": cum_vega_level,
                "cum_vega_skew_pnl": cum_vega_skew,
                "cum_vanna_level_pnl": cum_vanna_level,
                "cum_vanna_skew_pnl": cum_vanna_skew,
                "cum_volga_level_pnl": cum_volga_level,
                "cum_volga_skew_pnl": cum_volga_skew,
                "cum_leakage_pnl": cum_leakage,
                "cum_residual": cum_residual,
                "cum_std_residual": cum_std_residual,
            })

            if just_knocked_out:
                knocked_out = True

        pnl_arr = np.array(pnl_list, dtype=float)
        resid_arr = np.array(resid_list, dtype=float)
        pnl_var = float(np.var(pnl_arr[pnl_arr != 0.0])) if np.any(pnl_arr != 0.0) else 1.0
        resid_var = float(np.var(resid_arr[resid_arr != 0.0])) if np.any(resid_arr != 0.0) else 0.0
        r2 = 1.0 - resid_var / pnl_var if pnl_var > 1e-14 else 1.0

        summary_rows.append({
            "model": kernel.name, "model_label": kernel.label,
            "knockout_day": knockout_day if knockout_day is not None else -1,
            "total_actual_pnl": cum_actual,
            "total_explained_pnl": cum_explained,
            "total_gamma_theta": cum_gamma_theta,
            "total_vega_level": cum_vega_level,
            "total_vega_skew": cum_vega_skew,
            "total_vanna_level": cum_vanna_level,
            "total_vanna_skew": cum_vanna_skew,
            "total_volga_level": cum_volga_level,
            "total_volga_skew": cum_volga_skew,
            "total_leakage": cum_leakage,
            "total_residual": cum_residual,
            "total_std_residual": cum_std_residual,
            "explanation_ratio": r2,
        })
        kernel_rows.append({
            "model": kernel.name, "model_label": kernel.label,
            "cov_S_level_be": kernel.cov_S_level,
            "var_level_be": kernel.var_level,
            "cov_S_skew_be": kernel.cov_S_skew,
            "var_skew_be": kernel.var_skew,
            "leakage_sensitivity": kernel.leakage_sensitivity,
            "realized_cov_S_level": realized["cov_S_level"],
            "realized_var_level": realized["var_level"],
            "realized_cov_S_skew": realized["cov_S_skew"],
            "realized_var_skew": realized["var_skew"],
            "note": kernel.note,
        })

    return daily_rows, summary_rows, kernel_rows, bucket_rows, path


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def write_csv(path: Path, rows: Iterable[Dict]) -> None:
    rows = list(rows)
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def _event_window_rows(rows: List[Dict], knockout_day: int, window: int = 20) -> List[Dict]:
    """Return rows within [-window, +window] days of the knockout day."""
    return [r for r in rows if abs(r["day"] - knockout_day) <= window]


def plot_barrier_results(
    daily_rows: List[Dict],
    summary_rows: List[Dict],
    path: Dict[str, np.ndarray],
    knockout_day: Optional[int],
) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    spot = path["spot"]
    days_all = np.arange(TRADING_DAYS)

    # ── Figure 1: dashboard for Bergomi 2F ──────────────────────────────────
    model = "bergomi_2f"
    rows = [r for r in daily_rows if r["model"] == model]
    days = np.array([r["day"] for r in rows])

    fig, axes = plt.subplots(5, 1, figsize=(11, 14), sharex=True)
    axes[0].plot(days, [r["spot"] for r in rows], lw=1.4)
    axes[0].axhline(BARRIER, color="red", lw=1.0, ls="--", label=f"Barrier {BARRIER}")
    if knockout_day is not None:
        axes[0].axvline(knockout_day, color="red", lw=0.8, ls=":")
    axes[0].set_ylabel("Spot")
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].set_title("Single shark fin PnL breakdown — Bergomi 2F convention")

    axes[1].plot(days, [100.0 * r["barrier_distance"] for r in rows], color="#9467bd", lw=1.2)
    axes[1].axhline(0.0, color="red", lw=0.8, ls="--")
    if knockout_day is not None:
        axes[1].axvline(knockout_day, color="red", lw=0.8, ls=":")
    axes[1].set_ylabel("Barrier distance (%)")

    axes[2].plot(days, [100.0 * r["level"] for r in rows], label="Level factor", lw=1.1)
    axes[2].plot(days, [100.0 * r["skew_factor"] for r in rows], label="Skew factor", lw=1.1)
    if knockout_day is not None:
        axes[2].axvline(knockout_day, color="red", lw=0.8, ls=":")
    axes[2].set_ylabel("Surface factors\n(vol pts)")
    axes[2].legend(frameon=False, fontsize=8)

    axes[3].stackplot(
        days,
        [r["vega_level_pnl"] for r in rows],
        [r["vega_skew_pnl"] for r in rows],
        labels=["Vega-level", "Vega-skew"],
        alpha=0.6,
    )
    if knockout_day is not None:
        axes[3].axvline(knockout_day, color="red", lw=0.8, ls=":")
    axes[3].set_ylabel("Daily Vega PnL\n(bucket decomp.)")
    axes[3].legend(frameon=False, fontsize=8)

    axes[4].plot(days, [r["cum_actual_pnl"] for r in rows], label="actual", lw=1.4)
    axes[4].plot(days, [r["cum_explained_pnl"] for r in rows], label="explained", lw=1.4, ls="--")
    axes[4].plot(days, [r["cum_residual"] for r in rows], label="residual", lw=1.1, ls=":")
    if knockout_day is not None:
        axes[4].axvline(knockout_day, color="red", lw=0.8, ls=":", label="knockout")
    axes[4].set_ylabel("Cumulative PnL")
    axes[4].set_xlabel("Day")
    axes[4].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "sharkfin_pnl_dashboard_bergomi2f.png", dpi=180)
    plt.close(fig)

    # ── Figure 2: event window PnL around knockout ───────────────────────────
    if knockout_day is not None:
        WINDOW = 20
        fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
        for km, label, color in [
            ("bs_zero", "BS zero-be", "#4c78a8"),
            ("local_vol", "LV", "#f58518"),
            ("bergomi_2f", "Bergomi 2F", "#54a24b"),
            ("admissible_lsv", "Admissible LSV", "#b279a2"),
        ]:
            win_rows = [
                r for r in daily_rows
                if r["model"] == km and abs(r["day"] - knockout_day) <= WINDOW
            ]
            tau = [r["day"] - knockout_day for r in win_rows]
            axes[0].plot(tau, [r["actual_pnl"] for r in win_rows], label=label, lw=1.2, color=color)
            axes[1].plot(tau, [r["residual"] for r in win_rows], lw=1.2, color=color, ls="--")

        axes[0].axvline(0, color="red", lw=0.8, ls=":")
        axes[0].axhline(0, color="black", lw=0.6)
        axes[0].set_ylabel("Daily actual PnL")
        axes[0].legend(frameon=False, fontsize=8)
        axes[0].set_title(f"Event window ±{WINDOW} days around knockout (day {knockout_day})")

        axes[1].axvline(0, color="red", lw=0.8, ls=":")
        axes[1].axhline(0, color="black", lw=0.6)
        axes[1].set_ylabel("Daily residual")
        axes[1].set_xlabel("Days relative to knockout")
        fig.tight_layout()
        fig.savefig(FIGURES / "sharkfin_event_window.png", dpi=180)
        plt.close(fig)

    # ── Figure 3: model comparison waterfall ─────────────────────────────────
    components = [
        "total_gamma_theta", "total_vega_level", "total_vega_skew",
        "total_vanna_level", "total_vanna_skew",
        "total_volga_level", "total_volga_skew",
        "total_leakage", "total_residual",
    ]
    comp_labels = [
        "Gamma/Theta", "Vega-level", "Vega-skew",
        "Vanna-level", "Vanna-skew",
        "Volga-level", "Volga-skew",
        "Leakage", "Residual",
    ]
    colors = [
        "#4c78a8", "#f58518", "#e8a825",
        "#54a24b", "#72b7b2",
        "#b279a2", "#ff9da7",
        "#e45756", "#79706e",
    ]
    model_labels = [r["model_label"] for r in summary_rows]
    x = np.arange(len(model_labels))
    bot_pos = np.zeros(len(model_labels))
    bot_neg = np.zeros(len(model_labels))
    fig, ax = plt.subplots(figsize=(13, 6))
    for comp, lbl, clr in zip(components, comp_labels, colors):
        vals = np.array([r[comp] for r in summary_rows], dtype=float)
        bots = np.where(vals >= 0.0, bot_pos, bot_neg)
        ax.bar(x, vals, bottom=bots, label=lbl, color=clr)
        bot_pos += np.where(vals >= 0.0, vals, 0.0)
        bot_neg += np.where(vals < 0.0, vals, 0.0)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(model_labels, rotation=20, ha="right")
    ax.set_ylabel("Total PnL contribution")
    ax.set_title("Bucket-vega PnL breakdown: single shark fin across accounting conventions")
    ax.legend(frameon=False, ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "sharkfin_model_waterfall.png", dpi=180)
    plt.close(fig)

    # ── Figure 4: cumulative components for all models ───────────────────────
    selected = [
        ("bs_zero", "BS zero-breakeven"),
        ("local_vol", "Local volatility"),
        ("bergomi_2f", "Bergomi 2F"),
        ("admissible_lsv", "Admissible LSV"),
        ("nonadmissible_lsv", "Non-admissible LSV"),
    ]
    cum_specs = [
        ("cum_gamma_theta_pnl", "Gamma/Theta", "#4c78a8"),
        ("cum_vega_level_pnl", "Vega-level", "#f58518"),
        ("cum_vega_skew_pnl", "Vega-skew", "#e8a825"),
        ("cum_vanna_level_pnl", "Vanna-level", "#54a24b"),
        ("cum_vanna_skew_pnl", "Vanna-skew", "#72b7b2"),
        ("cum_volga_level_pnl", "Volga-level", "#b279a2"),
        ("cum_volga_skew_pnl", "Volga-skew", "#ff9da7"),
        ("cum_leakage_pnl", "Leakage", "#e45756"),
        ("cum_residual", "Residual", "#79706e"),
    ]
    fig, axes = plt.subplots(3, 2, figsize=(14, 13), sharex=True)
    for ax, (mn, lbl) in zip(axes.ravel(), selected):
        mr = [r for r in daily_rows if r["model"] == mn]
        ds = np.array([r["day"] for r in mr])
        ax.plot(ds, [r["cum_actual_pnl"] for r in mr], color="black", lw=1.5, label="Actual")
        ax.plot(ds, [r["cum_explained_pnl"] for r in mr], color="black", lw=1.1, ls="--", label="Explained")
        for key, cl, color in cum_specs:
            vals = np.array([r[key] for r in mr], dtype=float)
            if np.max(np.abs(vals)) > 1e-10:
                ax.plot(ds, vals, lw=0.9, color=color, label=cl)
        if knockout_day is not None:
            ax.axvline(knockout_day, color="red", lw=0.8, ls=":")
        ax.axhline(0, color="black", lw=0.5)
        ax.set_title(lbl, fontsize=9)
        ax.set_ylabel("Cumul. PnL", fontsize=8)
    axes[-1, -1].set_visible(False)
    axes[-1, 0].set_xlabel("Day")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower right", ncol=3, frameon=False, fontsize=8)
    fig.suptitle("Cumulative bucket-vega PnL components — single shark fin", y=0.995)
    fig.tight_layout(rect=[0.0, 0.04, 1.0, 0.97])
    fig.savefig(FIGURES / "sharkfin_cumulative_components.png", dpi=180)
    plt.close(fig)

    # ── Figure 5: leakage diagnostic ─────────────────────────────────────────
    lsv_rows = [r for r in daily_rows if r["model"] == "nonadmissible_lsv"]
    lsv_days = np.array([r["day"] for r in lsv_rows])
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(lsv_days, [r["cum_std_residual"] for r in lsv_rows], label="residual without leakage", lw=1.3)
    ax.plot(lsv_days, [r["cum_residual"] for r in lsv_rows], label="residual after leakage", lw=1.3)
    ax.plot(lsv_days, [r["cum_leakage_pnl"] for r in lsv_rows], label="cumulative leakage", lw=1.1, ls=":")
    if knockout_day is not None:
        ax.axvline(knockout_day, color="red", lw=0.8, ls=":", label="knockout")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("Day")
    ax.set_ylabel("Cumulative PnL")
    ax.set_title("Non-admissible LSV: shadow-state leakage diagnostic (single shark fin)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES / "sharkfin_lsv_leakage.png", dpi=180)
    plt.close(fig)


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    daily_rows, summary_rows, kernel_rows, bucket_rows, path = run_barrier_experiment()

    knockout_day: Optional[int] = None
    if summary_rows and summary_rows[0]["knockout_day"] >= 0:
        knockout_day = summary_rows[0]["knockout_day"]

    write_csv(RESULTS / "sharkfin_daily.csv", daily_rows)
    write_csv(RESULTS / "sharkfin_summary.csv", summary_rows)
    write_csv(RESULTS / "sharkfin_kernels.csv", kernel_rows)
    write_csv(RESULTS / "sharkfin_bucket_daily.csv", bucket_rows)

    plot_barrier_results(daily_rows, summary_rows, path, knockout_day)

    print("Generated mock-data single shark fin PnL accounting outputs.")
    print(f"Knockout day: {knockout_day}")
    print(f"Results: {RESULTS}")
    print(f"Figures: {FIGURES}")


if __name__ == "__main__":
    main()
