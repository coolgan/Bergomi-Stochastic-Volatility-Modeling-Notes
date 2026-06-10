"""
Numerical experiments for Bergomi chapter 8 smile approximations.

The script is pedagogical.  It compares the chapter-8 perturbation formulas
with lightweight Monte Carlo surfaces for Heston, one-factor lognormal SV, and
a two-factor Bergomi-style forward-variance model.  It also computes the ATMF
skew and curvature diagnostics discussed after section 8.5.
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
V0 = 0.04
SIGMA0 = math.sqrt(V0)
RATE = 0.0
SEED = 20260608
N_PATHS = 70_000
STEPS_PER_YEAR = 252
MAX_MATURITY = 5.0
MATURITIES = [1.0 / 12.0, 0.25, 1.0, 5.0]
LOG_MONEYNESS = np.array([-0.30, -0.20, -0.15, -0.10, -0.05, 0.0, 0.05, 0.10, 0.15, 0.20, 0.30])
ATM_FIT_MASK = np.abs(LOG_MONEYNESS) <= 0.10 + 1e-12


def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_call_price(spot: float, strike: float, maturity: float, vol: float) -> float:
    if maturity <= 0:
        return max(spot - strike, 0.0)
    vol = max(vol, 1e-10)
    sqrt_t = math.sqrt(maturity)
    d1 = (math.log(spot / strike) + 0.5 * vol * vol * maturity) / (vol * sqrt_t)
    d2 = d1 - vol * sqrt_t
    return spot * norm_cdf(d1) - strike * norm_cdf(d2)


def implied_vol_call(price: float, spot: float, strike: float, maturity: float) -> float:
    intrinsic = max(spot - strike, 0.0)
    price = min(max(price, intrinsic + 1e-12), spot - 1e-12)
    lo, hi = 1e-5, 4.0
    for _ in range(90):
        mid = 0.5 * (lo + hi)
        if bs_call_price(spot, strike, maturity, mid) > price:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def write_csv(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    rows = list(rows)
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def fit_quadratic_iv(xs: np.ndarray, vols: np.ndarray) -> Tuple[float, float, float]:
    coeff = np.polyfit(xs, vols, 2)
    curvature = 2.0 * coeff[0]
    skew = coeff[1]
    level = coeff[2]
    return float(level), float(skew), float(curvature)


def g_weight(kappa: float, maturity: float) -> float:
    x = max(kappa * maturity, 1e-12)
    return (x - (1.0 - math.exp(-x))) / (x * x)


def a_integral(kappa: float, maturity: float) -> float:
    return maturity * maturity * g_weight(kappa, maturity)


def b_integral(kappa: float, tau_to_maturity: float) -> float:
    if kappa <= 1e-12:
        return tau_to_maturity
    return (1.0 - math.exp(-kappa * tau_to_maturity)) / kappa


def integrate_1d(func: Callable[[np.ndarray], np.ndarray], maturity: float, n: int = 700) -> float:
    grid = np.linspace(0.0, maturity, n)
    vals = func(grid)
    return float(np.trapezoid(vals, grid))


@dataclass(frozen=True)
class ApproxCoefficients:
    c_xi: float
    c_xixi: float
    d_term: float = 0.0

    def first_order_metrics(self, maturity: float, sigma_vs: float = SIGMA0) -> Tuple[float, float, float]:
        q = sigma_vs * sigma_vs * maturity
        skew = sigma_vs * self.c_xi / (2.0 * q * q)
        level = sigma_vs * (1.0 + self.c_xi / (4.0 * q))
        return level, skew, 0.0

    def second_order_metrics(self, maturity: float, sigma_vs: float = SIGMA0) -> Tuple[float, float, float]:
        q = sigma_vs * sigma_vs * maturity
        level = sigma_vs * (
            1.0
            + self.c_xi / (4.0 * q)
            + (
                12.0 * self.c_xi * self.c_xi
                - q * (q + 4.0) * self.c_xixi
                + 4.0 * q * (q - 4.0) * self.d_term
            )
            / (32.0 * q**3)
        )
        skew = sigma_vs * (
            self.c_xi / (2.0 * q * q)
            + (4.0 * q * self.d_term - 3.0 * self.c_xi * self.c_xi) / (8.0 * q**3)
        )
        curvature = sigma_vs * (
            4.0 * q * self.d_term + q * self.c_xixi - 6.0 * self.c_xi * self.c_xi
        ) / (4.0 * q**4)
        return level, skew, curvature


@dataclass(frozen=True)
class HestonSpec:
    name: str = "heston"
    label: str = "Heston"
    kappa: float = 2.0
    theta: float = V0
    vol_of_var: float = 0.80
    rho: float = -0.35

    def coeffs(self, maturity: float) -> ApproxCoefficients:
        c_xi = self.rho * self.vol_of_var * V0 * a_integral(self.kappa, maturity)

        def integrand(t: np.ndarray) -> np.ndarray:
            b = np.array([b_integral(self.kappa, maturity - ti) for ti in t])
            return self.vol_of_var * self.vol_of_var * V0 * b * b

        c_xixi = integrate_1d(integrand, maturity)
        return ApproxCoefficients(c_xi=c_xi, c_xixi=c_xixi)


@dataclass(frozen=True)
class LognormalSingleSpec:
    name: str = "lognormal_1f"
    label: str = "Lognormal 1F"
    kappa: float = 1.5
    nu: float = 1.40
    rho: float = -0.30

    def coeffs(self, maturity: float) -> ApproxCoefficients:
        c_xi = 2.0 * self.nu * V0 * SIGMA0 * self.rho * a_integral(self.kappa, maturity)

        def integrand(t: np.ndarray) -> np.ndarray:
            b = np.array([b_integral(self.kappa, maturity - ti) for ti in t])
            return (2.0 * self.nu * V0 * b) ** 2

        c_xixi = integrate_1d(integrand, maturity)
        return ApproxCoefficients(c_xi=c_xi, c_xixi=c_xixi)


@dataclass(frozen=True)
class Bergomi2FSpec:
    name: str
    label: str
    nu: float = 3.00
    theta: float = 0.245
    k1: float = 5.35
    k2: float = 0.28
    rho12: float = 0.0
    rho_s1: float = -0.355
    rho_s2: float = -0.227

    @property
    def alpha(self) -> float:
        return 1.0 / math.sqrt((1.0 - self.theta) ** 2 + self.theta**2)

    def coeffs(self, maturity: float) -> ApproxCoefficients:
        w1 = 1.0 - self.theta
        w2 = self.theta
        c_xi = (
            2.0
            * self.nu
            * V0
            * SIGMA0
            * self.alpha
            * (w1 * self.rho_s1 * a_integral(self.k1, maturity) + w2 * self.rho_s2 * a_integral(self.k2, maturity))
        )

        def integrand(t: np.ndarray) -> np.ndarray:
            b1 = np.array([b_integral(self.k1, maturity - ti) for ti in t])
            b2 = np.array([b_integral(self.k2, maturity - ti) for ti in t])
            factor = 2.0 * self.nu * V0 * self.alpha
            return factor * factor * (
                w1 * w1 * b1 * b1
                + w2 * w2 * b2 * b2
                + 2.0 * self.rho12 * w1 * w2 * b1 * b2
            )

        c_xixi = integrate_1d(integrand, maturity)
        return ApproxCoefficients(c_xi=c_xi, c_xixi=c_xixi)

    def first_order_skew_formula(self, maturity: float) -> float:
        w1 = 1.0 - self.theta
        w2 = self.theta
        return self.nu * self.alpha * (
            w1 * self.rho_s1 * g_weight(self.k1, maturity) + w2 * self.rho_s2 * g_weight(self.k2, maturity)
        )

    def factor_contributions(self, maturity: float) -> Tuple[float, float]:
        w1 = 1.0 - self.theta
        w2 = self.theta
        c1 = self.nu * self.alpha * w1 * self.rho_s1 * g_weight(self.k1, maturity)
        c2 = self.nu * self.alpha * w2 * self.rho_s2 * g_weight(self.k2, maturity)
        return c1, c2


MODELS = [
    HestonSpec(),
    LognormalSingleSpec(),
    Bergomi2FSpec(name="bergomi_2f_set_ii", label="Bergomi 2F Set II"),
]

BERGOMI_BASE = Bergomi2FSpec(name="bergomi_2f_set_ii", label="Bergomi 2F Set II")
BERGOMI_HIGH_NU_SAME_SKEW = Bergomi2FSpec(
    name="bergomi_2f_high_nu_same_skew",
    label="Bergomi 2F high nu, lower rho",
    nu=BERGOMI_BASE.nu / 0.75,
    theta=BERGOMI_BASE.theta,
    k1=BERGOMI_BASE.k1,
    k2=BERGOMI_BASE.k2,
    rho12=BERGOMI_BASE.rho12,
    rho_s1=BERGOMI_BASE.rho_s1 * 0.75,
    rho_s2=BERGOMI_BASE.rho_s2 * 0.75,
)


def surface_from_metrics(xs: np.ndarray, level: float, skew: float, curvature: float) -> np.ndarray:
    return level + skew * xs + 0.5 * curvature * xs * xs


def safe_corr_cholesky(corr: np.ndarray) -> np.ndarray:
    eig = np.linalg.eigvalsh(corr)
    if float(np.min(eig)) < 1e-10:
        corr = corr + np.eye(corr.shape[0]) * (1e-10 - float(np.min(eig)))
    return np.linalg.cholesky(corr)


def simulate_heston(spec: HestonSpec, n_paths: int = N_PATHS) -> Dict[float, np.ndarray]:
    rng = np.random.default_rng(SEED + 11)
    max_steps = int(round(MAX_MATURITY * STEPS_PER_YEAR))
    dt = 1.0 / STEPS_PER_YEAR
    sqrt_dt = math.sqrt(dt)
    maturity_steps = {int(round(t * STEPS_PER_YEAR)): t for t in MATURITIES}
    chol = safe_corr_cholesky(np.array([[1.0, spec.rho], [spec.rho, 1.0]]))

    log_s = np.full(n_paths, math.log(S0))
    v = np.full(n_paths, V0)
    out: Dict[float, np.ndarray] = {}
    for step in range(1, max_steps + 1):
        z = rng.standard_normal((2, n_paths))
        z_s, z_v = chol @ z
        v_pos = np.maximum(v, 0.0)
        log_s += -0.5 * v_pos * dt + np.sqrt(v_pos) * sqrt_dt * z_s
        v = v + spec.kappa * (spec.theta - v_pos) * dt + spec.vol_of_var * np.sqrt(v_pos) * sqrt_dt * z_v
        v = np.maximum(v, 0.0)
        if step in maturity_steps:
            out[maturity_steps[step]] = np.exp(log_s.copy())
    return out


def simulate_lognormal_1f(spec: LognormalSingleSpec, n_paths: int = N_PATHS) -> Dict[float, np.ndarray]:
    rng = np.random.default_rng(SEED + 17)
    max_steps = int(round(MAX_MATURITY * STEPS_PER_YEAR))
    dt = 1.0 / STEPS_PER_YEAR
    sqrt_dt = math.sqrt(dt)
    maturity_steps = {int(round(t * STEPS_PER_YEAR)): t for t in MATURITIES}
    chol = safe_corr_cholesky(np.array([[1.0, spec.rho], [spec.rho, 1.0]]))

    log_s = np.full(n_paths, math.log(S0))
    x = np.zeros(n_paths)
    out: Dict[float, np.ndarray] = {}
    for step in range(1, max_steps + 1):
        t_prev = (step - 1) * dt
        var_x = (1.0 - math.exp(-2.0 * spec.kappa * t_prev)) / (2.0 * spec.kappa)
        v = V0 * np.exp(2.0 * spec.nu * x - 2.0 * spec.nu * spec.nu * var_x)
        v = np.clip(v, 1e-6, 3.0)
        z = rng.standard_normal((2, n_paths))
        z_s, z_x = chol @ z
        log_s += -0.5 * v * dt + np.sqrt(v) * sqrt_dt * z_s
        x += -spec.kappa * x * dt + sqrt_dt * z_x
        if step in maturity_steps:
            out[maturity_steps[step]] = np.exp(log_s.copy())
    return out


def simulate_bergomi_2f(spec: Bergomi2FSpec, n_paths: int = N_PATHS) -> Dict[float, np.ndarray]:
    rng = np.random.default_rng(SEED + 23)
    max_steps = int(round(MAX_MATURITY * STEPS_PER_YEAR))
    dt = 1.0 / STEPS_PER_YEAR
    sqrt_dt = math.sqrt(dt)
    maturity_steps = {int(round(t * STEPS_PER_YEAR)): t for t in MATURITIES}
    corr = np.array(
        [
            [1.0, spec.rho_s1, spec.rho_s2],
            [spec.rho_s1, 1.0, spec.rho12],
            [spec.rho_s2, spec.rho12, 1.0],
        ]
    )
    chol = safe_corr_cholesky(corr)
    w1 = 1.0 - spec.theta
    w2 = spec.theta

    log_s = np.full(n_paths, math.log(S0))
    x1 = np.zeros(n_paths)
    x2 = np.zeros(n_paths)
    out: Dict[float, np.ndarray] = {}
    for step in range(1, max_steps + 1):
        t_prev = (step - 1) * dt
        var_x1 = (1.0 - math.exp(-2.0 * spec.k1 * t_prev)) / (2.0 * spec.k1)
        var_x2 = (1.0 - math.exp(-2.0 * spec.k2 * t_prev)) / (2.0 * spec.k2)
        cov_x12 = spec.rho12 * (1.0 - math.exp(-(spec.k1 + spec.k2) * t_prev)) / (spec.k1 + spec.k2)
        var_y = spec.alpha * spec.alpha * (w1 * w1 * var_x1 + w2 * w2 * var_x2 + 2.0 * w1 * w2 * cov_x12)
        y = spec.alpha * (w1 * x1 + w2 * x2)
        v = V0 * np.exp(2.0 * spec.nu * y - 2.0 * spec.nu * spec.nu * var_y)
        v = np.clip(v, 1e-6, 3.0)
        z = rng.standard_normal((3, n_paths))
        z_s, z1, z2 = chol @ z
        log_s += -0.5 * v * dt + np.sqrt(v) * sqrt_dt * z_s
        x1 += -spec.k1 * x1 * dt + sqrt_dt * z1
        x2 += -spec.k2 * x2 * dt + sqrt_dt * z2
        if step in maturity_steps:
            out[maturity_steps[step]] = np.exp(log_s.copy())
    return out


def mc_surface_from_terminal(terminals: Dict[float, np.ndarray], model_label: str) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for maturity, spot_t in terminals.items():
        for x in LOG_MONEYNESS:
            strike = S0 * math.exp(float(x))
            price = float(np.mean(np.maximum(spot_t - strike, 0.0)))
            iv = implied_vol_call(price, S0, strike, maturity)
            rows.append(
                {
                    "model": model_label,
                    "maturity": maturity,
                    "log_moneyness": float(x),
                    "strike": strike,
                    "price": price,
                    "mc_iv": iv,
                }
            )
    return rows


def run_monte_carlo_surfaces() -> List[Dict[str, object]]:
    all_rows: List[Dict[str, object]] = []
    all_rows.extend(mc_surface_from_terminal(simulate_heston(MODELS[0]), MODELS[0].label))
    all_rows.extend(mc_surface_from_terminal(simulate_lognormal_1f(MODELS[1]), MODELS[1].label))
    all_rows.extend(mc_surface_from_terminal(simulate_bergomi_2f(MODELS[2]), MODELS[2].label))
    return all_rows


def approximate_surface_rows() -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for model in MODELS:
        for maturity in MATURITIES:
            coeffs = model.coeffs(maturity)
            level_1, skew_1, curv_1 = coeffs.first_order_metrics(maturity)
            level_2, skew_2, curv_2 = coeffs.second_order_metrics(maturity)
            for x in LOG_MONEYNESS:
                rows.append(
                    {
                        "model": model.label,
                        "maturity": maturity,
                        "log_moneyness": float(x),
                        "approx_1_iv": surface_from_metrics(np.array([x]), level_1, skew_1, curv_1)[0],
                        "approx_2_iv": surface_from_metrics(np.array([x]), level_2, skew_2, curv_2)[0],
                        "c_xi": coeffs.c_xi,
                        "c_xixi": coeffs.c_xixi,
                        "d_term": coeffs.d_term,
                        "level_1": level_1,
                        "skew_1": skew_1,
                        "curvature_1": curv_1,
                        "level_2": level_2,
                        "skew_2": skew_2,
                        "curvature_2": curv_2,
                    }
                )
    return rows


def merge_metrics(mc_rows: List[Dict[str, object]], approx_rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    approx_lookup = {
        (r["model"], round(float(r["maturity"]), 8), round(float(r["log_moneyness"]), 8)): r for r in approx_rows
    }
    rows: List[Dict[str, object]] = []
    for r in mc_rows:
        key = (r["model"], round(float(r["maturity"]), 8), round(float(r["log_moneyness"]), 8))
        a = approx_lookup[key]
        row = dict(r)
        row.update({k: a[k] for k in ["approx_1_iv", "approx_2_iv", "c_xi", "c_xixi", "d_term"]})
        row["err_1"] = row["approx_1_iv"] - row["mc_iv"]
        row["err_2"] = row["approx_2_iv"] - row["mc_iv"]
        rows.append(row)
    return rows


def summary_metric_rows(surface_rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for model in sorted({r["model"] for r in surface_rows}):
        for maturity in MATURITIES:
            group = [r for r in surface_rows if r["model"] == model and abs(float(r["maturity"]) - maturity) < 1e-10]
            xs = np.array([float(r["log_moneyness"]) for r in group])
            mc = np.array([float(r["mc_iv"]) for r in group])
            a1 = np.array([float(r["approx_1_iv"]) for r in group])
            a2 = np.array([float(r["approx_2_iv"]) for r in group])
            level_mc, skew_mc, curv_mc = fit_quadratic_iv(xs[ATM_FIT_MASK], mc[ATM_FIT_MASK])
            level_1, skew_1, curv_1 = fit_quadratic_iv(xs[ATM_FIT_MASK], a1[ATM_FIT_MASK])
            level_2, skew_2, curv_2 = fit_quadratic_iv(xs[ATM_FIT_MASK], a2[ATM_FIT_MASK])
            near = ATM_FIT_MASK
            rows.append(
                {
                    "model": model,
                    "maturity": maturity,
                    "mc_level": level_mc,
                    "mc_skew": skew_mc,
                    "mc_curvature": curv_mc,
                    "approx_1_level": level_1,
                    "approx_1_skew": skew_1,
                    "approx_1_curvature": curv_1,
                    "approx_2_level": level_2,
                    "approx_2_skew": skew_2,
                    "approx_2_curvature": curv_2,
                    "rmse_1_near_atm": float(np.sqrt(np.mean((a1[near] - mc[near]) ** 2))),
                    "rmse_2_near_atm": float(np.sqrt(np.mean((a2[near] - mc[near]) ** 2))),
                    "rmse_1_full": float(np.sqrt(np.mean((a1 - mc) ** 2))),
                    "rmse_2_full": float(np.sqrt(np.mean((a2 - mc) ** 2))),
                }
            )
    return rows


def short_indicator_rows() -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    heston_spec = MODELS[0]
    lognormal_spec = MODELS[1]
    rho_heston = heston_spec.rho
    rho_lognormal = lognormal_spec.rho
    heston_sigma = heston_spec.vol_of_var
    lognormal_nu = lognormal_spec.nu
    for sigma0 in [0.15, 0.20, 0.30]:
        h_skew = rho_heston * heston_sigma / (4.0 * sigma0)
        h_curv = (2.0 - 5.0 * rho_heston * rho_heston) * (heston_sigma / (2.0 * sigma0)) ** 2 / (6.0 * sigma0)
        ln_skew = rho_lognormal * lognormal_nu / 2.0
        ln_curv = (2.0 - 3.0 * rho_lognormal * rho_lognormal) * lognormal_nu * lognormal_nu / (6.0 * sigma0)
        rows.append(
            {
                "model": "Heston short limit",
                "sigma0": sigma0,
                "short_skew": h_skew,
                "short_curvature": h_curv,
            }
        )
        rows.append(
            {
                "model": "Lognormal short limit",
                "sigma0": sigma0,
                "short_skew": ln_skew,
                "short_curvature": ln_curv,
            }
        )
    return rows


def bergomi_term_structure_rows() -> List[Dict[str, object]]:
    terms = np.array([1 / 12, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0])
    rows: List[Dict[str, object]] = []
    base_1y_rr = -2.0 * BERGOMI_BASE.first_order_skew_formula(1.0) * math.log(1.05 / 0.95) / 2.0
    for t in terms:
        c1, c2 = BERGOMI_BASE.factor_contributions(float(t))
        skew = c1 + c2
        rr_95_105 = -2.0 * skew * math.log(1.05 / 0.95) / 2.0
        power_baseline = base_1y_rr / math.sqrt(t)
        rows.append(
            {
                "maturity": float(t),
                "skew": skew,
                "rr_95_105": rr_95_105,
                "short_factor_contribution": c1,
                "long_factor_contribution": c2,
                "power_law_baseline": power_baseline,
            }
        )
    return rows


def equal_skew_variant_rows() -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for spec in [BERGOMI_BASE, BERGOMI_HIGH_NU_SAME_SKEW]:
        for maturity in [0.25, 1.0]:
            coeffs = spec.coeffs(maturity)
            level, skew, curvature = coeffs.second_order_metrics(maturity)
            for x in LOG_MONEYNESS:
                iv = surface_from_metrics(np.array([x]), level, skew, curvature)[0]
                rows.append(
                    {
                        "scenario": spec.label,
                        "maturity": maturity,
                        "log_moneyness": float(x),
                        "iv_second_order": iv,
                        "level": level,
                        "skew": skew,
                        "curvature": curvature,
                        "nu": spec.nu,
                        "rho_s1": spec.rho_s1,
                        "rho_s2": spec.rho_s2,
                        "c_xi": coeffs.c_xi,
                        "c_xixi": coeffs.c_xixi,
                    }
                )
    return rows


def plot_surface_comparison(surface_rows: List[Dict[str, object]]) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=False)
    for ax, model in zip(axes, [m.label for m in MODELS]):
        group = [r for r in surface_rows if r["model"] == model and abs(float(r["maturity"]) - 1.0) < 1e-10]
        xs = np.array([float(r["log_moneyness"]) for r in group])
        order = np.argsort(xs)
        xs = xs[order]
        mc = np.array([float(r["mc_iv"]) for r in group])[order]
        a1 = np.array([float(r["approx_1_iv"]) for r in group])[order]
        a2 = np.array([float(r["approx_2_iv"]) for r in group])[order]
        ax.plot(xs, mc, "o", label="MC")
        ax.plot(xs, a1, "--", label=r"$C^{x\xi}$ only")
        ax.plot(xs, a2, "-", label=r"$C^{x\xi}+C^{\xi\xi}$")
        ax.axvspan(-0.10, 0.10, color="grey", alpha=0.12)
        ax.set_title(f"{model}, T=1Y")
        ax.set_xlabel("log(K/F)")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("implied volatility")
    axes[-1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES / "surface_approximation_vs_mc_1y.png", dpi=180)
    plt.close(fig)


def plot_error_summary(summary_rows: List[Dict[str, object]]) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.8))
    labels = []
    near_1 = []
    near_2 = []
    for r in summary_rows:
        if abs(float(r["maturity"]) - 1.0) < 1e-10:
            labels.append(str(r["model"]))
            near_1.append(float(r["rmse_1_near_atm"]) * 10000)
            near_2.append(float(r["rmse_2_near_atm"]) * 10000)
    x = np.arange(len(labels))
    width = 0.35
    ax.bar(x - width / 2, near_1, width, label=r"$C^{x\xi}$ only")
    ax.bar(x + width / 2, near_2, width, label=r"$C^{x\xi}+C^{\xi\xi}$")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("near-ATM RMSE (bp vol)")
    ax.set_title("Near-ATM approximation error, T=1Y")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES / "near_atm_rmse_1y.png", dpi=180)
    plt.close(fig)


def plot_short_indicators(rows: List[Dict[str, object]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for model in sorted({r["model"] for r in rows}):
        group = [r for r in rows if r["model"] == model]
        sigma = np.array([float(r["sigma0"]) for r in group])
        skew = np.array([float(r["short_skew"]) for r in group])
        curv = np.array([float(r["short_curvature"]) for r in group])
        order = np.argsort(sigma)
        axes[0].plot(sigma[order], skew[order], marker="o", label=model)
        axes[1].plot(sigma[order], curv[order], marker="o", label=model)
    axes[0].set_title("Short ATMF skew")
    axes[1].set_title("Short ATMF curvature")
    for ax in axes:
        ax.set_xlabel("ATM volatility")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES / "short_limit_heston_vs_lognormal.png", dpi=180)
    plt.close(fig)


def plot_term_structure(rows: List[Dict[str, object]]) -> None:
    terms = np.array([float(r["maturity"]) for r in rows])
    skew = np.array([float(r["skew"]) for r in rows])
    c1 = np.array([float(r["short_factor_contribution"]) for r in rows])
    c2 = np.array([float(r["long_factor_contribution"]) for r in rows])
    baseline = np.array([float(r["power_law_baseline"]) for r in rows])
    rr = np.array([float(r["rr_95_105"]) for r in rows])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(terms, rr * 100, marker="o", label="Bergomi 2F")
    axes[0].plot(terms, baseline * 100, "--", label="$T^{-1/2}$ baseline")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("maturity")
    axes[0].set_ylabel("95/105 risk reversal (vol pt)")
    axes[0].set_title("ATMF skew term structure")
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False)

    axes[1].plot(terms, c1, marker="o", label="short factor")
    axes[1].plot(terms, c2, marker="o", label="long factor")
    axes[1].plot(terms, skew, "k--", label="total")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("maturity")
    axes[1].set_title("Factor contributions to skew")
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES / "bergomi_2f_skew_term_structure.png", dpi=180)
    plt.close(fig)


def plot_equal_skew_variant(rows: List[Dict[str, object]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, maturity in zip(axes, [0.25, 1.0]):
        for scenario in sorted({r["scenario"] for r in rows}):
            group = [r for r in rows if r["scenario"] == scenario and abs(float(r["maturity"]) - maturity) < 1e-10]
            xs = np.array([float(r["log_moneyness"]) for r in group])
            iv = np.array([float(r["iv_second_order"]) for r in group])
            order = np.argsort(xs)
            ax.plot(xs[order], iv[order], marker="o", label=scenario)
        ax.axvline(0.0, color="grey", lw=1)
        ax.set_title(f"T={maturity:g}Y")
        ax.set_xlabel("log(K/F)")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("2nd-order implied volatility")
    axes[-1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES / "same_skew_different_vol_of_vol.png", dpi=180)
    plt.close(fig)


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    approx_rows = approximate_surface_rows()
    write_csv(RESULTS / "analytic_approximation_surface.csv", approx_rows)

    mc_rows = run_monte_carlo_surfaces()
    write_csv(RESULTS / "mc_implied_vol_surface.csv", mc_rows)

    surface_rows = merge_metrics(mc_rows, approx_rows)
    write_csv(RESULTS / "approximation_vs_mc_surface.csv", surface_rows)

    summary_rows = summary_metric_rows(surface_rows)
    write_csv(RESULTS / "approximation_error_summary.csv", summary_rows)

    short_rows = short_indicator_rows()
    write_csv(RESULTS / "short_limit_indicators.csv", short_rows)

    term_rows = bergomi_term_structure_rows()
    write_csv(RESULTS / "bergomi_2f_skew_term_structure.csv", term_rows)

    variant_rows = equal_skew_variant_rows()
    write_csv(RESULTS / "same_skew_different_vol_of_vol.csv", variant_rows)

    plot_surface_comparison(surface_rows)
    plot_error_summary(summary_rows)
    plot_short_indicators(short_rows)
    plot_term_structure(term_rows)
    plot_equal_skew_variant(variant_rows)

    print("Generated chapter 8 smile approximation experiment outputs.")


if __name__ == "__main__":
    main()
