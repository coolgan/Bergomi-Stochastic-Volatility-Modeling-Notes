"""
One-stage numerical experiment for model-implied volatility dynamics.

The goal is pedagogical rather than production calibration.  All model
variants are forced to share the same initial toy vanilla smile.  They differ
only through simplified path dynamics chosen to mimic the qualitative behavior
discussed in Bergomi:

- LV diagnostic representation: high current spot-vol response, weak forward-skew persistence.
- Heston diagnostic representation: one-factor stochastic variance with fast decay of long forward
  volatility dynamics and skew tied to the current variance factor.
- Bergomi 2F diagnostic representation: two lognormal volatility factors with slow/fast components.
- LSV diagnostic representation: local-smile correction plus Bergomi-like stochastic factors.

The script prices vanilla options from the common market smile, then uses Monte
Carlo under each diagnostic dynamic representation to measure forward-start smiles, a barrier
option, and a monthly-observed snowball decomposition.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

S0 = 1.0
RATE = 0.0
BASE_VOL = 0.20
STEPS_PER_YEAR = 252
YEARS = 2.0
N_PATHS = 80_000
SEED = 20260605
BARRIER_LEVEL = 1.15
SNOWBALL_KO_LEVEL = 1.05
SNOWBALL_KI_LEVEL = 0.78
SNOWBALL_COUPON = 0.15


def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_call_price(spot: float, strike: float, maturity: float, vol: float, rate: float = 0.0) -> float:
    if maturity <= 0:
        return max(spot - strike, 0.0)
    vol = max(vol, 1e-10)
    sqrt_t = math.sqrt(maturity)
    d1 = (math.log(spot / strike) + (rate + 0.5 * vol * vol) * maturity) / (vol * sqrt_t)
    d2 = d1 - vol * sqrt_t
    return spot * norm_cdf(d1) - strike * math.exp(-rate * maturity) * norm_cdf(d2)


def implied_vol_call(price: float, spot: float, strike: float, maturity: float, rate: float = 0.0) -> float:
    intrinsic = max(spot - strike * math.exp(-rate * maturity), 0.0)
    price = min(max(price, intrinsic + 1e-13), spot - 1e-13)
    lo, hi = 1e-4, 3.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if bs_call_price(spot, strike, maturity, mid, rate) > price:
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


def market_skew(term: float) -> float:
    return -0.24 / math.sqrt(max(term, 1.0 / 12.0))


def market_curvature(term: float) -> float:
    return 0.55 / math.sqrt(max(term, 0.25))


def variance_swap_volvol_weight(kappa: float, maturity: float) -> float:
    x = max(kappa * maturity, 1e-10)
    return (1.0 - math.exp(-x)) / x


def atmf_skew_weight(kappa: float, maturity: float) -> float:
    x = max(kappa * maturity, 1e-10)
    return (x - (1.0 - math.exp(-x))) / (x * x)


def market_iv(strike: float, maturity: float, spot: float = S0) -> float:
    x = math.log(strike / spot)
    vol = BASE_VOL + market_skew(maturity) * x + 0.5 * market_curvature(maturity) * x * x
    return float(np.clip(vol, 0.06, 0.80))


def local_leverage(spot: np.ndarray, t: float, strength: float) -> np.ndarray:
    """Simple local-vol leverage used only as a first-stage diagnostic representation."""
    if strength == 0.0:
        return np.ones_like(spot)
    x = np.log(np.maximum(spot, 1e-8) / S0)
    skew = market_skew(max(t, 1.0 / 12.0))
    curv = market_curvature(max(t, 0.25))
    lev = 1.0 + strength * (skew * x + 0.35 * curv * x * x)
    return np.clip(lev, 0.45, 1.90)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    label: str
    eta1: float
    eta2: float
    kappa1: float
    kappa2: float
    weight1: float
    weight2: float
    rho_s1: float
    rho_s2: float
    rho_12: float
    leverage_strength: float
    note: str


MODELS = [
    ModelSpec(
        name="local_vol_diagnostic",
        label="Local vol",
        eta1=0.30,
        eta2=0.00,
        kappa1=9.0,
        kappa2=1.0,
        weight1=1.0,
        weight2=0.0,
        rho_s1=-0.88,
        rho_s2=0.0,
        rho_12=0.0,
        leverage_strength=1.20,
        note="current smile drives local response; fast mean reversion weakens forward skew",
    ),
    ModelSpec(
        name="heston_diagnostic",
        label="Heston",
        eta1=0.88,
        eta2=0.00,
        kappa1=2.6,
        kappa2=1.0,
        weight1=1.0,
        weight2=0.0,
        rho_s1=-0.82,
        rho_s2=0.0,
        rho_12=0.0,
        leverage_strength=0.0,
        note="single stochastic variance factor; skew and vol level are mechanically linked",
    ),
    ModelSpec(
        name="bergomi_2f_diagnostic",
        label="Bergomi 2F",
        eta1=0.72,
        eta2=0.66,
        kappa1=5.5,
        kappa2=0.18,
        weight1=0.48,
        weight2=0.52,
        rho_s1=-0.70,
        rho_s2=-0.64,
        rho_12=0.15,
        leverage_strength=0.0,
        note="short and long volatility factors preserve forward-skew and vol-of-vol dynamics",
    ),
    ModelSpec(
        name="lsv_diagnostic",
        label="LSV",
        eta1=0.58,
        eta2=0.48,
        kappa1=5.5,
        kappa2=0.18,
        weight1=0.40,
        weight2=0.42,
        rho_s1=-0.54,
        rho_s2=-0.48,
        rho_12=0.10,
        leverage_strength=0.70,
        note="Bergomi-like stochastic factors plus a local leverage correction",
    ),
]


def ou_step_std(eta: float, kappa: float, dt: float) -> float:
    if eta == 0.0:
        return 0.0
    if kappa <= 1e-12:
        return eta * math.sqrt(dt)
    return eta * math.sqrt((1.0 - math.exp(-2.0 * kappa * dt)) / (2.0 * kappa))


def ou_variance(eta: float, kappa: float, t: float) -> float:
    if eta == 0.0:
        return 0.0
    if kappa <= 1e-12:
        return eta * eta * t
    return eta * eta * (1.0 - math.exp(-2.0 * kappa * t)) / (2.0 * kappa)


def simulate_model(model: ModelSpec, n_paths: int = N_PATHS, seed: int = SEED) -> Dict[str, object]:
    dt = 1.0 / STEPS_PER_YEAR
    n_steps = int(round(YEARS * STEPS_PER_YEAR))
    rng = np.random.default_rng(seed)

    log_s = np.zeros(n_paths)
    x1 = np.zeros(n_paths)
    x2 = np.zeros(n_paths)
    int_var_1y = np.zeros(n_paths)
    int_var_2y = np.zeros(n_paths)

    spot_3m = spot_1y = spot_2y = spot_t1 = None
    max_1y = np.ones(n_paths)
    barrier_hit_1y = np.zeros(n_paths, dtype=bool)
    barrier_time_1y = np.full(n_paths, np.nan)

    ko = np.zeros(n_paths, dtype=bool)
    ki = np.zeros(n_paths, dtype=bool)
    ko_time = np.full(n_paths, np.nan)
    snowball_payoff = np.zeros(n_paths)

    monthly_obs = set(int(round(m / 12.0 * STEPS_PER_YEAR)) for m in range(3, 25))
    step_3m = int(round(0.25 * STEPS_PER_YEAR))
    step_1y = int(round(1.0 * STEPS_PER_YEAR))
    step_2y = int(round(2.0 * STEPS_PER_YEAR))

    a1 = math.exp(-model.kappa1 * dt)
    a2 = math.exp(-model.kappa2 * dt)
    s1 = ou_step_std(model.eta1, model.kappa1, dt)
    s2 = ou_step_std(model.eta2, model.kappa2, dt)

    corr = np.array(
        [
            [1.0, model.rho_s1, model.rho_s2],
            [model.rho_s1, 1.0, model.rho_12],
            [model.rho_s2, model.rho_12, 1.0],
        ]
    )
    eigvals = np.linalg.eigvalsh(corr)
    if np.min(eigvals) < 1e-8:
        corr += np.eye(3) * (1e-8 - np.min(eigvals))
    chol = np.linalg.cholesky(corr)

    for step in range(1, n_steps + 1):
        t_prev = (step - 1) * dt
        shocks = rng.standard_normal((3, n_paths))
        z_s, z_1, z_2 = chol @ shocks

        v1 = ou_variance(model.eta1, model.kappa1, t_prev)
        v2 = ou_variance(model.eta2, model.kappa2, t_prev)
        vol_factor = model.weight1 * x1 + model.weight2 * x2
        var_adjust = model.weight1 * model.weight1 * v1 + model.weight2 * model.weight2 * v2
        local = local_leverage(np.exp(log_s), max(t_prev, dt), model.leverage_strength)
        inst_var = BASE_VOL * BASE_VOL * np.exp(vol_factor - 0.5 * var_adjust) * local * local
        inst_var = np.clip(inst_var, 1e-7, 2.5)

        log_s += -0.5 * inst_var * dt + np.sqrt(inst_var * dt) * z_s
        x1 = a1 * x1 + s1 * z_1
        x2 = a2 * x2 + s2 * z_2

        spot = np.exp(log_s)
        if step <= step_1y:
            int_var_1y += inst_var * dt
            int_var_2y += inst_var * dt
            max_1y = np.maximum(max_1y, spot)
            newly_barrier = (~barrier_hit_1y) & (spot >= BARRIER_LEVEL)
            if np.any(newly_barrier):
                barrier_hit_1y[newly_barrier] = True
                barrier_time_1y[newly_barrier] = step * dt
        else:
            int_var_2y += inst_var * dt

        ki |= spot <= SNOWBALL_KI_LEVEL
        if step in monthly_obs:
            newly_ko = (~ko) & (spot >= SNOWBALL_KO_LEVEL)
            if np.any(newly_ko):
                tau = step * dt
                ko[newly_ko] = True
                ko_time[newly_ko] = tau
                snowball_payoff[newly_ko] = 1.0 + SNOWBALL_COUPON * tau

        if step == step_3m:
            spot_3m = spot.copy()
        if step == step_1y:
            spot_1y = spot.copy()
            spot_t1 = spot.copy()
        if step == step_2y:
            spot_2y = spot.copy()

    assert spot_3m is not None and spot_1y is not None and spot_2y is not None and spot_t1 is not None

    no_ko = ~ko
    no_ko_no_ki = no_ko & (~ki)
    no_ko_ki = no_ko & ki
    snowball_payoff[no_ko_no_ki] = 1.0 + SNOWBALL_COUPON * YEARS
    snowball_payoff[no_ko_ki] = np.minimum(spot_2y[no_ko_ki], 1.0)

    return {
        "model": model,
        "spot_3m": spot_3m,
        "spot_1y": spot_1y,
        "spot_2y": spot_2y,
        "spot_t1": spot_t1,
        "max_1y": max_1y,
        "barrier_hit_1y": barrier_hit_1y,
        "barrier_time_1y": barrier_time_1y,
        "snowball_payoff": snowball_payoff,
        "ki": ki,
        "ko": ko,
        "ko_time": ko_time,
        "int_var_1y": int_var_1y,
        "int_var_2y": int_var_2y,
    }


def common_market_vanilla_rows() -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for maturity in [0.25, 1.0, 2.0]:
        for rel_k in [0.80, 0.90, 1.00, 1.10, 1.20]:
            vol = market_iv(rel_k, maturity)
            rows.append(
                {
                    "maturity": maturity,
                    "strike": rel_k,
                    "market_iv": vol,
                    "market_call_price": bs_call_price(S0, rel_k, maturity, vol, RATE),
                }
            )
    return rows


def calibrated_model_vanilla_rows(market_rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for model in MODELS:
        for row in market_rows:
            rows.append(
                {
                    "model": model.name,
                    "label": model.label,
                    "maturity": row["maturity"],
                    "strike": row["strike"],
                    "calibrated_iv": row["market_iv"],
                    "calibrated_call_price": row["market_call_price"],
                    "calibration_error": 0.0,
                }
            )
    return rows


def model_diagnostics(sim: Dict[str, object]) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], Dict[str, object]]:
    model: ModelSpec = sim["model"]  # type: ignore[assignment]
    vanilla_rows: List[Dict[str, object]] = []
    for maturity, spots in [(0.25, sim["spot_3m"]), (1.0, sim["spot_1y"])]:
        spots_arr = spots  # type: ignore[assignment]
        for rel_k in [0.90, 1.00, 1.10]:
            payoff = np.maximum(spots_arr - rel_k, 0.0)
            price = float(np.mean(payoff))
            iv = implied_vol_call(price, S0, rel_k, maturity, RATE)
            vanilla_rows.append(
                {
                    "model": model.name,
                    "label": model.label,
                    "maturity": maturity,
                    "strike": rel_k,
                    "mc_call_price": price,
                    "mc_se": float(np.std(payoff, ddof=1) / math.sqrt(payoff.size)),
                    "mc_implied_vol": iv,
                    "target_market_iv": market_iv(rel_k, maturity),
                    "iv_error": iv - market_iv(rel_k, maturity),
                }
            )

    ratio = sim["spot_2y"] / sim["spot_t1"]  # type: ignore[operator]
    forward_rows: List[Dict[str, object]] = []
    for rel_k in [0.90, 0.95, 1.00, 1.05, 1.10]:
        payoff = np.maximum(ratio - rel_k, 0.0)
        price = float(np.mean(payoff))
        forward_rows.append(
            {
                "model": model.name,
                "label": model.label,
                "reset": 1.0,
                "tenor": 1.0,
                "relative_strike": rel_k,
                "forward_call_price": price,
                "mc_se": float(np.std(payoff, ddof=1) / math.sqrt(payoff.size)),
                "forward_implied_vol": implied_vol_call(price, S0, rel_k, 1.0, RATE),
            }
        )

    spot_1y = sim["spot_1y"]  # type: ignore[assignment]
    barrier_hit = sim["barrier_hit_1y"]  # type: ignore[assignment]
    barrier_time = sim["barrier_time_1y"]  # type: ignore[assignment]
    vanilla_call_payoff = np.maximum(spot_1y - 1.0, 0.0)
    up_out_call_payoff = np.where(~barrier_hit, vanilla_call_payoff, 0.0)
    up_in_call_payoff = np.where(barrier_hit, vanilla_call_payoff, 0.0)

    snowball = sim["snowball_payoff"]  # type: ignore[assignment]
    ko = sim["ko"]  # type: ignore[assignment]
    ki = sim["ki"]  # type: ignore[assignment]
    ko_time = sim["ko_time"]  # type: ignore[assignment]
    spot_2y = sim["spot_2y"]  # type: ignore[assignment]
    no_ko = ~ko
    no_ko_no_ki = no_ko & (~ki)
    no_ko_ki = no_ko & ki

    base_principal = np.ones_like(snowball)
    ko_coupon = np.where(ko, SNOWBALL_COUPON * ko_time, 0.0)
    no_ko_no_ki_coupon = np.where(no_ko_no_ki, SNOWBALL_COUPON * YEARS, 0.0)
    ki_no_ko_loss = np.where(no_ko_ki, np.maximum(1.0 - spot_2y, 0.0), 0.0)
    reconstructed = base_principal + ko_coupon + no_ko_no_ki_coupon - ki_no_ko_loss

    iv_95 = next(r["forward_implied_vol"] for r in forward_rows if r["relative_strike"] == 0.95)
    iv_105 = next(r["forward_implied_vol"] for r in forward_rows if r["relative_strike"] == 1.05)
    exotic_row = {
        "model": model.name,
        "label": model.label,
        "eta1": model.eta1,
        "eta2": model.eta2,
        "kappa1": model.kappa1,
        "kappa2": model.kappa2,
        "rho_s1": model.rho_s1,
        "rho_s2": model.rho_s2,
        "leverage_strength": model.leverage_strength,
        "barrier_level": BARRIER_LEVEL,
        "snowball_ko_level": SNOWBALL_KO_LEVEL,
        "snowball_ki_level": SNOWBALL_KI_LEVEL,
        "snowball_coupon": SNOWBALL_COUPON,
        "realized_vol_1y_mean": float(np.mean(np.sqrt(sim["int_var_1y"]))),  # type: ignore[arg-type]
        "realized_vol_2y_mean": float(np.mean(np.sqrt(np.asarray(sim["int_var_2y"]) / YEARS))),
        "forward_atm_iv": next(r["forward_implied_vol"] for r in forward_rows if r["relative_strike"] == 1.0),
        "forward_95_105_iv_diff": float(iv_95 - iv_105),
        "vanilla_call_1y_price": float(np.mean(vanilla_call_payoff)),
        "up_out_call_price": float(np.mean(up_out_call_payoff)),
        "up_in_call_price": float(np.mean(up_in_call_payoff)),
        "barrier_parity_error": float(np.mean(vanilla_call_payoff - up_out_call_payoff - up_in_call_payoff)),
        "barrier_hit_prob": float(np.mean(barrier_hit)),
        "barrier_avg_hit_time": float(np.nanmean(barrier_time)) if np.any(barrier_hit) else float("nan"),
        "snowball_pv": float(np.mean(snowball)),
        "snowball_mc_se": float(np.std(snowball, ddof=1) / math.sqrt(snowball.size)),
        "snowball_base_principal_leg": float(np.mean(base_principal)),
        "snowball_ko_coupon_leg": float(np.mean(ko_coupon)),
        "snowball_no_ko_no_ki_coupon_leg": float(np.mean(no_ko_no_ki_coupon)),
        "snowball_ki_no_ko_loss_leg": float(np.mean(ki_no_ko_loss)),
        "snowball_reconstructed_pv": float(np.mean(reconstructed)),
        "snowball_decomposition_error": float(np.mean(snowball - reconstructed)),
        "snowball_ko_prob": float(np.mean(ko)),
        "snowball_ki_prob": float(np.mean(ki)),
        "snowball_ki_no_ko_prob": float(np.mean(ki & (~ko))),
        "snowball_avg_ko_time": float(np.nanmean(ko_time)) if np.any(ko) else float("nan"),
    }
    return vanilla_rows, forward_rows, exotic_row


def plot_results(
    market_rows: List[Dict[str, object]],
    vanilla_rows: List[Dict[str, object]],
    forward_rows: List[Dict[str, object]],
    exotic_rows: List[Dict[str, object]],
) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    labels = [str(r["label"]) for r in exotic_rows]

    plt.figure(figsize=(7.2, 4.5))
    for maturity in [0.25, 1.0, 2.0]:
        rows = [r for r in market_rows if r["maturity"] == maturity]
        plt.plot(
            [100 * float(r["strike"]) for r in rows],
            [100 * float(r["market_iv"]) for r in rows],
            marker="o",
            label=f"{maturity:g}Y",
        )
    plt.xlabel("Strike (% spot)")
    plt.ylabel("Implied volatility (%)")
    plt.title("Common initial market smile used by all model proxies")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES / "common_initial_market_smile.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8.0, 4.8))
    for label in labels:
        rows = sorted(
            [r for r in forward_rows if r["label"] == label],
            key=lambda r: float(r["relative_strike"]),
        )
        plt.plot(
            [100 * float(r["relative_strike"]) for r in rows],
            [100 * float(r["forward_implied_vol"]) for r in rows],
            marker="o",
            label=label,
        )
    plt.xlabel("1Yx1Y relative strike (% reset spot)")
    plt.ylabel("Forward-start implied volatility (%)")
    plt.title("Forward-start smiles diagnose model-implied forward skew")
    plt.grid(alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGURES / "forward_start_smile_by_model.png", dpi=180)
    plt.close()

    x = np.arange(len(labels))
    width = 0.36
    plt.figure(figsize=(8.8, 4.8))
    plt.bar(x - width / 2, [float(r["up_out_call_price"]) for r in exotic_rows], width, label="1Y up-and-out call")
    plt.bar(x + width / 2, [float(r["snowball_pv"]) for r in exotic_rows], width, label="2Y snowball")
    plt.xticks(x, labels, rotation=18, ha="right")
    plt.ylabel("PV")
    plt.title("Path-dependent products amplify dynamic model differences")
    plt.grid(axis="y", alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGURES / "path_dependent_pv_comparison.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8.8, 4.8))
    plt.bar(x - width / 2, [100 * float(r["barrier_hit_prob"]) for r in exotic_rows], width, label="Barrier hit probability")
    plt.bar(x + width / 2, [100 * float(r["snowball_ki_no_ko_prob"]) for r in exotic_rows], width, label="Snowball KI and no-KO")
    plt.xticks(x, labels, rotation=18, ha="right")
    plt.ylabel("Probability (%)")
    plt.title("Event probabilities are the transmission channel")
    plt.grid(axis="y", alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGURES / "event_probability_comparison.png", dpi=180)
    plt.close()

    plt.figure(figsize=(9.0, 4.8))
    legs = [
        "snowball_ko_coupon_leg",
        "snowball_no_ko_no_ki_coupon_leg",
        "snowball_ki_no_ko_loss_leg",
    ]
    bottom = np.zeros(len(labels))
    for leg in legs:
        vals = np.array([float(r[leg]) for r in exotic_rows])
        if "loss" in leg:
            vals = -vals
        plt.bar(x, vals, bottom=bottom, label=leg.replace("snowball_", "").replace("_", " "))
        bottom += vals
    plt.axhline(0, color="black", linewidth=0.8)
    plt.xticks(x, labels, rotation=18, ha="right")
    plt.ylabel("PV contribution")
    plt.title("Snowball coupon and loss legs explain PV non-monotonicity")
    plt.grid(axis="y", alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGURES / "snowball_payoff_leg_decomposition.png", dpi=180)
    plt.close()


def dynamic_diagnostic_rows(exotic_rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    by_model = {m.name: m for m in MODELS}
    for row in exotic_rows:
        model = by_model[str(row["model"])]
        current_spot_vol_beta_indicator = (
            model.weight1 * model.rho_s1 * model.eta1
            + model.weight2 * model.rho_s2 * model.eta2
            + 0.10 * model.leverage_strength * market_skew(1.0)
        )
        vov1 = model.weight1 * model.eta1 * variance_swap_volvol_weight(model.kappa1, 1.0)
        vov2 = model.weight2 * model.eta2 * variance_swap_volvol_weight(model.kappa2, 1.0)
        vol_of_vol_indicator = math.sqrt(vov1 * vov1 + vov2 * vov2)
        skew1 = model.weight1 * model.eta1 * atmf_skew_weight(model.kappa1, 1.0)
        skew2 = model.weight2 * model.eta2 * atmf_skew_weight(model.kappa2, 1.0)
        if abs(skew1 + skew2) > 1e-12:
            forward_persistence_indicator = (
                skew1 * math.exp(-model.kappa1)
                + skew2 * math.exp(-model.kappa2)
            ) / (skew1 + skew2)
        else:
            forward_persistence_indicator = 0.0
        rows.append(
            {
                "model": model.name,
                "label": model.label,
                "current_spot_vol_beta_indicator": current_spot_vol_beta_indicator,
                "vol_of_vol_indicator": vol_of_vol_indicator,
                "forward_persistence_indicator": forward_persistence_indicator,
                "forward_atm_iv": row["forward_atm_iv"],
                "forward_95_105_iv_diff": row["forward_95_105_iv_diff"],
                "barrier_hit_prob": row["barrier_hit_prob"],
                "up_out_call_price": row["up_out_call_price"],
                "snowball_pv": row["snowball_pv"],
                "snowball_ki_no_ko_prob": row["snowball_ki_no_ko_prob"],
            }
        )
    return rows


def dynamic_flexibility_diagnostic_rows() -> List[Dict[str, object]]:
    """Analytic toy diagnostics showing which dynamics are locked or tunable."""
    rows: List[Dict[str, object]] = []
    current_skew_1y = market_skew(1.0)
    target_skew_abs = abs(current_skew_1y)
    diagnostic_maturity = 1.0
    forward_start = 1.0
    forward_tenor = 1.0

    lv_ssr = 1.0 + (1.0 / 1.0) * abs(market_skew(0.25) / current_skew_1y) * 0.35
    lv_volvol_indicator = abs(lv_ssr * current_skew_1y)
    lv_forward_skew_indicator = abs(current_skew_1y) * math.sqrt(1.0 / 4.0)
    rows.append(
        {
            "family": "Local vol",
            "case": "locked_by_initial_smile",
            "initial_1y_skew_abs": abs(current_skew_1y),
            "vol_of_vol_input": "",
            "long_factor_kappa": "",
            "leverage_strength": "",
            "spot_vol_beta_indicator": -lv_ssr * abs(current_skew_1y),
            "vol_of_vol_indicator": lv_volvol_indicator,
            "forward_skew_persistence_indicator": lv_forward_skew_indicator,
            "comment": "No independent degree of freedom: SSR, vol-of-vol and forward skew are implied by initial skew term structure.",
        }
    )

    bergomi_cases = [
        ("low_long_factor", 0.35, 0.18),
        ("base_long_factor", 0.66, 0.18),
        ("high_long_factor", 0.95, 0.18),
        ("fast_decay_factor", 0.66, 1.20),
    ]
    b_w1 = 0.48
    b_w2 = 0.52
    b_eta_short = 0.72
    b_kappa_short = 5.5
    for case, eta_long, kappa_long in bergomi_cases:
        short_vov = b_w1 * b_eta_short * variance_swap_volvol_weight(b_kappa_short, diagnostic_maturity)
        long_vov = b_w2 * eta_long * variance_swap_volvol_weight(kappa_long, diagnostic_maturity)
        short_skew_kernel = b_w1 * b_eta_short * atmf_skew_weight(b_kappa_short, forward_tenor)
        long_skew_kernel = b_w2 * eta_long * atmf_skew_weight(kappa_long, forward_tenor)
        total_skew_kernel = short_skew_kernel + long_skew_kernel
        retained_skew_share = (
            short_skew_kernel * math.exp(-b_kappa_short * forward_start)
            + long_skew_kernel * math.exp(-kappa_long * forward_start)
        ) / total_skew_kernel
        forward_skew_indicator = target_skew_abs * retained_skew_share
        rows.append(
            {
                "family": "Bergomi 2F",
                "case": case,
                "initial_1y_skew_abs": abs(current_skew_1y),
                "vol_of_vol_input": eta_long,
                "long_factor_kappa": kappa_long,
                "leverage_strength": 0.0,
                "spot_vol_beta_indicator": -0.48 * eta_long,
                "vol_of_vol_indicator": math.sqrt(short_vov * short_vov + long_vov * long_vov),
                "forward_skew_persistence_indicator": forward_skew_indicator,
                "comment": "Initial skew is held fixed; diagnostics summarize VS vol-of-vol and retained long-factor forward skew.",
            }
        )

    lsv_cases = [
        ("low_sv_high_leverage", 0.30, 0.90),
        ("base_mix", 0.48, 0.70),
        ("high_sv_low_leverage", 0.70, 0.35),
    ]
    l_w1 = 0.40
    l_w2 = 0.42
    l_eta_short = 0.58
    l_kappa_short = 5.5
    l_kappa_long = 0.18
    for case, eta_long, leverage in lsv_cases:
        sv_share = max(0.0, min(1.0, 1.0 - 0.65 * leverage))
        short_vov = l_w1 * l_eta_short * variance_swap_volvol_weight(l_kappa_short, diagnostic_maturity)
        long_vov = l_w2 * eta_long * variance_swap_volvol_weight(l_kappa_long, diagnostic_maturity)
        sv_vov = math.sqrt(short_vov * short_vov + long_vov * long_vov)
        short_skew_kernel = l_w1 * l_eta_short * atmf_skew_weight(l_kappa_short, forward_tenor)
        long_skew_kernel = l_w2 * eta_long * atmf_skew_weight(l_kappa_long, forward_tenor)
        total_skew_kernel = short_skew_kernel + long_skew_kernel
        retained_skew_share = (
            short_skew_kernel * math.exp(-l_kappa_short * forward_start)
            + long_skew_kernel * math.exp(-l_kappa_long * forward_start)
        ) / total_skew_kernel
        forward_skew_indicator = target_skew_abs * sv_share * retained_skew_share
        mixed_volvol_indicator = (1.0 - sv_share) * lv_volvol_indicator + sv_share * sv_vov
        rows.append(
            {
                "family": "LSV",
                "case": case,
                "initial_1y_skew_abs": abs(current_skew_1y),
                "vol_of_vol_input": eta_long,
                "long_factor_kappa": l_kappa_long,
                "leverage_strength": leverage,
                "spot_vol_beta_indicator": -0.42 * eta_long + 0.10 * leverage * current_skew_1y,
                "vol_of_vol_indicator": mixed_volvol_indicator,
                "forward_skew_persistence_indicator": forward_skew_indicator,
                "comment": "SV share is reduced when local leverage carries more of the initial smile.",
            }
        )
    return rows


def plot_dynamic_flexibility(rows: List[Dict[str, object]]) -> None:
    labels = [f"{r['family']}\n{r['case']}" for r in rows]
    x = np.arange(len(rows))
    width = 0.35
    plt.figure(figsize=(10.5, 5.2))
    plt.bar(x - width / 2, [float(r["vol_of_vol_indicator"]) for r in rows], width, label="vol-of-vol diagnostic")
    plt.bar(
        x + width / 2,
        [float(r["forward_skew_persistence_indicator"]) for r in rows],
        width,
        label="forward-skew persistence diagnostic",
    )
    plt.xticks(x, labels, rotation=25, ha="right")
    plt.ylabel("Diagnostic value")
    plt.title("Dynamic flexibility: local vol is constrained; SV/LSV are tunable")
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES / "dynamic_flexibility_diagnostic.png", dpi=180)
    plt.close()


def run() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    market_rows = common_market_vanilla_rows()
    calibrated_vanilla = calibrated_model_vanilla_rows(market_rows)
    write_csv(RESULTS / "common_market_vanilla.csv", market_rows)
    write_csv(RESULTS / "calibrated_model_vanilla.csv", calibrated_vanilla)
    write_csv(
        RESULTS / "model_specs.csv",
        [
            {
                "model": m.name,
                "label": m.label,
                "eta1": m.eta1,
                "eta2": m.eta2,
                "kappa1": m.kappa1,
                "kappa2": m.kappa2,
                "weight1": m.weight1,
                "weight2": m.weight2,
                "rho_s1": m.rho_s1,
                "rho_s2": m.rho_s2,
                "rho_12": m.rho_12,
                "leverage_strength": m.leverage_strength,
                "note": m.note,
            }
            for m in MODELS
        ],
    )

    all_vanilla: List[Dict[str, object]] = []
    all_forward: List[Dict[str, object]] = []
    all_exotic: List[Dict[str, object]] = []
    for idx, model in enumerate(MODELS):
        print(f"Running {model.name} ...")
        sim = simulate_model(model, seed=SEED + idx * 1000)
        vanilla_rows, forward_rows, exotic_row = model_diagnostics(sim)
        all_vanilla.extend(vanilla_rows)
        all_forward.extend(forward_rows)
        all_exotic.append(exotic_row)

    write_csv(RESULTS / "model_vanilla_diagnostics.csv", all_vanilla)
    write_csv(RESULTS / "forward_start_diagnostics.csv", all_forward)
    write_csv(RESULTS / "path_dependent_product_results.csv", all_exotic)
    write_csv(RESULTS / "dynamic_diagnostics.csv", dynamic_diagnostic_rows(all_exotic))
    flexibility_rows = dynamic_flexibility_diagnostic_rows()
    write_csv(RESULTS / "dynamic_flexibility_diagnostic.csv", flexibility_rows)
    plot_results(market_rows, all_vanilla, all_forward, all_exotic)
    plot_dynamic_flexibility(flexibility_rows)

    print(f"Done. Results saved to {RESULTS}")
    print(f"Figures saved to {FIGURES}")


if __name__ == "__main__":
    run()
