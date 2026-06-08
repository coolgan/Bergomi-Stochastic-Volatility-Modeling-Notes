"""
Numerical experiments for Bergomi-style volatility dynamics intuition.

The script is intentionally self-contained.  It contrasts static vanilla-smile
sensitivities with path-dependent payoffs under a lightweight stochastic
volatility model whose parameters can be read as practical "knobs":

- eta: vol of vol, also a driver of smile curvature.
- rho: spot-vol correlation, the driver of equity skew / SSR.
- kappa: mean reversion of volatility shocks, a proxy for forward-skew
  persistence.

The model is not a production calibration model.  Its purpose is to make the
pricing mechanisms in Bergomi's notes visible in a reproducible way.
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


def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_call_price(spot: float, strike: float, maturity: float, vol: float, rate: float = 0.0) -> float:
    if maturity <= 0.0:
        return max(spot - strike, 0.0)
    if vol <= 1e-12:
        forward_intrinsic = max(spot - strike * math.exp(-rate * maturity), 0.0)
        return forward_intrinsic
    sqrt_t = math.sqrt(maturity)
    d1 = (math.log(spot / strike) + (rate + 0.5 * vol * vol) * maturity) / (vol * sqrt_t)
    d2 = d1 - vol * sqrt_t
    return spot * norm_cdf(d1) - strike * math.exp(-rate * maturity) * norm_cdf(d2)


def implied_vol_call(price: float, spot: float, strike: float, maturity: float, rate: float = 0.0) -> float:
    intrinsic = max(spot - strike * math.exp(-rate * maturity), 0.0)
    upper = spot
    price = min(max(price, intrinsic + 1e-14), upper - 1e-14)
    lo, hi = 1e-4, 4.0
    for _ in range(90):
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


def static_smile_vol(strike: float, forward: float, atm: float, skew: float, curvature: float) -> float:
    x = math.log(strike / forward)
    return max(0.03, atm + skew * x + 0.5 * curvature * x * x)


@dataclass(frozen=True)
class StaticSmileScenario:
    name: str
    atm: float
    skew: float
    curvature: float


@dataclass(frozen=True)
class DynamicScenario:
    name: str
    label: str
    theta_vol: float
    eta: float
    rho: float
    kappa: float
    description: str


def run_static_smile_experiment() -> List[Dict[str, object]]:
    scenarios = [
        StaticSmileScenario("flat_bs", 0.20, 0.00, 0.00),
        StaticSmileScenario("equity_skew", 0.20, -0.30, 0.60),
        StaticSmileScenario("steeper_skew", 0.20, -0.55, 0.60),
        StaticSmileScenario("higher_curvature", 0.20, -0.30, 1.80),
    ]
    strikes = [80, 90, 100, 110, 120]
    rows: List[Dict[str, object]] = []
    for sc in scenarios:
        for k in strikes:
            vol = static_smile_vol(k, 100.0, sc.atm, sc.skew, sc.curvature)
            rows.append(
                {
                    "scenario": sc.name,
                    "maturity": 1.0,
                    "strike": k,
                    "log_moneyness": math.log(k / 100.0),
                    "input_iv": vol,
                    "call_price": bs_call_price(100.0, k, 1.0, vol),
                }
            )
    write_csv(RESULTS / "static_vanilla_sensitivity.csv", rows)

    plt.figure(figsize=(7.2, 4.6))
    for sc in scenarios:
        vols = [static_smile_vol(k, 100.0, sc.atm, sc.skew, sc.curvature) for k in strikes]
        plt.plot(strikes, np.array(vols) * 100.0, marker="o", label=sc.name)
    plt.xlabel("Strike")
    plt.ylabel("Implied volatility (%)")
    plt.title("Static vanilla smile: skew and curvature are cross-sectional inputs")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES / "static_vanilla_smile_sensitivity.png", dpi=180)
    plt.close()
    return rows


def simulate_dynamic_scenario(
    scenario: DynamicScenario,
    n_paths: int = 70_000,
    seed: int = 20260605,
    years: float = 2.0,
    steps_per_year: int = 252,
) -> Dict[str, object]:
    dt = 1.0 / steps_per_year
    n_steps = int(round(years * steps_per_year))
    t_grid = np.arange(n_steps + 1) * dt
    theta_var = scenario.theta_vol * scenario.theta_vol
    rng = np.random.default_rng(seed)

    log_s = np.zeros(n_paths, dtype=np.float64)
    x = np.zeros(n_paths, dtype=np.float64)

    spot_3m = None
    spot_1y = None
    spot_2y = None
    spot_t1 = None
    max_1y = np.ones(n_paths, dtype=np.float64)

    ki = np.zeros(n_paths, dtype=bool)
    ko = np.zeros(n_paths, dtype=bool)
    ko_time = np.full(n_paths, np.nan, dtype=np.float64)
    snowball_payoff = np.zeros(n_paths, dtype=np.float64)

    monthly_obs = set(int(round(m / 12.0 * steps_per_year)) for m in range(3, 25))
    t1_step = int(round(1.0 * steps_per_year))
    t_3m_step = int(round(0.25 * steps_per_year))
    t_1y_step = int(round(1.0 * steps_per_year))
    t_2y_step = int(round(2.0 * steps_per_year))

    a = math.exp(-scenario.kappa * dt) if scenario.kappa > 0 else 1.0
    if scenario.kappa > 0 and scenario.eta > 0:
        ou_std = scenario.eta * math.sqrt((1.0 - math.exp(-2.0 * scenario.kappa * dt)) / (2.0 * scenario.kappa))
    else:
        ou_std = scenario.eta * math.sqrt(dt)

    for step in range(1, n_steps + 1):
        t_prev = t_grid[step - 1]
        if scenario.kappa > 0:
            var_x = scenario.eta * scenario.eta / (2.0 * scenario.kappa) * (
                1.0 - math.exp(-2.0 * scenario.kappa * t_prev)
            )
        else:
            var_x = scenario.eta * scenario.eta * t_prev
        inst_var = theta_var * np.exp(x - 0.5 * var_x)
        inst_var = np.clip(inst_var, 1e-6, 3.0)

        z_spot = rng.standard_normal(n_paths)
        z_ind = rng.standard_normal(n_paths)
        z_vol = scenario.rho * z_spot + math.sqrt(max(1.0 - scenario.rho * scenario.rho, 0.0)) * z_ind

        log_s += -0.5 * inst_var * dt + np.sqrt(inst_var * dt) * z_spot
        x = a * x + ou_std * z_vol
        spot = np.exp(log_s)

        if step <= t_1y_step:
            max_1y = np.maximum(max_1y, spot)
        ki |= spot <= 0.75

        if step in monthly_obs:
            newly_ko = (~ko) & (spot >= 1.03)
            if np.any(newly_ko):
                tau = step * dt
                snowball_payoff[newly_ko] = 1.0 + 0.15 * tau
                ko_time[newly_ko] = tau
                ko[newly_ko] = True

        if step == t_3m_step:
            spot_3m = spot.copy()
        if step == t1_step:
            spot_t1 = spot.copy()
        if step == t_1y_step:
            spot_1y = spot.copy()
        if step == t_2y_step:
            spot_2y = spot.copy()

    assert spot_3m is not None and spot_1y is not None and spot_2y is not None and spot_t1 is not None

    no_ko = ~ko
    no_ko_no_ki = no_ko & (~ki)
    no_ko_ki = no_ko & ki
    snowball_payoff[no_ko_no_ki] = 1.0 + 0.15 * years
    snowball_payoff[no_ko_ki] = np.minimum(spot_2y[no_ko_ki], 1.0)

    return {
        "scenario": scenario,
        "spot_3m": spot_3m,
        "spot_1y": spot_1y,
        "spot_2y": spot_2y,
        "spot_t1": spot_t1,
        "max_1y": max_1y,
        "snowball_payoff": snowball_payoff,
        "ki": ki,
        "ko": ko,
        "ko_time": ko_time,
    }


def summarize_dynamic_outputs(sim: Dict[str, object]) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], Dict[str, object]]:
    scenario: DynamicScenario = sim["scenario"]  # type: ignore[assignment]
    vanilla_rows: List[Dict[str, object]] = []
    maturity_spots = [(0.25, sim["spot_3m"]), (1.0, sim["spot_1y"])]
    strikes = [80, 90, 100, 110, 120]
    for maturity, spots in maturity_spots:
        spots_arr = spots  # type: ignore[assignment]
        for strike in strikes:
            payoff = np.maximum(spots_arr - strike / 100.0, 0.0)
            price = float(np.mean(payoff))
            se = float(np.std(payoff, ddof=1) / math.sqrt(payoff.size))
            vanilla_rows.append(
                {
                    "scenario": scenario.name,
                    "label": scenario.label,
                    "maturity": maturity,
                    "strike": strike,
                    "call_price": price,
                    "mc_se": se,
                    "implied_vol": implied_vol_call(price, 1.0, strike / 100.0, maturity),
                }
            )

    ratio = sim["spot_2y"] / sim["spot_t1"]  # type: ignore[operator]
    forward_rows: List[Dict[str, object]] = []
    for strike in [0.95, 1.0, 1.05]:
        payoff = np.maximum(ratio - strike, 0.0)
        price = float(np.mean(payoff))
        se = float(np.std(payoff, ddof=1) / math.sqrt(payoff.size))
        forward_rows.append(
            {
                "scenario": scenario.name,
                "label": scenario.label,
                "reset": 1.0,
                "tenor": 1.0,
                "relative_strike": strike,
                "forward_call_price": price,
                "mc_se": se,
                "forward_implied_vol": implied_vol_call(price, 1.0, strike, 1.0),
            }
        )

    spot_1y = sim["spot_1y"]  # type: ignore[assignment]
    up_out_payoff = np.where(sim["max_1y"] < 1.20, np.maximum(spot_1y - 1.0, 0.0), 0.0)  # type: ignore[operator]
    snowball = sim["snowball_payoff"]  # type: ignore[assignment]
    ko = sim["ko"]  # type: ignore[assignment]
    ki = sim["ki"]  # type: ignore[assignment]
    ko_time = sim["ko_time"]  # type: ignore[assignment]
    spot_2y = sim["spot_2y"]  # type: ignore[assignment]
    no_ko = ~ko
    no_ko_no_ki = no_ko & (~ki)
    no_ko_ki = no_ko & ki
    avg_ko_time = float(np.nanmean(ko_time)) if np.any(ko) else float("nan")
    base_principal_leg = np.ones_like(snowball)
    ko_coupon_leg = np.where(ko, 0.15 * ko_time, 0.0)
    no_ko_no_ki_coupon_leg = np.where(no_ko_no_ki, 0.15 * 2.0, 0.0)
    ki_no_ko_loss_leg = np.where(no_ko_ki, np.maximum(1.0 - spot_2y, 0.0), 0.0)
    snowball_reconstructed = base_principal_leg + ko_coupon_leg + no_ko_no_ki_coupon_leg - ki_no_ko_loss_leg
    exotic_row = {
        "scenario": scenario.name,
        "label": scenario.label,
        "eta": scenario.eta,
        "rho": scenario.rho,
        "kappa": scenario.kappa,
        "vanilla_atm_1y_iv": next(
            row["implied_vol"] for row in vanilla_rows if row["maturity"] == 1.0 and row["strike"] == 100
        ),
        "forward_atm_iv": next(row["forward_implied_vol"] for row in forward_rows if row["relative_strike"] == 1.0),
        "forward_95_105_skew_iv_diff": next(
            row["forward_implied_vol"] for row in forward_rows if row["relative_strike"] == 0.95
        )
        - next(row["forward_implied_vol"] for row in forward_rows if row["relative_strike"] == 1.05),
        "up_out_call_price": float(np.mean(up_out_payoff)),
        "up_out_call_mc_se": float(np.std(up_out_payoff, ddof=1) / math.sqrt(up_out_payoff.size)),
        "snowball_pv": float(np.mean(snowball)),
        "snowball_mc_se": float(np.std(snowball, ddof=1) / math.sqrt(snowball.size)),
        "snowball_base_principal_leg": float(np.mean(base_principal_leg)),
        "snowball_ko_coupon_leg": float(np.mean(ko_coupon_leg)),
        "snowball_no_ko_no_ki_coupon_leg": float(np.mean(no_ko_no_ki_coupon_leg)),
        "snowball_ki_no_ko_loss_leg": float(np.mean(ki_no_ko_loss_leg)),
        "snowball_reconstructed_pv": float(np.mean(snowball_reconstructed)),
        "snowball_decomposition_error": float(np.mean(snowball - snowball_reconstructed)),
        "snowball_ko_prob": float(np.mean(ko)),
        "snowball_ki_prob": float(np.mean(ki)),
        "snowball_ki_no_ko_prob": float(np.mean(ki & (~ko))),
        "snowball_avg_ko_time": avg_ko_time,
    }
    return vanilla_rows, forward_rows, exotic_row


def run_dynamic_experiments() -> Tuple[List[Dict[str, object]], List[Dict[str, object]], List[Dict[str, object]]]:
    scenarios = [
        DynamicScenario(
            "bs_flat",
            "Flat BS",
            0.20,
            0.00,
            0.00,
            1.00,
            "constant volatility benchmark",
        ),
        DynamicScenario(
            "high_curvature_volvol",
            "High vol-of-vol, rho=0",
            0.20,
            0.90,
            0.00,
            1.20,
            "symmetric vol-of-vol; raises curvature but not equity skew",
        ),
        DynamicScenario(
            "fast_decay_skew",
            "Negative spot-vol, fast decay",
            0.20,
            0.70,
            -0.70,
            4.00,
            "local-vol-like: strong current skew but weaker forward skew",
        ),
        DynamicScenario(
            "persistent_forward_skew",
            "Negative spot-vol, persistent",
            0.20,
            0.70,
            -0.70,
            0.35,
            "Bergomi-like: volatility shocks persist into forward-start dates",
        ),
        DynamicScenario(
            "stress_spot_vol",
            "Stress negative spot-vol",
            0.20,
            1.00,
            -0.85,
            0.35,
            "large vol-of-vol and strong spot-vol correlation",
        ),
    ]

    write_csv(
        RESULTS / "scenario_params.csv",
        [
            {
                "scenario": sc.name,
                "label": sc.label,
                "theta_vol": sc.theta_vol,
                "eta": sc.eta,
                "rho": sc.rho,
                "kappa": sc.kappa,
                "description": sc.description,
            }
            for sc in scenarios
        ],
    )

    all_vanilla: List[Dict[str, object]] = []
    all_forward: List[Dict[str, object]] = []
    all_exotic: List[Dict[str, object]] = []
    for idx, scenario in enumerate(scenarios):
        print(f"Running {scenario.name} ...")
        sim = simulate_dynamic_scenario(scenario, seed=20260605 + 1000 * idx)
        vanilla_rows, forward_rows, exotic_row = summarize_dynamic_outputs(sim)
        all_vanilla.extend(vanilla_rows)
        all_forward.extend(forward_rows)
        all_exotic.append(exotic_row)

    write_csv(RESULTS / "dynamic_vanilla_iv.csv", all_vanilla)
    write_csv(RESULTS / "forward_start_iv.csv", all_forward)
    write_csv(RESULTS / "exotic_prices.csv", all_exotic)

    plot_dynamic_results(all_vanilla, all_forward, all_exotic)
    return all_vanilla, all_forward, all_exotic


def plot_dynamic_results(
    vanilla_rows: List[Dict[str, object]],
    forward_rows: List[Dict[str, object]],
    exotic_rows: List[Dict[str, object]],
) -> None:
    labels = []
    for row in exotic_rows:
        labels.append(str(row["label"]))

    plt.figure(figsize=(7.8, 4.8))
    for label in labels:
        rows = [r for r in vanilla_rows if r["label"] == label and r["maturity"] == 1.0]
        rows = sorted(rows, key=lambda r: r["strike"])
        plt.plot(
            [r["strike"] for r in rows],
            [100.0 * float(r["implied_vol"]) for r in rows],
            marker="o",
            label=label,
        )
    plt.xlabel("Strike")
    plt.ylabel("1Y implied volatility (%)")
    plt.title("Dynamic model: 1Y vanilla implied-volatility smiles")
    plt.grid(alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGURES / "dynamic_vanilla_1y_smiles.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7.8, 4.8))
    for label in labels:
        rows = [r for r in forward_rows if r["label"] == label]
        rows = sorted(rows, key=lambda r: r["relative_strike"])
        plt.plot(
            [100.0 * float(r["relative_strike"]) for r in rows],
            [100.0 * float(r["forward_implied_vol"]) for r in rows],
            marker="o",
            label=label,
        )
    plt.xlabel("Forward-start relative strike (%)")
    plt.ylabel("1Yx1Y forward-start implied volatility (%)")
    plt.title("Forward-start smiles expose forward skew persistence")
    plt.grid(alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGURES / "forward_start_smiles.png", dpi=180)
    plt.close()

    x = np.arange(len(labels))
    width = 0.36
    barrier = [float(r["up_out_call_price"]) for r in exotic_rows]
    snowball = [float(r["snowball_pv"]) for r in exotic_rows]
    plt.figure(figsize=(8.8, 4.8))
    plt.bar(x - width / 2, barrier, width, label="1Y 100/120 up-and-out call")
    plt.bar(x + width / 2, snowball, width, label="2Y monthly snowball PV")
    plt.xticks(x, labels, rotation=18, ha="right")
    plt.ylabel("PV per unit notional")
    plt.title("Path-dependent products amplify volatility-dynamics assumptions")
    plt.grid(axis="y", alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGURES / "exotic_price_comparison.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8.8, 4.8))
    ko = [100.0 * float(r["snowball_ko_prob"]) for r in exotic_rows]
    ki_no_ko = [100.0 * float(r["snowball_ki_no_ko_prob"]) for r in exotic_rows]
    plt.bar(x - width / 2, ko, width, label="KO probability")
    plt.bar(x + width / 2, ki_no_ko, width, label="KI and no-KO probability")
    plt.xticks(x, labels, rotation=18, ha="right")
    plt.ylabel("Probability (%)")
    plt.title("Snowball event probabilities under different spot-vol dynamics")
    plt.grid(axis="y", alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGURES / "snowball_event_probabilities.png", dpi=180)
    plt.close()


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    run_static_smile_experiment()
    run_dynamic_experiments()
    print(f"Done. Results saved to {RESULTS}")
    print(f"Figures saved to {FIGURES}")


if __name__ == "__main__":
    main()
