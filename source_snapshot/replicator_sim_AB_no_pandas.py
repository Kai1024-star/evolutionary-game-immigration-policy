#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
replicator_sim_no_pandas_with_gov_spillover.py

Tripartite evolutionary game (Gov-Firm-Talent) with corrected payoff logic:

x : share of government choosing K1 (contraction)
1-x: choosing K2 (expansion)

y : share of firms choosing M1 (hire foreign)
1-y: choosing M2 (hire local)

z : share of talent choosing N1 (migrate to Japan)
1-z: choosing N2 (other countries)

Core (from your text):
Gov:
  pi_G(K1) = E2 - C2
  pi_G(K2) = E1 - C1 - y*z*S1 - z*S2        (subsidies are fiscal costs)
Firm:
  pi_F(M1) = -C3 + z*E3 + (1-x)*z*S1
  pi_F(M2) = E4 - C4
Talent:
  pi_T(N1) = E5 - C5 + (1-x)*S2
  pi_T(N2) = E6

IMPORTANT FIX (so x doesn't mechanically go to 1):
Add spillover benefits to Gov under expansion K2:
  + R1*(y*z)  (industrial/tax benefit when firms hire foreign AND talent comes)
  + R2*(z)    (innovation/tax benefit from migration itself)

So:
  pi_G(K2) = E1 - C1 - y*z*S1 - z*S2 + R1*y*z + R2*z

This keeps the model economically coherent and allows policy to be sustainable.
"""

import os
import csv
import math
import argparse
import numpy as np
import matplotlib.pyplot as plt


# ---------- utility ----------
def clamp01(v: float, eps: float = 1e-9) -> float:
    if v < eps:
        return eps
    if v > 1.0 - eps:
        return 1.0 - eps
    return v


def tail_stats(arr: np.ndarray, tail_frac: float = 0.30):
    n = len(arr)
    start = int((1.0 - tail_frac) * n)
    tail = arr[start:]
    return float(np.min(tail)), float(np.max(tail)), float(np.mean(tail))


# ---------- payoffs ----------
def payoffs(x, y, z, p):
    """
    Return:
      piG_K1, piG_K2, piF_M1, piF_M2, piT_N1, piT_N2
    """
    E1, E2, C1, C2 = p["E1"], p["E2"], p["C1"], p["C2"]
    E3, E4, C3, C4 = p["E3"], p["E4"], p["C3"], p["C4"]
    E5, E6, C5 = p["E5"], p["E6"], p["C5"]
    S1, S2 = p["S1"], p["S2"]

    # spillover terms (new knobs)
    R1, R2 = p["R1"], p["R2"]

    # Government
    piG_K1 = E2 - C2
    piG_K2 = (E1 - C1) - (y * z) * S1 - z * S2 + R1 * (y * z) + R2 * z

    # Firm
    piF_M1 = (-C3) + z * E3 + (1.0 - x) * z * S1
    piF_M2 = E4 - C4

    # Talent
    piT_N1 = (E5 - C5) + (1.0 - x) * S2
    piT_N2 = E6

    return piG_K1, piG_K2, piF_M1, piF_M2, piT_N1, piT_N2


def replicator_rhs(state, p):
    x, y, z = state
    x = clamp01(x); y = clamp01(y); z = clamp01(z)

    piG_K1, piG_K2, piF_M1, piF_M2, piT_N1, piT_N2 = payoffs(x, y, z, p)

    # Replicator: dx = x(1-x)(pi(K1)-pi_bar) = x(1-x)(piK1 - piK2)
    dx = x * (1.0 - x) * (piG_K1 - piG_K2)
    dy = y * (1.0 - y) * (piF_M1 - piF_M2)
    dz = z * (1.0 - z) * (piT_N1 - piT_N2)

    return np.array([dx, dy, dz], dtype=float)


# ---------- integrator (RK4) ----------
def simulate(x0, y0, z0, T=200.0, dt=0.05, p=None):
    steps = int(T / dt) + 1
    t = np.linspace(0.0, T, steps)

    X = np.zeros(steps); Y = np.zeros(steps); Z = np.zeros(steps)
    state = np.array([x0, y0, z0], dtype=float)

    for i in range(steps):
        X[i], Y[i], Z[i] = state
        k1 = replicator_rhs(state, p)
        k2 = replicator_rhs(state + 0.5 * dt * k1, p)
        k3 = replicator_rhs(state + 0.5 * dt * k2, p)
        k4 = replicator_rhs(state + dt * k3, p)
        state = state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

        # keep inside (0,1)
        state[0] = clamp01(state[0])
        state[1] = clamp01(state[1])
        state[2] = clamp01(state[2])

    return t, X, Y, Z


# ---------- plotting ----------
def plot_xyz(t, X, Y, Z, title, outpath):
    plt.figure(figsize=(8, 6))
    plt.plot(t, X, label="x (contraction)")
    plt.plot(t, Y, label="y (hire foreign)")
    plt.plot(t, Z, label="z (migrate to Japan)")
    plt.xlabel("t")
    plt.ylabel("proportion")
    plt.title(title)
    plt.grid(True, alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()


def run_case(case_name, x0, y0, z0, p, outdir, T=200.0, dt=0.05):
    t, X, Y, Z = simulate(x0, y0, z0, T=T, dt=dt, p=p)

    xmin, xmax, xmean = tail_stats(X)
    ymin, ymax, ymean = tail_stats(Y)
    zmin, zmax, zmean = tail_stats(Z)

    title = f"{case_name} — (x0,y0,z0)=({x0},{y0},{z0})"
    fname = case_name.replace("/", "_").replace(":", "_")
    outpng = os.path.join(outdir, f"{fname}.png")

    plot_xyz(t, X, Y, Z, title, outpng)

    print(f"\n=== {case_name} ===")
    print(f"Final state: x={X[-1]:.4f}, y={Y[-1]:.4f}, z={Z[-1]:.4f}")
    print("Tail ranges (last 30%):")
    print(f"x: {xmin:.4f} ~ {xmax:.4f}   (tail mean {xmean:.4f})")
    print(f"y: {ymin:.4f} ~ {ymax:.4f}   (tail mean {ymean:.4f})")
    print(f"z: {zmin:.4f} ~ {zmax:.4f}   (tail mean {zmean:.4f})")
    print(f"Saved figure -> {outpng}")

    return {
        "case": case_name,
        "x_final": float(X[-1]),
        "y_final": float(Y[-1]),
        "z_final": float(Z[-1]),
        "x_tail_min": xmin, "x_tail_max": xmax, "x_tail_mean": xmean,
        "y_tail_min": ymin, "y_tail_max": ymax, "y_tail_mean": ymean,
        "z_tail_min": zmin, "z_tail_max": zmax, "z_tail_mean": zmean,
        "png": outpng
    }


def scan_parameter_C3(base_p, x0, y0, z0, c3_list, outdir, T=200.0, dt=0.05):
    rows = []
    y_means = []
    y_maxs = []

    for c3 in c3_list:
        p = dict(base_p)
        p["C3"] = float(c3)
        t, X, Y, Z = simulate(x0, y0, z0, T=T, dt=dt, p=p)
        ymin, ymax, ymean = tail_stats(Y)

        rows.append({"C3": float(c3), "y_tail_mean": ymean, "y_tail_max": ymax})
        y_means.append(ymean)
        y_maxs.append(ymax)

    # save csv
    csv_path = os.path.join(outdir, "scan_C3_table.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["C3", "y_tail_mean", "y_tail_max"])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # plot
    plt.figure(figsize=(8, 6))
    plt.plot([r["C3"] for r in rows], y_means, marker="o", label="tail mean of y")
    plt.plot([r["C3"] for r in rows], y_maxs, marker="o", label="tail max of y")
    plt.xlabel("C3 (firm foreign-hiring friction cost)")
    plt.ylabel("y statistic (tail segment)")
    plt.title("Threshold scan: effect of C3 on y (hire foreign)")
    plt.grid(True, alpha=0.4)
    plt.legend()
    plt.tight_layout()
    fig_path = os.path.join(outdir, "Fig_C3_threshold.png")
    plt.savefig(fig_path, dpi=160)
    plt.close()

    print("\n--- C3 scan saved ---")
    print(f"CSV -> {csv_path}")
    print(f"FIG -> {fig_path}")

    return csv_path, fig_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="figures_spillover", help="output directory")
    ap.add_argument("--T", type=float, default=200.0)
    ap.add_argument("--dt", type=float, default=0.05)
    args = ap.parse_args()

    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)

    # -------------------------
    # Baseline parameters (YOU WILL TUNE THESE)
    # -------------------------
    p = {
        # Government
        "E1": 2.0,   # expansion benefit
        "E2": 1.5,   # contraction benefit
        "C1": 0.6,   # expansion cost
        "C2": 0.2,   # contraction hidden cost

        # Firm
        "E3": 2.0,   # benefit if hire foreign AND talent comes
        "E4": 1.8,   # benefit hire local
        "C3": 0.9,   # friction cost hire foreign (always paid if choose M1)
        "C4": 0.5,   # cost hire local

        # Talent
        "E5": 2.0,   # benefit migrate Japan
        "E6": 1.9,   # benefit other countries
        "C5": 0.6,   # migration cost

        # Subsidies only under K2
        "S1": 0.8,   # subsidy to firms when (K2,M1,N1)
        "S2": 0.6,   # subsidy to talent when (K2,*,N1)

        # NEW: spillover benefits to government under expansion (K2)
        # These are the key knobs to avoid x->1 mechanically.
        "R1": 1.2,   # benefit to gov when (M1 and N1) -> y*z
        "R2": 0.8,   # benefit to gov from migration itself -> z
    }

    T = args.T
    dt = args.dt

    # -------------------------
    # Cases (same as your style)
    # -------------------------
    summary = []
    summary.append(run_case("Fig1 Baseline (spillover)", 0.6, 0.3, 0.2, p, outdir, T=T, dt=dt))
    summary.append(run_case("Fig2 Robustness (higher y0)", 0.6, 0.75, 0.2, p, outdir, T=T, dt=dt))
    summary.append(run_case("Fig2b Path dependence", 0.3, 0.75, 0.6, p, outdir, T=T, dt=dt))

    # Policy lever example: lower C3 (easier to hire foreign)
    pA = dict(p)
    pA["C3"] = 0.6
    summary.append(run_case("Fig3 Policy lever (lower C3)", 0.6, 0.3, 0.2, pA, outdir, T=T, dt=dt))

    # Save summary csv (no pandas)
    summary_csv = os.path.join(outdir, "summary_tail_stats.csv")
    fields = list(summary[0].keys())
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in summary:
            w.writerow(row)
    print(f"\nSaved summary table -> {summary_csv}")

    # Threshold scan for C3
    c3_list = np.round(np.linspace(0.3, 1.4, 23), 2)
    scan_parameter_C3(p, 0.6, 0.3, 0.2, c3_list, outdir, T=T, dt=dt)

    print("\nAll figures saved in:", outdir)


if __name__ == "__main__":
    main()