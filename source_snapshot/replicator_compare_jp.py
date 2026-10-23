#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
replicator_compare_jp.py

三者進化ゲーム・卒業論文用図表 日本語版

元の構成：
1. 図0：スピルオーバーあり／なしの比較
2. 図1：基準ケース
3. 図2：ロバスト性分析
4. 図2b：経路依存性
5. 図3：政策変数分析
6. 図4：C3 閾値分析

Run:
  python replicator_compare_jp.py --outdir figures_spillover_jp --T 200 --dt 0.05
"""

import os
import csv
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm


# ---------- 日本語フォント設定 ----------
def setup_japanese_font():
    preferred_fonts = [
        "Hiragino Sans",
        "Yu Gothic",
        "Meiryo",
        "MS Gothic",
        "Noto Sans CJK JP",
        "Noto Serif CJK JP",
        "IPAexGothic",
        "TakaoGothic",
        "DejaVu Sans",
    ]

    available_fonts = {f.name for f in fm.fontManager.ttflist}

    for font_name in preferred_fonts:
        if font_name in available_fonts:
            plt.rcParams["font.family"] = font_name
            break

    plt.rcParams["axes.unicode_minus"] = False


setup_japanese_font()


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

    # spillover terms
    R1, R2 = p.get("R1", 0.0), p.get("R2", 0.0)

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
    x = clamp01(x)
    y = clamp01(y)
    z = clamp01(z)

    piG_K1, piG_K2, piF_M1, piF_M2, piT_N1, piT_N2 = payoffs(x, y, z, p)

    dx = x * (1.0 - x) * (piG_K1 - piG_K2)
    dy = y * (1.0 - y) * (piF_M1 - piF_M2)
    dz = z * (1.0 - z) * (piT_N1 - piT_N2)

    return np.array([dx, dy, dz], dtype=float)


# ---------- integrator ----------
def simulate(x0, y0, z0, T=200.0, dt=0.05, p=None):
    steps = int(T / dt) + 1
    t = np.linspace(0.0, T, steps)

    X = np.zeros(steps)
    Y = np.zeros(steps)
    Z = np.zeros(steps)

    state = np.array([x0, y0, z0], dtype=float)

    for i in range(steps):
        X[i], Y[i], Z[i] = state

        k1 = replicator_rhs(state, p)
        k2 = replicator_rhs(state + 0.5 * dt * k1, p)
        k3 = replicator_rhs(state + 0.5 * dt * k2, p)
        k4 = replicator_rhs(state + dt * k3, p)

        state = state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        state[0] = clamp01(state[0])
        state[1] = clamp01(state[1])
        state[2] = clamp01(state[2])

    return t, X, Y, Z


# ---------- plotting ----------
def plot_xyz(t, X, Y, Z, title, outpath):
    plt.figure(figsize=(8, 6))

    plt.plot(t, X, label="x（政府：収縮政策 K1）")
    plt.plot(t, Y, label="y（企業：外国人材採用 M1）")
    plt.plot(t, Z, label="z（人材：日本移動 N1）")

    plt.xlabel("時間 t")
    plt.ylabel("戦略選択比率")
    plt.title(title)

    plt.grid(True, alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()


def plot_compare_xyz(t, X_on, Y_on, Z_on, X_off, Y_off, Z_off, title, outpath):
    """
    Compare x, y, z trajectories with spillover ON vs OFF on one figure.
    """
    plt.figure(figsize=(10, 6))

    plt.plot(t, X_on, label="x：スピルオーバーあり")
    plt.plot(t, X_off, label="x：スピルオーバーなし")
    plt.plot(t, Y_on, label="y：スピルオーバーあり")
    plt.plot(t, Y_off, label="y：スピルオーバーなし")
    plt.plot(t, Z_on, label="z：スピルオーバーあり")
    plt.plot(t, Z_off, label="z：スピルオーバーなし")

    plt.xlabel("時間 t")
    plt.ylabel("戦略選択比率")
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

    title = f"{case_name}：初期値 (x0, y0, z0)=({x0}, {y0}, {z0})"

    fname = (
        case_name
        .replace("/", "_")
        .replace(":", "_")
        .replace("：", "_")
        .replace(" ", "_")
    )

    outpng = os.path.join(outdir, f"{fname}.png")

    plot_xyz(t, X, Y, Z, title, outpng)

    print(f"\n=== {case_name} ===")
    print(f"Final state: x={X[-1]:.4f}, y={Y[-1]:.4f}, z={Z[-1]:.4f}")
    print("Tail ranges: last 30%")
    print(f"x: {xmin:.4f} ~ {xmax:.4f}   tail mean={xmean:.4f}")
    print(f"y: {ymin:.4f} ~ {ymax:.4f}   tail mean={ymean:.4f}")
    print(f"z: {zmin:.4f} ~ {zmax:.4f}   tail mean={zmean:.4f}")
    print(f"Saved figure -> {outpng}")

    return {
        "case": case_name,
        "x_final": float(X[-1]),
        "y_final": float(Y[-1]),
        "z_final": float(Z[-1]),
        "x_tail_min": xmin,
        "x_tail_max": xmax,
        "x_tail_mean": xmean,
        "y_tail_min": ymin,
        "y_tail_max": ymax,
        "y_tail_mean": ymean,
        "z_tail_min": zmin,
        "z_tail_max": zmax,
        "z_tail_mean": zmean,
        "png": outpng,
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

        rows.append({
            "C3": float(c3),
            "y_tail_mean": ymean,
            "y_tail_max": ymax,
        })

        y_means.append(ymean)
        y_maxs.append(ymax)

    csv_path = os.path.join(outdir, "scan_C3_table.csv")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["C3", "y_tail_mean", "y_tail_max"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    plt.figure(figsize=(8, 6))

    plt.plot(
        [r["C3"] for r in rows],
        y_means,
        marker="o",
        label="y の終端平均",
    )

    plt.plot(
        [r["C3"] for r in rows],
        y_maxs,
        marker="o",
        label="y の終端最大",
    )

    plt.xlabel("C3（企業の外国人材採用摩擦コスト）")
    plt.ylabel("y の統計量（終端区間）")
    plt.title("図4：C3 が外国人材採用比率 y に与える影響")

    plt.grid(True, alpha=0.4)
    plt.legend()
    plt.tight_layout()

    fig_path = os.path.join(outdir, "図4_C3閾値分析.png")
    plt.savefig(fig_path, dpi=160)
    plt.close()

    print("\n--- C3 scan saved ---")
    print(f"CSV -> {csv_path}")
    print(f"FIG -> {fig_path}")

    return csv_path, fig_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default="figures_spillover_jp", help="output directory")
    parser.add_argument("--T", type=float, default=200.0)
    parser.add_argument("--dt", type=float, default=0.05)

    args = parser.parse_args()

    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)

    # -------------------------
    # Baseline parameters
    # -------------------------
    p = {
        # Government
        "E1": 2.0,
        "E2": 1.5,
        "C1": 0.6,
        "C2": 0.2,

        # Firm
        "E3": 2.0,
        "E4": 1.8,
        "C3": 0.9,
        "C4": 0.5,

        # Talent
        "E5": 2.0,
        "E6": 1.9,
        "C5": 0.6,

        # Subsidies under K2
        "S1": 0.8,
        "S2": 0.6,

        # Spillover benefits to government under K2
        "R1": 1.2,
        "R2": 0.8,
    }

    T = args.T
    dt = args.dt

    # -------------------------
    # 図0：spillover ON vs OFF
    # -------------------------
    x0_cmp, y0_cmp, z0_cmp = 0.6, 0.3, 0.2

    p_on = dict(p)

    p_off = dict(p)
    p_off["R1"] = 0.0
    p_off["R2"] = 0.0

    t, X_on, Y_on, Z_on = simulate(
        x0_cmp, y0_cmp, z0_cmp,
        T=T,
        dt=dt,
        p=p_on,
    )

    _, X_off, Y_off, Z_off = simulate(
        x0_cmp, y0_cmp, z0_cmp,
        T=T,
        dt=dt,
        p=p_off,
    )

    outpng_cmp = os.path.join(outdir, "図0_スピルオーバー有無の比較.png")

    plot_compare_xyz(
        t,
        X_on, Y_on, Z_on,
        X_off, Y_off, Z_off,
        title=f"図0：スピルオーバー有無の比較：初期値 (x0, y0, z0)=({x0_cmp}, {y0_cmp}, {z0_cmp})",
        outpath=outpng_cmp,
    )

    print(f"\nSaved comparison figure -> {outpng_cmp}")

    # -------------------------
    # 図1〜図3
    # -------------------------
    summary = []

    summary.append(
        run_case(
            "図1 基準ケース（スピルオーバーあり）",
            0.6, 0.3, 0.2,
            p,
            outdir,
            T=T,
            dt=dt,
        )
    )

    summary.append(
        run_case(
            "図2 ロバスト性分析（初期 y0 高水準）",
            0.6, 0.75, 0.2,
            p,
            outdir,
            T=T,
            dt=dt,
        )
    )

    summary.append(
        run_case(
            "図2b 経路依存性",
            0.3, 0.75, 0.6,
            p,
            outdir,
            T=T,
            dt=dt,
        )
    )

    # Policy lever example: lower C3
    pA = dict(p)
    pA["C3"] = 0.6

    summary.append(
        run_case(
            "図3 政策変数分析（C3 低下）",
            0.6, 0.3, 0.2,
            pA,
            outdir,
            T=T,
            dt=dt,
        )
    )

    # -------------------------
    # Summary CSV
    # -------------------------
    summary_csv = os.path.join(outdir, "summary_tail_stats.csv")
    fields = list(summary[0].keys())

    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in summary:
            writer.writerow(row)

    print(f"\nSaved summary table -> {summary_csv}")

    # -------------------------
    # 図4：C3 threshold scan
    # -------------------------
    c3_list = np.round(np.linspace(0.3, 1.4, 23), 2)

    scan_parameter_C3(
        p,
        0.6, 0.3, 0.2,
        c3_list,
        outdir,
        T=T,
        dt=dt,
    )

    print("\nAll figures saved in:", outdir)


if __name__ == "__main__":
    main()