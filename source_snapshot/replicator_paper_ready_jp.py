#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
replicator_paper_ready_jp.py

卒業論文用・日本語図表版
三者進化ゲーム：政府 x、企業 y、人材 z

出力：
- 図1 基準ケース
- 図2 ロバスト性分析
- 図2b 経路依存性
- 図3 政策変数分析
- C3 閾値分析
- S2 閾値分析
- summary_tail_stats.csv
"""

import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm


# -----------------------------
# 0) 日本語フォント設定
# -----------------------------
def setup_japanese_font():
    """
    matplotlib で日本語が文字化けしないようにする。
    使用可能な日本語フォントを自動で探す。
    """
    preferred_fonts = [
        "Noto Sans CJK JP",
        "Noto Serif CJK JP",
        "Yu Gothic",
        "Hiragino Sans",
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


# -----------------------------
# 1) Parameters
# -----------------------------
PARAMS = {
    # Government payoffs
    # K1 = contraction, K2 = expansion
    "E2": 1.70,
    "C2": 0.20,
    "E1": 2.60,
    "C1": 0.35,

    # Firm payoffs
    # M1 = hire foreign, M2 = hire domestic
    "E3": 1.20,
    "E4": 1.15,
    "C3": 1.05,
    "C4": 0.60,

    # Talent payoffs
    # N1 = migrate to Japan, N2 = stay / other countries
    "E5": 1.85,
    "C5": 0.80,
    "E6": 1.35,

    # Subsidies / policy instruments
    "S1": 0.10,
    "S2": 0.22,

    # Spillover: reduces effective C3 when z is high
    "eta": 0.25,

    # Replicator speed
    "rx": 1.0,
    "ry": 1.0,
    "rz": 1.0,

    # Small mutation
    "mu_x": 0.0005,
    "mu_y": 0.0010,
    "mu_z": 0.0005,

    # Numerics
    "dt": 0.05,
    "T": 200.0,
    "tail_frac": 0.30,
    "eps_clip": 1e-9,
}


# -----------------------------
# 2) Payoff specification
# -----------------------------
def payoffs(x, y, z, p):
    """
    Return expected payoffs for each population's two strategies:
      Gov:    K1 / K2
      Firm:   M1 / M2
      Talent: N1 / N2

    x: probability of K1
    y: probability of M1
    z: probability of N1
    """
    E1, C1 = p["E1"], p["C1"]
    E2, C2 = p["E2"], p["C2"]
    E3, E4 = p["E3"], p["E4"]
    C3, C4 = p["C3"], p["C4"]
    E5, C5 = p["E5"], p["C5"]
    E6 = p["E6"]
    S1, S2 = p["S1"], p["S2"]
    eta = p["eta"]

    # Spillover effect on foreign-hiring friction
    C3_eff = max(C3 - eta * z, 0.0)

    # Government
    pi_K1 = E2 - C2
    pi_K2 = (E1 - C1) - (y * S1) - (z * S2)

    # Firm
    pi_M1 = (E4 * 0.15) + (z * E3) + ((1 - x) * S1) - C3_eff
    pi_M2 = E4 - C4

    # Talent
    pi_N1 = (E5 - C5) + ((1 - x) * S2)
    pi_N2 = E6

    return (pi_K1, pi_K2), (pi_M1, pi_M2), (pi_N1, pi_N2)


# -----------------------------
# 3) Replicator dynamics with mutation
# -----------------------------
def replicator_step(x, y, z, p):
    (pi_K1, pi_K2), (pi_M1, pi_M2), (pi_N1, pi_N2) = payoffs(x, y, z, p)

    # Average payoffs
    piG = x * pi_K1 + (1 - x) * pi_K2
    piF = y * pi_M1 + (1 - y) * pi_M2
    piT = z * pi_N1 + (1 - z) * pi_N2

    # Standard replicator dynamics
    dx = p["rx"] * x * (pi_K1 - piG)
    dy = p["ry"] * y * (pi_M1 - piF)
    dz = p["rz"] * z * (pi_N1 - piT)

    # Small mutation / trembling hand
    dx += p["mu_x"] * (0.5 - x)
    dy += p["mu_y"] * (0.5 - y)
    dz += p["mu_z"] * (0.5 - z)

    # Euler update
    x2 = x + p["dt"] * dx
    y2 = y + p["dt"] * dy
    z2 = z + p["dt"] * dz

    # Clip to (0, 1)
    eps = p["eps_clip"]
    x2 = min(max(x2, eps), 1 - eps)
    y2 = min(max(y2, eps), 1 - eps)
    z2 = min(max(z2, eps), 1 - eps)

    return x2, y2, z2


def simulate(x0, y0, z0, p):
    n = int(p["T"] / p["dt"]) + 1
    t = np.linspace(0, p["T"], n)

    xs = np.zeros(n)
    ys = np.zeros(n)
    zs = np.zeros(n)

    x, y, z = x0, y0, z0
    xs[0], ys[0], zs[0] = x, y, z

    for i in range(1, n):
        x, y, z = replicator_step(x, y, z, p)
        xs[i], ys[i], zs[i] = x, y, z

    return t, xs, ys, zs


def tail_stats(arr, tail_frac):
    n = len(arr)
    start = int((1 - tail_frac) * n)
    tail = arr[start:]
    return float(np.min(tail)), float(np.max(tail)), float(np.mean(tail))


# -----------------------------
# 4) Plot helpers
# -----------------------------
def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def plot_paths(t, xs, ys, zs, title, outpath):
    plt.figure(figsize=(10, 7))

    plt.plot(t, xs, label="x（政府：収縮政策 K1）")
    plt.plot(t, ys, label="y（企業：外国人材採用 M1）")
    plt.plot(t, zs, label="z（人材：日本移動 N1）")

    plt.xlabel("時間 t")
    plt.ylabel("戦略選択比率")
    plt.title(title)

    plt.grid(True, alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def write_csv(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row in rows:
            writer.writerow(row)


# -----------------------------
# 5) Main experiments
# -----------------------------
def run_case(tag, x0, y0, z0, p, figdir):
    t, xs, ys, zs = simulate(x0, y0, z0, p)

    x_min, x_max, x_mean = tail_stats(xs, p["tail_frac"])
    y_min, y_max, y_mean = tail_stats(ys, p["tail_frac"])
    z_min, z_max, z_mean = tail_stats(zs, p["tail_frac"])

    print(f"\n=== {tag} — (x0,y0,z0)=({x0},{y0},{z0}) ===")
    print(f"Final state: x={xs[-1]:.4f}, y={ys[-1]:.4f}, z={zs[-1]:.4f}")
    print(f"Tail ranges: last {int(p['tail_frac'] * 100)}%")
    print(f"x: {x_min:.4f} ~ {x_max:.4f}   tail mean={x_mean:.4f}")
    print(f"y: {y_min:.4f} ~ {y_max:.4f}   tail mean={y_mean:.4f}")
    print(f"z: {z_min:.4f} ~ {z_max:.4f}   tail mean={z_mean:.4f}")

    safe_name = "".join(
        c if c.isalnum() or c in " _-().＝" else "_"
        for c in tag
    )
    outfig = os.path.join(figdir, f"{safe_name}.png")

    plot_paths(t, xs, ys, zs, tag, outfig)
    print(f"Saved figure -> {outfig}")

    return {
        "tag": tag,
        "x0": x0,
        "y0": y0,
        "z0": z0,
        "x_final": float(xs[-1]),
        "y_final": float(ys[-1]),
        "z_final": float(zs[-1]),
        "x_tail_mean": x_mean,
        "y_tail_mean": y_mean,
        "z_tail_mean": z_mean,
        "x_tail_min": x_min,
        "x_tail_max": x_max,
        "y_tail_min": y_min,
        "y_tail_max": y_max,
        "z_tail_min": z_min,
        "z_tail_max": z_max,
    }


def threshold_scan_C3(x0, y0, z0, p, figdir, c3_grid):
    rows = []
    y_means = []
    y_maxs = []

    for c3 in c3_grid:
        p2 = dict(p)
        p2["C3"] = float(c3)

        t, xs, ys, zs = simulate(x0, y0, z0, p2)
        y_min, y_max, y_mean = tail_stats(ys, p2["tail_frac"])

        rows.append([float(c3), y_mean, y_max])
        y_means.append(y_mean)
        y_maxs.append(y_max)

    csv_path = os.path.join(figdir, "scan_C3_table.csv")
    write_csv(csv_path, ["C3", "y_tail_mean", "y_tail_max"], rows)
    print(f"Saved scan table -> {csv_path}")

    plt.figure(figsize=(10, 7))
    plt.plot(c3_grid, y_means, marker="o", label="y の終端平均")
    plt.plot(c3_grid, y_maxs, marker="o", label="y の終端最大")

    plt.xlabel("C3（企業の外国人材採用摩擦コスト）")
    plt.ylabel("y の統計量（終端区間）")
    plt.title("閾値分析：C3が外国人材採用比率 y に与える影響")

    plt.grid(True, alpha=0.35)
    plt.legend()
    plt.tight_layout()

    fig_path = os.path.join(figdir, "図4_C3閾値分析.png")
    plt.savefig(fig_path, dpi=150)
    plt.close()

    print(f"Saved figure -> {fig_path}")


def threshold_scan_S2(x0, y0, z0, p, figdir, s2_grid):
    rows = []
    y_means = []
    y_maxs = []

    for s2 in s2_grid:
        p2 = dict(p)
        p2["S2"] = float(s2)

        t, xs, ys, zs = simulate(x0, y0, z0, p2)
        y_min, y_max, y_mean = tail_stats(ys, p2["tail_frac"])

        rows.append([float(s2), y_mean, y_max])
        y_means.append(y_mean)
        y_maxs.append(y_max)

    csv_path = os.path.join(figdir, "scan_S2_table.csv")
    write_csv(csv_path, ["S2", "y_tail_mean", "y_tail_max"], rows)
    print(f"Saved scan table -> {csv_path}")

    plt.figure(figsize=(10, 7))
    plt.plot(s2_grid, y_means, marker="o", label="y の終端平均")
    plt.plot(s2_grid, y_maxs, marker="o", label="y の終端最大")

    plt.xlabel("S2（拡張政策下の人材誘致補助）")
    plt.ylabel("y の統計量（終端区間）")
    plt.title("閾値分析：S2が人材移動を通じて y に与える影響")

    plt.grid(True, alpha=0.35)
    plt.legend()
    plt.tight_layout()

    fig_path = os.path.join(figdir, "図5_S2閾値分析.png")
    plt.savefig(fig_path, dpi=150)
    plt.close()

    print(f"Saved figure -> {fig_path}")


def main():
    figdir = "figures_jp"
    ensure_dir(figdir)

    # Baseline initial state
    x0, y0, z0 = 0.6, 0.3, 0.2

    summary = []

    # 図1：基準ケース
    summary.append(
        run_case(
            "図1 基準ケース",
            x0, y0, z0,
            PARAMS,
            figdir,
        )
    )

    # 図2：ロバスト性分析
    summary.append(
        run_case(
            "図2 ロバスト性分析（初期 y0 高水準）",
            0.6, 0.75, 0.2,
            PARAMS,
            figdir,
        )
    )

    # 図2b：経路依存性
    summary.append(
        run_case(
            "図2b 経路依存性",
            0.3, 0.75, 0.6,
            PARAMS,
            figdir,
        )
    )

    # 図3：政策変数分析
    p_policy = dict(PARAMS)
    p_policy["C3"] = 0.90

    summary.append(
        run_case(
            "図3 政策変数分析（C3低下＝0.90）",
            x0, y0, z0,
            p_policy,
            figdir,
        )
    )

    # 図4：C3 閾値分析
    c3_grid = np.round(np.linspace(0.6, 1.4, 17), 2)
    threshold_scan_C3(x0, y0, z0, PARAMS, figdir, c3_grid)

    # 図5：S2 閾値分析
    s2_grid = np.round(np.linspace(0.0, 0.6, 13), 2)
    threshold_scan_S2(x0, y0, z0, PARAMS, figdir, s2_grid)

    # Summary CSV
    summary_csv = os.path.join(figdir, "summary_tail_stats.csv")

    header = [
        "tag",
        "x0",
        "y0",
        "z0",
        "x_final",
        "y_final",
        "z_final",
        "x_tail_mean",
        "y_tail_mean",
        "z_tail_mean",
        "x_tail_min",
        "x_tail_max",
        "y_tail_min",
        "y_tail_max",
        "z_tail_min",
        "z_tail_max",
    ]

    rows = []
    for s in summary:
        rows.append([
            s["tag"],
            s["x0"],
            s["y0"],
            s["z0"],
            f"{s['x_final']:.6f}",
            f"{s['y_final']:.6f}",
            f"{s['z_final']:.6f}",
            f"{s['x_tail_mean']:.6f}",
            f"{s['y_tail_mean']:.6f}",
            f"{s['z_tail_mean']:.6f}",
            f"{s['x_tail_min']:.6f}",
            f"{s['x_tail_max']:.6f}",
            f"{s['y_tail_min']:.6f}",
            f"{s['y_tail_max']:.6f}",
            f"{s['z_tail_min']:.6f}",
            f"{s['z_tail_max']:.6f}",
        ])

    write_csv(summary_csv, header, rows)

    print(f"\nSaved summary table -> {summary_csv}")
    print(f"All figures saved in ./{figdir}/")


if __name__ == "__main__":
    main()