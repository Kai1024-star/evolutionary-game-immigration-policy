"""Regenerate English figures, source-table checks and numerical diagnostics."""
import argparse
import csv
import json
import platform
from pathlib import Path
import numpy as np
import scipy
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import root
from .models import (parameters, simulate, reference_solution, rhs, jacobian,
                     tail_statistics, corner_equilibria)

ROOT = Path(__file__).resolve().parents[1]
COLORS = ["#244b6b", "#16857a", "#c05a20"]
LABELS = ["x: restrictive policy", "y: foreign hiring", "z: migration to Japan"]
INITIALS = [(0.6, .3, .2), (.6, .75, .2), (.3, .75, .6), (.6, .3, .2)]
NAMES = ["baseline", "higher_initial_hiring", "alternative_initial_state", "lower_hiring_cost"]


def save_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def finish_plot(fig, ax, title, path, xlabel="Model time (arbitrary units)"):
    ax.set(title=title, xlabel=xlabel, ylabel="Strategy share", ylim=(-.02, 1.02))
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=.18)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def cases(model, out):
    rows, trajectories = [], []
    for name, initial in zip(NAMES, INITIALS):
        p = parameters(model)
        if name == "lower_hiring_cost":
            p["C3"] = .6 if model == "government" else .9
        t, q = simulate(initial, p, model)
        row = dict(case=name, x0=initial[0], y0=initial[1], z0=initial[2])
        row.update({f"{label}_final": float(q[-1, i]) for i, label in enumerate("xyz")})
        row.update(tail_statistics(q))
        rows.append(row)
        trajectories.append((t, q))
    save_csv(out / f"{model}_cases.csv", rows)
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.8), sharex=True, sharey=True)
    for ax, name, (t, q) in zip(axes.flat, NAMES, trajectories):
        for j in range(3):
            ax.plot(t, q[:, j], color=COLORS[j], label=LABELS[j], lw=1.8)
        ax.set(title=name.replace("_", " ").capitalize(), ylim=(-.02, 1.02))
        ax.grid(alpha=.18)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.supxlabel("Model time (arbitrary units)")
    fig.supylabel("Strategy share")
    fig.suptitle(f"{model.capitalize()}-spillover specification: initial-state and policy cases")
    fig.tight_layout()
    fig.savefig(out / "figures" / f"{model}_cases.png", dpi=180)
    plt.close(fig)
    return rows


def scan(model, key, grid, out):
    rows = []
    for value in grid:
        p = parameters(model, **{key: float(value)})
        _, q = simulate(INITIALS[0], p, model)
        stats = tail_statistics(q)
        rows.append({key: float(value), "y_tail_mean": stats["y_tail_mean"],
                     "y_tail_max": stats["y_tail_max"]})
    save_csv(out / f"{model}_{key}_scan.csv", rows)
    fig, ax = plt.subplots(figsize=(7.7, 4.6))
    for name, color in [("y_tail_mean", COLORS[1]), ("y_tail_max", COLORS[0])]:
        ax.plot(grid, [r[name] for r in rows], "o-", ms=3.5, color=color,
                label=name.replace("_", " "))
    ax.set(title=f"{model.capitalize()} specification: {key} scan",
           xlabel=f"{key} (illustrative payoff units)", ylabel="Foreign-hiring share: final 30%")
    ax.grid(alpha=.18)
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "figures" / f"{model}_{key}_scan.png", dpi=180)
    plt.close(fig)
    return rows


def check_snapshot(rows, path):
    old = list(csv.DictReader(path.open()))
    assert len(old) == len(rows)
    errors = []
    for new, archived in zip(rows, old):
        for key, value in new.items():
            if key in archived and key not in ("case",):
                try:
                    errors.append(abs(value - float(archived[key])))
                except (ValueError, TypeError):
                    continue
    worst = max(errors)
    # Firm summary values were rounded to six decimals in the original CSV.
    tolerance = 6e-7 if "firm_outputs/summary" in str(path) else 2e-8
    if worst > tolerance:
        raise AssertionError(f"Source-table mismatch in {path.name}: {worst}")
    return {"file": str(path.relative_to(ROOT)), "max_absolute_difference": worst,
            "tolerance": tolerance, "passed": True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    out = args.outdir.resolve()
    (out / "figures").mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})

    g_rows, f_rows = cases("government", out), cases("firm", out)
    g_scan = scan("government", "C3", np.round(np.linspace(.3, 1.4, 23), 2), out)
    f_c3 = scan("firm", "C3", np.round(np.linspace(.6, 1.4, 17), 2), out)
    f_s2 = scan("firm", "S2", np.round(np.linspace(0, .6, 13), 2), out)
    snapshots = ROOT / "source_snapshot"
    checks = [check_snapshot(g_rows, snapshots / "government_outputs/summary_tail_stats.csv"),
              check_snapshot(f_rows, snapshots / "firm_outputs/summary_tail_stats.csv"),
              check_snapshot(g_scan, snapshots / "government_outputs/scan_C3_table.csv"),
              check_snapshot(f_c3, snapshots / "firm_outputs/scan_C3_table.csv"),
              check_snapshot(f_s2, snapshots / "firm_outputs/scan_S2_table.csv")]

    fig, ax = plt.subplots(figsize=(8, 4.8))
    for enabled, style in [(True, "-"), (False, "--")]:
        p = parameters("government")
        if not enabled:
            p.update(R1=0., R2=0.)
        t, q = simulate(INITIALS[0], p)
        for j in range(3):
            ax.plot(t, q[:, j], style, color=COLORS[j], lw=1.7,
                    label=f"{LABELS[j]} ({'on' if enabled else 'off'})")
    finish_plot(fig, ax, "Government-benefit spillovers: on versus off",
                out / "figures/government_spillover_comparison.png")

    p = parameters()
    t, clipped = simulate(INITIALS[0], p)
    _, continuous = reference_solution(INITIALS[0], p)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax in axes:
        ax.plot(t, clipped[:, 1], label="Historical RK4, floor 1e-9", color=COLORS[1])
        ax.plot(t, continuous[:, 1], "--", label="Continuous ODE, logit DOP853", color=COLORS[0])
        ax.set(xlabel="Model time", ylabel="Foreign-hiring share y")
        ax.grid(alpha=.18)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_title("Linear scale")
    axes[1].set(yscale="log", title="Log scale: rare-strategy recovery")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("A numerical floor changes finite-horizon results")
    fig.tight_layout()
    fig.savefig(out / "figures/government_clipping_diagnostic.png", dpi=180)
    plt.close(fig)

    diagnostics = []
    for model in ("government", "firm"):
        p = parameters(model)
        target = (np.array([0., 1., 1.]) if model == "government" else
                  root(lambda q: rhs(q, p, model), [.0003, .0004, .003]).x)
        eig = np.linalg.eigvals(jacobian(target, p, model))
        if np.max(np.abs(rhs(target, p, model))) > 1e-9:
            raise AssertionError("Equilibrium residual too large")
        levels = []
        for step in (.05, .025, .0125):
            _, old_q = simulate(INITIALS[0], p, model, dt=step)
            _, ref_q = reference_solution(INITIALS[0], p, model, dt=step)
            levels.append({"step": step, "max_path_difference": float(np.max(np.abs(old_q-ref_q)))})
        _, source_q = simulate(INITIALS[0], p, model)
        _, continuous_q = reference_solution(INITIALS[0], p, model)
        diagnostics.append(dict(model=model, equilibrium=target.tolist(),
                                eigenvalue_real_parts=eig.real.tolist(),
                                eigenvalue_imaginary_parts=eig.imag.tolist(),
                                equilibrium_residual=float(np.max(np.abs(rhs(target, p, model)))),
                                source_final=source_q[-1].tolist(),
                                continuous_final=continuous_q[-1].tolist(),
                                source_vs_DOP853=levels))
    floor_scan = []
    for floor in [1e-6, 1e-9, 1e-18, 1e-30, 1e-60]:
        _, q = simulate(INITIALS[0], parameters(), eps=floor)
        floor_scan.append(dict(floor=floor, y_final=float(q[-1, 1])))
    save_csv(out / "government_floor_sensitivity.csv", floor_scan)
    (out / "corner_equilibria.json").write_text(json.dumps(corner_equilibria(parameters()), indent=2))
    report = {"release_date": "2026-10-23", "T": 200., "dt": .05, "tail_fraction": .30,
              "source_table_checks": checks, "numerical_diagnostics": diagnostics,
              "floor_sensitivity": floor_scan,
              "interpretation": "Source-table reproduction does not imply convergence to the unclipped ODE. See the clipping diagnostic.",
              "versions": {"python": platform.python_version(), "numpy": np.__version__,
                           "scipy": scipy.__version__, "matplotlib": matplotlib.__version__}}
    (out / "validation.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
