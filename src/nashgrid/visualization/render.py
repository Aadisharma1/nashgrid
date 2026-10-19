"""Matplotlib visualizations (Agg backend, headless-safe)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from nashgrid.core.grid import Grid


def render_grid(grid: Grid, out_path: str | Path, defender_pos=None, attacker_pos=None,
                title: str = "NashGrid world") -> Path:
    fig, ax = plt.subplots(figsize=(6, 5))
    mat = np.zeros((grid.height, grid.width))
    for (x, y) in grid.obstacles:
        mat[y, x] = 0.5
    ax.imshow(mat, cmap="Greys", vmin=0, vmax=1, origin="lower")
    for tid, (x, y) in grid.targets.items():
        ax.scatter(x, y, marker="*", s=350, color="goldenrod", edgecolor="black", zorder=3)
        ax.annotate(f"T{tid}(v={grid.target_values[tid]:.0f})", (x, y),
                    textcoords="offset points", xytext=(6, 6), fontsize=8)
    if defender_pos:
        ax.scatter(defender_pos[0], defender_pos[1], marker="o", s=120, color="royalblue",
                   edgecolor="black", zorder=4, label="defender")
    if attacker_pos:
        ax.scatter(attacker_pos[0], attacker_pos[1], marker="^", s=120, color="crimson",
                   edgecolor="black", zorder=4, label="attacker")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def render_strategy_dashboard(results: dict[str, dict], out_path: str | Path,
                              metric: str = "attacker_success_rate") -> Path:
    names = list(results.keys())
    means = [results[n][metric] for n in names]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(names, means, color=["#d62728", "#1f77b4", "#2ca02c", "#9467bd", "#8c564b"][:len(names)])
    ax.set_ylabel(metric.replace("_", " "))
    ax.set_title("Strategy comparison (Monte Carlo, measured)")
    ax.set_ylim(0, max(1.0, max(means) * 1.2))
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def render_voi_plot(voi_results: list[dict], out_path: str | Path) -> Path:
    p = [r["p_obs"] for r in voi_results]
    v = [r["attacker_mean_payoff"] for r in voi_results]
    e = [r["ci95"] for r in voi_results]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(p, v, yerr=e, marker="o", capsize=3)
    ax.set_xlabel("probability attacker observes defender patrol")
    ax.set_ylabel("attacker mean payoff")
    ax.set_title("Value of Information (measured)")
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out
