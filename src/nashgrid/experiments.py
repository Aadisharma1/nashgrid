"""Experiment orchestration used by the CLI demo and scripts/run_experiments.py."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from nashgrid.agents.attacker import BestResponseAttacker, GreedyAttacker
from nashgrid.agents.defender import DeterministicDefender, MixedDefender, UniformRandomDefender
from nashgrid.analysis.voi import value_of_information
from nashgrid.core.grid import make_grid
from nashgrid.game.matrix import build_payoff_matrix, solve_zero_sum
from nashgrid.simulation.engine import monte_carlo
from nashgrid.visualization.render import (
    render_grid,
    render_strategy_dashboard,
    render_voi_plot,
)


def build_game(cfg: dict, seed: int = 7):
    gcfg = cfg["grid"]
    tcfg = cfg["targets"]
    grid = make_grid(
        gcfg["width"], gcfg["height"], gcfg["n_obstacles"], tcfg["n"],
        tcfg.get("values"), seed=seed,
    )
    A = build_payoff_matrix(grid.target_values, cfg["episode"]["intercept_reward"])
    sol = solve_zero_sum(A)
    return grid, A, sol


def run_experiments(cfg: dict, out_dir: str | Path, seed: int = 7) -> dict[str, Any]:
    """All measured comparisons. Every number below is computed by simulation."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    n_eps = cfg["monte_carlo"]["n_episodes"]
    mc_seed = cfg["monte_carlo"]["seed"]
    intercept = cfg["episode"]["intercept_reward"]
    max_steps = cfg["episode"]["max_steps"]
    a_speed = cfg["attacker"]["speed"]

    grid, A, sol = build_game(cfg, seed)
    tids = sorted(grid.targets)
    results: dict[str, Any] = {"game_value": sol["game_value"],
                               "defender_strategy": sol["defender_strategy"].tolist()}

    render_grid(grid, out / "grid.png")

    attacker = BestResponseAttacker(A)
    pairings = {
        "deterministic_vs_bestresponse": monte_carlo(
            grid, DeterministicDefender(grid.target_values), attacker,
            n_eps, mc_seed, intercept, max_steps, a_speed),
        "uniform_random_vs_bestresponse": monte_carlo(
            grid, UniformRandomDefender(grid.target_values), attacker,
            n_eps, mc_seed, intercept, max_steps, a_speed),
        "mixed_equilibrium_vs_bestresponse": monte_carlo(
            grid, MixedDefender(sol["defender_strategy"], tids), attacker,
            n_eps, mc_seed, intercept, max_steps, a_speed),
    }

    greedy = GreedyAttacker(grid.target_values)
    for name, defender in (
        ("deterministic_vs_greedy", DeterministicDefender(grid.target_values)),
        ("mixed_equilibrium_vs_greedy", MixedDefender(sol["defender_strategy"], tids)),
    ):
        pairings[name] = monte_carlo(grid, defender, greedy, n_eps, mc_seed,
                                     intercept, max_steps, a_speed)
    results["pairings"] = pairings

    size_sweep = {}
    for size in (6, 8, 10):
        cfg_variant = json.loads(json.dumps(cfg))
        cfg_variant["grid"].update({"width": size, "height": max(4, size - 2)})
        g2, A2, s2 = build_game(cfg_variant, seed)
        size_sweep[f"grid_{size}"] = monte_carlo(
            g2, MixedDefender(s2["defender_strategy"], sorted(g2.targets)),
            BestResponseAttacker(A2), n_eps, mc_seed, intercept, max_steps, a_speed)
    results["grid_size_sweep"] = size_sweep

    target_sweep = {}
    for n_t in (2, 3, 4):
        cfg_variant = json.loads(json.dumps(cfg))
        cfg_variant["targets"]["n"] = n_t
        g3, A3, s3 = build_game(cfg_variant, seed)
        target_sweep[f"targets_{n_t}"] = monte_carlo(
            g3, MixedDefender(s3["defender_strategy"], sorted(g3.targets)),
            BestResponseAttacker(A3), n_eps, mc_seed, intercept, max_steps, a_speed)
    results["target_count_sweep"] = target_sweep

    strength = {}
    for speed in (1, 2):
        strength[f"speed_{speed}"] = monte_carlo(
            grid, MixedDefender(sol["defender_strategy"], tids),
            BestResponseAttacker(A), n_eps, mc_seed, intercept, max_steps, speed)
    results["attacker_strength_sweep"] = strength

    voi = value_of_information(
        grid, MixedDefender(sol["defender_strategy"], tids),
        cfg["voi"]["p_obs_values"], cfg["voi"]["n_episodes"], mc_seed, intercept, max_steps)
    results["voi"] = voi

    render_strategy_dashboard(pairings, out / "strategy_dashboard.png")
    render_voi_plot(voi, out / "voi.png")

    (out / "experiments.json").write_text(json.dumps(results, indent=2))
    return results
