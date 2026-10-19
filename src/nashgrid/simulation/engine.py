"""Monte Carlo evaluation of defender/attacker strategy pairings."""

from __future__ import annotations

import random
from typing import Any

import numpy as np

from nashgrid.core.episode import run_episode
from nashgrid.core.grid import Grid


def monte_carlo(grid: Grid, defender, attacker, n_episodes: int, seed: int,
                intercept_reward: float, max_steps: int = 30,
                attacker_speed: int = 1) -> dict[str, Any]:
    """Run n_episodes of (defender, attacker) and return measured statistics.

    attacker.choose receives the defender's patrol distribution (when the
    attacker strategy uses it); defender strategies may be randomized.
    """
    if n_episodes < 1:
        raise ValueError("n_episodes must be >= 1")
    tids = sorted(grid.targets)

    if hasattr(defender, "probs"):
        defender_dist = defender.probs
    elif hasattr(defender, "strategy"):
        defender_dist = defender.strategy
    else:
        defender_dist = np.ones(len(tids)) / len(tids)

    successes = 0
    intercepts = 0
    payoffs_d = []
    payoffs_a = []
    for ep in range(n_episodes):
        rng = random.Random(seed * 1_000_003 + ep)
        d_target = defender.choose(rng)
        a_target = attacker.choose(rng, defender_dist)
        res = run_episode(grid, d_target, a_target, intercept_reward,
                          max_steps=max_steps, attacker_speed=attacker_speed, seed=ep)
        intercepts += int(res.intercepted)
        successes += int(not res.intercepted)
        payoffs_d.append(res.defender_payoff)
        payoffs_a.append(res.attacker_payoff)

    pd = np.array(payoffs_d)
    pa = np.array(payoffs_a)
    return {
        "n_episodes": n_episodes,
        "attacker_success_rate": float(successes / n_episodes),
        "interception_rate": float(intercepts / n_episodes),
        "defender_mean_payoff": float(pd.mean()),
        "defender_payoff_ci95": float(1.96 * pd.std(ddof=1) / np.sqrt(n_episodes)) if n_episodes > 1 else 0.0,
        "attacker_mean_payoff": float(pa.mean()),
        "attacker_payoff_ci95": float(1.96 * pa.std(ddof=1) / np.sqrt(n_episodes)) if n_episodes > 1 else 0.0,
    }


def evaluate_pairing(grid: Grid, defender, attacker, n_episodes: int, seed: int,
                     intercept_reward: float, max_steps: int = 30,
                     attacker_speed: int = 1) -> dict[str, Any]:
    """Alias with a friendlier name for experiment scripts."""
    return monte_carlo(grid, defender, attacker, n_episodes, seed,
                       intercept_reward, max_steps, attacker_speed)
