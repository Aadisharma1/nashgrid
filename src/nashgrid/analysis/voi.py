"""Value of Information: attacker learns the defender's patrol target with
probability p_obs before choosing what to attack. Expected attacker payoff is
measured by Monte Carlo across the p_obs sweep."""

from __future__ import annotations

import random
from typing import Any

import numpy as np

from nashgrid.core.episode import run_episode
from nashgrid.core.grid import Grid


def value_of_information(grid: Grid, defender, p_obs_values: list[float],
                         n_episodes: int, seed: int, intercept_reward: float,
                         max_steps: int = 30) -> list[dict[str, Any]]:
    """For each p_obs, measure attacker payoff with partial observation.

    With probability p_obs the attacker is told the defender's sampled target
    and best-responds (attacks the highest-value target the defender is NOT
    patrolling); otherwise it best-responds to the distribution only.
    """
    tids = sorted(grid.targets)
    values = np.array([grid.target_values[t] for t in tids])

    if hasattr(defender, "probs"):
        dist = defender.probs
    elif hasattr(defender, "strategy"):
        dist = defender.strategy
    else:
        dist = np.ones(len(tids)) / len(tids)

    results = []
    for p_obs in p_obs_values:
        if not 0.0 <= p_obs <= 1.0:
            raise ValueError("p_obs values must be in [0, 1]")
        payoffs = []
        for ep in range(n_episodes):
            rng = random.Random(seed * 7919 + ep)
            d_target = defender.choose(rng)
            if rng.random() < p_obs:
                mask = np.array([t != d_target for t in tids])
                if mask.any():
                    a_target = int(tids[int(np.argmax(values * mask))])
                else:
                    a_target = int(tids[int(np.argmax(values))])
            else:
                a_target = int(tids[int(np.argmax(values * (1 - dist)))])
            res = run_episode(grid, d_target, a_target, intercept_reward,
                              max_steps=max_steps, seed=ep)
            payoffs.append(res.attacker_payoff)
        arr = np.array(payoffs)
        results.append({
            "p_obs": float(p_obs),
            "attacker_mean_payoff": float(arr.mean()),
            "ci95": float(1.96 * arr.std(ddof=1) / np.sqrt(n_episodes)) if n_episodes > 1 else 0.0,
        })
    return results
