"""Attacker strategies."""

from __future__ import annotations

import random

import numpy as np


class RandomAttacker:
    name = "random"

    def __init__(self, target_values: dict[int, float]):
        self.tids = sorted(target_values)

    def choose(self, rng: random.Random, defender_dist: np.ndarray | None = None) -> int:
        return rng.choice(self.tids)


class GreedyAttacker:
    """Attack the target maximizing value x (1 - p_defended)."""

    name = "greedy"

    def __init__(self, target_values: dict[int, float]):
        self.target_values = target_values
        self.tids = sorted(target_values)

    def choose(self, rng: random.Random, defender_dist: np.ndarray | None = None) -> int:
        if defender_dist is None:
            defender_dist = np.ones(len(self.tids)) / len(self.tids)
        best, best_score = self.tids[0], -float("inf")
        for i, t in enumerate(self.tids):
            score = self.target_values[t] * (1.0 - float(defender_dist[i]))
            if score > best_score:
                best, best_score = t, score
        return int(best)


class BestResponseAttacker:
    """Best response (minimax attacker): attack the column minimizing the
    defender's expected payoff given the defender's mixed strategy."""

    name = "best_response_minimax"

    def __init__(self, payoff_matrix: np.ndarray):
        self.A = payoff_matrix

    def choose(self, rng: random.Random, defender_dist: np.ndarray | None = None) -> int:
        if defender_dist is None:
            defender_dist = np.ones(self.A.shape[0]) / self.A.shape[0]
        col_expect = defender_dist @ self.A
        return int(np.argmin(col_expect))
