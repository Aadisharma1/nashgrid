"""Defender strategies: choose a patrol target each episode."""

from __future__ import annotations

import random

import numpy as np


class DeterministicDefender:
    """Always patrols the highest-value target (a fixed schedule)."""

    name = "deterministic"

    def __init__(self, target_values: dict[int, float]):
        self.target_values = target_values
        self.probs = self._probs()

    def _probs(self) -> np.ndarray:
        tids = sorted(self.target_values)
        p = np.zeros(len(tids))
        best = max(tids, key=lambda t: self.target_values[t])
        p[best] = 1.0
        return p

    def choose(self, rng: random.Random) -> int:
        return int(np.argmax(self.probs))


class UniformRandomDefender:
    """Uniform patrol over targets."""

    name = "uniform_random"

    def __init__(self, target_values: dict[int, float]):
        self.tids = sorted(target_values)

    def choose(self, rng: random.Random) -> int:
        return rng.choice(self.tids)


class MixedDefender:
    """Patrol target sampled from the zero-sum equilibrium mixed strategy."""

    name = "mixed_equilibrium"

    def __init__(self, strategy: np.ndarray, tids: list[int]):
        if abs(strategy.sum() - 1.0) > 1e-6:
            raise ValueError("strategy must sum to 1")
        self.strategy = strategy
        self.tids = tids

    def choose(self, rng: random.Random) -> int:
        return int(rng.choices(self.tids, weights=self.strategy.tolist(), k=1)[0])
