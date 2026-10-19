"""Site-specific zero-sum game: defender patrols one target, attacker attacks one.

Defender payoff matrix A[d, a]:
    d == a  -> +intercept_reward (attack foiled)
    d != a  -> -value_a          (attacker steals target a's value)

The spatial episode simulator (core/episode.py) realizes the same payoffs on a
grid; the matrix game is the tractable abstraction used for mixed strategies.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linprog


def build_payoff_matrix(target_values: dict[int, float], intercept_reward: float) -> np.ndarray:
    n = len(target_values)
    A = np.zeros((n, n))
    for d in range(n):
        for a in range(n):
            A[d, a] = intercept_reward if d == a else -target_values[a]
    return A


def solve_zero_sum(A: np.ndarray) -> dict:
    """Solve the zero-sum game (row = defender maximizer) exactly via LP.

    max v  s.t.  sum_j A[i,j] x_j >= v for all i,  sum x = 1, x >= 0
    Returns defender mixed strategy x, attacker best-response distribution q
    (minimax optimal over columns), and game value v.
    """
    m, n = A.shape
    if m == 0 or n == 0:
        raise ValueError("empty payoff matrix")
    c = np.zeros(m + 1)
    c[-1] = -1.0
    A_ub = np.hstack([-A.T, np.ones((n, 1))])
    b_ub = np.zeros(n)
    A_eq = np.zeros((1, m + 1))
    A_eq[0, :m] = 1.0
    b_eq = [1.0]
    bounds = [(0, None)] * m + [(None, None)]
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if not res.success:
        raise RuntimeError(f"LP failed: {res.message}")
    x = res.x[:m]
    x = np.clip(x, 0, None)
    x = x / x.sum()
    v = float(res.x[-1])
    col_expect = x @ A
    q = np.zeros(n)
    q[np.argmin(col_expect)] = 1.0
    return {"defender_strategy": x, "attacker_strategy": q, "game_value": v}
