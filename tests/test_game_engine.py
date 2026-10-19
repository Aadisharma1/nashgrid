import random

import numpy as np
import pytest

from nashgrid.agents.attacker import BestResponseAttacker, GreedyAttacker
from nashgrid.agents.defender import DeterministicDefender, MixedDefender, UniformRandomDefender
from nashgrid.core.grid import Grid, make_grid
from nashgrid.game.matrix import build_payoff_matrix, solve_zero_sum
from nashgrid.game.minimax import TreeState, minimax, minimax_best_child
from nashgrid.simulation.engine import monte_carlo


def test_grid_validation_and_path():
    g = Grid(5, 4, obstacles={(2, 2)}, targets={0: (0, 0), 1: (4, 3)},
             target_values={0: 1.0, 1: 2.0})
    with pytest.raises(ValueError):
        Grid(5, 4, obstacles={(9, 9)})
    with pytest.raises(ValueError):
        Grid(5, 4, obstacles={(0, 0)}, targets={0: (0, 0)})

    path = g.bfs_path((0, 1), (4, 3))
    assert path[0] == (0, 1) and path[-1] == (4, 3)
    assert len(path) == 7
    assert g.bfs_path((0, 0), (2, 2)) == []


def test_make_grid_targets_not_on_obstacles():
    g = make_grid(10, 8, 12, 3, [3.0, 2.0, 1.0], seed=7)
    for cell in g.targets.values():
        assert cell not in g.obstacles
    assert len(g.targets) == 3
    assert set(g.target_values) == set(g.targets)


def test_payoff_matrix_shape_and_values():
    A = build_payoff_matrix({0: 3.0, 1: 2.0}, intercept_reward=5.0)
    assert A.shape == (2, 2)
    assert A[0, 0] == 5.0 and A[0, 1] == -2.0 and A[1, 0] == -3.0 and A[1, 1] == 5.0


def test_solve_zero_sum_matching_pennies():
    A = np.array([[1.0, -1.0], [-1.0, 1.0]])
    sol = solve_zero_sum(A)
    assert abs(sol["game_value"]) < 1e-6
    assert np.allclose(sol["defender_strategy"], [0.5, 0.5], atol=1e-6)


def test_solve_zero_sum_rock_paper_scissors():
    A = np.array([[0.0, -1.0, 1.0], [1.0, 0.0, -1.0], [-1.0, 1.0, 0.0]])
    sol = solve_zero_sum(A)
    assert abs(sol["game_value"]) < 1e-6
    assert np.allclose(sol["defender_strategy"], [1 / 3, 1 / 3, 1 / 3], atol=1e-6)


def test_solve_zero_sum_dominant_row():
    A = np.array([[5.0, -3.0], [4.0, -1.0]])
    sol = solve_zero_sum(A)
    assert sol["game_value"] == pytest.approx(-1.0)


def _tree(depth):
    if depth == 0:
        return TreeState(children=[], value=float(random.choice([1, 2, 3, 4, 5])))
    return TreeState(children=[_tree(depth - 1) for _ in range(2)],
                     mover="max" if depth % 2 == 1 else "min")


def test_minimax_known_values():
    root = TreeState(children=[
        TreeState(mover="min", children=[
            TreeState(children=[], value=3.0), TreeState(children=[], value=5.0)]),
        TreeState(mover="min", children=[
            TreeState(children=[], value=2.0), TreeState(children=[], value=9.0)]),
    ], mover="max")
    assert minimax(root) == 3.0
    idx, val = minimax_best_child(root)
    assert idx == 0 and val == 3.0


def test_minimax_equals_bruteforce():
    random.seed(0)
    root = _tree(5)
    brute = _bruteforce(root, True)
    assert minimax(root, depth=10) == brute


def _bruteforce(state, maximizing):
    if not state.children:
        return state.value
    vals = [_bruteforce(c, not maximizing) for c in state.children]
    return max(vals) if maximizing else min(vals)


def test_defender_strategies():
    tv = {0: 1.0, 1: 3.0, 2: 2.0}
    det = DeterministicDefender(tv)
    assert det.choose(random.Random(0)) == 1
    uni = UniformRandomDefender(tv)
    picks = {uni.choose(random.Random(i)) for i in range(50)}
    assert picks == {0, 1, 2}
    mix = MixedDefender(np.array([0.2, 0.5, 0.3]), [0, 1, 2])
    assert mix.choose(random.Random(1)) in (0, 1, 2)
    with pytest.raises(ValueError):
        MixedDefender(np.array([0.5, 0.5, 0.5]), [0, 1, 2])


def test_attacker_best_response_uses_distribution():
    A = build_payoff_matrix({0: 3.0, 1: 2.0}, 5.0)
    br = BestResponseAttacker(A)
    assert br.choose(random.Random(0), np.array([1.0, 0.0])) == 1
    greedy = GreedyAttacker({0: 3.0, 1: 2.0})
    assert greedy.choose(random.Random(0), np.array([1.0, 0.0])) == 1


def test_monte_carlo_determinism_and_direction():
    g = Grid(5, 5, obstacles=set(),
             targets={0: (2, 1), 1: (2, 3), 2: (3, 2)},
             target_values={0: 3.0, 1: 2.0, 2: 1.0})
    A = build_payoff_matrix(g.target_values, 5.0)
    sol = solve_zero_sum(A)
    tids = sorted(g.targets)

    r1 = monte_carlo(g, MixedDefender(sol["defender_strategy"], tids),
                     BestResponseAttacker(A), 60, seed=3, intercept_reward=5.0)
    r2 = monte_carlo(g, MixedDefender(sol["defender_strategy"], tids),
                     BestResponseAttacker(A), 60, seed=3, intercept_reward=5.0)
    assert r1 == r2, "same seed must reproduce identical statistics"

    det = monte_carlo(g, DeterministicDefender(g.target_values),
                      GreedyAttacker(g.target_values), 150, seed=3, intercept_reward=5.0)
    mix = monte_carlo(g, MixedDefender(sol["defender_strategy"], tids),
                      GreedyAttacker(g.target_values), 150, seed=3, intercept_reward=5.0)
    assert det["interception_rate"] == 0.0
    assert mix["interception_rate"] > 0.0
    assert 0.0 <= det["attacker_success_rate"] <= 1.0
