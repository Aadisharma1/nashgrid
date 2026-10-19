import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib

matplotlib.use("Agg")

from nashgrid.analysis.voi import value_of_information
from nashgrid.core.grid import make_grid
from nashgrid.game.matrix import build_payoff_matrix, solve_zero_sum
from nashgrid.visualization.render import (
    render_grid,
    render_strategy_dashboard,
    render_voi_plot,
)


def _nonempty_png(path) -> bool:
    return path.exists() and path.stat().st_size > 1000


@pytest.fixture()
def grid():
    return make_grid(10, 8, 10, 3, [3.0, 2.0, 1.0], seed=7)


def test_render_grid_creates_png(tmp_path, grid):
    out = render_grid(grid, tmp_path / "grid.png", title="test world")
    assert _nonempty_png(out)


def test_render_grid_with_positions_and_outdir_creation(tmp_path, grid):
    target = grid.targets[0]
    out = render_grid(grid, tmp_path / "sub" / "dir" / "grid.png",
                      defender_pos=target, attacker_pos=(1, 1))
    assert _nonempty_png(out)


def test_render_strategy_dashboard_all_pairings(tmp_path):
    results = {
        "deterministic_vs_greedy": {"attacker_success_rate": 1.0},
        "mixed_vs_greedy": {"attacker_success_rate": 0.6},
        "uniform_vs_bestresponse": {"attacker_success_rate": 0.4},
    }
    out = render_strategy_dashboard(results, tmp_path / "dash.png")
    assert _nonempty_png(out)


def test_render_voi_plot_matches_input_points(tmp_path):
    voi = [{"p_obs": p / 10, "attacker_mean_payoff": p - 5.0, "ci95": 0.1}
           for p in range(11)]
    out = render_voi_plot(voi, tmp_path / "voi.png")
    assert _nonempty_png(out)


def test_voi_analysis_returns_consistent_structure(tmp_path, grid):
    A = build_payoff_matrix(grid.target_values, 5.0)
    sol = solve_zero_sum(A)
    from nashgrid.agents.defender import MixedDefender

    res = value_of_information(grid, MixedDefender(sol["defender_strategy"], sorted(grid.targets)),
                               [0.0, 1.0], n_episodes=40, seed=1, intercept_reward=5.0)
    assert len(res) == 2
    for r in res:
        assert 0.0 <= r["p_obs"] <= 1.0
        assert r["ci95"] >= 0.0
    assert res[1]["attacker_mean_payoff"] >= res[0]["attacker_mean_payoff"] - 1e-9
