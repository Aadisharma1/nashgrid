import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def test_voi_monotone_and_bounds(tmp_path):
    from nashgrid.agents.defender import MixedDefender
    from nashgrid.analysis.voi import value_of_information
    from nashgrid.core.grid import make_grid
    from nashgrid.game.matrix import build_payoff_matrix, solve_zero_sum

    g = make_grid(10, 8, 10, 3, [3.0, 2.0, 1.0], seed=7)
    A = build_payoff_matrix(g.target_values, 5.0)
    sol = solve_zero_sum(A)
    res = value_of_information(g, MixedDefender(sol["defender_strategy"], sorted(g.targets)),
                               [0.0, 0.5, 1.0], n_episodes=100, seed=3, intercept_reward=5.0)
    assert [r["p_obs"] for r in res] == [0.0, 0.5, 1.0]
    assert res[-1]["attacker_mean_payoff"] >= res[0]["attacker_mean_payoff"]


def test_cli_demo_runs(tmp_path, capsys, monkeypatch):
    from nashgrid import cli

    monkeypatch.chdir(Path(__file__).resolve().parents[1])
    rc = cli.main(["--demo"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "NASHGRID DEMO" in out
    assert (Path("outputs") / "experiments.json").exists()
    data = (Path("outputs") / "experiments.json").read_text()
    assert "pairings" in data and "voi" in data
