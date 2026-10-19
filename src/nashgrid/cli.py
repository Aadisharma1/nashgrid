"""NashGrid CLI: python -m nashgrid [--demo | --simulate ...]"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _demo(args) -> int:
    from nashgrid.experiments import run_experiments

    root = Path(__file__).resolve().parents[2]
    cfg = json.loads((root / "configs" / "demo.json").read_text(encoding="utf-8"))
    results = run_experiments(cfg, root / "outputs", seed=cfg["monte_carlo"]["seed"])

    print("=== NASHGRID DEMO (all numbers measured by this run) ===")
    print(f"matrix-game value (defender): {results['game_value']:.3f}")
    print(f"equilibrium defender strategy: "
          f"{[round(p, 3) for p in results['defender_strategy']]}")
    print("\nPairings (attacker success rate / interception rate):")
    for name, r in results["pairings"].items():
        print(f"  {name:<36} {r['attacker_success_rate']:.3f} / {r['interception_rate']:.3f}")
    print("\nGrid-size sweep (mixed defense, best-response attacker):")
    for name, r in results["grid_size_sweep"].items():
        print(f"  {name:<12} attacker success {r['attacker_success_rate']:.3f}")
    print("\nTarget-count sweep:")
    for name, r in results["target_count_sweep"].items():
        print(f"  {name:<12} attacker success {r['attacker_success_rate']:.3f}")
    print("\nAttacker-strength sweep:")
    for name, r in results["attacker_strength_sweep"].items():
        print(f"  {name:<12} attacker success {r['attacker_success_rate']:.3f}")
    print("\nValue of Information:")
    for r in results["voi"]:
        print(f"  p_obs={r['p_obs']:.1f} attacker payoff {r['attacker_mean_payoff']:+.3f} "
              f"(+/- {r['ci95']:.3f})")
    print("\nArtifacts: outputs/experiments.json, outputs/*.png")
    return 0


def _simulate(args) -> int:
    from nashgrid.core.grid import make_grid
    from nashgrid.game.matrix import build_payoff_matrix, solve_zero_sum
    from nashgrid.agents.defender import DeterministicDefender, UniformRandomDefender, MixedDefender
    from nashgrid.agents.attacker import GreedyAttacker, BestResponseAttacker
    from nashgrid.simulation.engine import monte_carlo

    grid = make_grid(args.width, args.height, args.obstacles, args.targets, None, args.seed)
    A = build_payoff_matrix(grid.target_values, args.intercept_reward)
    sol = solve_zero_sum(A)
    tids = sorted(grid.targets)
    defender = {
        "deterministic": DeterministicDefender(grid.target_values),
        "uniform": UniformRandomDefender(grid.target_values),
        "mixed": MixedDefender(sol["defender_strategy"], tids),
    }[args.defender]
    attacker = {
        "greedy": GreedyAttacker(grid.target_values),
        "bestresponse": BestResponseAttacker(A),
    }[args.attacker]
    res = monte_carlo(grid, defender, attacker, args.episodes, args.seed,
                      args.intercept_reward, args.max_steps)
    print(json.dumps(res, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="nashgrid", description=__doc__)
    sub = ap.add_subparsers(dest="mode")

    d = sub.add_parser("demo", aliases=["--demo"])
    d.set_defaults(func=_demo)

    s = sub.add_parser("simulate", aliases=["--simulate"])
    s.add_argument("--width", type=int, default=10)
    s.add_argument("--height", type=int, default=8)
    s.add_argument("--obstacles", type=int, default=10)
    s.add_argument("--targets", type=int, default=3)
    s.add_argument("--defender", choices=["deterministic", "uniform", "mixed"], default="mixed")
    s.add_argument("--attacker", choices=["greedy", "bestresponse"], default="bestresponse")
    s.add_argument("--episodes", type=int, default=200)
    s.add_argument("--max-steps", type=int, default=30)
    s.add_argument("--intercept-reward", type=float, default=5.0)
    s.add_argument("--seed", type=int, default=7)
    s.set_defaults(func=_simulate)

    if argv and argv[0] == "--demo":
        argv = ["demo"]
    if argv and argv[0] == "--simulate":
        argv = ["simulate", *argv[1:]]
    args = ap.parse_args(argv)
    if not hasattr(args, "func"):
        ap.print_help()
        return 2
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
