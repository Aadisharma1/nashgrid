"""Spatial episode: attacker crosses the perimeter near its chosen target and
walks inward; defender patrols. Interception = co-location at any step."""

from __future__ import annotations

from dataclasses import dataclass

from nashgrid.core.grid import Grid


@dataclass
class EpisodeResult:
    target: int
    intercepted: bool
    defender_payoff: float
    attacker_payoff: float
    steps: int


def _attacker_path(grid: Grid, target_id: int) -> list[tuple[int, int]]:
    """Attack approach path: cross the perimeter at the border cell nearest the
    target, then walk the BFS path inward.

    The path is ``[border] + border..target``, so the attacker spends step 0
    crossing the perimeter and always has >= 1 step of approach. (The previous
    design let the spawn cell coincide with the target for border targets,
    giving a zero-length instant-win attack that made interception rates
    degenerate.)
    """
    x, y = grid.targets[target_id]
    candidates = [(x, 0), (x, grid.height - 1), (0, y), (grid.width - 1, y)]
    border = min(candidates, key=lambda c: abs(c[0] - x) + abs(c[1] - y))
    path = grid.bfs_path(border, grid.targets[target_id]) or [grid.targets[target_id]]
    return [path[0]] + path


def run_episode(grid: Grid, defender_target: int, attacker_target: int,
                intercept_reward: float, max_steps: int = 30,
                defender_speed: int = 2, attacker_speed: int = 1,
                seed: int = 0) -> EpisodeResult:
    """One spatial episode. Defender walks from the next target over (cyclically)
    to its sampled patrol target and holds; attacker crosses the perimeter and
    walks to its target. Intercept if they ever occupy the same cell on the
    same step.

    Defender default speed 2 vs attacker 1: patrol units cover open ground
    faster than an infiltrator moving between cover. Without this asymmetry
    interception is nearly impossible and strategy comparisons are vacuous.
    ``seed`` is retained for API stability (episodes are deterministic in the
    current geometry; future stochastic effects will consume it).
    """
    tids = sorted(grid.targets)
    if defender_target not in tids or attacker_target not in tids:
        raise ValueError("invalid target id")

    d_target_cell = grid.targets[defender_target]
    d_start = grid.targets[tids[(defender_target + 1) % len(tids)]]

    d_path = grid.bfs_path(d_start, d_target_cell) or [d_start]
    a_path = _attacker_path(grid, attacker_target)

    d_route = (d_path[::defender_speed] + [d_path[-1]] * max_steps)[:max_steps + 1]
    steps = 0
    for t in range(min(max_steps, len(a_path) - 1) + 1):
        a_pos = a_path[min(t * attacker_speed, len(a_path) - 1)]
        d_pos = d_route[min(t, len(d_route) - 1)]
        steps = t
        if d_pos == a_pos:
            return EpisodeResult(attacker_target, True, intercept_reward, 0.0, steps)
        if t >= len(a_path) - 1:
            break

    return EpisodeResult(attacker_target, False, -grid.target_values[attacker_target],
                         grid.target_values[attacker_target], steps)
