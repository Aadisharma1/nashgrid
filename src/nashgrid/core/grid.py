"""Grid world: cells, obstacles, targets, BFS pathfinding."""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass, field


@dataclass
class Grid:
    width: int
    height: int
    obstacles: set[tuple[int, int]] = field(default_factory=set)
    targets: dict[int, tuple[int, int]] = field(default_factory=dict)
    target_values: dict[int, float] = field(default_factory=dict)

    def __post_init__(self):
        if self.width < 2 or self.height < 2:
            raise ValueError("grid must be at least 2x2")
        for (x, y) in self.obstacles:
            if not (0 <= x < self.width and 0 <= y < self.height):
                raise ValueError(f"obstacle {(x, y)} out of bounds")
        for tid, (x, y) in self.targets.items():
            if (x, y) in self.obstacles:
                raise ValueError(f"target {tid} on obstacle")
            if not (0 <= x < self.width and 0 <= y < self.height):
                raise ValueError(f"target {tid} out of bounds")

    def in_bounds(self, cell: tuple[int, int]) -> bool:
        x, y = cell
        return 0 <= x < self.width and 0 <= y < self.height

    def passable(self, cell: tuple[int, int]) -> bool:
        return self.in_bounds(cell) and cell not in self.obstacles

    def neighbors(self, cell: tuple[int, int]) -> list[tuple[int, int]]:
        x, y = cell
        cand = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
        return [c for c in cand if self.passable(c)]

    def bfs_path(self, start: tuple[int, int], goal: tuple[int, int]) -> list[tuple[int, int]]:
        """Shortest path start->goal inclusive; [] if unreachable."""
        if start == goal:
            return [start]
        if not self.passable(start) or not self.passable(goal):
            return []
        prev = {start: None}
        q = deque([start])
        while q:
            cur = q.popleft()
            for nxt in self.neighbors(cur):
                if nxt not in prev:
                    prev[nxt] = cur
                    if nxt == goal:
                        path = [nxt]
                        while prev[path[-1]] is not None:
                            path.append(prev[path[-1]])
                        return path[::-1]
                    q.append(nxt)
        return []

    def path_length(self, start: tuple[int, int], goal: tuple[int, int]) -> int:
        path = self.bfs_path(start, goal)
        return max(len(path) - 1, 0) if path else 10**9


def make_grid(width: int, height: int, n_obstacles: int, n_targets: int,
              target_values: list[float] | None, seed: int) -> Grid:
    """Randomly place obstacles and targets (seeded, always solvable layout)."""
    rng = random.Random(seed)
    all_cells = [(x, y) for x in range(width) for y in range(height)]
    obstacles: set[tuple[int, int]] = set()
    for cell in rng.sample(all_cells, min(n_obstacles, len(all_cells) // 3)):
        obstacles.add(cell)

    free = [c for c in all_cells if c not in obstacles]
    border = [c for c in free if c[0] in (0, width - 1) or c[1] in (0, height - 1)]
    target_cells = rng.sample(border, min(n_targets, len(border)))
    while any(abs(a[0] - b[0]) + abs(a[1] - b[1]) < 3
              for i, a in enumerate(target_cells) for b in target_cells[i + 1:]):
        target_cells = rng.sample(border, min(n_targets, len(border)))

    values = target_values or [1.0 + i for i in range(n_targets)]
    targets = {i: target_cells[i] for i in range(len(target_cells))}
    tvals = {i: float(values[i % len(values)]) for i in targets}
    return Grid(width, height, obstacles, targets, tvals)
