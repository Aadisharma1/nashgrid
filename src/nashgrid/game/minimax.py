"""Depth-limited minimax with alpha-beta over sequential patrol/attack moves."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TreeState:
    """Explicit small game tree used for tests and tiny boards."""

    children: list["TreeState"]
    value: float | None = None
    mover: str = "max"


def minimax(state: TreeState, depth: int = 10, alpha: float = float("-inf"),
            beta: float = float("inf")) -> float:
    """Classic alpha-beta minimax on an explicit tree. Returns node value."""
    if depth == 0 or (state.value is not None and not state.children):
        return state.value if state.value is not None else 0.0
    if state.mover == "max":
        best = float("-inf")
        for child in state.children:
            best = max(best, minimax(child, depth - 1, alpha, beta))
            alpha = max(alpha, best)
            if beta <= alpha:
                break
        return best
    best = float("inf")
    for child in state.children:
        best = min(best, minimax(child, depth - 1, alpha, beta))
        beta = min(beta, best)
        if beta <= alpha:
            break
    return best


def minimax_best_child(state: TreeState, depth: int = 10) -> tuple[int, float]:
    """Index and value of the maximizing mover's best child (with pruning)."""
    if not state.children:
        raise ValueError("state has no children")
    alpha = float("-inf")
    best_i, best_v = 0, float("-inf")
    for i, child in enumerate(state.children):
        v = minimax(child, depth - 1, alpha, float("inf"))
        if v > best_v:
            best_i, best_v = i, v
        alpha = max(alpha, best_v)
    return best_i, best_v
