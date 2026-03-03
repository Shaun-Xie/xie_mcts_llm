"""Perfect-play minimax agent for Tic-Tac-Toe."""

from __future__ import annotations

from functools import lru_cache
from typing import Tuple

from .game import GameState, apply_move, check_terminal, legal_moves, winner


@lru_cache(maxsize=None)
def _value(board: str, player: str) -> float:
    """Return exact game-theoretic value in [0,1] from player-to-move perspective."""
    state = GameState(board=board, player=player)
    if check_terminal(state):
        w = winner(state)
        if w is None:
            return 0.5
        return 1.0 if w == player else 0.0

    best = -1.0
    for move in legal_moves(state):
        child = apply_move(state, move)
        child_value = _value(child.board, child.player)
        value = 1.0 - child_value
        if value > best:
            best = value
    return best


def choose_move(board: str, player: str) -> Tuple[int, float]:
    """Return optimal move index (0-8) and exact value from current player's perspective."""
    state = GameState(board=board, player=player)
    if check_terminal(state):
        return -1, _value(board, player)

    best_move = -1
    best_value = -1.0
    for move in legal_moves(state):
        child = apply_move(state, move)
        child_value = _value(child.board, child.player)
        value = 1.0 - child_value
        if value > best_value:
            best_value = value
            best_move = move

    return best_move, best_value


def choose_move_state(state: GameState) -> Tuple[int, float]:
    """State-based convenience wrapper around choose_move."""
    return choose_move(state.board, state.player)
