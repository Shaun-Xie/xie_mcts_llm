"""Core Tic-Tac-Toe game logic."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

X = "X"
O = "O"
EMPTY = "."
WIN_LINES = (
    (0, 1, 2),
    (3, 4, 5),
    (6, 7, 8),
    (0, 3, 6),
    (1, 4, 7),
    (2, 5, 8),
    (0, 4, 8),
    (2, 4, 6),
)


@dataclass(frozen=True)
class GameState:
    """Immutable Tic-Tac-Toe state."""

    board: str
    player: str


def new_game() -> GameState:
    """Create a new game state with X to move."""
    return GameState(board=EMPTY * 9, player=X)


def legal_moves(state: GameState) -> List[int]:
    """Return legal move indices in row-major order (0..8)."""
    return [idx for idx, cell in enumerate(state.board) if cell == EMPTY]


def apply_move(state: GameState, move: int) -> GameState:
    """Apply a move and return the next state."""
    if move < 0 or move > 8:
        raise ValueError(f"Move out of bounds: {move}")
    if state.board[move] != EMPTY:
        raise ValueError(f"Illegal move {move}; cell is occupied")
    next_player = O if state.player == X else X
    next_board = state.board[:move] + state.player + state.board[move + 1 :]
    return GameState(board=next_board, player=next_player)


def winner(state: GameState) -> Optional[str]:
    """Return winner symbol (X/O) or None."""
    b = state.board
    for i, j, k in WIN_LINES:
        if b[i] != EMPTY and b[i] == b[j] == b[k]:
            return b[i]
    return None


def check_terminal(state: GameState) -> bool:
    """Return True if game has ended (win or draw)."""
    return winner(state) is not None or EMPTY not in state.board


def current_player(state: GameState) -> str:
    """Return player to move for the state."""
    return state.player


def board_to_text(board: str) -> str:
    """Render 9-char board string as three rows of text."""
    return "\n".join((board[0:3], board[3:6], board[6:9]))
