"""Monte Carlo Tree Search agents for Tic-Tac-Toe."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

from .game import GameState, apply_move, check_terminal, legal_moves, new_game, winner
from .llm_eval import evaluate_position


@dataclass
class Node:
    """MCTS node with value tracked from current player perspective."""

    state: GameState
    parent: Optional["Node"] = None
    move: Optional[int] = None
    children: Dict[int, "Node"] = field(default_factory=dict)
    untried_moves: List[int] = field(default_factory=list)
    N: int = 0
    W: float = 0.0

    def __post_init__(self) -> None:
        if not self.untried_moves:
            self.untried_moves = legal_moves(self.state)

    @property
    def Q(self) -> float:
        """Mean value W/N from this node's player-to-move perspective."""
        return self.W / self.N if self.N else 0.0

    @property
    def terminal(self) -> bool:
        """Whether this node state is terminal."""
        return check_terminal(self.state)

    def expand_one(self, rng: random.Random) -> "Node":
        """Expand by adding one child from an untried move."""
        move_index = rng.randrange(len(self.untried_moves))
        move = self.untried_moves.pop(move_index)
        child = Node(state=apply_move(self.state, move), parent=self, move=move)
        self.children[move] = child
        return child

    def best_ucb_child(self, exploration_c: float) -> "Node":
        """Select child maximizing UCB1."""

        def score(child: Node) -> float:
            if child.N == 0:
                return float("inf")
            return child.Q + exploration_c * math.sqrt(math.log(self.N) / child.N)

        return max(self.children.values(), key=score)


class BaseMCTSAgent:
    """Shared MCTS algorithm with pluggable leaf evaluation."""

    def __init__(self, iters: int = 500, exploration_c: float = 1.41421356237, seed: Optional[int] = None) -> None:
        self.iters = iters
        self.exploration_c = exploration_c
        self.rng = random.Random(seed)

    def choose_move(self, state: GameState) -> Tuple[int, float]:
        """Run MCTS and return (move, estimated_q_from_current_player_perspective)."""
        if check_terminal(state):
            return -1, self._terminal_value(state)

        root = Node(state=state)
        for _ in range(self.iters):
            node = root

            while (not node.terminal) and (not node.untried_moves) and node.children:
                node = node.best_ucb_child(self.exploration_c)

            if (not node.terminal) and node.untried_moves:
                node = node.expand_one(self.rng)

            value = self.evaluate_leaf(node.state)

            while node is not None:
                node.N += 1
                node.W += value
                value = 1.0 - value
                node = node.parent

        if not root.children:
            return -1, self._terminal_value(state)

        best_child = max(
            root.children.values(),
            key=lambda child: (child.N, 1.0 - child.Q),
        )
        assert best_child.move is not None
        root_perspective_q = 1.0 - best_child.Q
        return best_child.move, root_perspective_q

    def evaluate_leaf(self, state: GameState) -> float:
        """Leaf evaluation in [0,1] from player-to-move perspective."""
        raise NotImplementedError

    def _terminal_value(self, state: GameState) -> float:
        """Terminal utility from state.player perspective."""
        w = winner(state)
        if w is None:
            return 0.5
        return 1.0 if w == state.player else 0.0


class BaselineMCTSAgent(BaseMCTSAgent):
    """Baseline MCTS with random rollout simulation."""

    def evaluate_leaf(self, state: GameState) -> float:
        if check_terminal(state):
            return self._terminal_value(state)

        rollout_state = state
        start_player = state.player
        while not check_terminal(rollout_state):
            moves = legal_moves(rollout_state)
            move = self.rng.choice(moves)
            rollout_state = apply_move(rollout_state, move)

        w = winner(rollout_state)
        if w is None:
            return 0.5
        return 1.0 if w == start_player else 0.0


class LLMMCTSAgent(BaseMCTSAgent):
    """MCTS using LLM (or fallback heuristic) leaf evaluations with caching."""

    def __init__(self, iters: int = 300, exploration_c: float = 1.41421356237, seed: Optional[int] = None) -> None:
        super().__init__(iters=iters, exploration_c=exploration_c, seed=seed)
        self.cache: Dict[Tuple[str, str], Dict[str, object]] = {}
        self.eval_requests: int = 0
        self.cache_hits: int = 0
        self.cache_misses: int = 0

    def reset_cache_stats(self) -> None:
        """Reset counters used by experiment reporting."""
        self.eval_requests = 0
        self.cache_hits = 0
        self.cache_misses = 0

    def cache_stats(self) -> Dict[str, Union[int, float]]:
        """Return cache counters and hit rate."""
        total_lookups = self.cache_hits + self.cache_misses
        hit_rate = (self.cache_hits / total_lookups) if total_lookups else 0.0
        return {
            "eval_requests": self.eval_requests,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_rate": hit_rate,
            "cache_size": len(self.cache),
        }

    def evaluate_leaf(self, state: GameState) -> float:
        if check_terminal(state):
            return self._terminal_value(state)

        self.eval_requests += 1
        key = (state.board, state.player)
        if key not in self.cache:
            self.cache_misses += 1
            board_text = "\n".join((state.board[0:3], state.board[3:6], state.board[6:9]))
            self.cache[key] = evaluate_position(board_text=board_text, player=state.player)
        else:
            self.cache_hits += 1

        value = self.cache[key].get("value", 0.5)
        try:
            num_value = float(value)
        except (TypeError, ValueError):
            num_value = 0.5
        return min(1.0, max(0.0, num_value))


def play_self(agent: BaseMCTSAgent) -> GameState:
    """Run self-play from a fresh game for quick smoke tests."""
    state = new_game()
    while not check_terminal(state):
        move, _ = agent.choose_move(state)
        state = apply_move(state, move)
    return state
