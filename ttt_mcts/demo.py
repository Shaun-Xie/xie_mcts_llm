"""CLI demo for baseline and LLM-guided MCTS Tic-Tac-Toe agents."""

from __future__ import annotations

import argparse
from dataclasses import dataclass

from .game import GameState, apply_move, board_to_text, check_terminal, current_player, new_game, winner
from .mcts import BaselineMCTSAgent, LLMMCTSAgent


@dataclass
class MoveDecision:
    """Agent move selection metadata."""

    move: int
    q: float
    label: str


def _print_board(state: GameState) -> None:
    print(board_to_text(state.board))


def _build_agent(name: str, iters: int):
    if name == "baseline":
        return BaselineMCTSAgent(iters=iters)
    if name == "llm":
        return LLMMCTSAgent(iters=iters)
    raise ValueError(f"Unknown agent: {name}")


def _agent_move(state: GameState, agent_name: str, iters: int) -> MoveDecision:
    agent = _build_agent(agent_name, iters)
    move, q = agent.choose_move(state)
    return MoveDecision(move=move, q=q, label=agent_name)


def _human_move(state: GameState) -> MoveDecision:
    legal = [str(i) for i, c in enumerate(state.board) if c == "."]
    while True:
        raw = input(f"Your move ({', '.join(legal)}): ").strip()
        if raw in legal:
            return MoveDecision(move=int(raw), q=0.0, label="human")
        print("Invalid move.")


def _announce_move(state: GameState, decision: MoveDecision) -> None:
    player = current_player(state)
    print(f"Player {player} -> {decision.label} selects move {decision.move} with Q={decision.q:.3f}")


def run_human_vs_agent(agent_name: str, iters: int) -> None:
    """Human is X, selected agent is O."""
    state = new_game()
    print("Human is X. Agent is O.")
    _print_board(state)

    while not check_terminal(state):
        if current_player(state) == "X":
            decision = _human_move(state)
        else:
            decision = _agent_move(state, agent_name=agent_name, iters=iters)
        _announce_move(state, decision)
        state = apply_move(state, decision.move)
        _print_board(state)
        print()

    _print_result(state)


def run_self_play(agent_name: str, iters: int) -> None:
    """Run single game where one agent controls both sides."""
    state = new_game()
    print(f"Self-play with {agent_name} agent on both sides.")
    _print_board(state)

    while not check_terminal(state):
        decision = _agent_move(state, agent_name=agent_name, iters=iters)
        _announce_move(state, decision)
        state = apply_move(state, decision.move)
        _print_board(state)
        print()

    _print_result(state)


def _pick_match_agent(player: str, game_idx: int) -> str:
    if game_idx % 2 == 0:
        return "llm" if player == "X" else "baseline"
    return "baseline" if player == "X" else "llm"


def run_match(games: int, iters: int) -> None:
    """Run baseline vs llm for multiple games, alternating first player."""
    stats = {"llm": 0, "baseline": 0, "draw": 0}

    for game_idx in range(games):
        state = new_game()
        print(f"=== Match Game {game_idx + 1}/{games} ===")
        x_agent = _pick_match_agent("X", game_idx)
        o_agent = _pick_match_agent("O", game_idx)
        print(f"X={x_agent}, O={o_agent}")
        _print_board(state)

        while not check_terminal(state):
            agent_name = x_agent if current_player(state) == "X" else o_agent
            decision = _agent_move(state, agent_name=agent_name, iters=iters)
            _announce_move(state, decision)
            state = apply_move(state, decision.move)
            _print_board(state)
            print()

        game_winner = winner(state)
        if game_winner is None:
            stats["draw"] += 1
            print("Result: Draw")
        elif game_winner == "X":
            stats[x_agent] += 1
            print(f"Result: {x_agent} wins as X")
        else:
            stats[o_agent] += 1
            print(f"Result: {o_agent} wins as O")
        print()

    print("=== Match Summary ===")
    print(f"LLM wins: {stats['llm']}")
    print(f"Baseline wins: {stats['baseline']}")
    print(f"Draws: {stats['draw']}")


def _print_result(state: GameState) -> None:
    w = winner(state)
    if w is None:
        print("Result: Draw")
    else:
        print(f"Result: {w} wins")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LLM-guided MCTS Tic-Tac-Toe demo (Part 1)")
    parser.add_argument("--agent", choices=["baseline", "llm"], default="llm")
    parser.add_argument("--iters", type=int, default=300)
    parser.add_argument("--play", choices=["human", "self", "match"], default="self")
    parser.add_argument("--games", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.play == "human":
        run_human_vs_agent(agent_name=args.agent, iters=args.iters)
    elif args.play == "self":
        run_self_play(agent_name=args.agent, iters=args.iters)
    else:
        run_match(games=args.games, iters=args.iters)


if __name__ == "__main__":
    main()
