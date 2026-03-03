"""Experiment harness for PA4 Part 2 comparisons."""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import random
import socket
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .game import GameState, apply_move, check_terminal, current_player, new_game, winner
from .llm_eval import runtime_config
from .mcts import BaselineMCTSAgent, LLMMCTSAgent
from .minimax import choose_move as minimax_choose_move


@dataclass
class AgentAggregate:
    """Per-agent metrics accumulated over one matchup."""

    name: str
    wins: int = 0
    losses: int = 0
    draws: int = 0
    decisions: int = 0
    moves: int = 0
    sum_value: float = 0.0
    sum_time_per_move: float = 0.0
    sum_time_per_iter: float = 0.0
    iter_samples: int = 0

    def record_move(self, elapsed: float, value: float, iters: Optional[int]) -> None:
        self.moves += 1
        self.decisions += 1
        self.sum_time_per_move += elapsed
        self.sum_value += value
        if iters is not None and iters > 0:
            self.sum_time_per_iter += elapsed / iters
            self.iter_samples += 1

    def avg_time_per_move(self) -> float:
        return self.sum_time_per_move / self.moves if self.moves else 0.0

    def avg_time_per_iter(self) -> float:
        return self.sum_time_per_iter / self.iter_samples if self.iter_samples else 0.0

    def avg_value(self) -> float:
        return self.sum_value / self.decisions if self.decisions else 0.0


class Policy:
    """Runtime interface for move selection in experiments."""

    name: str

    def choose(self, state: GameState) -> Tuple[int, float]:
        raise NotImplementedError

    def iterations_per_move(self) -> Optional[int]:
        return None


class BaselinePolicy(Policy):
    """Wrapper around baseline MCTS agent."""

    def __init__(self, iters: int, seed: int) -> None:
        self.name = "baseline"
        self.agent = BaselineMCTSAgent(iters=iters, seed=seed)

    def choose(self, state: GameState) -> Tuple[int, float]:
        return self.agent.choose_move(state)

    def iterations_per_move(self) -> Optional[int]:
        return self.agent.iters


class LLMPolicy(Policy):
    """Wrapper around LLM-guided MCTS agent."""

    def __init__(self, iters: int, seed: int) -> None:
        self.name = "llm"
        self.agent = LLMMCTSAgent(iters=iters, seed=seed)
        self.agent.reset_cache_stats()

    def choose(self, state: GameState) -> Tuple[int, float]:
        return self.agent.choose_move(state)

    def iterations_per_move(self) -> Optional[int]:
        return self.agent.iters


class MinimaxPolicy(Policy):
    """Wrapper around exact minimax solver."""

    def __init__(self) -> None:
        self.name = "minimax"

    def choose(self, state: GameState) -> Tuple[int, float]:
        return minimax_choose_move(state.board, state.player)


def _make_policy(name: str, iters: int, seed: int) -> Policy:
    if name == "baseline":
        return BaselinePolicy(iters=iters, seed=seed)
    if name == "llm":
        return LLMPolicy(iters=iters, seed=seed)
    if name == "minimax":
        return MinimaxPolicy()
    raise ValueError(f"Unsupported policy: {name}")


def _winner_agent_name(state: GameState, x_name: str, o_name: str) -> Optional[str]:
    w = winner(state)
    if w is None:
        return None
    return x_name if w == "X" else o_name


def _run_one_matchup(iters: int, games: int, seed: int, agent_a: str, agent_b: str) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    rng = random.Random(seed)
    policy_a = _make_policy(agent_a, iters=iters, seed=rng.randrange(1 << 30))
    policy_b = _make_policy(agent_b, iters=iters, seed=rng.randrange(1 << 30))

    agg: Dict[str, AgentAggregate] = {
        agent_a: AgentAggregate(name=agent_a),
        agent_b: AgentAggregate(name=agent_b),
    }

    per_game_rows: List[Dict[str, object]] = []
    half = games // 2

    for game_idx in range(games):
        if game_idx < half:
            x_policy, o_policy = policy_a, policy_b
            x_name, o_name = agent_a, agent_b
        else:
            x_policy, o_policy = policy_b, policy_a
            x_name, o_name = agent_b, agent_a

        state = new_game()
        game_move_count = 0
        game_start = time.perf_counter()

        while not check_terminal(state):
            policy = x_policy if current_player(state) == "X" else o_policy
            started = time.perf_counter()
            move, value_est = policy.choose(state)
            elapsed = time.perf_counter() - started

            agg[policy.name].record_move(
                elapsed=elapsed,
                value=value_est,
                iters=policy.iterations_per_move(),
            )

            state = apply_move(state, move)
            game_move_count += 1

        winner_name = _winner_agent_name(state, x_name=x_name, o_name=o_name)
        if winner_name is None:
            agg[agent_a].draws += 1
            agg[agent_b].draws += 1
        else:
            loser_name = agent_b if winner_name == agent_a else agent_a
            agg[winner_name].wins += 1
            agg[loser_name].losses += 1

        per_game_rows.append(
            {
                "iters": iters,
                "matchup": f"{agent_a}_vs_{agent_b}",
                "game_index": game_idx,
                "x_agent": x_name,
                "o_agent": o_name,
                "winner": winner(state),
                "winner_agent": winner_name,
                "draw": winner_name is None,
                "moves": game_move_count,
                "duration_sec": time.perf_counter() - game_start,
            }
        )

    llm_policy: Optional[LLMPolicy] = None
    if isinstance(policy_a, LLMPolicy):
        llm_policy = policy_a
    elif isinstance(policy_b, LLMPolicy):
        llm_policy = policy_b

    llm_stats = {
        "llm_calls": 0,
        "llm_cache_hits": 0,
        "llm_cache_misses": 0,
        "llm_cache_hit_rate": 0.0,
        "llm_cache_size": 0,
        "llm_calls_per_move": 0.0,
    }
    if llm_policy is not None:
        stats = llm_policy.agent.cache_stats()
        llm_moves = agg["llm"].moves if "llm" in agg else 0
        llm_calls = int(stats["cache_misses"])
        llm_stats = {
            "llm_calls": llm_calls,
            "llm_cache_hits": int(stats["cache_hits"]),
            "llm_cache_misses": int(stats["cache_misses"]),
            "llm_cache_hit_rate": float(stats["cache_hit_rate"]),
            "llm_cache_size": int(stats["cache_size"]),
            "llm_calls_per_move": (llm_calls / llm_moves) if llm_moves else 0.0,
        }

    draws = agg[agent_a].draws
    row: Dict[str, object] = {
        "iters": iters,
        "games": games,
        "seed": seed,
        "matchup": f"{agent_a}_vs_{agent_b}",
        "agent_a": agent_a,
        "agent_b": agent_b,
        "agent_a_wins": agg[agent_a].wins,
        "agent_b_wins": agg[agent_b].wins,
        "draws": draws,
        "agent_a_losses": agg[agent_a].losses,
        "agent_b_losses": agg[agent_b].losses,
        "agent_a_avg_time_per_move": agg[agent_a].avg_time_per_move(),
        "agent_b_avg_time_per_move": agg[agent_b].avg_time_per_move(),
        "agent_a_avg_time_per_iteration": agg[agent_a].avg_time_per_iter(),
        "agent_b_avg_time_per_iteration": agg[agent_b].avg_time_per_iter(),
        "agent_a_avg_value": agg[agent_a].avg_value(),
        "agent_b_avg_value": agg[agent_b].avg_value(),
        "agent_a_moves": agg[agent_a].moves,
        "agent_b_moves": agg[agent_b].moves,
    }
    row.update(llm_stats)
    return row, per_game_rows


def _write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_jsonl(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _parse_iters(raw: str) -> List[int]:
    values: List[int] = []
    for part in raw.split(","):
        text = part.strip()
        if not text:
            continue
        value = int(text)
        if value <= 0:
            raise ValueError("Iteration counts must be positive")
        values.append(value)
    if not values:
        raise ValueError("At least one iteration count is required")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Part 2 Tic-Tac-Toe experiments")
    parser.add_argument("--iters", type=str, default="10,50,100,500", help="Comma-separated MCTS iterations")
    parser.add_argument("--games", type=int, default=200, help="Games per matchup per iteration")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--llm_mode",
        choices=["auto", "openai", "ollama", "heuristic"],
        default="auto",
        help="Backend preference for LLM evaluations",
    )
    parser.add_argument("--output_dir", type=str, default="results")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    iters_list = _parse_iters(args.iters)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    os.environ["TTT_LLM_PROVIDER"] = args.llm_mode

    matchup_pairs = [
        ("baseline", "llm"),
        ("baseline", "minimax"),
        ("llm", "minimax"),
    ]

    all_rows: List[Dict[str, object]] = []
    per_game_rows: List[Dict[str, object]] = []

    for iters in iters_list:
        for pair_idx, (agent_a, agent_b) in enumerate(matchup_pairs):
            matchup_seed = args.seed + (iters * 10_000) + (pair_idx * 1_000)
            row, games_rows = _run_one_matchup(
                iters=iters,
                games=args.games,
                seed=matchup_seed,
                agent_a=agent_a,
                agent_b=agent_b,
            )
            row["llm_mode"] = args.llm_mode
            all_rows.append(row)
            per_game_rows.extend(games_rows)
            print(
                f"iters={iters:>4} matchup={agent_a}_vs_{agent_b:<8} "
                f"A_wins={row['agent_a_wins']:>3} B_wins={row['agent_b_wins']:>3} draws={row['draws']:>3}"
            )

    results_csv = out_dir / "results.csv"
    per_game_jsonl = out_dir / "per_game.jsonl"
    metadata_json = out_dir / "metadata.json"

    _write_csv(results_csv, all_rows)
    _write_jsonl(per_game_jsonl, per_game_rows)

    metadata = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "config": {
            "iters": iters_list,
            "games": args.games,
            "seed": args.seed,
            "llm_mode": args.llm_mode,
            "matchups": ["baseline_vs_llm", "baseline_vs_minimax", "llm_vs_minimax"],
        },
        "llm_runtime": runtime_config(),
        "outputs": {
            "results_csv": str(results_csv),
            "per_game_jsonl": str(per_game_jsonl),
        },
    }

    with metadata_json.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved results to: {results_csv}")
    print(f"Saved per-game logs to: {per_game_jsonl}")
    print(f"Saved metadata to: {metadata_json}")


if __name__ == "__main__":
    main()
