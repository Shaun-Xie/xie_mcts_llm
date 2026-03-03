---
marp: true
theme: default
paginate: true
---

# PA4 Part 2
## LLM-Guided MCTS for Tic-Tac-Toe

- Course project follow-up to Part 1
- Goal: compare baseline rollout MCTS vs LLM-guided MCTS
- Added minimax oracle, experiment harness, and reproducible metrics

---

# Overview

- **Part 1 baseline**: Selection + Expansion + Random Rollout + Backprop
- **Part 1 LLM-guided**: Selection + Expansion + LLM/heuristic eval + Backprop
- **Part 2 additions**:
  - perfect minimax player
  - experiment CLI + CSV/JSON outputs
  - plotting CLI
  - written analysis and slides

---

# MCTS Recap (4 Phases)

1. **Selection**: traverse by UCB1
2. **Expansion**: add one untried child
3. **Evaluation**:
   - baseline: random rollout
   - llm-guided: `evaluate_position(...)`
4. **Backpropagation**: flip perspective each ply (`parent = 1 - child`)

---

# What Changed for LLM Guidance

- Replaced rollout with evaluator returning:
  - `value`, `confidence`, `best_move`, `reason`
- Cache key: `(board_state_string, player_to_move)`
- Added explicit runtime modes:
  - `auto`, `openai`, `ollama`, `heuristic`
- Added cache stats:
  - misses (= eval calls), hits, hit-rate, cache size

---

# Experimental Setup

- Iterations: `10, 50, 100, 500`
- Matchups per iteration:
  - baseline vs llm
  - baseline vs minimax
  - llm vs minimax
- Controlled conditions:
  - fixed seed
  - alternating first player (half games each side)
- Metrics:
  - W/D/L
  - avg time per move
  - avg time per iteration
  - avg chosen-move value estimate
  - LLM calls + cache hit rate

---

# Sample Results (80 games/matchup)

| Iters | Baseline Win | LLM Win | Draw |
|---:|---:|---:|---:|
| 10 | 0.2375 | 0.5500 | 0.2125 |
| 50 | 0.3375 | 0.4375 | 0.2250 |
| 100 | 0.4500 | 0.0250 | 0.5250 |
| 500 | 0.3875 | 0.3375 | 0.2750 |

- Run used `--llm_mode heuristic` for offline reproducibility.

---

# Against Perfect Minimax

- Baseline vs minimax: **0 wins** at all iteration settings
- LLM vs minimax: **0 wins** at all iteration settings
- Some draws occur at low iterations; vanish in higher-iteration sample runs here

Interpretation:
- Tic-Tac-Toe is solved and tiny
- minimax remains strict upper bound on quality

---

# Cost and Latency Tradeoffs

- Move latency scales with iteration budget for both MCTS variants
- LLM cache hit-rate increases with deeper search
  - sample: `0.44` at 10 iters -> `0.95` at 500 iters
- In larger domains, evaluator priors can reduce bad exploration
- In Tic-Tac-Toe, cheap rollouts already work well

---

# RAP Connection

- RAP idea: model-generated reward/value signals guide search
- This project instantiates that idea by replacing rollout with evaluator values
- Empirical takeaway:
  - strong gains are limited in tiny solved games
  - expected payoff grows with branching factor and depth

---

# Reproduce + Demo

```bash
# Experiments
python3 -m ttt_mcts.experiments --output_dir results --games 200 --seed 0 --llm_mode heuristic

# Plots
python3 -m ttt_mcts.plots --input results/results.csv --output_dir results/plots

# Demo game
python3 -m ttt_mcts.demo --play human --agent llm --iters 300
```

Files:
- `REPORT.md`
- `results/results.csv`, `results/per_game.jsonl`, `results/metadata.json`
- `README.md` for prompts/tools/repro details
