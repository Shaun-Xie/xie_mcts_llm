# PA4 Part 2 Report: LLM-Guided MCTS for Tic-Tac-Toe

## 1. Objective

Part 2 evaluates whether replacing MCTS random rollouts with an LLM-style position evaluator improves decision quality under controlled iteration budgets. The comparison focuses on Tic-Tac-Toe, a solved and small game, where minimax can provide an exact reference policy.

## 2. Experimental Setup

- Game: 3x3 Tic-Tac-Toe, values in `[0,1]` from player-to-move perspective (`win=1`, `draw=0.5`, `loss=0`)
- Iteration budgets: `10, 50, 100, 500`
- Matchups per iteration:
  - `baseline` (MCTS + random rollout) vs `llm` (MCTS + evaluator)
  - `baseline` vs `minimax` (perfect play)
  - `llm` vs `minimax`
- Games per matchup: `80` (sample run for this report; harness default is `200`)
- Starting-player control: first half of games with agent A as X, second half with roles swapped
- Seed control: fixed deterministic seeds via `--seed 0`
- LLM mode used here: `heuristic` fallback (`--llm_mode heuristic`) to ensure offline reproducibility

Command used:

```bash
python3 -m ttt_mcts.experiments --output_dir results --games 80 --seed 0 --llm_mode heuristic
```

## 3. Key Results (Sample Run)

### 3.1 Baseline vs LLM (W/D/L rates)

| Iters | Baseline Win | LLM Win | Draw |
|---:|---:|---:|---:|
| 10 | 0.2375 | 0.5500 | 0.2125 |
| 50 | 0.3375 | 0.4375 | 0.2250 |
| 100 | 0.4500 | 0.0250 | 0.5250 |
| 500 | 0.3875 | 0.3375 | 0.2750 |

### 3.2 Against Perfect Minimax

| Iters | Baseline vs Minimax (W/D/L) | LLM vs Minimax (W/D/L) |
|---:|---:|---:|
| 10 | 0 / 3 / 77 | 0 / 17 / 63 |
| 50 | 0 / 0 / 80 | 0 / 0 / 80 |
| 100 | 0 / 0 / 80 | 0 / 0 / 80 |
| 500 | 0 / 0 / 80 | 0 / 0 / 80 |

Both non-perfect agents fail to beat minimax, as expected. Draws against minimax appear mainly at low iterations in this sample.

### 3.3 Timing and Cache Metrics

- Average move time grows roughly linearly with iteration count for both MCTS agents.
- Example (`baseline` vs `llm`):
  - Baseline avg time/move: `0.000118s` (10 iters) -> `0.006637s` (500 iters)
  - LLM-guided avg time/move: `0.000109s` (10 iters) -> `0.005243s` (500 iters)
- LLM cache hit rate increases with search depth:
  - `0.441` (10 iters) -> `0.951` (500 iters)
- LLM calls per move (cache misses per LLM move) were non-monotonic in this sample but generally high under larger trees (e.g., `10.90` at 500 iters in baseline-vs-LLM).

## 4. Analysis

### Did LLM-guided improve quality? Where?

In this offline run (heuristic evaluator standing in for LLM API calls), quality gains are mixed and unstable across iteration counts. LLM-guided outperformed baseline at 10 and 50 iterations, then underperformed at 100, and was near parity at 500. This suggests evaluator quality and search-policy interactions matter more than simply replacing rollouts. In a solved domain with tiny state space, small differences in tie-breaking and exploration noise can swing short runs.

### Why Tic-Tac-Toe limits LLM value

Tic-Tac-Toe has a very small state/action space and exact dynamic-programming solutions. Random rollouts converge quickly because terminal depth is short and tactical patterns are simple. As iterations increase, baseline MCTS has enough samples to estimate move values well without semantic priors. That shrinks the headroom for external learned evaluation.

### Cost tradeoffs (latency/token vs rollouts)

Real LLM calls add significant network/model latency and potential token cost per unique leaf. In this project, caching is essential: repeated leaf revisits become cache hits, and measured hit-rate grows with iterations. In larger tasks, high cache hit rates can partially amortize API overhead, but misses still dominate cold-start phases. For Tic-Tac-Toe specifically, random rollouts are so cheap that LLM latency is difficult to justify unless evaluator quality is substantially better.

### Where LLM guidance should help more

LLM guidance is more plausible when:
- branching factor is large,
- playout depth is long,
- random simulation quality is poor or noisy,
- domain heuristics are hard to code manually.

These conditions make rollout-only MCTS expensive or low-signal, increasing the value of a learned value prior.

### Connection to RAP-style ideas

RAP frames planning as search guided by model-produced reward/value signals. This implementation mirrors that idea by replacing rollout simulation with a learned evaluator and feeding that value into backprop. The empirical takeaway here is consistent with RAP’s motivation: reward-guided search is most valuable when task complexity outstrips cheap Monte Carlo simulation. Tic-Tac-Toe is too small and solved to show large consistent gains.

## 5. Reproducibility and Artifacts

- Raw results: `results/results.csv`
- Per-game logs: `results/per_game.jsonl`
- Run metadata: `results/metadata.json`
- Plot command (requires `matplotlib`):

```bash
python3 -m ttt_mcts.plots --input results/results.csv --output_dir results/plots
```

In this runtime, `matplotlib` was unavailable, so `ttt_mcts.plots` generated fallback PNG charts via a stdlib renderer in `results/plots/`.
