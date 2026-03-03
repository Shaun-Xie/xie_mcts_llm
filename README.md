# LLM-Guided MCTS for Tic-Tac-Toe (PA4 Parts 1-2)

This repository contains:
- **Part 1**: game engine, baseline MCTS, LLM-guided MCTS, and demo CLI
- **Part 2**: minimax oracle, experiment harness, plotting pipeline, written report, and slide deck

## Project Structure

- `ttt_mcts/game.py`: Tic-Tac-Toe engine
- `ttt_mcts/mcts.py`: baseline and LLM-guided MCTS agents
- `ttt_mcts/llm_eval.py`: pluggable evaluator (OpenAI-compatible / Ollama / heuristic)
- `ttt_mcts/demo.py`: interactive demo CLI
- `ttt_mcts/minimax.py`: perfect-play minimax agent
- `ttt_mcts/experiments.py`: Part 2 experiment runner
- `ttt_mcts/plots.py`: chart generator from `results.csv`
- `REPORT.md`: 1-2 page analysis writeup
- `SLIDES.md`: slide deck content (Marp markdown)

## Requirements

- Python 3.10+
- `matplotlib` (optional but recommended for nicer charts in `ttt_mcts.plots`)

Install plotting dependency:

```bash
python3 -m pip install -r requirements.txt
```

## Part 1 Demo Commands

```bash
# Human vs LLM-guided MCTS
python3 -m ttt_mcts.demo --play human --agent llm --iters 300

# Human vs baseline MCTS
python3 -m ttt_mcts.demo --play human --agent baseline --iters 500

# Baseline vs LLM match mode
python3 -m ttt_mcts.demo --play match --games 20 --iters 300

# Self-play
python3 -m ttt_mcts.demo --play self --agent llm --iters 300
```

CLI flags:
- `--agent {baseline,llm}`
- `--iters <int>`
- `--play {human,self,match}`
- `--games <int>`

## LLM Backends and Env Vars

### OpenAI-compatible chat completions
- `OPENAI_API_KEY` (required)
- `OPENAI_BASE_URL` (optional, default: `https://api.openai.com/v1`)
- `OPENAI_MODEL` (optional, default: `gpt-4o-mini`)

### Ollama
- `OLLAMA_BASE_URL` (optional, default: `http://localhost:11434`)
- `OLLAMA_MODEL` (optional, default: `llama3.1`)

### Provider selection
- `TTT_LLM_PROVIDER=ollama` (default in code path): try Ollama first, then OpenAI if key exists
- `TTT_LLM_PROVIDER=openai`: try OpenAI first, then Ollama
- `TTT_LLM_PROVIDER=auto`: OpenAI first if key exists, then Ollama
- `TTT_LLM_PROVIDER=heuristic`: force offline heuristic evaluator (no API calls)

## EXACT Prompt Template Used

```text
You are evaluating a Tic-Tac-Toe position.
Board (rows):
{row1}
{row2}
{row3}
Player to move: {player}
Return JSON only with keys: value, confidence, best_move, reason.
value must be a number in [0,1] representing the probability the player-to-move eventually wins.
confidence must be in [0,1].
best_move is 0-8 row-major or -1 if terminal.
reason must be <= 30 words.
```

Expected strict JSON keys:
- `value` in `[0,1]`
- `confidence` in `[0,1]`
- `best_move` in `0..8` or `-1`
- `reason` (<= 30 words)

## Caching and Fallback

- LLM-guided MCTS caches leaf evaluations by `(board_state_string, player_to_move)`.
- Cache stats are tracked for experiments:
  - calls (cache misses)
  - cache hits / misses
  - hit rate
- If providers are unavailable or output is invalid, heuristic fallback is used.
- `TTT_LLM_PROVIDER=heuristic` forces fully offline runs.

## Part 2: Reproduce Experiments

### 1) Run experiments

Default assignment-style configuration:

```bash
python3 -m ttt_mcts.experiments --output_dir results --games 200 --seed 0
```

Offline reproducible mode (no API dependence):

```bash
python3 -m ttt_mcts.experiments --output_dir results --games 200 --seed 0 --llm_mode heuristic
```

Arguments:
- `--iters 10,50,100,500` (default)
- `--games <int>` (default `200`)
- `--seed <int>` (default `0`)
- `--llm_mode {auto,openai,ollama,heuristic}` (default `auto`)
- `--output_dir results/`

Outputs:
- `results/results.csv` (one row per matchup per iteration count)
- `results/per_game.jsonl` (per-game logs)
- `results/metadata.json` (runtime/config metadata)

### 2) Generate charts

```bash
python3 -m ttt_mcts.plots --input results/results.csv --output_dir results/plots
```

Behavior:
- If `matplotlib` is installed, standard matplotlib charts are generated.
- If not installed, a stdlib fallback still generates PNG charts.

Generated charts:
- `results/plots/winrate_baseline_vs_llm.png`
- `results/plots/winrate_vs_minimax.png`
- `results/plots/time_per_move_vs_iters.png`
- `results/plots/llm_calls_and_cache_vs_iters.png`

## Deliverables in Repo

- Analysis: `REPORT.md`
- Slides: `SLIDES.md`
- Code: `ttt_mcts/` package (Part 1 + Part 2)

## Tools Used

- Codex (implementation assistance)
- OpenAI-compatible API (optional runtime backend)
- Ollama HTTP API (optional runtime backend)
- Heuristic fallback evaluator (offline reproducibility)
