# LLM-Guided Monte Carlo Tree Search for Tic-Tac-Toe (Part 1)

This repository contains a demo-ready Part 1 implementation with:
- Tic-Tac-Toe engine
- Baseline MCTS agent (random rollout simulation)
- LLM-guided MCTS agent (LLM/heuristic leaf evaluation instead of rollout)
- CLI demo for human play, self-play, and baseline-vs-LLM matches

## Requirements

- Python 3.10+
- No third-party dependencies required

## Run Demo

From repo root:

```bash
python3 -m ttt_mcts.demo --play self --agent llm --iters 300
```

Useful commands:

```bash
# Human vs LLM-MCTS
python3 -m ttt_mcts.demo --play human --agent llm --iters 300

# Human vs Baseline MCTS
python3 -m ttt_mcts.demo --play human --agent baseline --iters 500

# Baseline vs LLM-MCTS match (alternating who starts)
python3 -m ttt_mcts.demo --play match --games 20 --iters 300

# LLM or baseline self-play
python3 -m ttt_mcts.demo --play self --agent llm --iters 300
python3 -m ttt_mcts.demo --play self --agent baseline --iters 500
```

CLI flags:
- `--agent {baseline,llm}`
- `--iters <int>`
- `--play {human,self,match}`
- `--games <int>` (used by `match` mode)

## LLM Tools Supported

### OpenAI-compatible Chat Completions
- `OPENAI_API_KEY` (required for this backend)
- `OPENAI_BASE_URL` (optional, default: `https://api.openai.com/v1`)
- `OPENAI_MODEL` (optional, default: `gpt-4o-mini`)

Example:

```bash
export OPENAI_API_KEY="your_key"
export OPENAI_MODEL="gpt-4o-mini"
python3 -m ttt_mcts.demo --play self --agent llm --iters 300
```

### Ollama (local HTTP)
- `OLLAMA_BASE_URL` (optional, default: `http://localhost:11434`)
- `OLLAMA_MODEL` (optional, default: `llama3.1`)

Example:

```bash
export TTT_LLM_PROVIDER="ollama"
export OLLAMA_MODEL="llama3.1"
python3 -m ttt_mcts.demo --play self --agent llm --iters 300
```

Backend selection behavior:
- `TTT_LLM_PROVIDER=ollama` (default): Ollama first, then OpenAI (if key exists).
- `TTT_LLM_PROVIDER=openai`: OpenAI first (if key exists), then Ollama.
- `TTT_LLM_PROVIDER=auto`: OpenAI first (if key exists), then Ollama.
- If neither is reachable, heuristic fallback is used automatically.

## Exact Prompt Template Used

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

Expected strict JSON response keys:
- `value` in `[0,1]`
- `confidence` in `[0,1]`
- `best_move` in `0..8` or `-1`
- `reason` (<= 30 words)

## Caching and Fallback Behavior

- LLM-guided MCTS caches evaluations by `(board_state_string, player_to_move)`.
- Repeated visits to the same leaf do not trigger repeated API calls.
- If API key is missing, provider is unavailable, response is malformed, or endpoint is unreachable, a deterministic heuristic evaluator is used.
- This ensures demo commands run end-to-end even without network/model access.

## Short Reflection (Part 1)

The LLM tooling was most useful for structuring and validating the interface between search and evaluation. It accelerated getting from abstract requirements to a concrete and testable module split. Prompt design mattered because strict JSON output is fragile without explicit constraints. Caching made LLM guidance practical by reducing repeated calls on transposed states. The fallback heuristic was essential for reliability in offline demo conditions. OpenAI-compatible and Ollama support gave flexibility across cloud and local environments. The biggest tradeoff was balancing robustness and minimal code size. For Part 1, the tooling was productive when paired with deterministic safeguards.
