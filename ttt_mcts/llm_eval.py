"""LLM and fallback position evaluation for Tic-Tac-Toe."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Callable, Dict, List, Optional, Tuple

from .game import EMPTY, GameState, check_terminal, legal_moves, winner

PROMPT_TEMPLATE = """You are evaluating a Tic-Tac-Toe position.
Board (rows):
{row1}
{row2}
{row3}
Player to move: {player}
Return JSON only with keys: value, confidence, best_move, reason.
value must be a number in [0,1] representing the probability the player-to-move eventually wins.
confidence must be in [0,1].
best_move is 0-8 row-major or -1 if terminal.
reason must be <= 30 words."""


def _post_json(url: str, payload: Dict[str, object], headers: Optional[Dict[str, str]] = None, timeout: float = 15.0) -> Dict[str, object]:
    """POST JSON and decode JSON response."""
    req_headers = {"Content-Type": "application/json"}
    if headers:
        req_headers.update(headers)

    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url=url, data=data, headers=req_headers, method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = response.read().decode("utf-8")
    return json.loads(body)


def _extract_json_object(raw: str) -> Dict[str, object]:
    """Parse strict JSON output; tolerate extra wrappers if provider misbehaves."""
    text = raw.strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    first = text.find("{")
    last = text.rfind("}")
    if first != -1 and last != -1 and last > first:
        snippet = text[first : last + 1]
        obj = json.loads(snippet)
        if isinstance(obj, dict):
            return obj

    raise ValueError("LLM response is not valid JSON object")


def _clip_float(value: object, default: float = 0.5) -> float:
    """Convert to float and clamp to [0,1]."""
    try:
        num = float(value)
    except (TypeError, ValueError):
        num = default
    return min(1.0, max(0.0, num))


def _coerce_best_move(value: object) -> int:
    """Convert best_move to int and keep in valid range."""
    try:
        move = int(value)
    except (TypeError, ValueError):
        move = -1
    if move == -1 or 0 <= move <= 8:
        return move
    return -1


def _normalize_result(payload: Dict[str, object]) -> Dict[str, object]:
    """Normalize provider output to required schema."""
    reason = str(payload.get("reason", "No reason provided"))
    words = reason.split()
    if len(words) > 30:
        reason = " ".join(words[:30])

    return {
        "value": _clip_float(payload.get("value", 0.5), default=0.5),
        "confidence": _clip_float(payload.get("confidence", 0.0), default=0.0),
        "best_move": _coerce_best_move(payload.get("best_move", -1)),
        "reason": reason,
    }


def _format_prompt(board_text: str, player: str) -> str:
    """Fill required prompt template."""
    rows = board_text.splitlines()
    if len(rows) != 3 or any(len(r) != 3 for r in rows):
        raise ValueError("board_text must be 3 lines with 3 chars each")
    return PROMPT_TEMPLATE.format(row1=rows[0], row2=rows[1], row3=rows[2], player=player)


def _openai_eval(prompt: str) -> Dict[str, object]:
    """Evaluate position using OpenAI-compatible chat completion API."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY missing")

    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    url = f"{base_url}/chat/completions"

    response = _post_json(
        url=url,
        headers={"Authorization": f"Bearer {api_key}"},
        payload={
            "model": model,
            "temperature": 0,
            "messages": [{"role": "user", "content": prompt}],
        },
    )

    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError("OpenAI response missing choices")

    first = choices[0]
    if not isinstance(first, dict):
        raise RuntimeError("OpenAI response malformed")

    message = first.get("message")
    if not isinstance(message, dict):
        raise RuntimeError("OpenAI response missing message")

    content = message.get("content", "")
    if not isinstance(content, str):
        raise RuntimeError("OpenAI response content is not text")

    parsed = _extract_json_object(content)
    return _normalize_result(parsed)


def _ollama_eval(prompt: str) -> Dict[str, object]:
    """Evaluate position using local Ollama chat API."""
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    model = os.getenv("OLLAMA_MODEL", "llama3.1")
    url = f"{base_url}/api/chat"

    response = _post_json(
        url=url,
        payload={
            "model": model,
            "stream": False,
            "messages": [{"role": "user", "content": prompt}],
            "options": {"temperature": 0},
        },
    )

    message = response.get("message")
    if not isinstance(message, dict):
        raise RuntimeError("Ollama response missing message")

    content = message.get("content", "")
    if not isinstance(content, str):
        raise RuntimeError("Ollama response content is not text")

    parsed = _extract_json_object(content)
    return _normalize_result(parsed)


def _player_winning_moves(state: GameState, player: str) -> List[int]:
    """Return moves that immediately win for player."""
    wins: List[int] = []
    for move in legal_moves(state):
        next_board = state.board[:move] + player + state.board[move + 1 :]
        next_state = GameState(board=next_board, player=state.player)
        if winner(next_state) == player:
            wins.append(move)
    return wins


def _find_fork_block(board: str, player: str) -> Optional[int]:
    """Simple tactical move ordering for fallback."""
    preferred = [4, 0, 2, 6, 8, 1, 3, 5, 7]
    for move in preferred:
        if board[move] == EMPTY:
            return move
    return None


def _heuristic_eval(board_text: str, player: str) -> Dict[str, object]:
    """Rule-based fallback when LLM backends are unavailable."""
    board = "".join(board_text.splitlines())
    state = GameState(board=board, player=player)

    if check_terminal(state):
        w = winner(state)
        if w is None:
            return {"value": 0.5, "confidence": 1.0, "best_move": -1, "reason": "Terminal draw position."}
        return {
            "value": 1.0 if w == player else 0.0,
            "confidence": 1.0,
            "best_move": -1,
            "reason": "Terminal position with known winner.",
        }

    opponent = "O" if player == "X" else "X"
    own_wins = _player_winning_moves(state, player)
    if own_wins:
        return {
            "value": 0.95,
            "confidence": 0.75,
            "best_move": own_wins[0],
            "reason": "Immediate winning move available.",
        }

    opp_wins = _player_winning_moves(state, opponent)
    if opp_wins:
        return {
            "value": 0.45,
            "confidence": 0.65,
            "best_move": opp_wins[0],
            "reason": "Blocking opponent immediate win.",
        }

    best_move = _find_fork_block(board, player)
    center_bonus = 0.55 if board[4] == EMPTY else 0.5
    return {
        "value": center_bonus,
        "confidence": 0.35,
        "best_move": -1 if best_move is None else best_move,
        "reason": "Heuristic fallback favors center, corners, then edges.",
    }


def evaluate_position(board_text: str, player: str) -> Dict[str, object]:
    """Return dict with keys: value, confidence, best_move, reason."""
    prompt = _format_prompt(board_text=board_text, player=player)

    provider_pref = os.getenv("TTT_LLM_PROVIDER", "ollama").strip().lower()
    has_openai_key = bool(os.getenv("OPENAI_API_KEY"))
    providers: List[Tuple[str, Callable[[str], Dict[str, object]]]] = []
    if provider_pref == "openai":
        if has_openai_key:
            providers.append(("openai", _openai_eval))
        providers.append(("ollama", _ollama_eval))
    elif provider_pref == "auto":
        if has_openai_key:
            providers.append(("openai", _openai_eval))
        providers.append(("ollama", _ollama_eval))
    else:
        providers.append(("ollama", _ollama_eval))
        if has_openai_key:
            providers.append(("openai", _openai_eval))

    for _, provider in providers:
        try:
            return provider(prompt)
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ValueError, RuntimeError, json.JSONDecodeError):
            continue

    return _heuristic_eval(board_text=board_text, player=player)
