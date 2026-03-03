"""Plot generation for Part 2 experiment results."""

from __future__ import annotations

import argparse
import binascii
import csv
import struct
import zlib
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, Dict, Iterable, List, Mapping, Sequence, Tuple


try:  # pragma: no cover - optional runtime dependency
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
except ImportError:  # pragma: no cover - optional runtime dependency
    plt = None
    PercentFormatter = None


Color = Tuple[int, int, int]
MATPLOTLIB_COLORS = {
    "baseline": "#1f77b4",
    "llm": "#ff7f0e",
    "draw": "#2ca02c",
    "minimax": "#d62728",
    "calls": "#1f77b4",
    "cache": "#ff7f0e",
}


def _to_float(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _to_int(value: str) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def _read_results(path: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "iters": _to_int(row.get("iters", "0")),
                    "games": _to_int(row.get("games", "0")),
                    "matchup": row.get("matchup", ""),
                    "agent_a": row.get("agent_a", ""),
                    "agent_b": row.get("agent_b", ""),
                    "agent_a_wins": _to_int(row.get("agent_a_wins", "0")),
                    "agent_b_wins": _to_int(row.get("agent_b_wins", "0")),
                    "draws": _to_int(row.get("draws", "0")),
                    "agent_a_avg_time_per_move": _to_float(row.get("agent_a_avg_time_per_move", "0")),
                    "agent_b_avg_time_per_move": _to_float(row.get("agent_b_avg_time_per_move", "0")),
                    "agent_a_avg_time_per_iteration": _to_float(row.get("agent_a_avg_time_per_iteration", "0")),
                    "agent_b_avg_time_per_iteration": _to_float(row.get("agent_b_avg_time_per_iteration", "0")),
                    "llm_calls_per_move": _to_float(row.get("llm_calls_per_move", "0")),
                    "llm_cache_hit_rate": _to_float(row.get("llm_cache_hit_rate", "0")),
                    "llm_calls": _to_int(row.get("llm_calls", "0")),
                }
            )
    return rows


def _row_by_matchup(rows: Iterable[Dict[str, object]], matchup: str) -> Dict[int, Dict[str, object]]:
    out: Dict[int, Dict[str, object]] = {}
    for row in rows:
        if row["matchup"] == matchup:
            out[int(row["iters"])] = row
    return out


def _apply_matplotlib_style() -> None:
    """Apply consistent, presentation-friendly matplotlib styling."""
    plt.style.use("default")
    plt.rcParams.update(
        {
            "figure.facecolor": "#f9fafb",
            "axes.facecolor": "white",
            "axes.edgecolor": "#4b5563",
            "axes.labelcolor": "#111827",
            "xtick.color": "#111827",
            "ytick.color": "#111827",
            "axes.titlesize": 14,
            "axes.titleweight": "semibold",
            "axes.labelsize": 12,
            "legend.frameon": True,
            "legend.facecolor": "#f3f4f6",
            "legend.edgecolor": "#d1d5db",
            "grid.color": "#d1d5db",
            "grid.linestyle": "--",
            "grid.alpha": 0.65,
            "font.size": 11,
            "axes.titlepad": 12,
            }
    )


def _save_win_rate_charts(rows: List[Dict[str, object]], output_dir: Path) -> None:
    baseline_llm = _row_by_matchup(rows, "baseline_vs_llm")
    if baseline_llm:
        x = sorted(baseline_llm.keys())
        baseline = [baseline_llm[i]["agent_a_wins"] / baseline_llm[i]["games"] for i in x]
        llm = [baseline_llm[i]["agent_b_wins"] / baseline_llm[i]["games"] for i in x]
        draws = [baseline_llm[i]["draws"] / baseline_llm[i]["games"] for i in x]
        games = int(baseline_llm[x[0]]["games"])

        fig, ax = plt.subplots(figsize=(10.5, 6.2), constrained_layout=True)
        ax.plot(x, baseline, marker="o", linewidth=2.4, markersize=7, color=MATPLOTLIB_COLORS["baseline"], label="Baseline Win Rate")
        ax.plot(x, llm, marker="o", linewidth=2.4, markersize=7, color=MATPLOTLIB_COLORS["llm"], label="LLM Win Rate")
        ax.plot(x, draws, marker="o", linewidth=2.4, markersize=7, color=MATPLOTLIB_COLORS["draw"], label="Draw Rate")

        ax.set_xlabel("MCTS Iterations Per Move")
        ax.set_ylabel("Rate")
        ax.set_xticks(x)
        ax.set_ylim(0, 1.02)
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
        ax.set_title("Baseline vs LLM Win/Draw Rates")
        ax.text(
            0.01,
            1.02,
            f"{games} games per setting (starting player alternated)",
            transform=ax.transAxes,
            fontsize=9,
            color="#4b5563",
        )
        ax.grid(True)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3)
        fig.savefig(output_dir / "winrate_baseline_vs_llm.png", dpi=220)
        plt.close(fig)

    baseline_minimax = _row_by_matchup(rows, "baseline_vs_minimax")
    llm_minimax = _row_by_matchup(rows, "llm_vs_minimax")
    if baseline_minimax or llm_minimax:
        x = sorted(set(baseline_minimax.keys()) | set(llm_minimax.keys()))
        baseline_vs_minimax = [
            (baseline_minimax[i]["agent_a_wins"] / baseline_minimax[i]["games"]) if i in baseline_minimax else 0.0
            for i in x
        ]
        llm_vs_minimax = [
            (llm_minimax[i]["agent_a_wins"] / llm_minimax[i]["games"]) if i in llm_minimax else 0.0
            for i in x
        ]

        fig, ax = plt.subplots(figsize=(10.5, 6.0), constrained_layout=True)
        ax.plot(
            x,
            baseline_vs_minimax,
            marker="o",
            linewidth=2.4,
            markersize=7,
            color=MATPLOTLIB_COLORS["baseline"],
            label="Baseline Wins vs Minimax",
        )
        ax.plot(
            x,
            llm_vs_minimax,
            marker="o",
            linewidth=2.4,
            markersize=7,
            color=MATPLOTLIB_COLORS["llm"],
            label="LLM Wins vs Minimax",
        )

        ax.set_xlabel("MCTS Iterations Per Move")
        ax.set_ylabel("Win Rate")
        ax.set_xticks(x)
        ax.set_ylim(0, 1.02)
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
        ax.set_title("MCTS Agents vs Perfect Minimax")
        ax.text(0.01, 1.02, "Expected near-zero wins in a solved game", transform=ax.transAxes, fontsize=9, color="#4b5563")
        ax.grid(True)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)
        fig.savefig(output_dir / "winrate_vs_minimax.png", dpi=220)
        plt.close(fig)


def _save_timing_chart(rows: List[Dict[str, object]], output_dir: Path) -> None:
    per_agent_time: DefaultDict[str, DefaultDict[int, List[float]]] = defaultdict(lambda: defaultdict(list))

    for row in rows:
        iters = int(row["iters"])
        agent_a = str(row["agent_a"])
        agent_b = str(row["agent_b"])
        per_agent_time[agent_a][iters].append(float(row["agent_a_avg_time_per_move"]))
        per_agent_time[agent_b][iters].append(float(row["agent_b_avg_time_per_move"]))

    fig, ax = plt.subplots(figsize=(10.5, 6.0), constrained_layout=True)
    for agent in sorted(per_agent_time.keys()):
        x = sorted(per_agent_time[agent].keys())
        y_sec = [sum(per_agent_time[agent][i]) / len(per_agent_time[agent][i]) for i in x]
        y_ms = [v * 1000.0 for v in y_sec]
        color = MATPLOTLIB_COLORS.get(agent, None)
        ax.plot(x, y_ms, marker="o", linewidth=2.4, markersize=7, label=agent.upper(), color=color)

    ax.set_xlabel("MCTS Iterations Per Move")
    ax.set_ylabel("Avg Time Per Move (ms)")
    ax.set_xticks(sorted({iters for iters_by_agent in per_agent_time.values() for iters in iters_by_agent.keys()}))
    ax.set_title("Move Latency vs Iteration Budget")
    ax.text(0.01, 1.02, "Lower is better", transform=ax.transAxes, fontsize=9, color="#4b5563")
    ax.grid(True)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=max(1, len(per_agent_time)))
    fig.savefig(output_dir / "time_per_move_vs_iters.png", dpi=220)
    plt.close(fig)


def _save_llm_cache_chart(rows: List[Dict[str, object]], output_dir: Path) -> None:
    llm_rows = [row for row in rows if int(row["llm_calls"]) > 0]
    if not llm_rows:
        return

    calls_per_move_by_iter: DefaultDict[int, List[float]] = defaultdict(list)
    hit_rate_by_iter: DefaultDict[int, List[float]] = defaultdict(list)

    for row in llm_rows:
        iters = int(row["iters"])
        calls_per_move_by_iter[iters].append(float(row["llm_calls_per_move"]))
        hit_rate_by_iter[iters].append(float(row["llm_cache_hit_rate"]))

    x = sorted(calls_per_move_by_iter.keys())
    calls = [sum(calls_per_move_by_iter[i]) / len(calls_per_move_by_iter[i]) for i in x]
    hits = [sum(hit_rate_by_iter[i]) / len(hit_rate_by_iter[i]) for i in x]

    fig, ax1 = plt.subplots(figsize=(10.5, 6.0), constrained_layout=True)
    ax2 = ax1.twinx()

    ax1.plot(x, calls, marker="o", linewidth=2.4, markersize=7, color=MATPLOTLIB_COLORS["calls"], label="LLM Calls per Move")
    ax2.plot(x, hits, marker="s", linewidth=2.4, markersize=7, color=MATPLOTLIB_COLORS["cache"], label="LLM Cache Hit Rate")
    ax1.set_xlabel("MCTS Iterations Per Move")
    ax1.set_ylabel("Calls Per Move", color=MATPLOTLIB_COLORS["calls"])
    ax2.set_ylabel("Cache Hit Rate", color=MATPLOTLIB_COLORS["cache"])
    ax1.set_xticks(x)
    ax2.set_ylim(0, 1)
    ax2.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax1.set_title("LLM Usage and Caching Efficiency")
    ax1.text(0.01, 1.02, "Calls should drop with better cache reuse", transform=ax1.transAxes, fontsize=9, color="#4b5563")
    ax1.grid(True)

    lines = ax1.get_lines() + ax2.get_lines()
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)

    fig.savefig(output_dir / "llm_calls_and_cache_vs_iters.png", dpi=220)
    plt.close(fig)


# -------- Fallback renderer (stdlib only) --------


def _new_canvas(width: int, height: int, color: Color = (255, 255, 255)) -> bytearray:
    buf = bytearray(width * height * 3)
    r, g, b = color
    for i in range(0, len(buf), 3):
        buf[i] = r
        buf[i + 1] = g
        buf[i + 2] = b
    return buf


def _set_px(buf: bytearray, width: int, height: int, x: int, y: int, color: Color) -> None:
    if x < 0 or y < 0 or x >= width or y >= height:
        return
    idx = (y * width + x) * 3
    buf[idx] = color[0]
    buf[idx + 1] = color[1]
    buf[idx + 2] = color[2]


def _draw_line(buf: bytearray, width: int, height: int, x0: int, y0: int, x1: int, y1: int, color: Color) -> None:
    dx = abs(x1 - x0)
    sx = 1 if x0 < x1 else -1
    dy = -abs(y1 - y0)
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    x, y = x0, y0
    while True:
        _set_px(buf, width, height, x, y, color)
        if x == x1 and y == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x += sx
        if e2 <= dx:
            err += dx
            y += sy


def _draw_polyline(buf: bytearray, width: int, height: int, pts: Sequence[Tuple[int, int]], color: Color) -> None:
    for i in range(1, len(pts)):
        _draw_line(buf, width, height, pts[i - 1][0], pts[i - 1][1], pts[i][0], pts[i][1], color)


def _write_png(path: Path, width: int, height: int, rgb: bytes) -> None:
    raw = bytearray()
    row_len = width * 3
    for y in range(height):
        raw.append(0)  # filter byte
        start = y * row_len
        raw.extend(rgb[start : start + row_len])

    def chunk(name: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + name
            + data
            + struct.pack(">I", binascii.crc32(name + data) & 0xFFFFFFFF)
        )

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    idat = zlib.compress(bytes(raw), level=9)

    png = bytearray()
    png.extend(b"\x89PNG\r\n\x1a\n")
    png.extend(chunk(b"IHDR", ihdr))
    png.extend(chunk(b"IDAT", idat))
    png.extend(chunk(b"IEND", b""))

    path.write_bytes(bytes(png))


def _render_simple_line_chart(path: Path, x_values: Sequence[int], series: Mapping[str, Sequence[float]]) -> None:
    width, height = 880, 520
    left, right, top, bottom = 70, 20, 25, 55
    plot_w = width - left - right
    plot_h = height - top - bottom

    buf = _new_canvas(width, height, color=(255, 255, 255))

    axis_color: Color = (30, 30, 30)
    _draw_line(buf, width, height, left, top, left, top + plot_h, axis_color)
    _draw_line(buf, width, height, left, top + plot_h, left + plot_w, top + plot_h, axis_color)

    all_y = [v for values in series.values() for v in values]
    if not all_y:
        _write_png(path, width, height, bytes(buf))
        return
    y_min = min(all_y)
    y_max = max(all_y)
    if y_max <= y_min:
        y_max = y_min + 1.0

    x_min = min(x_values)
    x_max = max(x_values)
    if x_max <= x_min:
        x_max = x_min + 1

    def map_x(xv: int) -> int:
        return left + int(((xv - x_min) / (x_max - x_min)) * plot_w)

    def map_y(yv: float) -> int:
        frac = (yv - y_min) / (y_max - y_min)
        return top + plot_h - int(frac * plot_h)

    colors: List[Color] = [
        (31, 119, 180),
        (255, 127, 14),
        (44, 160, 44),
        (214, 39, 40),
        (148, 103, 189),
    ]

    for idx, (_, values) in enumerate(series.items()):
        pts = [(map_x(x_values[i]), map_y(values[i])) for i in range(len(x_values))]
        _draw_polyline(buf, width, height, pts, colors[idx % len(colors)])

    _write_png(path, width, height, bytes(buf))


def _save_fallback_plots(rows: List[Dict[str, object]], output_dir: Path) -> None:
    baseline_llm = _row_by_matchup(rows, "baseline_vs_llm")
    if baseline_llm:
        x = sorted(baseline_llm.keys())
        _render_simple_line_chart(
            output_dir / "winrate_baseline_vs_llm.png",
            x,
            {
                "baseline": [baseline_llm[i]["agent_a_wins"] / baseline_llm[i]["games"] for i in x],
                "llm": [baseline_llm[i]["agent_b_wins"] / baseline_llm[i]["games"] for i in x],
                "draw": [baseline_llm[i]["draws"] / baseline_llm[i]["games"] for i in x],
            },
        )

    baseline_minimax = _row_by_matchup(rows, "baseline_vs_minimax")
    llm_minimax = _row_by_matchup(rows, "llm_vs_minimax")
    if baseline_minimax or llm_minimax:
        x = sorted(set(baseline_minimax.keys()) | set(llm_minimax.keys()))
        _render_simple_line_chart(
            output_dir / "winrate_vs_minimax.png",
            x,
            {
                "baseline_vs_minimax": [
                    (baseline_minimax[i]["agent_a_wins"] / baseline_minimax[i]["games"]) if i in baseline_minimax else 0.0
                    for i in x
                ],
                "llm_vs_minimax": [
                    (llm_minimax[i]["agent_a_wins"] / llm_minimax[i]["games"]) if i in llm_minimax else 0.0
                    for i in x
                ],
            },
        )

    per_agent_time: DefaultDict[str, DefaultDict[int, List[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        iters = int(row["iters"])
        a, b = str(row["agent_a"]), str(row["agent_b"])
        per_agent_time[a][iters].append(float(row["agent_a_avg_time_per_move"]))
        per_agent_time[b][iters].append(float(row["agent_b_avg_time_per_move"]))

    x = sorted({int(r["iters"]) for r in rows})
    series: Dict[str, List[float]] = {}
    for agent in sorted(per_agent_time.keys()):
        series[agent] = [
            (sum(per_agent_time[agent][i]) / len(per_agent_time[agent][i])) if per_agent_time[agent][i] else 0.0
            for i in x
        ]
    if series:
        _render_simple_line_chart(output_dir / "time_per_move_vs_iters.png", x, series)

    llm_rows = [row for row in rows if int(row["llm_calls"]) > 0]
    if llm_rows:
        calls_per_move_by_iter: DefaultDict[int, List[float]] = defaultdict(list)
        hit_rate_by_iter: DefaultDict[int, List[float]] = defaultdict(list)
        for row in llm_rows:
            iters = int(row["iters"])
            calls_per_move_by_iter[iters].append(float(row["llm_calls_per_move"]))
            hit_rate_by_iter[iters].append(float(row["llm_cache_hit_rate"]))

        x2 = sorted(calls_per_move_by_iter.keys())
        calls = [sum(calls_per_move_by_iter[i]) / len(calls_per_move_by_iter[i]) for i in x2]
        hits = [sum(hit_rate_by_iter[i]) / len(hit_rate_by_iter[i]) for i in x2]
        _render_simple_line_chart(
            output_dir / "llm_calls_and_cache_vs_iters.png",
            x2,
            {
                "calls_per_move": calls,
                "cache_hit_rate": hits,
            },
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Part 2 plots from results.csv")
    parser.add_argument("--input", type=str, default="results/results.csv")
    parser.add_argument("--output_dir", type=str, default="results/plots")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = _read_results(input_path)
    if plt is not None:
        _apply_matplotlib_style()
        _save_win_rate_charts(rows, output_dir)
        _save_timing_chart(rows, output_dir)
        _save_llm_cache_chart(rows, output_dir)
        print(f"Saved matplotlib plots to: {output_dir}")
    else:
        _save_fallback_plots(rows, output_dir)
        print(f"matplotlib not available; saved fallback PNG plots to: {output_dir}")


if __name__ == "__main__":
    main()
