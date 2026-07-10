#!/usr/bin/env python3
"""Plot memory usage metrics from a benchmark CSV.

The script expects columns like:
    frame,restart,allocated_gb,reserved_gb,max_allocated_gb,max_reserved_gb,cpu_gb,frame_time_ms,num_objects

It saves a PNG plot with GPU memory metrics on one panel and CPU memory on a
second panel. Restart frames are marked with vertical dashed lines.

Usage:
    python scripts/plot_memory_usage.py path/to/memory.csv
    python scripts/plot_memory_usage.py path/to/memory.csv --output memory.png
    python scripts/plot_memory_usage.py path/to/memory.csv --show
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_METRICS = [
    ("allocated_gb", "Allocated GB"),
    ("reserved_gb", "Reserved GB"),
    ("max_allocated_gb", "Max Allocated GB"),
    ("max_reserved_gb", "Max Reserved GB"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot memory allocation and reserved usage from a CSV benchmark file."
    )
    parser.add_argument("--csv_path", type=Path, help="Path to the benchmark CSV file")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output PNG path. Defaults to <csv_stem>_memory_plot.png next to the CSV.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the plot interactively after saving it.",
    )
    return parser.parse_args()


def load_data(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if "frame" not in df.columns:
        raise ValueError("CSV must contain a 'frame' column")

    if "restart" not in df.columns:
        df["restart"] = False

    return df.sort_values("frame").reset_index(drop=True)


def _plot_restart_markers(ax, restart_frames: list[int]) -> None:
    for idx, frame in enumerate(restart_frames):
        ax.axvline(
            frame,
            color="tab:red",
            linestyle="--",
            linewidth=1.0,
            alpha=0.55,
            label="Restart" if idx == 0 else None,
        )


def plot_memory(df: pd.DataFrame, csv_path: Path, output_path: Path, show: bool) -> None:
    fig, (ax_gpu, ax_cpu) = plt.subplots(
        2,
        1,
        figsize=(14, 8),
        sharex=True,
        constrained_layout=True,
    )

    frame_values = df["frame"]
    restart_frames = df.loc[df["restart"].astype(bool), "frame"].tolist()

    for column, label in DEFAULT_METRICS:
        if column in df.columns:
            ax_gpu.plot(frame_values, df[column], linewidth=1.8, label=label)

    if "cpu_gb" in df.columns:
        ax_cpu.plot(frame_values, df["cpu_gb"], color="tab:green", linewidth=1.8, label="CPU GB")

    for axis in (ax_gpu, ax_cpu):
        _plot_restart_markers(axis, restart_frames)
        axis.grid(True, alpha=0.25)
        axis.legend(loc="best")

    ax_gpu.set_title(f"Memory usage over time: {csv_path.name}")
    ax_gpu.set_ylabel("GPU memory (GB)")
    ax_cpu.set_ylabel("CPU memory (GB)")
    ax_cpu.set_xlabel("Frame")

    fig.suptitle("Memory allocation and reserved usage", fontsize=14, fontweight="bold")
    fig.savefig(output_path, dpi=160, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)


def main() -> None:
    args = parse_args()
    df = load_data(args.csv_path)

    output_path = args.output
    if output_path is None:
        output_path = args.csv_path.with_name(f"{args.csv_path.stem}_memory_plot.png")

    plot_memory(df, args.csv_path, output_path, args.show)
    print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    main()