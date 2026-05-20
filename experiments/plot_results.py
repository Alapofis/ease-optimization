import argparse
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def _read(path: Path):
    if not path.exists():
        print(f"skip missing {path}")
        return None
    return pd.read_csv(path)


def _save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"saved {path}")


def plot_epsilon(results_dir: Path, figures_dir: Path):
    df = _read(results_dir / "epsilon_sweep.csv")
    if df is None or df.empty:
        return
    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.set_xscale("log")
    ax1.plot(df["epsilon"], df["t_solve"], marker="o", label="solve time")
    ax1.set_xlabel("epsilon")
    ax1.set_ylabel("solve time, s")
    ax2 = ax1.twinx()
    ax2.plot(df["epsilon"], df["recall"], marker="s", color="tab:green", label="recall")
    ax2.plot(df["epsilon"], df["ndcg"], marker="^", color="tab:orange", label="ndcg")
    ax2.set_ylabel("quality")
    ax1.invert_xaxis()
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [line.get_label() for line in lines], loc="best")
    _save(fig, figures_dir / "epsilon_tradeoff.png")


def plot_rank(results_dir: Path, figures_dir: Path):
    df = _read(results_dir / "rank_sweep.csv")
    if df is None or df.empty:
        return
    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(df["nystrom_rank"], df["t_total"], marker="o", label="total time")
    ax1.plot(df["nystrom_rank"], df["t_solve"], marker="s", label="solve time")
    ax1.set_xlabel("nystrom rank")
    ax1.set_ylabel("time, s")
    ax2 = ax1.twinx()
    ax2.plot(
        df["nystrom_rank"],
        df["mean_iterations"],
        marker="^",
        color="tab:green",
        label="mean iterations",
    )
    ax2.set_ylabel("mean iterations")
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [line.get_label() for line in lines], loc="best")
    _save(fig, figures_dir / "rank_time_iterations.png")


def plot_block(results_dir: Path, figures_dir: Path):
    df = _read(results_dir / "block_sweep.csv")
    if df is None or df.empty:
        return
    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(df["block_size"], df["t_solve"], marker="o", label="solve time")
    ax1.set_xlabel("block size")
    ax1.set_ylabel("solve time, s")
    ax2 = ax1.twinx()
    ax2.plot(df["block_size"], df["memory_gb"], marker="s", color="tab:red", label="memory")
    ax2.set_ylabel("memory, GB")
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [line.get_label() for line in lines], loc="best")
    _save(fig, figures_dir / "block_time_memory.png")


def plot_baselines(results_dir: Path, figures_dir: Path):
    df = _read(results_dir / "external_baselines.csv")
    if df is None or df.empty:
        return
    df = df[df["status"] == "ok"].copy()
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    x = range(len(df))
    ax.bar([i - 0.18 for i in x], df["recall"], width=0.36, label="recall")
    ax.bar([i + 0.18 for i in x], df["ndcg"], width=0.36, label="ndcg")
    ax.set_xticks(list(x))
    ax.set_xticklabels(df["method"], rotation=15, ha="right")
    ax.set_ylabel("quality")
    ax.legend()
    _save(fig, figures_dir / "external_baselines_quality.png")


def plot_internal_baselines(results_dir: Path, figures_dir: Path):
    df = _read(results_dir / "internal_ease_baselines.csv")
    if df is None or df.empty:
        return
    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    x = range(len(df))
    ax1.bar(x, df["t_total"], width=0.5, label="total time")
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(df["method"], rotation=20, ha="right")
    ax1.set_ylabel("time, s")
    ax2 = ax1.twinx()
    ax2.plot(x, df["ndcg"], marker="o", color="tab:orange", label="NDCG")
    ax2.plot(x, df["recall"], marker="s", color="tab:green", label="Recall")
    ax2.set_ylabel("quality")
    lines = ax1.get_legend_handles_labels()
    lines2 = ax2.get_legend_handles_labels()
    ax1.legend(lines[0] + lines2[0], lines[1] + lines2[1], loc="best")
    _save(fig, figures_dir / "internal_ease_baselines.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()
    results_dir = Path(args.results_dir)
    figures_dir = results_dir / "figures"
    plot_epsilon(results_dir, figures_dir)
    plot_rank(results_dir, figures_dir)
    plot_block(results_dir, figures_dir)
    plot_internal_baselines(results_dir, figures_dir)
    plot_baselines(results_dir, figures_dir)


if __name__ == "__main__":
    main()
