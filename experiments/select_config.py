import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pandas as pd

from common import load_yaml, save_yaml


def _selection_cfg(cfg: dict | None = None) -> dict:
    cfg = cfg or {}
    return {
        "tau_recall": float(cfg.get("tau_recall", 0.005)),
        "tau_ndcg": float(cfg.get("tau_ndcg", 0.005)),
        "q_min": float(cfg.get("q_min", 0.95)),
        "tie_break": cfg.get("tie_break", "coarser"),
    }


def _filter_quality(df: pd.DataFrame, reference: pd.Series, selection: dict) -> pd.DataFrame:
    return df[
        (df["recall"] >= float(reference["recall"]) - selection["tau_recall"])
        & (df["ndcg"] >= float(reference["ndcg"]) - selection["tau_ndcg"])
        & (df["q_epsilon"] >= selection["q_min"])
    ].copy()


def select_epsilon(df: pd.DataFrame, selection_cfg: dict | None = None) -> dict:
    selection = _selection_cfg(selection_cfg)
    if df.empty:
        raise ValueError("epsilon sweep is empty")

    strict_idx = df["epsilon"].astype(float).idxmin()
    reference = df.loc[strict_idx]
    candidates = _filter_quality(df, reference, selection)
    if candidates.empty:
        raise ValueError("No admissible epsilon candidate")

    candidates = candidates.sort_values(["t_solve", "epsilon"], ascending=[True, False])
    winner = candidates.iloc[0]
    return {
        "epsilon": float(winner["epsilon"]),
        "reference_epsilon": float(reference["epsilon"]),
        "reference_recall": float(reference["recall"]),
        "reference_ndcg": float(reference["ndcg"]),
        "reference_q_epsilon": float(reference["q_epsilon"]),
        "selected_t_solve": float(winner["t_solve"]),
        "selected_recall": float(winner["recall"]),
        "selected_ndcg": float(winner["ndcg"]),
        "selected_q_epsilon": float(winner["q_epsilon"]),
    }


def select_rank(df: pd.DataFrame, selection_cfg: dict | None = None) -> dict:
    selection = _selection_cfg(selection_cfg)
    if df.empty:
        raise ValueError("rank sweep is empty")

    candidates = df[df["q_epsilon"] >= selection["q_min"]].copy()
    if candidates.empty:
        candidates = df.copy()
    candidates = candidates.sort_values(["t_total", "nystrom_rank"], ascending=[True, True])
    winner = candidates.iloc[0]
    return {
        "nystrom_rank": int(winner["nystrom_rank"]),
        "selected_t_total": float(winner["t_total"]),
        "selected_t_solve": float(winner["t_solve"]),
        "selected_mean_iterations": float(winner["mean_iterations"]),
        "selected_q_epsilon": float(winner["q_epsilon"]),
    }


def select_block(df: pd.DataFrame, selection_cfg: dict | None = None, max_memory_gb: float | None = None) -> dict:
    selection = _selection_cfg(selection_cfg)
    if df.empty:
        raise ValueError("block sweep is empty")

    reference = df.sort_values("ndcg", ascending=False).iloc[0]
    candidates = _filter_quality(df, reference, selection)
    if max_memory_gb is not None:
        candidates = candidates[candidates["memory_gb"] <= float(max_memory_gb)]
    if candidates.empty:
        raise ValueError("No admissible block_size candidate")

    candidates = candidates.sort_values(["t_solve", "block_size"], ascending=[True, True])
    winner = candidates.iloc[0]
    return {
        "block_size": int(winner["block_size"]),
        "selected_t_solve": float(winner["t_solve"]),
        "selected_t_total": float(winner["t_total"]) if "t_total" in winner else float(winner["t_solve"]),
        "selected_memory_gb": float(winner["memory_gb"]),
        "selected_recall": float(winner["recall"]),
        "selected_ndcg": float(winner["ndcg"]),
        "selected_q_epsilon": float(winner["q_epsilon"]),
    }


def select_from_run_dir(run_dir: str | Path, selection_cfg: dict | None = None, max_memory_gb: float | None = None) -> dict:
    run_dir = Path(run_dir)
    epsilon = select_epsilon(pd.read_csv(run_dir / "epsilon_sweep.csv"), selection_cfg)
    rank = select_rank(pd.read_csv(run_dir / "rank_sweep.csv"), selection_cfg)
    block = select_block(pd.read_csv(run_dir / "block_sweep.csv"), selection_cfg, max_memory_gb)
    return {"epsilon": epsilon, "rank": rank, "block": block}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--config")
    parser.add_argument("--output", default="selected_config.yaml")
    args = parser.parse_args()

    cfg = load_yaml(args.config) if args.config else {}
    selection_cfg = cfg.get("selection", cfg)
    selected = select_from_run_dir(
        args.run_dir,
        selection_cfg=selection_cfg,
        max_memory_gb=cfg.get("runtime", {}).get("max_memory_gb"),
    )
    out = Path(args.run_dir) / args.output
    save_yaml(out, selected)
    print(f"Saved selected config to {out}")


if __name__ == "__main__":
    main()
