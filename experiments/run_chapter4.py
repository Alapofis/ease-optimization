import argparse
import copy
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from common import (
    append_row,
    apply_dry_run_limits,
    dataset_stats_for_config,
    fit_and_evaluate,
    load_base_config,
    load_split,
    memory_preflight,
    prepare_run_dir,
    save_yaml,
    single_threaded,
    write_manifest,
)
from iterative_ease.baselines import ALSImplicitRecommender, ItemKNNRecommender, PopularityRecommender
from iterative_ease.memory import current_memory_gb
from iterative_ease.metrics import evaluate_recommender_topk, evaluate_topk
from iterative_ease.train import fit_iterative_ease
from make_chapter4_tables import make_tables, validate_artifacts
from plot_results import plot_baselines, plot_block, plot_epsilon, plot_internal_baselines, plot_rank
from select_config import select_block, select_epsilon, select_rank


def _run_smoke(run_dir: Path) -> dict:
    X = sparse.random(
        200,
        100,
        density=0.05,
        format="csr",
        random_state=42,
        dtype=np.float64,
    )
    X.data[:] = 1.0
    W, stats = fit_iterative_ease(
        X=X,
        reg=100.0,
        block_size=4,
        tol=1e-3,
        max_iter=20,
        nystrom_rank=16,
        nystrom_oversample=5,
        top_l=50,
        seed=42,
        show_progress=False,
        save_path=run_dir / "smoke_W.npz",
    )
    metrics = evaluate_topk(W, X, X, k=10, user_batch_size=64)
    return {**stats, **metrics}


def _sweep_values(base: dict, dry_run: bool) -> dict:
    if dry_run:
        return {
            "epsilon": [1.0e-2, 1.0e-3],
            "rank": [0, 16],
            "block": [1, 4],
            "oversample": 5,
        }
    return {
        "epsilon": [1.0e-2, 1.0e-3, 1.0e-4],
        "rank": [0, 32, 64, 128],
        "block": [1, 4, 8, 16],
        "oversample": int(base["solver"].get("nystrom_oversample", 5)),
    }


def _run_epsilon_sweep(base, X_train, X_valid, values, run_dir: Path) -> dict:
    out_path = run_dir / "epsilon_sweep.csv"
    for eps in values["epsilon"]:
        _, row = fit_and_evaluate(
            base,
            X_train,
            X_valid,
            epsilon=float(eps),
            block_size=int(base["solver"]["block_size"]),
            nystrom_rank=int(base["solver"]["nystrom_rank"]),
            nystrom_oversample=int(values["oversample"]),
            show_progress=True,
        )
        append_row(out_path, {"epsilon": float(eps), **row})
    return select_epsilon(pd.read_csv(out_path), base.get("selection"))


def _run_rank_sweep(base, X_train, X_valid, values, epsilon: float, run_dir: Path) -> dict:
    out_path = run_dir / "rank_sweep.csv"
    for rank in values["rank"]:
        _, row = fit_and_evaluate(
            base,
            X_train,
            X_valid,
            epsilon=float(epsilon),
            block_size=int(base["solver"]["block_size"]),
            nystrom_rank=int(rank),
            nystrom_oversample=int(values["oversample"]),
            show_progress=True,
        )
        append_row(out_path, {"nystrom_rank": int(rank), **row})
    return select_rank(pd.read_csv(out_path), base.get("selection"))


def _run_block_sweep(base, X_train, X_valid, values, epsilon: float, rank: int, run_dir: Path) -> dict:
    out_path = run_dir / "block_sweep.csv"
    for block_size in values["block"]:
        _, row = fit_and_evaluate(
            base,
            X_train,
            X_valid,
            epsilon=float(epsilon),
            block_size=int(block_size),
            nystrom_rank=int(rank),
            nystrom_oversample=int(values["oversample"]),
            show_progress=True,
        )
        append_row(out_path, {"block_size": int(block_size), **row})
    return select_block(
        pd.read_csv(out_path),
        base.get("selection"),
        max_memory_gb=base["runtime"].get("max_memory_gb"),
    )


def _evaluate_external(name, recommender, X_train, X_test, base):
    t0 = time.perf_counter()
    recommender.fit(X_train)
    t_fit = time.perf_counter() - t0
    metrics = evaluate_recommender_topk(
        recommender,
        X_train=X_train,
        X_test=X_test,
        k=int(base["model"]["k_recommend"]),
        user_batch_size=int(base["runtime"]["user_batch_size"]),
    )
    return {
        "method": name,
        "status": "ok",
        "reason": None,
        "t_fit": float(t_fit),
        **metrics,
        "memory_gb": current_memory_gb(),
        "seed": int(base["data"].get("seed", 42)),
    }


def _run_baselines(base, X_train, X_test, selected, run_dir: Path):
    internal_path = run_dir / "internal_ease_baselines.csv"
    selected_eps = float(selected["epsilon"]["epsilon"])
    selected_rank = int(selected["rank"]["nystrom_rank"])
    selected_block = int(selected["block"]["block_size"])
    oversample = int(base["solver"].get("nystrom_oversample", 5))
    internal_methods = [
        ("CG, b=1, no preconditioner", 1, 0, selected_eps),
        ("Block CG, no preconditioner", selected_block, 0, selected_eps),
        ("Block PCG, fixed epsilon", selected_block, selected_rank, float(base["solver"]["epsilon"])),
        ("Block PCG, selected epsilon/rank/block", selected_block, selected_rank, selected_eps),
    ]
    for name, block_size, rank, eps in internal_methods:
        _, row = fit_and_evaluate(
            base,
            X_train,
            X_test,
            epsilon=float(eps),
            block_size=int(block_size),
            nystrom_rank=int(rank),
            nystrom_oversample=oversample,
            show_progress=True,
        )
        append_row(internal_path, {"method": name, **row})

    external_path = run_dir / "external_baselines.csv"
    baselines_cfg = base.get("baselines", {})
    for name, recommender in [
        ("Popularity", PopularityRecommender()),
        ("ItemKNN", ItemKNNRecommender(top_k=int(baselines_cfg.get("itemknn_top_k", 100)))),
    ]:
        append_row(external_path, _evaluate_external(name, recommender, X_train, X_test, base))

    try:
        als = ALSImplicitRecommender(
            factors=int(baselines_cfg.get("als_factors", 64)),
            regularization=float(baselines_cfg.get("als_regularization", 0.01)),
            iterations=int(baselines_cfg.get("als_iterations", 15)),
            alpha=float(baselines_cfg.get("als_alpha", 40.0)),
            random_state=int(base["data"].get("seed", 42)),
        )
        row = _evaluate_external("ALS implicit", als, X_train, X_test, base)
    except ImportError as exc:
        row = {
            "method": "ALS implicit",
            "status": "skipped",
            "reason": str(exc),
            "t_fit": None,
            "recall": None,
            "ndcg": None,
            "evaluated_users": None,
            "memory_gb": current_memory_gb(),
            "seed": int(base["data"].get("seed", 42)),
        }
    append_row(external_path, row)


def _plot_all(run_dir: Path) -> None:
    figures_dir = run_dir / "figures"
    plot_epsilon(run_dir, figures_dir)
    plot_rank(run_dir, figures_dir)
    plot_block(run_dir, figures_dir)
    plot_internal_baselines(run_dir, figures_dir)
    plot_baselines(run_dir, figures_dir)


def run_chapter4(
    config: str,
    dry_run: bool = False,
    output_dir: str | None = None,
    run_dir: str | None = None,
    overwrite: bool = False,
) -> Path:
    base = load_base_config(config)
    if dry_run:
        apply_dry_run_limits(base)

    out_dir = prepare_run_dir(base, output_dir=output_dir, run_dir=run_dir, overwrite=overwrite)
    base["runtime"]["output_dir"] = str(out_dir)
    values = _sweep_values(base, dry_run)

    print(f"Chapter 4 run directory: {out_dir}")
    dataset_stats = dataset_stats_for_config(base)
    X_train, X_valid, X_test = load_split(base)
    preflight_cfg = copy.deepcopy(base)
    preflight_cfg["solver"]["block_size"] = max(values["block"])
    preflight_cfg["solver"]["nystrom_rank"] = max(values["rank"])
    memory_estimate = memory_preflight(preflight_cfg, X_train)

    smoke = _run_smoke(out_dir)
    print("smoke:", smoke)

    selected = {}
    selected["epsilon"] = _run_epsilon_sweep(base, X_train, X_valid, values, out_dir)
    selected_eps = selected["epsilon"]["epsilon"]

    selected["rank"] = _run_rank_sweep(base, X_train, X_valid, values, selected_eps, out_dir)
    selected_rank = selected["rank"]["nystrom_rank"]

    selected["block"] = _run_block_sweep(base, X_train, X_valid, values, selected_eps, selected_rank, out_dir)

    selected_config = copy.deepcopy(base)
    selected_config["solver"]["epsilon"] = float(selected_eps)
    selected_config["solver"]["nystrom_rank"] = int(selected_rank)
    selected_config["solver"]["block_size"] = int(selected["block"]["block_size"])
    selected_config["selected"] = selected
    save_yaml(out_dir / "selected_config.yaml", selected_config)

    _run_baselines(selected_config, X_train, X_test, selected, out_dir)
    write_manifest(
        out_dir / "run_manifest.json",
        selected_config,
        dataset_stats=dataset_stats,
        selected=selected,
        memory_estimate=memory_estimate,
    )
    _plot_all(out_dir)
    make_tables(out_dir)
    validate_artifacts(out_dir)
    return out_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="experiments/configs/default_m2.yaml")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output-dir")
    parser.add_argument("--run-dir")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    with single_threaded():
        out_dir = run_chapter4(
            config=args.config,
            dry_run=args.dry_run,
            output_dir=args.output_dir,
            run_dir=args.run_dir,
            overwrite=args.overwrite,
        )
    print(f"Chapter 4 artifacts are ready in {out_dir}")


if __name__ == "__main__":
    main()
