import argparse
import time

from common import (
    append_row,
    apply_dry_run_limits,
    ensure_output_dir,
    fit_and_evaluate,
    load_base_config,
    load_split,
    load_yaml,
    prepare_run_dir,
    single_threaded,
)

from iterative_ease.baselines import (
    ALSImplicitRecommender,
    ItemKNNRecommender,
    PopularityRecommender,
)
from iterative_ease.memory import current_memory_gb
from iterative_ease.metrics import evaluate_recommender_topk


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="experiments/configs/baselines_m2.yaml")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--run-dir")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    base = load_base_config(cfg["base_config"])
    if args.dry_run:
        apply_dry_run_limits(base)
        base["model"]["max_iter"] = 5

    if args.run_dir:
        base["runtime"]["output_dir"] = str(
            prepare_run_dir(base, run_dir=args.run_dir, overwrite=args.overwrite)
        )

    X_train, _, X_test = load_split(base)
    out_dir = ensure_output_dir(base)

    internal_path = out_dir / "internal_ease_baselines.csv"
    internal = cfg.get("internal", {})
    internal_methods = [
        (
            "CG, b=1, no preconditioner",
            1,
            0,
            internal.get("fixed_epsilon", base["solver"]["epsilon"]),
            base["solver"].get("nystrom_oversample", 5),
        ),
        (
            "Block CG, no preconditioner",
            base["solver"]["block_size"],
            0,
            internal.get("fixed_epsilon", base["solver"]["epsilon"]),
            base["solver"].get("nystrom_oversample", 5),
        ),
        (
            "Block PCG, fixed epsilon",
            base["solver"]["block_size"],
            base["solver"]["nystrom_rank"],
            internal.get("fixed_epsilon", base["solver"]["epsilon"]),
            base["solver"].get("nystrom_oversample", 5),
        ),
        (
            "Block PCG, selected epsilon/rank/block",
            internal.get("selected_block_size", base["solver"]["block_size"]),
            internal.get("selected_nystrom_rank", base["solver"]["nystrom_rank"]),
            internal.get("selected_epsilon", base["solver"]["epsilon"]),
            internal.get(
                "selected_nystrom_oversample",
                base["solver"].get("nystrom_oversample", 5),
            ),
        ),
    ]

    for name, block_size, rank, eps, oversample in internal_methods:
        _, row = fit_and_evaluate(
            base,
            X_train,
            X_test,
            epsilon=float(eps),
            block_size=int(block_size),
            nystrom_rank=int(rank),
            nystrom_oversample=int(oversample),
            show_progress=True,
        )
        row = {"method": name, **row}
        append_row(internal_path, row)
        print(row)

    external_path = out_dir / "external_baselines.csv"
    baselines_cfg = base.get("baselines", {})
    external = [
        ("Popularity", PopularityRecommender()),
        ("ItemKNN", ItemKNNRecommender(top_k=int(baselines_cfg.get("itemknn_top_k", 100)))),
    ]

    for name, recommender in external:
        row = _evaluate_external(name, recommender, X_train, X_test, base)
        append_row(external_path, row)
        print(row)

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
    print(row)

    print(f"Saved internal baselines to {internal_path}")
    print(f"Saved external baselines to {external_path}")


if __name__ == "__main__":
    with single_threaded():
        main()
