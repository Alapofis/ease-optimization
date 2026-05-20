import json
import os
import platform
import sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pandas as pd
import scipy
import yaml
from threadpoolctl import threadpool_limits

from iterative_ease.config import load_yaml, validate_default_config
from iterative_ease.data import describe_matrix, load_interactions_from_config, load_train_valid_test
from iterative_ease.memory import current_memory_gb, estimate_sparse_top_l_gb
from iterative_ease.metrics import evaluate_topk
from iterative_ease.train import fit_iterative_ease


GENERATED_FILENAMES = {
    "epsilon_sweep.csv",
    "rank_sweep.csv",
    "block_sweep.csv",
    "internal_ease_baselines.csv",
    "external_baselines.csv",
    "selected_config.yaml",
    "run_manifest.json",
}


@contextmanager
def single_threaded():
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    with threadpool_limits(limits=1):
        yield


def append_row(path, row: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new_df = pd.DataFrame([row])
    if path.exists():
        old_df = pd.read_csv(path)
        df = pd.concat([old_df, new_df], ignore_index=True, sort=False)
    else:
        df = new_df
    df.to_csv(path, index=False)


def load_base_config(path: str) -> dict:
    cfg = load_yaml(path)
    validate_default_config(cfg)
    return cfg


def apply_dry_run_limits(base: dict) -> None:
    base["data"]["max_users"] = 500
    base["data"]["max_items"] = 300
    base["model"]["max_iter"] = 5
    base["model"]["top_l"] = 50
    base["runtime"]["user_batch_size"] = 64
    base["runtime"]["item_batch_size"] = None


def load_split(base: dict):
    return load_train_valid_test(base["data"])


def load_full_matrix(base: dict):
    return load_interactions_from_config(base["data"])


def fit_and_evaluate(
    base: dict,
    X_train,
    X_eval,
    epsilon: float,
    block_size: int,
    nystrom_rank: int,
    nystrom_oversample: int,
    show_progress: bool = True,
    save_name: str | None = None,
) -> tuple[object, dict]:
    out_dir = Path(base["runtime"]["output_dir"])
    save_path = out_dir / save_name if save_name else None

    W, stats = fit_iterative_ease(
        X=X_train,
        reg=float(base["model"]["reg"]),
        block_size=int(block_size),
        tol=float(epsilon),
        max_iter=int(base["model"]["max_iter"]),
        nystrom_rank=int(nystrom_rank),
        nystrom_oversample=int(nystrom_oversample),
        top_l=base["model"].get("top_l"),
        seed=int(base["data"].get("seed", 42)),
        show_progress=show_progress,
        residual_recompute_every=base["solver"].get("residual_recompute_every", 20),
        save_path=save_path,
    )

    metrics = evaluate_topk(
        W=W,
        X_train=X_train,
        X_test=X_eval,
        k=int(base["model"]["k_recommend"]),
        user_batch_size=int(base["runtime"]["user_batch_size"]),
        item_batch_size=base["runtime"].get("item_batch_size"),
    )

    row = {
        **stats,
        **metrics,
        "memory_gb": current_memory_gb(),
        "seed": int(base["data"].get("seed", 42)),
    }
    return W, row


def ensure_output_dir(base: dict) -> Path:
    out_dir = Path(base["runtime"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def make_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def prepare_run_dir(
    base: dict,
    output_dir: str | None = None,
    run_dir: str | None = None,
    overwrite: bool = False,
) -> Path:
    if run_dir is not None:
        out = Path(run_dir)
    else:
        root = Path(output_dir) if output_dir is not None else Path(base["runtime"]["output_dir"]) / "chapter4"
        out = root / make_timestamp()

    out.mkdir(parents=True, exist_ok=True)
    if overwrite:
        for name in GENERATED_FILENAMES:
            path = out / name
            if path.exists():
                path.unlink()
        for subdir in ("figures", "tables"):
            d = out / subdir
            if d.exists():
                for child in d.iterdir():
                    if child.is_file() and child.suffix in {".png", ".tex"}:
                        child.unlink()
    return out


def save_yaml(path: str | Path, data: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def estimate_experiment_memory_gb(base: dict, n_users: int, n_items: int) -> dict[str, float]:
    top_l = base["model"].get("top_l")
    user_batch = int(base["runtime"].get("user_batch_size", 128))
    item_batch = base["runtime"].get("item_batch_size")
    item_width = n_items if item_batch is None else min(int(item_batch), n_items)

    if top_l is None:
        weight_gb = n_items * n_items * 8 / (1024**3)
    else:
        weight_gb = estimate_sparse_top_l_gb(n_items, int(top_l))

    score_gb = user_batch * item_width * 8 / (1024**3)
    work_gb = (n_users + n_items) * int(base["solver"].get("block_size", 4)) * 8 / (1024**3)
    rank_gb = n_items * int(base["solver"].get("nystrom_rank", 0)) * 8 / (1024**3)
    total = weight_gb + score_gb + work_gb + rank_gb
    return {
        "estimated_weight_gb": float(weight_gb),
        "estimated_score_batch_gb": float(score_gb),
        "estimated_solver_work_gb": float(work_gb),
        "estimated_preconditioner_gb": float(rank_gb),
        "estimated_total_gb": float(total),
    }


def memory_preflight(base: dict, X) -> dict[str, float]:
    stats = estimate_experiment_memory_gb(base, X.shape[0], X.shape[1])
    limit = float(base["runtime"].get("max_memory_gb", 10.0))
    if stats["estimated_total_gb"] > limit:
        raise MemoryError(
            f"Estimated experiment memory {stats['estimated_total_gb']:.2f} GB "
            f"exceeds runtime.max_memory_gb={limit:.2f} GB"
        )
    return stats


def write_manifest(
    path: str | Path,
    config: dict,
    dataset_stats: dict,
    selected: dict | None = None,
    memory_estimate: dict | None = None,
) -> dict:
    manifest = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "platform": {
            "python": sys.version.split()[0],
            "system": platform.platform(),
            "processor": platform.processor(),
        },
        "libraries": {
            "numpy": __import__("numpy").__version__,
            "scipy": scipy.__version__,
            "pandas": pd.__version__,
        },
        "thread_env": {
            key: os.environ.get(key)
            for key in [
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            ]
        },
        "dataset": dataset_stats,
        "memory_estimate": memory_estimate or {},
        "selected": selected or {},
        "config": config,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return manifest


def dataset_stats_for_config(base: dict) -> dict:
    X = load_full_matrix(base)
    return describe_matrix(X)
