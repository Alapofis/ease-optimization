import time
from pathlib import Path

import numpy as np
from scipy import sparse
from tqdm import tqdm

from .block_pcg import solve_block_pcg
from .nystrom import NystromPreconditioner
from .operator import GramOperator


def _make_rhs(n_items: int, indices: np.ndarray) -> np.ndarray:
    E = np.zeros((n_items, len(indices)), dtype=np.float64)
    E[indices, np.arange(len(indices))] = 1.0
    return E


def _inverse_columns_to_weight_block(P_block: np.ndarray, indices: np.ndarray) -> np.ndarray:
    W_block = np.empty_like(P_block)

    for local_j, item_i in enumerate(indices):
        p = P_block[:, local_j]
        p_ii = p[item_i]

        if not np.isfinite(p_ii) or abs(p_ii) < 1e-14:
            raise FloatingPointError(f"unstable diagonal p_ii={p_ii} at item {item_i}")

        w = -p / p_ii
        w[item_i] = 0.0
        W_block[:, local_j] = w

    return W_block


def _keep_top_l_per_column(W_block: np.ndarray, top_l: int) -> sparse.csc_matrix:
    if top_l <= 0:
        raise ValueError("top_l must be positive")

    n, s = W_block.shape

    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []

    for j in range(s):
        col = W_block[:, j]

        if top_l < n:
            idx = np.argpartition(np.abs(col), -top_l)[-top_l:]
        else:
            idx = np.arange(n)

        idx = idx[col[idx] != 0.0]

        rows.extend(idx.astype(int).tolist())
        cols.extend([j] * len(idx))
        vals.extend(col[idx].astype(float).tolist())

    return sparse.csc_matrix((vals, (rows, cols)), shape=W_block.shape)


def save_weight_matrix(path: str | Path, W) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if sparse.issparse(W):
        sparse.save_npz(path, W)
    else:
        np.save(path, np.asarray(W, dtype=np.float64))


def fit_iterative_ease(
    X,
    reg: float,
    block_size: int,
    tol: float,
    max_iter: int,
    nystrom_rank: int = 64,
    nystrom_oversample: int = 10,
    top_l: int | None = 300,
    seed: int = 42,
    show_progress: bool = True,
    residual_recompute_every: int | None = 20,
    save_path: str | Path | None = None,
):
    if reg <= 0:
        raise ValueError("reg must be positive")
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if tol <= 0:
        raise ValueError("tol must be positive")
    if max_iter <= 0:
        raise ValueError("max_iter must be positive")
    if nystrom_rank < 0:
        raise ValueError("nystrom_rank must be non-negative")
    if nystrom_oversample < 0:
        raise ValueError("nystrom_oversample must be non-negative")
    if top_l is not None and top_l <= 0:
        raise ValueError("top_l must be positive or None")
    if residual_recompute_every is not None and residual_recompute_every <= 0:
        raise ValueError("residual_recompute_every must be positive or None")

    G_op = GramOperator(X, reg)
    n_items = G_op.n_items

    if top_l is None and n_items > 3000:
        raise MemoryError(
            "Refusing to build dense W for n_items > 3000. "
            "Set top_l to a positive integer."
        )

    t0 = time.perf_counter()

    preconditioner = NystromPreconditioner.build(
        G_op=G_op,
        rank=nystrom_rank,
        oversample=nystrom_oversample,
        seed=seed,
    )

    t_build = time.perf_counter() - t0

    blocks = []
    residuals = []
    iterations = []
    converged = []

    breakdowns = []
    restarts = []
    regularized_small_solves = []
    pinv_small_solves = []

    t1 = time.perf_counter()

    starts = range(0, n_items, block_size)
    iterator = tqdm(starts, desc="Solving EASE blocks") if show_progress else starts

    for start in iterator:
        stop = min(start + block_size, n_items)
        indices = np.arange(start, stop, dtype=int)

        E = _make_rhs(n_items, indices)

        result = solve_block_pcg(
            G_op=G_op,
            E=E,
            tol=tol,
            max_iter=max_iter,
            preconditioner=preconditioner,
            residual_recompute_every=residual_recompute_every,
        )

        W_block = _inverse_columns_to_weight_block(result.P, indices)

        if top_l is not None:
            W_block = _keep_top_l_per_column(W_block, top_l)

        blocks.append(W_block)

        residuals.append(result.residual_norms)
        iterations.append(result.iterations)
        converged.append(result.converged)

        breakdowns.append(result.breakdowns)
        restarts.append(result.restarts)
        regularized_small_solves.append(result.regularized_small_solves)
        pinv_small_solves.append(result.pinv_small_solves)

    t_solve = time.perf_counter() - t1

    if top_l is None:
        W = np.concatenate(blocks, axis=1)
    else:
        W = sparse.hstack(blocks, format="csc")

    residuals_arr = np.concatenate(residuals) if residuals else np.array([])
    iterations_arr = np.concatenate(iterations) if iterations else np.array([])
    converged_arr = np.concatenate(converged) if converged else np.array([], dtype=bool)

    stats = {
        "t_build": float(t_build),
        "t_solve": float(t_solve),
        "t_total": float(t_build + t_solve),

        "mean_iterations": float(np.mean(iterations_arr)) if iterations_arr.size else 0.0,
        "median_iterations": float(np.median(iterations_arr)) if iterations_arr.size else 0.0,
        "max_iterations": float(np.max(iterations_arr)) if iterations_arr.size else 0.0,
        "p90_iterations": float(np.percentile(iterations_arr, 90)) if iterations_arr.size else 0.0,
        "p95_iterations": float(np.percentile(iterations_arr, 95)) if iterations_arr.size else 0.0,

        "q_epsilon": float(np.mean(converged_arr)) if converged_arr.size else 0.0,

        "mean_residual": float(np.mean(residuals_arr)) if residuals_arr.size else 0.0,
        "max_residual": float(np.max(residuals_arr)) if residuals_arr.size else 0.0,
        "median_residual": float(np.median(residuals_arr)) if residuals_arr.size else 0.0,

        "total_breakdowns": int(np.sum(breakdowns)),
        "total_restarts": int(np.sum(restarts)),
        "total_regularized_small_solves": int(np.sum(regularized_small_solves)),
        "total_pinv_small_solves": int(np.sum(pinv_small_solves)),

        "n_users": int(G_op.n_users),
        "n_items": int(n_items),
        "block_size": int(block_size),
        "tol": float(tol),

        "nystrom_rank": int(nystrom_rank),
        "actual_preconditioner_rank": int(getattr(preconditioner, "rank", 0)),
        "nystrom_oversample": int(nystrom_oversample),

        "top_l": -1 if top_l is None else int(top_l),
    }

    if save_path is not None:
        save_weight_matrix(save_path, W)
        stats["save_path"] = str(save_path)

    return W, stats
