import os

import psutil


def current_memory_gb() -> float:
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024**3)


def estimate_dense_matrix_gb(n_rows: int, n_cols: int, dtype_bytes: int = 8) -> float:
    return n_rows * n_cols * dtype_bytes / (1024**3)


def estimate_sparse_top_l_gb(n_items: int, top_l: int, dtype_bytes: int = 8) -> float:
    # CSC values + row indices + column pointer, ignoring small Python overhead.
    nnz = n_items * top_l
    return (nnz * (dtype_bytes + 4) + (n_items + 1) * 4) / (1024**3)
