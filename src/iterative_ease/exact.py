import numpy as np
from scipy.linalg import cho_factor, cho_solve


def fit_exact_ease(X, reg: float, max_items: int = 3000) -> np.ndarray:
    if reg <= 0:
        raise ValueError("reg must be positive")

    X = X.tocsr().astype(np.float64)
    n_items = X.shape[1]
    if n_items > max_items:
        raise MemoryError(f"Exact EASE is allowed only for n_items <= {max_items}.")

    G = (X.T @ X).toarray().astype(np.float64)
    G.flat[:: n_items + 1] += float(reg)

    factor = cho_factor(G, lower=True, check_finite=False)
    P = cho_solve(factor, np.eye(n_items, dtype=np.float64), check_finite=False)

    diag = np.diag(P)
    if np.any(~np.isfinite(diag)) or np.any(np.abs(diag) < 1e-14):
        raise FloatingPointError("unstable diagonal in exact EASE solution")

    W = -P / diag[None, :]
    np.fill_diagonal(W, 0.0)
    return W
