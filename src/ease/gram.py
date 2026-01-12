import numpy as np
import scipy.sparse as sp


def compute_gram(
    X: sp.csr_matrix,
    normalization: str = "none",
    alpha: float = 0.5,
    beta: float = 1.0,
) -> np.ndarray:

    if not sp.isspmatrix_csr(X):
        raise TypeError("X must be CSR matrix")

    if normalization in ("rw", "sym") and beta != 1.0:
        raise ValueError("beta must be 1.0 for rw/sym")

    item_deg = np.asarray(X.sum(axis=0)).ravel()
    user_deg = np.asarray(X.sum(axis=1)).ravel()

    def safe_pow(x, p):
        out = np.zeros_like(x, dtype=np.float64)
        mask = x > 0
        out[mask] = np.power(x[mask], p)
        return out

    if normalization == "none":
        P = X.T @ X

    elif normalization == "rw":
        D_I_inv = sp.diags(safe_pow(item_deg, -1.0))
        D_U_inv = sp.diags(safe_pow(user_deg, -1.0))
        P = D_I_inv @ (X.T @ D_U_inv @ X)

    elif normalization == "sym":
        D_I = sp.diags(safe_pow(item_deg, -0.5))
        D_U = sp.diags(safe_pow(user_deg, -1.0))
        P = D_I @ (X.T @ D_U @ X) @ D_I

    elif normalization == "dan":
        D_I_l = sp.diags(safe_pow(item_deg, -(1.0 - alpha)))
        D_I_r = sp.diags(safe_pow(item_deg, -alpha))
        D_U = sp.diags(safe_pow(user_deg, -beta))
        P = D_I_l @ (X.T @ D_U @ X) @ D_I_r

    else:
        raise ValueError(f"Unknown normalization: {normalization}")

    return P.toarray()
