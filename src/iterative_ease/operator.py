import numpy as np
from scipy import sparse


class GramOperator:
    """
    Matrix-free operator for:
        S = X.T @ X
        G = X.T @ X + reg * I

    Dense S, G and G^{-1} are never formed.
    """

    def __init__(self, X: sparse.spmatrix, reg: float):
        if reg <= 0:
            raise ValueError("reg must be positive")
        if not sparse.issparse(X):
            raise TypeError("X must be a scipy sparse matrix")

        self.X = X.tocsr().astype(np.float64)
        self.XT = self.X.T.tocsr()
        self.reg = float(reg)
        self.n_users, self.n_items = self.X.shape

    def _validate_block(self, D: np.ndarray, name: str) -> np.ndarray:
        D = np.asarray(D, dtype=np.float64)
        if D.ndim != 2:
            raise ValueError(f"{name} must be a 2D array")
        if D.shape[0] != self.n_items:
            raise ValueError(
                f"{name} has wrong shape: expected {self.n_items} rows, "
                f"got {D.shape[0]}"
            )
        return D

    def s_matmat(self, D: np.ndarray) -> np.ndarray:
        """Apply S = X.T @ X to a dense block D."""
        D = self._validate_block(D, "D")
        return self.XT @ (self.X @ D)

    def matmat(self, D: np.ndarray) -> np.ndarray:
        """Apply G = X.T @ X + reg * I to a dense block D."""
        D = self._validate_block(D, "D")
        return self.s_matmat(D) + self.reg * D

    def matvec(self, v: np.ndarray) -> np.ndarray:
        v = np.asarray(v, dtype=np.float64)
        if v.ndim != 1:
            raise ValueError("v must be a 1D array")
        if v.shape[0] != self.n_items:
            raise ValueError(
                f"v has wrong length: expected {self.n_items}, got {v.shape[0]}"
            )
        return self.matmat(v[:, None])[:, 0]
