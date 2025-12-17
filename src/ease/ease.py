import numpy as np
import scipy.sparse as sp
from .gram import compute_gram


class EASE:
    def __init__(self, solver, reg: float):
        self.solver = solver
        self.reg = float(reg)
        self.B = None

    def fit(self, X: sp.csr_matrix):
        if not sp.isspmatrix_csr(X):
            raise TypeError("X must be CSR matrix")

        G = compute_gram(X)
        P = self.solver.solve(G, self.reg)

        if not isinstance(P, np.ndarray):
            raise TypeError("Solver must return np.ndarray")
        if P.shape != G.shape:
            raise ValueError("Solver returned wrong shape")

        diag = np.diag(P)
        if np.any(diag == 0):
            raise ValueError("Zero on diagonal of P")

        B = -P / diag[None, :]
        np.fill_diagonal(B, 0.0)

        self.B = B
        return self

    def score(self, X: sp.csr_matrix) -> np.ndarray:
        if self.B is None:
            raise RuntimeError("Call fit() first")
        return X @ self.B