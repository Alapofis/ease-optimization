import numpy as np
from scipy.sparse.linalg import cg


class PCGSolver:
    def __init__(self, tol=1e-6, max_iter=1000):
        self.tol = tol
        self.max_iter = max_iter

    def solve(self, G: np.ndarray, reg: float) -> np.ndarray:
        n = G.shape[0]
        A = G + reg * np.eye(n)

        M_inv = np.diag(1.0 / np.diag(A))  # Jacobi preconditioner
        P = np.zeros((n, n))

        for j in range(n):
            e = np.zeros(n)
            e[j] = 1.0
            x, info = cg(
                A, e, M=M_inv, tol=self.tol, maxiter=self.max_iter
            )
            if info != 0:
                raise RuntimeError(f"PCG failed at column {j}")
            P[:, j] = x

        return P