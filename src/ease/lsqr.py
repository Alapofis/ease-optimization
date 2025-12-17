import numpy as np
from scipy.sparse.linalg import lsqr


class LSQRSolver:
    def __init__(self, tol=1e-6, max_iter=1000):
        self.tol = tol
        self.max_iter = max_iter

    def solve(self, G: np.ndarray, reg: float) -> np.ndarray:
        """
        Solves (G + reg*I) P = I column-wise using LSQR.
        """
        n = G.shape[0]
        A = G + reg * np.eye(n)
        P = np.zeros((n, n))

        for j in range(n):
            e = np.zeros(n)
            e[j] = 1.0
            x = lsqr(A, e, atol=self.tol, btol=self.tol, iter_lim=self.max_iter)[0]
            P[:, j] = x

        return P