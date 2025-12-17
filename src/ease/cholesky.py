import numpy as np


class CholeskySolver:
    def solve(self, G: np.ndarray, reg: float) -> np.ndarray:
        n = G.shape[0]
        A = G.copy()
        A[np.diag_indices(n)] += reg

        L = np.linalg.cholesky(A)
        I = np.eye(n)
        return np.linalg.solve(L.T, np.linalg.solve(L, I))