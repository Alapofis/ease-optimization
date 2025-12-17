import numpy as np


class InverseSolver:
    def solve(self, G: np.ndarray, reg: float) -> np.ndarray:
        n = G.shape[0]
        A = G.copy()
        A[np.diag_indices(n)] += reg
        return np.linalg.inv(A)