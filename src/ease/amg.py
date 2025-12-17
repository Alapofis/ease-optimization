import numpy as np
import pyamg


class AMGSolver:
    def __init__(self):
        pass

    def solve(self, G: np.ndarray, reg: float) -> np.ndarray:
        """
        Solves (G + reg*I) P = I using algebraic multigrid.
        """
        n = G.shape[0]
        A = G + reg * np.eye(n)
        ml = pyamg.ruge_stuben_solver(A)
        P = np.zeros((n, n))

        for j in range(n):
            e = np.zeros(n)
            e[j] = 1.0
            x = ml.solve(e, tol=1e-10)
            P[:, j] = x

        return P