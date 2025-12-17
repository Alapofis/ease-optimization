import numpy as np


def check_symmetric(A: np.ndarray, tol=1e-8):
    return np.allclose(A, A.T, atol=tol)