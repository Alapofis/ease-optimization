import numpy as np
import pytest
from ease.baseline_inverse import InverseSolver
from ease.cholesky import CholeskySolver
from ease.cg import CGSolver
from ease.pcg import PCGSolver
from ease.lsqr import LSQRSolver
from ease.amg import AMGSolver

@pytest.fixture
def small_G():
    np.random.seed(42)
    A = np.random.rand(5,5)
    G = A.T @ A
    return G

def test_solver_shapes(small_G):
    solvers = [
        InverseSolver(),
        CholeskySolver(),
        CGSolver(max_iter=100),
        PCGSolver(max_iter=100),
        LSQRSolver(),
        AMGSolver()
    ]

    for solver in solvers:
        P = solver.solve(small_G, reg=1e-3)
        assert P.shape == small_G.shape
        assert np.all(np.isfinite(P))

def test_solver_runs_without_errors(small_G):
    # проверка, что solver’ы работают на разных входах
    solvers = [
        InverseSolver(),
        CholeskySolver(),
        CGSolver(max_iter=100),
        PCGSolver(max_iter=100),
        LSQRSolver(),
        AMGSolver()
    ]
    for solver in solvers:
        try:
            P = solver.solve(small_G, reg=1e-3)
        except Exception:
            pytest.fail(f"Solver {solver.__class__.__name__} raised an exception")