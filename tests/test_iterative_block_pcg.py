import numpy as np
from scipy import sparse
from scipy.linalg import cho_factor, cho_solve

from iterative_ease.block_pcg import solve_block_pcg
from iterative_ease.nystrom import RegularizationPreconditioner
from iterative_ease.operator import GramOperator


def _small_problem():
    X = sparse.random(12, 7, density=0.5, format="csr", random_state=42)
    X = X.astype(np.float64)
    reg = 3.0
    op = GramOperator(X, reg)
    G = (X.T @ X + reg * sparse.eye(X.shape[1])).toarray()
    pre = RegularizationPreconditioner(reg)
    return op, G, pre


def test_block_pcg_solves_small_system_single_rhs():
    op, G, pre = _small_problem()
    E = np.eye(G.shape[0])[:, :1]

    result = solve_block_pcg(op, E, tol=1e-10, max_iter=100, preconditioner=pre)
    expected = cho_solve(cho_factor(G, lower=True), E)

    np.testing.assert_allclose(result.P, expected, rtol=1e-6, atol=1e-6)
    assert result.converged.all()


def test_block_pcg_solves_small_system_block_rhs():
    op, G, pre = _small_problem()
    E = np.eye(G.shape[0])[:, :3]

    result = solve_block_pcg(op, E, tol=1e-10, max_iter=100, preconditioner=pre)
    expected = cho_solve(cho_factor(G, lower=True), E)

    np.testing.assert_allclose(result.P, expected, rtol=1e-5, atol=1e-5)
    assert result.converged.all()


def test_block_pcg_active_mask_handles_initially_converged_column():
    op, _, pre = _small_problem()
    E = np.eye(op.n_items)[:, :2]
    E[:, 1] = 0.0

    result = solve_block_pcg(op, E, tol=1e-12, max_iter=50, preconditioner=pre)

    assert result.converged[1]
    assert result.iterations[1] == 0
    np.testing.assert_allclose(result.P[:, 1], 0.0)


def test_block_pcg_counts_breakdown_for_dependent_rhs():
    op, _, pre = _small_problem()
    e = np.eye(op.n_items)[:, :1]
    E = np.repeat(e, 2, axis=1)

    result = solve_block_pcg(op, E, tol=1e-8, max_iter=20, preconditioner=pre)

    assert np.all(np.isfinite(result.P))
    assert result.breakdowns > 0
