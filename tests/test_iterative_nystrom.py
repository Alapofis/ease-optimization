import numpy as np
from scipy import sparse

from iterative_ease.nystrom import NystromPreconditioner, RegularizationPreconditioner
from iterative_ease.operator import GramOperator


def test_nystrom_apply_shape_and_finiteness():
    X = sparse.random(20, 10, density=0.3, format="csr", random_state=42)
    X = X.astype(np.float64)
    op = GramOperator(X, reg=5.0)

    pre = NystromPreconditioner.build(op, rank=4, oversample=2, seed=42)
    R = np.random.default_rng(42).standard_normal((10, 3))
    Z = pre.apply(R)

    assert Z.shape == R.shape
    assert np.all(np.isfinite(Z))


def test_rank_zero_uses_regularization_preconditioner():
    X = sparse.eye(5, format="csr")
    op = GramOperator(X, reg=3.0)
    pre = NystromPreconditioner.build(op, rank=0)

    assert isinstance(pre, RegularizationPreconditioner)
    R = np.ones((5, 2))
    np.testing.assert_allclose(pre.apply(R), R / 3.0)


def test_nystrom_woodbury_apply_matches_dense_inverse():
    rng = np.random.default_rng(42)
    Q, _ = np.linalg.qr(rng.standard_normal((8, 3)))
    theta = np.array([5.0, 2.0, 0.5])
    reg = 4.0
    pre = NystromPreconditioner(Q=Q, theta=theta, reg=reg)

    R = rng.standard_normal((8, 4))
    M = reg * np.eye(8) + Q @ np.diag(theta) @ Q.T

    np.testing.assert_allclose(pre.apply(R), np.linalg.solve(M, R), rtol=1e-10, atol=1e-10)
