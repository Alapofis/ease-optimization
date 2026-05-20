import numpy as np
from scipy import sparse

from iterative_ease.operator import GramOperator


def test_gram_operator_matches_dense():
    rng = np.random.default_rng(42)
    X = sparse.random(6, 5, density=0.4, format="csr", random_state=42)
    X = X.astype(np.float64)
    D = rng.standard_normal((5, 3))

    op = GramOperator(X, reg=2.0)
    got = op.matmat(D)

    G = X.T @ X + 2.0 * sparse.eye(5)
    expected = G.toarray() @ D
    np.testing.assert_allclose(got, expected, rtol=1e-10, atol=1e-10)
