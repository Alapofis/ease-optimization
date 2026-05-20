import numpy as np
from scipy import sparse

from iterative_ease.exact import fit_exact_ease
from iterative_ease.train import fit_iterative_ease


def test_iterative_ease_close_to_exact_small():
    X = sparse.random(12, 8, density=0.4, format="csr", random_state=42)
    X = X.astype(np.float64)
    reg = 10.0

    W_exact = fit_exact_ease(X, reg)
    W_iter, stats = fit_iterative_ease(
        X=X,
        reg=reg,
        block_size=4,
        tol=1e-10,
        max_iter=100,
        nystrom_rank=0,
        nystrom_oversample=0,
        top_l=None,
        show_progress=False,
    )

    np.testing.assert_allclose(W_iter, W_exact, rtol=1e-5, atol=1e-5)
    assert stats["q_epsilon"] == 1.0


def test_iterative_ease_returns_sparse_top_l():
    X = sparse.random(20, 10, density=0.3, format="csr", random_state=42)
    X = X.astype(np.float64)

    W, stats = fit_iterative_ease(
        X=X,
        reg=20.0,
        block_size=2,
        tol=1e-6,
        max_iter=50,
        nystrom_rank=0,
        top_l=3,
        show_progress=False,
    )

    assert sparse.isspmatrix_csc(W)
    assert W.shape == (10, 10)
    assert stats["top_l"] == 3
