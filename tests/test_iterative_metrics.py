import numpy as np
from scipy import sparse

from iterative_ease.metrics import evaluate_topk


def test_evaluate_topk_runs_on_manual_example():
    X_train = sparse.csr_matrix([[1, 0, 0], [0, 1, 0]], dtype=float)
    X_test = sparse.csr_matrix([[0, 1, 0], [0, 0, 1]], dtype=float)
    W = sparse.csc_matrix(
        [
            [0, 1, 0],
            [1, 0, 1],
            [0, 1, 0],
        ],
        dtype=float,
    )

    result = evaluate_topk(W, X_train, X_test, k=1, user_batch_size=2)

    assert 0.0 <= result["recall"] <= 1.0
    assert 0.0 <= result["ndcg"] <= 1.0
    assert result["evaluated_users"] == 2


def test_streaming_and_full_evaluation_match():
    X_train = sparse.csr_matrix(
        [
            [1, 0, 0, 1],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
        ],
        dtype=float,
    )
    X_test = sparse.csr_matrix(
        [
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [1, 0, 0, 0],
        ],
        dtype=float,
    )
    W = sparse.csc_matrix(
        np.array(
            [
                [0.0, 2.0, 1.0, 0.0],
                [1.0, 0.0, 3.0, 1.0],
                [2.0, 1.0, 0.0, 2.0],
                [0.0, 1.0, 1.0, 0.0],
            ]
        )
    )

    full = evaluate_topk(W, X_train, X_test, k=2, user_batch_size=2)
    streaming = evaluate_topk(W, X_train, X_test, k=2, user_batch_size=2, item_batch_size=2)

    assert streaming == full
