import numpy as np
import pandas as pd
from scipy import sparse

from iterative_ease.data import (
    load_interactions_csv,
    load_movielens_1m,
    train_valid_test_split_per_user,
)


def test_csv_loader(tmp_path):
    path = tmp_path / "interactions.csv"
    pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u2", "u3"],
            "item_id": ["i1", "i2", "i2", "i3"],
            "rating": [5, 3, 4, 5],
        }
    ).to_csv(path, index=False)

    X = load_interactions_csv(path, rating_col="rating", min_rating=4)

    assert X.shape == (3, 3)
    assert X.nnz == 3


def test_movielens_loader(tmp_path):
    path = tmp_path / "ratings.dat"
    path.write_text(
        "1::10::5::111\n"
        "1::11::3::112\n"
        "2::10::4::113\n",
        encoding="utf-8",
    )

    X = load_movielens_1m(path, min_rating=4.0)

    assert X.shape == (2, 1)
    assert X.nnz == 2


def test_split_has_no_overlap_per_user():
    X = sparse.csr_matrix(
        np.array(
            [
                [1, 1, 1, 1, 0],
                [0, 1, 1, 1, 1],
                [1, 0, 0, 0, 0],
            ],
            dtype=float,
        )
    )

    X_train, X_valid, X_test = train_valid_test_split_per_user(X, seed=42)

    for u in range(X.shape[0]):
        train = set(X_train[u].indices.tolist())
        valid = set(X_valid[u].indices.tolist())
        test = set(X_test[u].indices.tolist())
        assert train.isdisjoint(valid)
        assert train.isdisjoint(test)
        assert valid.isdisjoint(test)
