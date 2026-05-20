import numpy as np
import pandas as pd
from scipy import sparse


def _build_binary_matrix(
    user_values,
    item_values,
    return_mappings: bool = False,
):
    users = pd.Categorical(user_values)
    items = pd.Categorical(item_values)
    data = np.ones(len(users.codes), dtype=np.float64)

    X = sparse.csr_matrix(
        (data, (users.codes, items.codes)),
        shape=(len(users.categories), len(items.categories)),
    )
    X.data[:] = 1.0
    X.eliminate_zeros()

    if return_mappings:
        return X, list(users.categories), list(items.categories)
    return X


def load_interactions_csv(
    path,
    user_col: str = "user_id",
    item_col: str = "item_id",
    rating_col: str | None = None,
    min_rating: float | None = None,
    sep: str = ",",
    return_mappings: bool = False,
):
    df = pd.read_csv(path, sep=sep)
    if user_col not in df.columns:
        raise ValueError(f"Missing user column: {user_col}")
    if item_col not in df.columns:
        raise ValueError(f"Missing item column: {item_col}")
    if rating_col is not None:
        if rating_col not in df.columns:
            raise ValueError(f"Missing rating column: {rating_col}")
        if min_rating is not None:
            df = df[df[rating_col] >= min_rating]

    return _build_binary_matrix(df[user_col], df[item_col], return_mappings=return_mappings)


def load_movielens_1m(
    path,
    min_rating: float = 4.0,
    return_mappings: bool = False,
):
    df = pd.read_csv(
        path,
        sep="::",
        engine="python",
        names=["user_id", "item_id", "rating", "timestamp"],
    )
    df = df[df["rating"] >= min_rating]
    return _build_binary_matrix(df["user_id"], df["item_id"], return_mappings=return_mappings)


def limit_matrix(X, max_users=None, max_items=None):
    if max_users is not None:
        X = X[: int(max_users)]
    if max_items is not None:
        X = X[:, : int(max_items)]
    return X.tocsr()


def train_valid_test_split_per_user(
    X,
    validation_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
):
    if validation_ratio < 0 or test_ratio < 0 or validation_ratio + test_ratio >= 1:
        raise ValueError("validation_ratio and test_ratio must be non-negative and sum to < 1")

    rng = np.random.default_rng(seed)
    X = X.tocsr()

    train_rows: list[int] = []
    train_cols: list[int] = []
    valid_rows: list[int] = []
    valid_cols: list[int] = []
    test_rows: list[int] = []
    test_cols: list[int] = []

    for u in range(X.shape[0]):
        start = X.indptr[u]
        end = X.indptr[u + 1]
        items = X.indices[start:end].copy()

        if items.size < 3:
            train_rows.extend([u] * items.size)
            train_cols.extend(items.astype(int).tolist())
            continue

        rng.shuffle(items)
        n_test = max(1, int(items.size * test_ratio)) if test_ratio > 0 else 0
        n_valid = max(1, int(items.size * validation_ratio)) if validation_ratio > 0 else 0

        if n_test + n_valid >= items.size:
            n_valid = max(0, items.size - n_test - 1)

        test_items = items[:n_test]
        valid_items = items[n_test : n_test + n_valid]
        train_items = items[n_test + n_valid :]

        train_rows.extend([u] * train_items.size)
        train_cols.extend(train_items.astype(int).tolist())
        valid_rows.extend([u] * valid_items.size)
        valid_cols.extend(valid_items.astype(int).tolist())
        test_rows.extend([u] * test_items.size)
        test_cols.extend(test_items.astype(int).tolist())

    shape = X.shape
    X_train = sparse.csr_matrix(
        (np.ones(len(train_rows), dtype=np.float64), (train_rows, train_cols)),
        shape=shape,
    )
    X_valid = sparse.csr_matrix(
        (np.ones(len(valid_rows), dtype=np.float64), (valid_rows, valid_cols)),
        shape=shape,
    )
    X_test = sparse.csr_matrix(
        (np.ones(len(test_rows), dtype=np.float64), (test_rows, test_cols)),
        shape=shape,
    )
    return X_train.tocsr(), X_valid.tocsr(), X_test.tocsr()


def describe_matrix(X) -> dict[str, float]:
    X = X.tocsr()
    n_users, n_items = X.shape
    nnz = X.nnz
    return {
        "n_users": int(n_users),
        "n_items": int(n_items),
        "nnz": int(nnz),
        "density": float(nnz / (n_users * n_items)) if n_users and n_items else 0.0,
        "mean_items_per_user": float(nnz / n_users) if n_users else 0.0,
        "mean_users_per_item": float(nnz / n_items) if n_items else 0.0,
    }


def load_interactions_from_config(data_cfg: dict):
    fmt = data_cfg.get("format", "csv")
    if fmt == "movielens_1m":
        X = load_movielens_1m(
            data_cfg["path"],
            min_rating=float(data_cfg.get("min_rating", 4.0)),
        )
    elif fmt == "csv":
        X = load_interactions_csv(
            data_cfg["path"],
            user_col=data_cfg.get("user_col", "user_id"),
            item_col=data_cfg.get("item_col", "item_id"),
            rating_col=data_cfg.get("rating_col"),
            min_rating=data_cfg.get("min_rating"),
            sep=data_cfg.get("sep", ","),
        )
    else:
        raise ValueError(f"Unknown data format: {fmt}")

    return limit_matrix(
        X,
        max_users=data_cfg.get("max_users"),
        max_items=data_cfg.get("max_items"),
    )


def load_train_valid_test(data_cfg: dict):
    X = load_interactions_from_config(data_cfg)
    return train_valid_test_split_per_user(
        X,
        validation_ratio=float(data_cfg.get("validation_ratio", 0.1)),
        test_ratio=float(data_cfg.get("test_ratio", 0.1)),
        seed=int(data_cfg.get("seed", 42)),
    )
