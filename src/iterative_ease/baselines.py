import numpy as np
from scipy import sparse


class PopularityRecommender:
    def fit(self, X):
        self.item_scores_ = np.asarray(X.sum(axis=0)).ravel().astype(np.float64)
        return self

    def score_matrix(self, X_user):
        if not hasattr(self, "item_scores_"):
            raise RuntimeError("Call fit() first")
        return np.tile(self.item_scores_, (X_user.shape[0], 1))


class ItemKNNRecommender:
    def __init__(self, top_k: int = 100):
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        self.top_k = int(top_k)

    def fit(self, X):
        X = X.tocsr().astype(np.float64)
        norms = np.sqrt(np.asarray(X.power(2).sum(axis=0)).ravel())
        norms[norms == 0.0] = 1.0

        Xn = X @ sparse.diags(1.0 / norms)
        S = (Xn.T @ Xn).tolil()
        S.setdiag(0.0)
        S = S.tocsc()
        S.eliminate_zeros()
        self.W_ = self._keep_top_k_columns(S, self.top_k)
        return self

    @staticmethod
    def _keep_top_k_columns(S, top_k: int):
        S = S.tocsc()
        rows: list[int] = []
        cols: list[int] = []
        vals: list[float] = []

        for j in range(S.shape[1]):
            start = S.indptr[j]
            end = S.indptr[j + 1]
            idx = S.indices[start:end]
            data = S.data[start:end]
            if data.size > top_k:
                keep = np.argpartition(data, -top_k)[-top_k:]
                idx = idx[keep]
                data = data[keep]
            rows.extend(idx.astype(int).tolist())
            cols.extend([j] * len(idx))
            vals.extend(data.astype(float).tolist())

        return sparse.csc_matrix((vals, (rows, cols)), shape=S.shape)

    def score_matrix(self, X_user):
        if not hasattr(self, "W_"):
            raise RuntimeError("Call fit() first")
        return X_user @ self.W_


class ALSImplicitRecommender:
    """
    Optional implicit-feedback ALS baseline.

    This class requires the external 'implicit' package. It is intentionally not
    listed as a hard dependency because the scientific work treats ALS as an
    auxiliary external baseline.
    """

    def __init__(
        self,
        factors: int = 64,
        regularization: float = 0.01,
        iterations: int = 15,
        alpha: float = 40.0,
        random_state: int = 42,
    ):
        self.factors = int(factors)
        self.regularization = float(regularization)
        self.iterations = int(iterations)
        self.alpha = float(alpha)
        self.random_state = int(random_state)

    def fit(self, X):
        try:
            from implicit.als import AlternatingLeastSquares
        except ImportError as exc:
            raise ImportError("Install optional dependency 'implicit' to run ALS baseline") from exc

        X = X.tocsr().astype(np.float32)
        self.model_ = AlternatingLeastSquares(
            factors=self.factors,
            regularization=self.regularization,
            iterations=self.iterations,
            random_state=self.random_state,
        )
        self.model_.fit((X.T * self.alpha).tocsr(), show_progress=False)
        return self

    def score_matrix(self, X_user, user_indices=None):
        if not hasattr(self, "model_"):
            raise RuntimeError("Call fit() first")
        if user_indices is None:
            raise ValueError("ALSImplicitRecommender requires user_indices")
        return self.model_.user_factors[user_indices] @ self.model_.item_factors.T
