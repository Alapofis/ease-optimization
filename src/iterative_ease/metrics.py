import inspect

import numpy as np
from scipy import sparse


def _as_dense(A) -> np.ndarray:
    if sparse.issparse(A):
        return A.toarray()
    return np.asarray(A, dtype=np.float64)


def _topk_indices_from_scores(scores: np.ndarray, k: int) -> np.ndarray:
    if k <= 0:
        raise ValueError("k must be positive")

    k = min(k, scores.shape[1])

    idx = np.argpartition(scores, -k, axis=1)[:, -k:]
    rows = np.arange(scores.shape[0])[:, None]
    order = np.argsort(scores[rows, idx], axis=1)[:, ::-1]

    return idx[rows, order]


def _mask_train_items(scores: np.ndarray, X_exclude_batch, offset: int = 0) -> np.ndarray:
    X_exclude_batch = X_exclude_batch.tocsr()
    width = scores.shape[1]

    for u in range(X_exclude_batch.shape[0]):
        start = X_exclude_batch.indptr[u]
        end = X_exclude_batch.indptr[u + 1]
        items = X_exclude_batch.indices[start:end]

        local = items[(items >= offset) & (items < offset + width)] - offset
        scores[u, local] = -np.inf

    return scores


def _merge_topk(
    current_scores: np.ndarray | None,
    current_items: np.ndarray | None,
    new_scores: np.ndarray,
    new_items: np.ndarray,
    k: int,
):
    if current_scores is None:
        combined_scores = new_scores
        combined_items = new_items
    else:
        combined_scores = np.concatenate([current_scores, new_scores], axis=1)
        combined_items = np.concatenate([current_items, new_items], axis=1)

    keep = min(k, combined_scores.shape[1])

    idx = np.argpartition(combined_scores, -keep, axis=1)[:, -keep:]
    rows = np.arange(combined_scores.shape[0])[:, None]
    order = np.argsort(combined_scores[rows, idx], axis=1)[:, ::-1]
    idx = idx[rows, order]

    return combined_scores[rows, idx], combined_items[rows, idx]


def _topk_full(W, Xb_train, Xb_exclude, k: int) -> np.ndarray:
    scores = _as_dense(Xb_train @ W)
    scores = _mask_train_items(scores, Xb_exclude, offset=0)
    return _topk_indices_from_scores(scores, k)


def _topk_streaming(W, Xb_train, Xb_exclude, k: int, item_batch_size: int) -> np.ndarray:
    if item_batch_size <= 0:
        raise ValueError("item_batch_size must be positive")

    n_items = Xb_train.shape[1]

    best_scores = None
    best_items = None

    for item_start in range(0, n_items, item_batch_size):
        item_stop = min(item_start + item_batch_size, n_items)

        W_chunk = W[:, item_start:item_stop]

        scores = _as_dense(Xb_train @ W_chunk)
        scores = _mask_train_items(scores, Xb_exclude, offset=item_start)

        items = np.tile(
            np.arange(item_start, item_stop, dtype=int),
            (scores.shape[0], 1),
        )

        best_scores, best_items = _merge_topk(
            best_scores,
            best_items,
            scores,
            items,
            k,
        )

    return best_items


def _ranking_metrics(topk: np.ndarray, X_test_batch, k: int) -> dict[str, float]:
    X_test_batch = X_test_batch.tocsr()

    discounts = 1.0 / np.log2(np.arange(2, k + 2))

    recalls = []
    ndcgs = []

    for local_u in range(X_test_batch.shape[0]):
        start = X_test_batch.indptr[local_u]
        end = X_test_batch.indptr[local_u + 1]
        true_items = X_test_batch.indices[start:end]

        if true_items.size == 0:
            continue

        true_set = set(true_items.tolist())

        hits = np.array(
            [1.0 if item in true_set else 0.0 for item in topk[local_u]],
            dtype=np.float64,
        )

        recalls.append(float(hits.sum() / true_items.size))

        dcg = float(np.sum(hits * discounts[: hits.size]))
        ideal_len = min(true_items.size, k)
        idcg = float(np.sum(discounts[:ideal_len]))

        ndcgs.append(dcg / idcg if idcg > 0 else 0.0)

    return {
        "recall": float(np.mean(recalls)) if recalls else 0.0,
        "ndcg": float(np.mean(ndcgs)) if ndcgs else 0.0,
        "evaluated_users": int(len(recalls)),
    }


def evaluate_topk(
    W,
    X_train,
    X_test,
    k: int,
    user_batch_size: int = 256,
    item_batch_size: int | None = None,
    X_exclude=None,
) -> dict[str, float]:
    """
    Evaluate Recall@K and NDCG@K.

    X_train is used to compute user scores: X_train @ W.
    X_exclude is used only for masking already known items.

    For validation:
        X_exclude = X_train

    For final test after hyperparameter selection:
        X_exclude = X_train + X_valid
    """

    if k <= 0:
        raise ValueError("k must be positive")
    if user_batch_size <= 0:
        raise ValueError("user_batch_size must be positive")
    if item_batch_size is not None and item_batch_size <= 0:
        raise ValueError("item_batch_size must be positive or None")

    X_train = X_train.tocsr()
    X_test = X_test.tocsr()

    if X_exclude is None:
        X_exclude = X_train
    else:
        X_exclude = X_exclude.tocsr()

    if X_train.shape != X_test.shape:
        raise ValueError("X_train and X_test must have the same shape")
    if X_exclude.shape != X_train.shape:
        raise ValueError("X_exclude must have the same shape as X_train")

    recalls = []
    ndcgs = []
    evaluated = 0

    n_users = X_train.shape[0]

    for start in range(0, n_users, user_batch_size):
        stop = min(start + user_batch_size, n_users)

        Xb_train = X_train[start:stop]
        Xb_test = X_test[start:stop]
        Xb_exclude = X_exclude[start:stop]

        if item_batch_size is None:
            topk = _topk_full(W, Xb_train, Xb_exclude, k)
        else:
            topk = _topk_streaming(W, Xb_train, Xb_exclude, k, item_batch_size)

        batch_metrics = _ranking_metrics(topk, Xb_test, k)

        if batch_metrics["evaluated_users"] == 0:
            continue

        recalls.append(batch_metrics["recall"] * batch_metrics["evaluated_users"])
        ndcgs.append(batch_metrics["ndcg"] * batch_metrics["evaluated_users"])
        evaluated += batch_metrics["evaluated_users"]

    return {
        "recall": float(np.sum(recalls) / evaluated) if evaluated else 0.0,
        "ndcg": float(np.sum(ndcgs) / evaluated) if evaluated else 0.0,
        "evaluated_users": int(evaluated),
    }


def evaluate_recommender_topk(
    recommender,
    X_train,
    X_test,
    k: int,
    user_batch_size: int = 256,
    X_exclude=None,
) -> dict[str, float]:
    """
    Evaluate a recommender object with score_matrix(...).

    X_train is passed into score_matrix.
    X_exclude is used only for masking already known items.
    """

    if k <= 0:
        raise ValueError("k must be positive")
    if user_batch_size <= 0:
        raise ValueError("user_batch_size must be positive")

    X_train = X_train.tocsr()
    X_test = X_test.tocsr()

    if X_exclude is None:
        X_exclude = X_train
    else:
        X_exclude = X_exclude.tocsr()

    if X_train.shape != X_test.shape:
        raise ValueError("X_train and X_test must have the same shape")
    if X_exclude.shape != X_train.shape:
        raise ValueError("X_exclude must have the same shape as X_train")

    recalls = []
    ndcgs = []
    evaluated = 0

    n_users = X_train.shape[0]

    sig = inspect.signature(recommender.score_matrix)
    accepts_user_indices = "user_indices" in sig.parameters

    for start in range(0, n_users, user_batch_size):
        stop = min(start + user_batch_size, n_users)

        Xb_train = X_train[start:stop]
        Xb_exclude = X_exclude[start:stop]

        if accepts_user_indices:
            scores = recommender.score_matrix(
                Xb_train,
                user_indices=np.arange(start, stop),
            )
        else:
            scores = recommender.score_matrix(Xb_train)

        scores = _mask_train_items(_as_dense(scores), Xb_exclude, offset=0)

        topk = _topk_indices_from_scores(scores, k)
        batch_metrics = _ranking_metrics(topk, X_test[start:stop], k)

        if batch_metrics["evaluated_users"] == 0:
            continue

        recalls.append(batch_metrics["recall"] * batch_metrics["evaluated_users"])
        ndcgs.append(batch_metrics["ndcg"] * batch_metrics["evaluated_users"])
        evaluated += batch_metrics["evaluated_users"]

    return {
        "recall": float(np.sum(recalls) / evaluated) if evaluated else 0.0,
        "ndcg": float(np.sum(ndcgs) / evaluated) if evaluated else 0.0,
        "evaluated_users": int(evaluated),
    }
