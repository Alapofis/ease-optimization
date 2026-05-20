import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np
from scipy import sparse

from iterative_ease.metrics import evaluate_topk
from iterative_ease.train import fit_iterative_ease


def main():
    X = sparse.random(
        200,
        100,
        density=0.05,
        format="csr",
        random_state=42,
        dtype=np.float64,
    )
    X.data[:] = 1.0

    W, stats = fit_iterative_ease(
        X=X,
        reg=100.0,
        block_size=4,
        tol=1e-3,
        max_iter=20,
        nystrom_rank=16,
        nystrom_oversample=5,
        top_l=50,
        seed=42,
        show_progress=False,
        save_path="results/smoke_W.npz",
    )

    metrics = evaluate_topk(
        W=W,
        X_train=X,
        X_test=X,
        k=10,
        user_batch_size=64,
    )

    print("stats:", stats)
    print("metrics:", metrics)
    print("OK")


if __name__ == "__main__":
    main()
