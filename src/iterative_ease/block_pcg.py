from dataclasses import dataclass

import numpy as np


@dataclass
class BlockPCGResult:
    P: np.ndarray
    residual_norms: np.ndarray
    iterations: np.ndarray
    converged: np.ndarray
    breakdowns: int
    restarts: int
    regularized_small_solves: int
    pinv_small_solves: int


@dataclass
class _SmallSolveResult:
    X: np.ndarray
    used_fallback: bool
    regularized: bool
    pinv: bool


def _sym(A: np.ndarray) -> np.ndarray:
    return 0.5 * (A + A.T)


def _small_solve(A: np.ndarray, B: np.ndarray, ridge: float = 1e-10) -> _SmallSolveResult:
    """
    Robust solve for small dense systems A X = B.

    The lhs is symmetrized. If the system is ill-conditioned, diagonal
    regularization is tried before the final pseudo-inverse fallback.
    """

    A = _sym(np.asarray(A, dtype=np.float64))
    B = np.asarray(B, dtype=np.float64)

    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be square")
    if B.ndim != 2 or B.shape[0] != A.shape[0]:
        raise ValueError("B has incompatible shape")
    if ridge <= 0:
        raise ValueError("ridge must be positive")

    scale = max(1.0, float(np.linalg.norm(A, ord=np.inf)))

    try:
        cond = np.linalg.cond(A)
    except np.linalg.LinAlgError:
        cond = np.inf

    candidates: list[tuple[np.ndarray, bool]] = []

    # If the small system looks numerically safe, try it as is.
    if np.isfinite(cond) and cond <= 1.0 / max(ridge, 1e-16):
        candidates.append((A, False))

    # Otherwise, or if the plain solve fails, try increasingly stronger ridge.
    for multiplier in (1.0, 100.0, 10000.0):
        A_reg = A.copy()
        A_reg.flat[:: A_reg.shape[0] + 1] += ridge * multiplier * scale
        candidates.append((A_reg, True))

    for candidate, regularized in candidates:
        try:
            X = np.linalg.solve(candidate, B)
            if np.all(np.isfinite(X)):
                return _SmallSolveResult(
                    X=X,
                    used_fallback=regularized,
                    regularized=regularized,
                    pinv=False,
                )
        except np.linalg.LinAlgError:
            continue

    X = np.linalg.pinv(A, rcond=1e-12) @ B
    if not np.all(np.isfinite(X)):
        raise FloatingPointError("small solve produced non-finite coefficients")

    return _SmallSolveResult(
        X=X,
        used_fallback=True,
        regularized=False,
        pinv=True,
    )


def _right_small_solve(A: np.ndarray, B: np.ndarray, ridge: float = 1e-10) -> _SmallSolveResult:
    """
    Solve X A = B for X.

    Implemented as A.T X.T = B.T.

    Kept for compatibility. The Block PCG beta update below does not use this
    helper because the implementation updates directions as D_old @ beta.
    """

    result = _small_solve(A.T, B.T, ridge=ridge)
    return _SmallSolveResult(
        X=result.X.T,
        used_fallback=result.used_fallback,
        regularized=result.regularized,
        pinv=result.pinv,
    )


def solve_block_pcg(
    G_op,
    E: np.ndarray,
    tol: float,
    max_iter: int,
    preconditioner,
    ridge: float = 1e-10,
    residual_recompute_every: int | None = 20,
) -> BlockPCGResult:
    """
    Active-mask preconditioned Block CG for G P = E.

    Columns satisfying ||r_i||_2 <= tol are fixed and removed from later
    coefficient matrices, while the physical work arrays keep their original
    width.
    """

    E = np.asarray(E, dtype=np.float64)

    if E.ndim != 2:
        raise ValueError("E must be 2D")
    if E.shape[0] != G_op.n_items:
        raise ValueError("E row count must match G_op.n_items")
    if tol <= 0:
        raise ValueError("tol must be positive")
    if max_iter <= 0:
        raise ValueError("max_iter must be positive")
    if ridge <= 0:
        raise ValueError("ridge must be positive")
    if residual_recompute_every is not None and residual_recompute_every <= 0:
        raise ValueError("residual_recompute_every must be positive or None")

    n, s = E.shape

    P = np.zeros((n, s), dtype=np.float64)
    R = E.copy()
    Z = preconditioner.apply(R)
    D = Z.copy()

    residual_norms = np.linalg.norm(R, axis=0)
    iterations = np.zeros(s, dtype=np.int32)
    active = np.where(residual_norms > tol)[0]

    breakdowns = 0
    restarts = 0
    regularized_small_solves = 0
    pinv_small_solves = 0

    for it in range(1, max_iter + 1):
        if active.size == 0:
            break

        R_a = R[:, active]
        Z_a = Z[:, active]
        D_a = D[:, active]

        GD_a = G_op.matmat(D_a)

        alpha_lhs = D_a.T @ GD_a
        alpha_rhs = R_a.T @ Z_a
        alpha = _small_solve(alpha_lhs, alpha_rhs, ridge=ridge)

        if alpha.regularized:
            regularized_small_solves += 1
        if alpha.pinv:
            pinv_small_solves += 1
            breakdowns += 1

        old_active = active.copy()

        R_old = R_a.copy()
        Z_old = Z_a.copy()
        D_old = D_a.copy()

        P[:, old_active] += D_a @ alpha.X
        R[:, old_active] -= GD_a @ alpha.X

        if residual_recompute_every and it % residual_recompute_every == 0:
            R[:, old_active] = E[:, old_active] - G_op.matmat(P[:, old_active])

        residual_norms[old_active] = np.linalg.norm(R[:, old_active], axis=0)
        iterations[old_active] = it

        new_active = old_active[residual_norms[old_active] > tol]

        if new_active.size == 0:
            active = new_active
            break

        active_pos = {int(col): pos for pos, col in enumerate(old_active)}
        keep_positions = np.array(
            [active_pos[int(col)] for col in new_active],
            dtype=int,
        )

        R_old = R_old[:, keep_positions]
        Z_old = Z_old[:, keep_positions]
        D_old = D_old[:, keep_positions]

        Z_new = preconditioner.apply(R[:, new_active])

        beta_lhs = R_old.T @ Z_old
        beta_rhs = R[:, new_active].T @ Z_new

        try:
            # Standard column-block PCG convention:
            #     (R_old^T Z_old) B_t = R_new^T Z_new
            # with
            #     D_new = Z_new + D_old B_t.
            beta = _small_solve(beta_lhs, beta_rhs, ridge=ridge)

            if beta.regularized:
                regularized_small_solves += 1
            if beta.pinv:
                pinv_small_solves += 1
                breakdowns += 1
                restarts += 1
                beta_X = np.zeros_like(beta.X)
            else:
                beta_X = beta.X

        except (FloatingPointError, ValueError, np.linalg.LinAlgError):
            breakdowns += 1
            restarts += 1
            beta_X = np.zeros((new_active.size, new_active.size), dtype=np.float64)

        Z[:, new_active] = Z_new
        D[:, new_active] = Z_new + D_old @ beta_X

        active = new_active

    converged = residual_norms <= tol

    return BlockPCGResult(
        P=P,
        residual_norms=residual_norms,
        iterations=iterations,
        converged=converged,
        breakdowns=int(breakdowns),
        restarts=int(restarts),
        regularized_small_solves=int(regularized_small_solves),
        pinv_small_solves=int(pinv_small_solves),
    )
