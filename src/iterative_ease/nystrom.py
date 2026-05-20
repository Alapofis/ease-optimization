import numpy as np


class RegularizationPreconditioner:
    """
    Scalar preconditioner M = reg * I.

    For PCG this is equivalent to the unpreconditioned method up to a constant
    scaling of residuals, and is used for rank=0 or numerical fallback.
    """

    def __init__(self, reg: float):
        if reg <= 0:
            raise ValueError("reg must be positive")
        self.reg = float(reg)
        self.rank = 0

    def apply(self, R: np.ndarray) -> np.ndarray:
        R = np.asarray(R, dtype=np.float64)
        if R.ndim != 2:
            raise ValueError("R must be 2D")
        return R / self.reg


class NystromPreconditioner:
    """
    Low-rank Nystrom preconditioner for:
        G = S + reg * I, S = X.T @ X.

    Approximation:
        S ~= Q diag(theta) Q.T

    Preconditioner:
        M = reg * I + Q diag(theta) Q.T
    """

    def __init__(self, Q: np.ndarray, theta: np.ndarray, reg: float):
        self.Q = np.asarray(Q, dtype=np.float64)
        self.theta = np.asarray(theta, dtype=np.float64)
        self.reg = float(reg)

        if self.Q.ndim != 2:
            raise ValueError("Q must be 2D")
        if self.theta.ndim != 1:
            raise ValueError("theta must be 1D")
        if self.Q.shape[1] != self.theta.shape[0]:
            raise ValueError("Q and theta have incompatible shapes")
        if self.reg <= 0:
            raise ValueError("reg must be positive")
        if self.theta.size == 0:
            raise ValueError("theta must be non-empty")
        if np.any(self.theta <= 0):
            raise ValueError("theta must contain positive values")

        self.rank = int(self.theta.shape[0])
        self._coef = self.theta / (self.reg * (self.reg + self.theta))

    @classmethod
    def build(
        cls,
        G_op,
        rank: int,
        oversample: int = 10,
        seed: int = 42,
        jitter: float = 1e-10,
    ):
        if rank <= 0:
            return RegularizationPreconditioner(G_op.reg)
        if oversample < 0:
            raise ValueError("oversample must be non-negative")

        n = G_op.n_items
        ell = min(n, int(rank) + int(oversample))
        if ell <= 0:
            return RegularizationPreconditioner(G_op.reg)

        rng = np.random.default_rng(seed)
        Omega = rng.standard_normal((n, ell))

        # Y = S Omega, with S applied matrix-free.
        Y = G_op.s_matmat(Omega)

        H = Omega.T @ Y
        H = 0.5 * (H + H.T)
        scale = max(1.0, float(np.linalg.norm(H, ord=np.inf)))
        H.flat[:: H.shape[0] + 1] += jitter * scale

        try:
            eigvals, U = np.linalg.eigh(H)
        except np.linalg.LinAlgError:
            return RegularizationPreconditioner(G_op.reg)

        keep = eigvals > jitter * scale
        if not np.any(keep):
            return RegularizationPreconditioner(G_op.reg)

        eigvals = eigvals[keep]
        U = U[:, keep]
        B = Y @ (U / np.sqrt(eigvals))

        try:
            Q, _ = np.linalg.qr(B, mode="reduced")
        except np.linalg.LinAlgError:
            return RegularizationPreconditioner(G_op.reg)

        SQ = G_op.s_matmat(Q)
        T = Q.T @ SQ
        T = 0.5 * (T + T.T)

        try:
            theta, V = np.linalg.eigh(T)
        except np.linalg.LinAlgError:
            return RegularizationPreconditioner(G_op.reg)

        order = np.argsort(theta)[::-1]
        theta = theta[order]
        V = V[:, order]

        t_scale = max(1.0, float(np.max(np.abs(theta))) if theta.size else 1.0)
        keep = theta > jitter * t_scale
        if not np.any(keep):
            return RegularizationPreconditioner(G_op.reg)

        theta = theta[keep]
        V = V[:, keep]
        r = min(int(rank), theta.size)

        return cls(Q=Q @ V[:, :r], theta=theta[:r], reg=G_op.reg)

    def apply(self, R: np.ndarray) -> np.ndarray:
        R = np.asarray(R, dtype=np.float64)
        if R.ndim != 2:
            raise ValueError("R must be 2D")

        T = self.Q.T @ R
        return R / self.reg - self.Q @ (self._coef[:, None] * T)
