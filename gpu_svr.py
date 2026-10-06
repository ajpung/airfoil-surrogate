"""Epsilon-SVR (RBF kernel) solved on the GPU with PyTorch.

Solves the same dual as sklearn.svm.SVR, in terms of beta = alpha - alpha*:

    min_beta  0.5 beta' K beta - y' beta + eps * |beta|_1,   -C <= beta <= C

with accelerated proximal gradient (FISTA + adaptive restart). The prox of
eps*|.| plus the box is soft-threshold-then-clip. The intercept is absorbed
by using K + 1 (a lightly regularized bias), which drops sklearn's
sum(beta) = 0 constraint; with standardized targets the difference is
negligible (checked against sklearn in check_gpu_svr.py).

The solver is batched: K is (b, n, n) and Y is (b, n, k), so e.g. all CV folds
(padded to a common n, padding masked out) and both targets (Cl, log Cd) are
solved in one pass. Memory is b * n^2 * 4 bytes.
"""
import numpy as np
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cuda.matmul.allow_tf32 = False  # keep kernel distances in true fp32


def rbf(A, B, gamma):
    return torch.exp(-gamma * torch.cdist(A, B).pow_(2))


@torch.no_grad()
def solve(K, Y, C, eps, mask=None, max_iter=20000, tol=1e-6):
    """K: (b, n, n), already + 1 for the bias. Y: (b, n, k). mask: (b, n, 1) bool,
    False rows are padding and are held at beta = 0. Returns beta (b, n, k), iters."""
    b, n, k = Y.shape
    m = torch.ones(b, n, 1, dtype=K.dtype, device=K.device) if mask is None else mask.to(K.dtype)
    # Lipschitz constant per batch = largest eigenvalue of K (power iteration)
    v = torch.randn(b, n, 1, dtype=K.dtype, device=K.device,
                    generator=torch.Generator(K.device).manual_seed(0)) * m
    for _ in range(50):
        v = K @ v
        v /= v.norm(dim=1, keepdim=True)
    step = 1.0 / ((v.transpose(1, 2) @ K @ v) * 1.01)  # (b, 1, 1)

    beta, Kb = torch.zeros_like(Y), torch.zeros_like(Y)
    z, Kz = beta.clone(), Kb.clone()
    t = torch.ones(b, 1, k, dtype=K.dtype, device=K.device)
    ynorm = (Y * m).norm(dim=1, keepdim=True).clamp_min(1e-12)
    for it in range(max_iter):
        new = z - step * (Kz - Y)
        new = torch.sign(new) * (new.abs() - step * eps).clamp_min(0)  # soft-threshold
        new = new.clamp(-C, C) * m                                     # box + padding
        Knew = K @ new                                                 # one bmm / iter
        d, Kd = new - beta, Knew - Kb
        # adaptive restart (O'Donoghue & Candes), per fold/target, no host sync
        t = torch.where(((z - new) * d).sum(1, keepdim=True) > 0, torch.ones_like(t), t)
        t_new = 0.5 * (1 + torch.sqrt(1 + 4 * t * t))
        mom = (t - 1) / t_new
        z, Kz = new + mom * d, Knew + mom * Kd
        beta, Kb, t = new, Knew, t_new
        # stop when the fitted values on the training set stop moving
        if it % 50 == 0 and float(((Kd * m).norm(dim=1, keepdim=True) / ynorm).max()) < tol:
            break
    return beta, it + 1


class GpuSVR:
    """sklearn-like multi-output RBF epsilon-SVR. Expects already-scaled X, Y."""

    def __init__(self, C=1.0, epsilon=0.1, gamma="scale", max_iter=20000, tol=1e-6):
        self.C, self.epsilon, self.gamma = C, epsilon, gamma
        self.max_iter, self.tol = max_iter, tol

    def fit(self, X, Y):
        Y = np.asarray(Y, np.float32)
        Y = Y[:, None] if Y.ndim == 1 else Y
        self.gamma_ = 1.0 / (X.shape[1] * X.var()) if self.gamma == "scale" else float(self.gamma)
        Xg = torch.tensor(X, dtype=torch.float32, device=DEVICE)
        K = rbf(Xg, Xg, self.gamma_)
        K += 1.0
        beta, self.n_iter_ = solve(K[None], torch.tensor(Y, device=DEVICE)[None],
                                   self.C, self.epsilon, None, self.max_iter, self.tol)
        del K
        beta = beta[0]
        sv = (beta.abs() > 0).any(1)
        self.sv_ = Xg[sv].cpu().numpy()
        self.beta_ = beta[sv].cpu().numpy()
        return self

    @torch.no_grad()
    def predict(self, X, batch=8192):
        sv = torch.tensor(self.sv_, device=DEVICE)
        beta = torch.tensor(self.beta_, device=DEVICE)
        out = []
        for i in range(0, len(X), batch):
            Xb = torch.tensor(X[i:i + batch], dtype=torch.float32, device=DEVICE)
            out.append(((rbf(Xb, sv, self.gamma_) + 1.0) @ beta).cpu().numpy())
        P = np.concatenate(out).astype(np.float64)
        return P[:, 0] if P.shape[1] == 1 else P
