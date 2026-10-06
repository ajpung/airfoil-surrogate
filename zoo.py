"""Model zoo. Every model maps X=[AoA, CST1..8] -> Y=[Cl, log Cd] via
fit(X, Y, groups) / predict(X). `groups` is only used for grouped inner
early-stopping splits so no geometry straddles train/early-stop data.
"""
import numpy as np
import torch
import torch.nn as nn
from sklearn.base import clone
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

from features import physics_features  # noqa: F401  (re-exported)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# --------------------------------------------------------------------------
# sklearn-style wrappers
# --------------------------------------------------------------------------
class PerTarget:
    """One clone of `est` per output column, optional X/Y standardization."""

    def __init__(self, est, scale_x=False, scale_y=False, subsample=None, seed=0):
        self.est, self.scale_x, self.scale_y = est, scale_x, scale_y
        self.subsample, self.seed = subsample, seed

    def fit(self, X, Y, groups=None):
        if self.subsample and len(X) > self.subsample:
            idx = np.random.default_rng(self.seed).choice(len(X), self.subsample, replace=False)
            X, Y = X[idx], Y[idx]
        self.sx = StandardScaler().fit(X) if self.scale_x else None
        self.sy = StandardScaler().fit(Y) if self.scale_y else None
        Xs = self.sx.transform(X) if self.sx else X
        Ys = self.sy.transform(Y) if self.sy else Y
        self.models = [clone(self.est).fit(Xs, Ys[:, j]) for j in range(Y.shape[1])]
        return self

    def predict(self, X):
        Xs = self.sx.transform(X) if self.sx else X
        P = np.column_stack([m.predict(Xs) for m in self.models])
        return self.sy.inverse_transform(P) if self.sy else P


class WithFeatures:
    """Append deterministic derived features (no labels involved) to X."""

    def __init__(self, model, fn):
        self.model, self.fn = model, fn

    def fit(self, X, Y, groups=None):
        self.model.fit(self.fn(X), Y, groups)
        return self

    def predict(self, X):
        return self.model.predict(self.fn(X))


# --------------------------------------------------------------------------
# Torch models
# --------------------------------------------------------------------------
class TorchReg:
    """Standardize X/Y, train with AdamW + cosine LR, grouped early stopping."""

    def __init__(self, build, epochs=300, lr=2e-3, wd=1e-4, batch=512,
                 patience=40, loss="mse", seed=0):
        self.build, self.epochs, self.lr, self.wd = build, epochs, lr, wd
        self.batch, self.patience, self.loss, self.seed = batch, patience, loss, seed

    def _loss(self, out, y):
        if self.loss == "mse":
            return ((out[:, :2] - y) ** 2).mean()
        if self.loss == "student_t":
            mu, log_s = out[:, :2], out[:, 2:4].clamp(-7, 3)
            nu = 3.0
            z2 = ((y - mu) / log_s.exp()) ** 2
            return (log_s + 0.5 * (nu + 1) * torch.log1p(z2 / nu)).mean()
        raise ValueError(self.loss)

    def fit(self, X, Y, groups=None):
        torch.manual_seed(self.seed)
        self.sx, self.sy = StandardScaler().fit(X), StandardScaler().fit(Y)
        Xs, Ys = self.sx.transform(X), self.sy.transform(Y)
        if groups is None:
            groups = np.arange(len(X))
        tr, va = next(GroupShuffleSplit(1, test_size=0.1, random_state=self.seed)
                      .split(Xs, groups=groups))
        t = lambda a: torch.tensor(a, dtype=torch.float32, device=DEVICE)
        Xt, Yt, Xv, Yv = t(Xs[tr]), t(Ys[tr]), t(Xs[va]), t(Ys[va])

        self.net = self.build(X.shape[1]).to(DEVICE)
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=self.wd)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, self.epochs)
        best, best_state, bad = np.inf, None, 0
        n = len(Xt)
        for ep in range(self.epochs):
            self.net.train()
            perm = torch.randperm(n, device=DEVICE)
            for i in range(0, n, self.batch):
                b = perm[i:i + self.batch]
                opt.zero_grad()
                self._loss(self.net(Xt[b]), Yt[b]).backward()
                opt.step()
            sched.step()
            self.net.eval()
            with torch.no_grad():
                v = ((self.net(Xv)[:, :2] - Yv) ** 2).mean().item()
            if v < best:
                best, bad = v, 0
                best_state = {k: x.detach().clone() for k, x in self.net.state_dict().items()}
            else:
                bad += 1
                if bad >= self.patience:
                    break
        self.net.load_state_dict(best_state)
        return self

    def predict(self, X):
        self.net.eval()
        with torch.no_grad():
            out = self.net(torch.tensor(self.sx.transform(X), dtype=torch.float32,
                                        device=DEVICE))[:, :2].cpu().numpy()
        return self.sy.inverse_transform(out)


class ResMLP(nn.Module):
    def __init__(self, d_in, width=256, depth=4, d_out=2, dropout=0.0):
        super().__init__()
        self.inp = nn.Linear(d_in, width)
        self.blocks = nn.ModuleList(nn.Sequential(
            nn.LayerNorm(width), nn.Linear(width, width), nn.SiLU(),
            nn.Dropout(dropout), nn.Linear(width, width)) for _ in range(depth))
        self.out = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, d_out))

    def forward(self, x):
        h = self.inp(x)
        for b in self.blocks:
            h = h + b(h)
        return self.out(h)


class FourierMLP(nn.Module):
    """Random Fourier feature embedding followed by an MLP."""

    def __init__(self, d_in, n_freq=128, sigma=1.0, width=256, depth=3):
        super().__init__()
        self.register_buffer("B", torch.randn(d_in, n_freq) * sigma)
        layers, d = [], 2 * n_freq + d_in
        for _ in range(depth):
            layers += [nn.Linear(d, width), nn.SiLU()]
            d = width
        self.mlp = nn.Sequential(*layers, nn.Linear(d, 2))

    def forward(self, x):
        p = 2 * np.pi * x @ self.B
        return self.mlp(torch.cat([x, torch.sin(p), torch.cos(p)], 1))


class KANLayer(nn.Module):
    """FastKAN-style layer: per-input Gaussian RBF basis + SiLU base branch."""

    def __init__(self, d_in, d_out, grid=8, lo=-2.5, hi=2.5):
        super().__init__()
        self.register_buffer("centers", torch.linspace(lo, hi, grid))
        self.h = (hi - lo) / (grid - 1)
        self.norm = nn.LayerNorm(d_in)
        self.spline = nn.Linear(d_in * grid, d_out)
        self.base = nn.Linear(d_in, d_out)

    def forward(self, x):
        z = self.norm(x)
        phi = torch.exp(-((z[..., None] - self.centers) / self.h) ** 2)
        return self.spline(phi.flatten(1)) + self.base(nn.functional.silu(x))


class KAN(nn.Module):
    def __init__(self, d_in, widths=(64, 64), grid=8):
        super().__init__()
        dims = [d_in, *widths, 2]
        self.layers = nn.Sequential(*[KANLayer(a, b, grid) for a, b in zip(dims, dims[1:])])

    def forward(self, x):
        return self.layers(x)


class DragPolarNet(nn.Module):
    """Physics-structured hypernetwork.

    A network reads the shape (standardized CST) and emits per-airfoil curve
    parameters; AoA enters only through closed forms:
      Cl(a)  = c0 + c1 a + c2 a^2 + c3 a^3                  (smooth lift curve)
      Cd     = Cd0 + k (Cl - Cl_m)^2 + k4 (Cl - Cl_m)^4     (drag polar)
    Outputs are re-standardized to match the training targets.
    Standardization stats for Y are injected after construction.
    """

    def __init__(self, d_in, width=256, depth=3):
        super().__init__()
        layers, d = [], d_in - 1
        for _ in range(depth):
            layers += [nn.Linear(d, width), nn.SiLU()]
            d = width
        self.body = nn.Sequential(*layers, nn.Linear(d, 8))
        self.register_buffer("ymu", torch.zeros(2))
        self.register_buffer("ysd", torch.ones(2))

    def forward(self, x):
        a, shape = x[:, :1], x[:, 1:]          # a is standardized AoA
        p = self.body(shape)
        cl = p[:, 0:1] + p[:, 1:2] * a + 0.1 * p[:, 2:3] * a ** 2 + 0.1 * p[:, 3:4] * a ** 3
        cl_phys = cl * self.ysd[0] + self.ymu[0]
        d = cl_phys - p[:, 6:7]
        cd = (torch.exp(p[:, 4:5] - 4.0) + torch.exp(p[:, 5:6] - 4.0) * d ** 2
              + torch.exp(p[:, 7:8] - 5.0) * d ** 4)
        logcd = (torch.log(cd) - self.ymu[1]) / self.ysd[1]
        return torch.cat([cl, logcd], 1)


class DragPolarReg(TorchReg):
    def fit(self, X, Y, groups=None):
        sy = StandardScaler().fit(Y)
        inner = self.build

        def build(d_in):
            net = inner(d_in)
            net.ymu.copy_(torch.tensor(sy.mean_, dtype=torch.float32))
            net.ysd.copy_(torch.tensor(sy.scale_, dtype=torch.float32))
            return net

        self.build = build
        try:
            return super().fit(X, Y, groups)
        finally:
            self.build = inner


class AvgEnsemble:
    """Placeholder marker: built post-hoc from saved out-of-fold predictions."""

    def __init__(self, members):
        self.members = members


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------
def registry():
    import lightgbm as lgb
    import xgboost as xgb
    from catboost import CatBoostRegressor
    from sklearn.ensemble import (ExtraTreesRegressor, HistGradientBoostingRegressor,
                                  RandomForestRegressor)
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
    from sklearn.kernel_approximation import Nystroem
    from sklearn.linear_model import Ridge, RidgeCV
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.svm import SVR

    lgb_params = dict(n_estimators=2000, learning_rate=0.03, num_leaves=31,
                      min_child_samples=10, subsample=0.8, subsample_freq=1,
                      colsample_bytree=0.9, random_state=0, verbose=-1)
    solid = {
        "S01 Poly4 Ridge": lambda: PerTarget(make_pipeline(
            PolynomialFeatures(4), RidgeCV(alphas=np.logspace(-6, 2, 17))), scale_x=True),
        "S02 KNN (k=10)": lambda: PerTarget(
            KNeighborsRegressor(10, weights="distance"), scale_x=True),
        "S03 SVR (RBF)": lambda: PerTarget(SVR(C=10, epsilon=0.02, gamma="scale"),
                                           scale_x=True, scale_y=True),
        "S04 Nystroem KRR": lambda: PerTarget(make_pipeline(
            Nystroem(gamma=0.1, n_components=3000, random_state=0), Ridge(1e-3)),
            scale_x=True, scale_y=True),
        "S05 GP (ARD, 2.5k subsample)": lambda: PerTarget(GaussianProcessRegressor(
            ConstantKernel() * RBF(np.ones(9)) + WhiteKernel(0.1), normalize_y=True,
            random_state=0), scale_x=True, subsample=2500),
        "S06 Random Forest": lambda: PerTarget(RandomForestRegressor(
            500, min_samples_leaf=2, max_features=0.6, n_jobs=-1, random_state=0)),
        "S07 Extra Trees": lambda: PerTarget(ExtraTreesRegressor(
            500, min_samples_leaf=2, max_features=0.8, n_jobs=-1, random_state=0)),
        "S08 HistGradBoost": lambda: PerTarget(HistGradientBoostingRegressor(
            max_iter=1500, learning_rate=0.05, max_leaf_nodes=31, random_state=0)),
        "S09 LightGBM": lambda: PerTarget(lgb.LGBMRegressor(**lgb_params)),
        "S10 LightGBM Huber": lambda: PerTarget(lgb.LGBMRegressor(
            objective="huber", alpha=0.1, **lgb_params), scale_y=True),
        "S11 XGBoost": lambda: PerTarget(xgb.XGBRegressor(
            n_estimators=2000, learning_rate=0.03, max_depth=6, subsample=0.8,
            colsample_bytree=0.9, tree_method="hist", random_state=0, n_jobs=-1)),
        "S12 CatBoost": lambda: PerTarget(CatBoostRegressor(
            iterations=3000, learning_rate=0.05, depth=6, verbose=0, random_seed=0,
            thread_count=-1)),
        "S13 sklearn MLP": lambda: PerTarget(MLPRegressor(
            hidden_layer_sizes=(128, 128, 128), alpha=1e-4, max_iter=500, early_stopping=True,
            n_iter_no_change=25, random_state=0), scale_x=True, scale_y=True),
        "S14 Torch ResMLP": lambda: TorchReg(lambda d: ResMLP(d)),
        "S15 Avg ensemble (S09+S12+S14)": lambda: AvgEnsemble(
            ["S09 LightGBM", "S12 CatBoost", "S14 Torch ResMLP"]),
    }
    wild = {
        "W1 Drag-polar hypernet": lambda: DragPolarReg(lambda d: DragPolarNet(d)),
        "W2 KAN (RBF splines)": lambda: TorchReg(lambda d: KAN(d)),
        "W3 Student-t heteroscedastic NN": lambda: TorchReg(
            lambda d: ResMLP(d, d_out=4), loss="student_t"),
        "W4 Fourier-feature MLP": lambda: TorchReg(lambda d: FourierMLP(d, sigma=0.5)),
        "W5 Thin-airfoil physics feats + LightGBM": lambda: WithFeatures(
            PerTarget(lgb.LGBMRegressor(**lgb_params)), physics_features),
    }
    return {**solid, **wild}
