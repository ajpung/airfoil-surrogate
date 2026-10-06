"""Check GpuSVR against sklearn SVR on one CV fold (baseline config)."""
import time

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from common import folds, load_dev, train_mask, xy
from gpu_svr import GpuSVR


def main():
    df = load_dev()
    X, Y = xy(df)
    tr, va = folds(df)[0]
    keep = tr[train_mask(df.iloc[tr], "conflict+jump")]
    sx, sy = StandardScaler().fit(X[keep]), StandardScaler().fit(Y[keep])
    Xs, Ys, Xv = sx.transform(X[keep]), sy.transform(Y[keep]), sx.transform(X[va])
    smooth = df.smooth.fillna(False).to_numpy(bool)[va]
    print(f"train n={len(keep)}, val n={len(va)}")

    t0 = time.time()
    g = GpuSVR(C=10, epsilon=0.02).fit(Xs, Ys)
    Pg = sy.inverse_transform(g.predict(Xv))
    print(f"GPU: {time.time() - t0:.1f}s, {g.n_iter_} iters, {len(g.sv_)} SVs")

    t0 = time.time()
    Ps = sy.inverse_transform(np.column_stack(
        [SVR(C=10, epsilon=0.02, gamma="scale").fit(Xs, Ys[:, j]).predict(Xv) for j in (0, 1)]))
    print(f"sklearn: {time.time() - t0:.1f}s")

    for name, P in [("sklearn", Ps), ("gpu", Pg)]:
        yv = Y[va][smooth]
        p = P[smooth]
        mdape = 100 * np.median(np.abs(np.exp(p[:, 1]) - np.exp(yv[:, 1])) / np.exp(yv[:, 1]))
        print(f"{name:8s} smooth Cd MdAPE {mdape:.3f}%  Cl MAE {np.mean(np.abs(p[:, 0] - yv[:, 0])):.5f}")
    d = np.abs(Pg - Ps)
    print(f"|gpu - sklearn| on val: Cl max {d[:, 0].max():.2e} mean {d[:, 0].mean():.2e}; "
          f"logCd max {d[:, 1].max():.2e} mean {d[:, 1].mean():.2e}")


if __name__ == "__main__":
    main()
