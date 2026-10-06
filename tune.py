"""Random-search tuning of the top CV model (RBF SVR), dev split only.

Runs on the GPU (gpu_svr.py): per config, all 5 folds x 2 targets are one
batched solve, so the CPU stays mostly idle.

Each target (Cl, log Cd) is an independent SVR, so each gets its own search,
scored on the same grouped folds as compare.py:
  Cd: median |rel. error| on smooth-Cd validation airfoils
  Cl: MAE on smooth-Cd validation airfoils
Selection uses the one-standard-error rule: among configs whose mean fold
score is within 1 SE of the best, take the smoothest (smallest C, then largest
epsilon) to avoid overfitting the CV.
"""
import json
import sys
import time

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from common import RESULTS, folds, load_dev, train_mask, xy
from features import physics_features
from gpu_svr import DEVICE, rbf, solve

torch.set_num_threads(2)  # leave the CPU free; the work is on the GPU

N_TRIALS = 40
SEED = 123
BASELINE = {"C": 10.0, "epsilon": 0.02, "gamma": "scale", "filter": "conflict+jump", "feats": "raw"}


def sample(rng):
    return {
        "C": float(np.exp(rng.uniform(np.log(0.3), np.log(300)))),
        "epsilon": float(np.exp(rng.uniform(np.log(1e-3), np.log(0.2)))),
        "gamma": float(np.exp(rng.uniform(np.log(0.02), np.log(2.0)))),
        "filter": str(rng.choice(["none", "conflict", "conflict+jump", "smooth"])),
        "feats": str(rng.choice(["raw", "phys"])),
    }


def fit_predict_folds(cfg, X, Y, df, fl):
    """Fit cfg on every fold at once (both targets). Returns per-fold val preds."""
    F = physics_features(X) if cfg["feats"] == "phys" else X
    parts = []
    for tr, va in fl:
        keep = tr[train_mask(df.iloc[tr], cfg["filter"])]
        sx, sy = StandardScaler().fit(F[keep]), StandardScaler().fit(Y[keep])
        parts.append((sx.transform(F[keep]), sy.transform(Y[keep]), sx.transform(F[va]), sy))
    b, n = len(parts), max(len(p[0]) for p in parts)
    t = lambda a: torch.tensor(a, dtype=torch.float32, device=DEVICE)
    Xtr = torch.zeros(b, n, F.shape[1], device=DEVICE)
    Ytr = torch.zeros(b, n, 2, device=DEVICE)
    mask = torch.zeros(b, n, 1, dtype=torch.bool, device=DEVICE)
    for i, (xs, ys, _, _) in enumerate(parts):
        Xtr[i, :len(xs)], Ytr[i, :len(xs)], mask[i, :len(xs)] = t(xs), t(ys), True
    g = 1.0 / F.shape[1] if cfg["gamma"] == "scale" else cfg["gamma"]
    K = rbf(Xtr, Xtr, g)
    K += 1.0
    beta, iters = solve(K, Ytr, cfg["C"], cfg["epsilon"], mask)
    del K
    out = []
    for i, (xs, _, xv, sy) in enumerate(parts):
        P = (rbf(t(xv), Xtr[i, :len(xs)], g) + 1.0) @ beta[i, :len(xs)]
        out.append(sy.inverse_transform(P.cpu().numpy().astype(np.float64)))
    return out, iters


def fold_score(j, y, p):
    if j == 1:
        return 100 * np.median(np.abs(np.exp(p) - np.exp(y)) / np.exp(y))
    return np.mean(np.abs(p - y))


def sample_r2(rng):
    """Round 2: extend gamma below round 1's floor (winners sat at 0.021-0.028),
    restricted to the filters/features that won round 1."""
    return {
        "C": float(np.exp(rng.uniform(np.log(0.3), np.log(30)))),
        "epsilon": float(np.exp(rng.uniform(np.log(1e-3), np.log(0.1)))),
        "gamma": float(np.exp(rng.uniform(np.log(0.002), np.log(0.05)))),
        "filter": str(rng.choice(["conflict+jump", "smooth"])),
        "feats": "raw",
    }


# round 1 winners, re-run in round 2 as anchors (Cd pick, Cl pick)
R1_CD = {"C": 5.0453836406919805, "epsilon": 0.0214697165330862,
         "gamma": 0.027652305241177226, "filter": "smooth", "feats": "raw"}
R1_CL = {"C": 0.9954424642921836, "epsilon": 0.005243597075148726,
         "gamma": 0.0213785846251826, "filter": "conflict+jump", "feats": "raw"}


def main():
    df = load_dev()
    X, Y = xy(df)
    fl = folds(df)
    smooth = df.smooth.fillna(False).to_numpy(bool)
    round2 = "--round2" in sys.argv
    if round2:
        rng = np.random.default_rng(SEED + 1)
        cfgs = [R1_CD, R1_CL] + [sample_r2(rng) for _ in range(20)]
    else:
        rng = np.random.default_rng(SEED)
        cfgs = [BASELINE] + [sample(rng) for _ in range(N_TRIALS)]
    tag = "_r2" if round2 else ""

    rows = []
    t0 = time.time()
    for c, cfg in enumerate(cfgs):
        t1 = time.time()
        preds, iters = fit_predict_folds(cfg, X, Y, df, fl)
        for k, ((_, va), p) in enumerate(zip(fl, preds)):
            m = smooth[va]
            for j in (0, 1):
                rows.append({"cfg": c, "target": ["Cl", "Cd"][j], "fold": k,
                             "score": fold_score(j, Y[va, j][m], p[m, j])})
        cd = np.mean([r["score"] for r in rows[-10:] if r["target"] == "Cd"])
        print(f"cfg {c:2d}/{len(cfgs) - 1}: {time.time() - t1:5.1f}s, {iters:5d} iters, "
              f"smooth Cd MdAPE {cd:.3f}%", flush=True)
    print(f"{len(cfgs)} configs x 5 folds x 2 targets in {time.time() - t0:.0f}s")

    r = (pd.DataFrame(rows).groupby(["target", "cfg"]).score
         .agg(mean="mean", se=lambda s: s.std(ddof=1) / np.sqrt(len(s))).reset_index())
    r = r.join(pd.DataFrame(cfgs), on="cfg")

    chosen = {}
    for t, g in r.groupby("target"):
        g = g.sort_values("mean")
        best = g.iloc[0]
        ok = g[g["mean"] <= best["mean"] + best["se"]]
        pick = ok.sort_values(["C", "epsilon"], ascending=[True, False]).iloc[0]
        unit = "MdAPE%" if t == "Cd" else "MAE"
        print(f"\n== {t} (smooth-val {unit}); cfg 0 = "
              f"{g[g.cfg == 0]['mean'].iloc[0]:.4f}")
        print(g.head(8)[["cfg", "mean", "se", "C", "epsilon", "gamma", "filter", "feats"]]
              .to_string(index=False, float_format="{:.4f}".format))
        print(f"best cfg {int(best.cfg)}: {best['mean']:.4f} +/- {best['se']:.4f}; "
              f"1-SE pick cfg {int(pick.cfg)}: {pick['mean']:.4f}")
        chosen[t] = {**cfgs[int(pick.cfg)], "cv_mean": float(pick["mean"]),
                     "cv_se": float(pick["se"])}

    r.to_csv(RESULTS / f"tune_svr{tag}.csv", index=False)
    (RESULTS / f"svr_tuned{tag}.json").write_text(json.dumps(chosen, indent=2))
    print(f"\nwrote {RESULTS / f'svr_tuned{tag}.json'}")


if __name__ == "__main__":
    main()
