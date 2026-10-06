"""Final evaluation on the held-out test geometries (touched only here).

Fits on the full dev split, predicts the test split once, and reports metrics
on three test subsets:
  raw       -- every test point (noisy labels included)
  unflagged -- test points not flagged conflict/cd_jump
  smooth    -- test airfoils whose Cd curve looks converged
Saves the tuned SVR to results/svr_final.joblib.
"""
import json
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from common import RESULTS, load, load_dev, score, train_mask, xy
from features import physics_features
from gpu_svr import GpuSVR
from zoo import registry

warnings.filterwarnings("ignore")


class TunedSVR:
    """Per-target SVRs with their own training filter and feature set.
    Uses the same GPU solver the configs were tuned with (gpu_svr.py)."""

    def __init__(self, cfgs):
        self.cfgs = cfgs  # {"Cl": {...}, "Cd": {...}}

    def fit(self, df):
        X, Y = xy(df)
        self.parts = []
        for j, t in enumerate(["Cl", "Cd"]):
            c = self.cfgs[t]
            keep = train_mask(df, c["filter"])
            F = self._feats(X, c)[keep]
            sx, sy = StandardScaler().fit(F), StandardScaler().fit(Y[keep, j:j + 1])
            m = GpuSVR(C=c["C"], epsilon=c["epsilon"], gamma=c["gamma"])
            m.fit(sx.transform(F), sy.transform(Y[keep, j:j + 1]))
            self.parts.append((sx, sy, m))
        return self

    @staticmethod
    def _feats(X, c):
        return physics_features(X) if c["feats"] == "phys" else X

    def predict(self, X):
        """X columns: AoA, CST1..8. Returns [:,0]=Cl, [:,1]=log Cd."""
        cols = []
        for (sx, sy, m), t in zip(self.parts, ["Cl", "Cd"]):
            F = sx.transform(self._feats(X, self.cfgs[t]))
            cols.append(sy.inverse_transform(m.predict(F)[:, None]).ravel())
        return np.column_stack(cols)


def fit_registry(name, dev, mode="conflict+jump"):
    X, Y = xy(dev)
    keep = train_mask(dev, mode)
    return registry()[name]().fit(X[keep], Y[keep], dev.geom_id.to_numpy()[keep])


def main():
    dev, test = load_dev(), load("test")
    Xt, _ = xy(test)
    # round-2 search (extended gamma, round-1 winners re-run as anchors) supersedes round 1
    tuned = RESULTS / "svr_tuned_r2.json"
    tuned = tuned if tuned.exists() else RESULTS / "svr_tuned.json"
    cfgs = json.loads(tuned.read_text())
    print(f"configs from {tuned.name}: {json.dumps(cfgs)}")

    preds = {}
    means = dev.groupby("AoA")[["Cl", "Cd"]].mean()
    preds["AoA-mean (no shape)"] = np.column_stack(
        [test.AoA.map(means.Cl), np.log(test.AoA.map(means.Cd))])
    preds["LightGBM L2 (original baseline)"] = fit_registry("S09 LightGBM", dev).predict(Xt)
    preds["SVR default (S03)"] = fit_registry("S03 SVR (RBF)", dev).predict(Xt)
    from final_test import TunedSVR as Importable  # pickle as final_test.TunedSVR, not __main__
    final = Importable(cfgs).fit(dev)
    preds["SVR tuned (final)"] = final.predict(Xt)
    joblib.dump(final, RESULTS / "svr_final.joblib")

    res = pd.DataFrame({k: score(test, P) for k, P in preds.items()}).T
    res.to_csv(RESULTS / "test_metrics.csv")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print(f"Held-out test: {len(test)} pts, {test.geom_id.nunique()} geometries "
          f"(unflagged {int((~(test.conflict | test.cd_jump)).sum())}, "
          f"smooth {int(test.smooth.fillna(False).sum())})\n")
    for sub in ["raw", "unflagged", "smooth"]:
        cols = [f"{sub} {m}" for m in ["Cd MdAPE%", "Cd MAE", "Cd R2", "Cl MAE", "Cl R2"]]
        print(f"-- {sub}")
        print(res[cols].to_string(float_format="{:.4f}".format), "\n")


if __name__ == "__main__":
    main()
