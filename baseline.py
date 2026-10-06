"""Baseline surrogates: (8 CST coeffs + AoA) -> (Cl, Cd).

5-fold GroupKFold on geom_id so no airfoil shape appears in both train and
test. Cd is modelled in log space. Reports metrics in physical units.
"""
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).parent
DATA = ROOT / "data" / "airfoil_cst_cfd_clean.csv"
FEATURES = ["AoA"] + [f"CST Coeff {i}" for i in range(1, 9)]
SEED = 0


def make_models():
    return {
        "LightGBM": lambda: lgb.LGBMRegressor(
            n_estimators=2000, learning_rate=0.03, num_leaves=31,
            min_child_samples=10, subsample=0.8, subsample_freq=1,
            colsample_bytree=0.9, random_state=SEED, verbose=-1,
        ),
        "MLP": lambda: make_pipeline(
            StandardScaler(),
            MLPRegressor(hidden_layer_sizes=(128, 128, 128), alpha=1e-4,
                         learning_rate_init=1e-3, max_iter=500,
                         early_stopping=True, n_iter_no_change=25,
                         random_state=SEED),
        ),
    }


def metrics(y, p):
    return {"MAE": mean_absolute_error(y, p), "R2": r2_score(y, p),
            "MAPE%": 100 * np.mean(np.abs((p - y) / y))}


def main():
    df = pd.read_csv(DATA)
    X = df[FEATURES].to_numpy()
    targets = {"Cl": df.Cl.to_numpy(), "logCd": np.log(df.Cd.to_numpy())}
    folds = list(GroupKFold(n_splits=5).split(X, groups=df.geom_id))

    rows = []
    for name, factory in make_models().items():
        preds = {t: np.zeros(len(df)) for t in targets}
        for tr, te in folds:
            for t, y in targets.items():
                m = factory().fit(X[tr], y[tr])
                preds[t][te] = m.predict(X[te])
        cl = metrics(df.Cl, preds["Cl"])
        cd = metrics(df.Cd, np.exp(preds["logCd"]))
        rows.append({"model": name,
                     "Cl MAE": cl["MAE"], "Cl R2": cl["R2"],
                     "Cd MAE": cd["MAE"], "Cd R2": cd["R2"], "Cd MAPE%": cd["MAPE%"]})

    # Trivial reference: per-AoA mean Cl/Cd from the training folds -- ignores shape.
    ref = {"Cl": np.zeros(len(df)), "Cd": np.zeros(len(df))}
    for tr, te in folds:
        for t in ref:
            means = df.iloc[tr].groupby("AoA")[t].mean()
            ref[t][te] = df.AoA.iloc[te].map(means).to_numpy()
    cl, cd = metrics(df.Cl, ref["Cl"]), metrics(df.Cd, ref["Cd"])
    rows.insert(0, {"model": "AoA-mean (no shape)",
                    "Cl MAE": cl["MAE"], "Cl R2": cl["R2"],
                    "Cd MAE": cd["MAE"], "Cd R2": cd["R2"], "Cd MAPE%": cd["MAPE%"]})

    res = pd.DataFrame(rows).set_index("model")
    pd.set_option("display.float_format", "{:.4f}".format)
    print(f"{len(df)} points, {df.geom_id.nunique()} geometries, 5-fold grouped CV\n")
    print(res)


if __name__ == "__main__":
    main()
