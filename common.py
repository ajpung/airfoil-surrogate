"""Shared data loading, training filters, CV folds and metrics.

Leakage rules enforced here:
  * The 'test' split (fixed in clean_data.py) is never loaded by load_dev().
  * Folds are grouped by geom_id (one CST vector; covers every Filename alias).
  * Training filters drop flagged rows from the *training* side only;
    validation/test are always scored on all rows and on subsets.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).parent
DATA = ROOT / "data" / "airfoil_cst_cfd_clean.csv"
RESULTS = ROOT / "results"
FEATURES = ["AoA"] + [f"CST Coeff {i}" for i in range(1, 9)]
N_FOLDS = 5
SEED = 0


def load(split):
    df = pd.read_csv(DATA)
    return df[df.split == split].reset_index(drop=True)


def load_dev():
    return load("train")


def xy(df):
    X = df[FEATURES].to_numpy(np.float64)
    Y = np.column_stack([df.Cl.to_numpy(), np.log(df.Cd.to_numpy())])
    return X, Y


def train_mask(df, mode):
    """Which rows of a *training* frame to keep."""
    if mode == "none":
        return np.ones(len(df), bool)
    if mode == "conflict":
        return ~df.conflict.to_numpy()
    if mode == "conflict+jump":
        return ~(df.conflict | df.cd_jump).to_numpy()
    if mode == "smooth":
        return df.smooth.fillna(False).to_numpy(bool)
    raise ValueError(mode)


def folds(df):
    gkf = GroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    return list(gkf.split(df, groups=df.geom_id))


def _r2(y, p):
    return 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)


def score(df, P):
    """P[:,0]=Cl, P[:,1]=log Cd. Scored on all rows, on unflagged rows, and on
    smooth-Cd airfoils."""
    out = {}
    subsets = {"raw": np.ones(len(df), bool),
               "unflagged": ~(df.conflict | df.cd_jump).to_numpy(),
               "smooth": df.smooth.fillna(False).to_numpy(bool)}
    for tag, m in subsets.items():
        cl, cl_p = df.Cl.to_numpy()[m], P[m, 0]
        cd, cd_p = df.Cd.to_numpy()[m], np.exp(P[m, 1])
        rel = np.abs(cd_p - cd) / cd
        out.update({
            f"{tag} Cd MdAPE%": 100 * np.median(rel),
            f"{tag} Cd MAE": np.mean(np.abs(cd_p - cd)),
            f"{tag} Cd R2": _r2(cd, cd_p),
            f"{tag} Cl MAE": np.mean(np.abs(cl_p - cl)),
            f"{tag} Cl R2": _r2(cl, cl_p),
        })
    return out
