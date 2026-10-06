"""Compare the model zoo with 5-fold grouped CV on the dev split only.

Usage: python compare.py [name-substring ...]   (default: all models)
Out-of-fold predictions are cached in results/oof/ so models can be re-run
individually; the summary table is rebuilt from whatever is cached.
"""
import re
import sys
import time
import warnings

import numpy as np
import pandas as pd

from common import RESULTS, folds, load_dev, score, train_mask, xy
from zoo import AvgEnsemble, registry

warnings.filterwarnings("ignore")
TRAIN_FILTER = "conflict+jump"
OOF = RESULTS / "oof"


def slug(name):
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")


def run_model(name, factory, df, X, Y, fl):
    path = OOF / f"{slug(name)}.npy"
    if path.exists():
        return np.load(path)
    proto = factory()
    if isinstance(proto, AvgEnsemble):
        return None  # built after members
    P = np.zeros_like(Y)
    t0 = time.time()
    for k, (tr, va) in enumerate(fl):
        keep = tr[train_mask(df.iloc[tr], TRAIN_FILTER)]
        m = factory().fit(X[keep], Y[keep], df.geom_id.to_numpy()[keep])
        P[va] = m.predict(X[va])
        print(f"  {name} fold {k}: {time.time() - t0:.0f}s", flush=True)
    np.save(path, P)
    return P


def main():
    OOF.mkdir(parents=True, exist_ok=True)
    df = load_dev()
    X, Y = xy(df)
    fl = folds(df)
    reg = registry()
    pick = sys.argv[1:]
    names = [n for n in reg if not pick or any(p.lower() in n.lower() for p in pick)]

    for name in names:
        print(f"== {name}", flush=True)
        try:
            run_model(name, reg[name], df, X, Y, fl)
        except Exception as e:  # keep the sweep going; report failures
            print(f"  FAILED: {e!r}", flush=True)

    rows = []
    for name, factory in reg.items():
        proto = factory()
        if isinstance(proto, AvgEnsemble):
            parts = [OOF / f"{slug(m)}.npy" for m in proto.members]
            if not all(p.exists() for p in parts):
                continue
            P = np.mean([np.load(p) for p in parts], axis=0)
        else:
            path = OOF / f"{slug(name)}.npy"
            if not path.exists():
                continue
            P = np.load(path)
        rows.append({"model": name, **score(df, P)})

    res = pd.DataFrame(rows).set_index("model").sort_values("smooth Cd MdAPE%")
    res.to_csv(RESULTS / "cv_summary.csv")
    cols = ["smooth Cd MdAPE%", "smooth Cd MAE", "smooth Cd R2", "raw Cd MdAPE%",
            "raw Cd MAE", "raw Cd R2", "smooth Cl MAE", "raw Cl MAE", "raw Cl R2"]
    pd.set_option("display.width", 250)
    pd.set_option("display.float_format", "{:.4f}".format)
    print(f"\nDev: {len(df)} pts, {df.geom_id.nunique()} geometries; "
          f"train filter '{TRAIN_FILTER}'; ranked by smooth Cd MdAPE%\n")
    print(res[cols])


if __name__ == "__main__":
    main()
