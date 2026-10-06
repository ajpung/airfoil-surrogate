"""Rescore the top tuning configs on every validation subset (raw / unflagged /
smooth), not just smooth airfoils, before picking the final model. Dev only."""
import numpy as np
import pandas as pd

from common import RESULTS, folds, load_dev, score, xy
from tune import BASELINE, N_TRIALS, SEED, fit_predict_folds, sample

TOP = [19, 12, 25, 16, 29, 0]


def main():
    df = load_dev()
    X, Y = xy(df)
    fl = folds(df)
    rng = np.random.default_rng(SEED)
    cfgs = [BASELINE] + [sample(rng) for _ in range(N_TRIALS)]
    rows = {}
    for i, c in enumerate(TOP):
        preds, _ = fit_predict_folds(cfgs[c], X, Y, df, fl)
        P = np.zeros_like(Y)
        for (_, va), p in zip(fl, preds):
            P[va] = p
        rows[f"cfg {c} ({cfgs[c]['filter']})"] = score(df, P)
        print(f"{100 * (i + 1) // len(TOP)}% done: cfg {c}", flush=True)
    r = pd.DataFrame(rows).T
    r.to_csv(RESULTS / "rescore_top.csv")
    pd.set_option("display.width", 250)
    for sub in ["raw", "unflagged", "smooth"]:
        cols = [f"{sub} {m}" for m in ["Cd MdAPE%", "Cd R2", "Cl MAE", "Cl R2"]]
        print(f"\n-- {sub}\n" + r[cols].to_string(float_format="{:.4f}".format))


if __name__ == "__main__":
    main()
