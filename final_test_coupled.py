"""Held-out test of the tuned parametric drag-polar hypernetwork (5-seed ensemble).

Trains on the full dev split (smooth filter, tuned config from
results/coupled_tuned.json), predicts the test geometries once, and scores with
the same standard + coupling metrics as coupled.py. The final decoupled SVR pair
(results/svr_final.joblib) is scored alongside as a reference.

The 5 ensemble members train in parallel as the batch dimension of one model:
same data, independent random initialisation and batch order. Their Cl and Cd
are averaged (an average of force vectors, so still a physical force).
"""
import json

import joblib
import numpy as np
import pandas as pd
import torch

from common import RESULTS, load, load_dev, xy
from coupled import build, coupled_score, predict, train

KIND, N_SEEDS = "hyper", 5


def main():
    dev, test = load_dev(), load("test")
    X, Y = xy(dev)
    Xt, _ = xy(test)
    cfg = json.loads((RESULTS / "coupled_tuned.json").read_text())[KIND]
    print(f"config: {cfg}")

    everything = np.arange(len(dev))
    D = build(dev, X, Y, [(everything, everything[:0])] * N_SEEDS, "smooth")
    model, secs = train(KIND, D, "training", {**cfg, "seed": 0})
    CL, CD = predict(model, D, Xt)
    torch.save({"state_dict": model.state_dict(), "cfg": cfg, "kind": KIND,
                "mu": D["mu"].cpu(), "sd": D["sd"].cpu(),
                "st": {k: v.cpu() for k, v in D["st"].items()}},
               RESULTS / "hyper_final.pt")
    print(f"trained {N_SEEDS} members on {int(D['trn'][0])} pts in {secs:.0f}s; "
          f"saved {RESULTS / 'hyper_final.pt'}")

    rows = {"Drag-polar hypernetwork (5-seed ens.)": coupled_score(test, CL.mean(0), CD.mean(0))}
    singles = [coupled_score(test, CL[i], CD[i]) for i in range(N_SEEDS)]
    rows["  single member (mean of 5)"] = {k: np.mean([s[k] for s in singles]) for k in singles[0]}
    svr = joblib.load(RESULTS / "svr_final.joblib").predict(Xt)
    rows["REF decoupled SVR pair"] = coupled_score(test, svr[:, 0], np.exp(svr[:, 1]))

    res = pd.DataFrame(rows).T
    res.to_csv(RESULTS / "test_metrics_hyper.csv")
    np.save(RESULTS / "test_pred_hyper.npy", np.column_stack([CL.mean(0), CD.mean(0)]))
    pd.set_option("display.width", 300)
    pd.set_option("display.max_columns", 40)
    print(f"\nHeld-out test: {len(test)} pts, {test.geom_id.nunique()} geometries")
    for sub in ["smooth", "unflagged", "raw"]:
        cols = [f"{sub} {m}" for m in ["Cd MdAPE%", "Cd MAE", "Cd R2", "Cl MAE", "Cl R2",
                                       "force-angle MdAE deg", "L/D MdAE"]]
        if sub == "smooth":
            cols += ["smooth maxL/D MdAPE%", "smooth alpha@maxL/D exact%"]
        print(f"\n-- {sub}\n" + res[cols].to_string(float_format="{:.4f}".format))
    print(f"\nCd<=0 predictions: {res['Cd<=0 count'].to_dict()}")


if __name__ == "__main__":
    main()
