"""Where are the final model's largest Cd errors on the test set?

Diagnosis only: the final model is fixed, nothing is retuned on these numbers.
Compares the tuned SVR against the untuned SVR and the LightGBM baseline,
broken down by AoA, by true-Cd level, by label flag, and by how far each test
shape is from the shapes the Cd model was trained on.
"""
import joblib
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from common import RESULTS, load, load_dev, train_mask, xy
from final_test import fit_registry

CST = [f"CST Coeff {i}" for i in range(1, 9)]


def table(df, by, models):
    rows = []
    for key, g in df.groupby(by, observed=True):
        r = {by: key, "n": len(g)}
        for m in models:
            ape = g[f"ape_{m}"]
            r[f"{m} MdAPE%"] = ape.median()
            r[f"{m} P95 APE%"] = ape.quantile(0.95)
            r[f"{m} SSE share%"] = 100 * g[f"se_{m}"].sum() / df[f"se_{m}"].sum()
        rows.append(r)
    return pd.DataFrame(rows).set_index(by)


def main():
    dev, test = load_dev(), load("test")
    Xt, _ = xy(test)
    final = joblib.load(RESULTS / "svr_final.joblib")
    preds = {"tuned": final.predict(Xt)[:, 1],
             "untuned": fit_registry("S03 SVR (RBF)", dev).predict(Xt)[:, 1],
             "lgbm": fit_registry("S09 LightGBM", dev).predict(Xt)[:, 1]}
    models = list(preds)

    d = test[["geom_id", "AoA", "Cd", "conflict", "cd_jump", "smooth"]].copy()
    d["smooth"] = d.smooth.fillna(False)
    d["label"] = np.where(d.conflict | d.cd_jump, "flagged",
                          np.where(d.smooth, "smooth", "unflagged, not smooth"))
    for m, p in preds.items():
        cd = np.exp(p)
        d[f"pred_{m}"] = cd
        d[f"ape_{m}"] = 100 * np.abs(cd - d.Cd) / d.Cd
        d[f"se_{m}"] = (cd - d.Cd) ** 2

    # distance from each test shape to the nearest shape the Cd model trained on
    cfg = final.cfgs["Cd"]
    tr_shapes = dev.loc[train_mask(dev, cfg["filter"]), CST].drop_duplicates().to_numpy()
    sc = StandardScaler().fit(tr_shapes)
    nn = NearestNeighbors(n_neighbors=1).fit(sc.transform(tr_shapes))
    dist = nn.kneighbors(sc.transform(test[CST].to_numpy()))[0][:, 0]
    d["shape_dist"] = dist
    d["dist_bin"] = pd.qcut(dist, 4, labels=["Q1 nearest", "Q2", "Q3", "Q4 farthest"])
    d["cd_bin"] = pd.qcut(d.Cd, 5, labels=["Cd Q1 lowest", "Q2", "Q3", "Q4", "Q5 highest"])
    d.to_csv(RESULTS / "test_cd_errors.csv", index=False)

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    f = "{:.2f}".format
    print(f"Cd model trained on {len(tr_shapes)} shapes (filter '{cfg['filter']}'); "
          f"test: {len(d)} pts, {d.geom_id.nunique()} shapes\n")
    print("MdAPE% = median abs % error; P95 = 95th-percentile abs % error;\n"
          "SSE share% = share of that model's total squared Cd error (what drives R2)\n")
    for by in ["AoA", "cd_bin", "label", "dist_bin"]:
        print(f"== by {by}")
        print(table(d, by, models).to_string(float_format=f), "\n")

    worst = d.nlargest(15, "se_tuned")
    print("== 15 largest tuned-SVR squared Cd errors")
    print(worst[["geom_id", "AoA", "Cd", "pred_tuned", "pred_untuned", "pred_lgbm",
                 "ape_tuned", "label", "shape_dist"]].to_string(index=False, float_format="{:.4f}".format))
    top = d.nlargest(int(0.05 * len(d)), "se_tuned")
    print(f"\nworst 5% of points ({len(top)}) hold "
          f"{100 * top.se_tuned.sum() / d.se_tuned.sum():.1f}% of tuned-SVR squared Cd error; "
          f"{top.geom_id.nunique()} shapes; labels: {top.label.value_counts().to_dict()}")


if __name__ == "__main__":
    main()
