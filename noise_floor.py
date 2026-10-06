"""Estimate solver noise from geometries that were run more than once.

The same CST vector appears under several Filenames (separate CFD runs). For
each (geometry, AoA) with >=2 runs, the run-to-run difference measures label
noise. If y_i = f + e_i with independent e, |e1 - e2| / sqrt(2) estimates |e|,
i.e. the error a *perfect* model would still show against a single-run label.

Caveat: identical CST does not prove identical CFD geometry if the meshes were
built from the original coordinate files rather than the CST reconstruction.
"""
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
CST = [f"CST Coeff {i}" for i in range(1, 9)]


def main():
    raw = pd.read_csv(ROOT / "data" / "airfoil_cst_cfd.csv")
    raw["geom_id"] = raw[CST].round(6).groupby(CST, sort=False).ngroup()
    clean = pd.read_csv(ROOT / "data" / "airfoil_cst_cfd_clean.csv")
    smooth = clean.groupby("geom_id").smooth.first()

    rows = []
    for (g, a), s in raw.groupby(["geom_id", "AoA"]):
        if len(s) < 2:
            continue
        for (cl1, cd1), (cl2, cd2) in combinations(s[["Cl", "Cd"]].to_numpy(), 2):
            rows.append({"geom_id": g, "dCl": abs(cl1 - cl2), "dCd": abs(cd1 - cd2),
                         "rCd": abs(np.log(cd1 / cd2))})
    p = pd.DataFrame(rows)
    p["smooth"] = p.geom_id.map(smooth).fillna(False).astype(bool)

    r2 = np.sqrt(2)
    print(f"{len(p)} run pairs from {p.geom_id.nunique()} repeated geometries\n")
    print("Implied per-run label noise (perfect-model error vs a single run):")
    for name, s in [("all repeats", p), ("smooth-Cd airfoils", p[p.smooth]),
                    ("rough-Cd airfoils", p[~p.smooth])]:
        print(f"  {name:20s} n={len(s):5d}  "
              f"Cl MAE~{s.dCl.mean() / r2:.4f}  "
              f"Cd MAE~{s.dCd.mean() / r2:.5f}  "
              f"Cd median rel err~{100 * (np.exp(s.rCd.median() / r2) - 1):.1f}%  "
              f"identical-pair frac={np.mean(s.dCd == 0):.2f}")


if __name__ == "__main__":
    main()
