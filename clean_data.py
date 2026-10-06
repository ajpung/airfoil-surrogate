"""Clean airfoil_cst_cfd.csv -> airfoil_cst_cfd_clean.csv.

Nothing is dropped here: suspect points are only *flagged*, so filtering can be
applied to training data alone and test metrics stay honest.

Steps:
  1. Strip/normalize airfoil names.
  2. Assign geom_id per unique CST vector (rounded to 1e-6). Several Filenames
     share one CST vector, so geom_id is the grouping key that prevents leakage
     (every Filename maps to exactly one geom_id).
  3. Collapse duplicate (geometry, AoA) rows to the median Cl/Cd, keeping the
     count and spread.
  4. Flags:
       conflict -- duplicate runs of the same geometry/AoA disagree by more than
                   CL_SPREAD_MAX or CD_SPREAD_MAX.
       cd_jump  -- log(Cd) deviates from a leave-one-out cubic fit of that
                   airfoil's own log(Cd)-vs-AoA curve by more than CD_JUMP_MAX.
                   Uses only that geometry's labels (no cross-airfoil info).
       cd_rough -- per-geometry median |LOO resid| (NaN if too few points).
       smooth   -- cd_rough <= CD_SMOOTH_MAX: this airfoil's Cd curve looks
                   converged. Used to define a trustworthy evaluation subset.
  5. Fixed held-out test split: TEST_FRAC of geometries, chosen at random.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).parent
RAW = ROOT / "data" / "airfoil_cst_cfd.csv"
OUT = ROOT / "data" / "airfoil_cst_cfd_clean.csv"

CST = [f"CST Coeff {i}" for i in range(1, 9)]
CL_SPREAD_MAX = 0.05
CD_SPREAD_MAX = 0.005
CD_JUMP_MAX = 0.5   # in log(Cd), i.e. ~65% off the airfoil's own smooth curve
CD_SMOOTH_MAX = 0.05  # per-airfoil median |LOO resid| below which the Cd curve is "smooth"
MIN_PTS_FOR_JUMP = 7
TEST_FRAC = 0.15
SEED = 42


def loo_cubic_resid(aoa, y):
    """Residual of each point vs a cubic fitted to the *other* points."""
    out = np.full(len(y), np.nan)
    if len(y) < MIN_PTS_FOR_JUMP:
        return out
    for i in range(len(y)):
        m = np.arange(len(y)) != i
        c = np.polyfit(aoa[m], y[m], 3)
        out[i] = y[i] - np.polyval(c, aoa[i])
    return out


def main():
    df = pd.read_csv(RAW)
    n_raw = len(df)
    df["Filename"] = df["Filename"].str.strip().str.split().str.join(" ")
    df["geom_id"] = df[CST].round(6).groupby(CST, sort=False).ngroup()

    agg = (
        df.groupby(["geom_id", "AoA"])
        .agg(
            names=("Filename", lambda s: ";".join(sorted(set(s)))),
            **{c: (c, "first") for c in CST},
            Cl=("Cl", "median"),
            Cd=("Cd", "median"),
            n_dup=("Cl", "size"),
            Cl_spread=("Cl", lambda s: s.max() - s.min()),
            Cd_spread=("Cd", lambda s: s.max() - s.min()),
        )
        .reset_index()
        .sort_values(["geom_id", "AoA"])
        .reset_index(drop=True)
    )

    agg["conflict"] = (agg.Cl_spread > CL_SPREAD_MAX) | (agg.Cd_spread > CD_SPREAD_MAX)
    agg["cd_resid"] = np.concatenate([
        loo_cubic_resid(g.AoA.to_numpy(float), np.log(g.Cd.to_numpy()))
        for _, g in agg.groupby("geom_id", sort=True)
    ])
    agg["cd_jump"] = agg.cd_resid.abs() > CD_JUMP_MAX
    agg["cd_rough"] = agg.groupby("geom_id").cd_resid.transform(
        lambda s: s.abs().median() if s.notna().any() else np.nan)
    agg["smooth"] = agg.cd_rough <= CD_SMOOTH_MAX

    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_FRAC, random_state=SEED)
    _, te = next(gss.split(agg, groups=agg.geom_id))
    agg["split"] = "train"
    agg.loc[te, "split"] = "test"

    agg.to_csv(OUT, index=False)

    flagged = agg.conflict | agg.cd_jump
    print(f"raw rows:                 {n_raw}")
    print(f"after merging duplicates: {len(agg)}  ({agg.geom_id.nunique()} geometries)")
    print(f"flagged conflict:         {agg.conflict.sum()}")
    print(f"flagged cd_jump:          {agg.cd_jump.sum()}")
    print(f"flagged either:           {flagged.sum()}")
    print(f"smooth-Cd geometries:     {agg[agg.smooth].geom_id.nunique()} ({agg.smooth.sum()} pts)")
    for s, g in agg.groupby("split"):
        print(f"{s:5s}: {len(g)} pts, {g.geom_id.nunique()} geometries, "
              f"{(g.conflict | g.cd_jump).sum()} flagged, {g.smooth.sum()} smooth")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
