"""Figure: a typical 'smooth' airfoil vs a typical 'non-smooth' one.

Smooth = per-airfoil median |leave-one-out cubic residual| of log(Cd) vs AoA
<= CD_SMOOTH_MAX (see clean_data.py). Examples are the 13-point airfoils whose
roughness is closest to the median of each class.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from clean_data import CD_SMOOTH_MAX, OUT
from common import RESULTS

df = pd.read_csv(OUT)
g = df.groupby("geom_id").agg(rough=("cd_rough", "first"), n=("AoA", "size"),
                              smooth=("smooth", "first"))
full = g[g.n == 13]
pick = {}
for lab, is_s in [("Smooth", True), ("Non-smooth", False)]:
    s = full[full.smooth == is_s].rough
    pick[lab] = (s - g[g.smooth == is_s].rough.median()).abs().idxmin()

fig = plt.figure(figsize=(12, 11))
gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.8], hspace=0.38, wspace=0.22)
for col, (lab, gid) in enumerate(pick.items()):
    a = df[df.geom_id == gid].sort_values("AoA")
    aoa, cd, cl = a.AoA.to_numpy(float), a.Cd.to_numpy(), a.Cl.to_numpy()
    jump = a.cd_jump.to_numpy(bool)
    x = np.linspace(aoa.min(), aoa.max(), 200)
    fit = np.exp(np.polyval(np.polyfit(aoa, np.log(cd), 3), x))
    color = "tab:green" if lab == "Smooth" else "tab:red"

    ax = fig.add_subplot(gs[0, col])
    ax.plot(x, fit, "--", color="gray", lw=1.2, label="cubic fit of log Cd")
    ax.plot(aoa, cd, "o-", color=color, ms=6, label="CFD Cd")
    if jump.any():
        ax.plot(aoa[jump], cd[jump], "kx", ms=12, mew=2, label="flagged Cd jump")
    ax.set_yscale("log")
    ax.set_title(f"{lab} airfoil (geom {gid})\nmedian |LOO resid| = "
                 f"{a.cd_rough.iloc[0]:.3f} log-Cd "
                 f"(threshold {CD_SMOOTH_MAX})", fontsize=11)
    ax.set_xlabel("AoA (deg)")
    ax.set_ylabel("Cd (log scale)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=9)

    ax = fig.add_subplot(gs[1, col])
    ax.plot(aoa, cl, "o-", color=color, ms=6)
    ax.set_xlabel("AoA (deg)")
    ax.set_ylabel("Cl")
    ax.set_title(f"{lab}: lift curve", fontsize=11)
    ax.grid(alpha=0.3)

ax = fig.add_subplot(gs[2, :])
r = g.rough.dropna()
bins = np.logspace(np.log10(r.min()), np.log10(r.max()), 60)
ax.hist(r[r <= CD_SMOOTH_MAX], bins=bins, color="tab:green", alpha=0.8,
        label=f"smooth ({(r <= CD_SMOOTH_MAX).sum()} airfoils)")
ax.hist(r[r > CD_SMOOTH_MAX], bins=bins, color="tab:red", alpha=0.8,
        label=f"non-smooth ({(r > CD_SMOOTH_MAX).sum()} airfoils)")
ax.axvline(CD_SMOOTH_MAX, color="k", ls="--", lw=1.2, label=f"threshold {CD_SMOOTH_MAX}")
for lab, gid in pick.items():
    ax.annotate(lab, (g.rough[gid], 0), xytext=(0, 40), textcoords="offset points",
                ha="center", arrowprops=dict(arrowstyle="->"))
ax.set_xscale("log")
ax.set_xlabel("per-airfoil median |leave-one-out cubic residual| of log Cd")
ax.set_ylabel("airfoils")
ax.set_title("Roughness of every airfoil's Cd curve", fontsize=11)
ax.legend(fontsize=9)

out = RESULTS / "smooth_vs_nonsmooth.png"
fig.savefig(out, dpi=120, bbox_inches="tight")
print(f"wrote {out}; picks {pick}")
