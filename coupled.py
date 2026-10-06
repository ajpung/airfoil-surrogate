"""Compare ONLY models that couple lift and drag (dev split, same grouped folds).

Lift and drag are the two components of one aerodynamic force, so every
candidate produces Cl and Cd from a single joint computation:

  body      body-axis force net: one net -> (Cn, Ca), rotated exactly by alpha
              Cl = Cn cos a - Ca sin a,  Cd = Cn sin a + Ca cos a
  resultant resultant-force net: one net -> (|C_R|, theta), theta in (0, pi)
              Cl = C_R cos theta,  Cd = C_R sin theta   (Cd > 0 by construction)
  polar     learned drag polar: net A (shape, alpha) -> Cl;
              net B (Cl, shape) -> Cd, trained end to end (alpha enters Cd only via Cl)
  hyper     parametric drag-polar hypernet (W1): shape -> curve params,
              cubic Cl(alpha), Cd = Cd0 + k (Cl - Clm)^2 + k4 (Cl - Clm)^4
  joint     shared-trunk net -> (Cl, log Cd): statistical coupling only (reference)

All five CV folds train in parallel as one batched model on the GPU (every
weight is per-fold, so no fold sees another fold's validation data). Early
stopping uses a grouped inner split of each fold's training geometries.
The tuned decoupled SVR pair is scored once as a reference, not a candidate.
"""
import json
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as Fn
from sklearn.model_selection import GroupShuffleSplit

from common import RESULTS, folds, load_dev, score, train_mask, xy

DEV = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_num_threads(2)
STEPS, BATCH, EVAL_EVERY, LR = 20000, 1024, 250, 2e-3
MODELS = ["body", "resultant", "polar", "hyper", "joint"]
FILTERS = ["smooth", "conflict+jump"]


# --------------------------------------------------------------------------
# Batched (per-fold) layers
# --------------------------------------------------------------------------
class BLinear(nn.Module):
    def __init__(self, F, i, o, scale=1.0):
        super().__init__()
        self.w = nn.Parameter(torch.randn(F, i, o) * scale * (2.0 / i) ** 0.5)
        self.b = nn.Parameter(torch.zeros(F, 1, o))

    def forward(self, x):
        return torch.baddbmm(self.b, x, self.w)


class BResMLP(nn.Module):
    """Residual MLP, one independent copy per fold. LayerNorm without affine
    parameters so nothing is shared across folds."""

    def __init__(self, F, d_in, d_out, width=256, depth=4):
        super().__init__()
        self.inp = BLinear(F, d_in, width)
        self.blocks = nn.ModuleList(nn.ModuleList([BLinear(F, width, width),
                                                   BLinear(F, width, width, scale=0.1)])
                                    for _ in range(depth))
        self.out = BLinear(F, width, d_out, scale=0.1)
        self.width = width

    def forward(self, x):
        h = self.inp(x)
        for a, b in self.blocks:
            h = h + b(Fn.silu(a(Fn.layer_norm(h, (self.width,)))))
        return self.out(Fn.layer_norm(h, (self.width,)))


class Coupled(nn.Module):
    def __init__(self, kind, F, width=256, depth=4):
        super().__init__()
        self.kind = kind
        kw = dict(width=width, depth=depth)
        if kind == "polar":
            self.net_cl = BResMLP(F, 9, 1, **kw)
            self.net_cd = BResMLP(F, 9, 1, **kw)  # input: [Cl (standardized), 8 shape coeffs]
        elif kind == "hyper":
            self.net = BResMLP(F, 8, 8, **kw)     # shape only; alpha enters via closed forms
        elif kind == "hyper_q5":
            self.net = BResMLP(F, 8, 10, **kw)    # 6 quintic Cl(alpha) coeffs + 4 polar params
        elif kind == "hyper_tat":
            self.net = BResMLP(F, 8, 6, **kw)     # lift slope, alpha0 + 4 polar params
            self.net_res = BResMLP(F, 9, 1, **kw)  # learned Cl correction (alpha, shape)
        elif kind == "hyper_net":
            self.net = BResMLP(F, 8, 4, **kw)     # 4 polar params from shape only
            self.net_cl = BResMLP(F, 9, 1, **kw)   # flexible Cl(alpha, shape)
        else:
            self.net = BResMLP(F, 9, 2, **kw)

    def forward(self, xs, a_rad, st):
        """xs: (F,B,9) standardized [AoA, CST1..8]; a_rad: (F,B). Returns Cl, Cd (F,B)."""
        k = self.kind
        if k == "joint":
            o = self.net(xs)
            cl = o[..., 0] * st["sdCl"] + st["muCl"]
            cd = torch.exp(o[..., 1] * st["sdL"] + st["muL"])
        elif k == "body":
            o = self.net(xs)
            cn = o[..., 0] * st["sdCn"] + st["muCn"]
            ca = o[..., 1] * st["sdCa"] + st["muCa"]
            c, s = torch.cos(a_rad), torch.sin(a_rad)
            cl, cd = cn * c - ca * s, cn * s + ca * c
        elif k == "resultant":
            o = self.net(xs)
            cr = torch.exp(o[..., 0] * st["sdR"] + st["muR"])
            th = torch.pi * torch.sigmoid(o[..., 1] * st["sdZ"] + st["muZ"])
            cl, cd = cr * torch.cos(th), cr * torch.sin(th)
        elif k == "polar":
            cl_s = self.net_cl(xs)[..., 0]
            cl = cl_s * st["sdCl"] + st["muCl"]
            cd = torch.exp(self.net_cd(torch.cat([cl_s[..., None], xs[..., 1:]], -1))[..., 0]
                           * st["sdL"] + st["muL"])
        elif k == "hyper":
            p = self.net(xs[..., 1:])
            a = xs[..., 0]
            cl_s = p[..., 0] + p[..., 1] * a + 0.1 * p[..., 2] * a ** 2 + 0.1 * p[..., 3] * a ** 3
            cl = cl_s * st["sdCl"] + st["muCl"]
            d = cl - p[..., 6]
            cd = (torch.exp(p[..., 4] - 4.0) + torch.exp(p[..., 5] - 4.0) * d ** 2
                  + torch.exp(p[..., 7] - 5.0) * d ** 4)
        elif k.startswith("hyper_"):
            a = xs[..., 0]
            if k == "hyper_q5":
                p = self.net(xs[..., 1:])
                cl_s = p[..., 0] + p[..., 1] * a + 0.1 * sum(p[..., i] * a ** i for i in range(2, 6))
                q = p[..., 6:]
                cl = cl_s * st["sdCl"] + st["muCl"]
            elif k == "hyper_tat":
                p = self.net(xs[..., 1:])
                # thin-airfoil lift 2*pi*s*(alpha - alpha0), s and alpha0 (rad) per shape
                lin = 2 * torch.pi * torch.exp(0.2 * p[..., 0]) * (a_rad - 0.05 * p[..., 1])
                cl = lin + self.net_res(xs)[..., 0] * st["sdCl"]
                q = p[..., 2:]
            else:  # hyper_net
                q = self.net(xs[..., 1:])
                cl = self.net_cl(xs)[..., 0] * st["sdCl"] + st["muCl"]
            # same parametric drag polar as "hyper", evaluated at the predicted Cl
            d = cl - q[..., 2]
            cd = (torch.exp(q[..., 0] - 4.0) + torch.exp(q[..., 1] - 4.0) * d ** 2
                  + torch.exp(q[..., 3] - 5.0) * d ** 4)
        return cl, cd


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
def pad(arrs, dim):
    n = max(len(a) for a in arrs)
    out = np.zeros((len(arrs), n, dim), np.float32)
    for i, a in enumerate(arrs):
        out[i, :len(a)] = a.reshape(len(a), dim)
    return torch.tensor(out, device=DEV), torch.tensor([len(a) for a in arrs], device=DEV)


def fold_stats(Y, a_rad):
    cl, cd = Y[:, 0], np.exp(Y[:, 1])
    th = np.arctan2(cd, cl)
    z = np.log(th / np.pi) - np.log1p(-th / np.pi)
    cols = {"Cl": cl, "L": np.log(cd), "Cn": cl * np.cos(a_rad) + cd * np.sin(a_rad),
            "Ca": -cl * np.sin(a_rad) + cd * np.cos(a_rad), "R": np.log(np.hypot(cl, cd)), "Z": z}
    st = {}
    for k, v in cols.items():
        st["mu" + k], st["sd" + k] = v.mean(), v.std()
    return st


def build(df, X, Y, fl, filt):
    tr_sets, iv_sets, stats = [], [], []
    for tr, _ in fl:
        keep = tr[train_mask(df.iloc[tr], filt)]
        itr, iva = next(GroupShuffleSplit(1, test_size=0.1, random_state=0)
                        .split(keep, groups=df.geom_id.to_numpy()[keep]))
        tr_sets.append(keep[itr])
        iv_sets.append(keep[iva])
        stats.append((X[keep].mean(0), X[keep].std(0), fold_stats(Y[keep], np.radians(X[keep, 0]))))
    F = len(fl)
    mu = torch.tensor(np.stack([s[0] for s in stats]), dtype=torch.float32, device=DEV)[:, None]
    sd = torch.tensor(np.stack([s[1] for s in stats]), dtype=torch.float32, device=DEV)[:, None]
    st = {k: torch.tensor([s[2][k] for s in stats], dtype=torch.float32, device=DEV)[:, None]
          for k in stats[0][2]}
    D = {"mu": mu, "sd": sd, "st": st, "F": F}
    for name, sets in [("tr", tr_sets), ("iv", iv_sets)]:
        D[name + "X"], D[name + "n"] = pad([X[s] for s in sets], 9)
        D[name + "Y"], _ = pad([Y[s] for s in sets], 2)
    return D


def loss_fn(cl, cd, y, st):
    """Same loss for every model: standardized Cl error + Cd relative error
    scaled by the spread of log Cd. Returns per-fold mean (F,)."""
    cl_t, cd_t = y[..., 0], torch.exp(y[..., 1])
    return (((cl - cl_t) / st["sdCl"]) ** 2 + (((cd - cd_t) / cd_t) / st["sdL"]) ** 2)


DEFAULT_CFG = {"width": 256, "depth": 4, "lr": LR, "wd": 1e-5, "batch": BATCH, "steps": STEPS,
               "seed": 0}


def train(kind, D, log_prefix, cfg=None):
    c = {**DEFAULT_CFG, "steps": STEPS, **(cfg or {})}
    STEPS_, BATCH_ = c["steps"], c["batch"]
    F = D["F"]
    torch.manual_seed(c["seed"])
    model = Coupled(kind, F, c["width"], c["depth"]).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=c["lr"], weight_decay=c["wd"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, c["lr"], total_steps=STEPS_, pct_start=0.05)
    best = [p.detach().clone() for p in model.parameters()]
    best_v = torch.full((F,), float("inf"), device=DEV)
    ar = torch.arange(F, device=DEV)[:, None]
    ivmask = (torch.arange(D["ivX"].shape[1], device=DEV)[None] < D["ivn"][:, None]).float()
    xs_iv = (D["ivX"] - D["mu"]) / D["sd"]
    a_iv = torch.deg2rad(D["ivX"][..., 0])
    t0 = time.time()
    for step in range(STEPS_):
        idx = (torch.rand(F, BATCH_, device=DEV) * D["trn"][:, None]).long()
        xb, yb = D["trX"][ar, idx], D["trY"][ar, idx]
        cl, cd = model((xb - D["mu"]) / D["sd"], torch.deg2rad(xb[..., 0]), D["st"])
        loss = loss_fn(cl, cd, yb, D["st"]).mean(1).sum()  # sum over folds: independent grads
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if (step + 1) % EVAL_EVERY == 0:
            with torch.no_grad():
                cl, cd = model(xs_iv, a_iv, D["st"])
                v = (loss_fn(cl, cd, D["ivY"], D["st"]) * ivmask).sum(1) / D["ivn"]
                v = torch.nan_to_num(v, nan=float("inf"))
                imp = v < best_v
                best_v = torch.where(imp, v, best_v)
                for b, p in zip(best, model.parameters()):
                    b.copy_(torch.where(imp.view(-1, *[1] * (p.dim() - 1)), p.detach(), b))
        if log_prefix and step + 1 == STEPS_ // 2:
            print(f"{log_prefix} half-way ({time.time() - t0:.0f}s)", flush=True)
    with torch.no_grad():
        for b, p in zip(best, model.parameters()):
            p.copy_(b)
    return model, time.time() - t0


def predict(model, D, X):
    Xa = torch.tensor(X, dtype=torch.float32, device=DEV)[None].expand(D["F"], -1, -1)
    with torch.no_grad():
        cl, cd = model((Xa - D["mu"]) / D["sd"], torch.deg2rad(Xa[..., 0]), D["st"])
    return cl.cpu().numpy().astype(np.float64), cd.cpu().numpy().astype(np.float64)


# --------------------------------------------------------------------------
# Metrics (standard + coupling)
# --------------------------------------------------------------------------
def coupled_score(df, cl, cd):
    P = np.column_stack([cl, np.log(np.clip(cd, 1e-6, None))])
    out = score(df, P)
    Cl, Cd = df.Cl.to_numpy(), df.Cd.to_numpy()
    out["Cd<=0 count"] = int((cd <= 0).sum())
    th_err = np.degrees(np.abs(np.arctan2(cd, cl) - np.arctan2(Cd, Cl)))
    ld_err = np.abs(cl / np.clip(cd, 1e-6, None) - Cl / Cd)
    subsets = {"raw": np.ones(len(df), bool),
               "unflagged": ~(df.conflict | df.cd_jump).to_numpy(),
               "smooth": df.smooth.fillna(False).to_numpy(bool)}
    for tag, m in subsets.items():
        out[f"{tag} force-angle MAE deg"] = th_err[m].mean()
        out[f"{tag} force-angle MdAE deg"] = np.median(th_err[m])
        out[f"{tag} L/D MAE"] = ld_err[m].mean()
        out[f"{tag} L/D MdAE"] = np.median(ld_err[m])
    # per smooth airfoil: max L/D and the alpha where it occurs
    s = df[subsets["smooth"]].assign(ld=Cl[subsets["smooth"]] / Cd[subsets["smooth"]],
                                     ldp=(cl / np.clip(cd, 1e-6, None))[subsets["smooth"]])
    g = s.groupby("geom_id")
    mx, mxp = g.ld.max(), g.ldp.max()
    a_t = s.loc[g.ld.idxmax(), ["geom_id", "AoA"]].set_index("geom_id").AoA
    a_p = s.loc[g.ldp.idxmax(), ["geom_id", "AoA"]].set_index("geom_id").AoA
    out["smooth maxL/D MdAPE%"] = 100 * np.median(np.abs(mxp - mx) / mx)
    out["smooth alpha@maxL/D exact%"] = 100 * (a_p == a_t).mean()
    out["smooth alpha@maxL/D MAE deg"] = (a_p - a_t).abs().mean()
    return out


def svr_reference(df, X, Y, fl):
    from tune import fit_predict_folds
    cfgs = json.loads((RESULTS / "svr_tuned_r2.json").read_text())
    P = np.zeros_like(Y)
    for j, t in enumerate(["Cl", "Cd"]):
        preds, _ = fit_predict_folds(cfgs[t], X, Y, df, fl)
        for (_, va), p in zip(fl, preds):
            P[va, j] = p[:, j]
    return P[:, 0], np.exp(P[:, 1])


# --------------------------------------------------------------------------
def main():
    df = load_dev()
    X, Y = xy(df)
    fl = folds(df)
    out_dir = RESULTS / "oof_coupled"
    out_dir.mkdir(exist_ok=True)
    rows = {}
    if "--no-ref" not in sys.argv:
        t0 = time.time()
        cl, cd = svr_reference(df, X, Y, fl)
        rows["REF decoupled SVR pair (not a candidate)"] = coupled_score(df, cl, cd)
        print(f"reference SVR pair scored ({time.time() - t0:.0f}s)", flush=True)

    variants = [(m, f) for f in FILTERS for m in MODELS]
    for i, (kind, filt) in enumerate(variants):
        D = build(df, X, Y, fl, filt)
        pre = f"[{100 * (2 * i + 1) // (2 * len(variants))}%] {kind} / {filt}"
        model, secs = train(kind, D, pre)
        CL, CD = predict(model, D, X)
        cl, cd = np.zeros(len(df)), np.zeros(len(df))
        for k, (_, va) in enumerate(fl):
            cl[va], cd[va] = CL[k, va], CD[k, va]
        np.save(out_dir / f"{kind}_{filt}.npy", np.column_stack([cl, cd]))
        r = coupled_score(df, cl, cd)
        rows[f"{kind} / {filt}"] = r
        print(f"[{100 * (i + 1) // len(variants)}%] {kind} / {filt} done ({secs:.0f}s): "
              f"smooth Cd MdAPE {r['smooth Cd MdAPE%']:.3f}%, smooth Cl MAE {r['smooth Cl MAE']:.4f}, "
              f"smooth force-angle MAE {r['smooth force-angle MAE deg']:.3f} deg", flush=True)
        del model, D
        torch.cuda.empty_cache()

    res = pd.DataFrame(rows).T.sort_values("smooth Cd MdAPE%")
    res.to_csv(RESULTS / "coupled_summary.csv")
    pd.set_option("display.width", 300)
    pd.set_option("display.max_columns", 40)
    cols = ["smooth Cd MdAPE%", "smooth Cl MAE", "smooth force-angle MdAE deg", "smooth L/D MdAE",
            "smooth maxL/D MdAPE%", "smooth alpha@maxL/D exact%", "unflagged Cd MdAPE%",
            "unflagged Cl MAE", "raw Cd MdAPE%", "raw Cl MAE", "Cd<=0 count"]
    print("\n" + res[cols].to_string(float_format="{:.4f}".format))
    print(f"\nwrote {RESULTS / 'coupled_summary.csv'}")


if __name__ == "__main__":
    main()
