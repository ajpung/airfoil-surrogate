"""Tune + seed-ensemble the top three physically coupled models (dev split only).

Stage 1: random search, 8 configs per model (width, depth, lr, weight decay,
         batch, steps), smooth training filter (won for every coupled model).
Stage 2: best config per model retrained with 5 seeds; predictions averaged.
         Averaging Cl and Cd across seeds averages the force vectors, so the
         ensemble is still a physical force (Cd > 0 preserved).

Selection score (lower is better), weighting lift and drag equally:
    smooth Cd MdAPE% / 1%  +  smooth Cl MAE / 0.01
Same grouped folds as compare.py / coupled.py; numbers are CV on dev.
"""
import json
import sys
import time

import numpy as np
import pandas as pd
import torch

from common import RESULTS, folds, load_dev, xy
from coupled import build, coupled_score, predict, train

# usage: tune_coupled.py [--kinds a,b,c] [--tag name]
_args = dict(zip(sys.argv[1::2], sys.argv[2::2]))
KINDS = _args.get("--kinds", "hyper,resultant,polar").split(",")
TAG = "_" + _args["--tag"] if "--tag" in _args else ""
N_CFG, N_SEEDS, SEED = 8, 5, 7
FILTER = "smooth"


def sample(rng):
    return {"width": int(rng.choice([128, 256, 512])),
            "depth": int(rng.choice([2, 3, 4, 6])),
            "lr": float(np.exp(rng.uniform(np.log(3e-4), np.log(5e-3)))),
            "wd": float(np.exp(rng.uniform(np.log(1e-6), np.log(1e-3)))),
            "batch": int(rng.choice([512, 1024, 2048])),
            "steps": int(rng.choice([10000, 20000, 30000]))}


def objective(r):
    return r["smooth Cd MdAPE%"] / 1.0 + r["smooth Cl MAE"] / 0.01


def run(kind, cfg, D, df, X, fl):
    model, secs = train(kind, D, None, cfg)
    CL, CD = predict(model, D, X)
    cl, cd = np.zeros(len(df)), np.zeros(len(df))
    for k, (_, va) in enumerate(fl):
        cl[va], cd[va] = CL[k, va], CD[k, va]
    del model
    torch.cuda.empty_cache()
    return cl, cd, secs


def main():
    df = load_dev()
    X, Y = xy(df)
    fl = folds(df)
    D = build(df, X, Y, fl, FILTER)
    rng = np.random.default_rng(SEED)
    default = {"width": 256, "depth": 4, "lr": 2e-3, "wd": 1e-5, "batch": 1024, "steps": 20000}
    cfgs = {k: [default] + [sample(rng) for _ in range(N_CFG - 1)] for k in KINDS}
    total = len(KINDS) * (N_CFG + N_SEEDS - 1)
    done, t0 = 0, time.time()
    out_dir = RESULTS / "oof_coupled"
    rows, best = [], {}

    def tick(msg):
        top = min(best.items(), key=lambda kv: kv[1]["objective"])
        print(f"[{100 * done / total:.1f}%] {msg} | overall best: {top[0]} cfg {top[1]['cfg']} "
              f"score {top[1]['objective']:.3f} | elapsed {(time.time() - t0) / 60:.1f} min",
              flush=True)

    # stage 1: random search
    for kind in KINDS:
        for i, cfg in enumerate(cfgs[kind]):
            cl, cd, secs = run(kind, {**cfg, "seed": 0}, D, df, X, fl)
            r = coupled_score(df, cl, cd)
            obj = objective(r)
            rows.append({"kind": kind, "cfg": i, **cfg, "objective": obj, **r})
            if kind not in best or obj < best[kind]["objective"]:
                best[kind] = {"cfg": i, "objective": obj, "pred": (cl, cd)}
            done += 1
            tick(f"tune {kind} cfg {i} ({secs:.0f}s): Cd {r['smooth Cd MdAPE%']:.3f}%, "
                 f"Cl MAE {r['smooth Cl MAE']:.4f}, score {obj:.3f} | best {kind}: "
                 f"cfg {best[kind]['cfg']} score {best[kind]['objective']:.3f}")
    pd.DataFrame(rows).to_csv(RESULTS / f"tune_coupled{TAG}.csv", index=False)

    # stage 2: seed ensembles of each model's best config (seed 0 reused)
    summary, chosen = {}, {}
    for kind in KINDS:
        cfg = cfgs[kind][best[kind]["cfg"]]
        chosen[kind] = cfg
        cls, cds = [best[kind]["pred"][0]], [best[kind]["pred"][1]]
        for seed in range(1, N_SEEDS):
            cl, cd, secs = run(kind, {**cfg, "seed": seed}, D, df, X, fl)
            cls.append(cl)
            cds.append(cd)
            done += 1
            r = coupled_score(df, np.mean(cls, 0), np.mean(cds, 0))
            tick(f"ensemble {kind} {len(cls)}/{N_SEEDS} seeds ({secs:.0f}s): "
                 f"Cd {r['smooth Cd MdAPE%']:.3f}%, Cl MAE {r['smooth Cl MAE']:.4f}")
        cl, cd = np.mean(cls, 0), np.mean(cds, 0)
        np.save(out_dir / f"{kind}{TAG}_tuned_ens.npy", np.column_stack([cl, cd]))
        summary[f"{kind} tuned, 1 seed"] = coupled_score(df, cls[0], cds[0])
        summary[f"{kind} tuned, {N_SEEDS}-seed ensemble"] = coupled_score(df, cl, cd)

    (RESULTS / f"coupled_tuned{TAG}.json").write_text(json.dumps(chosen, indent=2))
    prev = pd.read_csv(RESULTS / "coupled_tuned_summary.csv", index_col=0)
    keep = [i for i in prev.index if i.startswith("REF") or i.endswith("seed ensemble")]
    res = pd.concat([pd.DataFrame(summary).T, prev.loc[keep]])
    res = res.sort_values("smooth Cd MdAPE%")
    res.to_csv(RESULTS / f"coupled_tuned_summary{TAG}.csv")
    pd.set_option("display.width", 300)
    pd.set_option("display.max_columns", 40)
    cols = ["smooth Cd MdAPE%", "smooth Cd R2", "smooth Cl MAE", "smooth force-angle MdAE deg",
            "smooth L/D MdAE", "smooth maxL/D MdAPE%", "smooth alpha@maxL/D exact%",
            "unflagged Cd MdAPE%", "raw Cd MdAPE%", "Cd<=0 count"]
    print("\nchosen configs: " + json.dumps(chosen))
    print("\n" + res[cols].to_string(float_format="{:.4f}".format))
    print(f"\nwrote {RESULTS / f'coupled_tuned_summary{TAG}.csv'}")


if __name__ == "__main__":
    main()
