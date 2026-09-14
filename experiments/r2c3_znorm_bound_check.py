"""Bound check for \\Vbin vs measured MSE, physical vs z-normalized (R2-C3).

In physical gain units several lists show measured MSE below \\Vbin, which
reads as a violated lower bound. The aggregation weights dimensions by
absolute gain variance, so a few high-variance dimensions with large binning
bias dominate. Normalizing each gain dimension by its training-set std before
both quantities are computed removes that weighting; the bound should then
hold on every list if the apparent violations are binning bias, not broken
theory. Sweeps n_bins to show robustness. Writes
output/r2c3_znorm_bound_check.json.
"""
import sys, json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from itertools import combinations
from experiments.r2c8_vbin_bootstrap_ci import vhid, transform_to_pd, ROOT, FACTOR_NAMES
from experiments.vhid_factorlist_pd import LIST_TO_BASELINE, SEEDS

def measured_mse_z(baseline, mu, sd):
    vals = []
    for s in SEEDS:
        p = ROOT / f"seed{s}/{baseline}/predictions.npz"
        if not p.exists(): continue
        d = np.load(p, allow_pickle=True)
        t = (transform_to_pd(d["true_gains_test"]) - mu) / sd
        q = (transform_to_pd(d["pred_gains_test"]) - mu) / sd
        vals.append(float(np.mean((q - t) ** 2)))
    return float(np.mean(vals))

def main():
    p = np.load(ROOT / "seed42/paramid_3f_oracle/predictions.npz", allow_pickle=True)
    phys = p["phys_train"]
    g_pd = transform_to_pd(p["true_gains_train"])
    mu, sd = g_pd.mean(0, keepdims=True), g_pd.std(0, keepdims=True) + 1e-12
    g_z = (g_pd - mu) / sd
    LISTS = [c for s in range(1, 4) for c in combinations(range(3), s)]

    out = {"n_bins_sweep": {}, "lists": {}}
    print(f"  {'list':<12} {'V̂(phys)':>9} {'MSE(phys)':>10} {'ok?':>4}   "
          f"{'V̂(z)':>7} {'MSE(z)':>7} {'ok?':>4}")
    for L in LISTS:
        key = "_".join(FACTOR_NAMES[i].replace("_","") for i in L)
        vp = vhid(phys, g_pd, L, ddof=0)
        vz = vhid(phys, g_z, L, ddof=0)
        mp_ = __import__("experiments.vhid_factorlist_pd", fromlist=["measured_mse"]).measured_mse(LIST_TO_BASELINE[L])
        mz = measured_mse_z(LIST_TO_BASELINE[L], mu, sd)
        out["lists"][key] = {"v_phys": round(vp,2), "mse_phys": round(float(mp_),2),
                             "ok_phys": bool(mp_ >= vp),
                             "v_z": round(vz,4), "mse_z": round(mz,4),
                             "ok_z": bool(mz >= vz)}
        r = out["lists"][key]
        print(f"  {key:<12} {vp:>9.1f} {mp_:>10.1f} {'✓' if r['ok_phys'] else '✗':>4}   "
              f"{vz:>7.3f} {mz:>7.3f} {'✓' if r['ok_z'] else '✗':>4}")

    for nb in [3, 5, 8, 12, 20]:
        ok = sum(measured_mse_z(LIST_TO_BASELINE[L], mu, sd) >= vhid(phys, g_z, L, n_bins=nb, ddof=0)
                 for L in LISTS)
        out["n_bins_sweep"][nb] = f"{ok}/7"
        print(f"  n_bins={nb:>2}: z-space bound holds on {ok}/7 lists")

    dst = PROJECT_ROOT / "output/r2c3_znorm_bound_check.json"
    json.dump(out, open(dst, "w"), indent=2)
    print(f"Saved {dst}")

if __name__ == "__main__":
    main()
