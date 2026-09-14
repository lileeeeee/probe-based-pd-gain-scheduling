"""Bootstrap CIs for the quantile-bin estimator \\Vbin (R2-C3 / R2-C8).

Resolves the CI offset seen in the first bootstrap pass: with within-bin
variance computed at ddof=0, resampling with replacement biases each bin's
variance down by the factor (1 - 1/n_b), so the resample mean sits ~1/n_bar
below the point estimate and the point estimate lands on the CI's upper edge
(n_bar = N / 5^F: -5.2% for the full three-factor list, -1% at F=2, -0.2%
at F=1). Using ddof=1 inside the resampled statistic removes that first-order
bias; the point estimate itself stays at ddof=0, identical to Table 3.

Outputs percentile CIs from the debiased statistic, plus the analytic
prediction check, to output/r2c8_vbin_bootstrap_ci.json.
"""
import sys, json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from itertools import combinations

ROOT = PROJECT_ROOT / "output/f3_casadi_iid_valfrac"
DT = 0.04
N_BINS = 5
B = 500
SEED = 0
FACTOR_NAMES = ["m_p", "f_v", "f_c"]


def transform_to_pd(g):
    nJ = g.shape[1] // 3
    return np.concatenate([g[:, :nJ] + DT * g[:, nJ:2*nJ], g[:, 2*nJ:]], axis=1)


def vhid(phys, gains, factor_idxs, n_bins=N_BINS, ddof=0):
    if not factor_idxs:
        return float(gains.var(axis=0, ddof=ddof).mean())
    bin_idx = np.zeros(len(phys), dtype=int)
    for f in factor_idxs:
        col = phys[:, f]
        edges = np.quantile(col, np.linspace(0, 1, n_bins + 1))
        edges[-1] += 1e-6
        bin_idx = bin_idx * n_bins + np.digitize(col, edges[1:-1])
    total = 0.0; total_n = 0
    for b in np.unique(bin_idx):
        mask = bin_idx == b
        if mask.sum() < max(2, ddof + 1):
            continue
        total += gains[mask].var(axis=0, ddof=ddof).mean() * mask.sum()
        total_n += mask.sum()
    return total / total_n if total_n else float("nan")


def main():
    p = np.load(ROOT / "seed42/paramid_3f_oracle/predictions.npz", allow_pickle=True)
    phys, g = p["phys_train"], transform_to_pd(p["true_gains_train"])
    N = len(phys)
    rng = np.random.default_rng(SEED)
    idx_boot = rng.integers(0, N, size=(B, N))

    out = {"B": B, "N": int(N), "n_bins": N_BINS, "lists": {}}
    print(f"N={N}, B={B}, bins/factor={N_BINS}")
    print(f"  {'list':<16} {'point':>7} {'nbar':>6} {'pred bias':>9} "
          f"{'obs bias':>9} {'debiased CI (2.5-97.5%)':>24} {'in CI':>6}")
    for L_size in range(1, 4):
        for L in combinations(range(3), L_size):
            key = "_".join(FACTOR_NAMES[i].replace("_", "") for i in L)
            point = vhid(phys, g, L, ddof=0)
            nbar = N / N_BINS ** len(L)
            # biased (ddof=0) and debiased (ddof=1) resample distributions
            v0 = np.array([vhid(phys[i], g[i], L, ddof=0) for i in idx_boot])
            v1 = np.array([vhid(phys[i], g[i], L, ddof=1) for i in idx_boot])
            lo, hi = np.percentile(v1, [2.5, 97.5])
            out["lists"][key] = {
                "point_ddof0": round(point, 3),
                "resample_mean_ddof0": round(float(v0.mean()), 3),
                "pred_bias_frac": round(-1.0 / nbar, 4),
                "obs_bias_frac": round(float(v0.mean() / point - 1), 4),
                "ci_debiased": [round(float(lo), 3), round(float(hi), 3)],
                "resample_mean_debiased": round(float(v1.mean()), 3),
            }
            r = out["lists"][key]
            print(f"  {key:<16} {point:>7.2f} {nbar:>6.0f} {r['pred_bias_frac']:>8.1%} "
                  f"{r['obs_bias_frac']:>8.1%} "
                  f"{'[' + str(r['ci_debiased'][0]) + ', ' + str(r['ci_debiased'][1]) + ']':>24} "
                  f"{'yes' if lo <= point <= hi else 'NO':>6}")

    dst = PROJECT_ROOT / "output/r2c8_vbin_bootstrap_ci.json"
    json.dump(out, open(dst, "w"), indent=2)
    print(f"\nSaved {dst}")


def eq3_refit():
    """Eq (3) slope/intercept/r CIs with the debiased resampled \\Vbin."""
    p = np.load(ROOT / "seed42/paramid_3f_oracle/predictions.npz", allow_pickle=True)
    phys, g = p["phys_train"], transform_to_pd(p["true_gains_train"])
    N = len(phys)
    rng = np.random.default_rng(SEED)
    idx_boot = rng.integers(0, N, size=(B, N))
    LISTS = [c for s in range(1, 4) for c in combinations(range(3), s)]
    # measured MSE per list, fixed (property of the trained models)
    from experiments.vhid_factorlist_pd import measured_mse, LIST_TO_BASELINE
    mse = np.array([measured_mse(LIST_TO_BASELINE[L]) for L in LISTS])
    stats = []
    for i in idx_boot:
        v = np.array([vhid(phys[i], g[i], L, ddof=1) for L in LISTS])
        sl, ic = np.polyfit(v, mse, 1)
        stats.append((sl, ic, np.corrcoef(v, mse)[0, 1]))
    stats = np.array(stats)
    v0 = np.array([vhid(phys, g, L, ddof=0) for L in LISTS])
    sl0, ic0 = np.polyfit(v0, mse, 1)
    r0 = np.corrcoef(v0, mse)[0, 1]
    print("\nEq(3) refit (debiased bootstrap):")
    names = ["slope", "intercept", "r"]
    point = [sl0, ic0, r0]
    res = {}
    for j, nm in enumerate(names):
        lo, hi = np.percentile(stats[:, j], [2.5, 97.5])
        res[nm] = {"point": round(float(point[j]), 4),
                   "ci": [round(float(lo), 4), round(float(hi), 4)]}
        print(f"  {nm:<10} point={point[j]:8.4f}   CI=[{lo:8.4f}, {hi:8.4f}]")
    dst = PROJECT_ROOT / "output/r2c8_eq3_bootstrap_debiased.json"
    json.dump(res, open(dst, "w"), indent=2)
    print(f"Saved {dst}")


if __name__ == "__main__":
    if "--eq3" in sys.argv:
        eq3_refit()
    else:
        main()
