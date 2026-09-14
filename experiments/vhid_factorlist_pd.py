"""V_hidden(L) factor-list selection diagnostic on PD-effective gains.

For each candidate L ⊆ {m_p, f_v, f_c}:
  V_hid(L) = E[Var(g* | L)]
  computed on training set by binning factors into quantiles and
  averaging within-bin gain variance, weighted by bin counts.

Compare V_hid(L) (computed before training) to actual paramid model
test MSE on PD-effective gains (post-processed from existing 12-dim
predictions).

If r ≈ 1, slope ≈ 1: V_hid is a working a priori diagnostic.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from itertools import combinations

ROOT = PROJECT_ROOT / "output/f3_casadi_iid_valfrac"
SEEDS = [42, 7, 123, 2025, 2026]
DT = 0.04
N_BINS = 5  # quantile bins per factor for V_hid estimate

# Map candidate list (factor index tuple) → trained baseline name
LIST_TO_BASELINE = {
    (0,):       "paramid_1f_mp",
    (1,):       "paramid_1f_fv",
    (2,):       "paramid_1f_fc",
    (0, 1):     "paramid_2f_mp_fv",
    (0, 2):     "paramid_2f_mp_fc",
    (1, 2):     "paramid_2f_fv_fc",
    (0, 1, 2):  "paramid_3f_oracle",  # full oracle
}
FACTOR_NAMES = ["m_p", "f_v", "f_c"]


def transform_to_pd(g):
    nJ = g.shape[1] // 3
    return np.concatenate([g[:, :nJ] + DT * g[:, nJ:2*nJ], g[:, 2*nJ:]], axis=1)


def vhid_for_list(phys, gains, factor_idxs, n_bins=N_BINS):
    """E[Var(g | L)] estimated by quantile bins on the factor list."""
    if not factor_idxs:
        # No conditioning: V_hid = total var
        return float(gains.var(axis=0).mean())

    # Multi-dim quantile binning
    bin_idx = np.zeros(len(phys), dtype=int)
    for f in factor_idxs:
        col = phys[:, f]
        edges = np.quantile(col, np.linspace(0, 1, n_bins + 1))
        edges[-1] += 1e-6
        b = np.digitize(col, edges[1:-1])
        bin_idx = bin_idx * n_bins + b

    # E[Var(g | L)] over bins
    total = 0.0; total_n = 0
    for b in np.unique(bin_idx):
        mask = bin_idx == b
        if mask.sum() < 2: continue
        var_b = gains[mask].var(axis=0).mean()  # mean over gain dims
        total += var_b * mask.sum()
        total_n += mask.sum()
    return total / total_n if total_n > 0 else float("nan")


def measured_mse(baseline):
    """Mean PD-effective gain MSE across 5 seeds on IID test set."""
    mses = []
    for s in SEEDS:
        p = ROOT / f"seed{s}/{baseline}/predictions.npz"
        if not p.exists():
            continue
        d = np.load(p, allow_pickle=True)
        true_pd = transform_to_pd(d["true_gains_test"])
        pred_pd = transform_to_pd(d["pred_gains_test"])
        mses.append(np.mean((pred_pd - true_pd) ** 2))
    return np.mean(mses) if mses else float("nan")


def main():
    # Load training labels & factors from any baseline (3f_oracle has full split)
    p_ref = np.load(ROOT / "seed42/paramid_3f_oracle/predictions.npz", allow_pickle=True)
    phys_train = p_ref["phys_train"]   # (2400, 3)
    train_pd = transform_to_pd(p_ref["true_gains_train"])

    # Generate all candidate lists L ⊆ {0, 1, 2}
    print(f"\n=== V_hidden vs measured PD-MSE (F=3 CasADi IID) ===\n")
    print(f"  {'list':<28}  {'V_hid (sim)':>12}  {'baseline':<22}  {'measured MSE':>13}")
    print("  " + "-" * 80)

    rows = []
    for L_size in range(0, 4):
        for L_idx in combinations(range(3), L_size):
            L_str = "{" + ", ".join(FACTOR_NAMES[i] for i in L_idx) + "}" if L_idx else "{} (no factors)"
            vhid = vhid_for_list(phys_train, train_pd, list(L_idx))

            if L_idx in LIST_TO_BASELINE:
                bl = LIST_TO_BASELINE[L_idx]
                mse = measured_mse(bl)
                bl_str = bl
                mse_str = f"{mse:.3f}"
            else:
                bl_str = "(no model trained)"
                mse = float("nan")
                mse_str = "—"

            rows.append({"L_str": L_str, "vhid": vhid, "baseline": bl_str,
                         "mse": mse, "L_size": L_size})
            print(f"  {L_str:<28}  {vhid:>12.4f}  {bl_str:<22}  {mse_str:>13}")

    # Linear fit V_hid vs measured MSE on the 7 trained lists
    trained = [r for r in rows if not np.isnan(r["mse"])]
    if len(trained) >= 3:
        x = np.array([r["vhid"] for r in trained])
        y = np.array([r["mse"] for r in trained])
        # OLS y = a*x + b
        a, b = np.polyfit(x, y, 1)
        r = np.corrcoef(x, y)[0, 1]
        # R²
        y_pred = a * x + b
        ss_res = ((y - y_pred) ** 2).sum()
        ss_tot = ((y - y.mean()) ** 2).sum()
        r2 = 1 - ss_res / ss_tot

        print(f"\n  Linear fit on {len(trained)} trained lists:")
        print(f"    measured_MSE = {a:.4f} * V_hid(L) + {b:.4f}")
        print(f"    Pearson r = {r:.4f}")
        print(f"    R² = {r2:.4f}")

        print(f"\n  Per-point comparison (sorted by V_hid):")
        print(f"  {'list':<28}  {'V_hid':>10}  {'measured MSE':>13}  {'predicted MSE':>14}  {'residual':>10}")
        for r_row in sorted(trained, key=lambda x: x["vhid"]):
            pred = a * r_row["vhid"] + b
            res = r_row["mse"] - pred
            print(f"  {r_row['L_str']:<28}  {r_row['vhid']:>10.4f}  {r_row['mse']:>13.3f}  "
                  f"{pred:>14.3f}  {res:>+10.3f}")


if __name__ == "__main__":
    main()
