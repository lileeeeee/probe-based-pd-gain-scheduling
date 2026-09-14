"""First-order and total-effect variance shares per gain dimension (R2-C7).

Reviewer 2, comment 7 asks for effect sizes and Sobol indices, noting that
one-way ANOVA F-statistics on 3000 samples are significant even for modest
effects and that separate one-way tests neglect interactions.

Sobol indices normally need free sampling of the input space. Our labels live
on a fixed 3000-point Sobol grid (one BO gain vector per factor point), so we
estimate the same two quantities by quantile-bin variance decomposition, the
estimator already used for \\Vbin (experiments/r2c8_vbin_bootstrap_ci.vhid):

    S1_i  = 1 - E[Var(g | c_i)]        / Var(g)      first-order share
    ST_i  = E[Var(g | c_{-i})]         / Var(g)      total-effect share

with E[Var(g | .)] the bin-conditional variance. Both are computed per gain
dimension on z-scored gains so dimensions are comparable. Interaction share
is ST_i - S1_i. These are the bin estimates of the Sobol indices and carry the
same upward bias in E[Var(.)] that Section 5.3 documents for \\Vbin; the bias
inflates E[Var] and therefore deflates S1 and inflates ST. n_bins is swept so
the reader can see the sensitivity.

Writes output/r2c7_sobol_recompute.json.
"""
import sys, json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from experiments.r2c8_vbin_bootstrap_ci import vhid, transform_to_pd

DATASET = PROJECT_ROOT / "data" / "dataset_3d_envB_full.npz"
FACTORS = ["m_p", "f_v", "f_c"]
DIMS = ["Kp_base", "Kp_shld", "Kp_elbow", "Kp_wrist",
        "Kd_base", "Kd_shld", "Kd_elbow", "Kd_wrist"]
N_BINS_MAIN = 8
N_BINS_SWEEP = (5, 8, 12, 20)


def shares(phys, g_z, n_bins):
    """Per-dimension S1 and ST for every factor, at one bin count."""
    out = {}
    tot = g_z.var(axis=0)                                   # (D,) == 1.0 for z-scored
    for i, f in enumerate(FACTORS):
        others = [j for j in range(len(FACTORS)) if j != i]
        s1, st = [], []
        for d in range(g_z.shape[1]):
            col = g_z[:, d:d + 1]
            v_i = vhid(phys, col, [i], n_bins=n_bins)        # E[Var(g | c_i)]
            v_o = vhid(phys, col, others, n_bins=n_bins)     # E[Var(g | c_-i)]
            s1.append(1.0 - v_i / tot[d])
            st.append(v_o / tot[d])
        out[f] = {"S1": s1, "ST": st}
    return out


def main():
    z = np.load(DATASET, allow_pickle=True)
    phys = np.asarray(z["phys_factors"], dtype=np.float64)
    g_pd = transform_to_pd(np.asarray(z["gains"], dtype=np.float64))
    g_z = (g_pd - g_pd.mean(0)) / g_pd.std(0)
    print(f"dataset {DATASET.name}: phys {phys.shape}, PD gains {g_pd.shape}")

    main_tab = shares(phys, g_z, N_BINS_MAIN)
    print(f"\n=== variance shares, n_bins = {N_BINS_MAIN} (z-scored gains) ===")
    print(f"{'dim':10s} " + "  ".join(f"{f:>18s}" for f in FACTORS) + "   dominant(ST)")
    print(f"{'':10s} " + "  ".join(f"{'S1':>6s}{'ST':>6s}{'int':>6s}" for _ in FACTORS))
    rows = {}
    for d, name in enumerate(DIMS):
        line = f"{name:10s} "
        st_d = {}
        for f in FACTORS:
            s1 = main_tab[f]["S1"][d]; st = main_tab[f]["ST"][d]
            st_d[f] = st
            line += f"  {s1:6.3f}{st:6.3f}{st - s1:6.3f}"
        dom = max(st_d, key=st_d.get)
        rows[name] = {f: {"S1": round(main_tab[f]["S1"][d], 4),
                          "ST": round(main_tab[f]["ST"][d], 4),
                          "interaction": round(main_tab[f]["ST"][d] - main_tab[f]["S1"][d], 4)}
                      for f in FACTORS}
        rows[name]["dominant_ST"] = dom
        print(line + f"   {dom}")

    sweep = {str(nb): shares(phys, g_z, nb) for nb in N_BINS_SWEEP}
    print(f"\n=== n_bins sensitivity: mean over dimensions ===")
    print(f"{'n_bins':>7s} " + "  ".join(f"{f:>14s}" for f in FACTORS))
    for nb in N_BINS_SWEEP:
        line = f"{nb:7d} "
        for f in FACTORS:
            line += f"   S1 {np.mean(sweep[str(nb)][f]['S1']):5.3f} ST {np.mean(sweep[str(nb)][f]['ST']):5.3f}"
        print(line)

    out = {"dataset": DATASET.name, "n_points": int(len(phys)), "n_bins_main": N_BINS_MAIN,
           "estimator": "quantile-bin variance decomposition (same estimator as Vbin); "
                        "S1_i = 1 - E[Var(g|c_i)]/Var(g), ST_i = E[Var(g|c_-i)]/Var(g), z-scored gains",
           "factors": FACTORS, "per_dimension": rows,
           "n_bins_sweep": {nb: {f: {"S1": [round(v, 4) for v in sweep[nb][f]["S1"]],
                                     "ST": [round(v, 4) for v in sweep[nb][f]["ST"]]}
                                 for f in FACTORS} for nb in sweep}}
    p = PROJECT_ROOT / "output" / "r2c7_sobol_recompute.json"
    json.dump(out, open(p, "w"), indent=1)
    print(f"\nwrote {p.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
