"""ANOVA F-stat cluster characterization on PD-effective 8-dim labels.

PD transform: Kp_eff = Kp + dt*Ki, drop Ki, keep Kd → (N, 8).

For each PD dim:
  1. GMM/BIC pick K
  2. Per factor: one-way ANOVA F-stat + p-value
  3. Bonferroni-corrected α across (n_dim × n_factor) tests
  4. Dominant factor: max F-stat among p < α
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from scipy.stats import f_oneway
from sklearn.mixture import GaussianMixture

JOINT_4 = ("e", "d", "c", "b")
DT_DEFAULT = 0.04


def transform_to_pd(g, dt=DT_DEFAULT):
    nJ = g.shape[1] // 3
    Kp_eff = g[:, :nJ] + dt * g[:, nJ:2*nJ]
    Kd = g[:, 2*nJ:]
    return np.concatenate([Kp_eff, Kd], axis=1)


def fit_gmm_dim(y, max_k=4):
    if y.std() < 1e-3:
        return np.zeros(len(y), dtype=int), 1
    y_col = y.reshape(-1, 1)
    best_bic = np.inf; lab = None; bk = 1
    for k in range(1, max_k + 1):
        try:
            gmm = GaussianMixture(n_components=k, random_state=0, max_iter=200).fit(y_col)
            if gmm.bic(y_col) < best_bic:
                best_bic = gmm.bic(y_col); lab = gmm.predict(y_col); bk = k
        except Exception:
            pass
    return lab, bk


def analyse(fname_orig, label):
    """Load original 12-dim dataset, apply PD transform, run ANOVA."""
    d = np.load(f"/Users/lizheng/PycharmProjects/casadi-moe/data/{fname_orig}",
                allow_pickle=True)
    phys = d["phys_factors"].astype(float)
    g = d["gains"].astype(float)
    n_factors = phys.shape[1]
    fn = (["m_p", "f_v", "f_c"][:n_factors] if d.get("factor_names", None) is None
          else [str(s) for s in d["factor_names"]])

    # dt depends on platform: CasADi 0.04, MuJoCo 0.002
    dt = float(d["step_dt"]) if "step_dt" in d.keys() else DT_DEFAULT
    g_pd = transform_to_pd(g, dt=dt)
    n_dims = g_pd.shape[1]
    n_joints = n_dims // 2  # PD has 2 components: Kp_eff, Kd

    if n_joints == 4:
        joint_names = JOINT_4
    else:
        joint_names = tuple(str(i) for i in range(n_joints))

    pd_labels = [f"Kp_{joint_names[j]}" for j in range(n_joints)] + \
                [f"Kd_{joint_names[j]}" for j in range(n_joints)]

    p_threshold = 0.05 / (n_dims * n_factors)
    print(f"\n{'═' * 80}")
    print(f"{label}  (PD-effective)")
    print(f"  N={phys.shape[0]}, factors={fn}, dims={n_dims}")
    print(f"  Bonferroni α = 0.05 / {n_dims * n_factors} = {p_threshold:.2e}")
    print(f"{'═' * 80}")
    fmt_p = lambda p: ("< 1e-12" if p < 1e-12 else f"{p:.1e}")

    hdr = f"  {'dim':<6} {'K':>3}  "
    for f in fn:
        hdr += f"{'F('+f+')':>10}  {'p('+f+')':>10}  "
    hdr += f"  {'dominant':<10}  {'gain range':>15}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    rows = []
    for d_idx in range(n_dims):
        labels, K = fit_gmm_dim(g_pd[:, d_idx])
        if K < 2:
            print(f"  {pd_labels[d_idx]:<6} {K:>3}  (no partition)")
            continue
        F_stats = []; p_values = []
        for fi in range(n_factors):
            groups = [phys[labels == k, fi] for k in range(K) if (labels == k).sum() >= 2]
            if len(groups) < 2:
                F_stats.append(0.0); p_values.append(1.0); continue
            F, p = f_oneway(*groups)
            F_stats.append(F); p_values.append(p)
        F_stats = np.array(F_stats); p_values = np.array(p_values)
        sig = p_values < p_threshold
        if sig.sum() == 0:
            dom = "—"
        else:
            sig_idx = np.where(sig)[0]
            dom = fn[sig_idx[np.argmax(F_stats[sig_idx])]]

        means = []
        for k in range(K):
            mk = labels == k
            if mk.sum() >= 2:
                means.append(g_pd[mk, d_idx].mean())
        means = sorted(means)
        gr_str = f"[{means[0]:.2f}, {means[-1]:.2f}]"

        row = f"  {pd_labels[d_idx]:<6} {K:>3}  "
        for fi in range(n_factors):
            row += f"{F_stats[fi]:>10.1f}  {fmt_p(p_values[fi]):>10}  "
        row += f"  {dom:<10}  {gr_str:>15}"
        print(row)
        rows.append({"dim": pd_labels[d_idx], "K": K, "F": F_stats.tolist(),
                     "p": p_values.tolist(), "dominant": dom})

    print(f"\n  Dominant factor breakdown:")
    by_dom = {}
    for r in rows:
        by_dom.setdefault(r["dominant"], []).append(r["dim"])
    for k_dom in sorted(by_dom):
        print(f"    {k_dom:<10}: {', '.join(by_dom[k_dom])}")

    return rows


def main():
    analyse("dataset_3d_envB_full.npz", "F=3 CasADi 4-DoF")
    analyse("dataset_casadi.npz", "F=2 CasADi 4-DoF")
    analyse("dataset_mujoco.npz", "F=2 MuJoCo 7-DoF")


if __name__ == "__main__":
    main()
