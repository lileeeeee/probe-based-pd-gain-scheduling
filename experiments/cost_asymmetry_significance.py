"""Significance tests for the F=2 cost-space asymmetry (Table tab:cost_asymmetry).

Recomputes paired significance on the current §5.4 cost data (the per-point
30-task deployment costs).  Paired Wilcoxon signed-rank tests, one test
family, applied identically to CasADi and MuJoCo:

  * \textsc{ImplicitID} vs each \textsc{Partial-1F}  (probe escapes the floor)
  * \textsc{Partial-1F-mp} vs \textsc{Partial-1F-fv}  (the asymmetry itself)

Also reproduces the failure-rate table P(cost > 1.1 c_BO) as a sanity check.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
OUT_JSON = ROOT / "output" / "cost_asymmetry_significance.json"
SEEDS = [42, 7, 123, 2025, 2026]


def load_casadi():
    d = np.load(ROOT / "data" / "casadi2d_iid_cost_30tasks.npz", allow_pickle=True)
    return {
        "cbo": np.asarray(d["true_cost_30tasks_recomp"], float),
        "Oracle": np.asarray(d["paramid_2f_mp_fv_cost"], float),
        "ImplicitID": np.asarray(d["excite_e2e_cost"], float),
        "Partial-mp": np.asarray(d["paramid_1f_mp_cost"], float),
        "Partial-fv": np.asarray(d["paramid_1f_fv_cost"], float),
    }


def load_mujoco():
    d = np.load(ROOT / "data" / "mujoco2d_iid_cost_30task.npz", allow_pickle=True)
    cat = lambda key: np.concatenate([np.asarray(d[f"seed{s}__{key}"], float)
                                      for s in SEEDS])
    return {
        "cbo": cat("true_gains"),
        "Oracle": cat("paramid_2f_mp_fv"),
        "ImplicitID": cat("excite_e2e"),
        "Partial-mp": cat("paramid_1f_mp"),
        "Partial-fv": cat("paramid_1f_fv"),
    }


def analyse(name, c):
    cbo = c["cbo"]
    fail = {m: float(np.mean(c[m] > 1.1 * cbo)) for m in
            ["Oracle", "ImplicitID", "Partial-mp", "Partial-fv"]}
    # paired Wilcoxon signed-rank on per-point cost
    tests = {
        # Two-sided: is the probe distinguishable from the full-factor oracle
        # at deployment cost?  (Reported as "matches" only if not.)
        "ImplicitID~Oracle": wilcoxon(c["ImplicitID"], c["Oracle"]),
        "ImplicitID<Partial-mp": wilcoxon(c["ImplicitID"], c["Partial-mp"],
                                          alternative="less"),
        "ImplicitID<Partial-fv": wilcoxon(c["ImplicitID"], c["Partial-fv"],
                                          alternative="less"),
        "Partial-mp<Partial-fv": wilcoxon(c["Partial-mp"], c["Partial-fv"],
                                          alternative="less"),
    }
    med_excess = {m: float(np.median(c[m] / cbo - 1.0)) for m in
                  ["Oracle", "ImplicitID", "Partial-mp", "Partial-fv"]}
    print(f"\n=== {name}  (n={len(cbo)} points) ===")
    print("  median cost excess over c_BO: " +
          "  ".join(f"{m} {med_excess[m]*100:+.2f}%" for m in med_excess))
    print("  failure rate P(cost>1.1 c_BO): " +
          "  ".join(f"{m} {fail[m]*100:.1f}%" for m in fail))
    out = {"n": int(len(cbo)), "failure_rate_pct":
           {m: round(fail[m] * 100, 2) for m in fail},
           "median_cost_excess_pct": {m: round(med_excess[m] * 100, 3)
                                      for m in med_excess},
           "wilcoxon_p": {}}
    for k, r in tests.items():
        out["wilcoxon_p"][k] = float(r.pvalue)
        print(f"  {k:24s}  p = {r.pvalue:.3e}")
    return out


def main():
    result = {"test": "paired Wilcoxon signed-rank on per-point 30-task cost",
              "CasADi": analyse("CasADi", load_casadi()),
              "MuJoCo": analyse("MuJoCo", load_mujoco())}
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n[saved] {OUT_JSON}")


if __name__ == "__main__":
    main()
