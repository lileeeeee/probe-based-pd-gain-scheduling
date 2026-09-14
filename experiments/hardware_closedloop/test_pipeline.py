"""Synthetic test for the closed-loop analysis pipeline.

Generates fake CSV trials whose joint trajectories are perturbed
versions of the canonical reference. Verifies:
  1. Filename parsing recovers (payload, baseline, trial_index).
  2. trial_metrics() runs without error on a 200-sample 25 Hz CSV.
  3. Per-cell summary and cross-baseline stats produce expected
     output (e.g., the controller with smaller perturbation has lower
     RMSE).
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from analyze_rmse import (  # noqa: E402
    JOINTS,
    all_metric_stats,
    collect_trials,
    parse_filename,
    per_cell_summary,
    trials_to_long_df,
)
from reference_quintic import (  # noqa: E402
    DEFAULT_DT,
    make_reference,
    trial_duration,
)


def write_synthetic_csv(
    out_dir: Path,
    payload: float,
    baseline_tag: str,
    trial_index: int,
    error_scale: float,
    seed: int,
):
    """Write a synthetic trial CSV in the expected open-loop column
    convention. Joint state = reference + Gaussian noise scaled by
    `error_scale`.
    """
    T_total = trial_duration()
    t = np.arange(0, T_total, DEFAULT_DT)
    q_ref, qdot_ref, _ = make_reference(t)

    rng = np.random.default_rng(seed)
    q = q_ref + error_scale * rng.standard_normal(q_ref.shape)
    qdot = qdot_ref + error_scale * rng.standard_normal(qdot_ref.shape)
    tau_cmd = error_scale * rng.standard_normal(q_ref.shape)

    payload_str = f"{payload:.2f}".replace(".", "p") + "kg"
    fname = f"20260428_120000_synth_payload_{payload_str}_{baseline_tag}_pass{trial_index}.csv"
    path = out_dir / fname

    columns = [
        "wall_time_sec", "sim_time_sec", "replay_time_sec",
        "profile", "pass_index",
    ] + [f"q_alpha_axis_{j}" for j in JOINTS] \
      + [f"dq_alpha_axis_{j}" for j in JOINTS] \
      + [f"effort_alpha_axis_{j}" for j in JOINTS] \
      + [f"cmd_tau_axis_{j}" for j in JOINTS] \
      + ["payload_mass", "gravity"]

    with path.open("w") as f:
        f.write(",".join(columns) + "\n")
        for i in range(len(t)):
            row = [
                f"{1.7e9 + i*DEFAULT_DT:.6f}",
                f"{i*DEFAULT_DT:.6f}",
                f"{i*DEFAULT_DT:.6f}",
                f"synth_{baseline_tag}",
                str(trial_index),
            ]
            row += [f"{q[i, k]:.6f}" for k in range(4)]
            row += [f"{qdot[i, k]:.6f}" for k in range(4)]
            row += [f"{0.0:.6f}" for _ in range(4)]  # effort placeholder
            row += [f"{tau_cmd[i, k]:.6f}" for k in range(4)]
            row += [f"{payload:.4f}", "9.81"]
            f.write(",".join(row) + "\n")
    return path


def main():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        # Generate trials: baselines with increasing error scale
        # implicit (best) → partial-cheat (medium) → fixed (worst)
        scales = {
            "implicitid":     0.005,
            "partial_cheat":  0.010,
            "fixed":          0.030,
        }
        seed = 0
        for payload in (0.0, 0.38, 0.76):
            for tag, scale in scales.items():
                for trial in (1, 2, 3):
                    seed += 1
                    write_synthetic_csv(tmp, payload, tag, trial, scale, seed)

        print(f"Wrote {sum(1 for _ in tmp.glob('*.csv'))} synthetic trials.\n")

        # 1. Filename parsing
        for p in sorted(tmp.glob("*.csv"))[:3]:
            print(f"  parse {p.name}")
            print(f"    → {parse_filename(p)}")
        print()

        # 2. End-to-end load + metrics
        trials = collect_trials(tmp)
        print(f"Loaded {len(trials)} trials.")
        sample = trials[0]
        print(f"\nSample metrics (1st trial):")
        for k, v in sorted(sample["metrics"].items()):
            if hasattr(v, "shape") or isinstance(v, (list, tuple)):
                print(f"  {k}: {np.asarray(v).round(5)}")
            else:
                print(f"  {k}: {v:.5f}")

        # 3. Per-cell summary
        print("\nPer-cell summary:")
        summary = per_cell_summary(trials)
        import pandas as pd
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 220)
        print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

        # 4. Cross-baseline (expect implicit < partial-cheat < fixed)
        print("\nCross-baseline stats (ref = ImplicitID, expect ratios > 1):")
        stats = all_metric_stats(trials, ref_baseline="C1_implicitid")
        print(stats.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

        # Sanity assertion: median rmse_trial ranking matches scales
        ranked = (
            trials_to_long_df(trials)
            .groupby("baseline")["rmse_trial"]
            .median()
        )
        print("\nMedian rmse_trial per baseline:")
        print(ranked.to_string())
        assert ranked.idxmin() == "C1_implicitid", \
            "Sanity check failed: expected ImplicitID to have lowest median RMSE"
        assert ranked["C1_implicitid"] < ranked["C2_partial_cheat"] < ranked["C3_fixed"], \
            f"Sanity check failed: expected ranking C1 < C2 < C3, got {ranked}"
        print("\n✓ Pipeline end-to-end test passed.")


if __name__ == "__main__":
    main()
