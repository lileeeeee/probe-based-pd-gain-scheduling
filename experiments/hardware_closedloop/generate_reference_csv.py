"""Export the quintic closed-loop reference as CSV + JSON spec, mirroring
the format of `data/hardware_probe/`.

Default: single 8 s quintic segment q_A → q_B (200 samples @ 25 Hz, no
dwells, no return). See `reference_quintic.py` for parameter overrides.

Output files (default in `data/hardware_closedloop_ref/`):
  - reference_dense_25hz.csv : dense reference at sim_dt
        columns: time_sec, q_ref_axis_{e,d,c,b}, qdot_ref_axis_{e,d,c,b}
  - reference_waypoints.csv  : per-segment endpoints
        columns: time_sec, q_ref_axis_{e,d,c,b}
  - reference_spec.json      : metadata (waypoints, durations, schedule,
                                joint ordering, interpolant)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from reference_quintic import (  # noqa: E402
    DEFAULT_DT,
    DEFAULT_N_RETURN_CYCLES,
    DEFAULT_Q_A,
    DEFAULT_Q_B,
    DEFAULT_T_DWELL,
    DEFAULT_T_SEG,
    make_reference,
    trial_duration,
)

JOINT_NAMES = ("axis_e", "axis_d", "axis_c", "axis_b")
JOINT_DESCRIPTION = (
    "alpha5 4-DoF: axis_e (base) → axis_d → axis_c → axis_b (wrist)"
)


def export(
    output_dir: Path,
    q_A: np.ndarray = DEFAULT_Q_A,
    q_B: np.ndarray = DEFAULT_Q_B,
    t_dwell: float = DEFAULT_T_DWELL,
    T_seg: float = DEFAULT_T_SEG,
    n_return_cycles: int = DEFAULT_N_RETURN_CYCLES,
    dt: float = DEFAULT_DT,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    T_total = trial_duration(t_dwell=t_dwell, T_seg=T_seg,
                             n_return_cycles=n_return_cycles)
    H = int(round(T_total / dt))   # number of samples
    t = np.arange(H) * dt          # 0, dt, 2dt, ..., (H-1)*dt

    q_ref, qdot_ref, mask = make_reference(
        t, q_A=q_A, q_B=q_B,
        t_dwell=t_dwell, T_seg=T_seg, n_return_cycles=n_return_cycles,
    )

    # ----- dense CSV -----
    dense_path = output_dir / "reference_dense_25hz.csv"
    with dense_path.open("w") as f:
        cols = ["time_sec"]
        cols += [f"q_ref_{name}" for name in JOINT_NAMES]
        cols += [f"qdot_ref_{name}" for name in JOINT_NAMES]
        f.write(",".join(cols) + "\n")
        for i in range(H):
            vals = [f"{t[i]:.6f}"]
            vals += [f"{q_ref[i, j]:.6f}" for j in range(4)]
            vals += [f"{qdot_ref[i, j]:.6f}" for j in range(4)]
            f.write(",".join(vals) + "\n")

    # ----- waypoints CSV (segment endpoints) -----
    waypoints = [(0.0, q_A.copy())]
    cursor = 0.0
    if t_dwell > 0:
        cursor += t_dwell
        waypoints.append((cursor, q_A.copy()))
    # mandatory A → B
    cursor += T_seg
    waypoints.append((cursor, q_B.copy()))
    for _ in range(n_return_cycles):
        if t_dwell > 0:
            cursor += t_dwell
            waypoints.append((cursor, q_B.copy()))
        cursor += T_seg
        waypoints.append((cursor, q_A.copy()))
        if t_dwell > 0:
            cursor += t_dwell
            waypoints.append((cursor, q_A.copy()))
        cursor += T_seg
        waypoints.append((cursor, q_B.copy()))
    if t_dwell > 0:
        cursor += t_dwell
        waypoints.append((cursor, q_B.copy()))

    wp_path = output_dir / "reference_waypoints.csv"
    with wp_path.open("w") as f:
        f.write("time_sec," + ",".join(f"q_ref_{n}" for n in JOINT_NAMES) + "\n")
        for ti, qi in waypoints:
            f.write(f"{ti:.6f}," + ",".join(f"{qi[j]:.6f}" for j in range(4)) + "\n")

    # ----- spec JSON -----
    if n_return_cycles == 0 and t_dwell == 0:
        schedule_str = f"single quintic({T_seg:g}s) q_A → q_B"
    else:
        schedule_str = (
            f"dwell({t_dwell:g}s)? @ q_A → quintic({T_seg:g}s) A→B"
            + (f" → {n_return_cycles}× [dwell?({t_dwell:g}s) B → quintic({T_seg:g}s) B→A "
               f"→ dwell?({t_dwell:g}s) A → quintic({T_seg:g}s) A→B]"
               if n_return_cycles > 0 else "")
            + (f" → dwell({t_dwell:g}s) @ q_B" if t_dwell > 0 else "")
        )
    transition_total = (1 + 2 * n_return_cycles) * T_seg
    spec = {
        "duration_sec": float(T_total),
        "n_samples_25hz": int(H),
        "sim_dt": float(dt),
        "n_return_cycles": int(n_return_cycles),
        "T_seg_sec": float(T_seg),
        "t_dwell_sec": float(t_dwell),
        "q_A_rad": q_A.tolist(),
        "q_B_rad": q_B.tolist(),
        "joint_ordering": JOINT_DESCRIPTION,
        "schedule": schedule_str,
        "interpolant": "5th-order quintic, s(τ)=10τ³-15τ⁴+6τ⁵, "
                       "C² with zero start/end velocity & acceleration",
        "transition_phase_total_sec": float(transition_total),
        "note": (
            "Apply q_ref(t), qdot_ref(t) as PD controller setpoint at 25 Hz. "
            "Compute trial RMSE over transition phases (the quintic segments). "
            "With default settings (no dwells), transition mask covers the "
            "entire 8 s trial."
        ),
    }
    spec_path = output_dir / "reference_spec.json"
    with spec_path.open("w") as f:
        json.dump(spec, f, indent=2)

    return {
        "dense_csv": dense_path,
        "waypoints_csv": wp_path,
        "spec_json": spec_path,
        "n_samples": H,
        "duration_sec": T_total,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "-o", "--output_dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "data" / "hardware_closedloop_ref",
        help="Output directory (default: data/hardware_closedloop_ref/)",
    )
    args = p.parse_args()

    info = export(args.output_dir)
    print(f"Generated reference trajectory:")
    print(f"  Duration: {info['duration_sec']:.2f} s, samples: {info['n_samples']} @ 25 Hz")
    print(f"  Dense CSV     → {info['dense_csv']}")
    print(f"  Waypoints CSV → {info['waypoints_csv']}")
    print(f"  Spec JSON     → {info['spec_json']}")


if __name__ == "__main__":
    main()
