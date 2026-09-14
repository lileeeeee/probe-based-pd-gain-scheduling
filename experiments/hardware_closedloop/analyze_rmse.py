"""Closed-loop hardware RMSE analysis.

Reads CSV trial logs (same column convention as the open-loop probe
data in `data/real_robot/`) and computes:

  - Per-trial RMSE on the quintic-transition phases (excluding dwell).
  - Per-cell median RMSE: 3 trials per (payload, baseline) cell.
  - Cross-baseline statistics: Wilcoxon signed-rank + Cliff's delta on
    pooled trials.

The reference trajectory is regenerated in software via
`reference_quintic.make_reference`, so we don't depend on whether the
ROS2 logger included `q_ref` columns in the CSV.

Expected CSV column convention (4 active joints e, d, c, b):
    wall_time_sec, sim_time_sec, replay_time_sec,
    profile, pass_index,
    q_alpha_axis_e, q_alpha_axis_d, q_alpha_axis_c, q_alpha_axis_b,
    dq_alpha_axis_e, dq_alpha_axis_d, dq_alpha_axis_c, dq_alpha_axis_b,
    cmd_tau_axis_e, cmd_tau_axis_d, cmd_tau_axis_c, cmd_tau_axis_b,
    payload_mass

Filename / profile convention assumed:
    <timestamp>_<scenario>_payload_<XpYZkg>_baseline_<C>_pass<N>.csv
where C ∈ {C1_implicitid, C2_partial_cheat, C3_fixed, C4_handtuned}
and N is the trial index. Adapt `parse_filename()` if your logger uses
a different convention.

Usage:
    python analyze_rmse.py /path/to/closedloop_csv_dir
"""
from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# Local import — we run as `python analyze_rmse.py` so pyimport via
# explicit path
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from reference_quintic import (  # noqa: E402
    make_reference,
    trial_duration,
    DEFAULT_DT,
)
import json  # noqa: E402

# Reference used for error computation. Defaults reproduce the May-2026
# protocol (8 s A->B). Override per dataset with --ref_spec (a
# reference_spec.json written by generate_reference_csv.py), or bypass the
# analytic reference entirely with --use_logged_ref (error against the
# ref_alpha_axis_* columns the controller actually tracked).
REF_KW: dict = {}
USE_LOGGED_REF: bool = False


def load_ref_spec(path: Path) -> dict:
    spec = json.load(open(path))
    return dict(q_A=np.asarray(spec["q_A_rad"], float),
                q_B=np.asarray(spec["q_B_rad"], float),
                T_seg=float(spec["T_seg_sec"]),
                t_dwell=float(spec.get("t_dwell_sec", 0.0)),
                n_return_cycles=int(spec.get("n_return_cycles", 0)))


JOINTS = ("e", "d", "c", "b")


# ---------- file/metadata parsing ----------

@dataclass(frozen=True)
class TrialMeta:
    payload_kg: float
    baseline: str  # 'C1_implicitid', 'C2_partial_cheat', 'C3_fixed', 'C4_handtuned'
    trial_index: int
    csv_path: Path


_BASELINE_ALIASES = {
    "implicitid": "C1_implicitid",
    "implicit_id": "C1_implicitid",
    "c1": "C1_implicitid",
    "partial_cheat": "C2_partial_cheat",
    "partial1f": "C2_partial_cheat",
    "partial_1f": "C2_partial_cheat",
    "c2": "C2_partial_cheat",
    "fixed": "C3_fixed",
    "oracle0kg": "C3_fixed",
    "c3": "C3_fixed",
    "handtuned": "C4_handtuned",
    "hand_tuned": "C4_handtuned",
    "manual": "C4_handtuned",
    "c4": "C4_handtuned",
}


def _normalize_baseline(s: str) -> str | None:
    return _BASELINE_ALIASES.get(s.lower().replace("-", "_"))


def parse_filename(path: Path) -> TrialMeta | None:
    """Extract (payload_kg, baseline, trial_index) from filename.

    Tries a few flexible patterns. Returns None if no match — caller
    decides what to do.

    Recognised filename patterns:
      ..._payload_0p38kg_..._baseline_C1_implicitid_..._pass3.csv
      ..._payload_0_38_..._C2_partial_cheat_pass2.csv
      ..._payload_0p76kg_..._handtuned_pass1.csv

    Falls back to the parent directory name for the baseline tag
    when the filename omits it (e.g. unzipped layout `C1_0_57/<csv>`
    where the CSV name carries only payload + pass info).
    """
    name = path.stem

    # Payload like 0p38kg or 0_38
    m_pay = re.search(r"payload[_-](\d+)[p_](\d+)(?:kg)?", name, re.IGNORECASE)
    if not m_pay:
        return None
    integer, decimal = m_pay.group(1), m_pay.group(2)
    payload_kg = float(f"{integer}.{decimal}")

    # Baseline tag: try filename first, then parent directory name
    baseline = None
    for alias in _BASELINE_ALIASES:
        if alias.lower() in name.lower():
            baseline = _BASELINE_ALIASES[alias]
            break
    if baseline is None:
        parent = path.parent.name.lower()
        for alias in _BASELINE_ALIASES:
            # Match alias as a token in the parent dir name (e.g.,
            # "c1_0_57" → matches "c1"); avoid substring false
            # positives by requiring boundary
            if re.search(rf"(?:^|[_-]){re.escape(alias.lower())}(?:[_-]|$)", parent):
                baseline = _BASELINE_ALIASES[alias]
                break
    if baseline is None:
        return None

    # Pass / trial index
    m_pass = re.search(r"pass(\d+)|trial[_-]?(\d+)", name, re.IGNORECASE)
    if not m_pass:
        return None
    trial_index = int(m_pass.group(1) or m_pass.group(2))

    return TrialMeta(payload_kg, baseline, trial_index, path)


# ---------- CSV loading ----------

def load_trial_csv(meta: TrialMeta, dt: float = DEFAULT_DT) -> dict:
    """Load one trial CSV; resample to uniform `dt` grid via linear
    interpolation on `replay_time_sec`. Returns dict with t, q, qdot,
    tau_cmd arrays.
    """
    df = pd.read_csv(meta.csv_path)
    # Some loggers store cmd torques only for sub-axes; require all 4
    required = (
        ["replay_time_sec"]
        + [f"q_alpha_axis_{j}" for j in JOINTS]
        + [f"dq_alpha_axis_{j}" for j in JOINTS]
        + [f"cmd_tau_axis_{j}" for j in JOINTS]
    )
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{meta.csv_path.name}: missing columns {missing}")

    t_raw = df["replay_time_sec"].to_numpy()
    valid = np.isfinite(t_raw)
    t_raw = t_raw[valid]
    if len(t_raw) < 2:
        raise ValueError(f"{meta.csv_path.name}: too few rows after filtering")

    # Uniform grid spanning the trial. With --use_logged_ref the span comes
    # from the log itself (works for any reference); otherwise from the
    # analytic schedule so that a short log is not padded by extrapolation.
    if USE_LOGGED_REF:
        T_total = float(t_raw[-1] - t_raw[0]) + dt
        t_raw = t_raw - t_raw[0]
    else:
        T_total = trial_duration(**{k: REF_KW[k] for k in ("t_dwell", "T_seg", "n_return_cycles") if k in REF_KW})
    t_uniform = np.arange(0.0, T_total, dt)

    def _interp(col):
        v = df[col].to_numpy()[valid]
        return np.interp(t_uniform, t_raw, v)

    q = np.column_stack([_interp(f"q_alpha_axis_{j}") for j in JOINTS])
    qdot = np.column_stack([_interp(f"dq_alpha_axis_{j}") for j in JOINTS])
    tau_cmd = np.column_stack([_interp(f"cmd_tau_axis_{j}") for j in JOINTS])

    out = {
        "t": t_uniform,
        "q": q,
        "qdot": qdot,
        "tau_cmd": tau_cmd,
        "meta": meta,
    }
    # Logged reference (what the controller actually tracked), if present
    ref_cols = [f"ref_alpha_axis_{j}" for j in JOINTS]
    if all(c in df.columns for c in ref_cols):
        out["q_ref_logged"] = np.column_stack([_interp(c) for c in ref_cols])
        out["qdot_ref_logged"] = np.gradient(out["q_ref_logged"], dt, axis=0)
    return out


# ---------- RMSE computation ----------

def trial_metrics(trial: dict, include_joints: tuple[int, ...] | None = None) -> dict:
    """Compute a battery of per-joint and trial-level tracking metrics.

    `include_joints` selects which joint indices contribute to the
    trial-level scalar means (rmse_trial, mae_trial, ...). Per-joint
    arrays are always returned for all 4 joints regardless. Default
    None uses all 4 joints.

    All position-error metrics computed on the transition mask
    (quintic segments). Velocity-error metrics same. Final-position
    and settling metrics use the trial endpoint. Control effort
    uses the full trial.

    Returns dict with keys:
      Position-error (per-joint shape (4,) unless noted):
        rmse_per_joint            RMSE on transition mask
        mae_per_joint             MAE  on transition mask
        max_err_per_joint         max |e| over full trial
        iae_per_joint             ∫|e(t)|dt over transition mask
        itae_per_joint            ∫ t·|e(t)| dt over transition mask
                                  (t starts at 0 at segment start)
        final_pos_err_per_joint   |q[-1] - q_ref[-1]| (settle quality)
      Velocity-error:
        vel_rmse_per_joint        RMSE on qdot - qdot_ref (transition mask)
        peak_vel_err_per_joint    max |qdot_err| on transition mask
      Trial-level scalars:
        rmse_trial                mean of rmse_per_joint
        mae_trial                 mean of mae_per_joint
        max_err_trial             mean of max_err_per_joint
        iae_trial                 mean of iae_per_joint
        final_pos_err_trial       mean of final_pos_err_per_joint
        vel_rmse_trial            mean of vel_rmse_per_joint
        control_effort_rms        RMS of cmd torque over full trial (scalar)
        control_effort_per_joint  RMS of cmd torque per joint (4,)
        control_peak_per_joint    max |tau_cmd| per joint over full trial
    """
    t = trial["t"]
    q = trial["q"]
    qdot = trial["qdot"]
    tau_cmd = trial["tau_cmd"]
    dt = float(t[1] - t[0]) if len(t) > 1 else 0.04

    if USE_LOGGED_REF:
        if "q_ref_logged" not in trial:
            raise ValueError("--use_logged_ref set but CSV has no ref_alpha_axis_* columns")
        q_ref, qdot_ref = trial["q_ref_logged"], trial["qdot_ref_logged"]
        mask = np.ones(len(t), dtype=bool)   # whole logged trial is the transition
    else:
        q_ref, qdot_ref, mask = make_reference(t, **REF_KW)
    err = q - q_ref                  # (T, 4)
    vel_err = qdot - qdot_ref        # (T, 4)
    err_trans = err[mask]
    vel_err_trans = vel_err[mask]
    if err_trans.shape[0] == 0:
        raise ValueError("Empty transition mask — bad timing alignment?")

    # Position-error metrics on transition mask
    rmse = np.sqrt(np.mean(err_trans**2, axis=0))         # (4,)
    mae = np.mean(np.abs(err_trans), axis=0)              # (4,)
    iae = np.sum(np.abs(err_trans), axis=0) * dt          # (4,)
    # ITAE: time within the transition, starting from 0 at first masked
    # sample
    t_in_trans = np.arange(err_trans.shape[0]) * dt
    itae = np.sum(t_in_trans[:, None] * np.abs(err_trans), axis=0) * dt

    # Max error over full trial (catches overshoot/late-stage drift)
    max_err = np.max(np.abs(err), axis=0)                 # (4,)

    # Final-position error (last sample vs reference target)
    final_pos_err = np.abs(err[-1])                       # (4,)

    # Velocity tracking
    vel_rmse = np.sqrt(np.mean(vel_err_trans**2, axis=0))
    peak_vel_err = np.max(np.abs(vel_err_trans), axis=0)

    # Control effort
    tau_rms = np.sqrt(np.mean(tau_cmd**2, axis=0))
    tau_peak = np.max(np.abs(tau_cmd), axis=0)
    tau_rms_scalar = float(np.sqrt(np.mean(tau_cmd**2)))

    idx = (
        list(include_joints) if include_joints is not None else list(range(4))
    )
    return {
        # per-joint arrays (always full 4-joint)
        "rmse_per_joint": rmse,
        "mae_per_joint": mae,
        "max_err_per_joint": max_err,
        "iae_per_joint": iae,
        "itae_per_joint": itae,
        "final_pos_err_per_joint": final_pos_err,
        "vel_rmse_per_joint": vel_rmse,
        "peak_vel_err_per_joint": peak_vel_err,
        "control_effort_per_joint": tau_rms,
        "control_peak_per_joint": tau_peak,
        # trial-level scalars (mean across selected joints)
        "rmse_trial": float(rmse[idx].mean()),
        "mae_trial": float(mae[idx].mean()),
        "max_err_trial": float(max_err[idx].mean()),
        "iae_trial": float(iae[idx].mean()),
        "itae_trial": float(itae[idx].mean()),
        "final_pos_err_trial": float(final_pos_err[idx].mean()),
        "vel_rmse_trial": float(vel_rmse[idx].mean()),
        "peak_vel_err_trial": float(peak_vel_err[idx].mean()),
        "control_effort_rms": tau_rms_scalar,
        "_included_joints": idx,
    }


# Back-compat alias
trial_rmse = trial_metrics


# ---------- Aggregation ----------

def collect_trials(csv_dir: Path,
                    include_joints: tuple[int, ...] | None = None) -> list[dict]:
    """Walk csv_dir, parse and load every recognised trial."""
    trials = []
    for csv_path in sorted(csv_dir.rglob("*.csv")):
        meta = parse_filename(csv_path)
        if meta is None:
            print(f"  [skip: unparsed filename] {csv_path.name}")
            continue
        try:
            trial = load_trial_csv(meta)
            trial["metrics"] = trial_metrics(trial, include_joints=include_joints)
            trials.append(trial)
        except Exception as e:
            print(f"  [skip: load error {e}] {csv_path.name}")
    return trials


TRIAL_LEVEL_METRICS = (
    "rmse_trial",
    "mae_trial",
    "max_err_trial",
    "iae_trial",
    "itae_trial",
    "final_pos_err_trial",
    "vel_rmse_trial",
    "peak_vel_err_trial",
    "control_effort_rms",
)


def trials_to_long_df(trials: list[dict]) -> pd.DataFrame:
    """Flatten the trial list into a long-format DataFrame for summary
    + statistics. One row per trial, columns include all trial-level
    metrics + per-joint RMSE (for finer breakdown)."""
    rows = []
    for trial in trials:
        m = trial["metrics"]
        meta = trial["meta"]
        row = {
            "payload_kg": meta.payload_kg,
            "baseline": meta.baseline,
            "trial_index": meta.trial_index,
        }
        for k in TRIAL_LEVEL_METRICS:
            row[k] = m[k]
        for i, j in enumerate(JOINTS):
            row[f"rmse_{j}"] = m["rmse_per_joint"][i]
        rows.append(row)
    return pd.DataFrame(rows)


def per_cell_summary(trials: list[dict]) -> pd.DataFrame:
    """Group trials by (payload, baseline) and report median of every
    trial-level metric."""
    df = trials_to_long_df(trials)
    if df.empty:
        return df
    agg = {"n_trials": ("rmse_trial", "size")}
    for k in TRIAL_LEVEL_METRICS:
        agg[f"{k.replace('_trial', '')}_med"] = (k, "median")
    # Per-joint RMSE median
    for j in JOINTS:
        agg[f"rmse_{j}_med"] = (f"rmse_{j}", "median")
    return df.groupby(["payload_kg", "baseline"]).agg(**agg).reset_index()


# ---------- Statistics: Wilcoxon + Cliff's delta ----------

def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    """Cliff's delta: P(X>Y) - P(X<Y), in [-1, 1].
    Negative → x < y (x's controller is better if RMSE is lower)."""
    n = len(x)
    m = len(y)
    if n == 0 or m == 0:
        return float("nan")
    gt = (x[:, None] > y[None, :]).sum()
    lt = (x[:, None] < y[None, :]).sum()
    return float((gt - lt) / (n * m))


def cross_baseline_stats(trials: list[dict],
                          ref_baseline: str = "C1_implicitid",
                          metric: str = "rmse_trial") -> pd.DataFrame:
    """Pooled cross-baseline comparison (all payloads) on one metric.

    For each non-reference baseline, compares its trial-level metric to
    `ref_baseline`'s on the same (payload, trial_index) pair.

    `metric` must be one of `TRIAL_LEVEL_METRICS`.

    Returns DataFrame with median ratio, Wilcoxon p-value, Cliff's delta.
    """
    from scipy.stats import wilcoxon

    df_long = trials_to_long_df(trials)
    if df_long.empty:
        return pd.DataFrame()
    if metric not in df_long.columns:
        raise KeyError(f"Unknown metric '{metric}'; expected one of {TRIAL_LEVEL_METRICS}")

    ref_df = df_long[df_long["baseline"] == ref_baseline]
    rows = []
    for baseline in sorted(df_long["baseline"].unique()):
        if baseline == ref_baseline:
            continue
        cmp_df = df_long[df_long["baseline"] == baseline]
        merged = cmp_df.merge(
            ref_df[["payload_kg", "trial_index", metric]],
            on=["payload_kg", "trial_index"],
            suffixes=("_other", "_ref"),
        )
        if merged.empty:
            continue
        x = merged[f"{metric}_other"].to_numpy()
        y = merged[f"{metric}_ref"].to_numpy()
        # Avoid /0 in ratio if reference has near-zero values
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio_arr = np.where(np.abs(y) > 1e-12, x / y, np.nan)
        ratio = float(np.nanmedian(ratio_arr))
        try:
            _, pval = wilcoxon(x, y, zero_method="zsplit", alternative="two-sided")
        except ValueError:
            pval = float("nan")
        rows.append({
            "metric": metric,
            "vs_baseline": baseline,
            "n_pairs": len(merged),
            "median_ratio": ratio,
            "wilcoxon_p": pval,
            "cliffs_delta": cliffs_delta(x, y),
        })
    return pd.DataFrame(rows)


def all_metric_stats(trials: list[dict],
                      ref_baseline: str = "C1_implicitid") -> pd.DataFrame:
    """Run cross_baseline_stats for every trial-level metric, stacked."""
    parts = []
    for metric in TRIAL_LEVEL_METRICS:
        df = cross_baseline_stats(trials, ref_baseline, metric)
        if not df.empty:
            parts.append(df)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


# ---------- Top-level entry ----------

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("csv_dir", type=Path,
                   help="Directory containing closed-loop trial CSVs")
    p.add_argument("-o", "--output", type=Path, default=None,
                   help="Optional NPZ to save aggregate summary")
    p.add_argument("--ref_baseline", default="C1_implicitid",
                   help="Reference baseline for paired stats (default C1_implicitid)")
    p.add_argument("--exclude_joints", default="",
                   help="Comma-separated joint names to exclude from "
                        "trial-level scalars (e.g. 'b' for wrist). "
                        "Per-joint arrays still computed for all joints.")
    p.add_argument("--ref_spec", type=Path, default=None,
                   help="reference_spec.json of the reference these trials tracked "
                        "(default: May-2026 8 s A->B). Use one analyzer run per reference folder.")
    p.add_argument("--use_logged_ref", action="store_true",
                   help="Compute error against the ref_alpha_axis_* columns logged by the "
                        "controller instead of the analytic quintic (robust to any reference).")
    args = p.parse_args()

    global REF_KW, USE_LOGGED_REF
    if args.ref_spec is not None:
        REF_KW = load_ref_spec(args.ref_spec)
        print(f"Reference: {args.ref_spec}  T_seg={REF_KW['T_seg']}s  q_A={REF_KW['q_A'].tolist()}  q_B={REF_KW['q_B'].tolist()}")
    USE_LOGGED_REF = bool(args.use_logged_ref)
    if USE_LOGGED_REF:
        print("Reference: logged ref_alpha_axis_* columns (whole trial as transition mask)")

    excl = [j.strip() for j in args.exclude_joints.split(",") if j.strip()]
    bad = [j for j in excl if j not in JOINTS]
    if bad:
        raise SystemExit(f"Unknown joints in --exclude_joints: {bad}; expected subset of {JOINTS}")
    include_idx = tuple(i for i, j in enumerate(JOINTS) if j not in excl)
    if excl:
        kept = ",".join(JOINTS[i] for i in include_idx)
        print(f"Trial-level scalars use joints: {kept} (excluded: {','.join(excl)})")

    print(f"Scanning {args.csv_dir} ...")
    trials = collect_trials(args.csv_dir, include_joints=include_idx)
    print(f"Loaded {len(trials)} trials.")
    if not trials:
        return

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 220)

    summary = per_cell_summary(trials)
    long_df = trials_to_long_df(trials)

    # ---- Per-cell, primary metrics on a tight table ----
    print("\n=== Per-cell summary (median over trials) — primary metrics ===")
    primary_cols = ["payload_kg", "baseline", "n_trials",
                    "rmse_med", "mae_med", "max_err_med",
                    "final_pos_err_med", "control_effort_rms_med"]
    print(summary[primary_cols].to_string(
        index=False, float_format=lambda x: f"{x:.4f}"))

    # ---- Per-cell, integral / velocity metrics ----
    print("\n=== Per-cell summary — integral & velocity metrics ===")
    extra_cols = ["payload_kg", "baseline",
                  "iae_med", "itae_med",
                  "vel_rmse_med", "peak_vel_err_med"]
    print(summary[extra_cols].to_string(
        index=False, float_format=lambda x: f"{x:.4f}"))

    # ---- Per-joint RMSE breakdown ----
    print("\n=== Per-cell per-joint RMSE (median) ===")
    joint_cols = ["payload_kg", "baseline"] + [f"rmse_{j}_med" for j in JOINTS]
    print(summary[joint_cols].to_string(
        index=False, float_format=lambda x: f"{x:.4f}"))

    # ---- Cross-baseline statistics on every trial-level metric ----
    print(f"\n=== Cross-baseline pooled stats vs {args.ref_baseline} ===")
    stats = all_metric_stats(trials, ref_baseline=args.ref_baseline)
    if not stats.empty:
        print(stats.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    else:
        print(f"(no non-{args.ref_baseline} trials parsed)")

    if args.output is not None:
        out = {
            "payload_kg": summary["payload_kg"].to_numpy(),
            "baseline": summary["baseline"].to_numpy(),
            "n_trials": summary["n_trials"].to_numpy(),
        }
        for col in summary.columns:
            if col.endswith("_med"):
                out[col] = summary[col].to_numpy()
        if not stats.empty:
            out["stats_metric"] = stats["metric"].to_numpy()
            out["stats_vs"] = stats["vs_baseline"].to_numpy()
            out["stats_median_ratio"] = stats["median_ratio"].to_numpy()
            out["stats_p"] = stats["wilcoxon_p"].to_numpy()
            out["stats_cliffs"] = stats["cliffs_delta"].to_numpy()
        np.savez(args.output, **out)
        print(f"\nSaved summary → {args.output}")


if __name__ == "__main__":
    main()
