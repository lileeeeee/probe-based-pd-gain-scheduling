# Hardware Closed-Loop Analysis

Code for analysing closed-loop hardware trials per the protocol in
`paper_drafts/hardware_closedloop_protocol.md`.

## Files

- `reference_quintic.py` — quintic 5th-order polynomial reference
  trajectory generator. Single source of truth for the reference
  schedule (default: 8 s single quintic q_A → q_B at 25 Hz, H=200).
  Run as a script for self-test.
- `generate_reference_csv.py` — exports the reference as
  `reference_dense_25hz.csv` + `reference_waypoints.csv` +
  `reference_spec.json` to `data/hardware_closedloop_ref/`,
  mirroring the format of `data/hardware_probe/`.
- `analyze_rmse.py` — main entry. Reads CSV trial logs, computes a
  battery of trial-level metrics, runs pooled cross-baseline
  statistics.
- `test_pipeline.py` — synthetic end-to-end test. Run before any real
  data to verify nothing's broken.

## Metrics computed per trial

For each trial, `analyze_rmse.trial_metrics` returns:

| Group | Metric | Description |
|---|---|---|
| Position error | `rmse_per_joint`, `rmse_trial` | RMSE on quintic-segment mask |
| | `mae_per_joint`, `mae_trial` | MAE on transition mask |
| | `max_err_per_joint`, `max_err_trial` | max abs error over full trial |
| | `iae_per_joint`, `iae_trial` | ∫\|e(t)\|dt on transition mask |
| | `itae_per_joint`, `itae_trial` | ∫t·\|e(t)\|dt (later errors weighted more) |
| | `final_pos_err_per_joint`, `final_pos_err_trial` | \|q[T]−q_ref[T]\| (settling quality) |
| Velocity | `vel_rmse_per_joint`, `vel_rmse_trial` | RMSE on q̇−q̇_ref (transition mask) |
| | `peak_vel_err_per_joint`, `peak_vel_err_trial` | max \|q̇−q̇_ref\| |
| Control | `control_effort_per_joint`, `control_effort_rms` | RMS commanded torque |
| | `control_peak_per_joint` | max \|τ_cmd\| per joint over full trial |

`*_per_joint` arrays have shape (4,) for joints (e, d, c, b).
`*_trial` scalars are the mean across joints.

`cross_baseline_stats(metric=...)` runs paired Wilcoxon / Cliff's delta
on any of the trial-level scalars; `all_metric_stats` runs all of them
in one go.

## Workflow

### 0. Sanity-test the pipeline

```bash
cd <project_root>
python experiments/hardware_closedloop/test_pipeline.py
```

Generates 27 synthetic trials with controlled noise scales (so
ImplicitID < Partial-cheat < Fixed by design) and verifies the
analysis correctly ranks them. Should print `✓ Pipeline end-to-end
test passed.`

### 1. Generate reference trajectory

```bash
python experiments/hardware_closedloop/generate_reference_csv.py
```

Writes `data/hardware_closedloop_ref/`: dense CSV at 25 Hz, waypoints
CSV, and spec JSON. The hardware controller reads the dense CSV (or
re-evaluates the quintic on the fly) at runtime.

### 2. Run hardware trials

Per protocol §4: 3–4 baselines × 3 payloads × 3 trials. Each trial
logs a CSV with the same column convention as the open-loop probe
data in `data/real_robot/`:

```
wall_time_sec, sim_time_sec, replay_time_sec, profile, pass_index,
q_alpha_axis_e, q_alpha_axis_d, q_alpha_axis_c, q_alpha_axis_b,
dq_alpha_axis_e, dq_alpha_axis_d, dq_alpha_axis_c, dq_alpha_axis_b,
cmd_tau_axis_e, cmd_tau_axis_d, cmd_tau_axis_c, cmd_tau_axis_b,
payload_mass, ... (other columns ignored)
```

The reference trajectory is regenerated in software, so `q_ref`
columns are not required.

Filename convention recognised by `parse_filename`:

```
<timestamp>_<scenario>_payload_<XpYZkg>_<baseline_tag>_pass<N>.csv
```

`<baseline_tag>` is matched case-insensitively against:

| Tag | Resolves to |
|---|---|
| `implicitid`, `implicit_id`, `c1` | `C1_implicitid` |
| `partial_cheat`, `partial1f`, `partial_1f`, `c2` | `C2_partial_cheat` |
| `fixed`, `oracle0kg`, `c3` | `C3_fixed` |
| `handtuned`, `hand_tuned`, `manual`, `c4` | `C4_handtuned` |

### 3. Analyse

```bash
python experiments/hardware_closedloop/analyze_rmse.py \
    /path/to/closedloop_csv_dir \
    -o data/hardware_closedloop_summary.npz \
    --ref_baseline C1_implicitid
```

Prints three per-cell tables (primary metrics, integral/velocity
metrics, per-joint RMSE) and a cross-baseline pooled stats table
covering all 9 trial-level metrics.

### 4. Interpret per protocol §0 hypotheses

Match the printed tables to the pre-committed hypotheses in
`paper_drafts/hardware_closedloop_protocol.md` §0. **Do not change
metric definitions or test conventions after seeing the data.**

## Adapting

- Different CSV column names → edit `load_trial_csv()` in
  `analyze_rmse.py`.
- Different filename convention → edit `parse_filename()`.
- Different trial schedule (waypoints / segment time / dwells / return
  cycles) → edit defaults in `reference_quintic.py`. Both
  `make_reference()` and `trial_duration()` will pick up the change.
