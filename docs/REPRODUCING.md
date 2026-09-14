# Reproducing the reported numbers

Two things about the archived runs are not evident from the configs, and both
are needed to get the paper's numbers.

## 1. The Alpha 5 training target is an effective-PD fold

Runs with `ki_ratio: 0.0` in `train_configs.json` did not simply drop the
integral gain. They trained on

    Kp_eff = Kp + dt * Ki,   dt = 0.04 s,   Ki dropped

i.e. an 8-dimensional `[Kp_eff (4), Kd (4)]` target, and expanded predictions
back to 12 dimensions with `Ki = 0` when writing `predictions.npz`. The fold is
exact rather than an approximation: the closed-loop cost kernel applies the
integral term as `Ki * e * dt` without accumulation, so `(Kp, Ki)` and
`(Kp + dt*Ki, 0)` are the same controller.

Verification, if you want to confirm it on the archived artifacts: for every
seed and every subset (train / test / remain), `true_gains_*` equals
`fold(dataset gains[idx])` to the last bit. The parameter count agrees
independently: the archived `n_params = 756168` is a `TransformerEncoderHead`
with `out_dim = 8`, not 12.

MuJoCo is different and needs no fold: its bundle declares `ki_ratio = 0.2`,
the loader drops the Ki columns to a 14-dimensional target, and the current
`src/datasets.py` reproduces the archived runs as is.

## 2. The splits cannot be regenerated, only replayed

The archived runs record `split_mode: "fixed_iid"`, which the current
`src/training.py` does not offer (`band | iid | region`). Using `iid` with the
same seed and test fraction produces a different partition. Use
`data/split_indices.npz`, which stores the actual `train_idx` / `test_idx` /
`remain_idx` of every run.

## Metric conventions

- Gain error is z-MSE: the squared error of per-dimension z-scored gains, each
  dimension standardized by its training-set standard deviation, averaged over
  the 2J dimensions.
- Table values are computed over the full 100-point test pool, not the
  50-point `gain_mse_test` scalar stored inside `predictions.npz`.
- The training loss zeroes degenerate gain dimensions (raw std < 0.05 or
  >= 80 % of training values below 0.1); on Alpha 5 that is Kd on joints 1 and 4.
  The reported metric still averages over all dimensions, so those two
  contribute error that the model was never trained to reduce.

## Hardware

Gains entered on the robot are the simulation gains times a per-joint scaling
calibrated once at zero payload; both are in `data/`. Reference trajectories,
the run orders, and the analysis command line are in
`data/hardware_closedloop_ref_v2/RUN_SHEET.md`.
