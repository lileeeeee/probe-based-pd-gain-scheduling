# Reproducing the reported numbers

## Splits

Use `data/split_indices.npz`, which stores the `train_idx` / `test_idx` /
`remain_idx` actually used by each archived run, keyed `env/seed/baseline/field`.
Regenerating a split from a seed does not reproduce them.

## Metric conventions

- Gain error is z-MSE: the squared error of per-dimension z-scored gains, each
  dimension standardized by its training-set standard deviation, averaged over
  the 2J dimensions.
- Table values are computed over the full 100-point test pool. The
  `gain_mse_test` scalar stored inside `predictions.npz` is a legacy 50-point
  figure and is not what the paper reports.
- The training loss zeroes degenerate gain dimensions (raw std < 0.05, or
  >= 80 % of training values below 0.1); on Alpha 5 those are Kd on joints 1
  and 4. The reported metric still averages over all dimensions, so those two
  contribute error the model was never trained to reduce.

## Hardware

Gains entered on the robot are the simulation gains times a per-joint scaling
calibrated once at zero payload; both are in `data/`. Reference trajectories,
run orders, and the analysis command line are in
`data/hardware_closedloop_ref_v2/RUN_SHEET.md`.
