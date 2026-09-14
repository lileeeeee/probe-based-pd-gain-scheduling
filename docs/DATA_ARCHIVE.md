# Separate data archive

These are too large to version here. Upload as one archive (Zenodo or similar),
cite its DOI from the paper, and keep the layout below so the scripts in this
repository find the files by their documented relative paths.

## Datasets (BO-labelled conditions)

| File | Size | Contents |
|---|---|---|
| `data/dataset_casadi.npz` | 47 MB | Alpha 5, F=2, 500 conditions: `phys_factors` (500,2), `gains` (500,12), probe responses for four waveforms, per-joint torque limits, cost. |
| `data/dataset_mujoco.npz` | 510 MB | Franka Panda, F=2, 500 conditions, 7 joints, `gains` (500,21) with `Ki = 0.2*Kp`. |
| `data/dataset_3d_envB_full.npz` | 299 MB | Alpha 5, F=3, 3000 Sobol points. Used for Sections 5.2, 5.3 and the probing analysis. |

Each carries the BO-tuned gain vector per condition, the physics factors, and
the probe response that the encoder consumes. Labels come from batched Bayesian
optimization of the task-averaged closed-loop cost; Appendix A.1 of the paper
records the rounds and neighbour counts.

## Trained-run artifacts

| Directory | Size | Runs |
|---|---|---|
| `output/casadi2d_iid/` | 60 MB | 5 seeds x 4 baselines, Alpha 5 F=2 |
| `output/mujoco2d_iid/` | 60 MB | 5 seeds x 4 baselines, MuJoCo F=2 |
| `output/f3_casadi_iid_valfrac/` | 134 MB | 5 seeds x 9 baselines, Alpha 5 F=3 |
| `output/f3_casadi_iid_qpos_valfrac/` | 68 MB | 5 seeds x 5 baselines, position-channel encoder (deployed on hardware) |

Each run holds `predictions.npz` (predicted and true gains, the split indices,
the physics factors), `train_config.json`, `summary.json`, `model.pt`, and the
training log. The reported table values are recomputed from `predictions.npz`;
note that the `gain_mse_test` scalar inside it is a legacy 50-point figure and
is not what the paper reports (the paper uses the full 100-point test pool).

## Hardware logs

| Directory | Size | Contents |
|---|---|---|
| `data/real_robot/` | 40 MB | Open-loop probe trials, 7 payloads x 3 passes (Section 6.2). |
| `data/real_robot_25hz/` | 8 MB | The same trials resampled to 25 Hz, as fed to the encoder. |
| `data/real_robot_closed_loop/` | 7 MB | Closed-loop trials behind Table 7 (2026-05-01). |
| `data/hardware_closedloop_sept2026/` | 501 MB compressed | Closed-loop trials behind Table 8 (2026-09), three payloads, four references, randomized cell order, with a per-trial index and a README documenting reset quality and two known hardware effects. |

## Not included

Intermediate and superseded material: `dataset_*_raw.npz`, out-of-distribution
cost analyses not used in the paper, earlier architecture-comparison runs, and
the pre-fix hardware session of 2026-09-06 (discarded; its README explains why).
