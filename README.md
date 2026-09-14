# Probe-Based Proportional Derivative Gain Scheduling — code and data

Code, probe sequences, data splits, and evaluation scripts for the Robotica
submission *Probe-Based Proportional Derivative Gain Scheduling: Factor-List
Aliasing and Deployment Cost Asymmetry* (ROB-2026-0352).

This repository holds the implementation, the probe sequences, and the data
splits. The BO-labelled datasets and the raw hardware logs are not included
here; they are available from the authors on request.

## Layout

    src/                    training pipeline, models, dataset loaders, splits
    experiments/            scripts that produce specific paper numbers
      hardware_closedloop/  reference generation and trial analysis for Section 6
    data/                   probe sequences, split indices, task set, small results
    docs/                   what lives in the separate data archive, and how to reproduce

## Which script produces which number

| Paper location | Script |
|---|---|
| Sec. 5.2, Table 2 (factor sensitivity) | `experiments/anova_pd_effective.py` |
| Sec. 5.3, Table 3 and Eq. (3) | `experiments/r2c8_vbin_bootstrap_ci.py`, `experiments/vhid_factorlist_pd.py` |
| Sec. 5.3, z-normalized bound check | `experiments/r2c3_znorm_bound_check.py` |
| Sec. 5.3, first-order / total-effect variance shares | `experiments/r2c7_sobol_recompute.py` |
| Sec. 5.4, Table 4 (cost asymmetry significance) | `experiments/cost_asymmetry_significance.py` |
| Sec. 6, Tables 7 and 8 (hardware) | `experiments/hardware_closedloop/analyze_rmse.py` |
| Sec. 6, hardware gains | `experiments/hardware_closedloop/compute_baseline_gains.py` |

## data/

| File | What it is |
|---|---|
| `split_indices.npz` | Train / test / remain indices for all 110 archived runs, keyed `env/seed/baseline/field`. Use these rather than regenerating a split from a seed. |
| `train_configs.json` | The training configuration of each of those runs. See `docs/REPRODUCING.md` for the gain parameterization. |
| `hardware_probe/` | The open-loop probe: `probe_dense_25hz.csv` (25 Hz torque per joint), `probe_blocks.csv`, `probe_spec.json` (25 blocks of 0.8 s, seed 2025, per-joint limits 1.5 / 1.0 / 1.0 / 0.54 N·m). One fixed sequence, shared by the simulation training data and every hardware trial. |
| `hardware_closedloop_ref/`, `hardware_closedloop_ref_v2/` | Closed-loop reference trajectories (Sections 6.3 and 6.4): dense 25 Hz setpoints, waypoints, and specs. `_v2` also carries the gains entered on the robot (`GAINS_TO_ENTER.csv`, and `GAINS_BEFORE_AFTER_CALIBRATION.csv` showing them before and after the per-joint scaling), the pre-registered randomized cell orders, and the run sheet. |
| `tasks_30pairs_4dof.npz` | The 30 start/goal pairs behind every closed-loop cost in Section 5.4. **Required**: no generation script exists; deleting it causes the evaluator to silently generate 30 different tasks. |
| `hardware_closedloop_gains.npz` | Simulation gains per baseline and payload, before the per-joint scaling. |
| `hardware_closedloop_gain_scaling_may2026.json` | The per-joint scaling actually applied on hardware (Kp x 0.05 / 0.0667 / 0.04 / 0.2, Kd x 0.2), recovered by least squares from the logged torques. No original record of these factors exists. |
| `casadi2d_iid_cost_30tasks.npz` | Pooled 30-task closed-loop costs behind Table 4. |
| `r2c7_sobol_recompute.json`, `r1p10_inference_time_deployment_mac.json` | Variance shares per gain dimension; deployment-side inference timing. |

## Environment

Python 3.9+, `numpy`, `scipy`, `torch` (2.5.1 used for the reported runs),
`scikit-learn`, `pandas`, `matplotlib`. The MuJoCo and CasADi simulators are
needed only to regenerate datasets, not to reproduce the reported numbers from
the archived artifacts.

## License

MIT, see `LICENSE`.
