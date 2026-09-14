# Closed-loop supplement -- run sheet (2026-09-05), FINAL

Answers R2-C9 (several references / velocities / initial configs, randomized order, trial-level CIs)
and R2-C10 (probe reproducibility + safety numbers, calibration duration). Same CSV schema as 2026-05-01.

## What to run, per payload (mount once; payloads 0.00, 0.38, 0.57 kg)
1. PROBE (1x): reset to q_init = [3.1, 0.7, 0.4, 2.1], gravity comp OFF, play probe/probe_dense_25hz.csv (20 s, 25 blocks x 0.8 s,
   seed 2025, limits e/d/c/b = 1.5/1.0/1.0/0.54 N m). Log q, dq, cmd_tau -> probe_logs/payload_<p>kg_<ts>.csv.
   Not used for gains today; it is the probe-repeatability + safety record for R2-C10.
2. CLOSED LOOP: enter the gains from GAINS_TO_ENTER.csv for this payload (C1/C2/C3), then run trial_order_payload_<p>kg.txt
   in order: 39 trials = 4 rounds x (ref4s_AB, ref8s_BA, ref12s_AB) x (C1,C2,C3) + 3 anchor trials of May's 8 s A->B in round 1.
   Before every trial PD-lock 5 s at the reference's start pose (ref8s_BA starts at q_B = [3.8, 1.0, 0.7, 2.4]).
   Stop only at the end of a round (3 full rounds already make a complete design).
3. notes.txt: one line per trial (saturation buzz, unclean reset, e-stop, anything odd). Record actual payload mass.

## CRITICAL: start-pose control (found 2026-09-06 in the May logs)
In the May session the three baselines did NOT start from the same pose at 0.57 and 0.76 kg:
the wrist (axis_b) started at 2.02-2.05 rad for Fixed but 2.30-2.33 rad for ImplicitID/Explicit,
while the reference starts at 2.10 and ends at 2.40. Because the wrist gains are very weak
(Kp_b ~1.7-2.2, Kd_b ~0.001) the arm barely moves, so a trial that starts at 2.33 is already 77%
of the way to the target and scores a much lower wrist RMSE for free. The wrist then accounts for
75% (0.57 kg) and 69% (0.76 kg) of the ImplicitID-vs-Fixed gap. The offset appears only after 17:20
in the session, i.e. the reset procedure changed partway through.

Today, for EVERY trial:
1. PD-lock 5 s at the reference's own start pose (ref4s_AB / ref12s_AB / ref8s_AB_anchor: q_A = [3.10, 0.70, 0.40, 2.10];
   ref8s_BA: q_B = [3.80, 1.00, 0.70, 2.40]).
2. Before releasing the lock, CHECK the logged q against that pose. If any joint is off by more than
   0.05 rad, re-lock and wait; do not start the trial. Note in notes.txt whenever a re-lock was needed.
3. Use the SAME reset procedure for all three baselines. Do not change it mid-session.
The logs record q at t=0, so this is verifiable afterwards; the analyzer will report per-trial start pose.

## Before 0 kg
- Safety pre-test: one ref4s_AB trial with C3 at 0 kg. If a joint saturates continuously or oscillates -> fall back to T_seg = 6 s and say so.
  (Reference peak speeds are 1/2-1/6 of what the arm already reached in May's tracking and in the probes; risk is low.)
- Gains are May's: sim gains x per-joint scaling Kp (0.05, 0.0667, 0.04, 0.2), Kd 0.2 -- see GAINS_TO_ENTER.csv. Do NOT re-derive today.
  Calibration duration for the paper: from recollection of the May session, ~10-20 min, 3-4 closed-loop trials at 0 kg against a hand-tuned reference.

## Also today (no arm time)
- Copy the ros2_control yaml with joint limits / torque limits / e-stop thresholds into data/hardware_probe/ (R2-C10).
- Inference time: DONE, output/r1p10_inference_time_deployment_mac.json (4.41 ms single encoder, 22.8 ms 5-seed readout, Mac CPU).

## File layout expected by the analyzer
<ref folder>/C1_0_38/<ts>_..._payload_0p38kg_..._passN.csv   (baseline tag in dir or file name; payload as 0p38kg; passN = round)

## Analysis (tonight), one run per reference folder
python experiments/hardware_closedloop/analyze_rmse.py <logs>/ref4s_AB        --ref_spec data/hardware_closedloop_ref_v2/ref4s_AB/reference_spec.json
python experiments/hardware_closedloop/analyze_rmse.py <logs>/ref8s_BA        --ref_spec data/hardware_closedloop_ref_v2/ref8s_BA/reference_spec.json
python experiments/hardware_closedloop/analyze_rmse.py <logs>/ref12s_AB       --ref_spec data/hardware_closedloop_ref_v2/ref12s_AB/reference_spec.json
python experiments/hardware_closedloop/analyze_rmse.py <logs>/ref8s_AB_anchor                       # May's reference = analyzer default
Cross-check any folder with --use_logged_ref (no --ref_spec needed).

## Time
117 closed-loop trials x ~1.5 min = 2.9 h + 3 probes (5 min) + pre-test + 3 mounts  ->  ~3.5-4 h.
