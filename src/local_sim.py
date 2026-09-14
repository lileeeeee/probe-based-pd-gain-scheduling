"""Local CasADi sim replay utilities.

Loads `arm_rollout_sim_u0.casadi` (nightly-blazing2 CasADi required on Mac;
`pip install https://github.com/casadi/casadi/releases/download/nightly-blazing2/
casadi-3.7.2.dev%2Bblazing2-cp39-none-macosx_11_0_arm64.whl`)
and provides:

  - replay_openloop(m_p, fv_mult, fc_mult, torque_seq, q0, qd0)
      → bit-exact reproduction of dataset trajectories; use with probe signals.

  - closed_loop_pid(m_p, fv_mult, fc_mult, gain_vec_12, q_target, horizon, q0, qd0)
      → closed-loop PID rollout; use for Phase 2 pre-flight sanity check on
      predicted gains before sending to hardware.

  - sanity_check_gains(gain_vec_12, payload_mass, q_target, q0=None)
      → Stage 3 gate: converged? overshoot? tracking-error quantile? Returns
      dict suitable for "ship / don't ship" decision.

Parameter-injection formula (matches turbo_casadi_gpu_multifactor.py:111-135):

    params_81 = excit_params_template.copy()
    params_81[62]    = m_p
    params_81[40:44] = fv_mult * [2.396, 2.236, 0.820, 0.357]        # = override
    params_81[44:48] = fc_mult * [0.15, 0.10, 0.10, 0.054]           # 0.1×tau_max

Only the closed_loop_pid rollout path uses the 200-step mapaccum
N_HORIZON=200 times (taking just step 0 each call) — ~2 s per 200-step
trajectory on MPS/CPU. Open-loop path is one 200-step call; batches
trivially if needed.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np

# ------------------------------------------------------------------
# Constants (locked to dataset_3d_envB_full.npz excit_* generation)
# ------------------------------------------------------------------

_CASADI_FN_PATH = Path.home() / "Downloads" / "arm_rollout_sim_u0.casadi"
_DATASET_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "dataset_3d_envB_full.npz"
)

_DT = 0.03999999910593033  # excit_step_dt in dataset
_BAUMGARTE_ALPHA = 200.0   # excit_baumgarte_alpha in dataset

_BASELINE_FV = np.array(
    [2.39569756, 2.23596482, 0.819671021, 0.357249665], dtype=np.float64
)
_BASELINE_FC_BETA = np.array(
    [0.15, 0.10, 0.10, 0.054], dtype=np.float64
)  # = 0.1 * tau_max_per_joint

_N_HORIZON = 200  # fixed in mapaccum_step_u_fn signature

# Lazily-loaded; avoid casadi import at module-import time (saves ~1s)
_fn = None
_params_template = None


def _ensure_loaded():
    global _fn, _params_template
    if _fn is None:
        import casadi as ca  # noqa: E402
        if not _CASADI_FN_PATH.exists():
            raise FileNotFoundError(
                f"Missing CasADi function file: {_CASADI_FN_PATH}. "
                "Request from the hardware author."
            )
        _fn = ca.Function.load(str(_CASADI_FN_PATH))
    if _params_template is None:
        if not _DATASET_PATH.exists():
            raise FileNotFoundError(
                f"Missing F=3 dataset: {_DATASET_PATH}"
            )
        with np.load(_DATASET_PATH, allow_pickle=True) as d:
            _params_template = d["excit_params_template"].astype(np.float64)
    return _fn, _params_template


def build_params(m_p: float, fv_mult: float, fc_mult: float) -> np.ndarray:
    """Build 81-dim params vector for a physics point.

    Matches turbo_casadi_gpu_multifactor.py:111-135 injection rules.
    """
    _, template = _ensure_loaded()
    p = template.copy()
    p[62] = float(m_p)
    p[40:44] = float(fv_mult) * _BASELINE_FV
    p[44:48] = float(fc_mult) * _BASELINE_FC_BETA
    return p


# ------------------------------------------------------------------
# Open-loop: 200-step torque sequence → state trajectory (bit-exact)
# ------------------------------------------------------------------

def replay_openloop(
    m_p: float,
    fv_mult: float,
    fc_mult: float,
    torque_seq: np.ndarray,
    q0: np.ndarray,
    qd0: Optional[np.ndarray] = None,
    dt: float = _DT,
    baumgarte_alpha: float = _BAUMGARTE_ALPHA,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Replay an open-loop torque sequence through the CasADi sim.

    Args:
        m_p: payload mass
        fv_mult: viscous friction multiplier
        fc_mult: Coulomb friction multiplier
        torque_seq: (4, N_HORIZON)=(4, 200) torque to apply per step
        q0: (4,) initial joint positions
        qd0: (4,) initial joint velocities (defaults to zeros)
        dt: step size (default 0.04 s)
        baumgarte_alpha: Baumgarte stabilization gain (default 200)

    Returns:
        q_traj: (4, 200) joint position trajectory
        qd_traj: (4, 200) joint velocity trajectory
        tau_applied: (4, 200) actual torque after safety clamp
    """
    fn, _ = _ensure_loaded()
    if torque_seq.shape != (4, _N_HORIZON):
        raise ValueError(
            f"torque_seq must be (4, {_N_HORIZON}); got {torque_seq.shape}"
        )
    if qd0 is None:
        qd0 = np.zeros(4, dtype=np.float64)
    q0 = np.asarray(q0, dtype=np.float64).ravel()
    qd0 = np.asarray(qd0, dtype=np.float64).ravel()

    # State layout: first 12-dim init. Extra 4 slots (beyond q, qd) are zero
    # — matches dataset convention excit_x0 = [q(4), qd(4), zeros(4)].
    state0 = np.concatenate([q0, qd0, np.zeros(4)])[:, None]
    u = np.asarray(torque_seq, dtype=np.float64)
    dt_seq = np.full((1, _N_HORIZON), dt)
    alpha_seq = np.full((1, _N_HORIZON), baumgarte_alpha)
    params = build_params(m_p, fv_mult, fc_mult)
    params_seq = np.tile(params.reshape(81, 1), (1, _N_HORIZON))

    out = fn(state0, u, dt_seq, params_seq, alpha_seq)
    traj = np.asarray(out[0])   # (12, 200)
    tau_applied = np.asarray(out[1])  # (4, 200)
    return traj[:4], traj[4:8], tau_applied


# ------------------------------------------------------------------
# Closed-loop: compute PID τ at each step, sim 1 step, advance
# ------------------------------------------------------------------

def closed_loop_pid(
    m_p: float,
    fv_mult: float,
    fc_mult: float,
    gains: np.ndarray,
    q_target: np.ndarray,
    horizon: int = _N_HORIZON,
    q0: Optional[np.ndarray] = None,
    qd0: Optional[np.ndarray] = None,
    dt: float = _DT,
    baumgarte_alpha: float = _BAUMGARTE_ALPHA,
    tau_limits: Optional[np.ndarray] = None,
) -> dict:
    """Closed-loop PID step-response at a given payload.

    Slow (~2 s per 200-step trajectory) because mapaccum is fixed at 200 steps
    and we extract only step 0 per loop iteration.

    Args:
        m_p, fv_mult, fc_mult: physics
        gains: (12,) vector laid out as [Kp*4, Ki*4, Kd*4]
        q_target: (4,) target joint configuration to track
        horizon: N closed-loop steps (default 200 = 8 s at dt=0.04)
        q0, qd0: initial state (default q0=q_target+small offset, qd0=0)
        tau_limits: (4,) abs torque limit per joint (default [1.5,1.0,1.0,0.54])

    Returns dict with:
        q_traj (4,H), qd_traj (4,H), tau_traj (4,H),
        tracking_err (H,) — ||q - q_target||_2 per step,
        final_err (4,) — last-step joint error,
        cost — integrated tracking + effort (matches sim BO cost modulo scaling)
    """
    fn, _ = _ensure_loaded()
    gains = np.asarray(gains, dtype=np.float64).ravel()
    if gains.size != 12:
        raise ValueError(f"gains must be 12-dim (Kp×4, Ki×4, Kd×4); got {gains.size}")
    Kp, Ki, Kd = gains[:4], gains[4:8], gains[8:12]

    q_target = np.asarray(q_target, dtype=np.float64).ravel()
    if q_target.size != 4:
        raise ValueError(f"q_target must be 4-dim; got {q_target.size}")

    if q0 is None:
        q0 = q_target.copy()
    if qd0 is None:
        qd0 = np.zeros(4, dtype=np.float64)
    q0 = np.asarray(q0, dtype=np.float64).ravel()
    qd0 = np.asarray(qd0, dtype=np.float64).ravel()

    if tau_limits is None:
        tau_limits = np.array([1.5, 1.0, 1.0, 0.54], dtype=np.float64)
    tau_limits = np.asarray(tau_limits, dtype=np.float64).ravel()

    # Pre-build static sim arrays (reused every inner call — the sim runs
    # a 200-step rollout each time but we only use step 0).
    dt_seq = np.full((1, _N_HORIZON), dt)
    alpha_seq = np.full((1, _N_HORIZON), baumgarte_alpha)
    params = build_params(m_p, fv_mult, fc_mult)
    params_seq = np.tile(params.reshape(81, 1), (1, _N_HORIZON))

    state = np.concatenate([q0, qd0, np.zeros(4)])[:, None]

    q_traj = np.zeros((4, horizon))
    qd_traj = np.zeros((4, horizon))
    tau_traj = np.zeros((4, horizon))
    err_int = np.zeros(4)

    for t in range(horizon):
        q = state[:4, 0]
        qd = state[4:8, 0]
        err = q_target - q
        err_int += err * dt
        tau_t = Kp * err + Ki * err_int - Kd * qd
        tau_t = np.clip(tau_t, -tau_limits, tau_limits)

        # Broadcast tau_t across the 200-step horizon (we only use step 0;
        # the sim will integrate internally but we discard steps 1..199).
        tau_seq = np.tile(tau_t.reshape(4, 1), (1, _N_HORIZON))

        out = fn(state, tau_seq, dt_seq, params_seq, alpha_seq)
        traj_200 = np.asarray(out[0])
        next_state = traj_200[:, 0:1]  # only step 0

        q_traj[:, t] = next_state[:4, 0]
        qd_traj[:, t] = next_state[4:8, 0]
        tau_traj[:, t] = tau_t
        state = next_state

    # Cost matches sim BO cost_mode="mul" at γ=0.1:
    #   J = (1/T) Σ ||q_t - q_target||² · (1 + 0.1 · (1/T) Σ ||τ_t||²)
    mean_track = float(((q_traj - q_target[:, None]) ** 2).mean(axis=0).mean())
    mean_effort = float((tau_traj ** 2).mean(axis=0).mean())
    cost = mean_track * (1.0 + 0.1 * mean_effort)

    tracking_err = np.linalg.norm(q_traj - q_target[:, None], axis=0)
    final_err = q_traj[:, -1] - q_target

    return {
        "q_traj": q_traj,
        "qd_traj": qd_traj,
        "tau_traj": tau_traj,
        "tracking_err": tracking_err,
        "final_err": final_err,
        "cost": cost,
        "mean_tracking_sq": mean_track,
        "mean_effort_sq": mean_effort,
    }


# ------------------------------------------------------------------
# Sanity-check gate for Stage 3 pre-flight
# ------------------------------------------------------------------

def sanity_check_gains(
    gains: np.ndarray,
    m_p: float,
    q_target: np.ndarray,
    fv_mult: float = 1.0,
    fc_mult: float = 1.0,
    q0: Optional[np.ndarray] = None,
    horizon: int = _N_HORIZON,
    max_joint_err_rad: float = 0.5,   # > 30° = bad
    max_overshoot_rad: float = 1.0,   # > 60° overshoot = bad
    max_cost: float = 10.0,
) -> dict:
    """Phase 2 pre-flight: decide whether sim-predicted gains are safe to
    ship to the hardware. Returns dict with `ok: bool` + diagnostics."""
    res = closed_loop_pid(
        m_p=m_p, fv_mult=fv_mult, fc_mult=fc_mult,
        gains=gains, q_target=q_target, q0=q0, horizon=horizon,
    )

    final_err = res["final_err"]
    max_err_per_joint = np.abs(res["q_traj"] - q_target[:, None]).max(axis=1)
    # "Overshoot" = max excursion past target away from starting side
    if q0 is None:
        q0 = q_target.copy()
    q0 = np.asarray(q0, dtype=np.float64).ravel()
    side = np.sign(q_target - q0)[:, None]  # (4, 1) for broadcast
    overshoot_per_joint = np.maximum(
        0, side * (res["q_traj"] - q_target[:, None])
    ).max(axis=1)

    ok = (
        np.all(np.abs(final_err) < max_joint_err_rad)
        and np.all(overshoot_per_joint < max_overshoot_rad)
        and res["cost"] < max_cost
        and np.all(np.isfinite(res["q_traj"]))
    )

    return {
        "ok": bool(ok),
        "cost": res["cost"],
        "final_err": final_err.tolist(),
        "max_err_per_joint": max_err_per_joint.tolist(),
        "overshoot_per_joint": overshoot_per_joint.tolist(),
        "tracking_err_quantiles": np.quantile(
            res["tracking_err"], [0.5, 0.9, 1.0]
        ).tolist(),
    }


if __name__ == "__main__":
    # Quick self-test
    import time
    with np.load(_DATASET_PATH, allow_pickle=True) as d:
        phys = d["phys_factors"]
        gains_db = d["gains"]
        x0 = d["excit_x0"]
    # Pick a point, verify open-loop matches dataset
    idx = 322
    m_p, fv, fc = phys[idx]
    print(f"Self-test: point {idx}  m_p={m_p:.3f} fv={fv:.3f} fc={fc:.3f}")

    # Open-loop replay test
    with np.load(_DATASET_PATH, allow_pickle=True) as d:
        u = d["excit_applied_u"][idx]
        truth_q = d["excit_position"][idx, :, 1:]

    q, qd, tau_app = replay_openloop(
        m_p=m_p, fv_mult=fv, fc_mult=fc,
        torque_seq=u.astype(np.float64),
        q0=x0[:4], qd0=x0[4:8],
    )
    err = np.abs(q - truth_q).mean()
    print(f"  Open-loop mean |Δq| vs dataset: {err:.2e}  "
          f"(expected ~1e-6 if matched)")

    # Closed-loop: use BO-optimal gains from dataset, target = q_target default
    print(f"\n  Closed-loop PID pre-flight with BO-optimal gains from idx={idx}:")
    q_target = np.array([0.2, 2.1, -0.1, 0.5])  # from dataset excit_q_ref
    q0_offset = x0[:4]  # start at dataset x0
    t0 = time.time()
    report = sanity_check_gains(
        gains=gains_db[idx], m_p=m_p,
        fv_mult=fv, fc_mult=fc,
        q_target=q_target, q0=q0_offset,
    )
    print(f"  (took {time.time()-t0:.1f} s)")
    for k, v in report.items():
        if isinstance(v, list):
            v = [f"{x:.3f}" if isinstance(x, float) else x for x in v]
        print(f"    {k}: {v}")
