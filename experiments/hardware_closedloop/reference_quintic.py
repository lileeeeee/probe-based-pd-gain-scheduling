"""Quintic (5th-order polynomial) reference trajectory generator.

Default schedule: a single quintic segment q_A → q_B over T_seg seconds
(no dwells, no return). Quintic endpoints are zero-velocity and
zero-acceleration, so the trajectory is C² with smooth start and stop.

Used by `analyze_rmse.py` to *synthesise* the reference trajectory in
software (independent of whatever was logged by the closed-loop ROS2
node), so RMSE is computed against a single canonical reference.

Optional extensions (parameters):
  - n_return_cycles > 0: append B→A then A→B repeats (rarely needed)
  - t_dwell > 0: optional hold at endpoints (e.g., for settling-time analysis)
"""
from __future__ import annotations

import numpy as np


# ----- protocol §2 default schedule (8 s single A→B, H=200 @ 25 Hz) -----
DEFAULT_Q_A = np.array([3.10, 0.70, 0.40, 2.10])  # rad, joints (e, d, c, b)
DEFAULT_Q_B = np.array([3.80, 1.00, 0.70, 2.40])  # rad
DEFAULT_T_DWELL = 0.0    # s, optional hold at endpoints (default: none)
DEFAULT_T_SEG = 8.0      # s, duration of single quintic segment
DEFAULT_N_RETURN_CYCLES = 0  # number of B→A→B return cycles (default: none)
DEFAULT_DT = 0.04        # 25 Hz → 200 samples in 8 s


def quintic_interpolant(tau: np.ndarray) -> np.ndarray:
    """Smooth scalar interpolant s(tau) on tau in [0,1].

    s(0)=0, s(1)=1, s'(0)=s'(1)=0, s''(0)=s''(1)=0.
    Returns array same shape as tau.
    """
    tau = np.clip(tau, 0.0, 1.0)
    return 10 * tau**3 - 15 * tau**4 + 6 * tau**5


def quintic_velocity(tau: np.ndarray, T_seg: float) -> np.ndarray:
    """Time-derivative of the quintic interpolant (per second)."""
    tau = np.clip(tau, 0.0, 1.0)
    return (30 * tau**2 - 60 * tau**3 + 30 * tau**4) / T_seg


def make_reference(
    t: np.ndarray,
    q_A: np.ndarray = DEFAULT_Q_A,
    q_B: np.ndarray = DEFAULT_Q_B,
    t_dwell: float = DEFAULT_T_DWELL,
    T_seg: float = DEFAULT_T_SEG,
    n_return_cycles: int = DEFAULT_N_RETURN_CYCLES,
):
    """Generate (q_ref, qdot_ref, transition_mask) for the trial schedule.

    Default schedule (n_return_cycles=0, t_dwell=0):
        [0, T_seg]:  quintic A→B   (transition, full trial)

    With n_return_cycles=k and t_dwell=d, schedule is:
        dwell(d) @ A → quintic A→B → dwell(d) @ B
        (then k repeats of: quintic B→A → dwell @ A → quintic A→B → dwell @ B)

    Args:
        t: (T,) time stamps in seconds (relative to trial start).
        q_A, q_B: (n_joints,) waypoints.
        t_dwell: optional hold duration at endpoints.
        T_seg: quintic segment duration.
        n_return_cycles: number of B→A→B return cycles after the
            initial A→B segment.

    Returns:
        q_ref: (T, n_joints)
        qdot_ref: (T, n_joints)
        transition_mask: (T,) bool — True during quintic segments,
            False during dwells.
    """
    q_ref = np.tile(q_A[None, :], (len(t), 1))
    qdot_ref = np.zeros_like(q_ref)
    transition_mask = np.zeros(len(t), dtype=bool)

    cursor = 0.0

    def _quintic_segment(start, end, q_start, q_end):
        """Fill q_ref / qdot_ref / mask between cursor [start, start+T_seg]."""
        in_seg = (t >= start) & (t < end)
        tau = (t[in_seg] - start) / T_seg
        s = quintic_interpolant(tau)
        sdot = quintic_velocity(tau, T_seg)
        delta = q_end - q_start
        q_ref[in_seg] = q_start[None, :] + s[:, None] * delta[None, :]
        qdot_ref[in_seg] = sdot[:, None] * delta[None, :]
        transition_mask[in_seg] = True

    def _dwell(start, end, q_hold):
        in_dwell = (t >= start) & (t < end)
        q_ref[in_dwell] = q_hold[None, :]
        # qdot_ref already zero from initialisation.

    # Optional initial dwell at q_A
    if t_dwell > 0:
        _dwell(cursor, cursor + t_dwell, q_A)
        cursor += t_dwell

    # Mandatory single A → B
    _quintic_segment(cursor, cursor + T_seg, q_A, q_B)
    cursor += T_seg

    # Optional dwell at q_B and any return cycles
    for _ in range(n_return_cycles):
        if t_dwell > 0:
            _dwell(cursor, cursor + t_dwell, q_B)
            cursor += t_dwell
        # quintic B → A
        _quintic_segment(cursor, cursor + T_seg, q_B, q_A)
        cursor += T_seg
        if t_dwell > 0:
            _dwell(cursor, cursor + t_dwell, q_A)
            cursor += t_dwell
        # quintic A → B (start of next cycle, or final)
        _quintic_segment(cursor, cursor + T_seg, q_A, q_B)
        cursor += T_seg

    # Optional final dwell at q_B
    if t_dwell > 0:
        _dwell(cursor, cursor + t_dwell, q_B)

    return q_ref, qdot_ref, transition_mask


def trial_duration(
    t_dwell: float = DEFAULT_T_DWELL,
    T_seg: float = DEFAULT_T_SEG,
    n_return_cycles: int = DEFAULT_N_RETURN_CYCLES,
) -> float:
    """Total duration of one trial.

    Default (single A→B, no dwells): T_seg = 8 s.
    With dwells: 2*t_dwell (around the single segment) + T_seg.
    With return cycles: + n_return_cycles * (2*T_seg + 2*t_dwell).
    """
    base = T_seg
    if t_dwell > 0:
        base += 2 * t_dwell
    base += n_return_cycles * (2 * T_seg + (2 * t_dwell if t_dwell > 0 else 0))
    return base


if __name__ == "__main__":
    T = trial_duration()
    print(f"Total trial duration: {T:.2f} s")
    t = np.arange(0, T, DEFAULT_DT)
    q_ref, qdot_ref, mask = make_reference(t)
    print(f"Number of samples (H): {len(t)}")
    expected_trans = (1 + 2 * DEFAULT_N_RETURN_CYCLES) * DEFAULT_T_SEG
    actual_trans = mask.sum() * DEFAULT_DT
    print(f"Transition phase total: {actual_trans:.2f} s "
          f"(expected {expected_trans:.2f})")
    print(f"q_ref start: {q_ref[0]}")
    print(f"q_ref end:   {q_ref[-1]}")
    print(f"qdot_ref|_t=0: {qdot_ref[0].round(4)} (should be ≈ 0)")
    print(f"qdot_ref|_t=T: {qdot_ref[-1].round(4)} (should be ≈ 0)")
    print(f"qdot_ref max: {qdot_ref.max(axis=0).round(3)}")
