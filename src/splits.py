"""
Leave-one-fv_mult-slice-out splitter for the dense 2D grid.

Rationale (matches CLAUDE.md, 2026-04-09):
  Random splits let kNN / interp baselines cheat by borrowing neighbors
  in the held-out axis.  We force the held-out axis to be an entire
  fv_mult slice that the model never saw at training time.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Split:
    train_idx: np.ndarray
    test_idx: np.ndarray
    heldout_fv: float
    heldout_tol: float


def _unique_fv(phys_factors: np.ndarray, round_to: int = 4) -> np.ndarray:
    fvs = np.unique(np.round(phys_factors[:, 1], round_to))
    return fvs


def leave_one_fv_slice(
    phys_factors: np.ndarray,
    heldout_fv: float,
    tol: float = 1e-3,
) -> Split:
    """
    Put every point whose fv_mult is within `tol` of `heldout_fv` into the
    test set; everything else trains.
    """
    diff = np.abs(phys_factors[:, 1] - heldout_fv)
    test_mask = diff < tol
    train_mask = ~test_mask
    return Split(
        train_idx=np.where(train_mask)[0],
        test_idx=np.where(test_mask)[0],
        heldout_fv=float(heldout_fv),
        heldout_tol=float(tol),
    )


def leave_one_fv_slice_band(
    phys_factors: np.ndarray,
    fv_lo: float,
    fv_hi: float,
) -> Split:
    """
    For the dense Sobol grid where fv is not on a discrete lattice,
    hold out a *band* [fv_lo, fv_hi] as test.  Use this instead of
    `leave_one_fv_slice` on `results_mf_dense` (Sobol-sampled fv).
    """
    return leave_one_factor_band(phys_factors, fv_lo, fv_hi, axis=1)


def leave_one_factor_band(
    phys_factors: np.ndarray,
    lo: float,
    hi: float,
    axis: int = 1,
) -> Split:
    """
    Generic leave-one-band-out split along a specific physics factor axis.

    axis=0 → hold out points with m_p in [lo, hi]       (payload-space OOD)
    axis=1 → hold out points with fv_mult in [lo, hi]   (viscous-friction OOD)
    axis=2 → hold out points with fc_mult in [lo, hi]   (Coulomb-friction OOD, F=3 only)
    """
    if axis >= phys_factors.shape[1]:
        raise ValueError(f"axis={axis} out of range for phys_factors with "
                         f"{phys_factors.shape[1]} factors")
    vals = phys_factors[:, axis]
    test_mask = (vals >= lo) & (vals <= hi)
    return Split(
        train_idx=np.where(~test_mask)[0],
        test_idx=np.where(test_mask)[0],
        heldout_fv=float(0.5 * (lo + hi)),  # reused field: midpoint
        heldout_tol=float(0.5 * (hi - lo)),
    )


def region_holdout_split(
    phys_factors: np.ndarray,
    axis_a: int,
    a_lo: float,
    a_hi: float,
    axis_b: int,
    b_lo: float,
    b_hi: float,
) -> Split:
    """
    2D region holdout: test = points where
        axis_a ∈ [a_lo, a_hi]  AND  axis_b ∈ [b_lo, b_hi]
    Train = everything else.

    Used for F=3 compound-trap evaluation:
      - (fv, fc) corner R02: a=fv_mult ∈ [0.3, 0.63], b=fc_mult ∈ [0.97, 1.3]
      - (fv, fc) center R11: a=fv_mult ∈ [0.63, 0.97], b=fc_mult ∈ [0.63, 0.97]

    Under a (fv, fc) holdout with Partial = m_p-only, both fv and fc are
    hidden AND the held-out region is OOD → compound trap (two negative
    covariance contributions). See Appendix A (Theorem F≥2 extension).
    """
    if axis_a >= phys_factors.shape[1] or axis_b >= phys_factors.shape[1]:
        raise ValueError(f"axes out of range for phys_factors with "
                         f"{phys_factors.shape[1]} factors")
    if axis_a == axis_b:
        raise ValueError(f"axis_a and axis_b must differ (got {axis_a}=={axis_b})")
    vals_a = phys_factors[:, axis_a]
    vals_b = phys_factors[:, axis_b]
    test_mask = (
        (vals_a >= a_lo) & (vals_a <= a_hi)
        & (vals_b >= b_lo) & (vals_b <= b_hi)
    )
    return Split(
        train_idx=np.where(~test_mask)[0],
        test_idx=np.where(test_mask)[0],
        heldout_fv=float("nan"),  # reused field: not a band midpoint
        heldout_tol=float("nan"),
    )


def stiff_corner_split(
    phys_factors: np.ndarray,
    m_p_threshold: float = 3.0,
    fv_threshold: float = 0.5,
) -> Split:
    """
    F=3 stiff-corner split: hold out the (m_p >= m_p_threshold, fv <= fv_threshold)
    intrinsic-hard regime where factor-to-gain mapping is highly non-linear.

    Default thresholds produce ~237 points on the 3000-point F=3 dataset.
    This is NOT a leave-one-band-out split along a single axis — it is a
    regime-of-difficulty split for evaluating the probe-response advantage
    (§5.6.X in the paper). Train set = complement of the stiff corner.
    """
    if phys_factors.shape[1] < 2:
        raise ValueError("stiff_corner needs at least (m_p, fv) axes")
    m_p = phys_factors[:, 0]
    fv = phys_factors[:, 1]
    test_mask = (m_p >= m_p_threshold) & (fv <= fv_threshold)
    return Split(
        train_idx=np.where(~test_mask)[0],
        test_idx=np.where(test_mask)[0],
        heldout_fv=float("nan"),    # reused field: not a band midpoint
        heldout_tol=float("nan"),
    )


def random_iid_split(
    n_points: int,
    test_frac: float = 0.2,
    seed: int = 0,
) -> Split:
    """
    I.i.d. random train/test split over indices [0, n_points).  Use this
    when the goal is to measure interpolation quality inside the sampled
    (m_p, fv_mult) domain (robotics-style evaluation), not OOD robustness.
    """
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_points)
    n_test = max(1, int(round(n_points * test_frac)))
    test_idx = np.sort(perm[:n_test])
    train_idx = np.sort(perm[n_test:])
    return Split(
        train_idx=train_idx,
        test_idx=test_idx,
        heldout_fv=float("nan"),
        heldout_tol=float(test_frac),
    )


def leave_one_gain_cluster_out(
    gains: np.ndarray,
    n_clusters: int,
    heldout_cluster: int,
    seed: int = 0,
    standardise: bool = True,
) -> Split:
    """
    Cluster points by their optimal gain vectors (KMeans) and hold out one
    cluster as the test set.  This is a "gain-space OOD" split: the test
    points have PID gains whose multidimensional profile is distinct from
    training, regardless of where they sit in physics-factor space.

    gains: (N, G) optimal gain vectors
    n_clusters: how many clusters to form
    heldout_cluster: which cluster index to hold out (0..n_clusters-1)
    seed: KMeans random state for reproducibility
    standardise: z-score each gain dim before clustering (recommended — Kp
                 range can dwarf Kd range otherwise).

    Uses sklearn.cluster.KMeans. The cluster assignment is deterministic
    given (gains, n_clusters, seed).
    """
    from sklearn.cluster import KMeans

    g = gains.astype(np.float32)
    if standardise:
        g = (g - g.mean(axis=0, keepdims=True)) / (g.std(axis=0, keepdims=True) + 1e-8)
    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10).fit(g)
    labels = km.labels_
    test_mask = (labels == heldout_cluster)
    return Split(
        train_idx=np.where(~test_mask)[0],
        test_idx=np.where(test_mask)[0],
        heldout_fv=float("nan"),
        heldout_tol=float(heldout_cluster),  # reuse field as cluster id
    )


def gain_cluster_labels(
    gains: np.ndarray,
    n_clusters: int,
    seed: int = 0,
    standardise: bool = True,
) -> np.ndarray:
    """Return KMeans cluster labels for all points (deterministic given seed)."""
    from sklearn.cluster import KMeans
    g = gains.astype(np.float32)
    if standardise:
        g = (g - g.mean(axis=0, keepdims=True)) / (g.std(axis=0, keepdims=True) + 1e-8)
    return KMeans(n_clusters=n_clusters, random_state=seed, n_init=10).fit(g).labels_


def list_candidate_heldout_bands(
    phys_factors: np.ndarray,
    n_bands: int = 5,
    min_test_points: int = 15,
    axis: int = 1,
) -> list[tuple[float, float]]:
    """
    Suggest n_bands disjoint bands along axis (0=m_p, 1=fv_mult) for
    cross-validation style leave-one-band-out.  Each band contains at least
    `min_test_points` points if possible.
    """
    vals = np.sort(phys_factors[:, axis])
    if len(vals) < n_bands * min_test_points:
        raise ValueError(
            f"not enough points: len={len(vals)}, want {n_bands*min_test_points}"
        )
    # Equal-count quantile bands
    quantiles = np.linspace(0, 1, n_bands + 1)
    edges = np.quantile(vals, quantiles)
    bands = []
    for i in range(n_bands):
        lo, hi = float(edges[i]), float(edges[i + 1])
        if i == n_bands - 1:
            hi = hi + 1e-6
        bands.append((lo, hi))
    return bands
