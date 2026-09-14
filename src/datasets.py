"""
Shared data loading for the 4 baselines on the dense multi-factor grid.

Expected server layout (in ~/casadi-on-gpu):
  results_mf_dense/factor_grid.npz             # phys_factors (N, 2) = (m_p, fv_mult)
  results_mf_dense/epoch3/best_params_idx_*.npy  # gains per point, 21 dims each
  excitation/multifactor_dense_excitation_rollout.npz
      keys: excitation (N, T, dof), valid_mask (N,), phys_factors (N, 2)

For the 1f_ood baseline (trained on the 1D dataset, evaluated on the 2D grid),
we additionally need the kp100_ki5_kd20 style 1D dataset.  Paths are passed
in explicitly so you can point at whichever 1D dump is the "payload-only
practitioner" reference.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# GAIN_DIM and N_DOF are default hints only — the actual dim is auto-detected
# from whatever data the bundle/loader provides.  For Env-A 7-DoF the gains are
# 21-dim ([kp, ki, kd] × 7); for Env-B 4-DoF they are 12-dim.
GAIN_DIM = 12
N_DOF = 4
FACTOR_NAMES = ("m_p", "fv_mult")


# ---------------------------------------------------------------------------
# Low-level loaders
# ---------------------------------------------------------------------------

def _load_factor_grid(path: str) -> np.ndarray:
    with np.load(path, allow_pickle=True) as f:
        phys = f["phys_factors"].astype(np.float32)
    assert phys.ndim == 2 and phys.shape[1] == 2, f"expected (N,2), got {phys.shape}"
    return phys


def _load_gains_dir(gains_dir: str, n_points: int) -> np.ndarray:
    """Load best_params_idx_{i}.npy for i in [0, n_points)."""
    pattern = re.compile(r"best_params_idx_(\d+)\.npy$")
    files = {}
    for fn in os.listdir(gains_dir):
        m = pattern.match(fn)
        if m:
            files[int(m.group(1))] = os.path.join(gains_dir, fn)
    missing = [i for i in range(n_points) if i not in files]
    if missing:
        raise FileNotFoundError(f"missing gain files: {missing[:10]}... total {len(missing)}")
    arr = np.stack([np.load(files[i]).astype(np.float32).ravel() for i in range(n_points)], axis=0)
    return arr


def _load_excitation(excitation_npz: str):
    with np.load(excitation_npz, allow_pickle=True) as f:
        exc = f["excitation"].astype(np.float32)     # (N, T, dof) or (N, dof, T)
        valid = f["valid_mask"].astype(bool)
        phys = f["phys_factors"].astype(np.float32)
    # Normalise to (N, T, dof): if dim-1 is small (<=16) and dim-2 is large,
    # assume it is (N, dof, T) and transpose.
    if exc.ndim == 3 and exc.shape[1] <= 16 and exc.shape[2] > exc.shape[1]:
        exc = np.transpose(exc, (0, 2, 1))
    return exc, valid, phys


# ---------------------------------------------------------------------------
# Dense 2D dataset bundle
# ---------------------------------------------------------------------------

@dataclass
class DenseBundle:
    phys_factors: np.ndarray        # (N, F)  F=2 for dataset_casadi.npz, F=3 for F=3 datasets
    gains: np.ndarray               # (N, G) — G may be reduced if ki_ratio > 0
    excitation: np.ndarray          # (N, T, dof)
    valid_mask: np.ndarray          # (N,) bool
    # MuJoCo-style constrained parameterisation: if non-None, the full gain
    # vector is [Kp_1..J, Ki_1..J, Kd_1..J] with Ki_j = ki_ratio * Kp_j.
    # In this case `gains` here stores only [Kp_1..J, Kd_1..J] (2*J dims).
    ki_ratio: float | None = None
    n_joints: int | None = None     # needed for Ki reconstruction

    @property
    def n_factors(self) -> int:
        return int(self.phys_factors.shape[1])

    def filter_valid(self) -> "DenseBundle":
        m = self.valid_mask
        return DenseBundle(
            phys_factors=self.phys_factors[m],
            gains=self.gains[m],
            excitation=self.excitation[m],
            valid_mask=np.ones(int(m.sum()), dtype=bool),
            ki_ratio=self.ki_ratio,
            n_joints=self.n_joints,
        )

    def __len__(self) -> int:
        return self.phys_factors.shape[0]


def reconstruct_full_gains(
    predicted: np.ndarray,
    ki_ratio: float,
    n_joints: int,
) -> np.ndarray:
    """Reconstruct [Kp_1..J, Ki_1..J, Kd_1..J] from [Kp_1..J, Kd_1..J].

    predicted: (N, 2*J) with layout [Kp_1..J, Kd_1..J]
    Returns:   (N, 3*J) with layout [Kp_1..J, Ki_1..J, Kd_1..J]
    """
    assert predicted.shape[1] == 2 * n_joints, (
        f"expected 2*{n_joints}={2*n_joints} columns, got {predicted.shape[1]}")
    kp = predicted[:, :n_joints]
    kd = predicted[:, n_joints:]
    ki = ki_ratio * kp
    return np.concatenate([kp, ki, kd], axis=1)


def load_dense_2d(
    factor_grid_path: str,
    gains_dir: str,
    excitation_npz: str,
) -> DenseBundle:
    phys_grid = _load_factor_grid(factor_grid_path)
    n = phys_grid.shape[0]
    gains = _load_gains_dir(gains_dir, n)
    exc, valid, phys_exc = _load_excitation(excitation_npz)
    if exc.shape[0] != n:
        raise ValueError(f"excitation N={exc.shape[0]} vs factor_grid N={n}")
    # Sanity: factor_grid and excitation phys_factors should agree up to float eps
    if not np.allclose(phys_grid, phys_exc, atol=1e-4):
        raise ValueError("phys_factors mismatch between factor_grid.npz and excitation npz")
    return DenseBundle(phys_factors=phys_grid, gains=gains, excitation=exc, valid_mask=valid)


def load_mf_dense_bundle(
    bundle_npz: str,
    excitation_signal: str = "velocity",
    excitation_type: str | None = None,
) -> DenseBundle:
    """
    Loader for single-file bundled datasets. Two layouts are supported:

    1. Legacy (single-excitation): expects top-level keys
       ``velocity``, ``position``, ``applied_u``, ``valid_mask``.

    2. Multi-excitation (per EXCITATION_DATASET_SPEC): expects prefixed keys
       ``{type}_velocity``, ``{type}_position``, ``{type}_applied_u`` for
       type in {prbs, chirp, step, random}.  ``excitation_type`` selects
       which one to load; if None, falls back to 'random' → 'prbs' → any.

    The excitation stream is built from ``excitation_signal`` (default
    'velocity'), sliced to drop the initial (pre-probe) step so its length
    matches applied_u, then transposed to (N, T, dof).
    """
    ki_ratio = None
    with np.load(bundle_npz, allow_pickle=True) as f:
        files = set(f.files)
        phys = f["phys_factors"].astype(np.float32)
        gains = f["gains"].astype(np.float32)

        # ---- Choose excitation fields ---------------------------------------
        # If the file has per-type prefixed fields, pick one; otherwise use
        # the legacy top-level names.
        multi_types = [t for t in ("prbs", "chirp", "step", "random")
                       if f"{t}_{excitation_signal}" in files]
        if multi_types:
            if excitation_type is None:
                # Default preference order
                for t in ("random", "prbs", "chirp", "step"):
                    if t in multi_types:
                        excitation_type = t
                        break
            if excitation_type not in multi_types:
                raise ValueError(
                    f"excitation_type={excitation_type!r} not available; "
                    f"options in this file: {multi_types}")
            sig_key = f"{excitation_type}_{excitation_signal}"
            u_key = f"{excitation_type}_applied_u"
            print(f"[bundle] loaded excitation type = '{excitation_type}' "
                  f"from {os.path.basename(bundle_npz)}")
        else:
            sig_key = excitation_signal
            u_key = "applied_u"
        sig = f[sig_key].astype(np.float32)
        T_target = int(f[u_key].shape[-1]) if u_key in files else sig.shape[-1]

        # ---- valid_mask (optional) -----------------------------------------
        if "valid_mask" in files:
            valid = f["valid_mask"].astype(bool)
        else:
            valid = np.ones(phys.shape[0], dtype=bool)

        # ---- Auto-detect ki_ratio from meta (MuJoCo constrained BO) --------
        if "meta" in files:
            try:
                meta = f["meta"].item()
                if isinstance(meta, dict) and "ki_ratio" in meta:
                    kr = float(meta["ki_ratio"])
                elif isinstance(meta, str):
                    m = re.search(r"'ki_ratio'\s*:\s*([0-9.eE+-]+)", meta)
                    kr = float(m.group(1)) if m else 0.0
                else:
                    kr = 0.0
                if kr > 0:
                    ki_ratio = kr
            except Exception:
                pass

    # sig: (N, dof, T_raw).  Drop leading samples so it lines up with T_target.
    if sig.ndim != 3:
        raise ValueError(f"expected 3-D excitation signal, got shape {sig.shape}")
    N, dof, T_raw = sig.shape
    if T_raw == T_target + 1:
        sig = sig[:, :, 1:]   # drop initial sample
    elif T_raw == T_target:
        pass
    elif T_raw > T_target:
        sig = sig[:, :, -T_target:]
    else:
        raise ValueError(
            f"excitation signal T={T_raw} shorter than applied_u T={T_target}"
        )
    exc = np.transpose(sig, (0, 2, 1)).astype(np.float32)  # (N, T, dof)

    if gains.ndim != 2:
        raise ValueError(f"expected (N, G) gains, got {gains.shape}")
    if phys.ndim != 2 or phys.shape[0] != N:
        raise ValueError(f"expected phys (N, F), got {phys.shape}")
    if phys.shape[1] not in (2, 3):
        raise ValueError(f"phys must have 2 or 3 factors, got {phys.shape[1]}")

    # If Ki = ki_ratio * Kp, drop Ki columns from the training target.
    # Gain layout: [Kp_1..J, Ki_1..J, Kd_1..J] (J = dof = n_joints).
    n_joints = dof
    if ki_ratio is not None and gains.shape[1] == 3 * n_joints:
        # Verify the Ki=ki_ratio*Kp invariant (with some tolerance)
        kp = gains[:, :n_joints]
        ki = gains[:, n_joints:2*n_joints]
        kd = gains[:, 2*n_joints:]
        max_err = np.max(np.abs(ki - ki_ratio * kp))
        if max_err > 1e-3:
            print(f"[warn] ki_ratio={ki_ratio} but |Ki - r*Kp|_max = {max_err:.4f}; "
                  "dropping Ki columns anyway")
        gains = np.concatenate([kp, kd], axis=1)  # (N, 2*J)
        print(f"[bundle] ki_ratio={ki_ratio} detected → gains reduced to "
              f"{gains.shape[1]} dims ([Kp_1..{n_joints}, Kd_1..{n_joints}])")

    return DenseBundle(
        phys_factors=phys,
        gains=gains,
        excitation=exc,
        valid_mask=valid,
        ki_ratio=ki_ratio,
        n_joints=n_joints,
    )


# ---------------------------------------------------------------------------
# 1D OOD dataset (payload-only sweep, for the "naive practitioner" baseline)
# ---------------------------------------------------------------------------

@dataclass
class OneDBundle:
    m_p: np.ndarray                 # (N1,)
    gains: np.ndarray               # (N1, 21)
    excitation: np.ndarray | None   # (N1, T, dof) or None

    def __len__(self) -> int:
        return self.m_p.shape[0]


def load_1d_payload_sweep(
    gains_dir: str,
    factor_file: str,
    excitation_npz: str | None = None,
) -> OneDBundle:
    """
    Loads the 1D kp100_ki5_kd20 style dataset.
    factor_file: .npz or .npy holding payloads in order (N1,)
    gains_dir:   directory with best_params_idx_*.npy
    excitation_npz: optional, must contain 'excitation' and 'payloads'
    """
    if factor_file.endswith(".npz"):
        with np.load(factor_file, allow_pickle=True) as f:
            if "payloads" in f:
                payloads = f["payloads"].astype(np.float32).ravel()
            elif "phys_factors" in f:
                pf = f["phys_factors"].astype(np.float32)
                payloads = pf[:, 0].ravel()  # assume col 0 = m_p
            else:
                raise KeyError(f"{factor_file} has no 'payloads' or 'phys_factors'")
    else:
        payloads = np.load(factor_file).astype(np.float32).ravel()

    n1 = payloads.shape[0]
    gains = _load_gains_dir(gains_dir, n1)

    exc = None
    if excitation_npz is not None:
        with np.load(excitation_npz, allow_pickle=True) as f:
            exc = f["excitation"].astype(np.float32)
        if exc.ndim == 3 and exc.shape[1] == N_DOF and exc.shape[2] != N_DOF:
            exc = np.transpose(exc, (0, 2, 1))
        if exc.shape[0] != n1:
            raise ValueError(f"1D excitation N={exc.shape[0]} vs gains N={n1}")
    return OneDBundle(m_p=payloads, gains=gains, excitation=exc)
