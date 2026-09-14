"""
Train one of four baselines on the dense 2D (m_p, fv_mult) grid and
dump predictions on the held-out fv band for closed-loop cost eval.

Baselines
---------
  1f_indist  : input = (m_p,)          — trained on train slice of dense grid
  1f_ood     : input = (m_p,)          — trained on the 1D payload-only dump,
                                          evaluated on dense grid test slice
  2f_oracle  : input = (m_p, fv_mult)  — trained on train slice of dense grid
  e2e        : input = excitation (B,T,dof)
                                       — trained on train slice of dense grid

The first is an ablation ("does MLP capacity limit 1f?"); the second is the
*real* naive-practitioner baseline ("deploy a payload-only model to a fv-
varying robot").  The 2f oracle is the identifiability upper bound.  The e2e
is the proposed method.

Outputs (to --out-dir):
  predictions.npz with keys
      pred_gains_test   (M, 21)
      true_gains_test   (M, 21)
      phys_test         (M, 2)
      pred_gains_train  (K, 21)
      true_gains_train  (K, 21)
      phys_train        (K, 2)
      gain_mse_test     scalar
      train_idx         (K,)  — into dense grid ordering
      test_idx          (M,)
      baseline          str
      heldout_fv_lo/hi  floats
  train.log text file with per-epoch loss.

Usage (server side, inside conda env `exp`):

  # 1f_indist
  python train_baselines.py --baseline 1f_indist \
      --factor-grid results_mf_dense/factor_grid.npz \
      --gains-dir   results_mf_dense/epoch3 \
      --excitation  excitation/multifactor_dense_excitation_rollout.npz \
      --heldout-fv-lo 0.95 --heldout-fv-hi 1.05 \
      --out-dir baselines_out/1f_indist_fv1.0

  # 1f_ood (payload-only model trained on 1D dataset)
  python train_baselines.py --baseline 1f_ood \
      --factor-grid results_mf_dense/factor_grid.npz \
      --gains-dir   results_mf_dense/epoch3 \
      --excitation  excitation/multifactor_dense_excitation_rollout.npz \
      --one-d-gains-dir results_kp100_ki5_kd20/epoch30 \
      --one-d-factor-file results_kp100_ki5_kd20/payloads.npy \
      --heldout-fv-lo 0.95 --heldout-fv-hi 1.05 \
      --out-dir baselines_out/1f_ood_fv1.0

  # 2f_oracle
  python train_baselines.py --baseline 2f_oracle --heldout-fv-lo 0.95 --heldout-fv-hi 1.05 \
      --factor-grid ... --gains-dir ... --excitation ... \
      --out-dir baselines_out/2f_oracle_fv1.0

  # e2e
  python train_baselines.py --baseline e2e --heldout-fv-lo 0.95 --heldout-fv-hi 1.05 \
      --factor-grid ... --gains-dir ... --excitation ... \
      --out-dir baselines_out/e2e_fv1.0
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from datasets import (  # noqa: E402
    DenseBundle,
    GAIN_DIM,
    load_1d_payload_sweep,
    load_dense_2d,
    load_mf_dense_bundle,
)
from models import (  # noqa: E402
    MLPHead,
    PerJointMoEHead,
    TransformerEncoderHead,
    TransformerMVEHead,
)
from splits import Split, leave_one_fv_slice_band, random_iid_split   # noqa: E402


# ---------------------------------------------------------------------------
# Input builders per baseline
# ---------------------------------------------------------------------------

_ALIAS = {
    "1f_indist":  "paramid_1f_indist",
    "1f_ood":     "paramid_1f",
    "2f_oracle":  "paramid_2f_oracle",
    "3f_oracle":  "paramid_3f_oracle",  # F=3: full 3-factor MLP oracle
    "e2e":        "excite_e2e",
    "e2e_zaux":   "excite_e2e_zaux",
    "moe_zsup":   "excite_e2e_moe_zsup",
    "moe_pure":   "excite_e2e_moe_pure",
    # Transformer-on-scalar controls (match e2e model capacity; only input differs)
    "1f_trans":   "paramid_1f_trans",
    "2f_trans":   "paramid_2f_trans",
    "3f_trans":   "paramid_3f_trans",  # F=3: full 3-factor Transformer oracle
    # Bayesian uncertainty-aware variant of 1f_trans (Gaussian NLL / MVE).
    "1f_mve":     "paramid_1f_mve",
}

def canon(baseline: str) -> str:
    return _ALIAS.get(baseline, baseline)


# Which physics column the paramid_1f* baselines see as their input.
# 0 = m_p (default, the "knows payload" practitioner)
# 1 = fv_mult (the "knows friction" practitioner)
# Set from CLI --input-factor.
_INPUT_FACTOR_COL = 0


def build_inputs(baseline: str, bundle, split: Split):
    """Return (x_train, y_train, x_test, y_test, input_kind).
    input_kind ∈ {"vec", "seq", "moe"}.  "moe" is identical to "seq" but
    signals to main() that the head is the GMM-routed MoE.
    """
    baseline = canon(baseline)
    phys = bundle.phys_factors
    gains = bundle.gains
    col = _INPUT_FACTOR_COL  # 0=mp, 1=fv
    if baseline == "paramid_1f_indist":
        x_all = phys[:, col:col+1]          # (N, 1)
        x_train = x_all[split.train_idx]
        x_test = x_all[split.test_idx]
        return x_train, gains[split.train_idx], x_test, gains[split.test_idx], "vec"
    if baseline == "paramid_2f_oracle":
        # Input = first 2 columns (m_p, fv) regardless of dataset F
        # (On F=3 datasets this is the "2-factor with fc hidden" — the trap baseline)
        x_all = phys[:, :2]                 # (N, 2)
        return (
            x_all[split.train_idx],
            gains[split.train_idx],
            x_all[split.test_idx],
            gains[split.test_idx],
            "vec",
        )
    if baseline == "paramid_3f_oracle":
        # Requires F=3 dataset: input = all 3 factors (m_p, fv, fc)
        if phys.shape[1] < 3:
            raise ValueError(f"paramid_3f_oracle requires F>=3 dataset, got F={phys.shape[1]}")
        x_all = phys[:, :3]                 # (N, 3)
        return (
            x_all[split.train_idx],
            gains[split.train_idx],
            x_all[split.test_idx],
            gains[split.test_idx],
            "vec",
        )
    if baseline in ("excite_e2e",):
        exc = bundle.excitation             # (N, T, dof)
        return (
            exc[split.train_idx],
            gains[split.train_idx],
            exc[split.test_idx],
            gains[split.test_idx],
            "seq",
        )
    if baseline == "excite_e2e_zaux":
        exc = bundle.excitation             # (N, T, dof)
        # Same seq input as vanilla Trans; the "zaux" suffix only affects
        # model build + training loop (auxiliary m_p supervision head).
        return (
            exc[split.train_idx],
            gains[split.train_idx],
            exc[split.test_idx],
            gains[split.test_idx],
            "seq_zaux",
        )
    if baseline in ("excite_e2e_moe_zsup", "excite_e2e_moe_pure"):
        exc = bundle.excitation             # (N, T, dof)
        return (
            exc[split.train_idx],
            gains[split.train_idx],
            exc[split.test_idx],
            gains[split.test_idx],
            "moe",
        )
    if baseline in ("paramid_1f_trans", "paramid_2f_trans",
                    "paramid_3f_trans", "paramid_1f_mve"):
        # Capacity-controlled baselines: scalar physics input broadcast to a
        # time sequence, fed to the SAME Transformer as excite_e2e.  This
        # isolates "what input information is available" from "model capacity".
        T = bundle.excitation.shape[1]      # match e2e's time dim
        if baseline in ("paramid_1f_trans", "paramid_1f_mve"):
            phys_sub = phys[:, col:col+1]   # (N, 1) — single chosen factor
            n_factors = 1
        elif baseline == "paramid_2f_trans":
            # First 2 columns (m_p, fv). On F=3 datasets this is the "fc-hidden" trap baseline.
            phys_sub = phys[:, :2]          # (N, 2)
            n_factors = 2
        elif baseline == "paramid_3f_trans":
            if phys.shape[1] < 3:
                raise ValueError(f"paramid_3f_trans requires F>=3 dataset, got F={phys.shape[1]}")
            phys_sub = phys[:, :3]          # (N, 3) — all factors
            n_factors = 3
        else:
            raise AssertionError(baseline)
        # Broadcast (N, n_factors) -> (N, T, n_factors) by repeating along time
        seq = np.broadcast_to(
            phys_sub[:, None, :], (phys_sub.shape[0], T, n_factors)
        ).copy()
        kind = "seq_mve" if baseline == "paramid_1f_mve" else "seq"
        return (
            seq[split.train_idx],
            gains[split.train_idx],
            seq[split.test_idx],
            gains[split.test_idx],
            kind,
        )
    if baseline == "paramid_plus_excite":
        # Strong partial-ID baseline: observed scalar physics factor
        # CONCATENATED with the excitation response at each timestep. Input
        # is (N, T, dof + 1); model is the same Transformer as excite_e2e.
        # Tests whether partial ID fails because of (a) missing factor info
        # or (b) missing probe info: if this baseline matches excite_e2e it
        # means observed factor contributes nothing beyond the probe; if it
        # beats excite_e2e it means observed factor does add information
        # and our contribution framing should be "excitation dominates".
        exc = bundle.excitation              # (N, T, dof)
        T = exc.shape[1]
        phys_sub = phys[:, col:col+1]        # (N, 1)
        phys_seq = np.broadcast_to(
            phys_sub[:, None, :], (phys_sub.shape[0], T, 1)
        ).copy()
        seq = np.concatenate([exc, phys_seq], axis=-1)  # (N, T, dof+1)
        return (
            seq[split.train_idx],
            gains[split.train_idx],
            seq[split.test_idx],
            gains[split.test_idx],
            "seq",
        )
    raise ValueError(f"unknown baseline for this path: {baseline}")


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------

class Standardiser:
    def __init__(self, x: np.ndarray, eps: float = 1e-6):
        self.mean = x.mean(axis=0, keepdims=True)
        self.std = x.std(axis=0, keepdims=True) + eps

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std

    def inv(self, x: np.ndarray) -> np.ndarray:
        return x * self.std + self.mean


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_loop(
    model: torch.nn.Module,
    x_train_t: torch.Tensor,
    y_train_t: torch.Tensor,
    x_test_t: torch.Tensor,
    y_test_t: torch.Tensor,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    device: torch.device,
    log_fp,
    loss_weight: torch.Tensor | None = None,
    track_best: bool = True,
    val_frac: float = 0.0,
    val_seed: int = 1234,
):
    """Train loop with optional per-dim loss weighting + best-on-val tracking.

    loss_weight: (G,) tensor; weights applied to per-dim squared error.
        Default None → uniform. Use 0/1 mask to exclude degenerate dims.
    track_best: if True, restores the model state from the epoch with lowest
        validation loss.
    val_frac: if > 0, hold out this fraction of training data as val for
        model selection (proper methodology, no test-set leakage). If 0,
        falls back to selecting on test set (legacy behaviour, test-optimistic).
    val_seed: seed for the train/val split (distinct from training seed).
    """
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    # Optional proper val split (held out from training data)
    if val_frac > 0:
        n_full = x_train_t.size(0)
        n_val = max(1, int(round(n_full * val_frac)))
        g = torch.Generator().manual_seed(val_seed)
        perm = torch.randperm(n_full, generator=g)
        val_idx = perm[:n_val]
        tr_idx = perm[n_val:]
        x_val_t, y_val_t = x_train_t[val_idx], y_train_t[val_idx]
        x_train_t, y_train_t = x_train_t[tr_idx], y_train_t[tr_idx]
        msg0 = (f"[val-split] val_frac={val_frac} val_seed={val_seed}  "
                f"train={x_train_t.size(0)} val={x_val_t.size(0)}")
    else:
        # Legacy: track best on TEST set (test-optimistic; documented in paper §4).
        x_val_t, y_val_t = x_test_t, y_test_t
        msg0 = "[val-split] DISABLED — tracking best on test (test-optimistic)"
    print(msg0)
    print(msg0, file=log_fp, flush=True)

    ds = TensorDataset(x_train_t, y_train_t)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)

    if loss_weight is not None:
        loss_weight = loss_weight.to(device)
        w_sum = float(loss_weight.sum()) + 1e-9
    else:
        w_sum = None

    def weighted_mse(pred, target):
        if loss_weight is None:
            return F.mse_loss(pred, target)
        sq = (pred - target) ** 2
        # pred/target: (B, G), loss_weight: (G,) → broadcast
        return (sq * loss_weight).sum() / (sq.size(0) * w_sum)

    def batched_forward(x_tensor, bs: int = 128):
        """Forward in batches to avoid MPS attention-matrix OOM on large N.
        Returns concatenated CPU tensor (no grad)."""
        outs = []
        for i in range(0, x_tensor.size(0), bs):
            chunk = x_tensor[i:i + bs].to(device)
            out = model(chunk)
            outs.append(out.detach())
        return torch.cat(outs, dim=0)

    best_val = float("inf")
    best_ep = -1
    best_state = None
    sel_label = "val" if val_frac > 0 else "test"

    for ep in range(epochs):
        model.train()
        tot, n = 0.0, 0
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=False)
            yb = yb.to(device, non_blocking=False)
            pred = model(xb)
            loss = weighted_mse(pred, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += float(loss) * xb.size(0)
            n += xb.size(0)
        sched.step()
        train_mse = tot / max(n, 1)

        model.eval()
        with torch.no_grad():
            # Selection set: val if split enabled, else test (legacy)
            pred_val = batched_forward(x_val_t)
            y_val_dev = y_val_t.to(device)
            val_all = float(F.mse_loss(pred_val, y_val_dev))
            if loss_weight is not None:
                val_mse = float(
                    ((pred_val - y_val_dev) ** 2 * loss_weight).sum() /
                    (pred_val.size(0) * w_sum)
                )
            else:
                val_mse = val_all
            # Also compute test MSE for logging (never used for selection when val enabled)
            if val_frac > 0:
                pred_test = batched_forward(x_test_t)
                y_test_dev = y_test_t.to(device)
                test_all = float(F.mse_loss(pred_test, y_test_dev))
                if loss_weight is not None:
                    test_mse_log = float(
                        ((pred_test - y_test_dev) ** 2 * loss_weight).sum() /
                        (pred_test.size(0) * w_sum)
                    )
                else:
                    test_mse_log = test_all
            else:
                test_mse_log, test_all = val_mse, val_all
        # Track best by the val (or test in legacy mode) MSE
        is_best = val_mse < best_val
        if track_best and is_best:
            best_val = val_mse
            best_ep = ep
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        tag = " *" if is_best else ""
        if val_frac > 0:
            msg = (f"ep {ep+1:03d}/{epochs}  train_mse={train_mse:.6f}  "
                   f"val_mse={val_mse:.6f}  test_mse={test_mse_log:.6f}"
                   f" (all_dim={test_all:.4f}){tag}")
        elif loss_weight is not None:
            msg = (f"ep {ep+1:03d}/{epochs}  train_mse={train_mse:.6f}  "
                   f"test_mse={val_mse:.6f} (all_dim={val_all:.4f}){tag}")
        else:
            msg = f"ep {ep+1:03d}/{epochs}  train_mse={train_mse:.6f}  test_mse={val_mse:.6f}{tag}"
        print(msg)
        print(msg, file=log_fp, flush=True)
        if device.type == "mps" and (ep + 1) % 10 == 0:
            torch.mps.empty_cache()

    if track_best and best_state is not None:
        model.load_state_dict(best_state)
        final_msg = (f"[best-on-{sel_label}] restored model from ep "
                     f"{best_ep+1}/{epochs}  {sel_label}_mse={best_val:.6f}")
        print(final_msg)
        print(final_msg, file=log_fp, flush=True)
    return model


def train_loop_mve(
    model: torch.nn.Module,
    x_train_t: torch.Tensor,
    y_train_t: torch.Tensor,
    x_test_t: torch.Tensor,
    y_test_t: torch.Tensor,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    device: torch.device,
    log_fp,
    logvar_clamp: tuple = (-8.0, 4.0),
    loss_weight: torch.Tensor | None = None,
    track_best: bool = True,
):
    """Mean-Variance Estimator training loop with Gaussian NLL.

    Loss per element:
        NLL = 0.5 * [exp(-logvar) * (target - mu)^2 + logvar]

    Uses the same per-dim loss weighting (if provided) and best-on-test
    tracking convention as train_loop, but reports both NLL (objective) and
    MSE (comparable to train_loop's metric) during training.
    """
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    ds = TensorDataset(x_train_t, y_train_t)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)

    if loss_weight is not None:
        loss_weight = loss_weight.to(device)
        w_sum = float(loss_weight.sum()) + 1e-9
    else:
        w_sum = None

    def gaussian_nll(mu, logvar, target):
        logvar = logvar.clamp(min=logvar_clamp[0], max=logvar_clamp[1])
        precision = torch.exp(-logvar)
        per_dim = 0.5 * (precision * (target - mu) ** 2 + logvar)
        if loss_weight is None:
            return per_dim.mean()
        return (per_dim * loss_weight).sum() / (per_dim.size(0) * w_sum)

    def batched_mve(x_tensor, bs: int = 128):
        mus, lvs = [], []
        for i in range(0, x_tensor.size(0), bs):
            chunk = x_tensor[i:i + bs].to(device)
            mu_b, lv_b = model(chunk, return_mve=True)
            mus.append(mu_b.detach())
            lvs.append(lv_b.detach())
        return torch.cat(mus, dim=0), torch.cat(lvs, dim=0)

    best_test = float("inf")
    best_ep = -1
    best_state = None

    for ep in range(epochs):
        model.train()
        tot_nll, tot_mse, n = 0.0, 0.0, 0
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=False)
            yb = yb.to(device, non_blocking=False)
            mu, logvar = model(xb, return_mve=True)
            loss = gaussian_nll(mu, logvar, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            with torch.no_grad():
                mse = F.mse_loss(mu, yb)
            tot_nll += float(loss) * xb.size(0)
            tot_mse += float(mse) * xb.size(0)
            n += xb.size(0)
        sched.step()
        train_nll = tot_nll / max(n, 1)
        train_mse = tot_mse / max(n, 1)

        model.eval()
        with torch.no_grad():
            mu_test, _ = batched_mve(x_test_t)
            y_test_dev = y_test_t.to(device)
            test_all = float(F.mse_loss(mu_test, y_test_dev))
            if loss_weight is not None:
                test_mse = float(
                    ((mu_test - y_test_dev) ** 2 * loss_weight).sum() /
                    (mu_test.size(0) * w_sum)
                )
            else:
                test_mse = test_all
        is_best = test_mse < best_test
        if track_best and is_best:
            best_test = test_mse
            best_ep = ep
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        tag = " *" if is_best else ""
        msg = (f"ep {ep+1:03d}/{epochs}  train_nll={train_nll:.6f}  "
               f"train_mse={train_mse:.6f}  test_mse={test_mse:.6f}"
               f" (all_dim={test_all:.4f}){tag}")
        print(msg)
        print(msg, file=log_fp, flush=True)
        if device.type == "mps" and (ep + 1) % 10 == 0:
            torch.mps.empty_cache()

    if track_best and best_state is not None:
        model.load_state_dict(best_state)
        final_msg = (f"[best-on-test] restored model from ep {best_ep+1}/{epochs}"
                     f"  test_mse={best_test:.6f}")
        print(final_msg)
        print(final_msg, file=log_fp, flush=True)
    return model


def train_loop_moe(
    model: torch.nn.Module,
    x_train_t: torch.Tensor,
    y_train_t: torch.Tensor,
    z_target_train_t: torch.Tensor | None,   # (N, 1) normalized z target, or None
    x_test_t: torch.Tensor,
    y_test_t: torch.Tensor,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    lambda_z_sup: float,
    device: torch.device,
    log_fp,
):
    """Training loop for the per-joint GMM-routed MoE.

    If `z_target_train_t` is None or `lambda_z_sup` == 0, runs pure-MSE.
    Otherwise applies an auxiliary `lambda_z_sup * mean_j MSE(z_j, z_target)`.
    No annealing, no hard routing, no neighbor smoothing — pure GMM routing
    with optional z supervision.  Add complexity back only if needed.
    """
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    has_zsup = (z_target_train_t is not None) and (lambda_z_sup > 0)
    if has_zsup:
        ds = TensorDataset(x_train_t, y_train_t, z_target_train_t)
    else:
        ds = TensorDataset(x_train_t, y_train_t)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)

    for ep in range(epochs):
        model.train()
        tot_mse, tot_zsup, n = 0.0, 0.0, 0
        for batch in loader:
            if has_zsup:
                xb, yb, zb = [t.to(device, non_blocking=False) for t in batch]
            else:
                xb, yb = [t.to(device, non_blocking=False) for t in batch]
                zb = None

            if has_zsup:
                pred, z_pred = model(xb, return_z=True)
                z_target_expand = zb.expand(-1, z_pred.size(1))   # (B, J)
                l_mse = F.mse_loss(pred, yb)
                l_zsup = F.mse_loss(z_pred, z_target_expand)
                loss = l_mse + lambda_z_sup * l_zsup
            else:
                pred = model(xb)
                l_mse = F.mse_loss(pred, yb)
                l_zsup = torch.tensor(0.0, device=device)
                loss = l_mse

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            tot_mse += float(l_mse) * xb.size(0)
            tot_zsup += float(l_zsup) * xb.size(0)
            n += xb.size(0)
        sched.step()
        train_mse = tot_mse / max(n, 1)
        train_zsup = tot_zsup / max(n, 1)

        model.eval()
        with torch.no_grad():
            outs = []
            for i in range(0, x_test_t.size(0), 128):
                outs.append(model(x_test_t[i:i+128].to(device)).detach())
            pred_test = torch.cat(outs, dim=0)
            test_mse = float(F.mse_loss(pred_test, y_test_t.to(device)))
        msg = (f"ep {ep+1:03d}/{epochs}  train_mse={train_mse:.6f}  "
               f"test_mse={test_mse:.6f}  z_sup={train_zsup:.6f}")
        print(msg)
        print(msg, file=log_fp, flush=True)
        if device.type == "mps" and (ep + 1) % 10 == 0:
            torch.mps.empty_cache()
    return model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True,
                    choices=["paramid_1f", "paramid_1f_indist",
                             "paramid_2f_oracle", "paramid_3f_oracle",
                             "excite_e2e",
                             "excite_e2e_zaux",
                             "excite_e2e_moe_zsup", "excite_e2e_moe_pure",
                             "paramid_1f_trans", "paramid_2f_trans",
                             "paramid_3f_trans", "paramid_1f_mve",
                             "paramid_plus_excite",
                             # legacy aliases (kept for backward compat):
                             "1f_ood", "1f_indist", "2f_oracle", "3f_oracle",
                             "e2e", "e2e_zaux", "moe_zsup", "moe_pure",
                             "1f_trans", "2f_trans", "3f_trans", "1f_mve"])
    # Dense 2D inputs — either three-file layout (legacy) or bundled single-npz.
    ap.add_argument("--factor-grid", default=None)
    ap.add_argument("--gains-dir", default=None)
    ap.add_argument("--excitation", default=None)
    ap.add_argument("--bundle-npz", default=None,
                    help="path to the single-file dense bundle (mutually "
                         "exclusive with --factor-grid/--gains-dir/--excitation)")
    ap.add_argument("--bundle-signal", default="velocity",
                    help="which signal in the bundle to use as excitation "
                         "(velocity / position / applied_u)")
    ap.add_argument("--excitation-type", default=None,
                    choices=[None, "prbs", "chirp", "step", "random"],
                    help="for multi-excitation datasets, which type to load")
    # paramid_1f only
    ap.add_argument("--one-d-gains-dir", default=None)
    ap.add_argument("--one-d-factor-file", default=None)
    ap.add_argument("--one-d-excitation", default=None,
                    help="optional; not actually needed for paramid_1f since input is m_p")
    ap.add_argument("--paramid-1f-fv-tol", type=float, default=None,
                    help="if set and --one-d-* are not provided, synthesize the "
                         "paramid_1f training set by taking dense-grid points with "
                         "|other_factor - nominal| <= tol (excluding any held-out band).")
    ap.add_argument("--paramid-1f-nominal", type=float, default=1.0,
                    help="Nominal value of the OTHER factor for the paramid_1f "
                         "synthesised sweep (default 1.0).")
    ap.add_argument("--input-factor", choices=["mp", "fv"], default="mp",
                    help="Which physics factor the paramid_1f* baselines see as "
                         "input. 'mp' = practitioner who measured payload (default), "
                         "'fv' = practitioner who measured friction.")
    # Split
    ap.add_argument("--split-mode", choices=["band", "iid", "region"], default="band",
                    help="'band' = leave-one-factor-band-out (OOD, 1 axis); "
                         "'region' = 2D region holdout (OOD, 2 axes, compound); "
                         "'iid' = random train/test split (interpolation)")
    ap.add_argument("--split-axis", choices=["mp", "fv", "fc"], default="fv",
                    help="which physics axis to split along in band mode "
                         "('mp' = payload, 'fv' = viscous friction, "
                         "'fc' = Coulomb friction — F=3 datasets only)")
    ap.add_argument("--heldout-fv-lo", type=float, default=None,
                    help="(band mode) low edge of held-out band on the chosen axis")
    ap.add_argument("--heldout-fv-hi", type=float, default=None,
                    help="(band mode) high edge of held-out band on the chosen axis")
    # Region mode (2D holdout)
    ap.add_argument("--region-axis-a", choices=["mp", "fv", "fc"], default="fv",
                    help="(region mode) first axis of 2D region")
    ap.add_argument("--region-axis-b", choices=["mp", "fv", "fc"], default="fc",
                    help="(region mode) second axis of 2D region")
    ap.add_argument("--region-a-lo", type=float, default=None,
                    help="(region mode) low edge on axis A")
    ap.add_argument("--region-a-hi", type=float, default=None,
                    help="(region mode) high edge on axis A")
    ap.add_argument("--region-b-lo", type=float, default=None,
                    help="(region mode) low edge on axis B")
    ap.add_argument("--region-b-hi", type=float, default=None,
                    help="(region mode) high edge on axis B")
    ap.add_argument("--test-frac", type=float, default=0.2,
                    help="test fraction when --split-mode=iid")
    ap.add_argument("--split-seed", type=int, default=0,
                    help="rng seed for the iid split (independent of --seed)")
    # Training
    ap.add_argument("--duration-frac", type=float, default=0.5,
                    help="fraction of excitation to use (0.5 → T=100 out of 200)")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--val-frac", type=float, default=0.0,
                    help="fraction of train data held out as validation for "
                         "best-epoch selection. 0 = legacy (select on test, "
                         "test-optimistic). 0.1 = proper val split.")
    ap.add_argument("--val-seed", type=int, default=1234,
                    help="rng seed for the train/val split (distinct from --seed)")
    ap.add_argument("--hidden", type=int, default=256)
    # Transformer defaults = trans_S (best config)
    ap.add_argument("--d-model", type=int, default=160)
    ap.add_argument("--nhead", type=int, default=4)
    ap.add_argument("--nlayers", type=int, default=3)
    ap.add_argument("--ff", type=int, default=320)
    # MoE defaults = moe_S (best config)
    ap.add_argument("--moe-d-model", type=int, default=64)
    ap.add_argument("--moe-nhead", type=int, default=4)
    ap.add_argument("--moe-nlayers", type=int, default=2)
    ap.add_argument("--moe-n-experts", type=int, default=4)
    ap.add_argument("--moe-expert-hidden", type=int, default=64)
    ap.add_argument("--moe-temperature", type=float, default=1.0)
    ap.add_argument("--moe-noise-eps", type=float, default=5e-4)
    ap.add_argument("--moe-lambda-z-sup", type=float, default=0.1,
                    help="weight on z-supervision loss for excite_e2e_moe_zsup; "
                         "ignored by excite_e2e_moe_pure")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--nonneg-output", action="store_true",
                    help="enforce non-negative gain output via sharp softplus round-trip")
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    # Set the global input-factor column for paramid_1f* baselines
    global _INPUT_FACTOR_COL
    _INPUT_FACTOR_COL = 0 if args.input_factor == "mp" else 1
    if _INPUT_FACTOR_COL == 1:
        print(f"[input-factor] paramid_1f* baselines will use FV (col {_INPUT_FACTOR_COL})")

    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    print(f"device: {device}")

    # ---- Load dense 2D (always the test source) --------------------------
    if args.bundle_npz is not None:
        bundle = load_mf_dense_bundle(
            args.bundle_npz,
            excitation_signal=args.bundle_signal,
            excitation_type=args.excitation_type,
        )
    else:
        if not (args.factor_grid and args.gains_dir and args.excitation):
            raise SystemExit("need either --bundle-npz or "
                             "--factor-grid+--gains-dir+--excitation")
        bundle = load_dense_2d(args.factor_grid, args.gains_dir, args.excitation)
    print(f"dense bundle: {len(bundle)} raw points, valid={int(bundle.valid_mask.sum())}")
    bundle = bundle.filter_valid()
    print(f"dense bundle after valid filter: {len(bundle)}")

    # ---- Truncate excitation to duration_frac --------------------------------
    if args.duration_frac < 1.0 and bundle.excitation is not None:
        T_full = bundle.excitation.shape[1]
        T_use = max(2, int(T_full * args.duration_frac))
        bundle.excitation = bundle.excitation[:, :T_use, :]
        print(f"excitation truncated: T={T_use}/{T_full} ({args.duration_frac:.0%})")

    if args.split_mode == "band":
        if args.heldout_fv_lo is None or args.heldout_fv_hi is None:
            raise SystemExit("--split-mode=band needs --heldout-fv-lo and --heldout-fv-hi")
        from splits import leave_one_factor_band
        axis_idx = {"mp": 0, "fv": 1, "fc": 2}[args.split_axis]
        split = leave_one_factor_band(
            bundle.phys_factors, args.heldout_fv_lo, args.heldout_fv_hi, axis=axis_idx
        )
        print(f"split: {len(split.train_idx)} train / {len(split.test_idx)} test "
              f"(heldout {args.split_axis}-band [{args.heldout_fv_lo},{args.heldout_fv_hi}])")
        if len(split.test_idx) == 0:
            raise SystemExit("no test points fell in heldout band; adjust --heldout-fv-lo/hi")
    elif args.split_mode == "iid":
        split = random_iid_split(
            n_points=len(bundle),
            test_frac=args.test_frac,
            seed=args.split_seed,
        )
        print(f"split: {len(split.train_idx)} train / {len(split.test_idx)} test "
              f"(iid, test_frac={args.test_frac}, split_seed={args.split_seed})")
    elif args.split_mode == "region":
        if None in (args.region_a_lo, args.region_a_hi,
                    args.region_b_lo, args.region_b_hi):
            raise SystemExit("--split-mode=region needs --region-{a,b}-{lo,hi}")
        from splits import region_holdout_split
        axis_a_idx = {"mp": 0, "fv": 1, "fc": 2}[args.region_axis_a]
        axis_b_idx = {"mp": 0, "fv": 1, "fc": 2}[args.region_axis_b]
        split = region_holdout_split(
            bundle.phys_factors,
            axis_a_idx, args.region_a_lo, args.region_a_hi,
            axis_b_idx, args.region_b_lo, args.region_b_hi,
        )
        print(f"split: {len(split.train_idx)} train / {len(split.test_idx)} test "
              f"(region {args.region_axis_a} in [{args.region_a_lo},{args.region_a_hi}] "
              f"AND {args.region_axis_b} in [{args.region_b_lo},{args.region_b_hi}])")
        if len(split.test_idx) == 0:
            raise SystemExit("no test points fell in region; check bounds")
    else:
        raise SystemExit(f"unknown split mode: {args.split_mode}")

    baseline = canon(args.baseline)

    # ---- Build inputs -----------------------------------------------------
    if baseline in ("paramid_1f_indist",
                    "paramid_2f_oracle", "paramid_3f_oracle",
                    "excite_e2e",
                    "excite_e2e_zaux",
                    "excite_e2e_moe_zsup", "excite_e2e_moe_pure",
                    "paramid_1f_trans", "paramid_2f_trans", "paramid_3f_trans",
                    "paramid_1f_mve", "paramid_plus_excite"):
        x_tr, y_tr, x_te, y_te, kind = build_inputs(baseline, bundle, split)
    elif baseline == "paramid_1f":
        # Naive practitioner: trains a 1-D regression where the INPUT axis is
        # `_INPUT_FACTOR_COL` (mp by default, or fv if --input-factor=fv) and
        # the training subset is restricted to points where the OTHER factor
        # is near its nominal value (i.e. the practitioner only swept input on
        # a fixed nominal value of the unknown factor).
        col_in  = _INPUT_FACTOR_COL          # 0=mp, 1=fv  → input axis
        col_oth = 1 - col_in                  # the "fixed at nominal" axis
        if args.one_d_gains_dir and args.one_d_factor_file:
            od = load_1d_payload_sweep(args.one_d_gains_dir, args.one_d_factor_file)
            print(f"1D payload sweep: {len(od)} points, m_p range "
                  f"[{od.m_p.min():.3f},{od.m_p.max():.3f}]")
            x_tr = od.m_p.reshape(-1, 1)
            y_tr = od.gains
        elif args.paramid_1f_fv_tol is not None:
            # Synthesize the practitioner 1-D sweep by grabbing dense grid
            # points whose UNKNOWN factor is near `args.paramid_1f_nominal`
            # and only those NOT in the held-out test set.
            other_vals = bundle.phys_factors[:, col_oth]
            nominal = args.paramid_1f_nominal
            near_default = np.abs(other_vals - nominal) <= args.paramid_1f_fv_tol
            in_train = np.zeros(len(bundle), dtype=bool)
            in_train[split.train_idx] = True
            sel = np.where(near_default & in_train)[0]
            if len(sel) < 5:
                raise SystemExit(
                    f"paramid_1f synthesized train set too small: "
                    f"{len(sel)} points with |fac{col_oth}-{nominal}|<={args.paramid_1f_fv_tol} "
                    f"inside the train split"
                )
            other_name = "m_p" if col_oth == 0 else "fv_mult"
            input_name = "m_p" if col_in == 0 else "fv_mult"
            print(f"paramid_1f synthesized 1D sweep: {len(sel)} dense-grid points "
                  f"with |{other_name}-{nominal}|<={args.paramid_1f_fv_tol}, "
                  f"input axis = {input_name}")
            x_tr = bundle.phys_factors[sel, col_in:col_in+1]
            y_tr = bundle.gains[sel]
        else:
            raise SystemExit(
                "paramid_1f needs either --one-d-gains-dir + --one-d-factor-file, "
                "or --paramid-1f-fv-tol (synthesize from dense grid near nominal)"
            )
        # Test set = entire held-out band on dense 2D grid.  We evaluate on
        # every dense test point (not just nominal-other-factor ones) — this
        # is what makes the OOD drop dramatic.  Test inputs use the same
        # input axis as training (col_in).
        x_te = bundle.phys_factors[split.test_idx, col_in:col_in+1]
        y_te = bundle.gains[split.test_idx]
        kind = "vec"
    else:
        raise SystemExit(baseline)

    # ---- Normalise inputs (identity for excitation, per-dim stats for vec)
    if kind == "vec":
        x_std = Standardiser(x_tr)
        x_tr_n = x_std(x_tr).astype(np.float32)
        x_te_n = x_std(x_te).astype(np.float32)
    else:
        # Standardise per-channel over train-set time×batch (kind in {seq, seq_zaux, moe}).
        seq_x_mean = x_tr.mean(axis=(0, 1), keepdims=True)
        seq_x_std = x_tr.std(axis=(0, 1), keepdims=True) + 1e-6
        x_tr_n = ((x_tr - seq_x_mean) / seq_x_std).astype(np.float32)
        x_te_n = ((x_te - seq_x_mean) / seq_x_std).astype(np.float32)

    # ---- For z-supervised baselines: build a normalised m_p target -------
    z_target_train_t = None
    needs_zsup = (
        (kind == "moe" and baseline == "excite_e2e_moe_zsup")
        or (kind == "seq_zaux" and baseline == "excite_e2e_zaux")
    )
    if needs_zsup:
        mp_train = bundle.phys_factors[split.train_idx, 0].astype(np.float32)
        mp_mean = float(mp_train.mean())
        mp_sd = float(mp_train.std() + 1e-6)
        z_target_train = ((mp_train - mp_mean) / mp_sd).reshape(-1, 1).astype(np.float32)
        z_target_train_t = torch.from_numpy(z_target_train)
        print(f"z-supervision: target = standardized m_p "
              f"(mean={mp_mean:.4f} std={mp_sd:.4f})  lambda={args.moe_lambda_z_sup}")

    # Standardise targets — keep stats for inverse transform at eval time.
    y_std = Standardiser(y_tr)
    y_tr_n = y_std(y_tr).astype(np.float32)
    y_te_n = y_std(y_te).astype(np.float32)

    # --- Degenerate-dim detection (loss weighting) ---
    # A gain dim is "degenerate" if its RAW std is very small (≈ 0 constant)
    # or if ≥ 80% of training values are below an absolute threshold.
    # On CasADi F=3: Kd_1 (84% zero, std=0.14) and Kd_4 (100% zero, std=0.008)
    # qualify. These dims amplify in std-normalised MSE due to tiny std
    # (raw err 0.01 → std err 1.3+), hurting convergence for no signal.
    raw_std = y_tr.std(axis=0)
    frac_zero = (np.abs(y_tr) < 0.1).mean(axis=0)
    degenerate_mask = (raw_std < 0.05) | (frac_zero > 0.80)
    if degenerate_mask.any():
        print(f"[loss-weighting] degenerate dims (std<0.05 or ≥80% near-zero): "
              f"{np.where(degenerate_mask)[0].tolist()} (zeroed in loss)")
    loss_weight_np = (~degenerate_mask).astype(np.float32)
    loss_weight_t = torch.from_numpy(loss_weight_np)

    x_tr_t = torch.from_numpy(x_tr_n)
    y_tr_t = torch.from_numpy(y_tr_n)
    x_te_t = torch.from_numpy(x_te_n)
    y_te_t = torch.from_numpy(y_te_n)

    # ---- Compute gain stats for non-negative output constraint ------------
    nonneg_kwargs = {}
    if args.nonneg_output:
        gain_mean = y_std.mean.flatten().astype(np.float32)
        gain_std = y_std.std.flatten().astype(np.float32)
        nonneg_kwargs = dict(gain_mean=gain_mean, gain_std=gain_std)
        print(f"non-negative output enabled (sharp softplus β=20)")

    # ---- Build model ------------------------------------------------------
    gain_dim = int(y_tr.shape[1])
    if kind == "vec":
        in_dim = x_tr_t.shape[1]
        model = MLPHead(in_dim=in_dim, out_dim=gain_dim, hidden=args.hidden,
                         **nonneg_kwargs)
    elif kind == "seq":
        in_dim = x_tr_t.shape[-1]
        model = TransformerEncoderHead(
            in_dim=in_dim,
            out_dim=gain_dim,
            d_model=args.d_model,
            nhead=args.nhead,
            nlayers=args.nlayers,
            ff=args.ff,
            max_len=x_tr_t.shape[1] + 8,
            **nonneg_kwargs,
        )
    elif kind == "seq_zaux":
        in_dim = x_tr_t.shape[-1]
        model = TransformerEncoderHead(
            in_dim=in_dim,
            out_dim=gain_dim,
            d_model=args.d_model,
            nhead=args.nhead,
            nlayers=args.nlayers,
            ff=args.ff,
            max_len=x_tr_t.shape[1] + 8,
            use_z_aux=True,
            **nonneg_kwargs,
        )
    elif kind == "seq_mve":
        in_dim = x_tr_t.shape[-1]
        model = TransformerMVEHead(
            in_dim=in_dim,
            out_dim=gain_dim,
            d_model=args.d_model,
            nhead=args.nhead,
            nlayers=args.nlayers,
            ff=args.ff,
            max_len=x_tr_t.shape[1] + 8,
            **nonneg_kwargs,
        )
    elif kind == "moe":
        in_dim = x_tr_t.shape[-1]
        T = x_tr_t.shape[1]
        # n_joints default to dof; dims_per_joint inferred from gain_dim.
        # Supports both 3*J (Kp,Ki,Kd) and 2*J (Kp,Kd with Ki constrained).
        n_joints = bundle.n_joints if bundle.n_joints is not None else in_dim
        if gain_dim % n_joints != 0:
            raise SystemExit(f"gain_dim={gain_dim} not divisible by n_joints={n_joints}")
        dims_per_joint = gain_dim // n_joints
        model = PerJointMoEHead(
            in_dim=in_dim,
            n_joints=n_joints,
            dims_per_joint=dims_per_joint,
            d_model=args.moe_d_model,
            nhead=args.moe_nhead,
            nlayers=args.moe_nlayers,
            max_len=T + 8,
            n_experts=args.moe_n_experts,
            expert_hidden=args.moe_expert_hidden,
            proj_dim=args.moe_d_model,
            **nonneg_kwargs,
            temperature=args.moe_temperature,
            noise_eps=args.moe_noise_eps,
        )
    else:
        raise SystemExit(f"unknown kind: {kind}")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: {model.__class__.__name__}  params={n_params/1e6:.3f}M")

    # ---- Train ------------------------------------------------------------
    log_path = os.path.join(args.out_dir, "train.log")
    with open(log_path, "w") as fp:
        fp.write(f"# baseline={args.baseline}\n")
        fp.write(f"# train_n={len(x_tr_t)} test_n={len(x_te_t)}\n")
        fp.write(f"# heldout_fv=[{args.heldout_fv_lo},{args.heldout_fv_hi}]\n")
        t0 = time.time()
        if kind in ("moe", "seq_zaux"):
            # Shared auxiliary-loss training loop.  Works for both
            # PerJointMoEHead (z_pred shape (B, J)) and the TransformerEncoder
            # with use_z_aux=True (z_pred shape (B, 1)); the broadcast in
            # train_loop_moe handles both cases.
            lam = args.moe_lambda_z_sup if baseline in (
                "excite_e2e_moe_zsup", "excite_e2e_zaux") else 0.0
            model = train_loop_moe(
                model,
                x_tr_t, y_tr_t, z_target_train_t,
                x_te_t, y_te_t,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                weight_decay=args.weight_decay,
                lambda_z_sup=lam,
                device=device,
                log_fp=fp,
            )
        elif kind == "seq_mve":
            model = train_loop_mve(
                model,
                x_tr_t, y_tr_t, x_te_t, y_te_t,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                weight_decay=args.weight_decay,
                device=device,
                log_fp=fp,
                loss_weight=loss_weight_t,
                track_best=True,
            )
        else:
            model = train_loop(
                model,
                x_tr_t, y_tr_t, x_te_t, y_te_t,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                weight_decay=args.weight_decay,
                device=device,
                log_fp=fp,
                loss_weight=loss_weight_t,
                track_best=True,
                val_frac=args.val_frac,
                val_seed=args.val_seed,
            )
        fp.write(f"# wallclock_sec={time.time()-t0:.1f}\n")

    # ---- Dump predictions -------------------------------------------------
    # Batched inference — single-shot forward on large N allocates
    # MPS attention matrices O(N*T^2) that can OOM on F=3 (N≈2700).
    def _batched_infer(model, x_tensor, bs=128):
        outs = []
        for i in range(0, x_tensor.size(0), bs):
            outs.append(model(x_tensor[i:i+bs].to(device)).detach().cpu())
        return torch.cat(outs, dim=0).numpy()

    model.eval()
    with torch.no_grad():
        pred_tr = _batched_infer(model, x_tr_t)
        pred_te = _batched_infer(model, x_te_t)
    if device.type == "mps":
        torch.mps.empty_cache()
    pred_tr_raw = y_std.inv(pred_tr)
    pred_te_raw = y_std.inv(pred_te)

    # Primary metric: MSE in standardised gain space (matches run_zaux_ablation.py
    # and makes cross-experiment comparisons well-defined).
    gain_mse_test = float(np.mean((pred_te - y_te_n) ** 2))
    gain_mse_test_raw = float(np.mean((pred_te_raw - y_te) ** 2))
    print(f"gain_mse_test (standardised) = {gain_mse_test:.6f}  "
          f"raw = {gain_mse_test_raw:.6f}  "
          "(NB: gain space is plateaued; use closed-loop cost for final eval)")

    out_npz = os.path.join(args.out_dir, "predictions.npz")
    if baseline == "paramid_1f":
        # For paramid_1f the "training phys_factors" come from the 1-D source
        # (payloads only); pad the fv_mult column with zeros as a sentinel.
        phys_train_dump = np.stack(
            [x_tr[:, 0], np.zeros_like(x_tr[:, 0])], axis=1
        ).astype(np.float32)
    else:
        phys_train_dump = bundle.phys_factors[split.train_idx].astype(np.float32)

    np.savez(
        out_npz,
        pred_gains_test=pred_te_raw.astype(np.float32),
        true_gains_test=y_te.astype(np.float32),
        phys_test=bundle.phys_factors[split.test_idx].astype(np.float32),
        pred_gains_train=pred_tr_raw.astype(np.float32),
        true_gains_train=y_tr.astype(np.float32),
        phys_train=phys_train_dump,
        gain_mse_test=np.float32(gain_mse_test),            # standardised
        gain_mse_test_raw=np.float32(gain_mse_test_raw),    # raw (for reference)
        train_idx=split.train_idx.astype(np.int64),
        test_idx=split.test_idx.astype(np.int64),
        baseline=np.array(baseline),
        split_mode=np.array(args.split_mode),
        heldout_fv_lo=np.float32(args.heldout_fv_lo if args.heldout_fv_lo is not None else -1.0),
        heldout_fv_hi=np.float32(args.heldout_fv_hi if args.heldout_fv_hi is not None else -1.0),
        split_seed=np.int64(args.split_seed),
        test_frac=np.float32(args.test_frac),
    )
    print(f"wrote {out_npz}")

    # Save model weights for reproducibility / rebuttal re-analysis
    weights_path = os.path.join(args.out_dir, "model.pt")
    ckpt = {
        "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
        "baseline": baseline,
        "input_kind": kind,
        "in_dim": int(x_tr_t.shape[-1] if x_tr_t.dim() == 3 else x_tr_t.shape[1]),
        "out_dim": int(gain_dim),
        "model_class": model.__class__.__name__,
        "d_model": getattr(args, "d_model", None),
        "nhead": getattr(args, "nhead", None),
        "nlayers": getattr(args, "nlayers", None),
        "ff": getattr(args, "ff", None),
        "y_mean": y_std.mean.astype(np.float32),
        "y_std": y_std.std.astype(np.float32),
        "val_frac": float(getattr(args, "val_frac", 0.0)),
        "val_seed": int(getattr(args, "val_seed", 0)),
        "seed": int(args.seed),
        "epochs": int(args.epochs),
    }
    # x_mean/x_std for inference. vec kind uses Standardiser; seq kind uses inline.
    _x_std_obj = locals().get("x_std", None)
    if _x_std_obj is not None:
        try:
            ckpt["x_mean"] = _x_std_obj.mean.astype(np.float32)
            ckpt["x_std"] = _x_std_obj.std.astype(np.float32)
        except Exception:
            pass
    elif "seq_x_mean" in locals() and "seq_x_std" in locals():
        ckpt["x_mean"] = seq_x_mean.astype(np.float32)
        ckpt["x_std"] = seq_x_std.astype(np.float32)
        ckpt["x_signal"] = str(args.bundle_signal)
    torch.save(ckpt, weights_path)
    print(f"wrote {weights_path}")

    # Also dump a small json summary for quick diffing across baselines
    summary = {
        "baseline": baseline,
        "split_mode": args.split_mode,
        "train_n": int(len(x_tr_t)),
        "test_n": int(len(x_te_t)),
        "heldout_fv_lo": args.heldout_fv_lo,
        "heldout_fv_hi": args.heldout_fv_hi,
        "split_seed": args.split_seed,
        "test_frac": args.test_frac,
        "gain_mse_test": gain_mse_test,
        "n_params": int(n_params),
    }
    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # Full training config: every CLI arg + resolved model hyperparams + env info.
    # Lets later experiments load and replay exact params (see --load-config in
    # the launchers).
    train_config = {k: (str(v) if not isinstance(v, (int, float, bool, str, list, dict, type(None))) else v)
                     for k, v in vars(args).items()}
    train_config["pytorch_version"] = torch.__version__
    train_config["device_type"] = str(device)
    train_config["n_params"] = int(n_params)
    train_config["t_use_computed"] = int(bundle.excitation.shape[1]) if bundle.excitation is not None else None
    train_config["trans_config_effective"] = {
        "d_model": args.d_model, "nhead": args.nhead,
        "nlayers": args.nlayers, "ff": args.ff,
    }
    with open(os.path.join(args.out_dir, "train_config.json"), "w") as f:
        json.dump(train_config, f, indent=2)


if __name__ == "__main__":
    main()
