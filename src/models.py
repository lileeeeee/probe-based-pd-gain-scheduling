"""
Small, identical-capacity heads for the 4 baselines.

Capacity is deliberately matched so the eventual comparison bars are
about *what information the model is allowed to see*, not about model
size.  All heads produce a (B, 21)-dim gain vector.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class _NonNegOutputMixin:
    """Round-trip non-negativity constraint for heads that train in a
    standardised target space:

        standardised → raw → softplus(raw) → standardised

    The wrapped head's training loss (computed in standardised space) sees an
    output whose corresponding raw gain is guaranteed to be non-negative.

    Subclasses must call ``self._init_nonneg(out_dim, gain_mean, gain_std)`` in
    their ``__init__`` and call ``self._apply_nonneg(y)`` on the final
    standardised output before returning.
    """

    def _init_nonneg(self, out_dim, gain_mean, gain_std):
        if gain_mean is None or gain_std is None:
            self._nonneg_enabled = False
            return
        self._nonneg_enabled = True
        gm = torch.as_tensor(gain_mean, dtype=torch.float32).reshape(out_dim)
        gs = torch.as_tensor(gain_std, dtype=torch.float32).reshape(out_dim)
        # Avoid divide-by-zero on flat gain dimensions.
        gs = torch.clamp(gs, min=1e-6)
        self.register_buffer("nonneg_gain_mean", gm)
        self.register_buffer("nonneg_gain_std", gs)

    def _apply_nonneg(self, y):
        """y: (B, out_dim) standardised prediction → safe standardised.

        Uses sharp softplus ``softplus(β·raw)/β`` with β=20 so that small
        positive gains (e.g. Kd_4 ≈ 0.005) are not pushed up to log(2).
        At raw≈0 the offset is ~0.035; for raw < -0.5 the value is essentially 0.
        """
        if not getattr(self, "_nonneg_enabled", False):
            return y
        raw = y * self.nonneg_gain_std + self.nonneg_gain_mean
        beta = 20.0
        raw = F.softplus(beta * raw) / beta
        return (raw - self.nonneg_gain_mean) / self.nonneg_gain_std


class MLPHead(nn.Module, _NonNegOutputMixin):
    """2-layer MLP.  Used for 1f_indist, 1f_ood, 2f_oracle."""

    def __init__(self, in_dim: int, out_dim: int = 21, hidden: int = 256,
                 gain_mean=None, gain_std=None):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, out_dim),
        )
        self._init_nonneg(out_dim, gain_mean, gain_std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._apply_nonneg(self.net(x))


class LSTMEncoderHead(nn.Module, _NonNegOutputMixin):
    """Bidirectional LSTM over the excitation response.

    Drop-in replacement for TransformerEncoderHead: same (B, T, dof) → (B, out_dim).

    Supports three pooling strategies:
      - "mean": mean-pool over time (naive baseline)
      - "last": concat final forward hidden with final backward hidden (standard BiLSTM)
      - "attn": learned attention-pool over time (strongest)
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int = 21,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        pool: str = "attn",
        use_z_aux: bool = False,
        z_dim: int = 1,
        gain_mean=None,
        gain_std=None,
    ):
        super().__init__()
        assert pool in ("mean", "last", "attn")
        self.pool = pool
        self._init_nonneg(out_dim, gain_mean, gain_std)
        self.lstm = nn.LSTM(
            input_size=in_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        pool_dim = hidden_dim * 2  # bidirectional
        if pool == "attn":
            # Simple additive attention: score_t = v^T tanh(W h_t)
            self.attn_proj = nn.Linear(pool_dim, pool_dim)
            self.attn_v = nn.Linear(pool_dim, 1, bias=False)
        self.head = nn.Sequential(
            nn.LayerNorm(pool_dim),
            nn.Linear(pool_dim, pool_dim),
            nn.GELU(),
            nn.Linear(pool_dim, out_dim),
        )
        self.use_z_aux = use_z_aux
        self.z_dim = z_dim
        if use_z_aux:
            self.z_head = nn.Sequential(
                nn.LayerNorm(pool_dim),
                nn.Linear(pool_dim, 64),
                nn.Tanh(),
                nn.Linear(64, z_dim),
            )
        else:
            self.z_head = None

    def _pool(self, out, h_n):
        """out: (B, T, 2*h); h_n: (2*L, B, h) final hidden per layer per direction."""
        if self.pool == "mean":
            return out.mean(dim=1)                                # (B, 2h)
        elif self.pool == "last":
            # h_n shape: (num_layers*2, B, hidden)
            # Last layer forward = h_n[-2], last layer backward = h_n[-1]
            fwd_last = h_n[-2]                                     # (B, h)
            bwd_last = h_n[-1]                                     # (B, h)
            return torch.cat([fwd_last, bwd_last], dim=-1)         # (B, 2h)
        else:  # attn
            # Score each timestep and softmax
            scores = self.attn_v(torch.tanh(self.attn_proj(out)))  # (B, T, 1)
            weights = torch.softmax(scores, dim=1)                 # (B, T, 1)
            return (out * weights).sum(dim=1)                      # (B, 2h)

    def forward(self, x: torch.Tensor, return_z: bool = False):
        # x: (B, T, dof)
        out, (h_n, _) = self.lstm(x)         # out: (B, T, 2*hidden); h_n: (2L, B, h)
        pooled = self._pool(out, h_n)
        y = self._apply_nonneg(self.head(pooled))
        if return_z:
            if self.z_head is None:
                raise RuntimeError("return_z=True but use_z_aux=False")
            z = self.z_head(pooled)
            return y, z
        return y


class TransformerMVEHead(nn.Module, _NonNegOutputMixin):
    """Transformer encoder + Mean-Variance Estimator (Gaussian NLL training).

    Bayesian uncertainty-aware variant of TransformerEncoderHead.
    Outputs per-dim $(\\mu, \\log \\sigma^2)$ trained with Gaussian negative
    log-likelihood. At inference, returns $\\mu$ only (drop-in compatible
    with the rest of the pipeline). Pass ``return_mve=True`` to get the
    $(\\mu, \\log \\sigma^2)$ pair needed for NLL in the training loop.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int = 21,
        d_model: int = 128,
        nhead: int = 4,
        nlayers: int = 3,
        max_len: int = 2048,
        ff: int = 256,
        log_sigma2_init: float = -2.0,
        gain_mean=None,
        gain_std=None,
    ):
        super().__init__()
        self._init_nonneg(out_dim, gain_mean, gain_std)
        self.out_dim = out_dim
        self.input_proj = nn.Linear(in_dim, d_model)
        self.pos = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ff,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=nlayers)
        self.mean_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, ff),
            nn.GELU(),
            nn.Linear(ff, out_dim),
        )
        self.logvar_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, ff),
            nn.GELU(),
            nn.Linear(ff, out_dim),
        )
        # Start logvar as a small constant so precision is well-behaved early;
        # zero weight on the last linear delays learning-rate impact on σ.
        nn.init.constant_(self.logvar_head[-1].bias, log_sigma2_init)
        nn.init.zeros_(self.logvar_head[-1].weight)

    def forward(self, x: torch.Tensor, return_mve: bool = False):
        B, T, _ = x.shape
        h = self.input_proj(x) + self.pos[:, :T]
        h = self.encoder(h)
        pooled = h.mean(dim=1)
        mu = self._apply_nonneg(self.mean_head(pooled))
        if return_mve:
            logvar = self.logvar_head(pooled)
            return mu, logvar
        return mu


class TransformerEncoderHead(nn.Module, _NonNegOutputMixin):
    """Tiny transformer encoder over the excitation response.

    Input:  (B, T, dof)  — joint-state sequence from the probe rollout
    Output: (B, 21)      — predicted gains
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int = 21,
        d_model: int = 128,
        nhead: int = 4,
        nlayers: int = 3,
        max_len: int = 2048,
        ff: int = 256,
        use_z_aux: bool = False,
        z_dim: int = 1,
        gain_mean=None,
        gain_std=None,
    ):
        super().__init__()
        self._init_nonneg(out_dim, gain_mean, gain_std)
        self.input_proj = nn.Linear(in_dim, d_model)
        self.pos = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ff,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=nlayers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, ff),
            nn.GELU(),
            nn.Linear(ff, out_dim),
        )
        # Optional auxiliary head to predict normalized physics factors from
        # the same pooled feature.  z_dim controls the target dimensionality:
        #   z_dim=1  →  scalar (m_p only, or fv only)
        #   z_dim=2  →  vector (m_p, fv_mult)
        self.use_z_aux = use_z_aux
        self.z_dim = z_dim
        if use_z_aux:
            self.z_head = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, 64),
                nn.Tanh(),
                nn.Linear(64, z_dim),
            )
        else:
            self.z_head = None

    def forward(self, x: torch.Tensor, return_z: bool = False):
        # x: (B, T, dof)
        B, T, _ = x.shape
        h = self.input_proj(x) + self.pos[:, :T]
        h = self.encoder(h)
        pooled = h.mean(dim=1)              # mean-pool over time
        y = self._apply_nonneg(self.head(pooled))
        if return_z:
            if self.z_head is None:
                raise RuntimeError("return_z=True but TransformerEncoderHead "
                                   "was built with use_z_aux=False")
            z = self.z_head(pooled)         # (B, 1)
            return y, z
        return y


# ===========================================================================
# Per-Joint GMM-routed Mixture-of-Experts
# Ported (and minimally adapted) from MOE_payload_supervised_z.py::PerJointMoE
# Adaptations vs the original:
#   - Input convention is (B, T, dof) to match the rest of baselines/models.py
#     instead of the original (B, dof, T).  No internal permute.
#   - Output gain ordering matches `gain_names` from the bundled dataset:
#     [Kp_1..J, Ki_1..J, Kd_1..J].
#   - SortedRandomBatchSampler / neighbor_kl_smooth / load_balance_loss /
#     temperature annealing / hard routing are NOT included by default — they
#     can be added back if the pure-GMM version underperforms.
# ===========================================================================

_LOG_2PI = float(np.log(2.0 * np.pi))


def _gaussian_logpdf(z: torch.Tensor, mu: torch.Tensor, log_sigma: torch.Tensor):
    """z: (B,1)  mu/log_sigma: (E,)  ->  (B,E)"""
    inv_sigma = torch.exp(-log_sigma)
    diff = (z - mu)
    quad = (diff * inv_sigma) ** 2
    return -0.5 * quad - log_sigma - 0.5 * _LOG_2PI


def _sinusoid_table(n_position: int, d_hid: int) -> torch.Tensor:
    pos = torch.arange(n_position, dtype=torch.float32).unsqueeze(1)   # (T,1)
    i = torch.arange(d_hid, dtype=torch.float32).unsqueeze(0)           # (1,D)
    angle = pos / (10000.0 ** (2 * (i // 2) / d_hid))
    angle[:, 0::2] = torch.sin(angle[:, 0::2])
    angle[:, 1::2] = torch.cos(angle[:, 1::2])
    return angle.unsqueeze(0)  # (1, T, D)


class _JointExpertMLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int = 3, hidden: int = 256, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, h):
        return self.net(h)


class PerJointMoEHead(nn.Module, _NonNegOutputMixin):
    """
    Per-joint GMM-routed MoE excitation head.

    Input:  (B, T, dof)
    Output: y of shape (B, n_joints * dims_per_joint)
            with ordering:
              dims_per_joint=3: [Kp_1..J, Ki_1..J, Kd_1..J]
              dims_per_joint=2: [Kp_1..J, Kd_1..J]  (Ki constrained externally)

    If `return_z=True` is passed in forward, also returns the per-joint
    latent z values (B, J), which can be supervised against an external
    scalar (e.g. normalized payload) to encourage payload-aligned GMM modes.
    """

    def __init__(
        self,
        in_dim: int = 4,
        n_joints: int = 4,
        dims_per_joint: int = 3,
        d_model: int = 256,
        nhead: int = 8,
        nlayers: int = 4,
        ff: int | None = None,
        dropout: float = 0.1,
        max_len: int = 256,
        n_experts: int = 4,
        expert_hidden: int = 256,
        proj_dim: int = 256,
        temperature: float = 1.0,
        noise_eps: float = 5e-4,
        gmm_init_spread: float = 2.0,
        z_dim: int = 1,
        gain_mean=None,
        gain_std=None,
    ):
        super().__init__()
        assert dims_per_joint in (2, 3), "dims_per_joint must be 2 (Kp, Kd) or 3 (Kp, Ki, Kd)"
        self.n_joints = n_joints
        self.dims_per_joint = dims_per_joint
        self.output_dim = n_joints * dims_per_joint
        self.n_experts = n_experts
        self.temperature = temperature
        self.noise_eps = noise_eps
        self.z_dim = z_dim
        self._init_nonneg(self.output_dim, gain_mean, gain_std)

        self.input_linear = nn.Linear(in_dim, d_model)
        self.register_buffer("pos_encoding", _sinusoid_table(max_len, d_model))

        # Default FF = 4x d_model (matches PyTorch default behaviour downsized
        # from 2048 to something proportional to d_model). Pass ff=None for
        # 4*d_model or override explicitly.
        dim_ff = ff if ff is not None else 4 * d_model
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            activation="relu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=nlayers)

        self.proj = nn.Sequential(
            nn.Linear(d_model, proj_dim),
            nn.Tanh(),
        )

        # Per-joint latent z_j(x).  z_dim controls target dimensionality.
        # For GMM routing we always use the first dimension of z.
        self.z_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(proj_dim, 64),
                nn.Tanh(),
                nn.Linear(64, z_dim),
            )
            for _ in range(n_joints)
        ])

        # Per-joint GMM parameters: (J, E)
        self.pi_logits = nn.Parameter(torch.zeros(n_joints, n_experts))
        mus = torch.linspace(-gmm_init_spread, gmm_init_spread, n_experts)
        self.mu = nn.Parameter(mus.unsqueeze(0).repeat(n_joints, 1))
        self.log_sigma = nn.Parameter(torch.full((n_joints, n_experts), -0.2))

        # n_joints × n_experts expert MLPs, each producing (Kp, Ki, Kd) for one joint.
        self.experts = nn.ModuleList([
            nn.ModuleList([
                _JointExpertMLP(proj_dim, out_dim=dims_per_joint, hidden=expert_hidden)
                for _ in range(n_experts)
            ])
            for _ in range(n_joints)
        ])

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, dof) -> h: (B, proj_dim) (last-token pooling)."""
        B, T, _ = x.shape
        h = self.input_linear(x) + self.pos_encoding[:, :T]
        h = self.encoder(h)
        h = h[:, -1, :]      # last-token pool, matching original PerJointMoE
        h = self.proj(h)
        return h

    def forward(
        self,
        x: torch.Tensor,
        return_z: bool = False,
        return_gate_probs: bool = False,
    ):
        h = self.encode(x)
        B = h.size(0)
        y = h.new_zeros(B, self.output_dim)
        z_list = []
        g_list = []

        for j in range(self.n_joints):
            z_j = self.z_heads[j](h)                    # (B, z_dim)
            if self.training and self.noise_eps > 0:
                z_j = z_j + self.noise_eps * torch.randn_like(z_j)

            # GMM routing always uses the first dimension of z
            z_route = z_j[:, :1]                         # (B, 1)
            log_p = _gaussian_logpdf(z_route, self.mu[j], self.log_sigma[j])  # (B, E)
            log_w = log_p + self.pi_logits[j].unsqueeze(0)
            tau = max(self.temperature, 1e-6)
            weights = F.softmax(log_w / tau, dim=-1)    # (B, E)

            exp_outs = torch.stack(
                [self.experts[j][k](h) for k in range(self.n_experts)],
                dim=1,
            )                                            # (B, E, dims_per_joint)
            yj = torch.sum(weights.unsqueeze(-1) * exp_outs, dim=1)  # (B, dims_per_joint)

            # Layout:
            #   dims_per_joint=3: [Kp_1..J, Ki_1..J, Kd_1..J]
            #   dims_per_joint=2: [Kp_1..J, Kd_1..J]
            if self.dims_per_joint == 3:
                y[:, j] = yj[:, 0]
                y[:, j + self.n_joints] = yj[:, 1]
                y[:, j + 2 * self.n_joints] = yj[:, 2]
            else:  # 2
                y[:, j] = yj[:, 0]
                y[:, j + self.n_joints] = yj[:, 1]

            if return_z:
                z_list.append(z_j)
            if return_gate_probs:
                g_list.append(weights.unsqueeze(1))     # (B, 1, E)

        # Apply non-negative round-trip on the assembled gain vector
        y = self._apply_nonneg(y)

        out = (y,)
        if return_z:
            out = out + (torch.cat(z_list, dim=1),)     # (B, J)
        if return_gate_probs:
            out = out + (torch.cat(g_list, dim=1),)     # (B, J, E)
        return out if len(out) > 1 else y
