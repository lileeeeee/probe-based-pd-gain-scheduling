"""
Default model configurations and hyperparameters.

These are the best configs found through size sweeps on Env-B (epoch 8, ~224 samples).
"""
import torch

# ── Device ────────────────────────────────────────────────────────────
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

# ── Training defaults ─────────────────────────────────────────────────
SEED = 42
EPOCHS = 300
BATCH_SIZE = 32
LR = 1e-3
LAMBDA_ZAUX = 0.1
TEST_FRAC = 0.2
DURATION_FRAC = 0.5   # use 50% of excitation (T=100 out of 200); multi-seed optimal

# ── Transformer (trans_S) ────────────────────────────────────────────
TRANS_CONFIG = {
    "d_model": 160,
    "nhead": 4,
    "nlayers": 3,
    "ff": 320,
    # ~720K params
}

# ── MoE (moe_S) ──────────────────────────────────────────────────────
MOE_CONFIG = {
    "d_model": 64,
    "nhead": 4,
    "nlayers": 2,
    "n_experts": 4,
    "expert_hidden": 64,
    "proj_dim": 64,
    # ~200K params
}

# ── Z-aux configurations ─────────────────────────────────────────────
ZAUX_CONFIGS = {
    "none":    {"cols": [],     "z_dim": 0, "lam": 0.0},
    "mp_only": {"cols": [0],    "z_dim": 1, "lam": LAMBDA_ZAUX},
    "fv_only": {"cols": [1],    "z_dim": 1, "lam": LAMBDA_ZAUX},
    "both":    {"cols": [0, 1], "z_dim": 2, "lam": LAMBDA_ZAUX},
}

# ── OOD evaluation ───────────────────────────────────────────────────
OOD_N_BANDS = 5
OOD_MIN_TEST_POINTS = 15
