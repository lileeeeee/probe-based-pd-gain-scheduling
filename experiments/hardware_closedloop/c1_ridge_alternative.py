"""C1 alternative: replace ImplicitID encoder's nonlinear gain head with
a sim-fitted Ridge regression. Mirrors the m_p-prediction pipeline of
phase1_q_predictions.py, just predicting 8-dim PD gain instead of scalar
m_p.

For each of 5 seeds:
  1. Load q-based F=3 encoder body (no gain head used).
  2. Extract pooled features on sim training set (random_position window
     0..T_use, sim training z-score).
  3. Fit Ridge: sim features → sim BO PD gain (no LOPO; ridge is just a
     drop-in linear head).
  4. Forward encoder on real probe (10-20 s window, sim z-score) →
     features → ridge → predicted gain.

5-seed ensemble median per payload.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from models import TransformerEncoderHead  # noqa: E402

SEEDS = [42, 7, 123, 2025, 2026]
PAYLOADS_KG = [0.00, 0.38, 0.76]
PAYLOAD_DIRS = {0.00: "payload_0", 0.38: "payload_0_38", 0.76: "payload_0_76"}
JOINT_ORDER = ("e", "d", "c", "b")
Q_COLS = [f"q_alpha_axis_{j}" for j in JOINT_ORDER]
LO, HI = 10.0, 20.0
GAIN_NAMES = ["Kp_e", "Kp_d", "Kp_c", "Kp_b", "Kd_e", "Kd_d", "Kd_c", "Kd_b"]

REAL_DIR = REPO / "data" / "real_robot_25hz"
ENC_ROOT = REPO / "output" / "f3_casadi_iid_qpos_valfrac"
SIM_BUNDLE = REPO / "data" / "dataset_3d_envB_full_pd.npz"
DT = 0.04
RIDGE_ALPHA = 1.0


def to_pd(g):
    nJ = g.shape[-1] // 3
    return np.concatenate([g[..., :nJ] + DT * g[..., nJ:2*nJ], g[..., 2*nJ:]], axis=-1)


def load_encoder(model_path: Path):
    m = torch.load(model_path, map_location="cpu", weights_only=False)
    pos_len = m["state_dict"]["pos"].shape[1]
    enc = TransformerEncoderHead(
        in_dim=m["in_dim"], out_dim=m["out_dim"],
        d_model=m["d_model"], nhead=m["nhead"], nlayers=m["nlayers"],
        ff=m["ff"], max_len=pos_len,
    )
    sd = {k: v for k, v in m["state_dict"].items()
          if k not in ("nonneg_gain_mean", "nonneg_gain_std")}
    enc.load_state_dict(sd, strict=False)
    enc.eval()
    return enc, m


@torch.no_grad()
def encoder_pooled(enc, V):
    """V: (T, 4) float32 → pooled (160,)."""
    x = torch.from_numpy(V).unsqueeze(0)        # (1, T, 4)
    h = enc.input_proj(x) + enc.pos[:, :x.shape[1]]
    h = enc.encoder(h)
    return h.mean(dim=1).squeeze(0).numpy()


def windowed_q(df, lo_s, hi_s):
    t = df["sim_time_sec"].to_numpy()
    mask = ((t - t[0]) >= lo_s) & ((t - t[0]) <= hi_s)
    return df[Q_COLS].to_numpy(dtype=np.float32)[mask]


def main():
    # Sim training data
    with np.load(SIM_BUNDLE, allow_pickle=True) as f:
        sim_pos = f["random_position"].astype(np.float32)[:, :, 1:]
        T_use = sim_pos.shape[-1] // 2
        sim_pos = sim_pos[:, :, :T_use]                   # (3000, 4, T_use)
        sim_pos_t = np.transpose(sim_pos, (0, 2, 1))      # (3000, T_use, 4)
        gains_pd = to_pd(f["gains"].astype(np.float32))   # (3000, 8)

    q_mean = sim_pos_t.mean(axis=(0, 1), keepdims=True).astype(np.float32)
    q_std = sim_pos_t.std(axis=(0, 1), keepdims=True).astype(np.float32) + 1e-6
    sim_pos_n = ((sim_pos_t - q_mean) / q_std).astype(np.float32)
    print(f"Sim training set: {sim_pos_n.shape}, gains_pd: {gains_pd.shape}")

    out = {pl: [] for pl in PAYLOADS_KG}            # C1-ridge gains, list of (8,) per seed
    sim_test_r2 = []                                # held-out sim test R² per seed (sanity)

    for seed in SEEDS:
        # Load encoder (we'll only use its body, ignore gain head)
        enc, m_enc = load_encoder(ENC_ROOT / f"seed{seed}" / "excite_e2e" / "model.pt")
        d = np.load(ENC_ROOT / f"seed{seed}" / "excite_e2e" / "predictions.npz",
                     allow_pickle=True)
        train_idx = d["train_idx"]
        test_idx = d["test_idx"]

        # Encoder features on full sim set (we'll split later by train/test idx)
        feats_all = []
        with torch.no_grad():
            B = 16
            for i in range(0, len(sim_pos_n), B):
                x = torch.from_numpy(sim_pos_n[i:i+B])
                h = enc.input_proj(x) + enc.pos[:, :x.shape[1]]
                h = enc.encoder(h)
                feats_all.append(h.mean(dim=1).numpy())
        feats_all = np.concatenate(feats_all, axis=0)     # (3000, 160)

        # Fit Ridge on training split, evaluate on test split
        ridge = Ridge(alpha=RIDGE_ALPHA).fit(feats_all[train_idx], gains_pd[train_idx])
        pred_test = ridge.predict(feats_all[test_idx])
        true_test = gains_pd[test_idx]
        ss_res = ((pred_test - true_test) ** 2).sum()
        ss_tot = ((true_test - true_test.mean(axis=0)) ** 2).sum()
        r2 = 1 - ss_res / ss_tot
        sim_test_r2.append(r2)

        # Forward encoder on real probes
        for pl in PAYLOADS_KG:
            pdir = REAL_DIR / PAYLOAD_DIRS[pl]
            pass_gains = []
            for csv in sorted(pdir.glob("*.csv")):
                df = pd.read_csv(csv)
                V = windowed_q(df, LO, HI)                     # (T, 4)
                V_n = ((V - q_mean.reshape(1, -1)) / q_std.reshape(1, -1)).astype(np.float32)
                feat = encoder_pooled(enc, V_n)               # (160,)
                gain = ridge.predict(feat[None, :])[0]         # (8,)
                pass_gains.append(gain)
            out[pl].append(np.median(np.stack(pass_gains), axis=0))

    print(f"\nSim test R² per seed (ridge fit on encoder features → BO PD gain):")
    print(f"  {[f'{r:.3f}' for r in sim_test_r2]}, median {np.median(sim_test_r2):.3f}")

    # Aggregate
    print(f"\n{'Payload':<10} {'Method':<22} {'Kp_e':>8} {'Kp_d':>8} {'Kp_c':>8} {'Kp_b':>8} {'Kd_e':>7} {'Kd_d':>7} {'Kd_c':>7} {'Kd_b':>7}")
    print("-" * 104)
    final = {}
    for pl in PAYLOADS_KG:
        arr = np.array(out[pl])                              # (5, 8)
        med = np.median(arr, axis=0)
        final[pl] = {"median": med, "all_seeds": arr}
        cols = [f'{med[i]:>7.3f}' if i < 4 else f'{med[i]:>6.3f}' for i in range(8)]
        print(f'{pl:<10.2f} C1-ridge (encoder→Ridge) ' + '  '.join(cols))

    # Save
    save_dict = {
        "payloads_kg": np.array(PAYLOADS_KG),
        "seeds": np.array(SEEDS),
        "gain_names": np.array(GAIN_NAMES),
        "sim_test_r2": np.array(sim_test_r2),
    }
    for pl, res in final.items():
        key = f'C1ridge_payload_{pl:.2f}'.replace(".", "p")
        save_dict[f'{key}_median'] = res["median"]
        save_dict[f'{key}_all_seeds'] = res["all_seeds"]
    out_path = REPO / "data" / "hardware_closedloop_gains_c1ridge.npz"
    np.savez(out_path, **save_dict)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
