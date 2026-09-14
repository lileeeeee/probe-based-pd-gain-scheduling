"""Compute C1 (ImplicitID), C2 (Partial-1F-cheat), C3 (Fixed) gains for the
4 closed-loop hardware payloads {0.00, 0.38, 0.57, 0.76} kg.

C1: q-based F=3 ImplicitID encoder forwarded on real q-position probe traces
    (10-20 s window, sim-training-stat z-score).
C2: q-based F=3 paramid_1f_trans (input m_p) NN with actual m_p input.
C3: same as C2 but with m_p=0; one gain reused across all payloads.

5 seeds, ensemble median.
Output: data/hardware_closedloop_gains.npz
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from models import TransformerEncoderHead  # noqa: E402

SEEDS = [42, 7, 123, 2025, 2026]
PAYLOADS_KG = [0.00, 0.38, 0.57, 0.76]
PAYLOAD_DIRS = {0.00: "payload_0", 0.38: "payload_0_38",
                0.57: "payload_0_57", 0.76: "payload_0_76"}
JOINT_ORDER = ("e", "d", "c", "b")
Q_COLS = [f"q_alpha_axis_{j}" for j in JOINT_ORDER]
LO, HI = 10.0, 20.0   # window seconds (matches phase1_q_predictions)
GAIN_NAMES = ["Kp_e", "Kp_d", "Kp_c", "Kp_b", "Kd_e", "Kd_d", "Kd_c", "Kd_b"]

REAL_DIR = REPO / "data" / "real_robot_25hz"
ENC_ROOT = REPO / "output" / "f3_casadi_iid_qpos_valfrac"
SIM_BUNDLE = REPO / "data" / "dataset_3d_envB_full_pd.npz"  # PD-format F=3 dataset


def windowed_q(df, lo_s, hi_s):
    """Extract q (position) within [lo_s, hi_s] seconds of probe."""
    t = df["sim_time_sec"].to_numpy()
    mask = ((t - t[0]) >= lo_s) & ((t - t[0]) <= hi_s)
    return df[Q_COLS].to_numpy(dtype=np.float32)[mask]


def sim_q_stats():
    """Mirror sim_q_stats() in phase1_q_predictions.py."""
    with np.load(SIM_BUNDLE, allow_pickle=True) as f:
        sig = f["random_position"].astype(np.float32)[:, :, 1:]   # drop pre-probe step
        sig = sig[:, :, :sig.shape[-1] // 2]                       # duration_frac=0.5
        sig_t = np.transpose(sig, (0, 2, 1))                       # (N, T, 4)
    return (sig_t.mean(axis=(0, 1), keepdims=True).astype(np.float32),
            sig_t.std(axis=(0, 1), keepdims=True).astype(np.float32) + 1e-6)


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


def forward_encoder_gain(enc, V_n, y_mean, y_std):
    """Full forward: encoder outputs z-scored gain → un-z-score → raw gain."""
    with torch.no_grad():
        x = torch.from_numpy(V_n).unsqueeze(0)  # (1, T, 4)
        y_z = enc(x).squeeze(0).numpy()           # (8,) z-scored
    return y_z * y_std + y_mean                   # (8,) raw gain


def main():
    # Try PD-format dataset first; fall back to non-PD if missing
    bundle_path = SIM_BUNDLE if SIM_BUNDLE.exists() else REPO / "data" / "dataset_3d_envB_full.npz"
    print(f"Sim bundle: {bundle_path}")

    # Sim training-set q stats (for z-score normalization at deployment)
    with np.load(bundle_path, allow_pickle=True) as f:
        sig = f["random_position"].astype(np.float32)[:, :, 1:]
        sig = sig[:, :, :sig.shape[-1] // 2]
        sig_t = np.transpose(sig, (0, 2, 1))
        q_mean = sig_t.mean(axis=(0, 1), keepdims=True).astype(np.float32)
        q_std = sig_t.std(axis=(0, 1), keepdims=True).astype(np.float32) + 1e-6
    q_mean_4 = q_mean.reshape(1, -1)
    q_std_4 = q_std.reshape(1, -1)
    print(f"Sim q stats (mean / std per joint):")
    print(f"  mean = {q_mean.flatten().round(3)}")
    print(f"  std  = {q_std.flatten().round(3)}\n")

    # Load source phys for paramid_1f_trans seq normalization
    with np.load(bundle_path, allow_pickle=True) as f:
        phys_full = f["phys_factors"].astype(np.float32)            # (3000, 3)

    # Per-seed forward
    out = {pl: {"C1": [], "C2": [], "C3": []} for pl in PAYLOADS_KG}

    for seed in SEEDS:
        # ----- C1 ImplicitID -----
        enc, m_enc = load_encoder(ENC_ROOT / f"seed{seed}" / "excite_e2e" / "model.pt")
        y_mean_enc = np.asarray(m_enc["y_mean"]).squeeze()
        y_std_enc = np.asarray(m_enc["y_std"]).squeeze()
        for pl in PAYLOADS_KG:
            pdir = REAL_DIR / PAYLOAD_DIRS[pl]
            pass_gains = []
            for csv in sorted(pdir.glob("*.csv")):
                df = pd.read_csv(csv)
                V = windowed_q(df, LO, HI)                        # (T, 4)
                V_n = ((V - q_mean_4) / q_std_4).astype(np.float32)
                gain = forward_encoder_gain(enc, V_n, y_mean_enc, y_std_enc)
                pass_gains.append(gain)
            out[pl]["C1"].append(np.median(np.stack(pass_gains), axis=0))

        # ----- C2 Partial-1F-trans (input_factor=mp) & C3 Fixed -----
        m_p1f_path = ENC_ROOT / f"seed{seed}" / "paramid_1f_trans" / "model.pt"
        m_p1f = torch.load(m_p1f_path, map_location="cpu", weights_only=False)
        T_p1f_max = m_p1f["state_dict"]["pos"].shape[1]
        T_p1f = 500   # match training duration_frac=0.5
        net_p1f = TransformerEncoderHead(
            in_dim=m_p1f["in_dim"], out_dim=m_p1f["out_dim"],
            d_model=m_p1f["d_model"], nhead=m_p1f["nhead"], nlayers=m_p1f["nlayers"],
            ff=m_p1f["ff"], max_len=T_p1f_max,
        )
        sd_p1f = {k: v for k, v in m_p1f["state_dict"].items()
                  if k not in ("nonneg_gain_mean", "nonneg_gain_std")}
        net_p1f.load_state_dict(sd_p1f, strict=False)
        net_p1f.eval()
        y_mean_p1f = np.asarray(m_p1f["y_mean"]).squeeze()
        y_std_p1f = np.asarray(m_p1f["y_std"]).squeeze()

        # Re-compute training-time seq normalization for input_factor=mp
        d_p1f = np.load(ENC_ROOT / f"seed{seed}" / "paramid_1f_trans" / "predictions.npz",
                         allow_pickle=True)
        train_idx = d_p1f["train_idx"]
        mp_train = phys_full[train_idx, 0:1]            # (n_train, 1)
        seq_x_mean = mp_train.mean(axis=0, keepdims=True)[None, :, :]  # (1, 1, 1)
        seq_x_std = mp_train.std(axis=0, keepdims=True)[None, :, :] + 1e-6

        for pl in PAYLOADS_KG:
            for input_mp, key in [(pl, "C2"), (0.0, "C3")]:
                x_scalar = np.array([[input_mp]], dtype=np.float32)
                x_seq = np.broadcast_to(x_scalar[:, None, :], (1, T_p1f, 1)).copy()
                x_seq = (x_seq - seq_x_mean) / seq_x_std
                with torch.no_grad():
                    y_z = net_p1f(torch.tensor(x_seq.astype(np.float32))).numpy()
                y = y_z * y_std_p1f + y_mean_p1f
                out[pl][key].append(y[0])

    # Aggregate: median across 5 seeds
    print(f"\n{'Payload':<10} {'Baseline':<10} {'Kp_e':>8} {'Kp_d':>8} {'Kp_c':>8} {'Kp_b':>8} {'Kd_e':>7} {'Kd_d':>7} {'Kd_c':>7} {'Kd_b':>7}")
    print('-' * 100)
    final = {}
    for pl in PAYLOADS_KG:
        for ctrl in ["C1", "C2", "C3"]:
            arr = np.array(out[pl][ctrl])             # (5, 8)
            med = np.median(arr, axis=0)
            final[(pl, ctrl)] = {"median": med, "all_seeds": arr}
            cols = [f'{med[i]:>7.3f}' if i < 4 else f'{med[i]:>6.3f}' for i in range(8)]
            print(f'{pl:<10.2f} {ctrl:<10} ' + '  '.join(cols))

    save_dict = {
        "payloads_kg": np.array(PAYLOADS_KG),
        "seeds": np.array(SEEDS),
        "gain_names": np.array(GAIN_NAMES),
        "encoder_root": str(ENC_ROOT.relative_to(REPO)),
        "input_signal": "q (position)",
        "window_seconds": np.array([LO, HI]),
    }
    for (pl, ctrl), res in final.items():
        key = f'{ctrl}_payload_{pl:.2f}'.replace(".", "p")
        save_dict[f'{key}_median'] = res["median"]
        save_dict[f'{key}_all_seeds'] = res["all_seeds"]
    out_path = REPO / "data" / "hardware_closedloop_gains.npz"
    np.savez(out_path, **save_dict)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
