#!/usr/bin/env python3
"""Deployment-side inference timing for R1-P10 / R2-C10.

Times the ImplicitID readout exactly as deployed on hardware: the 5-seed
sim-trained encoder ensemble (Section 6.1, output/f3_casadi_iid_qpos_valfrac)
applied to ONE probe response of shape (T_probe=500, J=4), median over seeds.

Run this ON THE DEPLOYMENT LAPTOP (the machine that computes gains at the robot):
    python experiments/hardware_closedloop/time_inference.py [--device cpu|mps|cuda] [--reps 100]
It prints one JSON line; paste it into notes. Normalization constants do not
affect timing, so the input is a standardized random trace of the right shape.
"""
import argparse, json, platform, sys, time
from pathlib import Path
import numpy as np, torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src.models import TransformerEncoderHead  # noqa: E402

ENC_ROOT = REPO / "output" / "f3_casadi_iid_qpos_valfrac"
SEEDS = [42, 7, 123, 2025, 2026]
T_PROBE, J = 500, 4


def load_encoder(seed: int, device):
    ck = torch.load(ENC_ROOT / f"seed{seed}" / "excite_e2e" / "model.pt",
                    map_location="cpu", weights_only=False)
    assert ck["model_class"] == "TransformerEncoderHead", ck["model_class"]
    sd = ck["state_dict"]
    m = TransformerEncoderHead(in_dim=ck["in_dim"], out_dim=ck["out_dim"],
                               d_model=ck["d_model"], nhead=ck["nhead"],
                               nlayers=ck["nlayers"], ff=ck["ff"],
                               max_len=sd["pos"].shape[1])
    m.load_state_dict(sd)
    return m.to(device).eval(), np.asarray(ck["y_mean"]), np.asarray(ck["y_std"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--reps", type=int, default=100)
    a = ap.parse_args()
    dev = torch.device(a.device)
    torch.set_num_threads(max(1, torch.get_num_threads()))

    t0 = time.perf_counter()
    encs = [load_encoder(s, dev) for s in SEEDS]
    t_load = time.perf_counter() - t0

    x = torch.randn(1, T_PROBE, J, device=dev)          # one standardized probe trace
    sync = (lambda: torch.cuda.synchronize()) if dev.type == "cuda" else \
           (lambda: torch.mps.synchronize()) if dev.type == "mps" else (lambda: None)

    with torch.no_grad():
        for m, _, _ in encs: m(x); sync()             # warm-up
        single = []
        for _ in range(a.reps):
            t = time.perf_counter(); encs[0][0](x); sync(); single.append(time.perf_counter() - t)
        ens = []
        for _ in range(a.reps):
            t = time.perf_counter()
            outs = [m(x) for m, _, _ in encs]; sync()
            g = torch.median(torch.stack(outs), dim=0).values          # ensemble median (protocol C1)
            g = g.cpu().numpy() * encs[0][2] + encs[0][1]               # de-standardize -> 8 PD gains
            ens.append(time.perf_counter() - t)
    ms = lambda v: float(np.median(v) * 1e3)
    out = {
        "machine": platform.node(), "cpu": platform.processor() or platform.machine(),
        "device": str(dev), "torch": torch.__version__, "threads": torch.get_num_threads(),
        "n_seeds": len(encs), "input_shape": [1, T_PROBE, J], "reps": a.reps,
        "load_5_models_s": round(t_load, 3),
        "single_forward_ms_median": round(ms(single), 3),
        "single_forward_ms_p95": round(float(np.percentile(single, 95) * 1e3), 3),
        "ensemble5_readout_ms_median": round(ms(ens), 3),
        "ensemble5_readout_ms_p95": round(float(np.percentile(ens, 95) * 1e3), 3),
        "n_params_per_model": sum(p.numel() for p in encs[0][0].parameters()),
    }
    print(json.dumps(out))


if __name__ == "__main__":
    main()
