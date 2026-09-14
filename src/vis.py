"""
Visualization utilities for experiment results.

All plot functions accept data arrays and save to a specified output directory.
Uses Agg backend so plots work headless.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.cm as cm


# ── Style defaults ────────────────────────────────────────────────────
COLORS = {
    "trans": "#2196F3",
    "moe":   "#FF9800",
}
ZAUX_MARKERS = {
    "none":    "o",
    "mp_only": "s",
    "fv_only": "^",
    "both":    "D",
}


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


# ── 1. Training curves ───────────────────────────────────────────────

def plot_training_curves(
    logs: Dict[str, Dict],
    out_dir: str,
    filename: str = "training_curves.png",
):
    """Plot train/test MSE vs epoch for multiple configs.

    logs: {label: {"epochs": [...], "train_mse": [...], "test_mse": [...], "best_test": float, "best_ep": int}}
    """
    _ensure_dir(out_dir)

    n = len(logs)
    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)

    for idx, (label, log) in enumerate(logs.items()):
        ax = axes[idx // ncols, idx % ncols]
        epochs = log["epochs"]
        ax.plot(epochs, log["train_mse"], label="train", alpha=0.7, linewidth=1)
        ax.plot(epochs, log["test_mse"], label="test", alpha=0.9, linewidth=1.5)
        if "best_test" in log:
            ax.axhline(log["best_test"], color="red", linestyle="--", alpha=0.5, linewidth=0.8)
            ax.text(epochs[-1] * 0.6, log["best_test"] * 1.05,
                    f'best={log["best_test"]:.4f}@{log["best_ep"]}',
                    fontsize=8, color="red")
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        ax.set_ylim(bottom=0)

    # Hide unused axes
    for idx in range(n, nrows * ncols):
        axes[idx // ncols, idx % ncols].set_visible(False)

    fig.suptitle("Training Curves", fontsize=13)
    fig.tight_layout()
    path = os.path.join(out_dir, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── 2. Prediction scatter: pred vs true ─────────────────────────────

def plot_pred_vs_true(
    predictions: Dict[str, Dict],
    out_dir: str,
    filename: str = "pred_vs_true.png",
    gain_names: Optional[List[str]] = None,
):
    """Scatter pred vs true for each config, aggregated over all gains.

    predictions: {label: {"pred": (N, G), "true": (N, G)}}
    """
    _ensure_dir(out_dir)
    n = len(predictions)
    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows), squeeze=False)

    for idx, (label, data) in enumerate(predictions.items()):
        ax = axes[idx // ncols, idx % ncols]
        pred = data["pred"].flatten()
        true = data["true"].flatten()

        ax.scatter(true, pred, s=3, alpha=0.3, color="#2196F3", rasterized=True)
        lims = [min(true.min(), pred.min()), max(true.max(), pred.max())]
        ax.plot(lims, lims, "r--", linewidth=1, alpha=0.7)
        ax.set_xlabel("True gain")
        ax.set_ylabel("Predicted gain")
        ax.set_title(f"{label}\nMSE={np.mean((pred-true)**2):.4f}", fontsize=10)
        ax.set_aspect("equal", "box")
        ax.grid(alpha=0.3)

    for idx in range(n, nrows * ncols):
        axes[idx // ncols, idx % ncols].set_visible(False)

    fig.suptitle("Predictions vs Ground Truth (all gains)", fontsize=13)
    fig.tight_layout()
    path = os.path.join(out_dir, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_pred_vs_true_per_gain(
    pred: np.ndarray,
    true: np.ndarray,
    label: str,
    out_dir: str,
    gain_names: Optional[List[str]] = None,
):
    """Per-gain scatter for a single config. pred/true: (N, G)."""
    _ensure_dir(out_dir)
    G = pred.shape[1]
    if gain_names is None:
        gain_names = [f"gain_{i}" for i in range(G)]

    ncols = min(4, G)
    nrows = (G + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows), squeeze=False)

    for g in range(G):
        ax = axes[g // ncols, g % ncols]
        p, t = pred[:, g], true[:, g]
        ax.scatter(t, p, s=10, alpha=0.6, color="#2196F3")
        lims = [min(t.min(), p.min()), max(t.max(), p.max())]
        ax.plot(lims, lims, "r--", linewidth=1, alpha=0.7)
        mse_g = np.mean((p - t) ** 2)
        ax.set_title(f"{gain_names[g]}\nMSE={mse_g:.4f}", fontsize=9)
        ax.set_xlabel("True")
        ax.set_ylabel("Pred")
        ax.grid(alpha=0.3)

    for g in range(G, nrows * ncols):
        axes[g // ncols, g % ncols].set_visible(False)

    fig.suptitle(f"Per-Gain Predictions: {label}", fontsize=12)
    fig.tight_layout()
    path = os.path.join(out_dir, f"pred_per_gain_{label}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── 3. Error heatmap over (mp, fv) ──────────────────────────────────

def plot_error_heatmap(
    predictions: Dict[str, Dict],
    out_dir: str,
    filename: str = "error_heatmap.png",
):
    """Per-sample MSE as scatter over (mp, fv) for each config.

    predictions: {label: {"pred": (N,G), "true": (N,G), "mp": (N,), "fv": (N,)}}
    """
    _ensure_dir(out_dir)
    n = len(predictions)
    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols

    # Compute global vmax for consistent colorbar
    all_mse = []
    for data in predictions.values():
        per_sample = np.mean((data["pred"] - data["true"]) ** 2, axis=1)
        all_mse.append(per_sample)
    vmax = np.percentile(np.concatenate(all_mse), 95)

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows), squeeze=False)

    for idx, (label, data) in enumerate(predictions.items()):
        ax = axes[idx // ncols, idx % ncols]
        per_sample = np.mean((data["pred"] - data["true"]) ** 2, axis=1)
        sc = ax.scatter(data["mp"], data["fv"], c=per_sample,
                        cmap="RdYlGn_r", s=30, alpha=0.8, vmin=0, vmax=vmax,
                        edgecolors="gray", linewidths=0.3)
        ax.set_xlabel("$m_p$ (payload)")
        ax.set_ylabel("$f_v$ (friction)")
        ax.set_title(f"{label}\nmean={per_sample.mean():.4f}", fontsize=10)
        plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label="Sample MSE")

    for idx in range(n, nrows * ncols):
        axes[idx // ncols, idx % ncols].set_visible(False)

    fig.suptitle("Per-Sample Error in Physics Factor Space", fontsize=13)
    fig.tight_layout()
    path = os.path.join(out_dir, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── 4. Summary comparison bar chart ─────────────────────────────────

def plot_comparison_bars(
    results: List[Dict],
    out_dir: str,
    filename: str = "comparison_bars.png",
    metric_key: str = "mean_test_mse",
    std_key: Optional[str] = "std_test_mse",
    ylabel: str = "Test MSE",
    title: str = "Model Comparison",
):
    """Grouped bar chart: arch × z-config, with optional error bars.

    results: list of {"label": ..., "arch": ..., "z_config": ...,
                      metric_key: ..., std_key: ... (optional)}
    """
    _ensure_dir(out_dir)

    archs = sorted(set(r["arch"] for r in results))
    z_cfgs = sorted(set(r["z_config"] for r in results),
                    key=lambda x: ["none", "mp_only", "fv_only", "both"].index(x)
                    if x in ["none", "mp_only", "fv_only", "both"] else 99)

    x = np.arange(len(z_cfgs))
    w = 0.8 / len(archs)

    fig, ax = plt.subplots(figsize=(max(8, 2.5 * len(z_cfgs)), 5))

    for ai, arch in enumerate(archs):
        vals, errs = [], []
        for zc in z_cfgs:
            match = [r for r in results if r["arch"] == arch and r["z_config"] == zc]
            vals.append(match[0][metric_key] if match else 0)
            errs.append(match[0].get(std_key, 0) if (match and std_key) else 0)
        color = COLORS.get(arch, f"C{ai}")
        has_err = any(e > 0 for e in errs)
        bars = ax.bar(x + ai * w - 0.4 + w / 2, vals, w,
                      yerr=errs if has_err else None, capsize=3,
                      label=arch.upper(), color=color, alpha=0.85)
        for bar, v, e in zip(bars, vals, errs):
            if v > 0:
                offset = e + 0.005 if has_err else 0.003
                text = f"{v:.4f}±{e:.4f}" if has_err and e > 0 else f"{v:.4f}"
                ax.text(bar.get_x() + bar.get_width() / 2, v + offset,
                        text, ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(z_cfgs, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    path = os.path.join(out_dir, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── 5. Gain landscape visualization ─────────────────────────────────

def plot_gain_landscape(
    phys: np.ndarray,
    gains: np.ndarray,
    out_dir: str,
    gain_names: Optional[List[str]] = None,
    filename: str = "gain_landscape.png",
):
    """Scatter plot of each gain dimension over (mp, fv) space.

    phys: (N, 2)  gains: (N, G)
    """
    _ensure_dir(out_dir)
    G = gains.shape[1]
    if gain_names is None:
        gain_names = [f"gain_{i}" for i in range(G)]

    ncols = min(4, G)
    nrows = (G + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4 * nrows), squeeze=False)

    for g in range(G):
        ax = axes[g // ncols, g % ncols]
        sc = ax.scatter(phys[:, 0], phys[:, 1], c=gains[:, g],
                        cmap="viridis", s=15, alpha=0.8)
        ax.set_xlabel("$m_p$")
        ax.set_ylabel("$f_v$")
        ax.set_title(gain_names[g], fontsize=9)
        plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)

    for g in range(G, nrows * ncols):
        axes[g // ncols, g % ncols].set_visible(False)

    fig.suptitle("Gain Landscape over Physics Factor Space", fontsize=12)
    fig.tight_layout()
    path = os.path.join(out_dir, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
