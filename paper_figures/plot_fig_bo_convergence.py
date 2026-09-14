"""fig_bo_convergence: BO label repeatability, cold start vs warm start.

Median best-cost-so-far with min-max band over 5 seeds, three representative
physics points of the F=3 Alpha 5 grid. Data: output/from_zgpu3/bo_variance/
f3_casadi_5pts_5seeds_2conditions_curves.npz (experiments/from_zgpu3/
run_bo_repeatability_curves_casadi.py on the server). Style matches fig1.
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper_final/robotica/figures"
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
    "mathtext.fontset": "dejavusans",
    "font.size": 9, "axes.labelsize": 9.5, "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5, "legend.fontsize": 8, "axes.linewidth": 0.8,
    "xtick.direction": "out", "ytick.direction": "out",
    "savefig.bbox": "tight", "savefig.pad_inches": 0.03, "savefig.dpi": 600,
})
RED, BLUE = "#C44E52", "#4C72B0"
c = np.load(ROOT / "output/from_zgpu3/bo_variance/f3_casadi_5pts_5seeds_2conditions_curves.npz",
            allow_pickle=True)
curves = c["best_cost_curve"]; conds = list(c["conditions"])
vi, wi = conds.index("vanilla"), conds.index("with_warmstart")
pick = [0, 2, 4]   # P1 nominal, P3 hardest, P5 most extreme cold-start failure
titles = {0: r"$(m_p, f_v, f_c)=(1.0,\,1.0,\,1.0)$", 2: r"$(4.5,\,0.4,\,1.25)$", 4: r"$(3.0,\,0.75,\,1.25)$"}
x = np.arange(64, 401)
fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.3), sharex=True)
for ax, p in zip(axes, pick):
    for ci, color, lab in [(vi, RED, "cold start"), (wi, BLUE, "warm start")]:
        Y = curves[p, ci, :, 63:]
        ax.fill_between(x, Y.min(0), Y.max(0), color=color, alpha=0.22, lw=0)
        ax.plot(x, np.median(Y, 0), color=color, lw=1.5, label=lab)
    ax.set_yscale("log"); ax.set_title(titles[p], fontsize=8.5)
    ax.set_xlabel("evaluations")
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
axes[0].set_ylabel("best cost so far")
axes[0].legend(frameon=False, loc="upper right", handlelength=1.6)
fig.tight_layout(w_pad=1.2)
fig.savefig(OUT / "fig_bo_convergence.pdf")
fig.savefig("/tmp/fig_bo_convergence_preview.png", dpi=150)
print("wrote", OUT / "fig_bo_convergence.pdf")
