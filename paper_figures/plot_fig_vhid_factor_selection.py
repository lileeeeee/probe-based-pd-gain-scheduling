"""fig_vhid_factor_selection: \\Vbin vs measured z-MSE scatter (Table 3 / Eq 3).

Z-normalized version (R2-C3/C8 revision): both axes in the z-MSE metric of the
main text; every list sits on or above the LOTV reference line. Reads numbers
from output/r2c3_table3_znorm.json (experiments/r2c3_table3_znorm.py).
Recovered and parameterized from the 2026-05-29 session that produced the
submitted physical-units figure; style matches fig1 (DejaVu Sans, despined).
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper_final/robotica/figures"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "mathtext.fontset": "dejavusans",
    "font.size": 10, "axes.labelsize": 11,
    "xtick.labelsize": 9.5, "ytick.labelsize": 9.5, "legend.fontsize": 9,
    "axes.linewidth": 0.8,
    "xtick.direction": "out", "ytick.direction": "out",
    "xtick.major.size": 3.5, "ytick.major.size": 3.5,
    "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.03, "savefig.dpi": 600,
})
BLUE = "#4C72B0"
REF = "0.6"
LBL = dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.7)

d = json.load(open(ROOT / "output/r2c3_table3_znorm.json"))
ORDER = ["mp_fv_fc", "mp_fv", "mp_fc", "mp", "fv_fc", "fv", "fc"]
LABELS = [r"$\{m_p,f_v,f_c\}$", r"$\{m_p,f_v\}$", r"$\{m_p,f_c\}$",
          r"$\{m_p\}$", r"$\{f_v,f_c\}$", r"$\{f_v\}$", r"$\{f_c\}$"]
vhid = np.array([d["casadi"][k]["v"] for k in ORDER])
mse = np.array([d["casadi"][k]["mse"] for k in ORDER])
slope, ic, r = (d["eq3_z"][k] for k in ["slope", "intercept", "r"])

# all points sit above the diagonal now; labels go to the lower right except
# the two rightmost, which run out of x-room
off = [(6, -9, "left", "top"), (6, -9, "left", "top"), (-6, -9, "right", "top"),
       (7, -2, "left", "center"), (-6, -9, "right", "top"),
       (7, -6, "left", "top"), (7, -2, "left", "center")]

fig, ax = plt.subplots(figsize=(3.6, 3.05))
xr = np.array([0.38, 1.17])
ax.plot(xr, xr, "-", lw=1.0, color=REF, zorder=2,
        label=r"LOTV floor  $\mathrm{MSE}=\widehat{V}^{\,\mathrm{bin}}_{\mathrm{hidden}}$")
ax.plot(xr, slope * xr + ic, "--", lw=1.5, color=BLUE, zorder=3,
        label=rf"fit:  slope {slope:.2f},  $r={r:.3f}$")
ax.scatter(vhid, mse, s=34, color=BLUE, edgecolor="white", linewidth=0.6, zorder=4)
for x, y, lab, (dx, dy, ha, va) in zip(vhid, mse, LABELS, off):
    ax.annotate(lab, (x, y), textcoords="offset points", xytext=(dx, dy),
                ha=ha, va=va, fontsize=7.5, color="0.4", bbox=LBL, zorder=5)
ax.set_xlim(*xr); ax.set_ylim(*xr)
ticks = np.arange(0.4, 1.11, 0.2)
ax.set_xticks(ticks); ax.set_yticks(ticks)
ax.set_xlabel(r"$\widehat{V}^{\,\mathrm{bin}}_{\mathrm{hidden}}(\mathcal{O})$   (training-data estimate)")
ax.set_ylabel(r"IID test $z$-MSE   (5-seed mean)")
ax.legend(loc="upper left", frameon=False, handlelength=1.8, borderpad=0.2)
for s in ["top", "right"]:
    ax.spines[s].set_visible(False)
fig.savefig(OUT / "fig_vhid_factor_selection.pdf")
fig.savefig("/tmp/fig_vhid_preview.png", dpi=150)
print("wrote", OUT / "fig_vhid_factor_selection.pdf")
