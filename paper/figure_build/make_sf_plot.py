#!/usr/bin/env python3
"""Panel (b) SF_k(d_ij) plot, drawn from the paper's own scoring terms.

The curves are NOT eyeballed: they are utils/vina_rules.py's five Vina terms with
VINA_WEIGHT, reimplemented in numpy so this runs without torch.

    steric              = w_g1*gauss1 + w_g2*gauss2 + w_rep*repulsion
    steric + hydrophobic= steric + w_hyd*hydrophobic
    steric + h-bond     = steric + w_hb *hbonding
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# utils/vina_rules.py :: VINA_WEIGHT = [gauss1, gauss2, repulsion, hydrophobic, hbonding]
W_G1, W_G2, W_REP, W_HYD, W_HB = -0.0356, -0.00516, 0.840, -0.0351, -0.587

gauss1      = lambda d: np.exp(-(2 * d) ** 2)
gauss2      = lambda d: np.exp(-((d - 3.0) / 2) ** 2)
repulsion   = lambda d: np.where(d < 0, d ** 2, 0.0)
hydrophobic = lambda d: np.clip(1.5 - d, 0.0, 1.0)
hbonding    = lambda d: np.clip(d / (-0.7), 0.0, 1.0)

TEAL, MAGENTA = "#17A2B8", "#E040A0"


def curves(d):
    steric = W_G1 * gauss1(d) + W_G2 * gauss2(d) + W_REP * repulsion(d)
    return steric, steric + W_HYD * hydrophobic(d), steric + W_HB * hbonding(d)


def main(out_dir=Path(__file__).parent / "assets"):
    out_dir.mkdir(parents=True, exist_ok=True)
    d = np.linspace(-1.0, 6.0, 4001)
    ste, hyd, hb = curves(d)

    # report the minima so the numbers printed on the figure are checkable
    for name, y in (("Steric", ste), ("Steric+Hydrophobic", hyd), ("Steric+H-Bond", hb)):
        i = int(np.argmin(y))
        print(f"{name:20s} min = {y[i]:+.4f} at d = {d[i]:+.3f}")

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 15,
                         "axes.linewidth": 1.4, "xtick.major.width": 1.4,
                         "ytick.major.width": 1.4})
    # Figure aspect is pinned to the slot this plot occupies in the composite figure
    # (figure_build/main_figure_fixed2.pptx, panel b). Measured from the composite: the
    # embedded axes frame reads 1.3254 while this script's native axes is 1.6075, so the
    # old 6.6x4.6 art was being squeezed to 0.8245 of its width on the slide. The slot is
    # therefore 1.4348*0.8245 = 1.183 wide-to-tall; 6.6/1.183 = 5.58in fills it undistorted.
    fig, ax = plt.subplots(figsize=(6.6, 5.58))

    ax.axhline(0, color="#9AA0A6", lw=1.0, zorder=1)
    ax.axvline(0, color="#9AA0A6", lw=1.0, ls="--", dashes=(5, 4), zorder=1)

    LW = 4.2  # ~3x the axis weight, per the figure spec
    ax.plot(d, ste, color="black", lw=LW, zorder=3, label="Steric",
            solid_capstyle="round")
    ax.plot(d, hyd, color=TEAL, lw=LW, zorder=4, label="Steric + Hydrophobic",
            solid_capstyle="round")
    # magenta last so it hides black where the two coincide (d >= 0, hbonding == 0)
    ax.plot(d, hb, color=MAGENTA, lw=LW, zorder=5, label="Steric + H-Bond",
            solid_capstyle="round")

    ax.set_xlim(-1, 6)
    ax.set_ylim(-0.25, 0.10)
    ax.set_xticks(range(-1, 7))
    ax.set_yticks(np.arange(-0.25, 0.101, 0.05))
    ax.set_yticklabels([" 0.00" if abs(v) < 1e-9 else f"{v:+.2f}"
                        for v in np.arange(-0.25, 0.101, 0.05)])
    ax.set_xlabel("Surface Distance $\\bf{d}$ (Å)")
    ax.set_ylabel("Pair Energy")
    ax.set_title("$SF_k(\\,d_{ij}\\,)$", fontsize=18, pad=10)
    ax.grid(False)
    ax.set_facecolor("white")

    # No per-minimum value annotations: the plot is reproduced at ~1/3 scale inside
    # the composite figure, where 13.5pt text is unreadable. The minima are printed to
    # stdout above and belong in the running text, not on the axes.

    ax.legend(loc="lower right", frameon=True, framealpha=1.0, edgecolor="#9AA0A6",
              fontsize=17, borderpad=0.6, handlelength=1.8, labelspacing=0.45,
              borderaxespad=0.5).set_zorder(7)

    fig.tight_layout()
    for ext in ("pdf", "png", "svg"):
        fig.savefig(out_dir / f"sf_plot.{ext}", dpi=600, transparent=False)
    print(f"\nwrote {out_dir}/sf_plot.{{pdf,png,svg}}")


if __name__ == "__main__":
    main()
