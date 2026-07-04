#!/usr/bin/env python3
"""
analyse.py
==========
Collect LCM disorder sweep results and produce the two-panel figure
analogous to Mildner et al. Fig. 4:

    Panel (a): disorder-averaged spin Chern marker c_bar vs disorder W,
               one curve per system size L.

    Panel (b): data collapse  c_bar vs (W - Wc) * L^{1/nu},
               with a zoomed inset around the critical region.

Physics-blind: all FSS algorithm lives in AS/physics/fss_tools.py.
This script only handles data loading, statistics, and plotting.

Run from AS/lcm_disorder/:
    python analyse.py
    python analyse.py --datadir data --outdir figures
"""

import os
import sys
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.ticker import MultipleLocator

# --------------------------------------------------------------------------- #
#  Add physics directory to path
# --------------------------------------------------------------------------- #
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'physics'))
from fss_tools import (pairwise_crossings, fit_fss,
                       bootstrap_fss, master_curve)

# --------------------------------------------------------------------------- #
#  Plotting style
# --------------------------------------------------------------------------- #
plt.rcParams.update({
    'font.size':        11,
    'axes.labelsize':   12,
    'legend.fontsize':   9,
    'figure.dpi':       150,
    'lines.linewidth':  1.2,
    'lines.markersize': 3.0,
})


# --------------------------------------------------------------------------- #
#  Data loading
# --------------------------------------------------------------------------- #
def load_data(datadir):
    """
    Scan datadir for all lcm_W*_L*.npz files.
    Returns a dict:
        data[(W, L)] = {'cs': array(n_real,), 'gaps': array(n_real,),
                        'cbar': float, 'csem': float}
    and sorted lists W_list, L_list.
    """
    pattern = os.path.join(datadir, 'lcm_W*_L*.npz')
    files   = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"No data files found in {datadir}. "
            "Run the cluster job first (sbatch submit.slurm).")

    data    = {}
    W_set   = set()
    L_set   = set()

    for fpath in files:
        d = np.load(fpath)
        W = float(d['W'])
        L = int(d['L'])
        cs    = d['cs']
        gaps  = d['gaps']
        n     = len(cs)
        cbar  = float(cs.mean())
        csem  = float(cs.std(ddof=1) / np.sqrt(n))   # standard error of mean

        data[(W, L)] = {'cs': cs, 'gaps': gaps,
                        'cbar': cbar, 'csem': csem}
        W_set.add(W)
        L_set.add(L)

    W_list = sorted(W_set)
    L_list = sorted(L_set)

    print(f"Loaded {len(files)} files: "
          f"{len(W_list)} W points x {len(L_list)} L values")
    print(f"  W range : [{min(W_list):.3f}, {max(W_list):.3f}]")
    print(f"  L values: {L_list}")

    return data, np.array(W_list), L_list


def build_flat_arrays(data, W_arr, L_list):
    """
    Flatten the (W, L) grid into 1D arrays for FSS fitting.
    Skips missing (W, L) combinations gracefully.
    """
    W_flat, L_flat, cbar_flat, csem_flat = [], [], [], []

    for L in L_list:
        for W in W_arr:
            key = (W, L)
            if key not in data:
                continue
            W_flat.append(W)
            L_flat.append(L)
            cbar_flat.append(data[key]['cbar'])
            csem_flat.append(data[key]['csem'])

    return (np.array(W_flat), np.array(L_flat, dtype=float),
            np.array(cbar_flat), np.array(csem_flat))


# --------------------------------------------------------------------------- #
#  Color scheme: one color per L value (blue -> red for increasing L)
# --------------------------------------------------------------------------- #
def make_colors(L_list):
    cmap   = cm.plasma
    n      = len(L_list)
    colors = {L: cmap(0.1 + 0.8 * i / (n - 1)) for i, L in enumerate(L_list)}
    return colors


# --------------------------------------------------------------------------- #
#  Panel (a): c_bar vs W
# --------------------------------------------------------------------------- #
def plot_panel_a(ax, data, W_arr, L_list, colors, Wc=None):
    """
    Plot disorder-averaged c_bar vs W for each L.
    Vertical line at Wc if provided.
    """
    for L in L_list:
        cbar_arr = []
        csem_arr = []
        W_plot   = []
        for W in W_arr:
            key = (W, L)
            if key not in data:
                continue
            W_plot.append(W)
            cbar_arr.append(data[key]['cbar'])
            csem_arr.append(data[key]['csem'])

        ax.plot(W_plot, cbar_arr, '-o',
                color=colors[L], label=f'$L={L}$', markersize=3)

    # vertical line at Wc
    if Wc is not None:
        ax.axvline(Wc, color='k', linewidth=1.0, linestyle='-')
        ax.text(Wc + 0.05, 0.05, r'$W_c$', fontsize=11)

    # arrow annotation for increasing L
    ax.annotate('Increasing $L$',
                xy=(1.5, 0.85), xytext=(1.5, 0.55),
                arrowprops=dict(arrowstyle='->', color='k', lw=1.0),
                fontsize=9, ha='left')

    ax.set_xlabel(r'$W$', fontsize=12)
    ax.set_ylabel(r'$\bar{c}$', fontsize=12)
    ax.set_xlim(W_arr.min() - 0.1, W_arr.max() + 0.1)
    ax.set_ylim(-0.05, 1.10)
    ax.xaxis.set_minor_locator(MultipleLocator(0.5))
    ax.yaxis.set_minor_locator(MultipleLocator(0.25))
    ax.text(0.04, 0.92, '(a)', transform=ax.transAxes, fontsize=12)


# --------------------------------------------------------------------------- #
#  Panel (b): data collapse + inset
# --------------------------------------------------------------------------- #
def plot_panel_b(ax, data, W_arr, L_list, colors,
                 Wc, nu, W_flat, L_flat, cbar_flat, csem_flat, fig):
    """
    Plot c_bar vs (W - Wc) * L^{1/nu} for each L (the collapsed data).
    Overlays the master curve polynomial.
    Adds a zoomed inset of the critical region.
    """
    # --- collapsed data for each L ---
    for L in L_list:
        cbar_arr = []
        x_arr    = []
        for W in W_arr:
            key = (W, L)
            if key not in data:
                continue
            x_arr.append((W - Wc) * (float(L) ** (1.0 / nu)))
            cbar_arr.append(data[key]['cbar'])

        ax.plot(x_arr, cbar_arr, '-o',
                color=colors[L], markersize=3)

    # --- master curve ---
    x_fine, c_fine = master_curve(Wc, nu,
                                  W_flat, L_flat,
                                  cbar_flat, csem_flat)
    ax.plot(x_fine, c_fine, 'k-', linewidth=1.0, alpha=0.4,
            label='master curve', zorder=0)

    ax.set_xlabel(r'$(W - W_c)\,L^{1/\nu}$', fontsize=12)
    ax.set_ylabel(r'$\bar{c}$', fontsize=12)
    ax.set_xlim(-7, 7)
    ax.set_ylim(-0.05, 1.10)
    ax.xaxis.set_minor_locator(MultipleLocator(1.0))
    ax.yaxis.set_minor_locator(MultipleLocator(0.25))
    ax.text(0.04, 0.92, '(b)', transform=ax.transAxes, fontsize=12)

    # --- inset: zoom into critical region x in [-1, 1], c in [0.2, 0.5] ---
    # position: upper right of panel (b) in axes-fraction coordinates
    ax_inset = ax.inset_axes([0.62, 0.55, 0.35, 0.40])

    for L in L_list:
        x_arr    = []
        cbar_arr = []
        for W in W_arr:
            key = (W, L)
            if key not in data:
                continue
            x = (W - Wc) * (float(L) ** (1.0 / nu))
            if abs(x) <= 1.2:
                x_arr.append(x)
                cbar_arr.append(data[key]['cbar'])
        if x_arr:
            ax_inset.plot(x_arr, cbar_arr, '-o',
                          color=colors[L], markersize=2.5, linewidth=0.9)

    ax_inset.set_xlim(-1, 1)
    ax_inset.set_ylim(0.18, 0.50)
    ax_inset.xaxis.set_minor_locator(MultipleLocator(0.5))
    ax_inset.yaxis.set_minor_locator(MultipleLocator(0.1))
    ax_inset.tick_params(labelsize=8)

    # draw a rectangle on the main panel showing the inset region
    ax.indicate_inset_zoom(ax_inset, edgecolor='gray', alpha=0.5)


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(
        description="Collect LCM data and plot Fig 4 (c_bar vs W + FSS collapse)")
    parser.add_argument('--datadir',  type=str, default='data',
                        help="Directory containing lcm_W*_L*.npz files")
    parser.add_argument('--outdir',   type=str, default='figures',
                        help="Directory to save the figure")
    parser.add_argument('--n_boot',   type=int, default=300,
                        help="Number of bootstrap samples for error bars")
    parser.add_argument('--Wc_init',  type=float, default=None,
                        help="Initial guess for Wc (auto from crossings if omitted)")
    parser.add_argument('--nu_init',  type=float, default=2.5,
                        help="Initial guess for nu (default 2.5, Mildner's value)")
    args = parser.parse_args()

    # ------------------------------------------------------------------ #
    #  Load data
    # ------------------------------------------------------------------ #
    data, W_arr, L_list = load_data(args.datadir)
    colors = make_colors(L_list)

    # flatten to 1D arrays for FSS fitting
    W_flat, L_flat, cbar_flat, csem_flat = build_flat_arrays(
        data, W_arr, L_list)

    # ------------------------------------------------------------------ #
    #  Initial Wc estimate from pairwise crossings
    # ------------------------------------------------------------------ #
    cbar_by_L = {}
    for L in L_list:
        cbar_by_L[L] = np.array([
            data.get((W, L), {'cbar': np.nan})['cbar'] for W in W_arr])

    crossings = pairwise_crossings(W_arr, cbar_by_L, L_list)
    if len(crossings) > 0:
        Wc_cross = float(np.median(crossings))
        print(f"\nPairwise crossings: {len(crossings)} found, "
              f"median Wc = {Wc_cross:.4f}")
    else:
        Wc_cross = (W_arr.min() + W_arr.max()) / 2.0
        print(f"\nNo crossings found; using midpoint Wc_init = {Wc_cross:.4f}")

    Wc_init = args.Wc_init if args.Wc_init is not None else Wc_cross
    nu_init = args.nu_init

    # ------------------------------------------------------------------ #
    #  FSS optimization
    # ------------------------------------------------------------------ #
    print(f"\nOptimizing FSS collapse (init: Wc={Wc_init:.3f}, nu={nu_init:.3f})")
    fss_result = fit_fss(W_flat, L_flat, cbar_flat, csem_flat,
                         Wc_init, nu_init)
    Wc = fss_result['Wc']
    nu = fss_result['nu']
    print(f"  Wc = {Wc:.4f}   nu = {nu:.4f}   chi2 = {fss_result['chi2']:.6f}"
          f"   converged = {fss_result['success']}")

    # ------------------------------------------------------------------ #
    #  Bootstrap errors
    # ------------------------------------------------------------------ #
    print(f"\nBootstrap ({args.n_boot} samples) ...")
    boot = bootstrap_fss(W_flat, L_flat, cbar_flat, csem_flat,
                         Wc, nu, n_boot=args.n_boot)
    Wc_err = boot['Wc_err']
    nu_err = boot['nu_err']
    print(f"  Wc = {Wc:.4f} +/- {Wc_err:.4f}")
    print(f"  nu = {nu:.4f} +/- {nu_err:.4f}")

    # ------------------------------------------------------------------ #
    #  Figure
    # ------------------------------------------------------------------ #
    fig, (ax_a, ax_b) = plt.subplots(2, 1, figsize=(6.5, 9.0),
                                      constrained_layout=True)

    # panel (a)
    plot_panel_a(ax_a, data, W_arr, L_list, colors, Wc=Wc)

    # panel (b)
    plot_panel_b(ax_b, data, W_arr, L_list, colors,
                 Wc, nu, W_flat, L_flat, cbar_flat, csem_flat, fig)

    # shared caption box
    caption = (rf"$W_c = {Wc:.2f} \pm {Wc_err:.2f}$"
               rf"$\quad \nu = {nu:.2f} \pm {nu_err:.2f}$")
    fig.text(0.5, 0.01, caption, ha='center', fontsize=11,
             bbox=dict(boxstyle='round', facecolor='lightyellow',
                       edgecolor='gray', alpha=0.8))

    # save
    os.makedirs(args.outdir, exist_ok=True)
    outpath = os.path.join(args.outdir, 'fig4_lcm_disorder.pdf')
    fig.savefig(outpath, bbox_inches='tight')
    print(f"\nFigure saved to {outpath}")

    # also save a png for quick viewing
    outpath_png = outpath.replace('.pdf', '.png')
    fig.savefig(outpath_png, dpi=150, bbox_inches='tight')
    print(f"Figure saved to {outpath_png}")

    plt.show()


if __name__ == '__main__':
    main()
