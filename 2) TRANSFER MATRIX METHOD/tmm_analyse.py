#!/usr/bin/env python3
"""
analyse.py  (TMM — disorder-driven transition)
===============================================

Reproduces the three figures from Mildner et al.
Supplementary Material, Section H (disorder-driven transition):

  Figure 1  [analog of Fig. S-36]:
      Lambda = (xi / Ly)^{-1}  vs  W  for each strip width Ny.
      Lambda has a minimum at the critical disorder Wc.
      Key feature: the minimum location barely drifts with Ny
      (unlike the mass-driven case where it drifts strongly).
      Inset: Wc(Ny) vs Ny confirming negligible drift.

  Figure 2  [analog of Fig. S-37]:
      Log-log plot of the curvature d²Lambda/dW²|_{Wc}  vs  Ny.
      From FSS theory (Mildner Eq. D.6), this scales as Ny^{2/nu}.
      The slope of a straight-line fit gives 2/nu -> extracts nu.

  Figure 3  [analog of Fig. S-38]:
      Scaling collapse. For each Ny, shift each curve:
          Lambda_tilde = Lambda(W) - Lambda_min(Ny)
      so every minimum sits at zero. Then plot vs
          x = (W - Wc_Ny) * Ny^{1/nu}
      If nu is correct, all curves overlap onto one master curve.

Run from AS/tmm_disorder/:
    python3 analyse.py
    python3 analyse.py --datadir data --outdir figures --Wc_guess 3.5
"""

import os
import sys
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.stats import linregress

# ============================================================
# PLOTTING STYLE
# ============================================================
plt.rcParams.update({
    'font.size'       : 11,
    'axes.labelsize'  : 13,
    'legend.fontsize' : 8,
    'figure.dpi'      : 150,
    'lines.linewidth' : 1.5,
    'lines.markersize': 5,
})


# ============================================================
# SECTION 1 — DATA LOADING
# ============================================================

def load_data(datadir):
    """
    Scan datadir for all tmm_W*_Ny*.npz files produced by run_point.py.

    Each file stores one (W, Ny) point with:
        Lambda = (xi / Ny)^{-1}   -- the key dimensionless observable
        xi, xi_err                 -- localization length and its error

    Returns
    -------
    data    : dict keyed by (W, Ny) -> {'Lambda': float, 'xi': float, 'xi_err': float}
    W_arr   : 1D array of unique W values, sorted ascending
    Ny_list : list of unique Ny values, sorted ascending
    """
    pattern = os.path.join(datadir, 'tmm_W*_Ny*.npz')
    files   = sorted(glob.glob(pattern))

    if not files:
        raise FileNotFoundError(
            f"No TMM data files found in '{datadir}'.\n"
            "Run the cluster job first: sbatch submit.slurm")

    data   = {}
    W_set  = set()
    Ny_set = set()

    for fpath in files:
        d  = np.load(fpath)
        W  = float(d['W'])
        Ny = int(d['Ny'])

        # Lambda = Ly/xi = (xi/Ly)^{-1}
        # Small Lambda near the transition (large xi = weak localization)
        data[(W, Ny)] = {
            'Lambda' : float(d['Lambda']),
            'xi'     : float(d['xi']),
            'xi_err' : float(d['xi_err']),
        }
        W_set.add(W)
        Ny_set.add(Ny)

    W_arr   = np.array(sorted(W_set))
    Ny_list = sorted(Ny_set)

    print(f"Loaded {len(files)} files")
    print(f"  W  range : [{W_arr.min():.3f}, {W_arr.max():.3f}]  ({len(W_arr)} points)")
    print(f"  Ny values: {Ny_list}")

    return data, W_arr, Ny_list


# ============================================================
# SECTION 2 — MINIMUM FINDING
# (Mildner: minimum of Lambda identifies finite-size Wc(Ny))
# ============================================================

def find_minimum(W_arr, Lambda_arr, n_fit=7):
    """
    Locate the minimum of Lambda(W) for one strip width Ny.

    Method: fit a quadratic polynomial to the n_fit points
    closest to the raw data minimum, then solve for the exact
    minimum analytically. This is smoother than just reading
    off the lowest data point.

    Parameters
    ----------
    W_arr      : 1D array of W values for this Ny
    Lambda_arr : 1D array of Lambda values
    n_fit      : number of points around minimum to use in quadratic fit

    Returns
    -------
    Wc_Ny      : W at the minimum (finite-size critical point)
    Lambda_min : Lambda value at the minimum
    """
    # Step 1: find the raw index of the minimum
    imin = int(np.argmin(Lambda_arr))

    # Step 2: take n_fit points centred on that index (clamp to array bounds)
    lo = max(0, imin - n_fit // 2)
    hi = min(len(W_arr), lo + n_fit)
    lo = max(0, hi - n_fit)          # re-clamp lower bound

    W_fit   = W_arr[lo:hi]
    Lam_fit = Lambda_arr[lo:hi]

    if len(W_fit) < 3:
        # Not enough points -- return raw minimum
        return float(W_arr[imin]), float(Lambda_arr[imin])

    # Step 3: fit quadratic:  Lambda ≈ a2*(W - W0)^2 + a0
    # (centre around the raw minimum for numerical stability)
    W0 = W_arr[imin]
    coeffs = np.polyfit(W_fit - W0, Lam_fit, 2)   # [a2, a1, a0]

    # Minimum of quadratic  a2*x^2 + a1*x + a0  is at x* = -a1 / (2*a2)
    a2, a1, a0 = coeffs
    if abs(a2) < 1e-12:
        return float(W_arr[imin]), float(Lambda_arr[imin])

    x_star    = -a1 / (2 * a2)
    Wc_Ny     = float(W0 + x_star)
    Lambda_min = float(np.polyval(coeffs, x_star))

    return Wc_Ny, Lambda_min


def compute_curvature(W_arr, Lambda_arr, Wc, n_fit=7):
    """
    Compute the second derivative d²Lambda/dW²|_{Wc} for one strip width.

    From Mildner Eq. D.6:  d²Lambda/dW²|_{Wc} ~ Ny^{2/nu}
    This divergence with Ny is how we extract the critical exponent nu.

    Method: fit quadratic Lambda ≈ a2*(W-Wc)^2 + a1*(W-Wc) + a0
    near Wc. The second derivative is 2*a2.

    Parameters
    ----------
    W_arr      : 1D array of W values for this Ny
    Lambda_arr : 1D array of Lambda values
    Wc         : critical disorder (thermodynamic, from average of minima)
    n_fit      : number of points around Wc to use in the fit

    Returns
    -------
    curvature : d²Lambda/dW²|_{Wc}  (positive, since Lambda has a minimum)
    """
    # Find the n_fit points closest to Wc
    dist  = np.abs(W_arr - Wc)
    order = np.argsort(dist)
    idx   = np.sort(order[:n_fit])

    W_fit   = W_arr[idx]
    Lam_fit = Lambda_arr[idx]

    if len(W_fit) < 3:
        return np.nan

    # Fit quadratic centred at Wc for numerical stability
    coeffs = np.polyfit(W_fit - Wc, Lam_fit, 2)   # [a2, a1, a0]
    a2 = coeffs[0]

    # Second derivative of  a2*x^2 + a1*x + a0  is  2*a2
    return float(2 * a2)


# ============================================================
# SECTION 3 — FIGURE 1 (analog of Mildner Fig. S-36)
# Lambda vs W for each Ny, with minimum markers and inset
# ============================================================

def make_colors(Ny_list):
    """
    Assign one colour per strip width, going cool->warm for increasing Ny.
    Matches the style of Mildner's figures.
    """
    cmap = cm.plasma
    n    = len(Ny_list)
    return {Ny: cmap(0.1 + 0.8 * i / max(n - 1, 1))
            for i, Ny in enumerate(Ny_list)}


def plot_figure1(data, W_arr, Ny_list, colors, Wc, outdir):
    """
    Plot Lambda = (xi/Ly)^{-1} vs W for each strip width Ny.

    Main panel: Lambda curves, each showing a minimum at Wc.
    Inset:      Wc(Ny) vs Ny -- should be nearly flat for the
                disorder-driven transition (contrast with mass-driven,
                where there is a strong drift).

    Saves: figures/fig1_lambda_vs_W.pdf
    """
    fig, ax = plt.subplots(figsize=(7, 5))

    Wc_list     = []   # finite-size critical points
    Lambda_mins = []   # Lambda at each minimum

    for Ny in Ny_list:
        # Collect Lambda(W) for this Ny, skipping missing points
        W_plot   = []
        Lam_plot = []
        for W in W_arr:
            if (W, Ny) in data:
                W_plot.append(W)
                Lam_plot.append(data[(W, Ny)]['Lambda'])

        if len(W_plot) < 3:
            continue

        W_plot   = np.array(W_plot)
        Lam_plot = np.array(Lam_plot)

        # Find and store the finite-size minimum
        Wc_Ny, Lam_min = find_minimum(W_plot, Lam_plot)
        Wc_list.append((Ny, Wc_Ny, Lam_min))
        Lambda_mins.append(Lam_min)

        # Plot the Lambda curve
        ax.plot(W_plot, Lam_plot, '-o',
                color=colors[Ny], label=f'$N_y={Ny}$')

        # Mark the minimum with a cross
        ax.plot(Wc_Ny, Lam_min, 'x',
                color=colors[Ny], markersize=8, markeredgewidth=2)

    # Vertical line at the thermodynamic Wc
    ax.axvline(Wc, color='k', linewidth=1.2, linestyle='--',
               label=f'$W_c = {Wc:.3f}$')

    ax.set_xlabel(r'$W$')
    ax.set_ylabel(r'$\Lambda = (\xi/L_y)^{-1}$')
    ax.set_title(r'$\Lambda$ vs $W$ — minimum locates $W_c$')
    ax.legend(loc='upper right', ncol=2, fontsize=7)
    ax.text(0.04, 0.95, '(a)', transform=ax.transAxes,
            fontsize=13, va='top')

    # ---- Inset: Wc(Ny) vs Ny ----
    # For the disorder-driven transition, this should be flat
    # (small drift) — confirming we are at the right fixed point.
    if len(Wc_list) >= 2:
        ax_in = ax.inset_axes([0.55, 0.55, 0.40, 0.38])
        Ny_arr_in  = np.array([x[0] for x in Wc_list])
        Wc_arr_in  = np.array([x[1] for x in Wc_list])
        ax_in.plot(Ny_arr_in, Wc_arr_in, 'o-', color='gray', markersize=4)
        ax_in.axhline(Wc, color='k', linewidth=1.0, linestyle='--')
        ax_in.set_xlabel(r'$N_y$', fontsize=9)
        ax_in.set_ylabel(r'$W_c(N_y)$', fontsize=9)
        ax_in.tick_params(labelsize=8)
        ax_in.set_title('negligible drift', fontsize=8)

    plt.tight_layout()
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, 'fig1_lambda_vs_W.pdf')
    fig.savefig(path, bbox_inches='tight')
    fig.savefig(path.replace('.pdf', '.png'), dpi=150, bbox_inches='tight')
    print(f"Saved {path}")
    plt.close(fig)

    return Wc_list


# ============================================================
# SECTION 4 — FIGURE 2 (analog of Mildner Fig. S-37)
# Log-log curvature scaling -> extract nu
# ============================================================

def plot_figure2(data, W_arr, Ny_list, colors, Wc, outdir):
    """
    Log-log plot of  d²Lambda/dW²|_{Wc}  vs  Ny.

    From FSS theory (Mildner Eq. D.6):
        d²Lambda/dW²|_{Wc}  ~  Ny^{2/nu}

    So on a log-log plot the slope is 2/nu, giving nu directly.

    This is the main quantitative result: the critical exponent.

    Saves: figures/fig2_curvature_scaling.pdf
    """
    fig, ax = plt.subplots(figsize=(6, 5))

    Ny_vals       = []
    curvature_vals = []

    for Ny in Ny_list:
        # Collect Lambda(W) for this Ny
        W_arr_Ny  = []
        Lam_arr_Ny = []
        for W in W_arr:
            if (W, Ny) in data:
                W_arr_Ny.append(W)
                Lam_arr_Ny.append(data[(W, Ny)]['Lambda'])

        if len(W_arr_Ny) < 3:
            continue

        W_arr_Ny   = np.array(W_arr_Ny)
        Lam_arr_Ny = np.array(Lam_arr_Ny)

        # Compute the curvature at Wc via parabolic fit
        curv = compute_curvature(W_arr_Ny, Lam_arr_Ny, Wc, n_fit=7)

        if np.isnan(curv) or curv <= 0:
            print(f"  Warning: bad curvature for Ny={Ny} (curv={curv:.4f}), skipping")
            continue

        Ny_vals.append(Ny)
        curvature_vals.append(curv)

        # Plot individual point coloured by Ny
        ax.scatter(Ny, curv, color=colors[Ny], s=60, zorder=5)

    if len(Ny_vals) < 2:
        print("Not enough valid curvature points for log-log fit.")
        plt.close(fig)
        return None, None

    Ny_arr   = np.array(Ny_vals, dtype=float)
    curv_arr = np.array(curvature_vals)

    # Straight-line fit in log-log space:  log(curv) = (2/nu)*log(Ny) + const
    log_Ny   = np.log(Ny_arr)
    log_curv = np.log(curv_arr)

    slope, intercept, r_value, p_value, std_err = linregress(log_Ny, log_curv)

    # slope = 2/nu  ->  nu = 2/slope
    nu     = 2.0 / slope
    nu_err = 2.0 * std_err / slope**2   # error propagation: d(nu)/d(slope) = -2/slope^2

    print(f"\nCurvature scaling fit:")
    print(f"  slope = {slope:.4f} ± {std_err:.4f}")
    print(f"  nu    = {nu:.4f} ± {nu_err:.4f}")
    print(f"  R²    = {r_value**2:.6f}")

    # Plot the fit line
    Ny_fine  = np.linspace(Ny_arr.min() * 0.9, Ny_arr.max() * 1.1, 100)
    curv_fit = np.exp(intercept) * Ny_fine**slope
    ax.plot(Ny_fine, curv_fit, 'k-', linewidth=1.5,
            label=rf'fit:  $\partial^2_W\Lambda \propto N_y^{{{slope:.2f}}}$')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$N_y$')
    ax.set_ylabel(r'$\partial^2_W \Lambda \big|_{W_c}$')
    ax.set_title(
        rf'Curvature scaling:  $\partial^2_W\Lambda \sim N_y^{{2/\nu}}$'
        '\n'
        rf'$\nu = {nu:.2f} \pm {nu_err:.2f}$  ($R^2 = {r_value**2:.4f}$)')
    ax.legend()
    ax.text(0.04, 0.95, '(b)', transform=ax.transAxes,
            fontsize=13, va='top')

    plt.tight_layout()
    path = os.path.join(outdir, 'fig2_curvature_scaling.pdf')
    fig.savefig(path, bbox_inches='tight')
    fig.savefig(path.replace('.pdf', '.png'), dpi=150, bbox_inches='tight')
    print(f"Saved {path}")
    plt.close(fig)

    return nu, nu_err


# ============================================================
# SECTION 5 — FIGURE 3 (analog of Mildner Fig. S-38)
# Scaling collapse of the centred variable
# ============================================================

def plot_figure3(data, W_arr, Ny_list, colors, Wc, nu, outdir):
    """
    Scaling collapse of the centred variable Lambda_tilde.

    For each strip width Ny (Mildner Eq. F.1):
        Lambda_tilde(Ny, W) = Lambda(W) - Lambda_min(Ny)

    This shifts every curve's minimum to zero vertically.
    Then shift horizontally by the finite-size critical point Wc(Ny):
        x = (W - Wc_Ny) * Ny^{1/nu}

    If nu is the correct critical exponent, all curves collapse
    onto one master curve regardless of Ny.

    Note: the quality of collapse is a direct visual test of nu.
    A bad nu will show the curves fanning out instead of overlapping.

    Saves: figures/fig3_scaling_collapse.pdf
    """
    fig, ax = plt.subplots(figsize=(7, 5))

    for Ny in Ny_list:
        # Collect Lambda(W) for this Ny
        W_plot   = []
        Lam_plot = []
        for W in W_arr:
            if (W, Ny) in data:
                W_plot.append(W)
                Lam_plot.append(data[(W, Ny)]['Lambda'])

        if len(W_plot) < 3:
            continue

        W_plot   = np.array(W_plot)
        Lam_plot = np.array(Lam_plot)

        # Find this Ny's finite-size minimum position and value
        Wc_Ny, Lambda_min_Ny = find_minimum(W_plot, Lam_plot)

        # Centred vertical variable (Mildner Eq. F.1):
        #   Lambda_tilde = Lambda(W) - Lambda_min(Ny)
        Lambda_tilde = Lam_plot - Lambda_min_Ny

        # Rescaled horizontal variable:
        #   x = (W - Wc_Ny) * Ny^{1/nu}
        # This is the standard FSS scaling variable that
        # makes curves of different Ny collapse.
        x = (W_plot - Wc_Ny) * (Ny ** (1.0 / nu))

        ax.plot(x, Lambda_tilde, '-o',
                color=colors[Ny], label=f'$N_y={Ny}$')

    ax.set_xlabel(r'$(W - W_{c,N_y})\,N_y^{1/\nu}$')
    ax.set_ylabel(r'$\tilde{\Lambda} = \Lambda - \Lambda_{\min}(N_y)$')
    ax.set_title(
        rf'Scaling collapse  ($\nu = {nu:.2f}$, $W_c = {Wc:.3f}$)'
        '\nAll $N_y$ curves should overlap if $\nu$ is correct')
    ax.legend(loc='upper left', ncol=2, fontsize=7)
    ax.set_ylim(bottom=-0.05)
    ax.text(0.04, 0.95, '(c)', transform=ax.transAxes,
            fontsize=13, va='top')

    plt.tight_layout()
    path = os.path.join(outdir, 'fig3_scaling_collapse.pdf')
    fig.savefig(path, bbox_inches='tight')
    fig.savefig(path.replace('.pdf', '.png'), dpi=150, bbox_inches='tight')
    print(f"Saved {path}")
    plt.close(fig)


# ============================================================
# SECTION 6 — MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="TMM analysis: Lambda vs W, curvature scaling, collapse")
    parser.add_argument('--datadir',  type=str,   default='data',
                        help="Directory containing tmm_W*_Ny*.npz files")
    parser.add_argument('--outdir',   type=str,   default='figures',
                        help="Directory to save figures")
    parser.add_argument('--Wc_guess', type=float, default=None,
                        help="Initial guess for Wc (auto-detected if omitted)")
    parser.add_argument('--nu_guess', type=float, default=2.5,
                        help="Initial guess for nu (used only if curvature fit fails)")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # ---- Step 1: Load data ----
    print("\n=== Loading data ===")
    data, W_arr, Ny_list = load_data(args.datadir)
    colors = make_colors(Ny_list)

    # ---- Step 2: Estimate Wc from the cluster of minima ----
    print("\n=== Locating Wc from minima ===")
    Wc_estimates = []
    for Ny in Ny_list:
        W_Ny  = np.array([W for W in W_arr if (W, Ny) in data])
        L_Ny  = np.array([data[(W, Ny)]['Lambda'] for W in W_Ny])
        if len(W_Ny) < 3:
            continue
        Wc_Ny, _ = find_minimum(W_Ny, L_Ny)
        Wc_estimates.append(Wc_Ny)
        print(f"  Ny={Ny:>4}:  Wc(Ny) = {Wc_Ny:.4f}")

    if args.Wc_guess is not None:
        # User override
        Wc = args.Wc_guess
        print(f"\nUsing user-supplied Wc = {Wc:.4f}")
    elif Wc_estimates:
        # For disorder-driven transition, average the minima of the
        # LARGER strips (they are closest to the thermodynamic limit)
        n_large = max(1, len(Wc_estimates) // 2)
        Wc = float(np.mean(Wc_estimates[-n_large:]))
        print(f"\nWc from average of largest {n_large} strips: {Wc:.4f}")
    else:
        Wc = W_arr[len(W_arr) // 2]
        print(f"\nCould not estimate Wc, using midpoint: {Wc:.4f}")

    # ---- Step 3: Figure 1 — Lambda vs W ----
    print("\n=== Figure 1: Lambda vs W ===")
    plot_figure1(data, W_arr, Ny_list, colors, Wc, args.outdir)

    # ---- Step 4: Figure 2 — Curvature scaling -> nu ----
    print("\n=== Figure 2: Curvature scaling ===")
    nu, nu_err = plot_figure2(data, W_arr, Ny_list, colors, Wc, args.outdir)

    if nu is None:
        nu = args.nu_guess
        nu_err = 0.0
        print(f"Curvature fit failed; using nu_guess = {nu}")

    # ---- Step 5: Figure 3 — Scaling collapse ----
    print(f"\n=== Figure 3: Scaling collapse (nu={nu:.3f}) ===")
    plot_figure3(data, W_arr, Ny_list, colors, Wc, nu, args.outdir)

    # ---- Summary ----
    print(f"\n{'='*50}")
    print(f"RESULTS:")
    print(f"  Wc = {Wc:.4f}")
    print(f"  nu = {nu:.4f} ± {nu_err:.4f}")
    print(f"  Figures saved to: {args.outdir}/")
    print(f"{'='*50}")


if __name__ == '__main__':
    main()
