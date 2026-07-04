#!/usr/bin/env python3
"""
fss_tools.py
============
General finite-size scaling (FSS) tools. Physics-blind: works for any
order parameter that obeys the single-parameter scaling ansatz

    O(W, L) = f[ (W - Wc) * L^{1/nu} ]

Three steps:
    1. pairwise_crossings  -- initial estimate of Wc from curve intersections
    2. fit_fss             -- optimize (Wc, nu) by minimizing collapse chi2
    3. bootstrap_fss       -- parametric bootstrap for error bars on (Wc, nu)
"""

import numpy as np
from scipy.optimize import minimize


# --------------------------------------------------------------------------- #
#  Step 1 — initial Wc estimate from pairwise crossings
# --------------------------------------------------------------------------- #
def pairwise_crossings(W_arr, cbar_by_L, L_list):
    """
    Find all W values where curves for different L cross, via linear
    interpolation between adjacent W grid points.

    Parameters
    ----------
    W_arr     : 1D array of disorder values (length N_W), sorted ascending
    cbar_by_L : dict {L: 1D array of length N_W}  -- mean marker per (W, L)
    L_list    : list of L values to consider

    Returns
    -------
    crossings : 1D array of crossing W values (may be empty if no crossings)
    """
    crossings = []
    L_sorted  = sorted(L_list)

    for i, L1 in enumerate(L_sorted):
        for L2 in L_sorted[i + 1:]:
            diff = cbar_by_L[L1] - cbar_by_L[L2]
            for j in range(len(diff) - 1):
                # sign change -> crossing between W[j] and W[j+1]
                if diff[j] * diff[j + 1] < 0:
                    # linear interpolation for the crossing W
                    frac   = diff[j] / (diff[j] - diff[j + 1])
                    W_cross = W_arr[j] + frac * (W_arr[j + 1] - W_arr[j])
                    crossings.append(W_cross)

    return np.array(crossings)


# --------------------------------------------------------------------------- #
#  Step 2 — collapse quality metric and optimization
# --------------------------------------------------------------------------- #
def _collapse_chi2(params, W_flat, L_flat, cbar_flat, csem_flat,
                   poly_degree=4, x_range=8.0):
    """
    For given (Wc, nu), compute the chi2 of a polynomial fit to the
    collapsed data  x = (W - Wc) * L^{1/nu},  y = cbar.

    A perfect collapse -> all curves land on one master curve -> small chi2.
    Poor (Wc, nu) -> curves do not overlap -> large chi2.

    Parameters
    ----------
    params     : (Wc, nu)
    W_flat     : 1D array, disorder value for each data point
    L_flat     : 1D array, system size for each data point
    cbar_flat  : 1D array, mean marker for each data point
    csem_flat  : 1D array, standard error of mean for each data point
    poly_degree: degree of the polynomial used as the master curve
    x_range    : only use points with |x| < x_range in the fit

    Returns
    -------
    chi2 : float (lower = better collapse)
    """
    Wc, nu = params

    # hard bounds to keep optimizer in physical region
    W_min, W_max = W_flat.min(), W_flat.max()
    if nu < 0.3 or nu > 15.0:
        return 1e10
    if Wc < W_min + 0.05 or Wc > W_max - 0.05:
        return 1e10

    # scaled variable
    x = (W_flat - Wc) * (L_flat ** (1.0 / nu))

    # restrict to overlap region and valid error bars
    mask = (np.abs(x) < x_range) & (csem_flat > 0)
    if mask.sum() < poly_degree + 3:
        return 1e10

    # weighted polynomial fit to master curve
    weights = 1.0 / csem_flat[mask] ** 2
    try:
        coeffs = np.polyfit(x[mask], cbar_flat[mask], poly_degree,
                            w=np.sqrt(weights))
        c_fit  = np.polyval(coeffs, x[mask])
        # weighted chi2 per data point
        chi2   = float(np.sum(weights * (cbar_flat[mask] - c_fit) ** 2)
                       / mask.sum())
        return chi2
    except Exception:
        return 1e10


def fit_fss(W_flat, L_flat, cbar_flat, csem_flat,
            Wc_init, nu_init, poly_degree=4):
    """
    Optimize (Wc, nu) by minimizing the collapse chi2 via Nelder-Mead.

    Parameters
    ----------
    W_flat, L_flat   : 1D arrays of shape (N_points,)
    cbar_flat        : 1D array, mean marker values
    csem_flat        : 1D array, standard errors of the mean
    Wc_init, nu_init : initial guesses

    Returns
    -------
    result : dict with keys Wc, nu, chi2, success
    """
    def objective(p):
        return _collapse_chi2(p, W_flat, L_flat, cbar_flat, csem_flat,
                              poly_degree=poly_degree)

    res = minimize(objective, x0=[Wc_init, nu_init],
                   method='Nelder-Mead',
                   options={'xatol': 1e-5, 'fatol': 1e-8,
                            'maxiter': 20000, 'adaptive': True})

    Wc_opt, nu_opt = res.x
    chi2_opt       = float(res.fun)

    return {"Wc": float(Wc_opt), "nu": float(nu_opt),
            "chi2": chi2_opt, "success": res.success}


# --------------------------------------------------------------------------- #
#  Step 3 — bootstrap error bars
# --------------------------------------------------------------------------- #
def bootstrap_fss(W_flat, L_flat, cbar_flat, csem_flat,
                  Wc_init, nu_init, n_boot=300, poly_degree=4):
    """
    Parametric bootstrap for uncertainties on (Wc, nu).

    For each bootstrap sample:
        - perturb each cbar_i by Normal(0, csem_i)   [parametric resampling]
        - reoptimize (Wc, nu) from the same initial point

    Returns std of the bootstrap distribution as the 1-sigma error.

    Parameters
    ----------
    n_boot : number of bootstrap samples (300 gives stable 1-sigma estimates)

    Returns
    -------
    result : dict with keys Wc_err, nu_err, Wc_boot, nu_boot (arrays)
    """
    rng      = np.random.default_rng(42)
    Wc_boot  = np.zeros(n_boot)
    nu_boot  = np.zeros(n_boot)

    for b in range(n_boot):
        # perturb c_bar values within their statistical errors
        cbar_b = cbar_flat + rng.normal(0.0, csem_flat)

        res = fit_fss(W_flat, L_flat, cbar_b, csem_flat,
                      Wc_init, nu_init, poly_degree=poly_degree)
        Wc_boot[b] = res["Wc"]
        nu_boot[b] = res["nu"]

    return {"Wc_err": float(np.std(Wc_boot, ddof=1)),
            "nu_err": float(np.std(nu_boot, ddof=1)),
            "Wc_boot": Wc_boot,
            "nu_boot": nu_boot}


# --------------------------------------------------------------------------- #
#  Utility: evaluate master curve polynomial for plotting
# --------------------------------------------------------------------------- #
def master_curve(Wc, nu, W_flat, L_flat, cbar_flat, csem_flat,
                 poly_degree=4, x_range=8.0):
    """
    Given optimal (Wc, nu), return the master curve polynomial evaluated
    on a fine x-grid, for plotting on top of the collapsed data.

    Returns (x_fine, c_fine) for the master curve.
    """
    x    = (W_flat - Wc) * (L_flat ** (1.0 / nu))
    mask = (np.abs(x) < x_range) & (csem_flat > 0)

    weights = 1.0 / csem_flat[mask] ** 2
    coeffs  = np.polyfit(x[mask], cbar_flat[mask], poly_degree,
                         w=np.sqrt(weights))

    x_fine = np.linspace(x[mask].min(), x[mask].max(), 500)
    c_fine = np.polyval(coeffs, x_fine)
    # clamp to [0, 1] (physical range of the marker)
    c_fine = np.clip(c_fine, 0.0, 1.0)

    return x_fine, c_fine
