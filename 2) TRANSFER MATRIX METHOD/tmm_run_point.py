#!/usr/bin/env python3
"""
run_point.py  (TMM)
===================
Runner for one (W, Ny) point of the TMM disorder-driven transition sweep.

Physics-blind: all model physics lives in AS/physics/pdw_tmm.py.

For each (W, Ny) point the computation is:
    - Run lyapunov_min() independently for N_TWISTS twist angles
    - Each twist is one complete strip run of length L_STRIP slices
    - Parallelize over twists with joblib (one worker per twist)
    - Average xi over twists -> Lambda = xi / Ny

Why parallelize over twists (not realizations as in LCM):
    The TMM strip is self-averaging -- one long strip gives the
    disorder-averaged Lyapunov exponent directly (by Oseledets'
    theorem). There is no ensemble of independent realizations to
    parallelize over. Instead, multiple twist angles give independent
    estimates of xi (each sees a different transverse momentum grid),
    and averaging over them reduces the systematic error from any
    single resonant momentum. With N_TWISTS = 188, all 188 cores
    are used and the twist-averaged xi is very well converged.

Output: one .npz file per (W, Ny) point.
    tmm_W{W:.4f}_Ny{Ny:03d}.npz

Called by submit.slurm. Parameter grids here MUST match submit.slurm.

BLAS guard must come before import numpy.
"""

import os
import sys

# ------------------------------------------------------------------ #
#  BLAS thread guard -- MUST come before import numpy
# ------------------------------------------------------------------ #
os.environ["OMP_NUM_THREADS"]      = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"]      = "1"

import argparse
import numpy as np
from joblib import Parallel, delayed

# ------------------------------------------------------------------ #
#  Add physics directory to path
# ------------------------------------------------------------------ #
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'physics'))
from pdw_tmm import Config, lyapunov_min

# ------------------------------------------------------------------ #
#  Parameter grids
#  MUST match submit.slurm: N_W = len(W_LIST), N_NY = len(NY_LIST)
# ------------------------------------------------------------------ #
W_LIST  = np.linspace(1.5, 7.0, 25)                       # 25 disorder values

# Strip widths following Mildner supplement Sec. H:
# {19, 23, 27, ..., 59} step-4 plus two large sizes 99, 139
NY_LIST = [19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 99, 139]

N_W  = len(W_LIST)    # 25
N_NY = len(NY_LIST)   # 13

# ------------------------------------------------------------------ #
#  Run settings
# ------------------------------------------------------------------ #
# Strip length: number of transfer matrix multiplications per run.
# L=40_000 gives a first-pass result. For production (Mildner used
# L=10^6 to 10^7) increase this and resubmit.
L_STRIP = 40_000

# One worker per twist angle; reads SLURM allocation, falls back to 8
N_JOBS   = int(os.environ.get('SLURM_CPUS_PER_TASK', 8))
N_TWISTS = N_JOBS     # one twist per available core -> 188 for production

# ------------------------------------------------------------------ #
#  Model parameters (same as pdw_tmm.py defaults, satisfy all
#  insulating-phase constraints with t=1)
#  t'=1, d0x=3, d0y=1  -->  mu*=-0.5, indirect gap=1.0
# ------------------------------------------------------------------ #
TP  = 1.0
D0X = 3.0
D0Y = 1.0
MU  = -0.5


# ------------------------------------------------------------------ #
#  Single-twist worker (called by joblib)
# ------------------------------------------------------------------ #
def _run_one_twist(W: float, Ny: int, twist: float, seed: int):
    """
    Run one complete strip for a single twist angle.
    Returns the lyapunov_min result dict: {xi, xi_err, gamma_min, gamma_err}

    Each twist sees a different transverse momentum grid, so results
    from different twists are statistically independent.
    """
    cfg = Config(Ly=Ny, tp=TP, d0x=D0X, d0y=D0Y, mu=MU)
    return lyapunov_min(cfg, W,
                        E     = 0.0,
                        sigma = +1,
                        twist = float(twist),
                        seed  = seed,
                        L     = L_STRIP,
                        q     = 8,
                        nseg  = 20)


# ------------------------------------------------------------------ #
#  Main
# ------------------------------------------------------------------ #
def main():
    parser = argparse.ArgumentParser(
        description="Run one (W, Ny) TMM point")
    parser.add_argument('--W_idx',  type=int, required=True,
                        help="Index into W_LIST (0 to N_W-1)")
    parser.add_argument('--Ny_idx', type=int, required=True,
                        help="Index into NY_LIST (0 to N_NY-1)")
    parser.add_argument('--outdir', type=str, default='data',
                        help="Directory to write output .npz file")
    args = parser.parse_args()

    W  = float(W_LIST[args.W_idx])
    Ny = int(NY_LIST[args.Ny_idx])

    # Twist angles: N_TWISTS values uniformly spread over (0, 2pi),
    # avoiding exact 0 and 2pi to prevent any accidentally resonant momentum.
    twists = np.linspace(0.05 * np.pi, 1.95 * np.pi, N_TWISTS)

    # Deterministic seed per twist, unique to this (W_idx, Ny_idx, twist_idx)
    seeds = [
        args.W_idx * N_NY * N_TWISTS + args.Ny_idx * N_TWISTS + k
        for k in range(N_TWISTS)
    ]

    print(f"[tmm run_point] W={W:.4f} (idx={args.W_idx})  "
          f"Ny={Ny} (idx={args.Ny_idx})  "
          f"n_twists={N_TWISTS}  L_strip={L_STRIP}",
          flush=True)

    # Parallel loop: one worker per twist angle
    results = Parallel(n_jobs=N_JOBS)(
        delayed(_run_one_twist)(W, Ny, twists[k], seeds[k])
        for k in range(N_TWISTS)
    )

    # Collect per-twist results
    xis      = np.array([r['xi']        for r in results])
    xi_errs  = np.array([r['xi_err']    for r in results])
    gammas   = np.array([r['gamma_min'] for r in results])

    # Twist-averaged localization length and its error
    # (combined in quadrature: within-twist convergence error
    #  + between-twist variance)
    xi     = float(xis.mean())
    xi_err = float(np.sqrt(
        np.mean(xi_errs ** 2) +
        (xis.std(ddof=1) ** 2 if len(xis) > 1 else 0.0)
    ))
    Lambda = xi / Ny

    # Save -- one fully tagged file per (W, Ny) point
    os.makedirs(args.outdir, exist_ok=True)
    fname = os.path.join(args.outdir, f'tmm_W{W:.4f}_Ny{Ny:03d}.npz')

    np.savez(fname,
             # key observables
             xi      = xi,
             xi_err  = xi_err,
             Lambda  = Lambda,
             # per-twist raw data (needed for bootstrap in analysis)
             xis     = xis,
             xi_errs = xi_errs,
             gammas  = gammas,
             twists  = twists,
             # parameters (file is self-contained)
             W       = W,
             Ny      = Ny,
             tp      = TP,
             d0x     = D0X,
             d0y     = D0Y,
             mu      = MU,
             L_strip = L_STRIP,
             n_twists = N_TWISTS)

    print(f"[tmm run_point] Saved {fname}  "
          f"xi={xi:.4f}  xi_err={xi_err:.4f}  Lambda={Lambda:.4f}",
          flush=True)


if __name__ == '__main__':
    main()
