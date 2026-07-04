#!/usr/bin/env python3
"""
run_point.py
============
Runner for one (W, L) point of the LCM disorder-driven transition sweep.

This file is PHYSICS-BLIND: it knows nothing about the PDW model.
All model physics lives in AS/physics/pdw_lcm.py. This file only:
    1. Parses (W_idx, L_idx) from the command line
    2. Maps indices to physical values via the parameter grids below
    3. Calls physics functions in a joblib parallel loop
    4. Saves one fully-tagged .npz file to --outdir

Called by submit.slurm. The parameter grids defined here MUST match the
decomposition used in submit.slurm exactly -- both files use the same
N_W and N_L so that SLURM_ARRAY_TASK_ID maps to the right point.

Output file per point:
    data/lcm_W{W:.4f}_L{L:03d}.npz
    contains: cs (n_real,), gaps (n_real,), W, L, all model params, seed
"""

import os
import sys
import argparse
import numpy as np
from joblib import Parallel, delayed

# ------------------------------------------------------------------ #
#  Add physics directory to path
# ------------------------------------------------------------------ #
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'physics'))
from pdw_lcm import Config, centre_marker_upspin, make_disorder

# ------------------------------------------------------------------ #
#  Parameter grids
#  MUST match submit.slurm: N_W = len(W_LIST), N_L = len(L_LIST)
# ------------------------------------------------------------------ #
W_LIST = np.linspace(0.5, 7.0, 30)                       # 30 disorder values
L_LIST = [15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35]   # 11 system sizes (all odd)

N_W = len(W_LIST)   # 30
N_L = len(L_LIST)   # 11

# ------------------------------------------------------------------ #
#  Run settings
# ------------------------------------------------------------------ #
N_REAL = 30_000   # disorder realizations per (W, L) point  [Mildner: 3e4]
N_JOBS = 8        # joblib parallel workers (match --cpus-per-task in SLURM)

# ------------------------------------------------------------------ #
#  Model parameters (satisfy all insulating-phase constraints, t=1)
#  t'=1, d0x=3, d0y=1  -->  mu*=-0.5, indirect gap=1.0
# ------------------------------------------------------------------ #
TP  = 1.0
D0X = 3.0
D0Y = 1.0
MU  = -0.5


# ------------------------------------------------------------------ #
#  Single-realization worker (called by joblib)
# ------------------------------------------------------------------ #
def _run_one_realization(real_idx: int, W: float, L: int,
                         child_seed: np.random.SeedSequence):
    """
    Run one disorder realization and return (C_s, gap).

    Each realization gets its own SeedSequence child, spawned from the
    root seed for this (W_idx, L_idx) point -- fully deterministic and
    independent of run order or number of workers.
    """
    rng = np.random.default_rng(child_seed)
    cfg = Config(L=L, tp=TP, d0x=D0X, d0y=D0Y, mu=MU, W=W)
    eps = make_disorder(cfg, rng)
    c, gap = centre_marker_upspin(cfg, eps, window=1)
    return float(c), float(gap)


# ------------------------------------------------------------------ #
#  Main
# ------------------------------------------------------------------ #
def main():
    parser = argparse.ArgumentParser(description="Run one (W, L) LCM point")
    parser.add_argument('--W_idx',  type=int, required=True,
                        help="Index into W_LIST (0 to N_W-1)")
    parser.add_argument('--L_idx',  type=int, required=True,
                        help="Index into L_LIST (0 to N_L-1)")
    parser.add_argument('--outdir', type=str, default='data',
                        help="Directory to write output .npz file")
    args = parser.parse_args()

    # map indices to physical values
    W = float(W_LIST[args.W_idx])
    L = int(L_LIST[args.L_idx])

    # root seed: unique and reproducible for each (W_idx, L_idx) pair
    # using a simple integer encoding: W_idx * N_L + L_idx
    root_seed = args.W_idx * N_L + args.L_idx
    root_ss   = np.random.SeedSequence(root_seed)

    # spawn one child SeedSequence per realization -- order-independent
    child_seeds = root_ss.spawn(N_REAL)

    print(f"[run_point] W={W:.4f} (idx={args.W_idx})  "
          f"L={L} (idx={args.L_idx})  "
          f"n_real={N_REAL}  n_jobs={N_JOBS}  root_seed={root_seed}")

    # parallel loop over realizations
    results = Parallel(n_jobs=N_JOBS)(
        delayed(_run_one_realization)(i, W, L, child_seeds[i])
        for i in range(N_REAL)
    )

    cs   = np.array([r[0] for r in results], dtype=np.float64)
    gaps = np.array([r[1] for r in results], dtype=np.float64)

    # save -- one file per (W, L) point, fully parameter-tagged filename
    os.makedirs(args.outdir, exist_ok=True)
    fname = os.path.join(args.outdir, f'lcm_W{W:.4f}_L{L:03d}.npz')

    np.savez(fname,
             # raw per-realization data
             cs   = cs,
             gaps = gaps,
             # physical parameters (for self-contained files)
             W    = W,
             L    = L,
             tp   = TP,
             d0x  = D0X,
             d0y  = D0Y,
             mu   = MU,
             # bookkeeping
             n_real    = N_REAL,
             root_seed = root_seed)

    print(f"[run_point] Saved {fname}  "
          f"mean_cs={cs.mean():.4f}  gap_mean={gaps.mean():.4f}  "
          f"gap_min={gaps.min():.4f}")


if __name__ == '__main__':
    main()
