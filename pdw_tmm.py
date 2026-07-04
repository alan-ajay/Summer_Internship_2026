#!/usr/bin/env python3
"""
pdw_tmm.py
==========

Physics core: transfer-matrix / Lyapunov-exponent (TMM) probe of the
disordered PDW transition. Full y-hopping is kept throughout.

PARAMETER CONSTRAINTS (derived from Hsu's band structure, t=1):
    #1a  t' > 0.5                  -- nonzero spin Chern number
    #1b  d0x * d0y != 0            -- nonzero spin Chern number
    #2   d0x > 2                   -- closes gap at (pi/2, 0)
    #3   4*t' + d0x > 6            -- closes gap at (0,0) vs (pi/2, pi)

Gap-centred chemical potential (exact):
    mu* = 0.5 * (max(lambda_-) + min(lambda_+))

Default parameters: t=1, t'=1, d0x=3, d0y=1
    min(lambda_+) = 0,  max(lambda_-) = -1  -->  mu* = -0.5,  gap = 1.0

WHAT TMM COMPUTES:
Build a quasi-1D strip of transverse width M, push the transfer matrix
along the strip length, extract the smallest positive Lyapunov exponent
gamma_min, and form xi_M = 1/gamma_min and Lambda_M = xi_M / M.
At the topological transition Lambda_M peaks and becomes approximately
M-independent; away from it Lambda_M decreases with M (localized bulk).

WHY ONE SPIN BLOCK:
Real spin-independent on-site disorder gives H_down = H_up^*, so both
spin blocks share the same Lyapunov spectrum. Only sigma=+1 is built.

WHY TWISTED BOUNDARY CONDITIONS:
The inter-slice hopping T+ is singular when a transverse momentum hits
cos(theta) = (t - d0x/2) / (2*t'). A generic irrational twist shifts
all momenta off that resonance simultaneously.
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from numpy.linalg import qr, solve


# --------------------------------------------------------------------------- #
#  Configuration
# --------------------------------------------------------------------------- #
@dataclass
class Config:
    """
    Strip configuration. Ly = M = transverse width.
    Default parameters satisfy all insulating-phase constraints with
    full y-hopping kept (see module docstring).
    """
    Ly:  int
    t:   float = 1.0    # nearest-neighbour hopping (energy unit)
    tp:  float = 1.0    # next-nearest (diagonal) hopping; need t' > 0.5
    d0x: float = 3.0    # singlet p_x pairing amplitude; need d0x > 2
    d0y: float = 1.0    # triplet p_y pairing amplitude; need d0y != 0
    mu:  float = -0.5   # chemical potential; mu* = -0.5 for default (t,tp,d0x)


# --------------------------------------------------------------------------- #
#  Per-parity slice constants (disorder-independent)
# --------------------------------------------------------------------------- #
def build_slice_constants(cfg: Config, sigma: int, twist: float):
    """
    Pre-compute the disorder-independent parts of each slice, split by
    the staggering parity (-1)^jx of the x-column index.

    Returns {parity: (Tplus, Hy)} where:
        Tplus : M x M inter-slice hopping matrix (x-direction)
        Hy    : M x M intra-slice hopping matrix (y-direction, no disorder)

    x-bond amplitude for column of parity p (sx = +1 if p=0, -1 if p=1):
        -t - sx * d0x/2          (real, diagonal in Tplus)

    y-bond amplitude (full hopping + PDW pairing, sigma = +1 or -1):
        -t - sx * i * sigma * d0y/2

    Diagonal t' bonds couple adjacent rows within and between slices.
    Twisted PBC with phase exp(i*twist) wraps the transverse direction.
    """
    M  = cfg.Ly
    t, tp, d0x, d0y = cfg.t, cfg.tp, cfg.d0x, cfg.d0y
    eph = np.exp(1j * twist)

    consts = {}
    for parity in (0, 1):
        sx = 1.0 if parity == 0 else -1.0

        # --- inter-slice hopping Tplus (x-bond diagonal + t' off-diagonal) ---
        Tp = np.zeros((M, M), dtype=np.complex128)
        np.fill_diagonal(Tp, -t - sx * d0x / 2.0)   # x-bond: full -t + PDW
        for m in range(M - 1):
            Tp[m + 1, m] += tp                        # t' sub-diagonal
            Tp[m, m + 1] += tp                        # t' super-diagonal
        if M > 2:
            Tp[0, M - 1] += tp * eph                  # twisted PBC wrap
            Tp[M - 1, 0] += tp * np.conj(eph)

        # --- intra-slice y-hopping Hy (full -t + PDW pairing) ---
        Hy  = np.zeros((M, M), dtype=np.complex128)
        ay  = -t - sx * 1j * sigma * d0y / 2.0       # full y-bond amplitude
        for m in range(M - 1):
            Hy[m, m + 1] += ay
            Hy[m + 1, m] += np.conj(ay)
        if M > 2:
            Hy[0, M - 1] += ay * eph
            Hy[M - 1, 0] += np.conj(ay) * np.conj(eph)

        consts[parity] = (Tp, Hy)
    return consts


# --------------------------------------------------------------------------- #
#  Lyapunov exponent (smallest positive) via the QR method
# --------------------------------------------------------------------------- #
def lyapunov_min(cfg: Config, W: float, E: float = 0.0, sigma: int = +1,
                 twist: float = 0.7, seed: int = 0,
                 L: int = 40000, q: int = 8, nseg: int = 20):
    """
    Push the transfer matrix for L slices, re-orthonormalising (QR) every
    q steps (Benettin's method). log|diag(R)| is accumulated at a fixed
    column index and sorted once at the very end.

    The run is split into nseg segments; the spread of per-segment
    estimates over the stationary second half gives the statistical error.

    Returns dict: {xi, xi_err, gamma_min, gamma_err}
    """
    M  = cfg.Ly
    rng = np.random.default_rng(seed)
    consts = build_slice_constants(cfg, sigma, twist)
    (A, HyA) = consts[0]
    (B, HyB) = consts[1]
    I  = np.eye(M)
    Z  = np.zeros((M, M))

    Q   = np.eye(2 * M, dtype=np.complex128)[:, :M]   # M orthonormal seed vectors
    acc = np.zeros(M)                                  # accumulated log|R_ii|
    seg_len   = max(1, L // nseg)
    ckpt_acc  = []
    ckpt_steps = []

    total = 0
    for n in range(L):
        eps = rng.uniform(-W, W, M)

        # alternate slice parity along the strip
        if n % 2 == 0:
            Tp, Hy, Tprev = A, HyA, B
        else:
            Tp, Hy, Tprev = B, HyB, A

        # intra-slice matrix including disorder and chemical potential
        Hin = Hy.copy()
        Hin[np.diag_indices(M)] += eps - cfg.mu

        # 2M x M transfer matrix block applied to current Q
        TL = solve(Tp, E * I - Hin)    # Tplus^{-1} (E - H_intra)
        TR = -solve(Tp, Tprev)         # -Tplus^{-1} Tplus_prev
        Mn = np.block([[TL, TR], [I, Z]])
        Q  = Mn @ Q
        total += 1

        # periodic re-orthonormalisation
        if (n + 1) % q == 0:
            Q, R = qr(Q)
            acc += np.log(np.abs(np.diag(R)))

        if total % seg_len == 0:
            ckpt_acc.append(acc.copy())
            ckpt_steps.append(total)

    # smallest positive Lyapunov exponent
    exponents = np.sort(acc / total)
    gamma_min = exponents[0]
    imin      = int(np.argmin(acc))

    # statistical error from per-segment estimates (stationary tail)
    ck = np.array(ckpt_acc)
    cs = np.array(ckpt_steps, dtype=float)
    if len(cs) >= 4:
        seg_vals  = np.diff(ck[:, imin]) / np.diff(cs)
        tail      = seg_vals[len(seg_vals) // 2:]
        gamma_err = (tail.std(ddof=1) / np.sqrt(len(tail))
                     if len(tail) > 1 else abs(gamma_min) * 0.1)
    else:
        gamma_err = abs(gamma_min) * 0.1

    xi     = 1.0 / gamma_min
    xi_err = gamma_err / gamma_min ** 2
    return {"xi": xi, "xi_err": xi_err,
            "gamma_min": gamma_min, "gamma_err": gamma_err}


# --------------------------------------------------------------------------- #
#  Localization length (twist-averaged)
# --------------------------------------------------------------------------- #
def localization_length(cfg: Config, W: float, E: float = 0.0, sigma: int = +1,
                        twists=(0.7,), seed: int = 0,
                        L: int = 40000, q: int = 8, nseg: int = 20):
    """
    Localization length averaged over a set of transverse twist angles.
    Returns xi, xi_err, Lambda = xi/M, and per-twist xi values
    (needed for a proper bootstrap error, not just a propagated SEM).
    """
    xis, errs = [], []
    for k, tw in enumerate(twists):
        out = lyapunov_min(cfg, W, E=E, sigma=sigma, twist=tw,
                           seed=seed + 1000 * k, L=L, q=q, nseg=nseg)
        xis.append(out["xi"])
        errs.append(out["xi_err"])

    xis    = np.asarray(xis)
    xi     = float(xis.mean())
    xi_err = float(np.sqrt(np.mean(np.square(errs)) +
                           (xis.std() ** 2 if len(xis) > 1 else 0.0)))
    return {"xi": xi, "xi_err": xi_err, "Lambda": xi / cfg.Ly, "M": cfg.Ly,
            "xis_per_twist": xis, "errs_per_twist": np.asarray(errs)}


# --------------------------------------------------------------------------- #
#  Clean-limit self-test  (run:  python pdw_tmm.py)
# --------------------------------------------------------------------------- #
def _selftest():
    print("Transfer-matrix self-test: full y-hopping, E=0")
    print("Parameters: t=1, t'=1, d0x=3, d0y=1, mu=-0.5  (indirect gap=1.0)")
    print("Expect Lambda_M to peak near W_c and decrease away from it")
    print("=" * 65)
    print(f"{'W':>5}   " + "   ".join(f"M={M:>2}" for M in (8, 16, 24)))
    for W in (1.0, 2.0, 3.0, 4.0, 5.0):
        row = []
        for M in (8, 16, 24):
            cfg = Config(Ly=M, t=1.0, tp=1.0, d0x=3.0, d0y=1.0, mu=-0.5)
            out = localization_length(cfg, W, twists=(0.7,), L=8000, seed=1)
            row.append(out["Lambda"])
        print(f"  W={W:.1f}   " + "   ".join(f"{v:6.3f}" for v in row))


if __name__ == "__main__":
    _selftest()
