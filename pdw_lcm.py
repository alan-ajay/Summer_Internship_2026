#!/usr/bin/env python3
"""
pdw_lcm.py
==========

Physics core: local spin Chern marker of the disordered
(p_x + i sigma p_y) density wave (Hsu 2012), evaluated at the centre
of an L x L sample. Full y-hopping is kept throughout.

PARAMETER CONSTRAINTS (derived from Hsu's band structure, t=1):
    #1a  t' > 0.5                  -- nonzero spin Chern number
    #1b  d0x * d0y != 0            -- nonzero spin Chern number
    #2   d0x > 2                   -- closes gap at (pi/2, 0)
    #3   4*t' + d0x > 6            -- closes gap at (0,0) vs (pi/2, pi)

Gap-centred chemical potential (exact, no numerical search needed):
    mu* = 0.5 * (max(lambda_-) + min(lambda_+))
        = 0.5 * (2 - d0x) + 0.5 * min(4t'-4, d0x-2)

Default parameters used here: t=1, t'=1, d0x=3, d0y=1
    min(lambda_+) = 0  at (0,0)
    max(lambda_-) = -1 at (pi/2, pi)
    mu* = -0.5,  indirect gap = 1.0

TWO PHYSICS SHORTCUTS (both exact):

1. C_s = C_up. Real spin-independent on-site disorder gives
   H_down = H_up^*, hence C_down(r) = -C_up(r) at every site and
   every realisation. So C_s(r) = C_up(r) and only the up-spin block
   is ever built or diagonalised.

2. MARKER FORMULA (Bianco-Resta commutator form):
       C(r) = -2 pi i <r| [P X P, P Y P] |r>
   evaluated via 6 matrix-vector products (O(N^2) per site) rather than
   forming the full N x N operator products (O(N^3)).
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.linalg import eigh


# --------------------------------------------------------------------------- #
#  Configuration
# --------------------------------------------------------------------------- #
@dataclass
class Config:
    """
    Model parameters. Use an ODD L so there is a unique centre site.
    Default parameters satisfy all insulating-phase constraints with
    full y-hopping kept (see module docstring).
    """
    L:   int
    t:   float = 1.0     # nearest-neighbour hopping (set to 1, energy unit)
    tp:  float = 1.0     # next-nearest (diagonal) hopping; need t' > 0.5
    d0x: float = 3.0     # singlet p_x pairing amplitude; need d0x > 2
    d0y: float = 1.0     # triplet p_y pairing amplitude; need d0y != 0
    mu:  float = -0.5    # chemical potential; mu* = -0.5 for default (t,tp,d0x)
    W:   float = 0.0     # disorder strength: eps_j ~ Uniform[-W, W]

    @property
    def N(self) -> int:
        return self.L * self.L

    def nocc(self) -> int:
        """Half filling: occupy the lower N/2 states of the up-spin block."""
        return self.N // 2

    def centre_index(self) -> int:
        """Site index (idx = jx*L + jy) of the literal centre of the grid."""
        c = self.L // 2
        return c * self.L + c


# --------------------------------------------------------------------------- #
#  Hamiltonian (up-spin block only)
# --------------------------------------------------------------------------- #
def build_hamiltonian_upspin(cfg: Config, eps: np.ndarray) -> np.ndarray:
    """
    Dense up-spin (sigma=+1) Hamiltonian, open boundary conditions,
    full y-hopping kept. Implements Hsu Eq. 22 with site index
    idx = jx*L + jy and staggering sx = (-1)^jx from Q=(pi,0).

    x-bond amplitude (Hsu Eq. 22, sigma=+1):
        -t - (-1)^jx * d0x/2          (real)

    y-bond amplitude (Hsu Eq. 22, sigma=+1):
        -t - (-1)^jx * i * d0y/2      (complex; full -t hopping kept)

    diagonal t' bond: tp (real, no staggering)

    eps : on-site disorder array of length N, Uniform[-W, W].
          The SAME array would be used for the down-spin block, making
          the C_s = C_up shortcut exact.
    """
    L, t, tp, d0x, d0y, mu = cfg.L, cfg.t, cfg.tp, cfg.d0x, cfg.d0y, cfg.mu
    N = cfg.N
    H = np.zeros((N, N), dtype=np.complex128)

    def idx(jx, jy):
        return jx * L + jy

    for jx in range(L):
        sx = 1.0 if (jx % 2 == 0) else -1.0    # (-1)^jx staggering
        for jy in range(L):
            i = idx(jx, jy)

            # on-site: disorder + chemical potential
            H[i, i] += eps[i] - mu

            # x-bond: -t - (-1)^jx * d0x/2  (real -> no conj needed)
            if jx + 1 < L:
                j   = idx(jx + 1, jy)
                amp = -t - sx * d0x / 2.0
                H[i, j] += amp
                H[j, i] += amp

            # y-bond: -t - (-1)^jx * i * d0y/2  (full hopping + PDW pairing)
            if jy + 1 < L:
                j   = idx(jx, jy + 1)
                amp = -t - sx * 1j * d0y / 2.0
                H[i, j] += amp
                H[j, i] += np.conj(amp)

            # diagonal t' bonds (forward pairs only; loop covers both directions)
            for dx, dy in ((1, 1), (1, -1)):
                jx2, jy2 = jx + dx, jy + dy
                if 0 <= jx2 < L and 0 <= jy2 < L:
                    j = idx(jx2, jy2)
                    H[i, j] += tp
                    H[j, i] += tp

    return H


# --------------------------------------------------------------------------- #
#  Projector onto the occupied (lowest nocc) states
# --------------------------------------------------------------------------- #
def occupied_projector(H: np.ndarray, nocc: int):
    """
    P = sum_{n=0}^{nocc-1} |psi_n><psi_n|

    Asks LAPACK for only the lowest nocc+1 eigenpairs (occupied states +
    the first empty one, so we can compute the single-particle gap).

    Returns (P, gap) where gap = E[nocc] - E[nocc-1].
    """
    n   = H.shape[0]
    top = min(nocc, n - 1)
    vals, vecs = eigh(H, subset_by_index=[0, top])
    occ  = vecs[:, :nocc]
    gap  = float(vals[nocc] - vals[nocc - 1]) if nocc < len(vals) else float("nan")
    P    = occ @ occ.conj().T
    return P, gap


# --------------------------------------------------------------------------- #
#  Centre-site spin Chern marker
# --------------------------------------------------------------------------- #
def centre_marker_upspin(cfg: Config, eps: np.ndarray, window: int = 1):
    """
    Spin Chern marker C_s averaged over a window x window block of central
    sites (window=1 -> the single centre site).

    Uses the O(N^2) contraction trick: for each site r, compute
        term1 = <r| P X P Y P |r>
        term2 = <r| P Y P X P |r>
    via 3 matrix-vector products each, rather than forming the full N x N
    operators P X P and P Y P explicitly.

    Returns (C_s_value, single_particle_gap).
    """
    L, N = cfg.L, cfg.N
    H    = build_hamiltonian_upspin(cfg, eps)
    P, gap = occupied_projector(H, cfg.nocc())

    site = np.arange(N)
    xs   = (site // L).astype(np.float64)   # x-coordinate of each site
    ys   = (site %  L).astype(np.float64)   # y-coordinate of each site

    c    = L // 2
    half = window // 2
    idxs = [jx * L + jy
            for jx in range(c - half, c + half + 1)
            for jy in range(c - half, c + half + 1)]

    vals = []
    for r in idxs:
        e_r      = np.zeros(N, dtype=complex)
        e_r[r]   = 1.0

        # term1 = <r| P X P Y P |r>  (right-to-left contraction)
        w = P @ e_r       # P|r>
        w = ys * w        # Y P|r>
        w = P @ w         # P Y P|r>
        w = xs * w        # X P Y P|r>
        w = P @ w         # P X P Y P|r>
        term1 = w[r]

        # term2 = <r| P Y P X P |r>  (swap X <-> Y)
        w = P @ e_r
        w = xs * w
        w = P @ w
        w = ys * w
        w = P @ w
        term2 = w[r]

        vals.append(np.real(-2j * np.pi * (term1 - term2)))

    return float(np.mean(vals)), gap


def make_disorder(cfg: Config, rng: np.random.Generator) -> np.ndarray:
    """Spin-independent on-site disorder: eps_j ~ Uniform[-W, W]."""
    return rng.uniform(-cfg.W, cfg.W, size=cfg.N)


# --------------------------------------------------------------------------- #
#  Clean-limit self-test  (run:  python pdw_lcm.py)
# --------------------------------------------------------------------------- #
def _selftest():
    print("Clean-limit self-test: centre-site spin Chern marker")
    print("Parameters: t=1, t'=1, d0x=3, d0y=1, mu=-0.5  (full y-hopping)")
    print("Indirect gap = 1.0 in the clean limit")
    print("=" * 65)
    for L in (15, 23, 31):
        cfg = Config(L=L, t=1.0, tp=1.0, d0x=3.0, d0y=1.0, mu=-0.5, W=0.0)
        eps = np.zeros(cfg.N)
        c, gap = centre_marker_upspin(cfg, eps, window=1)
        print(f"  L={L:>3}  single-particle gap={gap:.4f}  C_s(centre)={c:+.4f}"
              f"  (expect -> +1 as L grows)")


if __name__ == "__main__":
    _selftest()
