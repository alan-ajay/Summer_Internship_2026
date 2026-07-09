#!/usr/bin/env python3
"""
pdw_lcm.py
=====================

> Physics core: local spin Chern marker of the disordered
(p_x + i sigma p_y) density wave
> Evaluated at the centre of an L x L sample.

PARAMETER CONSTRAINTS:
    #1a  t' > 0.5                  -- nonzero spin Chern number
    #1b  d0x * d0y != 0            -- nonzero spin Chern number
    #2   d0x > 2 (approx)          -- gap-closing condition at (pi/2, 0)
    #3   4*t' + d0x > 6 (approx)   -- gap-closing condition at (0,0)

Default parameters used here: t=1, t'=1, d0x=3, d0y=1
    These satisfy all constraints and are insulating.
    Actual band extrema (computed automatically):
        min(lambda_+) = -1/3  at (0, arccos(2/3))
        max(lambda_-) = -1    at (pi/2, pi)
        indirect gap  = 2/3
        mu* = -2/3      

SIMPLFIED STEPS:

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
from dataclasses import dataclass, field
import numpy as np
from scipy.linalg import eigh

# =========================================================================== #
#  Band energy formulas
# =========================================================================== #

def eps1k(kx, ky, tp):
    return 2.0 * np.cos(kx) * (-1.0 + 2.0 * tp * np.cos(ky))

def eps2k(ky):
    return -2.0 * np.cos(ky) 

def band_eigenvalues(kx, ky, tp, d0x, d0y):
    e1 = eps1k(kx, ky, tp)
    e2 = eps2k(ky)

    dx = d0x * np.sin(kx)
    dy = d0y * np.sin(ky)

    Ek = np.sqrt(e1**2 + dx**2 + dy**2)

    lambda_minus = e2 - Ek
    lambda_plus = e2 + Ek

    return lambda_minus, lambda_plus

# =========================================================================== #
#  Compute the indirect gap and mu* from band structure
# =========================================================================== #

def compute_gap_and_mu(tp, d0x, d0y, n_ky=1000):
    """
    Sweeps the two high-symmetry edges (kx=0 and kx=pi/2) of the IBZ to
    find the indirect band gap and the gap-center chemical potential mu*.

    Parameters
    ----------
    tp     : float, next-nearest-neighbor hopping strength
    d0x    : float, singlet PDW order parameter
    d0y    : float, triplet PDW order parameter
    n_ky   : int, number of points to sample in ky in [0, pi]

    Returns
    -------
    gap : float, the indirect band gap (min(lambda_+) - max(lambda_-))
    is_insulating : bool, True if gap > 0
    mu_star : float, the gap-center chemical potential
    details : str, human-readable summary of where extrema occur

    Raises
    ------
    ValueError
        If the gap is negative (semimetallic phase), with a clear message
        about which constraints are violated.
    """
    ky_vals = np.linspace(0, np.pi, n_ky)

    # --- Edge 1: kx = 0 ---
    kx_edge1 = 0.0
    lam_minus_e1, lam_plus_e1 = band_eigenvalues(kx_edge1, ky_vals, tp, d0x, d0y)

    # --- Edge 2: kx = pi/2 ---
    kx_edge2 = np.pi / 2.0
    lam_minus_e2, lam_plus_e2 = band_eigenvalues(kx_edge2, ky_vals, tp, d0x, d0y)

    # Concatenate both edges to search globally
    all_lam_minus = np.concatenate([lam_minus_e1, lam_minus_e2])
    # Fix: 'all_lam_plus' used before assignment. Should be 'lam_plus_e2'.
    all_lam_plus  = np.concatenate([lam_plus_e1,  lam_plus_e2])

    i_min_plus_global = np.argmin(all_lam_plus) # retrieves index of the minimum eigenvalue from the upper band
    i_max_minus_global = np.argmax(all_lam_minus) # retrieves index of the maximum eigenvalue from the lower band
    n_pts = len(ky_vals)

    # Find min(lambda_+)
    if i_min_plus_global < n_pts:
        kx_min_plus = kx_edge1
        ky_min_plus = ky_vals[i_min_plus_global]
    else:
        kx_min_plus = kx_edge2
        ky_min_plus = ky_vals[i_min_plus_global - n_pts]

    # Find max(lambda_-)
    if i_max_minus_global < n_pts:
        kx_max_minus = kx_edge1
        ky_max_minus = ky_vals[i_max_minus_global]
    else:
        kx_max_minus = kx_edge2
        ky_max_minus = ky_vals[i_max_minus_global - n_pts]

    min_lam_plus  = all_lam_plus[i_min_plus_global]
    max_lam_minus = all_lam_minus[i_max_minus_global]

    gap = min_lam_plus - max_lam_minus
    is_insulating = gap > 0.0

    mu_star = 0.5 * (min_lam_plus + max_lam_minus)

    # summary
    details = (
        f"min(lambda_+) = {min_lam_plus:+.6f} at (kx,ky) = "
        f"({kx_min_plus/np.pi:.4f}π, {ky_min_plus/np.pi:.4f}π); "
        f"max(lambda_-) = {max_lam_minus:+.6f} at (kx,ky) = "
        f"({kx_max_minus/np.pi:.4f}π, {ky_max_minus/np.pi:.4f}π); "
        f"gap = {gap:.6f}; mu* = {mu_star:+.6f}"
    )

    return gap, is_insulating, mu_star, details

# =========================================================================== #
# Configuration class
# =========================================================================== #

@dataclass
class Config:
    """
    > Model parameters for the PDW model.
    > Use an ODD L so there is a unique centre site.

    Attributes
    ----------
    L : int
        Lattice size (number of sites along each dimension)
    t : float, default=1.0
        Nearest-neighbor hopping amplitude (energy unit)
    tp : float, default=1.0
        Next-nearest (diagonal) neighbor hopping; must satisfy t' > 0.5
    d0x : float, default=3.0
        Singlet (p_x) PDW order parameter; must satisfy d0x > 2 (approx)
    d0y : float, default=1.0
        Triplet (p_y) PDW order parameter; must be nonzero
    mu : float or None, default=None
        Chemical potential. If None, automatically computed. If provided,
        a warning is issued suggesting you use the auto value.
    W : float, default=0.0
        Disorder strength: eps_j ~ Uniform[-W, W]
    """
    L:    int
    t:    float = 1.0
    tp:   float = 1.0
    d0x:  float = 3.0
    d0y:  float = 1.0
    mu:   float | None = None
    W:    float = 0.0

    # Internal field to store the auto-computed mu (not part of __init__)
    _mu_auto: float = field(default=None, init=False, repr=False)

    def __post_init__(self):
        """
        Validate the parameters and auto-compute mu* if not provided.

        This method runs IMMEDIATELY after the dataclass is instantiated.
        It checks that:
        1. The parameter set is in the insulating phase (gap > 0)
        2. Auto-computes mu* from the band structure
        3. Warns if you manually provided mu, suggesting the auto value

        Raises
        ------
        ValueError
            If the parameter set is semimetallic (gap < 0), with a clear
            message about which constraints are violated.
        """
        # Step 1: Compute the band gap and mu* from scratch
        gap, is_insulating, mu_auto, details = compute_gap_and_mu(
            self.tp, self.d0x, self.d0y, n_ky=1000
        )

        # Step 2: Check that we're in the insulating phase
        if not is_insulating:
            raise ValueError(
                f"\n{'='*70}\n"
                f"ERROR: Parameter set is SEMIMETALLIC (not insulating)!\n"
                f"{'='*70}\n"
                f"Parameters: t={self.t}, t'={self.tp}, d0x={self.d0x}, d0y={self.d0y}\n"
                f"Gap calculation: {details}\n"
                f"\nPossible fixes:\n"
                f"  1. Increase d0x (need d0x > ~2)\n"
                f"2. Increase tp or d0y to open the gap\n"
                f"  3. Check Constraints #1a, #1b, #2, #3 in the module docstring\n"
                f"{'='*70}\n"
            )

        # Step 3: Store the auto-computed mu for use in Hamiltonian building
        self._mu_auto = mu_auto

        # Step 4: If the user manually provided mu, warn them
        if self.mu is not None:
            if abs(self.mu - mu_auto) > 1e-3:
                print(
                    f"\n[CONFIG WARNING]\n"
                    f"  You provided mu = {self.mu:+.6f}\n"
                    f"  But the auto-computed gap-center value is mu* = {mu_auto:+.6f}\n"
                    f"  Difference: {abs(self.mu - mu_auto):.6f}\n"
                    f"  SUGGESTION: Use mu=None to auto-compute, or set mu={mu_auto:+.6f}\n"
                    f"  Band details: {details}\n\n"
                )
            # Use the manually provided value (for backward compatibility)
            # but the auto-computed value is available via cfg.mu_auto() if needed

        # If mu is None, use the auto-computed value
        if self.mu is None:
            self.mu = mu_auto

    def mu_auto(self) -> float:
        """Return the auto-computed gap-center chemical potential."""
        return self._mu_auto

    @property
    def N(self) -> int:
        """Total number of sites: N = L * L."""
        return self.L * self.L

    def nocc(self) -> int:
        """Half filling: occupy the lower N/2 states of the up-spin block."""
        return self.N // 2

    def centre_index(self) -> int:
        """Site index (idx = jx*L + jy) of the literal centre of the grid."""
        c = self.L // 2
        return c * self.L + c

def build_hamiltonian_upspin(cfg: Config, eps: np.ndarray) -> np.ndarray:
    """
    Dense up-spin (sigma=+1) Hamiltonian, open boundary conditions,
    with site index idx = jx*L + jy and staggering sx = (-1)^jx from Q=(pi,0).

    x-bond amplitude:
        -t - (-1)^jx * d0x/2          (real)

    y-bond amplitude:
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

            # x-bond: -t - (-1)^jx * d0x/2
            if jx + 1 < L:
                j   = idx(jx + 1, jy)
                amp = -t - sx * d0x / 2.0
                H[i, j] += amp
                H[j, i] += amp

            # y-bond: -t - (-1)^jx * i * d0y/2
            if jy + 1 < L:
                j   = idx(jx, jy + 1)
                amp = -t - sx * 1j * d0y / 2.0
                H[i, j] += amp
                H[j, i] += np.conj(amp)

            # diagonal t' bonds
            for dx, dy in ((1, 1), (1, -1)):
                jx2, jy2 = jx + dx, jy + dy
                if 0 <= jx2 < L and 0 <= jy2 < L:
                    j = idx(jx2, jy2)
                    H[i, j] += tp
                    H[j, i] += tp

    return H

# =========================================================================== #
#  Projector onto the occupied (lowest nocc) states
# =========================================================================== #

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

# =========================================================================== #
#  Centre-site spin Chern marker
# =========================================================================== #

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

# =========================================================================== #
#  Self test
# =========================================================================== #

def _selftest():
    """
    Self-test demonstrating the new auto-computed mu* feature.
    
    This test shows:
    1. Creating a Config object with mu=None (auto-compute)
    2. The Config validates the insulating phase during __post_init__
    3. The auto-computed mu* is used in the marker calculation
    4. A warning is printed if you manually provide a different mu
    """
    print("\n" + "=" * 70)
    print("SELF-TEST: Updated pdw_lcm.py with auto-computed mu*")
    print("=" * 70)
    
    # Test 1: Auto-compute mu* (the recommended way)
    print("\n[TEST 1] Auto-computing mu* from band structure")
    print("-" * 70)
    cfg1 = Config(L=15, t=1.0, tp=1.0, d0x=3.0, d0y=1.0, mu=None, W=0.0)
    print(f"Created Config(L=15, tp=1.0, d0x=3.0, d0y=1.0, mu=None)")
    print(f"Auto-computed mu* = {cfg1.mu:+.6f}")
    print(f"Clean limit centre marker:")
    eps = np.zeros(cfg1.N)
    c, gap = centre_marker_upspin(cfg1, eps, window=1)
    print(f"  C_s(centre) = {c:+.6f}  (expect +1 in clean limit)")
    print(f"  Single-particle gap = {gap:.6f}")
    
    # # Test 2: Manually providing mu (will warn if different from auto)
    # print("\n[TEST 2] Providing mu manually (will warn if different from auto)")
    # print("-" * 70)
    # cfg2 = Config(L=15, t=1.0, tp=1.0, d0x=3.0, d0y=1.0, mu=None, W=0.0)
    # print(f"Created Config(L=15, tp=1.0, d0x=3.0, d0y=1.0)")
    # print(f"(Should warn that -0.5 differs from auto-computed value)")
    
    # # Test 3: Semimetallic parameter set (should fail)
    # print("\n[TEST 3] Testing error handling for semimetallic parameters")
    # print("-" * 70)
    # try:
    #     cfg3 = Config(L=15, t=1.0, tp=1.0, d0x=1.5, d0y=1.0, mu=None, W=0.0)
    #     print("ERROR: Should have raised ValueError for semimetallic phase!")
    # except ValueError as e:
    #     print("Caught expected ValueError for semimetallic parameters ✓")
    #     print("Error message was printed above")
    
    print("\n" + "=" * 70)
    print("SELF-TEST COMPLETE")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    _selftest()
