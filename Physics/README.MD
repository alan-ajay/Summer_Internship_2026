# Physics Folder — Function Reference

This file is a lookup directory for every function in `AS/physics/`.
Use it to quickly find what a function does, what it needs, and what
it returns — without re-reading the full source each time.

The physics folder is **physics-only**: no I/O, no SLURM, no file
saving. Everything here is called by the `run_point.py` runners in
`lcm_disorder/` and `tmm_disorder/`.

---

## Contents

- [`pdw_lcm.py`](#pdw_lcmpy) — Local Chern Marker (real-space, exact diagonalization)
- [`pdw_tmm.py`](#pdw_tmmpy) — Transfer Matrix Method (Lyapunov exponents)
- [`fss_tools.py`](#fss_toolspy) — Finite-Size Scaling (physics-blind, works for both probes)

---

## `pdw_lcm.py`

Computes the disorder-averaged spin Chern marker at the centre of a
finite `L × L` sample, using full y-hopping (not the earlier
insulating-parent shortcut). Default parameters `t=1, t'=1, d0x=3, d0y=1,
mu=-0.5` satisfy all insulating-phase constraints and give an indirect
gap of 1.0.

### `Config` (dataclass)

Holds all model parameters for one LCM run.

| Field | Type | Default | Meaning |
|---|---|---|---|
| `L` | `int` | — | Sample size (use ODD L for a unique centre site) |
| `t` | `float` | `1.0` | Nearest-neighbour hopping (energy unit) |
| `tp` | `float` | `1.0` | Next-nearest (diagonal) hopping; need `tp > 0.5` |
| `d0x` | `float` | `3.0` | Singlet p_x pairing amplitude; need `d0x > 2` |
| `d0y` | `float` | `1.0` | Triplet p_y pairing amplitude; need `d0y != 0` |
| `mu` | `float` | `-0.5` | Chemical potential (gap-centred for the defaults above) |
| `W` | `float` | `0.0` | Disorder strength; `eps_j ~ Uniform[-W, W]` |

Properties: `.N` (= L²), `.nocc()` (half-filling occupation count),
`.centre_index()` (site index of the sample centre).

### `build_hamiltonian_upspin(cfg, eps) -> np.ndarray`

Builds the dense up-spin Hamiltonian (open boundary conditions, full
y-hopping) for one disorder realization.

- **Input:** `cfg` (Config), `eps` (disorder array, length `cfg.N`)
- **Output:** complex `N×N` Hamiltonian matrix
- **Note:** only the up-spin block is ever built — see the
  `C_s = C_up` shortcut explained in the module docstring.

### `occupied_projector(H, nocc) -> (P, gap)`

Diagonalizes `H` and returns the projector onto the lowest `nocc`
states.

- **Input:** `H` (Hamiltonian), `nocc` (number of occupied states)
- **Output:** `P` (projector matrix), `gap` (single-particle gap
  between the highest occupied and lowest unoccupied state)
- Uses `scipy.linalg.eigh` with `subset_by_index` so only the needed
  eigenpairs are computed (not the full spectrum).

### `centre_marker_upspin(cfg, eps, window=1) -> (C_s, gap)`

**Main entry point.** Computes the spin Chern marker at the sample
centre for one disorder realization.

- **Input:** `cfg` (Config), `eps` (disorder array), `window` (block
  size to average over; `1` = single centre site)
- **Output:** `C_s` (marker value, ideally ±1 in the topological
  phase), `gap` (single-particle gap)
- Uses the Bianco–Resta commutator formula, evaluated via an O(N²)
  right-to-left contraction trick (not the O(N³) full-operator form).

### `make_disorder(cfg, rng) -> np.ndarray`

Generates one disorder realization.

- **Input:** `cfg` (Config), `rng` (`np.random.Generator`, e.g. from
  `np.random.default_rng(seed)`)
- **Output:** array of length `cfg.N`, `eps_j ~ Uniform[-W, W]`

### `_selftest()`

Runs a clean-limit (`W=0`) check across several `L` values, printing
the marker and gap. Run directly: `python3 pdw_lcm.py`. Confirms the
marker approaches ±1 as `L` grows (see the finite-size convergence
discussion in the project notes for why small `L` deviates from 1).

---

## `pdw_tmm.py`

Computes the localization length via the transfer-matrix / Lyapunov
exponent method on a quasi-1D strip. Full y-hopping, same default
parameters as `pdw_lcm.py`.

### `Config` (dataclass)

Holds all model parameters for one TMM strip run.

| Field | Type | Default | Meaning |
|---|---|---|---|
| `Ly` | `int` | — | Strip transverse width (= M) |
| `t` | `float` | `1.0` | Nearest-neighbour hopping |
| `tp` | `float` | `1.0` | Next-nearest hopping; need `tp > 0.5` |
| `d0x` | `float` | `3.0` | Singlet p_x pairing; need `d0x > 2` |
| `d0y` | `float` | `1.0` | Triplet p_y pairing; need `d0y != 0` |
| `mu` | `float` | `-0.5` | Chemical potential |

### `build_slice_constants(cfg, sigma, twist) -> dict`

Pre-computes the disorder-independent parts of one strip slice, split
by column-index parity.

- **Input:** `cfg` (Config), `sigma` (±1, spin sector), `twist`
  (transverse boundary twist angle in radians)
- **Output:** `{parity: (Tplus, Hy)}` for `parity in (0, 1)`, where
  `Tplus` is the inter-slice hopping matrix and `Hy` is the
  disorder-independent intra-slice matrix.
- Twisted boundary conditions dodge the singular inter-slice hopping
  resonance that occurs at exact PBC.

### `lyapunov_min(cfg, W, E=0.0, sigma=+1, twist=0.7, seed=0, L=40000, q=8, nseg=20) -> dict`

Pushes the transfer matrix down one strip of length `L` slices,
extracting the smallest positive Lyapunov exponent via the QR method
(Benettin's algorithm).

- **Input:** `cfg`, `W` (disorder strength), `E` (energy, usually 0),
  `sigma` (spin sector), `twist` (boundary angle), `seed` (RNG seed),
  `L` (number of slices — **this is what should scale with `Ny`, see
  `ASPECT_RATIO_TARGET` in `run_point.py`**), `q` (steps between
  QR re-orthonormalizations), `nseg` (number of segments for the
  error-bar estimate)
- **Output:** `{"xi", "xi_err", "gamma_min", "gamma_err"}`
- `xi = 1/gamma_min` is the localization length for this one twist
  angle.

### `localization_length(cfg, W, E=0.0, sigma=+1, twists=(0.7,), seed=0, L=40000, q=8, nseg=20) -> dict`

Averages `lyapunov_min` over multiple twist angles (Mildner-style
twist averaging).

- **Input:** same as `lyapunov_min`, plus `twists` (tuple/list of
  angles to average over)
- **Output:** `{"xi", "xi_err", "Lambda", "M", "xis_per_twist",
  "errs_per_twist"}`
- **Important convention note:** `Lambda` here is `xi / cfg.Ly`
  (NOT Mildner's `(xi/Ly)^-1`). When reproducing Mildner-style figures,
  invert this at the point of use in the analysis script:
  `Lambda_mildner = Ny / xi`.

### `_selftest()`

Runs a small clean sweep over `W` and `M` values, printing `Lambda`.
Run directly: `python3 pdw_tmm.py`.

---

## `fss_tools.py`

General finite-size scaling tools. **Physics-blind** — works for any
order parameter obeying the single-parameter scaling ansatz
`O(W, L) = f[(W - Wc) * L^(1/nu)]`. Used by both the LCM and TMM
analysis scripts.

### `pairwise_crossings(W_arr, cbar_by_L, L_list) -> np.ndarray`

Finds every W where two different-L curves cross, via linear
interpolation between adjacent grid points. Used to get an initial
guess for `Wc` before running the full optimizer.

- **Input:** `W_arr` (1D disorder grid), `cbar_by_L` (dict
  `{L: array of length N_W}`), `L_list` (which L's to compare)
- **Output:** 1D array of crossing W values (may be empty)

### `fit_fss(W_flat, L_flat, cbar_flat, csem_flat, Wc_init, nu_init, poly_degree=4) -> dict`

**Main entry point.** Optimizes `(Wc, nu)` by minimizing the data
collapse chi². Uses Nelder-Mead on a weighted polynomial fit to the
rescaled data `x = (W-Wc)*L^(1/nu)`.

- **Input:** flattened 1D arrays (same length, one entry per (W,L)
  data point), `Wc_init`/`nu_init` (starting guesses), `poly_degree`
  (degree of the master-curve polynomial)
- **Output:** `{"Wc", "nu", "chi2", "success"}`

### `bootstrap_fss(W_flat, L_flat, cbar_flat, csem_flat, Wc_init, nu_init, n_boot=300, poly_degree=4) -> dict`

Parametric bootstrap for error bars on `(Wc, nu)`. Perturbs each data
point by its own statistical error and re-optimizes, `n_boot` times.

- **Input:** same as `fit_fss`, plus `n_boot` (number of bootstrap
  samples — 300 is generally stable)
- **Output:** `{"Wc_err", "nu_err", "Wc_boot", "nu_boot"}` (the last
  two are the full bootstrap distributions, for diagnostics)

### `master_curve(Wc, nu, W_flat, L_flat, cbar_flat, csem_flat, poly_degree=4, x_range=8.0) -> (x_fine, c_fine)`

Evaluates the fitted master-curve polynomial on a fine grid, for
overlaying on top of collapsed data in plots.

- **Input:** optimal `(Wc, nu)` plus the same flattened data arrays
- **Output:** `x_fine` (fine x-grid), `c_fine` (polynomial evaluated
  on that grid, clipped to `[0, 1]`)

### `_collapse_chi2(...)` (internal, not called directly)

The objective function minimized inside `fit_fss`. Computes the
weighted chi² of a polynomial fit to the collapsed data for a given
`(Wc, nu)` — lower is a better collapse.

---

## Quick lookup — "I need to..."

| Task | Function |
|---|---|
| Build one LCM Hamiltonian | `pdw_lcm.build_hamiltonian_upspin` |
| Get the LCM marker for one realization | `pdw_lcm.centre_marker_upspin` |
| Generate disorder | `pdw_lcm.make_disorder` / (TMM: `W` passed directly into `lyapunov_min`) |
| Get the localization length for one (W, Ny) point | `pdw_tmm.localization_length` |
| Get an initial `Wc` guess before fitting | `fss_tools.pairwise_crossings` |
| Extract `(Wc, nu)` from data | `fss_tools.fit_fss` |
| Get error bars on `(Wc, nu)` | `fss_tools.bootstrap_fss` |
| Plot the fitted collapse curve | `fss_tools.master_curve` |
