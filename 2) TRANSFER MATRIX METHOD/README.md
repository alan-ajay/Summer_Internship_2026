# tmm_disorder Folder — Function & Workflow Reference

This file is a lookup directory for `AS/tmm_disorder/`. Use it to
quickly find what each script does, what parameters it uses, and how
the pieces connect — without re-reading the full source each time.

This folder is the **runner + analysis layer** for the TMM
disorder-driven transition sweep (reproduces Mildner et al.
Supplementary Section H / Fig. S-36–S-38). All physics (slice
construction, Lyapunov exponent) lives in `AS/physics/pdw_tmm.py` —
nothing here computes physics directly, it only calls into that file.

---

## Contents

```
tmm_disorder/
├── run_point.py    -- cluster runner: computes ONE (W, Ny) point
├── submit.slurm     -- SLURM array job: launches all (W, Ny) points
├── analyse.py       -- collects results, extracts (Wc, nu), plots 4-panel figure
├── data/             -- one .npz file per (W, Ny) point (created by run_point.py)
└── logs/             -- one .out/.err pair per SLURM array task
```

---

## `run_point.py`

Runs **one (W, Ny) point** of the sweep: `N_TWISTS` twist angles, each
one complete strip run, in parallel across all cores allocated by
SLURM.

### Why this parallelizes over TWISTS, not realizations (unlike LCM)

The TMM strip is self-averaging — one long strip gives the
disorder-averaged Lyapunov exponent directly (Oseledets' theorem).
There's no ensemble of independent realizations to parallelize over.
Instead, each twist angle sees a different transverse momentum grid
and gives an independent estimate of `xi`; averaging over `N_TWISTS`
twists reduces the systematic error from any single resonant
momentum.

### Parameter grids (must match `submit.slurm` exactly)

| Name | Value | Meaning |
|---|---|---|
| `W_LIST` | `np.linspace(1.5, 7.0, 25)` | 25 disorder strengths |
| `NY_LIST` | `[19, 23, 27, ..., 59, 99, 139]` | 13 strip widths (matches Mildner Sec. H) |
| `N_JOBS` | reads `SLURM_CPUS_PER_TASK`, falls back to `8` | One worker per twist angle |
| `N_TWISTS` | `= N_JOBS` | Number of twist angles averaged (188 in production) |

### Strip length — scales with `Ny`, not fixed

```python
ASPECT_RATIO_TARGET = 5000     # L_strip / Ny target
L_strip = ASPECT_RATIO_TARGET * Ny   # computed per-task in main()
```

**Why:** a single fixed `L_strip` under-converges the larger strips —
the smallest Lyapunov exponent gets harder to resolve as `Ny` grows
(more exponents packed into the same energy range), so a longer strip
is needed at large `Ny`. This was the fix for a systematic drift
previously seen in `Wc(Ny)`. Mildner used `N_x=10^7` at `Ny=139`
(aspect ratio ~72,000); we target a more modest but still convergent
5,000.

`L_strip` is passed **explicitly** into every worker call — not read
from a module-level global — because it depends on `Ny`, which is
only known inside `main()`. See the docstring of `_run_one_twist` for
the scoping reasoning.

### Model parameters (must match `pdw_tmm.py` defaults)

`TP=1.0, D0X=3.0, D0Y=1.0, MU=-0.5` — satisfies all insulating-phase
constraints, indirect gap = 1.0. Same as `lcm_disorder/run_point.py`.

### `_run_one_twist(W, Ny, twist, seed, L_strip) -> dict`

The joblib worker function. Builds one `Config`, calls
`pdw_tmm.lyapunov_min` for a single twist angle.

- **Input:** `W` (disorder), `Ny` (strip width), `twist` (boundary
  angle), `seed` (per-twist RNG seed), `L_strip` (strip length for
  this task — see above)
- **Output:** `{"xi", "xi_err", "gamma_min", "gamma_err"}` (the
  `lyapunov_min` result dict for this one twist)

### `main()`

1. Parses `--W_idx`, `--Ny_idx`, `--outdir` from the command line
2. Maps indices → physical `(W, Ny)` via the grids above
3. Computes `L_strip = ASPECT_RATIO_TARGET * Ny` for this task
4. Builds `N_TWISTS` twist angles uniformly spread over
   `(0.05π, 1.95π)` (avoids exact `0`/`2π` to dodge any accidentally
   resonant momentum)
5. Builds deterministic per-twist seeds:
   `W_idx * N_NY * N_TWISTS + Ny_idx * N_TWISTS + k`
6. Runs all `N_TWISTS` twists in parallel with `joblib.Parallel`
7. Averages `xi` over twists (combining within-twist convergence
   error and between-twist variance in quadrature)
8. Saves one `.npz` file: `data/tmm_W{W:.4f}_Ny{Ny:03d}.npz`,
   containing per-twist raw arrays (`xis`, `xi_errs`, `gammas`,
   `twists`) as well as the twist-averaged `xi`, `xi_err`, `Lambda`

### ⚠️ `Lambda` convention — read before using saved data

`run_point.py` saves `Lambda = xi / Ny`. **This is NOT Mildner's
convention.** Mildner's `Lambda = (xi/Ny)^{-1} = Ny/xi` (has a
**minimum** at the critical point). Our saved field has a **maximum**
instead. `analyse.py` corrects this by recomputing
`Lambda = Ny / xi` from the raw `xi` field at load time — the raw
`xi` was always saved correctly, so no cluster re-run is needed if
you hit this. If you write any new analysis script against this data,
recompute `Lambda` the same way; don't use the saved `Lambda` field
directly.

### Run manually (outside SLURM, for testing)

```bash
python3 run_point.py --W_idx 0 --Ny_idx 0 --outdir data
```

---

## `submit.slurm`

SLURM array job that launches `run_point.py` once per
`(W_idx, Ny_idx)` pair — 325 tasks total (`25 × 13`).

### Task decomposition

```bash
N_NY=13
W_IDX=$(( SLURM_ARRAY_TASK_ID / N_NY ))
NY_IDX=$(( SLURM_ARRAY_TASK_ID % N_NY ))
```

This is the **only** coupling between `submit.slurm` and
`run_point.py` — both files must agree on `N_NY`.

### Key settings

| Setting | Value | Why |
|---|---|---|
| `--partition` | `edr1-al9_large,edr2-al9_large` | **NOT** `global` — `global` has `MaxTime=03:00:00`, shorter than our `--time`; `sbatch` rejects the whole submission if ANY listed partition can't satisfy the requested time, it doesn't just silently drop that partition |
| `--cpus-per-task` | `188` | Not 192 — avoids the node-race condition |
| `--mem` | `64G` | Actual footprint, not full node RAM |
| `--time` | `04:00:00` | Sized for the worst case: `Ny=139` (`L_strip=695,000`) takes ~2.5 hr |
| `--array` | `0-324` | 325 tasks = 25 W × 13 Ny |
| BLAS guard | set both here and (load-bearing) before `import numpy` in `run_point.py` | Prevents thread oversubscription across 188 joblib workers |

### Runtime by strip width (approximate, 188 twists in parallel)

| `Ny` | `L_strip` | Est. wall time |
|---|---|---|
| 19 | 95,000 | ~5 s |
| 59 | 295,000 | ~5 min |
| 139 | 695,000 | ~2.5 hr |

### Submit

```bash
sbatch submit.slurm
squeue -u alanajay -o "%i %j %T %R"     # confirm it's running
```

---

## `analyse.py`

Collects every `.npz` file in `data/`, extracts `(Wc, nu)` via the
curvature-scaling method, and produces a single 4-panel figure
matching Mildner's Supplementary Fig. S-39 layout.

### `load_data(datadir) -> (data, W_arr, Ny_list)`

Scans `datadir` for all `tmm_W*_Ny*.npz` files.

- **Output:** `data` — dict keyed by `(W, Ny)` → `Lambda` (computed
  as `Ny / xi`, the corrected Mildner convention — see the warning
  above); `W_arr` — sorted array of W values; `Ny_list` — sorted list
  of strip widths

### `find_minimum(W_arr, Lambda_arr, n_fit=7) -> (Wc_Ny, Lambda_min)`

Locates the minimum of one `Lambda(W)` curve by fitting a parabola to
the `n_fit` points nearest the raw minimum and solving for the exact
vertex — smoother than reading off the lowest grid point directly.

- **Input:** `W_arr`/`Lambda_arr` (one strip's data), `n_fit` (points
  used in the local parabolic fit)
- **Output:** `(W` at the minimum, `Lambda` value there`)`

### `curvature_at_Wc(W_arr, Lambda_arr, Wc, n_fit=7) -> float`

Fits a parabola centred at the thermodynamic `Wc` and returns the
second derivative `2*a2` there.

- **Why:** FSS theory predicts this curvature scales as `Ny^(2/nu)`
  (Mildner Eq. D.6) — this is the quantity whose log-log slope gives
  `nu` directly.

### `main()`

1. Loads data (`load_data`)
2. Finds each strip's finite-size minimum `Wc(Ny)` via `find_minimum`
3. Sets the thermodynamic `Wc` = average of all finite-size minima
   (valid because the disorder-driven transition shows negligible
   drift — see panel (b) below)
4. Computes the curvature at `Wc` for each `Ny` via `curvature_at_Wc`,
   fits a straight line in log-log space (`scipy.stats.linregress`),
   extracts `nu = 2/slope`
5. Builds the single 2×2 figure (see panels below) and saves both
   `.pdf` and `.png`

### The four panels (matches Mildner Fig. S-39 layout)

| Panel | Content | What to look for |
|---|---|---|
| (a) | `Lambda` vs `W` for each `Ny` | Minimum at `Wc`, zoomed to `Wc ± 1.2` |
| (b) | `Wc(Ny)` vs `Ny` | Should be **flat** — negligible drift confirms disorder-driven transition |
| (c) | log-log curvature vs `Ny` | Straight-line fit slope = `2/nu` |
| (d) | Scaling collapse: `Lambda_tilde` vs `w·Ny^(1/nu)`, where `w=(W-Wc)/Wc` | All curves should overlap if `nu` is correct |

### Run

```bash
python3 analyse.py
python3 analyse.py --datadir data --outdir figures
```

---

## Cross-check against LCM

Both probes target the **same** disorder-driven transition and should
agree on `Wc` and `nu` within error bars. Compare:

```bash
# LCM result
cat lcm_disorder/figures/... # or read the printed Wc/nu from analyse.py output

# TMM result
cat tmm_disorder/figures/... # or read the printed Wc/nu from analyse.py output
```

If the two disagree significantly, suspect either LCM finite-size
convergence (see `AS/physics/README.md` note on small-`L` deviation
from `c̄=1`) or TMM strip-length convergence (see `ASPECT_RATIO_TARGET`
above) before suspecting a physics discrepancy.

---

## End-to-end workflow

```bash
cd AS/tmm_disorder/

# 1. Submit the sweep (325 independent tasks, one node each)
sbatch submit.slurm

# 2. Wait, check progress
squeue -u alanajay -o "%i %j %T %R"
ls data/ | wc -l          # should reach 325 when done

# 3. Analyse once all (or most) files exist
python3 analyse.py
```

---

## Quick lookup — "I need to..."

| Task | Where |
|---|---|
| Change the W or Ny grid | `W_LIST` / `NY_LIST` in `run_point.py` (must also update `N_NY` in `submit.slurm`) |
| Change strip length / convergence | `ASPECT_RATIO_TARGET` in `run_point.py` |
| Change model parameters (t', d0x, d0y, mu) | `TP`/`D0X`/`D0Y`/`MU` in `run_point.py` (must match `pdw_tmm.py`) |
| Fix a "curve has a maximum not minimum" bug | Check you're using `Ny/xi`, not the saved `Lambda` field directly |
| Find where one (W,Ny) point's raw data lives | `data/tmm_W{W:.4f}_Ny{Ny:03d}.npz` |
| Check a specific SLURM task's log | `logs/tmm_{jobid}_{taskid}.out` / `.err` |
