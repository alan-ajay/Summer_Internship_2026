# lcm_disorder Folder — Function & Workflow Reference

This file is a lookup directory for `AS/lcm_disorder/`. Use it to
quickly find what each script does, what parameters it uses, and how
the pieces connect — without re-reading the full source each time.

This folder is the **runner + analysis layer** for the LCM
disorder-driven transition sweep (reproduces Mildner et al. Fig. 4).
All physics (Hamiltonian, marker formula) lives in
`AS/physics/pdw_lcm.py` — nothing here computes physics directly,
it only calls into that file.

---

## Contents

```
lcm_disorder/
├── run_point.py    -- cluster runner: computes ONE (W, L) point
├── submit.slurm     -- SLURM array job: launches all (W, L) points
├── analyse.py       -- collects results, extracts (Wc, nu), plots Fig 4
├── data/             -- one .npz file per (W, L) point (created by run_point.py)
└── logs/             -- one .out/.err pair per SLURM array task
```

---

## `run_point.py`

Runs **one (W, L) point** of the sweep: `N_REAL` disorder realizations,
in parallel across all cores allocated by SLURM.

### Parameter grids (must match `submit.slurm` exactly)

| Name | Value | Meaning |
|---|---|---|
| `W_LIST` | `np.linspace(0.5, 7.0, 30)` | 30 disorder strengths |
| `L_LIST` | `[15, 17, ..., 35]` | 11 system sizes (all odd, matches Mildner) |
| `N_REAL` | `30_000` | Realizations per point (matches Mildner's `3×10⁴`) |
| `N_JOBS` | reads `SLURM_CPUS_PER_TASK`, falls back to `8` | One worker per realization |

### Model parameters (must match `pdw_lcm.py` defaults)

`TP=1.0, D0X=3.0, D0Y=1.0, MU=-0.5` — satisfies all insulating-phase
constraints, indirect gap = 1.0.

### `_run_one_realization(W, L, child_seed) -> (C_s, gap)`

The joblib worker function. Builds one `Config`, generates one
disorder array from `child_seed`, calls
`pdw_lcm.centre_marker_upspin`, returns the marker and gap for that
single realization.

- **Input:** `W` (disorder strength), `L` (system size), `child_seed`
  (a `np.random.SeedSequence`, spawned per-realization — see Seeding
  below)
- **Output:** `(C_s, gap)` tuple

### `main()`

1. Parses `--W_idx`, `--L_idx`, `--outdir` from the command line
2. Maps indices → physical `(W, L)` via the grids above
3. Builds a deterministic root seed: `root_seed = W_idx * N_L + L_idx`
4. Spawns `N_REAL` child seeds from that root via
   `np.random.SeedSequence.spawn()`
5. Runs all `N_REAL` realizations in parallel with
   `joblib.Parallel`
6. Saves one `.npz` file: `data/lcm_W{W:.4f}_L{L:03d}.npz`, containing
   the full raw arrays `cs` and `gaps` (not just the mean — so any
   statistic can be recomputed later without re-running the cluster job)

### Seeding — why it's deterministic

Root seed = `W_idx * N_L + L_idx` (a unique integer per grid point).
Every realization within that point gets its own child `SeedSequence`.
This means: (a) the same `(W, L)` point always produces the exact
same set of realizations regardless of how many workers ran it or in
what order, and (b) no two `(W, L)` points ever accidentally share a
random stream.

### Run manually (outside SLURM, for testing)

```bash
python3 run_point.py --W_idx 0 --L_idx 0 --outdir data
```

---

## `submit.slurm`

SLURM array job that launches `run_point.py` once per `(W_idx, L_idx)`
pair — 330 tasks total (`30 × 11`).

### Task decomposition

```bash
N_L=11
W_IDX=$(( SLURM_ARRAY_TASK_ID / N_L ))
L_IDX=$(( SLURM_ARRAY_TASK_ID % N_L ))
```

This is the **only** coupling between `submit.slurm` and
`run_point.py` — both files must agree on `N_L`.

### Key settings

| Setting | Value | Why |
|---|---|---|
| `--partition` | `edr1-al9_large,edr2-al9_large,global` | All three 192-core partitions |
| `--cpus-per-task` | `188` | Not 192 — avoids the node-race condition |
| `--mem` | `64G` | Actual footprint, not full node RAM |
| `--array` | `0-329` | 330 tasks = 30 W × 11 L |
| BLAS guard | set both here and (load-bearing) before `import numpy` in `run_point.py` | Prevents thread oversubscription across 188 joblib workers |

### Submit

```bash
sbatch submit.slurm
squeue -u alanajay -o "%i %j %T %R"     # confirm it's running
```

---

## `analyse.py`

Collects every `.npz` file in `data/`, extracts `(Wc, nu)` via finite-
size scaling, and produces the two-panel figure analogous to Mildner
Fig. 4. All FSS math lives in `AS/physics/fss_tools.py` — this script
only loads data, calls the FSS functions, and plots.

### `load_data(datadir) -> (data, W_arr, L_list)`

Scans `datadir` for all `lcm_W*_L*.npz` files.

- **Output:** `data` — dict keyed by `(W, L)`, each entry
  `{'cs', 'gaps', 'cbar', 'csem'}` (`cbar` = mean marker, `csem` =
  standard error of the mean); `W_arr` — sorted array of W values;
  `L_list` — sorted list of L values

### `build_flat_arrays(data, W_arr, L_list) -> (W_flat, L_flat, cbar_flat, csem_flat)`

Flattens the `(W, L)` grid into 1D arrays, ready to hand to the FSS
fitting functions in `fss_tools.py`. Skips any missing `(W, L)`
combination gracefully (so a partially-finished cluster run can still
be analyzed).

### `make_colors(L_list) -> dict`

One color per `L` value, using the `plasma` colormap (blue → yellow
for increasing `L`, matching Mildner's figure style).

### `plot_panel_a(ax, data, W_arr, L_list, colors, Wc=None)`

Plots $\bar{c}$ vs $W$, one curve per `L`. Draws a vertical line at
`Wc` if given, plus the "Increasing L" arrow annotation.

### `plot_panel_b(ax, data, W_arr, L_list, colors, Wc, nu, W_flat, L_flat, cbar_flat, csem_flat, fig)`

Plots the collapsed data $\bar{c}$ vs $(W-W_c)L^{1/\nu}$, overlays
the fitted master curve (dashed gray), and adds a zoomed inset around
the critical region $x \in [-1, 1]$.

### `main()`

1. Loads data (`load_data`)
2. Builds `L_list_fit` = `L_list` minus any `--exclude_L` values
   (excluded sizes still appear in panel (a), but are dropped from
   the fit and from panel (b) — see below)
3. Gets an initial `Wc` guess from `fss_tools.pairwise_crossings`
4. Optimizes `(Wc, nu)` via `fss_tools.fit_fss`
5. Gets error bars via `fss_tools.bootstrap_fss`
6. Builds the two-panel figure and saves both `.pdf` and `.png`

### Command-line flags

| Flag | Default | Meaning |
|---|---|---|
| `--datadir` | `data` | Where to read `.npz` files from |
| `--outdir` | `figures` | Where to save the figure |
| `--n_boot` | `300` | Bootstrap samples for error bars |
| `--Wc_init` | auto (from crossings) | Override the initial `Wc` guess |
| `--nu_init` | `2.5` | Initial `nu` guess (Mildner's value) |
| `--exclude_L` | `[]` (none) | `L` values to drop from the fit only, e.g. `--exclude_L 15 17` |

### Run

```bash
python3 analyse.py
python3 analyse.py --exclude_L 15 17
```

---

## End-to-end workflow

```bash
cd AS/lcm_disorder/

# 1. Submit the sweep (330 independent tasks, one node each)
sbatch submit.slurm

# 2. Wait, check progress
squeue -u alanajay -o "%i %j %T %R"
ls data/ | wc -l          # should reach 330 when done

# 3. Analyse once all (or most) files exist
python3 analyse.py --exclude_L 15 17
```

---

## Quick lookup — "I need to..."

| Task | Where |
|---|---|
| Change the W or L grid | `W_LIST` / `L_LIST` in `run_point.py` (must also update `N_L` in `submit.slurm`) |
| Change number of realizations | `N_REAL` in `run_point.py` |
| Change model parameters (t', d0x, d0y, mu) | `TP`/`D0X`/`D0Y`/`MU` in `run_point.py` (must match `pdw_lcm.py`) |
| Drop noisy small-L points from the fit | `--exclude_L` flag on `analyse.py` |
| Change bootstrap sample count | `--n_boot` flag on `analyse.py` |
| Find where one (W,L) point's raw data lives | `data/lcm_W{W:.4f}_L{L:03d}.npz` |
| Check a specific SLURM task's log | `logs/lcm_{jobid}_{taskid}.out` / `.err` |
