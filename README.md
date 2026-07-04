# Disorder-Driven Topological Phase Transitions in a 2D PDW Model

This repository contains the numerical pipelines used to characterize the
disorder-driven topological phase transition in the 2D
(p<sub>x</sub> + iσp<sub>y</sub>) topological density-wave (PDW) model
(Hsu, 2012), benchmarked against the disordered Haldane model results of
Mildner *et al.* (arXiv:2312.16689).

The critical disorder strength **W_c** and correlation-length exponent **ν**
are extracted using **two independent numerical probes**:

1. A **local (spin) Chern marker** (LCM) pipeline, based on Bianco–Resta
   (Phys. Rev. B 84, 241106(R), 2011).
2. A **transfer matrix method** (TMM) localization-length pipeline, based on
   the generalized transfer matrix construction for tight-binding models.

The two probes are cross-validated against each other and against the
disordered Haldane model benchmark.

---

## Repository structure

```
.
├── physics/          # Shared physics cores used by both pipelines
├── lcm_disorder/      # Disorder-driven LCM sweep (Mildner Fig. 4 analog)
└── tmm_disorder/      # Disorder-driven TMM sweep (Mildner Supp. Sec. H analog)
```

---

## `physics/`

Shared, model-level code imported by both `lcm_disorder/` and `tmm_disorder/`.
Keeping the physics core in one place ensures both probes act on the
*same* Hamiltonian and disorder realizations, which is what makes the
cross-validation between LCM and TMM meaningful.

| File | Purpose |
|---|---|
| `pdw_lcm.py` | Real-space PDW tight-binding Hamiltonian construction and local Chern marker (Bianco–Resta) evaluation. |
| `pdw_tmm.py` | PDW Hamiltonian in transfer-matrix (strip) form and Lyapunov exponent extraction for the localization length. |
| `fss_tools.py` | Shared finite-size scaling (FSS) utilities: single-parameter scaling collapse, crossing-point extraction via linear interpolation, and χ² optimization for (W_c, ν). |

**Model parameters (validated defaults):**

| Parameter | Value | Note |
|---|---|---|
| t = t' | 1.0 | nearest-neighbor hoppings |
| Δ₀ₓ | 3.0 | singlet PDW order parameter |
| Δ₀y | 1.0 | triplet PDW order parameter (required for nonzero spin Chern number) |
| μ | −0.5 | chemical potential |
| Indirect gap | 1.0 | fixed by the above choice |
| `ty_uniform` | `False` | mandatory for the insulating parent used in the disorder-driven transition |

Insulating-phase conditions (derived analytically, enforced when choosing
parameters):

1. t' > t/2
2. Δ₀ₓ > 2t
3. 4t' + Δ₀ₓ > 6t

---

## `lcm_disorder/`

Disorder-driven transition study using the **local spin Chern marker**.
Replicates the structure of Mildner *et al.*, Fig. 4.

**Method:** For a range of system sizes L and disorder strengths W, the
local Chern marker is averaged over disorder realizations and twisted
boundary conditions, then finite-size scaling is used to locate the
critical point and extract ν via a data collapse.

**Production results:**

- W_c = 3.91 ± 0.06
- ν = 3.49 ± 0.10
- Good finite-size scaling collapse across L ∈ {15, 19, 23, 27, 31, 35, 39, 43}

This ν is anomalously large relative to both the disordered Haldane
benchmark (ν ≈ 2.42 ± 0.11) and the quantum-Hall/Chalker–Coddington
consensus value (ν ≈ 2.6); resolving this discrepancy is an open question
tracked separately.

**Contents typically include:**
- Per-(L, W) run scripts (cluster submission + single-task driver)
- Raw output `.npz` files, one per run, tagged with full parameters
  (L, W, seed, ntwist) — never overwritten
- Analysis/plotting scripts that perform the FSS collapse and save
  publication-ready PDFs with embedded parameter/runtime panels

---

## `tmm_disorder/`

Disorder-driven transition study using the **transfer matrix method**,
extracting the localization length from the Lyapunov exponent spectrum of
long strips.

**Method:** For a range of strip widths M and disorder strengths W, strips
of length L are propagated via the transfer matrix; the relevant bulk
localization length is obtained from **λ₁** (the second-smallest positive
Lyapunov exponent — not λ₀, which corresponds to the chiral edge mode).
Cross-twist averaging and within-twist QR convergence are both required
for a clean estimate.

**Production results:**

- W_c = 3.74 ± 0.10 (consistent with the LCM estimate)
- ν: currently unreliable for the M = 128 tier due to strip-length
  under-convergence; being re-extracted with `ntwist` tripled to 48 for
  the affected tiers.

**Contents typically include:**
- Per-(M, W) run scripts (cluster submission + single-task driver)
- Raw output `.npz` files, one per run, tagged with full parameters
  (M, W, ntwist, seed) — never overwritten
- Convergence diagnostics separating cross-twist relative error
  (`Lambda_err/Lambda`) from within-twist QR error (`gamma_min_err_frac`)
- Analysis/plotting scripts for the FSS collapse and W_c/ν extraction

---

## Requirements

- Python 3 (`python3` — required explicitly on cluster compute nodes)
- `numpy`, `scipy`, `joblib`, `matplotlib`

On multi-core cluster nodes, set BLAS thread limits **before** importing
`numpy` inside each script:

```python
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import numpy as np
```

---

## Output conventions

- Every run's output is saved as a **separate file**, never overwritten.
- Filenames encode the full parameter set (system size, W or M, seed,
  ntwist, etc.) to prevent stale-data contamination across parameter
  sweeps.
- Every `.npz` output includes the per-task wall-clock runtime.
- Every analysis plot embeds mean runtime, key model parameters, and the
  terminal-style FSS diagnostic text as a monospace panel directly in the
  saved PDF.

---

## References

- Hsu, notes on the Topological PDW model (2012).
- Mildner *et al.*, "Topological Phase Transitions in the Disordered
  Haldane Model," arXiv:2312.16689.
- Bianco, R. & Resta, R., "Mapping topological order in coordinate space,"
  Phys. Rev. B 84, 241106(R) (2011).
- Generalized transfer matrices for tight-binding models (TMM construction
  reference).
- MacKinnon, A. & Kramer, B., "One-parameter scaling of localization
  length and conductance in disordered systems," Phys. Rev. Lett. 47, 1546
  (1981).
