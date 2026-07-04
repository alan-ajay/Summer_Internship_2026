#!/bin/bash
# =============================================================================
# submit.slurm  (TMM)
# SLURM array job for the TMM disorder-driven transition sweep.
#
# Array layout:
#   Total tasks = N_W * N_NY = 25 * 13 = 325
#   task_id = W_idx * N_NY + Ny_idx
#   W_idx   = task_id / N_NY   (integer division)
#   Ny_idx  = task_id % N_NY   (remainder)
#
# N_W=25 and N_NY=13 MUST match run_point.py exactly.
#
# Parallelism strategy (different from LCM):
#   LCM parallelised over disorder realisations (30k per point).
#   TMM parallelises over twist angles (one per core = 188 twists).
#   Each twist is one independent strip run.
#   With 188 twists, twist-averaged xi is very well converged.
#
# Strip length now SCALES with Ny (L_strip = 5000 * Ny), instead of
# being fixed, so the aspect ratio stays roughly constant across all
# strip widths and the largest strips (Ny=99, 139) get properly
# converged results -- a fixed L_strip previously caused a systematic
# drift in the finite-size critical point Wc(Ny).
#
# Runtime estimate (188 twists in parallel, L_strip = 5000*Ny):
#   Ny=19  (L_strip=95,000)  : ~5    s per task
#   Ny=59  (L_strip=295,000) : ~5    min per task
#   Ny=139 (L_strip=695,000) : ~2.5  hr per task
#   --> 4 hour wall time gives safety margin for the largest strips.
#
# Submit from AS/tmm_disorder/:
#   sbatch submit.slurm
# =============================================================================

#SBATCH --job-name=tmm_disorder
#SBATCH --account=qmt
#SBATCH --partition=edr1-al9_large,edr2-al9_large,global
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=188                          # NOT 192 (race condition)
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --array=0-324
#SBATCH --output=logs/tmm_%A_%a.out
#SBATCH --error=logs/tmm_%A_%a.err

# ------------------------------------------------------------------ #
#  Grid sizes -- MUST match run_point.py
# ------------------------------------------------------------------ #
N_NY=13

# Decompose SLURM_ARRAY_TASK_ID into (W_idx, Ny_idx)
W_IDX=$(( SLURM_ARRAY_TASK_ID / N_NY ))
NY_IDX=$(( SLURM_ARRAY_TASK_ID % N_NY ))

# ------------------------------------------------------------------ #
#  BLAS thread guard (belt-and-suspenders; load-bearing location
#  is before import numpy in run_point.py)
# ------------------------------------------------------------------ #
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

# ------------------------------------------------------------------ #
#  Environment
# ------------------------------------------------------------------ #
cd $SLURM_SUBMIT_DIR
mkdir -p logs data

echo "Task ${SLURM_ARRAY_TASK_ID}: W_idx=${W_IDX}  Ny_idx=${NY_IDX}"
echo "Host: $(hostname)   CPUs: ${SLURM_CPUS_PER_TASK}"
echo "Started: $(date)"

# ------------------------------------------------------------------ #
#  Run
# ------------------------------------------------------------------ #
python3 run_point.py \
    --W_idx  ${W_IDX} \
    --Ny_idx ${NY_IDX} \
    --outdir data

echo "Finished: $(date)"
