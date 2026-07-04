#!/bin/bash
# =============================================================================
# submit.slurm
# SLURM array job for the LCM disorder-driven transition sweep.
#
# Array layout:
#   Total tasks = N_W * N_L = 30 * 11 = 330
#   task_id = W_idx * N_L + L_idx
#   W_idx   = task_id / N_L   (integer division)
#   L_idx   = task_id % N_L   (remainder)
#
# N_W=30 and N_L=11 MUST match the grids in run_point.py.
#
# Partition strategy (from sinfo, 2025-07-01):
#   edr1-al9_large : 20 nodes x 192 cores
#   edr2-al9_large : 10 nodes x 192 cores
#   global         : union of ALL 30 edr1+edr2 nodes -- gives scheduler
#                    maximum flexibility to place tasks on any free node
#
#   EXCLUDED -- intel-g4-al9_large : 128-core nodes, cannot satisfy
#               188-CPU request.
#   EXCLUDED -- *_serial partitions : reject multi-CPU requests by design.
#
# Runtime estimate (188 cores, each task independent):
#   L=15 : ~10 s      L=35 : ~5 min
#   16 idle nodes right now -> 330 tasks finish in ~11 rounds -> ~1 hr wall
#   Safety margin: 2 hr per task is more than sufficient.
#
# Submit from AS/lcm_disorder/:
#   sbatch submit.slurm
# =============================================================================

#SBATCH --job-name=lcm_disorder
#SBATCH --account=qmt
#SBATCH --partition=edr1-al9_large,edr2-al9_large,global
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=188                          # NOT 192: race condition on 192-core nodes
#SBATCH --mem=64G                                    # actual footprint ~50 GB; not full node RAM
#SBATCH --time=02:00:00
#SBATCH --array=0-329
#SBATCH --output=logs/lcm_%A_%a.out
#SBATCH --error=logs/lcm_%A_%a.err

# ------------------------------------------------------------------ #
#  Grid sizes -- MUST match run_point.py
# ------------------------------------------------------------------ #
N_L=11

# decompose SLURM_ARRAY_TASK_ID into (W_idx, L_idx)
W_IDX=$(( SLURM_ARRAY_TASK_ID / N_L ))
L_IDX=$(( SLURM_ARRAY_TASK_ID % N_L ))

# ------------------------------------------------------------------ #
#  BLAS thread guard (belt-and-suspenders: also set in run_point.py
#  before numpy import, which is the load-bearing location)
# ------------------------------------------------------------------ #
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

# ------------------------------------------------------------------ #
#  Environment
# ------------------------------------------------------------------ #
cd $SLURM_SUBMIT_DIR
mkdir -p logs data

echo "Task ${SLURM_ARRAY_TASK_ID}: W_idx=${W_IDX}  L_idx=${L_IDX}"
echo "Host: $(hostname)   CPUs allocated: ${SLURM_CPUS_PER_TASK}"
echo "Started: $(date)"

# ------------------------------------------------------------------ #
#  Run
# ------------------------------------------------------------------ #
python run_point.py \
    --W_idx  ${W_IDX} \
    --L_idx  ${L_IDX} \
    --outdir data

echo "Finished: $(date)"
