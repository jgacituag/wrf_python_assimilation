#!/bin/bash
#PBS -N WS_DA
#PBS -l nodes=1:ppn=48
#PBS -o /tmp/ws_da_pbs_dummy.log
#PBS -j oe
#PBS -V

# ---------------------------------------------------------------------------
# queue_ws.sh -- WRF Single-Cycle Assimilation job script
#
# For sweep and single_obs modes ONLY.
# Use queue_multiobs.sh for multi_obs experiments.
#
# One job = one truth member = one node.
# Workers are set automatically from available cores minus 2 for OS/IO.
# OMP_NUM_THREADS is free during setup (H(xf) uses all cores), then
# reset to 1 inside each worker via ctypes after fork.
#
# Submit one truth member:
#   qsub -v CONFIG=configs/ws1.yaml,TM=0 src/queue_ws.sh
#
# Submit all truth members:
#   for tm in $(seq 0 59); do
#       qsub -v CONFIG=configs/ws1.yaml,TM=$tm src/queue_ws.sh
#   done
#
# Optional overrides via -v:
#   CONFIG   path to yaml config     (default: configs/ws1.yaml)
#   TM       truth member index      (required)
#   WORKERS  override worker count   (default: N_CORES - 2)
# ---------------------------------------------------------------------------

REPO=/nfsmounts/storage/scratch/jorge.gacitua/WRF_Single_Cycle_Assimilation
LOG_DIR=$REPO/logs
mkdir -p "$LOG_DIR"
cd "$REPO"

# --- environment ------------------------------------------------------------
source /opt/load-libs.sh 3
source /nfsmounts/storage/scratch/jorge.gacitua/miniconda3/etc/profile.d/conda.sh
conda activate intermediate_exp

# --- build Fortran if needed ------------------------------------------------
bash src/build_fortran.sh
if [ $? -ne 0 ]; then
    echo "ERROR: Fortran build failed on $(hostname)"
    exit 1
fi
echo "Fortran build OK"

# --- parameters -------------------------------------------------------------
CONFIG=${CONFIG:-configs/ws1.yaml}

# Detect available cores from PBS or fallback to nproc
if [ -n "${PBS_NP:-}" ]; then
    N_CORES=$PBS_NP
else
    N_CORES=$(nproc)
fi

# Reserve 2 cores for OS/IO overhead
SAFE_CORES=$(( N_CORES - 2 ))
[ "$SAFE_CORES" -lt 1 ] && SAFE_CORES=1

WORKERS=${WORKERS:-$SAFE_CORES}

# --- threading --------------------------------------------------------------
# OMP is left free so setup (H(xf) over full domain) uses all cores.
# After fork, each worker resets OMP to 1 thread via ctypes (_worker_init).
# MKL/OpenBLAS are forced to 1 to avoid nested thread contention.
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
# OMP_NUM_THREADS intentionally NOT set here — Python runner controls it

# --- logging ----------------------------------------------------------------
# PBS cannot expand variables in #PBS -o, so we redirect manually.
# The dummy PBS log goes to /tmp on the compute node and is discarded.
TM=${TM:-notm}
LOG_FILE="$LOG_DIR/ws_da_tm${TM}_W${WORKERS}_new_2.log"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "[queue] logging to $LOG_FILE"

# --- TM argument ------------------------------------------------------------
TM_ARG=""
if [ -n "${TM:-}" ] && [ "$TM" != "notm" ]; then
    TM_ARG="--tm $TM"
fi

# --- info -------------------------------------------------------------------
echo "[queue] node=$(hostname)  cores_avail=$N_CORES  safe_cores=$SAFE_CORES"
echo "[queue] config=$CONFIG  mode=sweep/single_obs  workers=$WORKERS  tm=${TM}"
echo "[queue] OMP_NUM_THREADS=$(echo ${OMP_NUM_THREADS:-unset})"

# --- run --------------------------------------------------------------------
echo "[queue] starting at $(date)"
t_start=$SECONDS

python -u src/runners/run_experiment.py \
    --config  "$CONFIG"  \
    --workers "$WORKERS" \
    --verbose 1          \
    $TM_ARG

echo "[queue] finished at $(date)  elapsed=$(( SECONDS - t_start ))s"