#!/bin/bash
#PBS -N WS_MULTIOBS
#PBS -l nodes=1:ppn=120
#PBS -o /tmp/ws_multiobs_pbs_dummy.log
#PBS -j oe
#PBS -V

# ---------------------------------------------------------------------------
# queue_multiobs.sh -- WRF Single-Cycle multi-obs assimilation job script
#
# For multi_obs mode ONLY.
# Use queue_ws.sh for sweep and single_obs experiments.
#
# Single Python process — Fortran uses all available cores via OpenMP.
# OMP_NUM_THREADS set to N_CORES - 2 to leave headroom for OS.
#
# Submit:
#   qsub -v CONFIG=configs/ws_multiobs_test.yaml,TM=0 src/queue_multiobs.sh
#
# Optional overrides via -v:
#   CONFIG   path to yaml config   (default: configs/ws_multiobs_test.yaml)
#   TM       truth member index    (required)
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
CONFIG=${CONFIG:-configs/ws_multiobs_test.yaml}

if [ -n "${PBS_NP:-}" ]; then
    N_CORES=$PBS_NP
else
    N_CORES=$(nproc)
fi

SAFE_CORES=$(( N_CORES - 2 ))
[ "$SAFE_CORES" -lt 1 ] && SAFE_CORES=1

# --- threading --------------------------------------------------------------
# Single process — Fortran uses all safe cores via OpenMP.
# MKL/OpenBLAS kept at 1 to avoid conflict with OpenMP.
export OMP_NUM_THREADS=$SAFE_CORES
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

# --- logging ----------------------------------------------------------------
TM=${TM:-notm}
LOG_FILE="$LOG_DIR/ws_multiobs_tm${TM}_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "[queue] logging to $LOG_FILE"

# --- TM argument ------------------------------------------------------------
TM_ARG=""
if [ -n "${TM:-}" ] && [ "$TM" != "notm" ]; then
    TM_ARG="--tm $TM"
fi

# --- info -------------------------------------------------------------------
echo "[queue] node=$(hostname)  cores_avail=$N_CORES  OMP_NUM_THREADS=$OMP_NUM_THREADS"
echo "[queue] config=$CONFIG  mode=multi_obs  tm=${TM}"

# --- run --------------------------------------------------------------------
echo "[queue] starting at $(date)"
t_start=$SECONDS

python -u src/runners/run_experiment.py \
    --config  "$CONFIG"  \
    --workers 1          \
    --verbose 2          \
    $TM_ARG

echo "[queue] finished at $(date)  elapsed=$(( SECONDS - t_start ))s"