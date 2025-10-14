#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <single|2d> <CONFIG.yaml>"
  exit 1
fi

MODE="$1"
CONFIG="$2"

mkdir -p logs
STAMP=$(date +"%Y%m%d_%H%M%S")
LOGFILE="logs/run_${MODE}_$(basename "$CONFIG" .yaml)_${STAMP}.log"

echo "=== wrf_python_assimilation runner ===" | tee "$LOGFILE"
echo "Mode   : $MODE" | tee -a "$LOGFILE"
echo "Config : $CONFIG" | tee -a "$LOGFILE"
echo "Date   : $(date -Iseconds)" | tee -a "$LOGFILE"
echo "Log    : $LOGFILE" | tee -a "$LOGFILE"
echo "--------------------------------------" | tee -a "$LOGFILE"

case "$MODE" in
  single)
    python -u runners/run_single_obs.py --config "$CONFIG" 2>&1 | tee -a "$LOGFILE"
    ;;
  2d)
    python -u runners/run_full2d_multicycle.py --config "$CONFIG" 2>&1 | tee -a "$LOGFILE"
    ;;
  2d_full)
    python -u runners/run_full2d_multicycle_exps.py --config "$CONFIG" 2>&1 | tee -a "$LOGFILE"
    ;;
  *)
    echo "[ERROR] Unknown mode: $MODE" | tee -a "$LOGFILE"
    exit 2
    ;;
esac

echo "Done." | tee -a "$LOGFILE"
