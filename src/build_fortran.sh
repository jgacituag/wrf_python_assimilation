#!/usr/bin/env bash
# scripts/build_fortran.sh
# Build the cletkf_wloc Fortran module with f2py + OpenMP.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
SRC_DIR="$ROOT/src/fortran"

cd "$SRC_DIR"
rm -f *.mod *.so
# Ensure sources exist
for f in netlib.f90 SFMT.f90 common_tools.f90 common_mtx.f90 common_letkf.f90 common_da_wloc.f90; do
  if [[ ! -f $f ]]; then
    echo "[ERROR] Missing $f in $SRC_DIR"
    exit 1
  fi
done

F2PY=f2py
MODNAME="cletkf_wloc"
FFLAGS='-O3'
#export FC=gfortran
#export F90=gfortran
echo "[build] Sources: SFMT.f90 common_tools.f90 common_mtx.f90 common_letkf.f90 common_da_wloc.f90"
#$F2PY -c -lgomp --opt="-fopenmp -lgomp" netlib.f90 SFMT.f90 common_tools.f90 common_mtx.f90 common_letkf.f90 common_da_wloc.f90 -m cletkf_wloc
#$F2PY -c -lgomp --opt="-fopenmp -lgomp" netlib.f90 SFMT.f90 common_tools.f90 common_mtx.f90 common_letkf.f90 common_da.f90 -m cletkf > compile_cletkf.out 2>&1
$F2PY -c -lgomp --opt="-fopenmp -lgomp" netlib.f90 SFMT.f90 common_tools.f90 common_mtx.f90 common_letkf.f90 common_da_wloc.f90 -m cletkf_wloc > compile_cletkf_wloc.out 2>&1

SOFILE=$(ls ${MODNAME}*.so | head -n1 || true)
if [[ -n "${SOFILE}" ]]; then
  echo "[build] Installed: $SRC_DIR/${SOFILE}"
else
  echo "[warn] Build finished but .so not found."
fi
