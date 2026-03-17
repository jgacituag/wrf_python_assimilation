# WRF Single-Cycle Assimilation

Radar data assimilation experiments with a real cases WRF ensemble,
using the Local Ensemble Transform Kalman Filter (LETKF) and likelihood
tempering (TEnKF).
---

## Repository layout

```
.
├── src/
│   ├── da/
│   │   └── core.py                  # All DA methods (LETKF, TEnKF, AOEI, ATEnKF, TAOEI)
│   ├── runners/
│   │   └── run_experiment.py        # Unified runner for all WS experiments
│   ├── extract_3d_subset.py         # Extract WRF ensemble subsets to .npz
│   └── fortran/                     # Fortran LETKF source + Makefile
├── configs/
│   ├── template.yaml                # Full reference template — start here
│   └── build_3D_section.yaml        # Data extraction config (Notebook 1)
├── Notebooks/
│   ├── S1_Explore_and_extract_3d_sections_WRF.ipynb
│   └── S2_obs_explorer_ws2.ipynb
└── tests/
    └── test_da_core.py
```

---

## Setup

### 1. Create the environment

```bash
conda env create -f environment.yml
conda activate wrf_python_assimilation
```

### 2. Build the Fortran LETKF module

```bash
cd src/fortran && bash ../build_fortran.sh && cd ../..
```

This compiles `cletkf_wloc` via `f2py` and places the `.so` in `src/fortran/`.
All runners add that path to `sys.path` automatically.

---

## Data preparation

Before running any experiment you need to extract the 3D WRF ensemble subset
from the raw `wrfout` files.

**Interactive** — open `Notebooks/1__Explore_and_extract_3d_sections_WRF.ipynb`
and follow the four steps: choose region → visualise → extract → sanity check.

**Command line** — once `configs/build_3D_section.yaml` is configured:

```bash
python src/extract_3d_subset.py --config configs/build_3D_section.yaml
```

The output is a compressed `.npz` file with the following arrays:

| Key | Shape | Description |
|-----|-------|-------------|
| `state_ensemble` | `(nx, ny, nz, Ne, 8)` | All members |
| `lats` | `(ny, nx)` | Latitude [°] |
| `lons` | `(ny, nx)` | Longitude [°] |
| `z_heights` | `(nz, ny, nx)` | Height above sea level [m] |

Variable order in the last axis of `state_ensemble`:

| Index | Variable | Units |
|-------|----------|-------|
| 0 | QGRAUP | kg/kg |
| 1 | QRAIN  | kg/kg |
| 2 | QSNOW  | kg/kg |
| 3 | T (temperature) | K |
| 4 | P (pressure) | Pa |
| 5 | UA (u-wind) | m/s |
| 6 | VA (v-wind) | m/s |
| 7 | WA (w-wind) | m/s |

---

## DA methods

All methods live in `src/da/core.py`.

| Method | Function | Description |
|--------|----------|-------------|
| LETKF | `letkf_update` | Standard single-step LETKF |
| TEnKF | `tenkf_update` | Tempered LETKF — fixed Ntemp, H(x) recomputed at each step |
| AOEI | `aoei_update` | LETKF + Adaptive Observation Error Inflation (single step) |
| ATEnKF | `atenkf_update` | Locally adaptive tempering — Ntemp determined per observation from AOEI inflation ratio |
| TAOEI | `taoei_update` | TEnKF with AOEI recomputed at every tempering step |

### Tempering schedule

Weights follow :

```
alpha_i = exp(-(Nt+1)*alpha_s / i) / sum_j exp(-(Nt+1)*alpha_s / j)
```

`sum(alpha_i) = 1` guarantees that total information across all steps equals
`R0` (information-preserving property). Larger `alpha_s` back-loads weight
toward later iterations; `alpha_s = 0` gives equal weights.

### Localization

R-localization (Greybush et al. 2011). The Fortran inflates observation error
by `exp(0.5*(d/L)^2)` at distance `d` from a grid point with scale `L`.
Set `loc_x/y/z: 99999` (or `null`) to disable localization on an axis.
---

## Running experiments

### 1. Sanity check first

Before any full run, verify all methods work correctly on a single point:

```bash
python tests/run_sanity_check.py --config configs/ws2.yaml \
    --truth 0 --x 10 --y 0 --z 15
```

Prints a table showing prior diagnostics, AOEI inflation ratio, ATEnKF
`Ntemp_j`, posterior mean per method, and innovation reduction percentage.

### 2. Run an experiment

All experiments use the same runner — the config controls everything:

```bash
python src/runners/run_experiment.py --config configs/ws1.yaml
python src/runners/run_experiment.py --config configs/ws2.yaml --workers 30
python src/runners/run_experiment.py --config configs/ws2.yaml --verbose 1
```

### 3. Submit to cluster

```bash
qsub -v CFG=configs/ws2.yaml,WORKERS=30 queue_ws.sh
qsub -v CFG=configs/ws3.yaml,WORKERS=30 queue_ws.sh
```
---

## Output files

A copy of the config yaml is written to `outdir` before any results, so
every output folder is self-contained.

### Filename convention

**Single-obs:**
```
{tag}_{method}_Nt{ntemp}_as{alpha_s}_Lx{lx}Ly{ly}Lz{lz}_Ne{ne}_obs{x}_{y}_{z}_qc{qc}_True{tm}.npz
```
 
**Multi-obs:**
```
{tag}_{method}_Nt{ntemp}_as{alpha_s}_Lx{lx}Ly{ly}Lz{lz}_Ne{ne}_str{stride}_qc{qc}_True{tm}.npz
```

**QC code:**

| Code | Meaning |
|------|---------|
| `none` | no filtering |
| `E` | ensemble mean filter only |
| `T` | truth filter only |
| `ET_and` | both filters, AND logic |
| `ET_or` | both filters, OR logic |

### Array contents per file

| Key | Description |
|-----|-------------|
| `xa` | posterior ensemble `(nx,ny,nz,Ne,nvar)` |
| `yo` | observations used |
| `hxf_mean` | prior ensemble mean in obs space |
| `dep` | innovation `yo - H(x̄^f)` |
| `spread` | prior ensemble spread in obs space |
| `obs_error` | obs error variance used (after AOEI if applicable) |
| `ox, oy, oz` | obs grid indices |
| `truth_member` | which member was used as truth |
| `xatemp` *(TEnKF/ATEnKF)* | ensemble at each tempering step |
| `ntemps_per_obs` *(ATEnKF)* | per-observation Ntemp_j |

---

## Config reference

See `configs/template.yaml` for the full documented reference with all
options, accepted formats, and defaults. Key points:

- `obs_error_var` is variance (dBZ²), not std
- `prior_size: null` uses all remaining members (default); set to scalar or list for ensemble size sensitivity — always recorded as `_Ne{N}` in the filename
- Sweep parameters accept scalar, list, or `{start, stop, num}` (stop inclusive)
- `loc_x/y/z: null` disables localization (equivalent to L=99999)
- `skip_existing: true` resumes a partial run without recomputing finished files
- `verbose: 1` is recommended for cluster runs (one line per truth member)
```
