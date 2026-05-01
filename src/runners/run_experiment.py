"""
src/runners/run_experiment.py
==============================
Three experiment modes, selected via sweep.obs_points.mode in the config:

  single_obs   One fixed observation point, all method combos.
               Always stores metrics only (no fields option).

  sweep        Every QC-passing stride point as an independent single-obs
               assimilation. All combos per point. Sequential by default.
               Use --workers N for parallel execution (fork-based, Linux).

  multi_obs    All QC-passing stride points assimilated together in one
               Fortran call per combo. Stores xa_mean and a shared reference
               file (truth, xf_mean, yo, positions) written once per tm.
               Set OMP_NUM_THREADS in the environment for multi-core Fortran.

Localization scales are in km throughout. pos_km (nx,ny,nz,3) must be
present in the prepared .npz.

Method deduplication
--------------------
AOEI and LETKF produce identical results for any ntemp value (single-step
methods). They are run once per (method, alpha_s, lx, ly, lz) regardless
of the ntemp sweep. ntemp is recorded as 1 in the output.

Usage
-----
  python src/runners/run_experiment.py --config configs/exp.yaml
  python src/runners/run_experiment.py --config configs/exp.yaml --tm 3
  python src/runners/run_experiment.py --config configs/exp.yaml --workers 16
"""

import argparse
import itertools
import math
import os
import pathlib
import shutil
import sys
import time
from multiprocessing import Pool

import numpy as np
import yaml

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "src" / "fortran"))

import da.core as core
from da.core import tenkf_update, aoei_update
from da.metrics import compute_single_obs_metrics, compute_multi_obs_metrics
# ---------------------------------------------------------------------------
# Module-level shared arrays
# Set once in main before any fork. Workers read them via copy-on-write.
# ---------------------------------------------------------------------------
_XF       = None   # (nx, ny, nz, Ne, nvar)  float32 F-order  — prior ensemble
_ENS_HX   = None   # (nx, ny, nz, Ne)         float32          — H(xf) full domain
_TRUTH    = None   # (nx, ny, nz, nvar)        float32          — truth state
_TRUTH_HX = None   # (nx, ny, nz)              float32          — H(truth) full domain
_POS_KM   = None   # (nx, ny, nz, 3)           float32          — [x_km, y_km, z_km]
_NOISE    = None   # (nx, ny, nz)              float32          — noise field


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _expand(val, is_int=False):
    """Expand a config value to a flat list (scalar, list, or {start,stop,num})."""
    if isinstance(val, dict):
        arr = np.linspace(val["start"], val["stop"], int(val["num"]))
        if is_int:
            arr = np.unique(np.round(arr).astype(int))
        return arr.tolist()
    if isinstance(val, list):
        return val
    return [val]


def _build_combos(sweep_cfg):
    """
    Return deduplicated list of (method, ntemp, alpha_s, lx_km, ly_km, lz_km).

    AOEI, LETKF  — ntemp irrelevant; one entry per (method, alpha_s, lx, ly, lz).
    TEnKF        — one entry per distinct ntemp value.
    """
    methods = _expand(sweep_cfg.get("methods",  ["TEnKF"]))
    ntemps  = _expand(sweep_cfg.get("ntemp",    [1]), is_int=True)
    alphas  = _expand(sweep_cfg.get("alpha_s",  [2.0]))
    lxs     = _expand(sweep_cfg.get("loc_x",    [10.0]))
    lys     = _expand(sweep_cfg.get("loc_y",    [10.0]))
    lzs     = _expand(sweep_cfg.get("loc_z",    [4.0]))

    combos = []
    seen_single = set()
    for method, ntemp, alpha_s, lx, ly, lz in itertools.product(
            methods, ntemps, alphas, lxs, lys, lzs):
        if method in ("AOEI", "LETKF"):
            key = (method, float(alpha_s), float(lx), float(ly), float(lz))
            if key in seen_single:
                continue
            seen_single.add(key)
            combos.append((method, 1, float(alpha_s), float(lx), float(ly), float(lz)))
        else:
            combos.append((method, int(ntemp), float(alpha_s),
                           float(lx), float(ly), float(lz)))
    return combos


def _qc_pass(yo_val, hxf_max_val, qc_cfg):
    if not qc_cfg:
        return True
    dbz_min = float(qc_cfg.get("dbz_min", 5.0))
    fe   = bool(qc_cfg.get("filter_ensemble", True))
    ft   = bool(qc_cfg.get("filter_truth",    False))
    mode = qc_cfg.get("filter_mode", "and").lower()
    fail_e = fe and (float(hxf_max_val) < dbz_min)
    fail_t = ft and (float(yo_val)      < dbz_min)
    if fe and ft:
        return not (fail_e and fail_t) if mode == "or" else not (fail_e or fail_t)
    return not fail_e if fe else not fail_t


# ---------------------------------------------------------------------------
# H(x) computation
# ---------------------------------------------------------------------------

def _calc_hx_domain(state, var_idx):
    """
    Vectorised H(x) over the full (sub)domain via Fortran calc_ref_ens.

    state : (nx,ny,nz,nvar)      single member -> returns (nx,ny,nz)
          | (nx,ny,nz,Ne,nvar)   ensemble      -> returns (nx,ny,nz,Ne)
    """
    from cletkf_wloc import common_da as cda
    vi = var_idx
    if state.ndim == 4:                          # single member
        s = state[:, :, :, np.newaxis, :]
        ref = cda.calc_ref_ens(
            s[:,:,:,:,vi["qr"]].astype(np.float64),
            s[:,:,:,:,vi["qs"]].astype(np.float64),
            s[:,:,:,:,vi["qg"]].astype(np.float64),
            s[:,:,:,:,vi["T"] ].astype(np.float64),
            s[:,:,:,:,vi["P"] ].astype(np.float64),
        )
        return ref[:, :, :, 0].astype(np.float32)
    else:                                        # ensemble
        ref = cda.calc_ref_ens(
            state[:,:,:,:,vi["qr"]].astype(np.float64),
            state[:,:,:,:,vi["qs"]].astype(np.float64),
            state[:,:,:,:,vi["qg"]].astype(np.float64),
            state[:,:,:,:,vi["T"] ].astype(np.float64),
            state[:,:,:,:,vi["P"] ].astype(np.float64),
        )
        return ref.astype(np.float32)


# ---------------------------------------------------------------------------
# Setup — runs once, sets module globals
# ---------------------------------------------------------------------------

def _setup(cfg, tm):
    """
    Load ensemble, slice xf and truth, free ens, compute H(x) and noise.
    Sets all module-level globals.
    Returns (pts, Ne):
      pts : list of (i, j, k) tuples passing QC and stride filter
      Ne  : actual prior ensemble size
    """
    global _XF, _ENS_HX, _TRUTH, _TRUTH_HX, _POS_KM, _NOISE

    var_idx = cfg["state"]["var_idx"]

    # --- load ---
    t0 = time.time()
    core._log(1, f"[setup tm={tm:02d}] loading {cfg['paths']['prepared']} ...")
    data = np.load(cfg["paths"]["prepared"])
    ens  = data["state_ensemble"] if "state_ensemble" in data else data["cross_sections"]
    # ens is already float32 from extract_3d_subset.py — no copy needed here

    if "pos_km" not in data:
        raise KeyError(
            "'pos_km' not found in prepared .npz. "
            "Re-run extract_3d_subset.py to add grid positions in km.")
    pos_km = data["pos_km"].astype(np.float32)   # (nx,ny,nz,3)

    core._log(1, f"[setup tm={tm:02d}] loaded  {time.time()-t0:.1f}s  "
                 f"ens={ens.nbytes/1e9:.2f} GB")

    nx, ny, nz = ens.shape[:3]
    Ne_tot     = ens.shape[3]
    core._log(2, f"[setup tm={tm:02d}] domain={nx}x{ny}x{nz}  Ne_tot={Ne_tot}")

    # --- slice truth and prior, then free ens ---
    prior_size = cfg["sweep"].get("prior_size", None)
    all_others = [i for i in range(Ne_tot) if i != tm]
    if prior_size is not None:
        prior_size = int(prior_size)
        if prior_size > len(all_others):
            raise ValueError(
                f"prior_size={prior_size} exceeds available members "
                f"({len(all_others)}) for truth member {tm}.")
        all_others = all_others[:prior_size]
    Ne = len(all_others)

    core._log(2, f"[setup tm={tm:02d}] slicing truth (tm={tm}) and prior (Ne={Ne}) ...")
    truth = ens[:, :, :, tm, :].copy()                    # (nx,ny,nz,nvar) C-order copy
    xf    = np.asfortranarray(ens[:, :, :, all_others, :])# (nx,ny,nz,Ne,nvar) F-order
    del ens
    core._log(1, f"[setup tm={tm:02d}] ens freed  Ne={Ne}  "
                 f"xf={xf.nbytes/1e9:.2f} GB")

    # --- H(x) over full domain ---
    t1 = time.time()
    core._log(2, f"[setup tm={tm:02d}] computing H(truth) over {nx}x{ny}x{nz} domain ...")
    truth_hx = _calc_hx_domain(truth, var_idx)        # (nx,ny,nz)
    core._log(1, f"[setup tm={tm:02d}] H(truth) done  {time.time()-t1:.1f}s")

    t1 = time.time()
    core._log(2, f"[setup tm={tm:02d}] computing H(xf)   over {nx}x{ny}x{nz}x{Ne} points ...")
    ens_hx = _calc_hx_domain(xf, var_idx)             # (nx,ny,nz,Ne)
    core._log(1, f"[setup tm={tm:02d}] H(xf)   done  {time.time()-t1:.1f}s")

    # --- reproducible noise field ---
    sigma = float(np.sqrt(float(cfg["obs"]["obs_error_var"])))
    rng   = np.random.default_rng(42 + tm)

    add_noise = bool(cfg["obs"].get("add_noise", False))
    if add_noise:
        noise = rng.normal(0.0, sigma, (nx, ny, nz)).astype(np.float32)
    else:
        noise = np.zeros((nx, ny, nz), dtype=np.float32)

    core._log(2, f"[setup tm={tm:02d}] noise field ready  sigma={sigma:.3f} dBZ  seed={42+tm}")

    # --- QC-filtered stride points ---
    stride  = int(cfg["sweep"].get("stride", 1))
    qc_cfg  = cfg.get("qc", {})
    ens_max = ens_hx.max(axis=3)                       # (nx,ny,nz)

    t1  = time.time()
    core._log(2, f"[setup tm={tm:02d}] applying QC (stride={stride}) ...")
    pts = []
    for i in range(0, nx, stride):
        for j in range(0, ny, stride):
            for k in range(0, nz):
                if _qc_pass(truth_hx[i, j, k], ens_max[i, j, k], qc_cfg):
                    pts.append((i, j, k))
    del ens_max
    core._log(1, f"[setup tm={tm:02d}] {len(pts)} stride-{stride} pts pass QC"
                 f"  ({time.time()-t1:.1f}s)")

    # --- set globals ---
    _XF       = xf
    _ENS_HX   = ens_hx
    _TRUTH    = truth
    _TRUTH_HX = truth_hx
    _POS_KM   = pos_km
    _NOISE    = noise

    return pts, Ne


# ---------------------------------------------------------------------------
# Subdomain and localization helpers
# ---------------------------------------------------------------------------

def _subdomain_slices(i0, j0, k0, lx_km, ly_km, lz_km,
                      pos_km, nx, ny, nz, cutoff_factor=4.0):
    """
    Conservative rectangular bounding box containing the full localization
    cutoff zone around obs point (i0, j0, k0).

    Horizontal: approximate using local dx/dy (nearly uniform ~2 km grid).
    Vertical:   exact, using actual non-uniform z levels from pos_km.

    Returns (si, sj, sk) Python slices into full-domain arrays.
    """
    di  = 1 if i0 + 1 < nx else -1
    dj  = 1 if j0 + 1 < ny else -1
    dx  = max(abs(float(pos_km[i0+di, j0, k0, 0] - pos_km[i0, j0, k0, 0])), 0.1)
    dy  = max(abs(float(pos_km[i0, j0+dj, k0, 1] - pos_km[i0, j0, k0, 1])), 0.1)

    half_i = int(np.ceil(cutoff_factor * lx_km / dx))
    half_j = int(np.ceil(cutoff_factor * ly_km / dy))

    i_min = max(0, i0 - half_i);  i_max = min(nx, i0 + half_i + 1)
    j_min = max(0, j0 - half_j);  j_max = min(ny, j0 + half_j + 1)

    # vertical: exact km distances using actual z levels
    z0    = float(pos_km[i0, j0, k0, 2])
    z_lev = pos_km[i0, j0, :, 2]              # (nz,)
    k_msk = np.abs(z_lev - z0) <= cutoff_factor * lz_km
    k_idx = np.where(k_msk)[0]
    if len(k_idx) == 0:
        k_min, k_max = 0, nz
    else:
        k_min = int(k_idx[0])
        k_max = int(k_idx[-1]) + 1

    return slice(i_min, i_max), slice(j_min, j_max), slice(k_min, k_max)


def _compute_rho(pos_km_sub, x0, y0, z0, lx_km, ly_km, lz_km):
    """
    Gaussian R-localization weight field over the subdomain.
    Shape (nx_s, ny_s, nz_s), float32.
    Points beyond the compact-support cutoff (2*sqrt(10/3)*L) receive 0.
    """
    dx = pos_km_sub[:, :, :, 0] - x0
    dy = pos_km_sub[:, :, :, 1] - y0
    dz = pos_km_sub[:, :, :, 2] - z0

    d2 = np.zeros(dx.shape, dtype=np.float32)
    if lx_km > 0: d2 += (dx / lx_km) ** 2
    if ly_km > 0: d2 += (dy / ly_km) ** 2
    if lz_km > 0: d2 += (dz / lz_km) ** 2

    cutoff = (2.0 * np.sqrt(10.0 / 3.0)) ** 2
    return np.where(d2 <= cutoff, np.exp(-0.5 * d2), 0.0).astype(np.float32)

# ---------------------------------------------------------------------------
# DA dispatch (subdomain, single observation)
# ---------------------------------------------------------------------------

def _da_subdomain(xf_sub, yo, R0_val, ox_s, oy_s, oz_s,
                  pos_km_sub, loc_scales_km, var_idx, method, ntemp, alpha_s):
    """
    Run one DA combo on a subdomain slice.

    ox_s, oy_s, oz_s : int   obs position within the subdomain, 0-based
    pos_km_sub       : (nx_s, ny_s, nz_s, 3)  km positions over subdomain
    loc_scales_km    : sequence [lx_km, ly_km, lz_km]

    Returns xa_sub (nx_s, ny_s, nz_s, Ne, nvar).
    """
    yo_a = np.array([yo],     np.float32)
    R0_a = np.array([R0_val], np.float32)
    ox_a = np.array([ox_s],   np.int32)
    oy_a = np.array([oy_s],   np.int32)
    oz_a = np.array([oz_s],   np.int32)
    loc  = np.asarray(loc_scales_km, np.float32)

    if method == "TEnKF":
        return tenkf_update(xf_sub, yo_a, R0_a, ox_a, oy_a, oz_a,
                            loc, var_idx, ntemp, alpha_s, pos_km_sub)["xa"]
    if method == "AOEI":
        return aoei_update(xf_sub, yo_a, R0_a, ox_a, oy_a, oz_a,
                           loc, var_idx, pos_km_sub)["xa"]
    raise ValueError(f"Unknown method: {method}")


# ---------------------------------------------------------------------------
# Single-point processing (shared by sweep and single_obs modes)
# ---------------------------------------------------------------------------

def _process_point(i0, j0, k0, combos, var_idx, R0_val,
                   cutoff_factor=4.0, return_fields=False):
    """
    Run all combos for one observation point.
 
    Returns
    -------
    row    : flat dict, each key -> (n_combos,) array.
             Contains all meta and all metrics. Ready to concatenate across points.
    fields : dict {combo_index: xa_sub} if return_fields else None
    """
    xf       = _XF
    ens_hx   = _ENS_HX
    truth    = _TRUTH
    truth_hx = _TRUTH_HX
    pos_km   = _POS_KM
    noise    = _NOISE
    nx, ny, nz, Ne, nvar = xf.shape
 
    # variable names in index order — consistent with var_idx
    var_names = [k for k, _ in sorted(var_idx.items(), key=lambda x: x[1])]
 
    x0       = float(pos_km[i0, j0, k0, 0])
    y0       = float(pos_km[i0, j0, k0, 1])
    z0       = float(pos_km[i0, j0, k0, 2])
    yo_clean = float(truth_hx[i0, j0, k0])
    yo       = float(yo_clean + noise[i0, j0, k0])
 
    n_c = len(combos)
 
    # fixed meta — same for every combo at this point
    fixed_meta = dict(
        i        = np.full(n_c, i0,       np.int32),
        j        = np.full(n_c, j0,       np.int32),
        k        = np.full(n_c, k0,       np.int32),
        x_km     = np.full(n_c, x0,       np.float32),
        y_km     = np.full(n_c, y0,       np.float32),
        z_km     = np.full(n_c, z0,       np.float32),
        yo       = np.full(n_c, yo,       np.float32),
        yo_clean = np.full(n_c, yo_clean, np.float32),
    )
 
    # combo-varying meta
    method_arr  = np.empty(n_c, dtype="U8")
    ntemp_arr   = np.empty(n_c, np.int32)
    alpha_s_arr = np.empty(n_c, np.float32)
    lx_arr      = np.empty(n_c, np.float32)
    ly_arr      = np.empty(n_c, np.float32)
    lz_arr      = np.empty(n_c, np.float32)
 
    metrics_rows = [None] * n_c
    fields       = {} if return_fields else None
    sub_cache    = {}
 
    for c, (method, ntemp, alpha_s, lx_km, ly_km, lz_km) in enumerate(combos):
        method_arr[c]  = method
        ntemp_arr[c]   = ntemp
        alpha_s_arr[c] = alpha_s
        lx_arr[c]      = lx_km
        ly_arr[c]      = ly_km
        lz_arr[c]      = lz_km
 
        loc_key = (lx_km, ly_km, lz_km)
        if loc_key not in sub_cache:
            si, sj, sk = _subdomain_slices(
                i0, j0, k0, lx_km, ly_km, lz_km, pos_km, nx, ny, nz, cutoff_factor)
            xf_sub       = np.asfortranarray(xf[si, sj, sk, :, :])
            ens_hx_sub   = ens_hx[si, sj, sk, :]
            truth_sub    = truth[si, sj, sk, :]
            truth_hx_sub = truth_hx[si, sj, sk]
            pos_km_sub   = np.asfortranarray(pos_km[si, sj, sk, :])
            ox_s = i0 - si.start
            oy_s = j0 - sj.start
            oz_s = k0 - sk.start
            rho  = _compute_rho(pos_km_sub, x0, y0, z0, lx_km, ly_km, lz_km)
            sub_cache[loc_key] = (xf_sub, ens_hx_sub, truth_sub, truth_hx_sub,
                                  pos_km_sub, ox_s, oy_s, oz_s, rho)
        (xf_sub, ens_hx_sub, truth_sub, truth_hx_sub,
         pos_km_sub, ox_s, oy_s, oz_s, rho) = sub_cache[loc_key]
 
        xa_sub  = _da_subdomain(xf_sub, yo, R0_val, ox_s, oy_s, oz_s,
                                pos_km_sub, (lx_km, ly_km, lz_km),
                                var_idx, method, ntemp, alpha_s)
        hxa_sub = _calc_hx_domain(xa_sub, var_idx)   # (nx_s, ny_s, nz_s, Ne)
 
        metrics_rows[c] = compute_single_obs_metrics(
            xf_sub, xa_sub, truth_sub,
            ens_hx_sub, hxa_sub, truth_hx_sub,
            rho, ox_s, oy_s, oz_s, yo, var_names)
 
        if return_fields:
            fields[c] = xa_sub
 
    # assemble flat output dict — each key -> (n_combos,) array
    combo_meta = dict(
        method  = method_arr,
        ntemp   = ntemp_arr,
        alpha_s = alpha_s_arr,
        lx_km   = lx_arr,
        ly_km   = ly_arr,
        lz_km   = lz_arr,
    )
 
    # stack metric scalars: metrics_rows[c] is a flat dict of floats
    metric_keys = list(metrics_rows[0].keys())
    metrics_flat = {
        k: np.array([metrics_rows[c][k] for c in range(n_c)], dtype=np.float32)
        for k in metric_keys
    }
 
    row = {**fixed_meta, **combo_meta, **metrics_flat}
    return row, fields



# ---------------------------------------------------------------------------
# Sweep mode
# ---------------------------------------------------------------------------

def _run_sweep_sequential(pts, combos, cfg, outdir, tag, tm, Ne):
    """Process all stride points sequentially, write one npz at the end."""
    var_idx = cfg["state"]["var_idx"]
    R0_val  = float(cfg["obs"]["obs_error_var"])
    cutoff  = float(cfg.get("cutoff_factor", 4.0))
    n_pts   = len(pts)
    n_c     = len(combos)
 
    core._log(1, f"[sweep tm={tm:02d}] {n_pts} pts x {n_c} combos = {n_pts*n_c} rows")
 
    all_rows = []
    t0 = time.time()
 
    for p_idx, (i0, j0, k0) in enumerate(pts):
        if p_idx % 500 == 0:
            elapsed = time.time() - t0
            rate    = p_idx / elapsed if p_idx > 0 else 0.0
            eta     = (n_pts - p_idx) / rate if rate > 0 else 0.0
            core._log(1, f"  [sweep] pt {p_idx}/{n_pts}  "
                         f"{rate:.1f} pts/s  ETA {eta/60:.0f} min")
        row, _ = _process_point(i0, j0, k0, combos, var_idx, R0_val, cutoff)
        all_rows.append(row)
 
    merged = {k: np.concatenate([r[k] for r in all_rows]) for k in all_rows[0]}
    var_names = [k for k, _ in sorted(var_idx.items(), key=lambda x: x[1])]
 
    fname = f"{tag}_sweep_Ne{Ne:03d}_tm{tm:02d}.npz"
    out   = os.path.join(outdir, fname)
    np.savez_compressed(out, var_names=np.array(var_names + ["ref"]), **merged)
    sz = os.path.getsize(out) / 1e6
    core._log(1, f"[sweep tm={tm:02d}] saved {n_pts*n_c} rows  "
                 f"{sz:.1f} MB  {time.time()-t0:.1f}s -> {fname}")

    
def _worker_init():
    """
    Called once per worker process immediately after fork.
    Forces OMP to use 1 thread regardless of how the parent initialized it.
    Setting os.environ alone is insufficient — OMP runtime is already
    initialized in the parent and inherited via fork.
    """
    import ctypes
    omp_env = os.environ.get("OMP_NUM_THREADS", "unset")
    try:
        libgomp = ctypes.CDLL("libgomp.so.1")
        libgomp.omp_set_num_threads(1)
        core._log(1, f"[worker init pid={os.getpid()}] "
                     f"omp_set_num_threads(1) OK  "
                     f"OMP_NUM_THREADS_env={omp_env}")
    except Exception as e:
        core._log(1, f"[worker init pid={os.getpid()}] "
                     f"WARNING: ctypes OMP reset failed: {e}  "
                     f"OMP_NUM_THREADS_env={omp_env}")

def _sweep_worker(args):
    """Worker for parallel sweep: process a chunk of points, return merged rows."""
    import os as _os
    _os.environ["OMP_NUM_THREADS"]     = "1"
    _os.environ["MKL_NUM_THREADS"]     = "1"
    _os.environ["OPENBLAS_NUM_THREADS"] = "1"
 
    pts_chunk, combos, var_idx, R0_val, cutoff = args
    all_rows = []
    for (i0, j0, k0) in pts_chunk:
        row, _ = _process_point(i0, j0, k0, combos, var_idx, R0_val, cutoff)
        all_rows.append(row)
    merged = {k: np.concatenate([r[k] for r in all_rows]) for k in all_rows[0]}
    return merged
 



def _run_sweep_parallel(pts, combos, cfg, outdir, tag, tm, Ne, n_workers):
    """
    Process stride points across n_workers processes (fork-based, Linux).
    Uses imap_unordered so results stream in as workers finish, allowing
    live progress reporting without waiting for all workers to complete.
    """
    var_idx = cfg["state"]["var_idx"]
    R0_val  = float(cfg["obs"]["obs_error_var"])
    cutoff  = float(cfg.get("cutoff_factor", 4.0))
    n_pts   = len(pts)
    n_c     = len(combos)

    core._log(1, f"[sweep tm={tm:02d}] parallel  workers={n_workers}  "
                 f"{n_pts} pts x {n_c} combos = {n_pts*n_c} rows")

    import os as _os
    _os.environ["OMP_NUM_THREADS"]     = "1"
    _os.environ["MKL_NUM_THREADS"]     = "1"
    _os.environ["OPENBLAS_NUM_THREADS"]= "1"

    chunk_size  = max(1, n_pts // n_workers)
    chunks      = [pts[i:i+chunk_size] for i in range(0, n_pts, chunk_size)]
    worker_args = [(c, combos, var_idx, R0_val, cutoff) for c in chunks]
    n_chunks    = len(chunks)

    all_rows = []
    t0       = time.time()
 
    with Pool(processes=n_workers, initializer=_worker_init) as pool:
        for done, merged_chunk in enumerate(
                pool.imap_unordered(_sweep_worker, worker_args), start=1):
            all_rows.append(merged_chunk)
            elapsed   = time.time() - t0
            rows_done = sum(len(r["i"]) for r in all_rows)
            rate      = rows_done / n_c / elapsed if elapsed > 0 else 0.0
            eta       = (n_pts - rows_done // n_c) / rate if rate > 0 else 0.0
            core._log(1, f"  [sweep] chunk {done}/{n_chunks}  "
                         f"pt {rows_done//n_c}/{n_pts}  "
                         f"{rate:.1f} pts/s  ETA {eta/60:.0f} min")
 
    merged    = {k: np.concatenate([r[k] for r in all_rows]) for k in all_rows[0]}
    var_names = [k for k, _ in sorted(var_idx.items(), key=lambda x: x[1])]
 
    fname = f"{tag}_sweep_Ne{Ne:03d}_tm{tm:02d}.npz"
    out   = os.path.join(outdir, fname)
    np.savez_compressed(out, var_names=np.array(var_names + ["ref"]), **merged)
    sz = os.path.getsize(out) / 1e6
    core._log(1, f"[sweep tm={tm:02d}] saved {n_pts*n_c} rows  "
                 f"{sz:.1f} MB  {time.time()-t0:.1f}s -> {fname}")



# ---------------------------------------------------------------------------
# Single-obs mode (one fixed point)
# ---------------------------------------------------------------------------

def _run_single_obs(combos, cfg, outdir, tag, tm, Ne):
    """
    One fixed obs point, all combos.
    Always stores metrics (per-combo scalars + 8×9 metric arrays).
    Fields (xf_sub, xa_sub) are never stored here — use multi_obs
    with store_fields: true for full ensemble output.
    """
    var_idx = cfg["state"]["var_idx"]
    R0_val  = float(cfg["obs"]["obs_error_var"])
    cutoff  = float(cfg.get("cutoff_factor", 4.0))

    obs_loc    = cfg["sweep"]["obs_points"]["loc"]
    i0, j0, k0 = int(obs_loc["x"]), int(obs_loc["y"]), int(obs_loc["z"])

    core._log(1, f"[single_obs tm={tm:02d}] obs=({i0},{j0},{k0})  {len(combos)} combos")

    row, _ = _process_point(
        i0, j0, k0, combos, var_idx, R0_val, cutoff,
        return_fields=False)
 
    var_names = [k for k, _ in sorted(var_idx.items(), key=lambda x: x[1])]
    fname = f"{tag}_single_obs_{i0}_{j0}_{k0}_Ne{Ne:03d}_tm{tm:02d}.npz"
    out   = os.path.join(outdir, fname)
    np.savez_compressed(out, var_names=np.array(var_names + ["ref"]), **row)
    sz = os.path.getsize(out) / 1e6
    core._log(1, f"[single_obs tm={tm:02d}] saved  {sz:.1f} MB -> {fname}")



# ---------------------------------------------------------------------------
# Multi-obs mode
# ---------------------------------------------------------------------------

def _run_multi_obs(pts, combos, cfg, outdir, tag, tm, Ne):
    """
    Assimilate all QC-passing stride points together in one Fortran call
    per combo.

    store_fields: false (default)
        Saves xa_mean (nx,ny,nz,nvar) + global RMSE/spread per combo.
        Compact — suitable for all truth members and parameter sweeps.

    store_fields: true
        Saves full xa (nx,ny,nz,Ne,nvar) and xf (nx,ny,nz,Ne,nvar) per combo.
        Use only for selected cases — each file is ~9 GB uncompressed.

    A shared reference file (truth, xf_mean, yo, obs positions) is written
    once per truth member regardless of store_fields.

    For multi-core Fortran: set OMP_NUM_THREADS in the environment before launch.
    """
    var_idx      = cfg["state"]["var_idx"]
    R0_val       = float(cfg["obs"]["obs_error_var"])
    store_fields = bool(cfg.get("store_fields", False))

    xf       = _XF
    truth    = _TRUTH
    truth_hx = _TRUTH_HX
    noise    = _NOISE
    pos_km   = _POS_KM
    nx, ny, nz, Ne_, nvar = xf.shape

    # obs arrays from stride pts
    ix  = np.array([p[0] for p in pts], np.int32)
    iy  = np.array([p[1] for p in pts], np.int32)
    iz  = np.array([p[2] for p in pts], np.int32)
    yo_clean = truth_hx[ix, iy, iz].astype(np.float32)
    yo       = (yo_clean + noise[ix, iy, iz]).astype(np.float32)
    R0       = np.full(len(pts), R0_val, np.float32)

    # shared reference file — written once per tm
    ref_fname = f"{tag}_multi_obs_ref_Ne{Ne:03d}_tm{tm:02d}.npz"
    ref_out   = os.path.join(outdir, ref_fname)
    if not os.path.exists(ref_out):
        np.savez_compressed(ref_out,
            truth    = truth,
            xf_mean  = xf.mean(axis=3).astype(np.float32),
            truth_hx = truth_hx,
            yo       = yo,
            yo_clean = yo_clean,
            ix=ix, iy=iy, iz=iz,
            var_names = np.array(list(var_idx.keys())),
        )
        sz = os.path.getsize(ref_out) / 1e6
        core._log(1, f"[multi_obs tm={tm:02d}] ref file  {sz:.0f} MB -> {ref_fname}")

    for (method, ntemp, alpha_s, lx_km, ly_km, lz_km) in combos:
        fname = (f"{tag}_multi_obs_{method}_Nt{ntemp:02d}"
                 f"_as{alpha_s:.1f}_Lx{lx_km}Ly{ly_km}Lz{lz_km}"
                 f"_Ne{Ne:03d}_tm{tm:02d}.npz")
        out = os.path.join(outdir, fname)

        if cfg.get("skip_existing", False) and os.path.exists(out):
            core._log(2, f"  [skip] {fname}")
            continue

        loc = np.array([lx_km, ly_km, lz_km], np.float32)
        core._log(2, f"  {method} Nt={ntemp} as={alpha_s} "
                     f"L=[{lx_km},{ly_km},{lz_km} km]  nobs={len(pts)}"
                     f"  store_fields={store_fields}")
        t1 = time.time()

        if method == "TEnKF":
            res = tenkf_update(xf, yo, R0, ix, iy, iz,
                               loc, var_idx, ntemp, alpha_s,
                               np.asfortranarray(pos_km))
        elif method == "AOEI":
            res = aoei_update(xf, yo, R0, ix, iy, iz, loc, var_idx,
                              np.asfortranarray(pos_km))
        else:
            raise ValueError(f"Unknown method: {method}")

        xa = res["xa"]   # (nx, ny, nz, Ne, nvar)
 
        # H(xa_mean) field — precompute for metrics (single-member path of _calc_hx_domain)
        hxa_mean_field = _calc_hx_domain(xa.mean(axis=3), var_idx)  # (nx, ny, nz)
        hxf_mean_field = _ENS_HX.mean(axis=3)                        # (nx, ny, nz)
        truth_hx_field = _TRUTH_HX                                    # (nx, ny, nz)
 
        var_names = [k for k, _ in sorted(var_idx.items(), key=lambda x: x[1])]
 
        m = compute_multi_obs_metrics(
            xa, xf, truth,
            hxf_mean_field, hxa_mean_field, truth_hx_field,
            var_names, store_fields=store_fields)
 
        np.savez_compressed(out,
            method       = method,
            ntemp        = np.int32(ntemp),
            alpha_s      = np.float32(alpha_s),
            lx_km        = np.float32(lx_km),
            ly_km        = np.float32(ly_km),
            lz_km        = np.float32(lz_km),
            Ne           = np.int32(Ne),
            truth_member = np.int32(tm),
            var_names    = np.array(var_names),
            ref_file     = ref_fname,
            **m,
        )
 
        del xa
        sz = os.path.getsize(out) / 1e6
        core._log(1, f"  saved {sz:.0f} MB  {time.time()-t1:.1f}s -> {fname}")



# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="WRF single-cycle assimilation experiment runner.")
    ap.add_argument("--config",  required=True,
                    help="Path to YAML config file.")
    ap.add_argument("--workers", type=int, default=1,
                    help="Parallel workers for sweep mode. 1 = sequential (default).")
    ap.add_argument("--verbose", type=int, default=None,
                    help="Verbosity level 0-3 (overrides config).")
    ap.add_argument("--tm",      type=int, default=None,
                    help="Truth member index (overrides config).")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    verbose = args.verbose if args.verbose is not None else int(cfg.get("verbose", 1))
    core.set_verbose(verbose)

    outdir = cfg["paths"]["outdir"]
    tag    = cfg.get("experiment_tag", "EXP")
    os.makedirs(outdir, exist_ok=True)
    shutil.copy2(args.config, os.path.join(outdir, f"{tag}_config.yaml"))
    print(f"[{tag}] config saved -> {outdir}")

    # resolve truth member (single value per run — no inner loop over tm)
    if args.tm is not None:
        tm = args.tm
    else:
        tm_cfg = cfg["sweep"].get("truth_members", 0)
        tm = tm_cfg[0] if isinstance(tm_cfg, list) else int(tm_cfg)

    # resolve mode
    obs_cfg  = cfg["sweep"]["obs_points"]
    obs_mode = obs_cfg if isinstance(obs_cfg, str) else obs_cfg.get("mode", "sweep")

    n_workers = max(1, args.workers)
    core._log(1, f"[{tag}] mode={obs_mode}  tm={tm:02d}  workers={n_workers}")

    t_start = time.time()

    core._log(1, f"[main] OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS','unset')} "
                 f"before setup  pid={os.getpid()}")

    if obs_mode == "sweep" and n_workers > 1:
        os.environ["OMP_NUM_THREADS"] = "1"
        core._log(1, f"[main] OMP forced to 1 before setup (fork-safe)  "
                    f"H(xf) will be single-threaded")

    pts, Ne = _setup(cfg, tm)
    combos  = _build_combos(cfg["sweep"])

    core._log(1, f"[main] OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS','unset')} "
                 f"after setup  pid={os.getpid()}")
    core._log(1, f"[{tag}] {len(combos)} combos  {len(pts)} obs candidates")

    if obs_mode == "single_obs":
        _run_single_obs(combos, cfg, outdir, tag, tm, Ne)

    elif obs_mode == "sweep":
        if n_workers == 1:
            _run_sweep_sequential(pts, combos, cfg, outdir, tag, tm, Ne)
        else:
            os.environ["OMP_NUM_THREADS"] = "1"   # ADD THIS before forking
            core._log(1, "[main] OMP_NUM_THREADS set to 1 for parallel sweep")
            _run_sweep_parallel(pts, combos, cfg, outdir, tag, tm, Ne, n_workers)

    elif obs_mode == "multi_obs":
        _run_multi_obs(pts, combos, cfg, outdir, tag, tm, Ne)

    else:
        raise ValueError(
            f"Unknown obs_mode '{obs_mode}'. "
            "Expected: single_obs | sweep | multi_obs")

    core._log(1, f"[{tag}] done  total={time.time()-t_start:.1f}s")


if __name__ == "__main__":
    main()