"""
src/runners/run_experiment.py
==============================
Unified experiment runner for all WS experiments.

The YAML config controls everything: experiment type, obs construction,
sweep parameters, QC, methods, parallelism, and verbosity.

Usage
-----
  python src/runners/run_experiment.py --config configs/ws1.yaml
  python src/runners/run_experiment.py --config configs/ws2.yaml --workers 30
  python src/runners/run_experiment.py --config configs/ws2.yaml --verbose 1

Obs modes (set via sweep.obs_points in config)
-----------------------------------------------
  single              one fixed point -> one LETKF call per combo
  full_grid           every QC-passing point as independent single obs (WS-2)
  strided: N          all QC-passing points on ::N grid, one LETKF call
  all                 all QC-passing points, one LETKF call

Sweep dimensions (all support scalar, list, or {start, stop, num})
-------------------------------------------------------------------
  truth_members, prior_size, methods, ntemp, alpha_s, loc_x, loc_y, loc_z

Output files
------------
  Single-obs:
    {tag}_{method}_Nt{nt}_as{as}_Lx{lx}Ly{ly}Lz{lz}_Ne{ne}_obs{x}_{y}_{z}_qc{qc}_True{tm}.npz
  Multi-obs:
    {tag}_{method}_Nt{nt}_as{as}_Lx{lx}Ly{ly}Lz{lz}_Ne{ne}_str{stride}_qc{qc}_True{tm}.npz

  Config copy: {outdir}/{tag}_config.yaml  (written before any results)
"""

import argparse
import itertools
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
from da.core import (
    letkf_update, tenkf_update, aoei_update,
    atenkf_update, taoei_update,
)

def _get_cda():
    """Lazy import of the Fortran module — fails clearly if not built."""
    from cletkf_wloc import common_da as cda
    return cda


# ## sweep parameter helpers ################################################

def _expand(val, is_int=False):
    """
    Expand a sweep parameter to a flat list.
      scalar       -> [scalar]
      [a, b, c]    -> [a, b, c]
      {start, stop, num}  -> linspace (inclusive both ends)
                             if is_int: rounded to unique ints
    """
    if isinstance(val, dict):
        import numpy as np
        arr = np.linspace(val["start"], val["stop"], int(val["num"]))
        if is_int:
            arr = np.unique(np.round(arr).astype(int))
        return arr.tolist()
    if isinstance(val, list):
        return val
    return [val]


def _expand_loc(cfg_val):
    """
    Expand a localization axis value.
    Accepts scalar, list, or {start,stop,num}.
    9999 or null/None means no localization.
    """
    vals = _expand(cfg_val if cfg_val is not None else 9999)
    return [float(v) for v in vals]


def _qc_code(qc_cfg):
    """Short code for QC settings used in filenames."""
    if not qc_cfg:
        return "none"
    fe = qc_cfg.get("filter_ensemble", True)
    ft = qc_cfg.get("filter_truth",    False)
    mode = qc_cfg.get("filter_mode", "and")
    if fe and ft:
        return f"ET_{mode}"
    if fe:
        return "E"
    if ft:
        return "T"
    return "none"


def _qc_pass(yo_val, hxf_mean_val, qc_cfg):
    if not qc_cfg:
        return True
    dbz_min = float(qc_cfg.get("dbz_min", 5.0))
    fe   = bool(qc_cfg.get("filter_ensemble", True))
    ft   = bool(qc_cfg.get("filter_truth",    False))
    mode = qc_cfg.get("filter_mode", "and").lower()
    fail_e = fe and (float(hxf_mean_val) < dbz_min)
    fail_t = ft and (float(yo_val)       < dbz_min)
    if fe and ft:
        return not (fail_e and fail_t) if mode == "or" \
               else not (fail_e or fail_t)
    return not fail_e if fe else not fail_t


# ## observation operator ###################################################

def _calc_hx_domain(state_nvar, var_idx):
    """
    Compute reflectivity H(x) over the full 3D domain using the
    vectorised Fortran calc_ref_ens — one call instead of nx*ny*nz loops.

    state_nvar : (nx,ny,nz,nvar)       single member
               | (nx,ny,nz,Ne,nvar)    ensemble
    Returns    : (nx,ny,nz)            or (nx,ny,nz,Ne)  float32
    """
    from cletkf_wloc import common_da as cda
    vi   = var_idx
    ndim = state_nvar.ndim

    if ndim == 4:                          # single member (nx,ny,nz,nvar)
        nx, ny, nz, _ = state_nvar.shape
        # wrap as nbv=1 ensemble, call, then squeeze
        s = state_nvar[:, :, :, np.newaxis, :]   # (nx,ny,nz,1,nvar)
        ref = cda.calc_ref_ens(
            s[:,:,:,:,vi["qr"]].astype(np.float64),
            s[:,:,:,:,vi["qs"]].astype(np.float64),
            s[:,:,:,:,vi["qg"]].astype(np.float64),
            s[:,:,:,:,vi["T"] ].astype(np.float64),
            s[:,:,:,:,vi["P"] ].astype(np.float64),
        )                                  # (nx,ny,nz,1)
        return ref[:, :, :, 0].astype(np.float32)

    else:                                  # ensemble (nx,ny,nz,Ne,nvar)
        ref = cda.calc_ref_ens(
            state_nvar[:,:,:,:,vi["qr"]].astype(np.float64),
            state_nvar[:,:,:,:,vi["qs"]].astype(np.float64),
            state_nvar[:,:,:,:,vi["qg"]].astype(np.float64),
            state_nvar[:,:,:,:,vi["T"] ].astype(np.float64),
            state_nvar[:,:,:,:,vi["P"] ].astype(np.float64),
        )                                  # (nx,ny,nz,Ne)
        return ref.astype(np.float32)


def _hx_point(state_nvar, i, j, k, var_idx):
    """H(x) at one grid point. state_nvar: (nvar,) or (Ne, nvar)."""
    from cletkf_wloc import common_da as cda
    vi = var_idx
    if state_nvar.ndim == 1:
        return float(cda.calc_ref(state_nvar[vi["qr"]], state_nvar[vi["qs"]],
                                  state_nvar[vi["qg"]], state_nvar[vi["T"]],
                                  state_nvar[vi["P"]]))
    return np.array([cda.calc_ref(state_nvar[m,vi["qr"]], state_nvar[m,vi["qs"]],
                                  state_nvar[m,vi["qg"]], state_nvar[m,vi["T"]],
                                  state_nvar[m,vi["P"]])
                     for m in range(state_nvar.shape[0])], np.float32)


# ## method dispatcher ######################################################

def _run_method(method, xf, yo, R0, ox, oy, oz, loc_scales, var_idx, ntemp, alpha_s):
    if method == "LETKF":
        return letkf_update(xf, yo, R0, ox, oy, oz, loc_scales, var_idx)
    if method == "TEnKF":
        return tenkf_update(xf, yo, R0, ox, oy, oz, loc_scales, var_idx,
                            ntemp=ntemp, alpha_s=alpha_s)
    if method == "AOEI":
        return aoei_update(xf, yo, R0, ox, oy, oz, loc_scales, var_idx)
    if method == "ATEnKF":
        return atenkf_update(xf, yo, R0, ox, oy, oz, loc_scales, var_idx,
                             alpha_s=alpha_s)
    if method == "TAOEI":
        return taoei_update(xf, yo, R0, ox, oy, oz, loc_scales, var_idx,
                            ntemp=ntemp, alpha_s=alpha_s)
    raise ValueError(f"Unknown method: {method}")


# ## filename builders ######################################################

def _fmt(v):
    """Format a float for filenames: no trailing zeros."""
    return f"{v:.2f}".rstrip("0").rstrip(".")


def _select_prior(ens, tm, prior_size=None):
    """
    Split ensemble into truth and prior.

    Parameters
    ----------
    ens        : (nx, ny, nz, Ne_tot, nvar)
    tm         : index of the truth member
    prior_size : number of prior members to use (None = all remaining,
                 sequential starting from index 0, skipping tm)

    Returns
    -------
    truth : (nx, ny, nz, nvar)
    xf    : (nx, ny, nz, Ne, nvar)  float32
    Ne    : int  actual prior ensemble size used
    """
    Ne_tot     = ens.shape[3]
    all_others = [i for i in range(Ne_tot) if i != tm]
    if prior_size is not None:
        if prior_size > len(all_others):
            raise ValueError(
                f"prior_size={prior_size} exceeds available members "
                f"({len(all_others)}) for truth member {tm}."
            )
        all_others = all_others[:prior_size]
    truth = ens[:, :, :, tm, :]
    xf    = np.asfortranarray(ens[:, :, :, all_others, :].astype(np.float32))
    return truth, xf, len(all_others)


def _fname_single(tag, method, ntemp, alpha_s, lx, ly, lz,
                  ne, ox, oy, oz, qc, tm):
    return (f"{tag}_{method}_Nt{ntemp:02d}_as{_fmt(alpha_s)}"
            f"_Lx{_fmt(lx)}Ly{_fmt(ly)}Lz{_fmt(lz)}"
            f"_Ne{ne:03d}_obs{ox}_{oy}_{oz}_qc{qc}_True{tm:02d}.npz")


def _fname_multi(tag, method, ntemp, alpha_s, lx, ly, lz,
                 ne, stride_str, qc, tm):
    return (f"{tag}_{method}_Nt{ntemp:02d}_as{_fmt(alpha_s)}"
            f"_Lx{_fmt(lx)}Ly{_fmt(ly)}Lz{_fmt(lz)}"
            f"_Ne{ne:03d}_str{stride_str}_qc{qc}_True{tm:02d}.npz")


# ## per-truth-member worker ################################################

def _worker(args):
    (tm, ens, cfg, verbose) = args
    core.set_verbose(verbose)

    sweep    = cfg["sweep"]
    qc_cfg   = cfg.get("qc", {})
    var_idx  = cfg["state"]["var_idx"]
    outdir   = cfg["paths"]["outdir"]
    tag      = cfg.get("experiment_tag", "EXP")
    R0_val   = float(cfg["obs"]["obs_error_var"])
    qc       = _qc_code(qc_cfg)

    obs_cfg  = sweep["obs_points"]
    if isinstance(obs_cfg, str):
        obs_mode = obs_cfg           # "full_grid" or "all"
        obs_loc  = None
        stride   = 1
    elif isinstance(obs_cfg, dict):
        obs_mode = obs_cfg.get("mode", "single")
        obs_loc  = obs_cfg.get("loc", None)      # {x,y,z} for single
        stride   = int(obs_cfg.get("stride", 2)) # for strided

    methods     = _expand(sweep.get("methods",    ["LETKF"]))
    ntemps      = _expand(sweep.get("ntemp",      [1]), is_int=True)
    alphas      = _expand(sweep.get("alpha_s",    [2.0]))
    lxs         = _expand_loc(sweep.get("loc_x",  5.0))
    lys         = _expand_loc(sweep.get("loc_y",  5.0))
    lzs         = _expand_loc(sweep.get("loc_z",  5.0))
    prior_sizes = _expand(sweep["prior_size"], is_int=True) \
                  if sweep.get("prior_size") is not None else [None]

    # ── truth H(x): fixed for this tm across all prior sizes ────────────
    t0 = time.time()
    truth_base = ens[:, :, :, tm, :]
    nx, ny, nz = truth_base.shape[:3]
    core._log(1, f"[truth {tm:02d}] start  domain={nx}x{ny}x{nz}"
                 f"  prior_sizes={prior_sizes}  mode={obs_mode}")

    _t = time.time()
    truth_hx = _calc_hx_domain(truth_base, var_idx)   # (nx,ny,nz)
    core._log(2, f"  [truth {tm:02d}] H(truth) done  {time.time()-_t:.1f}s"
                 f"  ({nx*ny*nz} pts)")

    combos = list(itertools.product(methods, ntemps, alphas, lxs, lys, lzs))
    core._log(2, f"  [truth {tm:02d}] {len(combos)} combo(s) to run")
    saved  = []

    for prior_size in prior_sizes:

        # ── prior ensemble and its H(x) ─────────────────────────────────
        _, xf, Ne = _select_prior(ens, tm, prior_size)

        _t = time.time()
        ens_hx   = _calc_hx_domain(xf, var_idx)   # (nx,ny,nz,Ne)
        ens_mean = ens_hx.mean(axis=3)             # (nx,ny,nz)
        core._log(2, f"  [truth {tm:02d}] H(xf) Ne={Ne} done  {time.time()-_t:.1f}s"
                     f"  ({nx*ny*nz*Ne} pts)")

        # ── build QC-passing obs list ────────────────────────────────────
        _t = time.time()
        if obs_mode == "single":
            i0, j0, k0 = int(obs_loc["x"]), int(obs_loc["y"]), int(obs_loc["z"])
            obs_sets = [(
                np.array([i0], np.int32),
                np.array([j0], np.int32),
                np.array([k0], np.int32),
                np.array([truth_hx[i0,j0,k0]], np.float32),
                f"{i0}_{j0}_{k0}",
            )]
            core._log(2, f"  [truth {tm:02d}] obs: single ({i0},{j0},{k0})"
                         f"  yo={truth_hx[i0,j0,k0]:.2f} dBZ")

        elif obs_mode == "full_grid":
            obs_sets = []
            for i in range(nx):
                for j in range(ny):
                    for k in range(nz):
                        if _qc_pass(truth_hx[i,j,k], ens_mean[i,j,k], qc_cfg):
                            obs_sets.append((
                                np.array([i], np.int32),
                                np.array([j], np.int32),
                                np.array([k], np.int32),
                                np.array([truth_hx[i,j,k]], np.float32),
                                f"{i}_{j}_{k}",
                            ))
            core._log(1, f"  [truth {tm:02d}] Ne={Ne}  full_grid: "
                         f"{len(obs_sets)}/{nx*ny*nz} pts pass QC"
                         f"  ({time.time()-_t:.1f}s)")

        else:  # "strided" or "all"
            s = stride if obs_mode == "strided" else 1
            ix, iy, iz, yo_vals = [], [], [], []
            for i in range(0, nx, s):
                for j in range(0, ny, s):
                    for k in range(0, nz, s):
                        if _qc_pass(truth_hx[i,j,k], ens_mean[i,j,k], qc_cfg):
                            ix.append(i); iy.append(j); iz.append(k)
                            yo_vals.append(truth_hx[i,j,k])
            obs_sets = [(
                np.array(ix, np.int32),
                np.array(iy, np.int32),
                np.array(iz, np.int32),
                np.array(yo_vals, np.float32),
                f"str{s}",
            )]
            core._log(1, f"  [truth {tm:02d}] Ne={Ne}  {obs_mode}: "
                         f"{len(ix)}/{nx*ny*nz} pts pass QC"
                         f"  ({time.time()-_t:.1f}s)")

        if not obs_sets:
            core._log(1, f"  [truth {tm:02d}] Ne={Ne}  no valid obs, skipping")
            continue

        # ── sweep over parameter combos ──────────────────────────────────
        for (ox_arr, oy_arr, oz_arr, yo_arr, loc_str) in obs_sets:
            R0 = np.full(len(yo_arr), R0_val, np.float32)

            for (method, ntemp, alpha_s, lx, ly, lz) in combos:
                loc_scales = np.array([lx, ly, lz], np.float32)

                if obs_mode in ("single", "full_grid"):
                    fname = _fname_single(tag, method, ntemp, alpha_s,
                                          lx, ly, lz, Ne,
                                          ox_arr[0], oy_arr[0], oz_arr[0],
                                          qc, tm)
                else:
                    fname = _fname_multi(tag, method, ntemp, alpha_s,
                                         lx, ly, lz, Ne, loc_str, qc, tm)

                out_path = os.path.join(outdir, fname)
                if os.path.exists(out_path):
                    if cfg.get("skip_existing", False):
                        core._log(2, f"  [skip] {fname}")
                        saved.append(fname)
                        continue

                core._log(2, f"  {method} Ne={Ne} Nt={ntemp} as={alpha_s} "
                             f"L=[{lx},{ly},{lz}] obs={loc_str}")

                _t = time.time()
                res = _run_method(method, xf, yo_arr, R0,
                                  ox_arr, oy_arr, oz_arr,
                                  loc_scales, var_idx, ntemp, alpha_s)
                core._log(2, f"    DA done  {time.time()-_t:.2f}s")

                _t = time.time()
                hxf_pt = ens_hx[ox_arr, oy_arr, oz_arr, :]  # (nobs, Ne)

                if obs_mode in ("single", "full_grid"):
                    # ── single-obs: scalars only, xa discarded ───────────
                    # The full analysis (nx,ny,nz,Ne,nvar) is computed by
                    # the Fortran but we extract only what we need from it
                    # and immediately free it. Nothing > (nvar,) is stored.
                    xa      = res["xa"]                   # (nx,ny,nz,Ne,nvar)
                    xa_mean = xa.mean(axis=3)             # (nx,ny,nz,nvar)
                    i0, j0, k0 = int(ox_arr[0]), int(oy_arr[0]), int(oz_arr[0])
                    cda = _get_cda()
                    vi  = var_idx

                    # H(xa) at the obs point for each ensemble member
                    hxa_pt = np.array([
                        cda.calc_ref(
                            xa[i0,j0,k0,m,vi["qr"]], xa[i0,j0,k0,m,vi["qs"]],
                            xa[i0,j0,k0,m,vi["qg"]], xa[i0,j0,k0,m,vi["T"]],
                            xa[i0,j0,k0,m,vi["P"]],
                        ) for m in range(Ne)
                    ], dtype=np.float32)

                    save = dict(
                        # observation identity
                        obs_x=np.int32(i0),
                        obs_y=np.int32(j0),
                        obs_z=np.int32(k0),
                        # obs-space scalars at the obs point
                        yo=np.float32(yo_arr[0]),
                        hxf_mean_obs=np.float32(hxf_pt.mean()),
                        hxa_mean_obs=np.float32(hxa_pt.mean()),
                        spread_f_obs=np.float32(hxf_pt.std(ddof=1)),
                        spread_a_obs=np.float32(hxa_pt.std(ddof=1)),
                        dep_b=np.float32(yo_arr[0] - hxf_pt.mean()),
                        dep_a=np.float32(yo_arr[0] - hxa_pt.mean()),
                        obs_error=np.float32(
                            res.get("obs_error", res.get("obs_error_raw", R0))[0]),
                        # state-space point values (nvar,) each
                        xf_mean_pt=xf[i0,j0,k0,:,:].mean(axis=0).astype(np.float32),
                        xa_mean_pt=xa_mean[i0,j0,k0,:].astype(np.float32),
                        truth_pt=truth_base[i0,j0,k0,:].astype(np.float32),
                        # variable name index for reading the (nvar,) arrays
                        var_names=np.array(list(var_idx.keys())),
                        # metadata
                        truth_member=np.int32(tm),
                        Ne=np.int32(Ne),
                    )
                    # per-tempering-step departures — cheap, useful diagnostic
                    if "deps" in res:
                        save["deps"] = res["deps"].astype(np.float32)
                    if "alpha_weights" in res:
                        save["alpha_weights"] = res["alpha_weights"]

                    del xa, xa_mean   # free immediately

                else:
                    # ── multi-obs: full matrices stored ──────────────────
                    save = dict(
                        xa=res["xa"],
                        yo=yo_arr,
                        hxf_mean=hxf_pt.mean(axis=1).astype(np.float32),
                        dep=(yo_arr - hxf_pt.mean(axis=1)).astype(np.float32),
                        spread=hxf_pt.std(axis=1, ddof=1).astype(np.float32),
                        obs_error=np.asarray(res.get("obs_error",
                                             res.get("obs_error_raw", R0)),
                                             np.float32),
                        ox=ox_arr, oy=oy_arr, oz=oz_arr,
                        truth_member=np.int32(tm),
                        Ne=np.int32(Ne),
                    )
                    for key in ("xatemp","hxfs","deps","alpha_weights",
                                "ntemps_per_obs","obs_error_aoei","obs_error_eff"):
                        if key in res:
                            save[key] = res[key]

                np.savez_compressed(out_path, **save)
                _sz = os.path.getsize(out_path) / 1e3
                core._log(2, f"    saved {_sz:.1f} KB  {time.time()-_t:.2f}s"
                             f"  -> {fname}")
                saved.append(fname)

    elapsed = time.time() - t0
    core._log(1, f"[truth {tm:02d}] done  {len(saved)} files  {elapsed:.1f}s")
    return saved


# ## main ###################################################################

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config",  required=True)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--verbose", type=int, default=None)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    verbose   = args.verbose if args.verbose is not None \
                else int(cfg.get("verbose", 1))
    outdir    = cfg["paths"]["outdir"]
    tag       = cfg.get("experiment_tag", "EXP")
    n_members = int(cfg["state"].get("n_members", 30))

    os.makedirs(outdir, exist_ok=True)

    # copy config to output dir before anything else
    config_copy = os.path.join(outdir, f"{tag}_config.yaml")
    shutil.copy2(args.config, config_copy)
    print(f"[info] config saved -> {config_copy}")

    # expand truth members
    tm_cfg = cfg["sweep"].get("truth_members", "all")
    if tm_cfg == "all":
        truth_members = list(range(n_members))
    else:
        truth_members = _expand(tm_cfg, is_int=True)

    # load ensemble once in main process
    data = np.load(cfg["paths"]["prepared"])
    ens  = data["state_ensemble"] if "state_ensemble" in data \
           else data["cross_sections"]
    
    print(f"[{tag}] Optimizing memory layout for fortran logic...")
    ens = np.asfortranarray(ens.astype(np.float32))

    n_workers   = args.workers or len(truth_members)
    worker_args = [(tm, ens, cfg, verbose) for tm in truth_members]

    t_start = time.time()
    print(f"[{tag}] {len(truth_members)} truth members  "
          f"workers={n_workers}  verbose={verbose}")

    if n_workers == 1:
        all_saved = []
        for a in worker_args:
            all_saved.extend(_worker(a))
    else:
        with Pool(processes=n_workers) as pool:
            results = pool.map(_worker, worker_args)
        all_saved = [f for r in results for f in r]

    elapsed = time.time() - t_start
    print(f"[{tag}] done  {len(all_saved)} files  total={elapsed:.1f}s")


if __name__ == "__main__":
    main()