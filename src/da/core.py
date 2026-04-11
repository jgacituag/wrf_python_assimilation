import os, sys
import numpy as np
import time
#### Verbosity 
# 0=silent  1=method start/finish  2=per-step  3=debug
# Set once per process with set_verbose(level).

_VERBOSE = 1

def set_verbose(level: int):
    """Set verbosity for all DA methods in this process."""
    global _VERBOSE
    _VERBOSE = int(level)

def _log(level: int, msg: str):
    if _VERBOSE >= level:
        print(msg, flush=True)

_cda = None


def _get_cda():
    global _cda
    if _cda is not None:
        return _cda
    for attempt in range(2):
        try:
            from cletkf_wloc import common_da as cda
            _cda = cda
            return _cda
        except ImportError:
            if attempt == 0:
                here = os.path.dirname(os.path.abspath(__file__))
                fort_dir = os.path.normpath(os.path.join(here, "..", "fortran"))
                if fort_dir not in sys.path:
                    sys.path.insert(0, fort_dir)
    raise RuntimeError(
        "Fortran backend (cletkf_wloc) not found. "
        "Run src/build_fortran.sh from the repo root first."
    )


def tempering_schedule(ntemp: int, alpha_s: float) -> np.ndarray:
    """
    Back-loaded exponential weights (Eq. 12).

        alpha_i = exp(-(Nt+1)*alpha_s / i) / sum_j exp(-(Nt+1)*alpha_s / j)

    for i = 1 ... Ntemp.  Weights sum to 1.
    Larger alpha_s -> more back-loading (stronger inflation in early steps).
    alpha_s = 0  -> equal weights = 1/Ntemp for all steps.

    At step i the obs error is inflated to R / alpha_i.
    """
    if ntemp == 1:
        return np.array([1.0], dtype=np.float32)
    i = np.arange(1, ntemp + 1, dtype=np.float64)
    w = np.exp(-(ntemp + 1) * float(alpha_s) / i)
    w /= w.sum()
    return w.astype(np.float32)


def compute_hxf(xf_grid: np.ndarray,
                ox: np.ndarray,
                oy: np.ndarray,
                oz: np.ndarray,
                var_idx: dict) -> np.ndarray:
    """
    Apply the nonlinear reflectivity operator H to every ensemble member
    at every observation location.

    Parameters
    ----------
    xf_grid : (nx, ny, nz, Ne, nvar)
    ox, oy, oz : (nobs,) integer arrays, 0-based
    var_idx    : dict  {"qg":0, "qr":1, "qs":2, "T":3, "P":4, ...}

    Returns
    -------
    hxf : (nobs, Ne), float32
    """
    cda  = _get_cda()
    nobs = len(ox)
    Ne   = xf_grid.shape[3]
    hxf  = np.empty((nobs, Ne), dtype=np.float32, order="F")
    _log(3, f"Computing H(xf) for {nobs} obs and {Ne} ensemble members...")
    for ii in range(nobs):
        i, j, k = int(ox[ii]), int(oy[ii]), int(oz[ii])
        for m in range(Ne):
            hxf[ii, m] = cda.calc_ref(
                xf_grid[i, j, k, m, var_idx["qr"]],
                xf_grid[i, j, k, m, var_idx["qs"]],
                xf_grid[i, j, k, m, var_idx["qg"]],
                xf_grid[i, j, k, m, var_idx["T"]],
                xf_grid[i, j, k, m, var_idx["P"]],
            )
    return hxf


def aoei(yo: np.ndarray,
         hxf: np.ndarray,
         R0: np.ndarray) -> np.ndarray:
    """
    Adaptive Observation Error Inflation (Minamide & Zhang 2017, Eq. 4).

        R_tilde_j = max( R0_j,  d_j^2 - sigma2_f_j )

    where d_j = yo_j - mean(hxf_j) and sigma2_f_j = var(hxf_j, ddof=1).

    Inflation activates when d^2 > R0 + sigma2_f, i.e. the squared
    innovation exceeds the combined obs + background variance.

    Parameters
    ----------
    yo  : (nobs,)      observations
    hxf : (nobs, Ne)   ensemble in obs space
    R0  : (nobs,)      nominal obs error VARIANCE (floor)

    Returns
    -------
    R_tilde : (nobs,), float32, always >= R0
    """
    yo_  = np.asarray(yo,  np.float64)
    hxf_ = np.asarray(hxf, np.float64)
    R0_  = np.asarray(R0,  np.float64)
    d        = yo_ - hxf_.mean(axis=1)
    sigma2_f = hxf_.var(axis=1, ddof=1)
    return np.maximum(R0_, d**2 - sigma2_f).astype(np.float32)


def _letkf_step(xf_grid, hxf, yo, obs_error_var,
                ox_km, oy_km, oz_km, loc_scales_km, pos_km):
    """
    One LETKF analysis via Fortran. Localization is km-based throughout.

    Parameters
    ----------
    xf_grid      : (nx, ny, nz, Ne, nvar)  prior ensemble, float32
    hxf          : (nobs, Ne)               H(xf) ensemble in obs space
    yo           : (nobs,)                  observations
    obs_error_var: (nobs,) or scalar        observation error variance R
    ox_km        : (nobs,) float32          obs x-position [km]
    oy_km        : (nobs,) float32          obs y-position [km]
    oz_km        : (nobs,) float32          obs z-position [km]
    loc_scales_km: (3,) float32             [lx, ly, lz] in km
    pos_km       : (nx, ny, nz, 3) float32  grid-point positions [km]

    Returns
    -------
    xa : (nx, ny, nz, Ne, nvar), float32
    """
    cda = _get_cda()
    nx, ny, nz, Ne, nvar = xf_grid.shape
    nobs = len(yo)

    ox_f   = np.asarray(ox_km,        np.float32)
    oy_f   = np.asarray(oy_km,        np.float32)
    oz_f   = np.asarray(oz_km,        np.float32)
    dep    = (yo - hxf.mean(axis=1)).astype(np.float32)
    oerr_f = np.asarray(obs_error_var, np.float32)
    locs_f = np.asarray(loc_scales_km, np.float32)

    _log(3, f"Running LETKF step: nobs={nobs}  "
            f"oerr_mean={oerr_f.mean():.2f}  locs={locs_f} km")

    t0 = time.time()
    xa_out, n_updated = cda.simple_letkf_wloc(
        nx=nx, ny=ny, nz=nz,
        nbv=Ne, nvar=nvar, nobs=nobs,
        hxf=np.asfortranarray(hxf),
        xf=np.asfortranarray(xf_grid),
        dep=dep,
        ox=ox_f, oy=oy_f, oz=oz_f,
        locs=locs_f, oerr=oerr_f,
        pos_km=np.asfortranarray(pos_km),
    )
    dt = time.time() - t0
    total_pts = nx * ny * nz
    pct_skipped = ((total_pts - n_updated) / total_pts) * 100
    _log(3, f"      [Fortran LETKF] {dt:.3f}s | "
            f"Updated {int(n_updated)}/{total_pts} pts "
            f"(Skipped {pct_skipped:.1f}%)")
    return xa_out.astype(np.float32)


def letkf_update(xf_grid, yo, obs_error_var, ox, oy, oz,
                 loc_scales_km, var_idx, pos_km):
    """
    Standard LETKF: single step, no AOEI, no tempering.

    ox, oy, oz    : (nobs,) int   0-based grid indices
    loc_scales_km : (3,)          [lx, ly, lz] in km
    pos_km        : (nx,ny,nz,3)  grid-point positions [km]

    Returns dict: xa, hxf, dep, obs_error
    """
    R0  = np.asarray(obs_error_var, np.float32)
    hxf = compute_hxf(xf_grid, ox, oy, oz, var_idx)
    idx = (np.asarray(ox, np.intp), np.asarray(oy, np.intp), np.asarray(oz, np.intp))
    xa  = _letkf_step(xf_grid, hxf, yo, R0,
                      pos_km[idx[0], idx[1], idx[2], 0],
                      pos_km[idx[0], idx[1], idx[2], 1],
                      pos_km[idx[0], idx[1], idx[2], 2],
                      loc_scales_km, pos_km)
    return dict(
        xa=xa,
        hxf=hxf,
        dep=(yo - hxf.mean(axis=1)).astype(np.float32),
        obs_error=R0,
    )


def tenkf_update(xf_grid, yo, obs_error_var, ox, oy, oz,
                 loc_scales_km, var_idx, ntemp, alpha_s, pos_km):
    """
    TEnKF (LETKF-T): Ntemp sequential steps with back-loaded inflation.

    ox, oy, oz    : (nobs,) int   0-based grid indices
    loc_scales_km : (3,)          [lx, ly, lz] in km
    pos_km        : (nx,ny,nz,3)  grid-point positions [km]

    Returns dict: xa, hxfs, deps, alpha_weights, obs_error
    """
    t_start = time.time()
    steps = tempering_schedule(ntemp, alpha_s)
    R0    = np.asarray(obs_error_var, np.float32)
    nobs  = len(yo)
    Nt    = len(steps)

    # Extract km positions for each observation once
    idx   = (np.asarray(ox, np.intp), np.asarray(oy, np.intp), np.asarray(oz, np.intp))
    ox_km = pos_km[idx[0], idx[1], idx[2], 0].astype(np.float32)
    oy_km = pos_km[idx[0], idx[1], idx[2], 1].astype(np.float32)
    oz_km = pos_km[idx[0], idx[1], idx[2], 2].astype(np.float32)

    x_af = xf_grid.copy(order="F")
    hxfs = np.empty((Nt, nobs, xf_grid.shape[3]), dtype=np.float32)
    deps = np.empty((Nt, nobs),                    dtype=np.float32)

    _log(3, f"  [TEnKF] Starting {Nt} tempering steps (alpha_s={alpha_s:.2f})")
    for it in range(Nt):
        t_step = time.time()
        hxf = compute_hxf(x_af, ox, oy, oz, var_idx)
        dep = (yo - hxf.mean(axis=1)).astype(np.float32)
        hxfs[it] = hxf
        deps[it]  = dep
        oerr = R0 / steps[it]

        _log(2, f"  [TEnKF]  step {it+1}/{Nt}  alpha={steps[it]:.4f}  "
                f"R/alpha={oerr.mean():.2f}  |dep|={np.abs(dep).mean():.3f}")

        x_af = _letkf_step(x_af, hxf, yo, oerr,
                            ox_km, oy_km, oz_km, loc_scales_km, pos_km)

        _log(3, f"    Step {it+1} complete in {time.time()-t_step:.3f}s")

    _log(3, f"  [TEnKF Total] {Nt} steps in {time.time()-t_start:.3f}s")
    return dict(xa=x_af, hxfs=hxfs, deps=deps,
                alpha_weights=steps, obs_error=R0)


def aoei_update(xf_grid, yo, obs_error_var, ox, oy, oz,
                loc_scales_km, var_idx, pos_km):
    """
    LETKF + AOEI: inflate once from the prior, then one LETKF step.

    ox, oy, oz    : (nobs,) int   0-based grid indices
    loc_scales_km : (3,)          [lx, ly, lz] in km
    pos_km        : (nx,ny,nz,3)  grid-point positions [km]

    Returns dict: xa, hxf, dep, obs_error_raw, obs_error
    """
    t_start = time.time()
    R0  = np.asarray(obs_error_var, np.float32)
    hxf = compute_hxf(xf_grid, ox, oy, oz, var_idx)
    _log(3, f"    [AOEI] H(x) calculated in {time.time()-t_start:.3f}s")

    R_t    = aoei(yo, hxf, R0)
    n_inf  = int((R_t > R0).sum())
    _log(3, f"    [AOEI] Inflated {n_inf}/{len(yo)} obs | "
            f"R_tilde={R_t.mean():.2f} | R0={R0.mean():.2f}")

    idx   = (np.asarray(ox, np.intp), np.asarray(oy, np.intp), np.asarray(oz, np.intp))
    xa    = _letkf_step(xf_grid, hxf, yo, R_t,
                        pos_km[idx[0], idx[1], idx[2], 0],
                        pos_km[idx[0], idx[1], idx[2], 1],
                        pos_km[idx[0], idx[1], idx[2], 2],
                        loc_scales_km, pos_km)

    _log(3, f"  [AOEI Total] cycle finished in {time.time()-t_start:.3f}s")
    return dict(
        xa=xa,
        hxf=hxf,
        dep=(yo - hxf.mean(axis=1)).astype(np.float32),
        obs_error_raw=R0,
        obs_error=R_t,
    )


def atenkf_update(xf_grid, yo, obs_error_var, ox, oy, oz, loc_scales, var_idx,
                  ntemp, alpha_s):
    """
    ATEnKF: TEnKF with AOEI recomputed at every tempering step.

    At each step i: recompute H(x), apply AOEI -> R_tilde,
    then inflate R_tilde / alpha_i, run LETKF.

    Returns
    -------
    dict: xa, xatemp, hxfs, deps, obs_error_aoei, obs_error_eff,
          alpha_weights, obs_error_raw
    """
    steps = tempering_schedule(ntemp, alpha_s)
    R0    = np.asarray(obs_error_var, np.float32)
    nx, ny, nz, Ne, nvar = xf_grid.shape
    nobs = len(yo)
    Nt   = len(steps)

    xatemp         = np.empty((nx, ny, nz, Ne, nvar, Nt + 1), dtype=np.float32, order="F")
    xatemp[..., 0] = xf_grid.astype(np.float32)
    hxfs           = np.empty((Nt, nobs, Ne), dtype=np.float32)
    deps           = np.empty((Nt, nobs),     dtype=np.float32)
    obs_error_aoei = np.empty((Nt, nobs),     dtype=np.float32)
    obs_error_eff  = np.empty((Nt, nobs),     dtype=np.float32)

    for it in range(Nt):
        hxf  = compute_hxf(xatemp[..., it], ox, oy, oz, var_idx)
        dep  = (yo - hxf.mean(axis=1)).astype(np.float32)
        R_t  = aoei(yo, hxf, R0)
        oerr = (R_t / steps[it]).astype(np.float32)
        hxfs[it] = hxf;  deps[it] = dep
        obs_error_aoei[it] = R_t;  obs_error_eff[it] = oerr
        n_inf = int((R_t > R0).sum())
        _log(2, f"  [ATEnKF] step {it+1}/{Nt}  alpha={steps[it]:.4f}  R_eff={oerr.mean():.2f}  AOEI={n_inf}/{nobs}  |dep|={np.abs(dep).mean():.3f}")
        xatemp[..., it + 1] = _letkf_step(
            xatemp[..., it], hxf, yo, oerr, ox, oy, oz, loc_scales)

    return dict(xa=xatemp[..., -1], xatemp=xatemp, hxfs=hxfs, deps=deps,
                obs_error_aoei=obs_error_aoei, obs_error_eff=obs_error_eff,
                alpha_weights=steps, obs_error_raw=R0)


#### ATEnKF helpers #########################################################

def _solve_ntemp(inflation_ratio: float,
                 alpha_s: float,
                 ntemp_max: int = 20) -> int:
    """
    Find the smallest Ntemp such that alpha_1(Ntemp, alpha_s) <= target,
    where target = R0 / R_tilde = 1 / inflation_ratio.

    alpha_1 is the SMALLEST weight (first step, most inflated).
    We want R0 / alpha_1 >= R_tilde, i.e. alpha_1 <= R0 / R_tilde.

    Returns Ntemp in [1, ntemp_max].
    """
    if inflation_ratio <= 1.0 + 1e-6:
        return 1
    target = 1.0 / inflation_ratio          # alpha_1 must be <= this
    for nt in range(1, ntemp_max + 1):
        w = tempering_schedule(nt, alpha_s)
        if w[0] <= target + 1e-9:           # alpha_1 small enough
            return nt
    return ntemp_max


def _per_obs_ntemp(R0: np.ndarray,
                   R_tilde: np.ndarray,
                   alpha_s: float,
                   ntemp_max: int = 20) -> np.ndarray:
    """
    Compute per-observation Ntemp_j from AOEI inflation ratios.

    Parameters
    ----------
    R0      : (nobs,) nominal obs error variance
    R_tilde : (nobs,) AOEI-inflated obs error variance
    alpha_s : tempering slope (default 1.0)
    ntemp_max : cap on Ntemp

    Returns
    -------
    ntemps : (nobs,) int array, each entry in [1, ntemp_max]
    """
    ratios = R_tilde / np.maximum(R0, 1e-30)
    ntemps = np.array([_solve_ntemp(float(r), alpha_s, ntemp_max)
                       for r in ratios], dtype=int)
    return ntemps


def atenkf_update(xf_grid, yo, obs_error_var, ox, oy, oz, loc_scales, var_idx,
                  alpha_s: float = 1.0, ntemp_max: int = 20):
    """
    ATEnKF: locally adaptive tempering guided by AOEI.

    Algorithm
    ---------
    1. Compute H(x^f) from the prior and apply AOEI -> R_tilde_j per obs.
    2. For each obs j compute Ntemp_j: the number of tempering steps needed
       so that the first (most conservative) step uses R_tilde_j:
           R0_j / alpha_1(Ntemp_j) = R_tilde_j
           => alpha_1(Ntemp_j) = R0_j / R_tilde_j
       Obs where AOEI did not fire (R_tilde_j = R0_j) get Ntemp_j = 1.
    3. Global loop for k = 1 ... Ntemp_max:
         For each obs j:
           if k <= Ntemp_j : oerr_j = R0_j / alpha_k(Ntemp_j)
           else             : oerr_j = INF  (obs j is exhausted)
         Recompute H(x) from current ensemble.
         Run one LETKF step with this oerr array.

    Information-preserving property: sum_k alpha_k(Ntemp_j) = 1 for all j,
    so the total information assimilated per obs is 1/R0_j, identical to
    a single standard LETKF step.

    Parameters
    ----------
    xf_grid       : (nx, ny, nz, Ne, nvar)  prior ensemble, float32
    yo            : (nobs,)                 observations
    obs_error_var : (nobs,) or scalar       nominal obs error VARIANCE R0
    ox, oy, oz    : (nobs,) int             obs grid indices, 0-based
    loc_scales    : (3,)                    localization length scales
    var_idx       : dict                    variable index mapping
    alpha_s       : float (default 1.0)     tempering slope
    ntemp_max     : int   (default 20)      maximum tempering steps

    Returns
    -------
    dict with keys:
      xa              (nx,ny,nz,Ne,nvar)       posterior ensemble
      xatemp          (nx,ny,nz,Ne,nvar,Nt+1)  ensemble at each step
      hxfs            (Nt, nobs, Ne)           H(x) at each step
      deps            (Nt, nobs)               innovations at each step
      ntemps_per_obs  (nobs,)                  Ntemp_j per observation
      obs_error_raw   (nobs,)                  nominal R0
      obs_error_aoei  (nobs,)                  AOEI-inflated R_tilde
      oerr_per_step   (Nt, nobs)               effective oerr used at each step
      alpha_s         float
      ntemp_max       int
    """
    INF = np.float32(1e30)
    R0  = np.broadcast_to(np.asarray(obs_error_var, np.float32), len(yo)).copy()

    # Step 1: AOEI from prior
    hxf0   = compute_hxf(xf_grid, ox, oy, oz, var_idx)
    R_tilde = aoei(yo, hxf0, R0)

    # Step 2: per-obs Ntemp
    ntemps = _per_obs_ntemp(R0, R_tilde, alpha_s, ntemp_max)
    Nt_global = int(ntemps.max())

    n_inflated = int((ntemps > 1).sum())
    _log(1, f"  [ATEnKF]  AOEI inflated {n_inflated}/{len(yo)} obs  Nt_global={Nt_global}  alpha_s={alpha_s}")

    # Precompute schedules for each unique Ntemp value
    unique_nts  = np.unique(ntemps)
    schedules   = {nt: tempering_schedule(nt, alpha_s) for nt in unique_nts}

    # Step 3: global tempering loop
    nx, ny, nz, Ne, nvar = xf_grid.shape
    nobs = len(yo)

    xatemp       = np.empty((nx, ny, nz, Ne, nvar, Nt_global + 1),
                             dtype=np.float32, order="F")
    xatemp[..., 0] = xf_grid.astype(np.float32)
    hxfs         = np.empty((Nt_global, nobs, Ne), dtype=np.float32)
    deps         = np.empty((Nt_global, nobs),     dtype=np.float32)
    oerr_per_step = np.empty((Nt_global, nobs),    dtype=np.float32)

    for k in range(Nt_global):           # k = 0 ... Nt_global-1  (step k+1)
        hxf = compute_hxf(xatemp[..., k], ox, oy, oz, var_idx)
        dep = (yo - hxf.mean(axis=1)).astype(np.float32)
        hxfs[k] = hxf;  deps[k] = dep

        # Build oerr for this step
        oerr = np.full(nobs, INF, np.float32)
        for j in range(nobs):
            nt_j = ntemps[j]
            if k < nt_j:                 # step k+1 is within obs j's schedule
                alpha_k = schedules[nt_j][k]
                oerr[j] = R0[j] / alpha_k

        oerr_per_step[k] = oerr
        active = int((oerr < INF).sum())
        _log(2, f"    step {k+1}/{Nt_global}  active_obs={active}  oerr_mean(active)={oerr[oerr < INF].mean():.2f}")

        xatemp[..., k + 1] = _letkf_step(
            xatemp[..., k], hxf, yo, oerr, ox, oy, oz, loc_scales)

    return dict(
        xa=xatemp[..., -1],
        xatemp=xatemp,
        hxfs=hxfs,
        deps=deps,
        ntemps_per_obs=ntemps,
        obs_error_raw=R0,
        obs_error_aoei=R_tilde,
        oerr_per_step=oerr_per_step,
        alpha_s=alpha_s,
        ntemp_max=ntemp_max,
    )


def taoei_update(xf_grid, yo, obs_error_var, ox, oy, oz, loc_scales, var_idx,
                 ntemp: int, alpha_s: float):
    """
    TAOEI: TEnKF with AOEI recomputed at every tempering step.

    At each step i:
      1. Recompute H(x) from the current (updated) ensemble.
      2. Apply AOEI using the current innovation and spread -> R_tilde.
      3. Inflate: effective obs error = R_tilde / alpha_i.
      4. Run one LETKF step.

    Unlike ATEnKF, every observation is active at every step and the
    schedule is fixed (same Ntemp for all obs). AOEI modulates the
    *amount* of inflation per obs and step, but does not determine Ntemp.

    Returns
    -------
    dict with keys:
      xa              (nx,ny,nz,Ne,nvar)
      xatemp          (nx,ny,nz,Ne,nvar, Ntemp+1)
      hxfs            (Ntemp, nobs, Ne)
      deps            (Ntemp, nobs)
      obs_error_aoei  (Ntemp, nobs)    R_tilde at each step
      obs_error_eff   (Ntemp, nobs)    R_tilde / alpha_i
      alpha_weights   (Ntemp,)
      obs_error_raw   (nobs,)          nominal R0
    """
    steps = tempering_schedule(ntemp, alpha_s)
    R0    = np.broadcast_to(np.asarray(obs_error_var, np.float32), len(yo)).copy()
    nx, ny, nz, Ne, nvar = xf_grid.shape
    nobs = len(yo);  Nt = len(steps)

    xatemp         = np.empty((nx, ny, nz, Ne, nvar, Nt + 1),
                               dtype=np.float32, order="F")
    xatemp[..., 0] = xf_grid.astype(np.float32)
    hxfs           = np.empty((Nt, nobs, Ne), dtype=np.float32)
    deps           = np.empty((Nt, nobs),     dtype=np.float32)
    obs_error_aoei = np.empty((Nt, nobs),     dtype=np.float32)
    obs_error_eff  = np.empty((Nt, nobs),     dtype=np.float32)

    for it in range(Nt):
        hxf  = compute_hxf(xatemp[..., it], ox, oy, oz, var_idx)
        dep  = (yo - hxf.mean(axis=1)).astype(np.float32)
        R_t  = aoei(yo, hxf, R0)
        oerr = (R_t / steps[it]).astype(np.float32)
        hxfs[it] = hxf;  deps[it] = dep
        obs_error_aoei[it] = R_t;  obs_error_eff[it] = oerr
        n_inf = int((R_t > R0).sum())
        _log(2, f"  [TAOEI]  step {it+1}/{Nt}  alpha={steps[it]:.4f}  R_eff_mean={oerr.mean():.2f}  AOEI_inf={n_inf}/{nobs}  |dep|={np.abs(dep).mean():.3f}")
        xatemp[..., it + 1] = _letkf_step(
            xatemp[..., it], hxf, yo, oerr, ox, oy, oz, loc_scales)

    return dict(xa=xatemp[..., -1], xatemp=xatemp, hxfs=hxfs, deps=deps,
                obs_error_aoei=obs_error_aoei, obs_error_eff=obs_error_eff,
                alpha_weights=steps, obs_error_raw=R0)