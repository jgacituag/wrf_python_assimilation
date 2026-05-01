"""
src/da/metrics.py
=================
All metric computation for single-obs and multi-obs experiments.

compute_single_obs_metrics(...)
    Called once per (point × combo) from _process_point.
    All H(x) arrays are precomputed by the caller — no Fortran calls here.
    Returns a flat dict of scalars ready to be stacked into npz rows.

compute_multi_obs_metrics(...)
    Called once per combo from _run_multi_obs.
    Returns a dict ready to be passed to np.savez_compressed.

Naming convention
-----------------
  _f        : forecast (prior)
  _a        : analysis (posterior)
  _obs      : in observation space (reflectivity, at obs point)
  _obs_*    : in observation space aggregated over the localization zone
  _w        : rho-weighted (Gaussian localization weight)
  _u        : unweighted, uniform over updated points (rho > 0)
  _local    : aggregated over the localization zone
  _pt       : value at the obs point location (single grid point)
  {v}       : state variable name (e.g. qr, qs, qg, T, P, u, v, w)

rho convention
--------------
  rho = 0   outside compact-support cutoff
  rho > 0   inside cutoff (Gaussian value)
  mask      rho > 0  (boolean)
"""

import numpy as np


# ---------------------------------------------------------------------------
# Internal helpers — work with rho=0 outside (no NaN convention)
# ---------------------------------------------------------------------------

def _wmean(field, rho):
    """Weighted mean: sum(rho * field) / sum(rho). Returns nan if sum==0."""
    w_sum = float(rho.sum())
    if w_sum == 0.0:
        return np.nan
    return float((rho * field).sum() / w_sum)


def _weighted_rmse(err_field, rho):
    """sqrt( sum(rho * err^2) / sum(rho) )"""
    w_sum = float(rho.sum())
    if w_sum == 0.0:
        return np.nan
    return float(np.sqrt((rho * err_field ** 2).sum() / w_sum))


def _weighted_bias(err_field, rho):
    """sum(rho * err) / sum(rho)  where err = mean - truth"""
    return _wmean(err_field, rho)


def _weighted_spread(std_field, rho):
    """sqrt( sum(rho * std^2) / sum(rho) )"""
    w_sum = float(rho.sum())
    if w_sum == 0.0:
        return np.nan
    return float(np.sqrt((rho * std_field ** 2).sum() / w_sum))


def _unweighted_rmse(err_field, mask):
    """sqrt( mean(err^2) ) over mask."""
    n = int(mask.sum())
    if n == 0:
        return np.nan
    return float(np.sqrt((err_field[mask] ** 2).mean()))


def _unweighted_bias(err_field, mask):
    """mean(err) over mask."""
    n = int(mask.sum())
    if n == 0:
        return np.nan
    return float(err_field[mask].mean())


def _unweighted_spread(std_field, mask):
    """sqrt( mean(std^2) ) over mask."""
    n = int(mask.sum())
    if n == 0:
        return np.nan
    return float(np.sqrt((std_field[mask] ** 2).mean()))


# ---------------------------------------------------------------------------
# Single-obs metrics
# ---------------------------------------------------------------------------

def compute_single_obs_metrics(
        xf_sub,        # (nx_s, ny_s, nz_s, Ne, nvar)  prior ensemble
        xa_sub,        # (nx_s, ny_s, nz_s, Ne, nvar)  posterior ensemble
        truth_sub,     # (nx_s, ny_s, nz_s, nvar)       truth state
        ens_hx_sub,    # (nx_s, ny_s, nz_s, Ne)          H(xf) ensemble — precomputed
        hxa_sub,       # (nx_s, ny_s, nz_s, Ne)          H(xa) ensemble — precomputed
        truth_hx_sub,  # (nx_s, ny_s, nz_s)              H(truth) — precomputed
        rho,           # (nx_s, ny_s, nz_s)              localization weights (0 outside)
        ox_s, oy_s, oz_s,  # int, obs position within subdomain (0-based)
        yo,            # scalar, observed value (with noise)
        var_names,     # list[str], state variable names in index order
) -> dict:
    """
    Compute all metrics for one (point × combo) single-observation experiment.

    No Fortran calls — all H(x) arrays must be precomputed by the caller.

    Returns a flat dict of float scalars. Keys are described below.

    Obs-point scalars
    -----------------
    hxf_mean_obs      prior ensemble mean H(x) at obs point
    hxa_mean_obs      analysis ensemble mean H(x) at obs point
    spread_f_obs      prior ensemble spread H(x) at obs point
    spread_a_obs      analysis ensemble spread H(x) at obs point
    dep_b             yo - hxf_mean_obs  (innovation)
    dep_a             yo - hxa_mean_obs  (residual)
    inc_obs           hxa_mean_obs - hxf_mean_obs  (analysis increment in obs space)

    Localization zone
    -----------------
    n_updated         number of grid points with rho > 0
    loc_weights_sum   sum of rho over subdomain

    Environment (characterises surrounding storm)
    ---------------------------------------------
    truth_hx_mean_local   rho-weighted mean H(truth) in zone
    truth_hx_max_local    max H(truth) inside rho > 0
    truth_hx_std_local    std H(truth) inside rho > 0
    hxf_mean_local        rho-weighted mean of ensemble-mean H(xf) in zone
    hxf_spread_local      rho-weighted mean ensemble spread H(xf) in zone

    Per state variable (replace {v} with variable name)
    -------------------------------------------------------
    rmse_f_w_{v}, rmse_a_w_{v}    rho-weighted RMSE
    bias_f_w_{v}, bias_a_w_{v}    rho-weighted bias  (mean - truth)
    spread_f_w_{v}, spread_a_w_{v} rho-weighted spread
    rmse_f_u_{v}, rmse_a_u_{v}    unweighted RMSE  (over rho>0 points)
    bias_f_u_{v}, bias_a_u_{v}    unweighted bias
    spread_f_u_{v}, spread_a_u_{v} unweighted spread
    xf_mean_pt_{v}                 prior ensemble mean at obs point
    xa_mean_pt_{v}                 analysis ensemble mean at obs point
    truth_pt_{v}                   truth value at obs point

    Obs-space zone metrics (reflectivity over localization zone)
    ------------------------------------------------------------
    rmse_f_obs_w, rmse_a_obs_w    rho-weighted RMSE of H(x) field
    bias_f_obs_w, bias_a_obs_w    rho-weighted bias of H(x) field
    rmse_f_obs_u, rmse_a_obs_u    unweighted RMSE  (over rho>0 points)
    bias_f_obs_u, bias_a_obs_u    unweighted bias
    """

    mask = rho > 0   # (nx_s, ny_s, nz_s) bool

    # ---- ensemble means and stds -------------------------------------------
    xf_mean = xf_sub.mean(axis=3)          # (nx_s, ny_s, nz_s, nvar)
    xa_mean = xa_sub.mean(axis=3)
    xf_std  = xf_sub.std(axis=3, ddof=1)
    xa_std  = xa_sub.std(axis=3, ddof=1)

    hxf_mean = ens_hx_sub.mean(axis=3)    # (nx_s, ny_s, nz_s)
    hxa_mean = hxa_sub.mean(axis=3)
    hxf_std  = ens_hx_sub.std(axis=3, ddof=1)
    hxa_std  = hxa_sub.std(axis=3, ddof=1)

    # ---- obs point metrics -------------------------------------------------
    hxf_mean_obs   = float(ens_hx_sub[ox_s, oy_s, oz_s, :].mean())
    hxf_spread_obs = float(ens_hx_sub[ox_s, oy_s, oz_s, :].std(ddof=1))
    hxa_mean_obs   = float(hxa_sub[ox_s, oy_s, oz_s, :].mean())
    hxa_spread_obs = float(hxa_sub[ox_s, oy_s, oz_s, :].std(ddof=1))

    dep_b   = float(yo) - hxf_mean_obs
    dep_a   = float(yo) - hxa_mean_obs
    inc_obs = hxa_mean_obs - hxf_mean_obs

    # ---- localization zone summary -----------------------------------------
    n_updated    = int(mask.sum())
    loc_wsum     = float(rho.sum())

    # ---- environment metrics -----------------------------------------------
    truth_hx_mean_local = _wmean(truth_hx_sub, rho)
    truth_hx_max_local  = float(truth_hx_sub[mask].max()) if n_updated > 0 else np.nan
    truth_hx_std_local  = float(truth_hx_sub[mask].std()) if n_updated > 0 else np.nan
    hxf_mean_local      = _wmean(hxf_mean, rho)
    hxf_spread_local    = _weighted_spread(hxf_std, rho)

    # ---- obs-space zone error fields ---------------------------------------
    err_f_obs = hxf_mean - truth_hx_sub   # (nx_s, ny_s, nz_s)
    err_a_obs = hxa_mean - truth_hx_sub

    out = dict(
        # obs point
        hxf_mean_obs    = hxf_mean_obs,
        hxa_mean_obs    = hxa_mean_obs,
        spread_f_obs    = hxf_spread_obs,
        spread_a_obs    = hxa_spread_obs,
        dep_b           = dep_b,
        dep_a           = dep_a,
        inc_obs         = inc_obs,
        # localization zone
        n_updated       = float(n_updated),   # float so npz stacking works uniformly
        loc_weights_sum = loc_wsum,
        # environment
        truth_hx_mean_local = truth_hx_mean_local,
        truth_hx_max_local  = truth_hx_max_local,
        truth_hx_std_local  = truth_hx_std_local,
        hxf_mean_local      = hxf_mean_local,
        hxf_spread_local    = hxf_spread_local,
        # obs-space zone — weighted
        rmse_f_obs_w = _weighted_rmse(err_f_obs, rho),
        rmse_a_obs_w = _weighted_rmse(err_a_obs, rho),
        bias_f_obs_w = _weighted_bias(err_f_obs, rho),
        bias_a_obs_w = _weighted_bias(err_a_obs, rho),
        # obs-space zone — unweighted
        rmse_f_obs_u = _unweighted_rmse(err_f_obs, mask),
        rmse_a_obs_u = _unweighted_rmse(err_a_obs, mask),
        bias_f_obs_u = _unweighted_bias(err_f_obs, mask),
        bias_a_obs_u = _unweighted_bias(err_a_obs, mask),
    )

    # ---- per state variable ------------------------------------------------
    for iv, vname in enumerate(var_names):
        err_f = xf_mean[..., iv] - truth_sub[..., iv]
        err_a = xa_mean[..., iv] - truth_sub[..., iv]
        sf    = xf_std[..., iv]
        sa    = xa_std[..., iv]

        out[f"rmse_f_w_{vname}"]   = _weighted_rmse(err_f, rho)
        out[f"rmse_a_w_{vname}"]   = _weighted_rmse(err_a, rho)
        out[f"bias_f_w_{vname}"]   = _weighted_bias(err_f, rho)
        out[f"bias_a_w_{vname}"]   = _weighted_bias(err_a, rho)
        out[f"spread_f_w_{vname}"] = _weighted_spread(sf,  rho)
        out[f"spread_a_w_{vname}"] = _weighted_spread(sa,  rho)

        out[f"rmse_f_u_{vname}"]   = _unweighted_rmse(err_f, mask)
        out[f"rmse_a_u_{vname}"]   = _unweighted_rmse(err_a, mask)
        out[f"bias_f_u_{vname}"]   = _unweighted_bias(err_f, mask)
        out[f"bias_a_u_{vname}"]   = _unweighted_bias(err_a, mask)
        out[f"spread_f_u_{vname}"] = _unweighted_spread(sf, mask)
        out[f"spread_a_u_{vname}"] = _unweighted_spread(sa, mask)

        out[f"xf_mean_pt_{vname}"] = float(xf_mean[ox_s, oy_s, oz_s, iv])
        out[f"xa_mean_pt_{vname}"] = float(xa_mean[ox_s, oy_s, oz_s, iv])
        out[f"truth_pt_{vname}"]   = float(truth_sub[ox_s, oy_s, oz_s, iv])

    return out


# ---------------------------------------------------------------------------
# Multi-obs metrics
# ---------------------------------------------------------------------------

def compute_multi_obs_metrics(
        xa,                # (nx, ny, nz, Ne, nvar)  analysis ensemble
        xf,                # (nx, ny, nz, Ne, nvar)  prior ensemble
        truth,             # (nx, ny, nz, nvar)        truth state
        hxf_mean_field,    # (nx, ny, nz)              H(xf_mean) — precomputed
        hxa_mean_field,    # (nx, ny, nz)              H(xa_mean) — precomputed
        truth_hx_field,    # (nx, ny, nz)              H(truth) — precomputed
        var_names,         # list[str]                 state variable names
        store_fields=False,
) -> dict:
    """
    Compute metrics and assemble save dict for one multi-obs combo.

    store_fields=False (default, compact)
    --------------------------------------
    xa_mean            (nx, ny, nz, nvar)  analysis ensemble mean
    hxf_mean_field     (nx, ny, nz)        prior mean reflectivity field
    hxa_mean_field     (nx, ny, nz)        analysis mean reflectivity field
    truth_hx_field     (nx, ny, nz)        truth reflectivity field
    innovation_field   (nx, ny, nz)        hxf_mean - truth_hx
    residual_field     (nx, ny, nz)        hxa_mean - truth_hx
    rmse_f_global_{v}  scalar per variable
    rmse_a_global_{v}  scalar per variable
    bias_f_global_{v}  scalar per variable
    bias_a_global_{v}  scalar per variable
    spread_f_global_{v} scalar per variable
    spread_a_global_{v} scalar per variable

    store_fields=True (full, for selected cases only)
    --------------------------------------------------
    All of the above, plus:
    xa                 (nx, ny, nz, Ne, nvar)  full analysis ensemble
    xf                 (nx, ny, nz, Ne, nvar)  full prior ensemble
    """
    xa_mean = xa.mean(axis=3).astype(np.float32)   # (nx, ny, nz, nvar)
    xf_mean = xf.mean(axis=3)
    xa_std  = xa.std(axis=3, ddof=1)
    xf_std  = xf.std(axis=3, ddof=1)

    innovation_field = (hxf_mean_field - truth_hx_field).astype(np.float32)
    residual_field   = (hxa_mean_field - truth_hx_field).astype(np.float32)

    out = dict(
        xa_mean          = xa_mean,
        hxf_mean_field   = hxf_mean_field.astype(np.float32),
        hxa_mean_field   = hxa_mean_field.astype(np.float32),
        truth_hx_field   = truth_hx_field.astype(np.float32),
        innovation_field = innovation_field,
        residual_field   = residual_field,
    )

    # global scalars per variable
    for iv, vname in enumerate(var_names):
        err_f = xf_mean[..., iv] - truth[..., iv]
        err_a = xa_mean[..., iv] - truth[..., iv]
        out[f"rmse_f_global_{vname}"]    = float(np.sqrt((err_f ** 2).mean()))
        out[f"rmse_a_global_{vname}"]    = float(np.sqrt((err_a ** 2).mean()))
        out[f"bias_f_global_{vname}"]    = float(err_f.mean())
        out[f"bias_a_global_{vname}"]    = float(err_a.mean())
        out[f"spread_f_global_{vname}"]  = float(xf_std[..., iv].mean())
        out[f"spread_a_global_{vname}"]  = float(xa_std[..., iv].mean())

    if store_fields:
        out["xa"] = xa.astype(np.float32)
        out["xf"] = xf.astype(np.float32)

    return out