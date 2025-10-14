import sys, pathlib
# Add repo root to sys.path
REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import argparse, yaml, numpy as np
from da.fortran_interpreter import tempered_wloc
import pandas as pd
import os, json
from datetime import datetime

from cletkf_wloc      import common_da        as cda
calc_reflectivity = cda.calc_ref
import obs.selectors as sel

# simple saver

def _save_run(outdir: str, tag: str, **arrays_and_meta):
    os.makedirs(outdir, exist_ok=True)
    meta = arrays_and_meta.pop("meta", {})
    meta["timestamp"] = datetime.utcnow().isoformat() + "Z"
    with open(os.path.join(outdir, f"{tag}.meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    np.savez_compressed(os.path.join(outdir, f"{tag}.npz"), **arrays_and_meta)
    print(f"[save] {tag} -> {outdir}")

def tempering_steps(ntemp: int, alpha: float) -> np.ndarray:
    if ntemp < 1:
        return np.array([1.0], dtype=np.float32)
    dt = 1.0 / (ntemp + 1)
    grid = np.arange(dt, 1.0 - dt/100.0, dt)
    steps = np.exp(alpha / grid)
    steps = (1/steps) / np.sum((1/steps))
    #steps /= steps.sum()
    #steps = (1.0 / steps) / (1.0 / steps).sum()
    return steps.astype("float32")

def _calc_ref_truth(truth, i,j,k, var_idx):
    qr = truth[i,j,k, var_idx["qr"]]
    qs = truth[i,j,k, var_idx["qs"]]
    qg = truth[i,j,k, var_idx["qg"]]
    TT = truth[i,j,k, var_idx["T"]]
    PP = truth[i,j,k, var_idx["P"]]
    return cda.calc_ref(qr, qs, qg, TT, PP)

def _calc_ref_member(xf, i,j,k, m, var_idx):
    qr = xf[i,j,k, m, var_idx["qr"]]
    qs = xf[i,j,k, m, var_idx["qs"]]
    qg = xf[i,j,k, m, var_idx["qg"]]
    TT = xf[i,j,k, m, var_idx["T"]]
    PP = xf[i,j,k, m, var_idx["P"]]
    return cda.calc_ref(qr, qs, qg, TT, PP)

def build_and_qc_dbz_obs(truth, xf, ox_in, oy_in, oz_in, var_idx,
                         sigma_dbz=5.0, dbz_min=5.0, dbz_max=70.0,
                         detect_prob_min=0.20, use_gross_check=True, k_sigma=5.0):
    """
    truth: (nx,ny,nz,nvar)    single-member 'truth'
    xf   : (nx,ny,nz,Ne,nvar) forecast ensemble
    (ox,oy,oz): integer indices on your 2D cross-section (0-based)
    """
  
    ox, oy, oz = [], [], []
    for xi in ox_in:
        for yi in oy_in:
            for zi in oz_in:
                ox.append(xi)
                oy.append(yi)
                oz.append(zi)


    ox = np.asarray(ox).astype(int)
    oy = np.asarray(oy).astype(int)
    oz = np.asarray(oz).astype(int)
    nobs_cand = ox.size
    nx, ny, nz, Ne, nvar = xf.shape

    # 1) Build yo (truth) and Hx_f ensemble for all candidate points
    yo = np.empty(nobs_cand, dtype=np.float32)
    Hx_f = np.empty((nobs_cand, Ne), dtype=np.float32)
    for ii in range(nobs_cand):
        i,j,k = ox[ii], oy[ii], oz[ii]
        yo[ii] = _calc_ref_truth(truth, i,j,k, var_idx)
        for m in range(Ne):
            Hx_f[ii, m] = _calc_ref_member(xf, i,j,k, m, var_idx)

    keep = np.isfinite(yo) & np.all(np.isfinite(Hx_f), axis=1)

    # 2) Range check (physically plausible)
    keep &= (yo >= dbz_min) & (yo <= dbz_max)

    # 3) Ensemble detectability gate
    p_detect = (Hx_f > dbz_min).mean(axis=1)
    keep &= (p_detect >= detect_prob_min)

    # 4) Gross-error check (optional)
    if use_gross_check:
        d_f = yo - Hx_f.mean(axis=1)
        keep &= (np.abs(d_f) <= k_sigma * float(sigma_dbz))

    # Apply current mask
    idx = np.where(keep)[0]
    yo, ox, oy, oz = yo[idx], ox[idx], oy[idx], oz[idx]

    # 5) Dedup exact overlaps
    if yo.size > 0:
        key = np.stack([ox, oy, oz], axis=1)
        _, uniq_idx = np.unique(key, axis=0, return_index=True)
        uniq_idx.sort()
        yo, ox, oy, oz = yo[uniq_idx], ox[uniq_idx], oy[uniq_idx], oz[uniq_idx]

    return yo.astype('float32'), ox.astype(int), oy.astype(int), oz.astype(int)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    paths = cfg["paths"]; st = cfg["state"]; obs_cfg = cfg["obs"]; da_cfg = cfg["da"]
    temps = da_cfg.get("ntemps")
    alphas = da_cfg.get("alphas")
    loc_scales_files = da_cfg.get("loc_scale")
    sigma_dbz = obs_cfg.get("sigma_dbz", 5.0)
    date_ini = da_cfg.get("init_date")
    date_end = da_cfg.get("end_date")
    freq     = da_cfg.get("freq")
    members = [20]#np.arange(30)
    dates = pd.date_range(start=pd.to_datetime(date_ini,format="%Y-%m-%d_%H:%M:%S"), end=pd.to_datetime(date_end,format="%Y-%m-%d_%H:%M:%S"), freq=freq)
    print(f"[info] running data assimilation for {len(dates)} dates from {date_ini} to {date_end} every {freq}")
    for date in dates:
        date = date.strftime("%Y-%m-%d_%H:%M:%S")
        print(f"[info] processing date: {date}")
        for temp in temps:
            for alpha in alphas:
                for loc_scale in loc_scales_files:
                    for truth_member in members:
                        print(f"[info] running with truth member {truth_member}, temp {temp}, alpha {alpha}, loc_scale {loc_scale}")

                        steps = tempering_steps(temp, alpha)
                        loc_scales = np.array( [loc_scale, loc_scale, loc_scale], dtype="float32")


                        prepared_file = paths["prepared"].format(date=date)

                        data = np.load(prepared_file)["cross_sections"]  # [nx,1,nz,nbv,nvar]
                        mask = np.zeros(data.shape[3], dtype=bool); mask[truth_member] = True
                        truth = data[:, :, :, mask, :][:, :, :, 0, :]        # [nx,1,nz,nvar]
                        xf = data[:, :, :, ~mask, :]                         # [nx,1,nz,Ne,nvar]
                        nx, ny, nz, Ne, nvar = xf.shape
                        
                        ox = np.arange(0, nx, 2)
                        oy = np.arange(0, ny, 2)
                        oz = np.arange(0, nz, 2) 
                        # Build y^o from truth and set obs error

                        yo, ox_arr, oy_arr, oz_arr = build_and_qc_dbz_obs(truth=truth, xf=xf, ox_in=ox, oy_in=oy, oz_in=oz,var_idx=st["var_idx"],sigma_dbz=sigma_dbz, dbz_min=5.0, dbz_max=70.0,detect_prob_min=0.20, use_gross_check=True, k_sigma=5.0)

                        print(f"\n[DIAGNOSTIC] Observation Density Check:")
                        print(f"Grid size: {nx} × {ny} × {nz}")
                        print(f"Total observations: {len(ox_arr)}")
                        print(f"Obs density: {len(ox_arr)/(nx*ny*nz)*100:.1f}% of grid points")
                        print(f"Obs spacing: every {2} grid points")
                        print(f"Localization scale: {loc_scale} grid points")
                        print(f"Effective radius: ~{loc_scale*2}-{loc_scale*3} grid points")
                        print(f"Expected obs influencing each point: ~{(loc_scale*2/2)**2:.0f}")
                        
                        kind_tag = "FULL_2D"
                        if len(ox_arr) == 0:
                            raise RuntimeError("No observation points selected — check obs config.")
                        print(f"[obs] kind={kind_tag}, Nobs={len(ox_arr)}")
                        obs_error = (obs_cfg.get("sigma_dbz", 1.0) * np.ones_like(yo)).astype("float32")

                        # Tempering & localization

                        Xf_grid = xf.astype("float32")
                        xatemp, deps, hxf = tempered_wloc(st=st,
                            xf_grid=Xf_grid, yo=yo,
                            obs_error=obs_error, loc_scales=loc_scales,
                            ox=ox_arr, oy=oy_arr, oz=oz_arr,
                            steps=steps
                        )
                        Xa = xatemp[..., -1]

                        # save
                        meta = dict(config=cfg)
                        tag = cfg.get("experiment_tag", "full2d_multicycle_v1")
                        outtag = f"{tag}_{date}_temp{temp}_alpha{int(alpha)}_Loc{int(loc_scale)}_True{int(truth_member)}_kind{kind_tag}"
                        _save_run(paths["outdir"], outtag,
                                xa=Xa, xf=xf, hxf=hxf,yo=yo,
                                deps=deps, steps=steps,obs_error=obs_error,
                                ox=ox_arr, oy=oy_arr, oz=oz_arr,
                                truth=truth, meta=meta)

if __name__ == "__main__":
    main()
