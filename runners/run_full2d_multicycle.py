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
    steps /= steps.sum()
    steps = (1.0 / steps) / (1.0 / steps).sum()
    return steps.astype("float32")


def _truth_to_yo(truth,xf, ox, oy, oz, var_idx):
    print(f"[info] building yo from truth at {len(ox)*len(oy)*len(oz)} obs locations")
    yo = []
    x0 = []
    y0 = []
    z0 = []

    for xi in ox:
        for yi in oy:
            for zi in oz:
                qr = truth[xi, yi, zi, var_idx["qr"]]
                qs = truth[xi, yi, zi, var_idx["qs"]]
                qg = truth[xi, yi, zi, var_idx["qg"]]
                tt = truth[xi, yi, zi, var_idx["T"]]
                pp = truth[xi, yi, zi, var_idx["P"]]

                qr_f = np.mean(xf[xi, yi, zi, :, var_idx["qr"]])
                qs_f = np.mean(xf[xi, yi, zi, :, var_idx["qs"]])
                qg_f = np.mean(xf[xi, yi, zi, :, var_idx["qg"]])
                tt_f = np.mean(xf[xi, yi, zi, :, var_idx["T"]])
                pp_f = np.mean(xf[xi, yi, zi, :, var_idx["P"]])

                yf = calc_reflectivity(qr_f, qs_f, qg_f, tt_f, pp_f)
                yt = calc_reflectivity(qr, qs, qg, tt, pp)
                if yf < 5:
                    yf = 0.0
                if yt < 5:
                    yt = 0.0
                if np.isnan(yf) or np.isnan(yt):
                    continue
                elif yt<0.1 and yf<0.1:
                    continue
                else:
                    yo.append(yt)
                    x0.append(xi)
                    y0.append(yi)
                    z0.append(zi)

    yo = np.array(yo, dtype="float32")
    x0 = np.array(x0, dtype="int32")
    y0 = np.array(y0, dtype="int32")
    z0 = np.array(z0, dtype="int32")
    print(f"[info] built yo with {len(yo)} valid obs")
    return yo, x0, y0, z0

def _select_obs(obs_cfg, nx, nz):
    kind = obs_cfg.get("kind", "FULL_2D").upper()
    if kind == "FULL_2D":
        return sel.full2d(nx, nz), "FULL_2D"
    if kind == "EVERY_OTHER":
        stride_x = int(obs_cfg.get("stride_x", 2))
        stride_z = int(obs_cfg.get("stride_z", 2))
        offset_x = int(obs_cfg.get("offset_x", 0))
        offset_z = int(obs_cfg.get("offset_z", 0))
        return sel.every_other(nx, nz, stride_x, stride_z, offset_x, offset_z), f"EVERY_OTHER_sx{stride_x}_sz{stride_z}"
    if kind == "RHI":
        rhi_cfg = obs_cfg.get("rhi", {})
        ox0 = int(rhi_cfg.get("origin_x", nx//2))
        oz0 = int(rhi_cfg.get("origin_z", 0))
        angles = list(map(float, rhi_cfg.get("angles_deg", [10, 20, 30, 40, 50, 60, 70])))
        max_range = float(rhi_cfg.get("max_range", max(nx, nz)))
        dr = float(rhi_cfg.get("dr", 1.0))
        return sel.rhi(nx, nz, ox0, oz0, angles, max_range, dr), f"RHI_{ox0}_{oz0}_{len(angles)}ang"
    raise ValueError(f"Unknown obs.kind: {kind}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    paths = cfg["paths"]; st = cfg["state"]; obs_cfg = cfg["obs"]; da_cfg = cfg["da"]

    date_ini = da_cfg.get("init_date")
    date_end = da_cfg.get("end_date")
    freq     = da_cfg.get("freq")
    dates = pd.date_range(start=pd.to_datetime(date_ini,format="%Y-%m-%d_%H:%M:%S"), end=pd.to_datetime(date_end,format="%Y-%m-%d_%H:%M:%S"), freq=freq)
    print(f"[info] running data assimilation for {len(dates)} dates from {date_ini} to {date_end} every {freq}")
    for date in dates:
        date = date.strftime("%Y-%m-%d_%H:%M:%S")
        print(f"[info] processing date: {date}")
        prepared_file = paths["prepared"].format(date=date)

        data = np.load(prepared_file)["cross_sections"]  # [nx,1,nz,nbv,nvar]
        truth_member = st["truth_member"]
        mask = np.zeros(data.shape[3], dtype=bool); mask[truth_member] = True
        truth = data[:, :, :, mask, :][:, :, :, 0, :]        # [nx,1,nz,nvar]
        xf = data[:, :, :, ~mask, :]                         # [nx,1,nz,Ne,nvar]
        nx, ny, nz, Ne, nvar = xf.shape

        # --- select observation points ---
        #(ox, oy, oz), kind_tag = _select_obs(obs_cfg, nx, nz)
        ox = np.arange(0, nx, 1)
        oy = np.arange(0, ny, 1)
        oz = np.arange(0, nz, 1)
        kind_tag = "FULL_2D"
        if len(ox) == 0:
            raise RuntimeError("No observation points selected — check obs config.")
        print(f"[obs] kind={kind_tag}, Nobs={len(ox)}")

        # Build y^o from truth and set obs error
        yo,ox_arr,oy_arr,oz_arr = _truth_to_yo(truth, xf, ox, oy, oz, st["var_idx"])
        obs_error = (obs_cfg.get("sigma_dbz", 1.0) * np.ones_like(yo)).astype("float32")

        # Tempering & localization
        steps = tempering_steps(da_cfg["ntemp"][0], da_cfg["alpha"][0])
        loc_scales = np.array(da_cfg.get("loc_scales", [5,5,5]), dtype="float32")

        Xf_grid = xf.astype("float32")
        print(f"[info] running tempered WLOC with {len(steps)} steps, loc_scales={loc_scales.tolist()}")
        xatemp, deps = tempered_wloc(st=st,
            xf_grid=Xf_grid, yo=yo,
            obs_error=obs_error, loc_scales=loc_scales,
            ox=ox_arr, oy=oy_arr, oz=oz_arr,
            steps=steps
        )
        Xa = xatemp[..., -1]

        # save
        meta = dict(config=cfg)
        tag = cfg.get("experiment_tag", "full2d_multicycle_v1")
        outtag = f"{tag}_{date}_temp{da_cfg['ntemp'][0]}_alpha{da_cfg['alpha'][0]}_kind{kind_tag}"
        _save_run(paths["outdir"], outtag,
                xa=Xa, xf=xf, yo=yo,
                deps=deps, steps=steps,
                ox=ox_arr, oy=oy_arr, oz=oz_arr,
                truth=truth, meta=meta)

if __name__ == "__main__":
    main()
