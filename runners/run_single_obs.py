import sys, pathlib
# Add repo root to sys.path
REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import argparse, yaml, numpy as np
from da.fortran_interpreter import tempered_wloc
#from obs.reflectivity import calc_reflectivity
from cletkf_wloc      import common_da        as cda
calc_reflectivity = cda.calc_ref

# simple saver
import os, json
from datetime import datetime
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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    paths = cfg["paths"]; st = cfg["state"]; obs_cfg = cfg["obs"]; da_cfg = cfg["da"]

    # load data
    data = np.load(paths["prepared"])["cross_sections"]  # [nx,1,nz,nbv,nvar]
    truth_member = st["truth_member"]
    mask = np.zeros(data.shape[3], dtype=bool); mask[truth_member] = True
    truth = data[:, :, :, mask, :][:, :, :, 0, :]        # [nx,1,nz,nvar]
    xf = data[:, :, :, ~mask, :]                         # [nx,1,nz,Ne,nvar]
    nx, ny, nz, Ne, nvar = xf.shape

    # single observation location
    ox, oy, oz = obs_cfg["loc"]["x"], obs_cfg["loc"]["y"], obs_cfg["loc"]["z"]

    # truth obs
    qg = truth[ox, oy, oz, st["var_idx"]["qg"]]
    qr = truth[ox, oy, oz, st["var_idx"]["qr"]]
    qs = truth[ox, oy, oz, st["var_idx"]["qs"]]
    tt = truth[ox, oy, oz, st["var_idx"]["T"]]
    pp = truth[ox, oy, oz, st["var_idx"]["P"]]
    yo = np.array([calc_reflectivity(qr, qs, qg, tt, pp)])#, dtype="float32")

    # forecasted obs Hx_f for all members

    # LETKF (Fortran)
    steps = tempering_steps(da_cfg["ntemp"][0], da_cfg["alpha"][0])
    obs_error = np.array([obs_cfg["sigma_dbz"]], dtype="float32")  # std
    loc_scales = np.array(da_cfg.get("loc_scales", [5,5,5]), dtype="float32")
    ox_arr = np.array([ox], dtype="int32")
    oy_arr = np.array([oy], dtype="int32")
    oz_arr = np.array([oz], dtype="int32")

    xatemp, deps = tempered_wloc(st=st,
        xf_grid=xf.astype("float32"),
        yo=yo, obs_error=obs_error,
        loc_scales=loc_scales, ox=ox_arr, oy=oy_arr, oz=oz_arr,
        steps=steps
    )
    Xa = xatemp[..., -1]  # [nx,1,nz,Ne,nvar]

    # save
    meta = dict(config=cfg)
    outtag = f"single_obs_ntemp{da_cfg['ntemp'][0]}_alpha{da_cfg['alpha'][0]}"
    _save_run(paths["outdir"], outtag,
              xa=Xa,xatemp=xatemp, xf=xf, yo=yo, deps=deps, steps=steps,
              obs_loc=(ox, oy, oz), truth=truth, meta=meta)

if __name__ == "__main__":
    main()
