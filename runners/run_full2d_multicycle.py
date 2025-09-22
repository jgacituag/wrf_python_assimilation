import sys, pathlib
# Add repo root to sys.path
REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import argparse, yaml, numpy as np
from da.fortran_interpreter import tempered_wloc

from cletkf_wloc      import common_da        as cda
calc_reflectivity = cda.calc_ref

from obs import selectors as sel

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


def _truth_to_yo(truth, ox, oy, oz, var_idx):
    yo = np.zeros(len(ox), dtype="float32")
    for i in range(len(ox)):
        xi, yi, zi = int(ox[i]), int(oy[i]), int(oz[i])
        qr = truth[xi, yi, zi, var_idx["qr"]]
        qs = truth[xi, yi, zi, var_idx["qs"]]
        qg = truth[xi, yi, zi, var_idx["qg"]]
        tt = truth[xi, yi, zi, var_idx["T"]]
        pp = truth[xi, yi, zi, var_idx["P"]]
        yo[i] = calc_reflectivity(qr, qs, qg, tt, pp)
    return yo

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

    data = np.load(paths["prepared"])["cross_sections"]  # [nx,1,nz,nbv,nvar]
    truth_member = st["truth_member"]
    mask = np.zeros(data.shape[3], dtype=bool); mask[truth_member] = True
    truth = data[:, :, :, mask, :][:, :, :, 0, :]        # [nx,1,nz,nvar]
    xf = data[:, :, :, ~mask, :]                         # [nx,1,nz,Ne,nvar]
    nx, ny, nz, Ne, nvar = xf.shape

    # --- select observation points ---
    (ox, oy, oz), kind_tag = _select_obs(obs_cfg, nx, nz)
    if len(ox) == 0:
        raise RuntimeError("No observation points selected — check obs config.")
    print(f"[obs] kind={kind_tag}, Nobs={len(ox)}")

    # Build y^o from truth and set obs error
    yo = _truth_to_yo(truth, ox, oy, oz, st["var_idx"])
    obs_error = (obs_cfg.get("sigma_dbz", 1.0) * np.ones_like(yo)).astype("float32")

    # Tempering & localization
    steps = tempering_steps(da_cfg["ntemp"][0], da_cfg["alpha"][0])
    loc_scales = np.array(da_cfg.get("loc_scales", [5,5,5]), dtype="float32")
    ox_arr, oy_arr, oz_arr = ox.astype("int32"), oy.astype("int32"), oz.astype("int32")

    Xf_grid = xf.astype("float32")
    all_cycle = []
    cycles = int(da_cfg.get("cycles", 1))

    for cyc in range(cycles): 
        print(f"[cycle] {cyc+1}/{cycles}: LETKF (Fortran) …")
        xatemp, deps = tempered_wloc(st=st,
            xf_grid=Xf_grid yo=yo,
            obs_error=obs_error, loc_scales=loc_scales,
            ox=ox_arr, oy=oy_arr, oz=oz_arr,
            steps=steps
        )
        Xa_grid = xatemp[..., -1]
        all_cycle.append(dict(Xa=Xa_grid, deps=deps, steps=steps))

        # Identity model between cycles (plug in WRF step here later if desired)
        Xf_grid = Xa_grid

    Xa = all_cycle[-1]["Xa"]
    hxf_last = hxf  # from last cycle

    # save
    meta = dict(config=cfg)
    tag = cfg.get("experiment_tag", "full2d_multicycle_v1")
    outtag = f"{tag}__cyc={cycles}__nt={da_cfg['ntemp'][0]}__a={da_cfg['alpha'][0]}__kind={kind_tag}"
    _save_run(paths["outdir"], outtag,
              xa=Xa, xf=xf, yo=yo, hxf=hxf_last,
              deps=all_cycle[-1]["deps"], steps=all_cycle[-1]["steps"],
              obs_loc=(ox.tolist(), oy.tolist(), oz.tolist()),
              truth=truth, meta=meta)

if __name__ == "__main__":
    main()
