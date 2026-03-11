"""
src/extract_3d_subset.py
========================
Extract a 3D WRF ensemble subset and save it as a compressed .npz file.

usage:

    python src/extract_3d_subset.py --config configs/build_3D_section.yaml

or imported in a notebook / script:

    from extract_3d_subset import process_data
    process_data("configs/build_3D_section.yaml")

Output array layout
-------------------
state_ensemble : (nx, ny, nz, Ne, 8)  float32
    Variable index mapping (last axis):
      0 – QGRAUP  [kg/kg]
      1 – QRAIN   [kg/kg]
      2 – QSNOW   [kg/kg]
      3 – T       [K]
      4 – P       [Pa]
      5 – UA      [m/s]
      6 – VA      [m/s]
      7 – WA      [m/s]
lats       : (ny, nx)      latitude  [°]
lons       : (ny, nx)      longitude [°]
z_heights  : (nz, ny, nx)  height above sea level [m]

YAML config schema
------------------
cross_sections_job:
  paths:
    pattern:    "/path/{member}/wrfout_d01_{date}"   # {member} and {date} are substituted
    output:     "/path/to/output/subset_{date}.npz"
    init_date:  "2023-12-16_19:00:00"
    end_date:   "2023-12-16_19:00:00"
    freq:       "1H"
  ensemble:
    mem_ini: 1
    mem_end: 30
    pad:     3     # zero-padding width for member number string
  subset_3d:
    timeidx: -1    # WRF time index (-1 = last)
    k_start: ~     # vertical level start (null = 0)
    k_end:   ~     # vertical level end   (null = top)
    j_start: ~     # south-north start
    j_end:   ~     # south-north end
    i_start: ~     # west-east start
    i_end:   ~     # west-east end
"""

import argparse
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml
from netCDF4 import Dataset
from tqdm import tqdm
import wrf


############## Helper functions ##############

def _expand_members(mem_ini: int, mem_end: int, pad: int) -> List[str]:
    """Return zero-padded member strings from mem_ini to mem_end inclusive."""
    return [str(i).zfill(pad) for i in range(mem_ini, mem_end + 1)]


def _resolve_paths(cfg: dict, date: str) -> Tuple[List, List, str]:
    """
    Resolve file paths for a single date from the config.

    Returns (members, nc_paths, out_path).
    """
    p   = cfg["cross_sections_job"]["paths"]
    ens = cfg["cross_sections_job"]["ensemble"]

    pattern = p.get("pattern") or p.get("template")
    if pattern is None:
        raise ValueError(
            "cross_sections_job.paths.pattern is required in the YAML config."
        )

    members  = _expand_members(ens["mem_ini"], ens["mem_end"], ens.get("pad", 0))
    nc_paths = [pattern.format(member=m, date=date) for m in members]
    out_path = p["output"].format(date=date)
    return members, nc_paths, out_path


def _get_vars(nc: Dataset, timeidx: int) -> dict:
    """
    Extract required WRF variables from an open Dataset.

    Returns a dict of xarray.DataArray keyed by variable name.
    """
    return {
        "QGRAUP":   wrf.getvar(nc, "QGRAUP", timeidx=timeidx, meta=True),
        "QRAIN":    wrf.getvar(nc, "QRAIN",  timeidx=timeidx, meta=True),
        "QSNOW":    wrf.getvar(nc, "QSNOW",  timeidx=timeidx, meta=True),
        "tk":       wrf.getvar(nc, "temp",   timeidx=timeidx, meta=True),
        "pressure": wrf.getvar(nc, "pres",   timeidx=timeidx, meta=True),
        "ua":       wrf.getvar(nc, "ua",     timeidx=timeidx, meta=True),
        "va":       wrf.getvar(nc, "va",     timeidx=timeidx, meta=True),
        "wa":       wrf.getvar(nc, "wa",     timeidx=timeidx, meta=True),
        "z":        wrf.getvar(nc, "z",      timeidx=timeidx, meta=True),
    }


def _slices_from_cfg(sub_cfg: dict):
    """Build k, j, i slices from the subset_3d config block."""
    return (
        slice(sub_cfg.get("k_start"), sub_cfg.get("k_end")),
        slice(sub_cfg.get("j_start"), sub_cfg.get("j_end")),
        slice(sub_cfg.get("i_start"), sub_cfg.get("i_end")),
    )


################ Main processing function ##############

def process_data(config_path: str) -> None:
    """
    Extract 3D WRF ensemble subsets for all dates in the config and save
    each as a compressed .npz file.

    Parameters
    ----------
    config_path : str
        Path to the YAML configuration file.
    """
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    paths_cfg = cfg["cross_sections_job"]["paths"]
    date_ini  = paths_cfg.get("init_date")
    date_end  = paths_cfg.get("end_date")
    freq      = paths_cfg.get("freq", "1H")

    if date_ini is None or date_end is None:
        raise ValueError("init_date and end_date must be set in the YAML config.")

    dates = pd.date_range(
        start=pd.to_datetime(date_ini, format="%Y-%m-%d_%H:%M:%S"),
        end=pd.to_datetime(date_end,   format="%Y-%m-%d_%H:%M:%S"),
        freq=freq,
    )
    print(f"[info] {len(dates)} date(s) to process  ({date_ini} → {date_end}, freq={freq})")

    for dt in dates:
        date = dt.strftime("%Y-%m-%d_%H:%M:%S")
        print(f"\n--- {date} ---")

        members, nc_paths, out_path = _resolve_paths(cfg, date)

        sub_cfg = cfg["cross_sections_job"]["subset_3d"]
        timeidx = sub_cfg.get("timeidx", -1)
        k_slice, j_slice, i_slice = _slices_from_cfg(sub_cfg)

        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        ##### extract metadata and sample shape from the first member #####
        print(f"[info] probing from {nc_paths[0]}")
        with Dataset(nc_paths[0]) as nc0:
            v0  = _get_vars(nc0, timeidx)
            lat = wrf.getvar(nc0, "lat", timeidx=timeidx, meta=True)
            lon = wrf.getvar(nc0, "lon", timeidx=timeidx, meta=True)

            samp        = v0["tk"][k_slice, j_slice, i_slice]
            nz, ny, nx  = samp.shape
            lats_sub    = wrf.to_np(lat[j_slice, i_slice])
            lons_sub    = wrf.to_np(lon[j_slice, i_slice])
            z_heights   = wrf.to_np(v0["z"][k_slice, j_slice, i_slice])

        Ne   = len(nc_paths)
        nvar = 8
        out  = np.zeros((nx, ny, nz, Ne, nvar), dtype=np.float32)

        print(f"[info] output shape: (nx={nx}, ny={ny}, nz={nz}, Ne={Ne}, nvar={nvar})")
        print("[info] variable order: [QGRAUP, QRAIN, QSNOW, T, P, UA, VA, WA]")

        ###### fill ensemble array from all members #######
        for j, path in enumerate(tqdm(nc_paths, desc="members")):
            if not os.path.isfile(path):
                print(f"[warning] missing file for member {members[j]}: {path}")
                out[:, :, :, j, :] = np.nan
                continue

            with Dataset(path) as nc:
                v = _get_vars(nc, timeidx)
                # WRF arrays are (nz, ny, nx); transpose to (nx, ny, nz)
                out[:, :, :, j, 0] = wrf.to_np(v["QGRAUP"][k_slice, j_slice, i_slice]).T
                out[:, :, :, j, 1] = wrf.to_np(v["QRAIN" ][k_slice, j_slice, i_slice]).T
                out[:, :, :, j, 2] = wrf.to_np(v["QSNOW" ][k_slice, j_slice, i_slice]).T
                out[:, :, :, j, 3] = wrf.to_np(v["tk"    ][k_slice, j_slice, i_slice]).T
                out[:, :, :, j, 4] = wrf.to_np(v["pressure"][k_slice, j_slice, i_slice]).T
                out[:, :, :, j, 5] = wrf.to_np(v["ua"    ][k_slice, j_slice, i_slice]).T
                out[:, :, :, j, 6] = wrf.to_np(v["va"    ][k_slice, j_slice, i_slice]).T
                out[:, :, :, j, 7] = wrf.to_np(v["wa"    ][k_slice, j_slice, i_slice]).T

        # clean up any all-NaN vertical levels (common when k_start is above the surface)
        finite_z = np.isfinite(out).any(axis=(0, 1, 3, 4))
        n_dropped = int((~finite_z).sum())
        if n_dropped:
            print(f"[clean] dropping {n_dropped} all-NaN z-level(s) "
                  f"— consider adjusting k_start in the config.")
            out       = out[:, :, finite_z, :, :]
            z_heights = z_heights[finite_z, :, :]

        np.savez_compressed(
            out_path,
            state_ensemble=out,
            lats=lats_sub,
            lons=lons_sub,
            z_heights=z_heights,
        )
        print(f"[done] {out_path}  shape={out.shape}")

    print("\n[info] all dates processed.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Extract a 3D WRF ensemble subset to .npz")
    ap.add_argument("--config", required=True,
                    help="Path to the YAML configuration file")
    args = ap.parse_args()
    process_data(args.config)
