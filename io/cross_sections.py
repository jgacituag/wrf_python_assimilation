#!/usr/bin/env python
"""
Builds ensemble cross-sections from WRF outputs.

Output file:
  npz with key "cross_sections" of shape [nx, 1, nz, Ne, nvar]
  var order:
    [QGRAUP, QRAIN, QSNOW, T(K), P(Pa), UA(m/s), VA(m/s), WA(m/s)]

Config YAML expected keys (see configs/build_cross_sections.yaml):
cross_sections_job:
  paths:
    output: /abs/path/to/ensemble_cross_sections.npz
    pattern: /abs/path/to/{member}/wrfout_d01_YYYY-mm-dd_HH:MM:SS
  ensemble:
    mem_ini: 1
    mem_end: 30
    pad: 3
  variables: ["QGRAUP","QRAIN","QSNOW","temp","pressure","ua","va","wa"]
  cross_section:
    start: [-39.2, -65.5]   # [lat, lon]
    end:   [-39.2, -62.0]   # [lat, lon]
    timeidx: -1
"""

import os
import argparse
import yaml
import numpy as np
from tqdm import tqdm

from netCDF4 import Dataset
import wrf

def _expand_members(mem_ini: int, mem_end: int, pad: int):
    return [str(i).zfill(pad) for i in range(mem_ini, mem_end + 1)]

def _resolve_paths(cfg):
    p = cfg["cross_sections_job"]["paths"]
    ens = cfg["cross_sections_job"]["ensemble"]
    pattern = p.get("pattern") or p.get("template")
    if pattern is None:
        raise ValueError("Provide cross_sections_job.paths.pattern (recommended) or .template")

    members = _expand_members(ens["mem_ini"], ens["mem_end"], ens.get("pad", 0))
    if "{member}" in pattern:
        nc_paths = [pattern.format(member=m) for m in members]
    else:
        nc_paths = [pattern for _ in members]
    out_path = p["output"]
    return members, nc_paths, out_path

def _get_vars(nc, timeidx):
    """
    Returns dict with required WRF variables as xarray.DataArray:
      QGRAUP, QRAIN, QSNOW, tk (K), pressure (Pa), ua, va, wa, and z (m).
    wrf-python names:
      - 'ua','va','wa' are mass-grid, de-staggered winds (m/s).
      - 'tk' is temp (K), 'pressure' is Pa.
    """
    qg = wrf.getvar(nc, "QGRAUP",  timeidx=timeidx, meta=True)
    qr = wrf.getvar(nc, "QRAIN",   timeidx=timeidx, meta=True)
    qs = wrf.getvar(nc, "QSNOW",   timeidx=timeidx, meta=True)

    tk = wrf.getvar(nc, "temp",       timeidx=timeidx, meta=True)
    p  = wrf.getvar(nc, "pressure", timeidx=timeidx, meta=True)

    ua = wrf.getvar(nc, "ua", timeidx=timeidx, meta=True)  # m/s
    va = wrf.getvar(nc, "va", timeidx=timeidx, meta=True)  # m/s
    wa = wrf.getvar(nc, "wa", timeidx=timeidx, meta=True)  # m/s

    z = wrf.getvar(nc, "z", timeidx=timeidx, meta=True)    # m
    return dict(QGRAUP=qg, QRAIN=qr, QSNOW=qs, tk=tk, pressure=p, ua=ua, va=va, wa=wa, z=z)

def _vert_xsection(da, z, start_latlon, end_latlon, nc):
    """
    Interpolate a 3D field da(z,y,x) onto a vertical cross-section between
    (start_lat, start_lon) and (end_lat, end_lon). Returns ndarray (nz, nx).
    """
    sp = wrf.CoordPair(lat=float(start_latlon[0]), lon=float(start_latlon[1]))
    ep = wrf.CoordPair(lat=float(end_latlon[0]),   lon=float(end_latlon[1]))
    cs = wrf.vertcross(da, z,
                       start_point=sp, end_point=ep,
                       wrfin=nc, latlon=True, meta=True)#,
                       #autolevels=da.shape[0])
    #return np.asarray(cs)  # (nz, nx)
    return wrf.to_np(cs)  # (nz, nx)
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="YAML with cross_sections_job")
    args = ap.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    job = cfg["cross_sections_job"]
    members, nc_paths, out_path = _resolve_paths(cfg)

    cs_cfg = job["cross_section"]
    start = cs_cfg["start"]
    end   = cs_cfg["end"]
    timeidx = cs_cfg.get("timeidx", -1)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Probe shapes from first member
    with Dataset(nc_paths[0]) as nc0:
        v0 = _get_vars(nc0, timeidx)
        samp = _vert_xsection(v0["tk"], v0["z"], start, end, nc0)  # (nz, nx)
        nz, nx = samp.shape

    # var order (nvar = 8):
    #   0: QGRAUP, 1: QRAIN, 2: QSNOW, 3: T(K), 4: P(hPa), 5: UA, 6: VA, 7: WA
    nvar = 8
    Ne   = len(nc_paths)
    out = np.zeros((nx, 1, nz, Ne, nvar))#, dtype=np.float32, order="F")

    print(f"[info] cross-section dims: nx={nx}, nz={nz}, Ne={Ne}, nvar={nvar}")
    print("[info] variable order: [QGRAUP, QRAIN, QSNOW, T, P, UA, VA, WA]")

    for j, path in enumerate(tqdm(nc_paths, desc="members")):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Missing WRF file for member {members[j]}: {path}")
        with Dataset(path) as nc:
            v = _get_vars(nc, timeidx)

            xs_qg = _vert_xsection(v["QGRAUP"],  v["z"], start, end, nc)
            xs_qr = _vert_xsection(v["QRAIN"],   v["z"], start, end, nc)
            xs_qs = _vert_xsection(v["QSNOW"],   v["z"], start, end, nc)
            xs_tk = _vert_xsection(v["tk"],      v["z"], start, end, nc)
            xs_p  = _vert_xsection(v["pressure"],v["z"], start, end, nc)
            xs_ua = _vert_xsection(v["ua"],      v["z"], start, end, nc)
            xs_va = _vert_xsection(v["va"],      v["z"], start, end, nc)
            xs_wa = _vert_xsection(v["wa"],      v["z"], start, end, nc)

            # transpose to [nx, nz] and assign
            out[:, 0, :, j, 0] = xs_qg.T
            out[:, 0, :, j, 1] = xs_qr.T
            out[:, 0, :, j, 2] = xs_qs.T
            out[:, 0, :, j, 3] = xs_tk.T
            out[:, 0, :, j, 4] = xs_p.T
            out[:, 0, :, j, 5] = xs_ua.T
            out[:, 0, :, j, 6] = xs_va.T
            out[:, 0, :, j, 7] = xs_wa.T

    np.savez_compressed(out_path, cross_sections=out)
    print(f"[done] wrote: {out_path}")

if __name__ == "__main__":
    main()
