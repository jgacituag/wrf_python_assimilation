#!/usr/bin/env python
import os, re, glob, argparse
from datetime import datetime
import yaml
import numpy as np
from tqdm import tqdm
from netCDF4 import Dataset
import wrf

TS_FMT = "%Y-%m-%d_%H:%M:%S"
FNAME_TS_RE = re.compile(r"(\d{4}-\d{2}-\d{2}[_-]\d{2}:\d{2}:\d{2})")

def _expand_members(a, b, pad): return [str(i).zfill(pad) for i in range(a, b+1)]

def _resolve_globs(cfg):
    p = cfg["cross_sections_job"]["paths"]
    ens = cfg["cross_sections_job"]["ensemble"]
    if "glob" not in p:
        raise ValueError("Provide cross_sections_job.paths.glob like /root/{member}/wrfout_d01_*")
    members = _expand_members(ens["mem_ini"], ens["mem_end"], ens.get("pad",0))
    out_path = p["output"]
    pattern = p["glob"]
    per_member_files = []
    for m in members:
        g = pattern.format(member=m)
        files = sorted(glob.glob(g))
        if not files:
            raise FileNotFoundError(f"No wrfout files for member {m} using glob: {g}")
        per_member_files.append(files)
    return members, per_member_files, out_path

def _time_from_fname(path):
    m = FNAME_TS_RE.search(os.path.basename(path))
    if not m: return None
    s = m.group(1).replace("-", "_", 1)  # normalize 2019-10-10_18:00:00
    if "-" in s.split("_")[0]:  # already has date-
        s = s.replace("-", "_", 1)
    s = s.replace("-", "_") if "_" not in s else s
    s = s.replace("__","_")
    s = s.replace("-",":",2) if "-" in s.split("_")[-1] else s
    # be permissive: try both variants
    for fmt in (TS_FMT, "%Y_%m_%d_%H:%M:%S"):
        try: return datetime.strptime(s, fmt)
        except Exception: pass
    return None

def _extract_time_from_file(path):
    with Dataset(path) as nc:
        dt = wrf.extract_times(nc, wrf.ALL_TIMES)
        if len(dt) == 0:
            raise ValueError(f"No time in file: {path}")
        return dt[0]

def _member_time_table(files):
    """Return list[(path, dt)] sorted by dt."""
    pairs = []
    for f in files:
        dt = _time_from_fname(f)
        if dt is None:
            dt = _extract_time_from_file(f)
        pairs.append((f, dt))
    pairs.sort(key=lambda x: x[1])
    return pairs

def _intersect_time_axes(member_tables):
    """Ensure all members share the same time axis; align paths accordingly."""
    # Build sets of datetimes per member
    dt_lists = [ [dt for _, dt in tbl] for tbl in member_tables ]
    # Intersect (keep order of the first member)
    base = dt_lists[0]
    common = [dt for dt in base if all(dt in dts for dts in dt_lists[1:])]
    if not common:
        raise ValueError("No common times across members. Ensure consistent times.")
    # For each member, map dt -> path
    aligned = []
    for tbl in member_tables:
        d2p = {dt: p for p, dt in tbl}
        paths = [d2p[dt] for dt in common]
        aligned.append(paths)
    return common, aligned

def _select_time_indices(cfg_time, py_times):
    # Accept same selectors as before. py_times is a list[datetime]
    if "timeidx" in cfg_time and cfg_time["timeidx"] is not None:
        idx = int(cfg_time["timeidx"])
        if idx < 0: idx = len(py_times) + idx
        if not (0 <= idx < len(py_times)): raise IndexError("timeidx out of bounds")
        return [idx]
    if str(cfg_time.get("times","")).lower() == "all":
        return list(range(len(py_times)))
    if "indices" in cfg_time:
        idxs = [int(i) for i in cfg_time["indices"]]
        for i in idxs:
            if not (0 <= i < len(py_times)): raise IndexError(f"indices contains {i} out of bounds")
        return idxs
    if "range" in cfg_time:
        r = cfg_time["range"]
        t0 = datetime.strptime(r["start"], TS_FMT)
        t1 = datetime.strptime(r["end"],   TS_FMT)
        if t1 < t0: t0, t1 = t1, t0
        idxs = [i for i,t in enumerate(py_times) if t0 <= t <= t1]
        if not idxs: raise ValueError("No times in requested range.")
        return idxs
    # default last
    return [len(py_times)-1]

def _compute_ij_bbox(nc, scfg):
    mode = scfg.get("mode","full").lower()
    if mode == "full":
        nx = nc.dimensions["west_east"].size
        ny = nc.dimensions["south_north"].size
        return 0, nx, 0, ny
    if mode == "ij":
        ij = scfg["ij"]; return int(ij["i0"]), int(ij["i1"]), int(ij["j0"]), int(ij["j1"])
    if mode == "bbox":
        bbox = scfg["bbox"]
        lats, lons = wrf.latlon_coords(wrf.getvar(nc, "T2", timeidx=0, meta=True))
        lat = np.array(lats); lon = np.array(lons)
        mask = (lat >= bbox["lat_min"]) & (lat <= bbox["lat_max"]) & \
               (lon >= bbox["lon_min"]) & (lon <= bbox["lon_max"])
        js, is_ = np.where(mask)
        if js.size == 0: raise ValueError("BBox did not intersect the domain.")
        return is_.min(), is_.max()+1, js.min(), js.max()+1
    raise ValueError("spatial.mode must be full|ij|bbox")

def _get_vars_3d(nc):
    qg = wrf.getvar(nc, "QGRAUP",  timeidx=0, meta=True)
    qr = wrf.getvar(nc, "QRAIN",   timeidx=0, meta=True)
    qs = wrf.getvar(nc, "QSNOW",   timeidx=0, meta=True)
    tk = wrf.getvar(nc, "temp",    timeidx=0, meta=True)
    p  = wrf.getvar(nc, "pressure",timeidx=0, meta=True)
    ua = wrf.getvar(nc, "ua",      timeidx=0, meta=True)
    va = wrf.getvar(nc, "va",      timeidx=0, meta=True)
    wa = wrf.getvar(nc, "wa",      timeidx=0, meta=True)
    return dict(QGRAUP=qg, QRAIN=qr, QSNOW=qs, tk=tk, pressure=p, ua=ua, va=va, wa=wa)

def _subset_ij_3d(da, i0, i1, j0, j1):
    sub = da.isel(south_north=slice(j0,j1), west_east=slice(i0,i1))
    return wrf.to_np(sub)  # (z,y,x)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config,"r"))

    job = cfg["cross_sections_job"]
    members, per_member_files, out_path = _resolve_globs(cfg)

    # 1) Build per-member (path, time) tables and align by common times
    tables = [ _member_time_table(files) for files in per_member_files ]
    common_times, aligned_paths = _intersect_time_axes(tables)  # common order across members

    # 2) Select time indices per config
    tsel = _select_time_indices(job.get("time",{}), common_times)
    sel_times = [ common_times[i] for i in tsel ]
    sel_str   = [ t.strftime(TS_FMT) for t in sel_times ]
    nt = len(sel_times)

    # 3) Open one sample file to get ij window and shapes
    sample_path = aligned_paths[0][tsel[0]]
    with Dataset(sample_path) as nc0:
        i0, i1, j0, j1 = _compute_ij_bbox(nc0, job.get("spatial", {"mode":"full"}))
        samp = _subset_ij_3d(_get_vars_3d(nc0)["tk"], i0,i1,j0,j1)  # (z,y,x)
        nz, ny, nx = samp.shape

    nvar = 8
    Ne   = len(members)
    volumes = np.zeros((nt, nx, ny, nz, Ne, nvar), dtype=np.float32)

    print(f"[info] dims: nt={nt}, nx={nx}, ny={ny}, nz={nz}, Ne={Ne}, nvar={nvar}")
    print(f"[info] ij bbox: i=[{i0}:{i1}) j=[{j0}:{j1})")
    print(f"[info] first time: {sel_str[0]}  last time: {sel_str[-1]}")

    # 4) Fill
    for j, m in enumerate(tqdm(members, desc="members")):
        paths_j = aligned_paths[j]
        for it, idx in enumerate(tqdm(tsel, leave=False, desc=f"times({m})")):
            f = paths_j[idx]
            with Dataset(f) as nc:
                v = _get_vars_3d(nc)
                a_qg = _subset_ij_3d(v["QGRAUP"],  i0,i1,j0,j1)
                a_qr = _subset_ij_3d(v["QRAIN"],   i0,i1,j0,j1)
                a_qs = _subset_ij_3d(v["QSNOW"],   i0,i1,j0,j1)
                a_tk = _subset_ij_3d(v["tk"],      i0,i1,j0,j1)
                a_p  = _subset_ij_3d(v["pressure"],i0,i1,j0,j1)
                a_ua = _subset_ij_3d(v["ua"],      i0,i1,j0,j1)
                a_va = _subset_ij_3d(v["va"],      i0,i1,j0,j1)
                a_wa = _subset_ij_3d(v["wa"],      i0,i1,j0,j1)

                def xyz(a): return np.transpose(a, (2,1,0))  # (z,y,x) -> (x,y,z)
                volumes[it, :, :, :, j, 0] = xyz(a_qg)
                volumes[it, :, :, :, j, 1] = xyz(a_qr)
                volumes[it, :, :, :, j, 2] = xyz(a_qs)
                volumes[it, :, :, :, j, 3] = xyz(a_tk)
                volumes[it, :, :, :, j, 4] = xyz(a_p)
                volumes[it, :, :, :, j, 5] = xyz(a_ua)
                volumes[it, :, :, :, j, 6] = xyz(a_va)
                volumes[it, :, :, :, j, 7] = xyz(a_wa)

    np.savez_compressed(
        out_path,
        volumes=volumes,
        times=np.array(sel_str, dtype="U19"),
        members=np.array(members, dtype="U"),
        ij_bbox=np.array([i0,i1,j0,j1], dtype=np.int32),
    )
    print(f"[done] wrote: {out_path}")

if __name__ == "__main__":
    main()
