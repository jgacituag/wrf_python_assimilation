from typing import Tuple, List
import numpy as np

def full2d(nx: int, nz: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mask = np.ones((nx, nz), dtype=bool)
    ox, oz = np.where(mask)
    oy = np.zeros_like(ox, dtype=int)
    return ox.astype(int), oy, oz.astype(int)

def every_other(nx: int, nz: int,
                stride_x: int = 2, stride_z: int = 2,
                offset_x: int = 0, offset_z: int = 0
               ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs = np.arange(offset_x, nx, max(1, stride_x))
    zs = np.arange(offset_z, nz, max(1, stride_z))
    mask = np.zeros((nx, nz), dtype=bool)
    mask[np.ix_(xs, zs)] = True
    ox, oz = np.where(mask)
    oy = np.zeros_like(ox, dtype=int)
    return ox.astype(int), oy, oz.astype(int)

def rhi(nx: int, nz: int, origin_x: int, origin_z: int,
        angles_deg: List[float], max_range: float, dr: float = 1.0
       ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    ox_list, oz_list = [], []
    for th in angles_deg:
        th_rad = np.deg2rad(th)
        ux, uz = np.cos(th_rad), np.sin(th_rad)
        nsteps = int(max_range / max(dr, 1e-6))
        for k in range(1, nsteps + 1):
            x = int(round(origin_x + k * dr * ux))
            z = int(round(origin_z + k * dr * uz))
            if x < 0 or x >= nx or z < 0 or z >= nz:
                break
            if not ox_list or (x != ox_list[-1] or z != oz_list[-1]):
                ox_list.append(x); oz_list.append(z)

    if not ox_list:
        empty = np.array([], dtype=int)
        return empty, empty, empty

    ox = np.array(ox_list, dtype=int)
    oz = np.array(oz_list, dtype=int)
    oy = np.zeros_like(ox, dtype=int)

    # Deduplicate preserving order
    seen = set()
    keep = []
    for i in range(len(ox)):
        key = (int(ox[i]), int(oz[i]))
        if key not in seen:
            seen.add(key)
            keep.append(i)

    keep = np.array(keep, dtype=int)
    return ox[keep], oy[keep], oz[keep]