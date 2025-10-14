import os, sys
import numpy as np

def _import_fortran():
    try:
        from cletkf_wloc import common_da as cda
        return cda
    except Exception:
        here = os.path.dirname(os.path.abspath(__file__))
        third_party = os.path.normpath(os.path.join(here, "..", "fortran"))
        if third_party not in sys.path:
            sys.path.insert(0, third_party)
        from cletkf_wloc import common_da as cda
        return cda

cda = _import_fortran()

def check():
    print("Fortran LETKF loaded from:", cda.__file__)

def tempered_wloc(st,xf_grid, yo, obs_error, loc_scales, ox, oy, oz, steps):
    nx, ny, nz, Ne, nvar = xf_grid.shape
    nobs = len(yo)
    ntemp = len(steps)

    xf_grid = np.asfortranarray(xf_grid.astype("float32"))
    yo      = np.asarray(yo, dtype="float32")
    obs_error = np.asarray(obs_error, dtype="float32")
    obs_error = obs_error * np.ones( nobs )
    loc_scales = np.asarray(loc_scales, dtype="float32")
    ox = np.asarray(ox, dtype="int32")
    oy = np.asarray(oy, dtype="int32")
    oz = np.asarray(oz, dtype="int32")

    ox_f = (ox.astype(np.int64) + 1).astype("float32")
    oy_f = (oy.astype(np.int64) + 1).astype("float32")
    oz_f = (oz.astype(np.int64) + 1).astype("float32")
    steps = np.asarray(steps, dtype="float32")

    xatemp = np.zeros((nx,ny,nz,Ne,nvar, ntemp+1), dtype="float32", order="F")
    xatemp[..., 0] = xf_grid
    deps = np.zeros((ntemp, nobs), dtype="float32")
    hxf = np.zeros((nobs,Ne), dtype=np.float32, order="F")
    hxfs = np.zeros((ntemp,nobs, Ne), dtype=np.float32, order="F")

    for it in range(ntemp):
        #print(f"[info] Tempering step {it+1}/{ntemp}, alpha_temp={steps[it]:.3f}")
        for ii in range(nobs):
            oxi, oyi, ozi = ox[ii], oy[ii], oz[ii]
            for jj in range(Ne):
                qr = xatemp[oxi, oyi, ozi, jj, st["var_idx"]["qr"], it]
                qs = xatemp[oxi, oyi, ozi, jj, st["var_idx"]["qs"], it]
                qg = xatemp[oxi, oyi, ozi, jj, st["var_idx"]["qg"], it]
                tt = xatemp[oxi, oyi, ozi, jj, st["var_idx"]["T"],  it]
                pp = xatemp[oxi, oyi, ozi, jj, st["var_idx"]["P"],  it]
                hxf[ii, jj] = cda.calc_ref(qr, qs, qg, tt, pp)
        
        hxfs[it, :,:] = hxf
        dep = np.asfortranarray(yo - hxf.mean(axis=1).astype("float32"), dtype="float32")
        deps[it, :] = dep
        #print(f'hxf shape: {hxf.shape}')
        #print(f'dep shape: {dep.shape}')
        oerr_temp = (obs_error / steps[it]).astype("float32")
        print(f'starting LETKF with loc_scales={loc_scales}, oerr mean={oerr_temp.mean():.3f}')
        Xa = cda.simple_letkf_wloc(
            nx=nx, ny=ny, nz=nz,
            nbv=Ne, nvar=nvar, nobs=nobs,
            hxf=hxf, xf=xatemp[..., it],
            dep=dep, ox=ox_f, oy=oy_f, oz=oz_f,
            locs=loc_scales, oerr=oerr_temp
        ).astype("float32")

        xatemp[..., it+1] = Xa
        spread = xatemp[..., it].std(axis=3).mean()
        increment = np.abs(xatemp[..., it+1] - xatemp[..., it]).mean()
        print(f"Step {it}: increment magnitude = {increment:.6f}, spread = {spread:.4f}")
    return xatemp, deps, hxfs