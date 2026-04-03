MODULE common_da
!=======================================================================
!
! [PURPOSE:] LETKF-based data assimilation for 3D WRF ensembles.
!
!   simple_letkf_wloc : run LETKF at every grid point with Gaussian
!                       R-localisation.
!   calc_ref           : forward operator — radar reflectivity from
!                        WRF microphysics variables (Tong & Xue 2006).
!
! [NOTE:] No observation screening is performed here. All QC must be
!         applied before calling simple_letkf_wloc.
!
! [DEPENDENCY:] common_tools, common_letkf, BLAS (dgemm via common_letkf)
!
!=======================================================================
  USE common_tools
  USE common_letkf
  IMPLICIT NONE
  PUBLIC

CONTAINS

!=======================================================================
SUBROUTINE simple_letkf_wloc(nx, ny, nz, nbv, nvar, nobs, &
                              hxf, xf, dep, ox, oy, oz,    &
                              locs, oerr, xa, n_updated)
!=======================================================================
!
!  Run one LETKF analysis step at every grid point (ix, iy, iz).
!  Localisation is Gaussian R-localisation: observation error is
!  inflated by 1/rho, where
!
!    rho(ix,iy,iz) = exp(-0.5 * [((ix-ox)/lx)^2
!                               + ((iy-oy)/ly)^2
!                               + ((iz-oz)/lz)^2])
!
!  Points beyond max_dist = 2*sqrt(10/3)*max(lx,ly,lz) receive
!  rho -> 0, so the observation is effectively ignored there.
!
!  INPUT
!    nx,ny,nz    : domain dimensions
!    nbv         : ensemble size
!    nvar        : number of state variables
!    nobs        : number of observations
!    hxf(nobs,nbv)         : H(xf) — ensemble in obs space
!    xf(nx,ny,nz,nbv,nvar) : prior ensemble
!    dep(nobs)             : departures  yo - H(xf_mean)
!    ox,oy,oz(nobs)        : observation locations (1-based grid indices)
!    locs(3)               : localisation scales [lx, ly, lz] in grid points
!    oerr(nobs)            : observation error variance
!
!  OUTPUT
!    xa(nx,ny,nz,nbv,nvar) : posterior ensemble
!
!=======================================================================
  IMPLICIT NONE
  INTEGER,      INTENT(IN)  :: nx, ny, nz, nbv, nvar, nobs
  REAL(r_sngl), INTENT(IN)  :: hxf(nobs, nbv)
  REAL(r_sngl), INTENT(IN)  :: xf(nx, ny, nz, nbv, nvar)
  REAL(r_sngl), INTENT(IN)  :: dep(nobs)
  REAL(r_sngl), INTENT(IN)  :: ox(nobs), oy(nobs), oz(nobs)
  REAL(r_sngl), INTENT(IN)  :: locs(3)
  REAL(r_sngl), INTENT(IN)  :: oerr(nobs)
  REAL(r_sngl), INTENT(OUT) :: xa(nx, ny, nz, nbv, nvar)
  INTEGER,      INTENT(OUT) :: n_updated

  INTEGER      :: ix, iy, iz, iv, im, im2, io
  REAL(r_size) :: hxfmean(nobs)
  REAL(r_size) :: hxfpert(nobs, nbv)
  REAL(r_size) :: xfmean(nvar)
  REAL(r_size) :: xfpert(nbv, nvar)

  INTEGER      :: local_nobs
  REAL(r_size) :: rloc_loc(nobs)
  REAL(r_size) :: hxfpert_loc(nobs, nbv)
  REAL(r_size) :: dep_loc(nobs)

  REAL(r_size) :: wa(nbv, nbv), wamean(nbv), pa(nbv, nbv)
  REAL(r_size) :: oerr_dp(nobs), dep_dp(nobs)
  REAL(r_size) :: infl
  REAL(r_size) :: max_dist, dist
  REAL(r_sngl) :: lx, ly, lz

  
  oerr_dp  = REAL(oerr, r_size)
  dep_dp   = REAL(dep,  r_size)
  lx = locs(1);  ly = locs(2);  lz = locs(3)

  ! Compact-support cutoff (horizontal): beyond this dist observations
  ! have negligible weight and are skipped.
  max_dist = 2.0d0 * SQRT(10.0d0/3.0d0) * MAX(REAL(lx,r_size), MAX(REAL(ly,r_size),REAL(lz,r_size)))
  n_updated = 0
  ! Pre-compute H(xf) ensemble perturbations (same for all grid points)
  DO io = 1, nobs
    CALL com_mean(nbv, REAL(hxf(io,:), r_size), hxfmean(io))
    hxfpert(io, :) = REAL(hxf(io,:), r_size) - hxfmean(io)
  END DO

!$OMP PARALLEL DO PRIVATE(ix,iy,iz,iv,im,im2,io,dist,xfmean,xfpert,wa,wamean,pa, &
!$OMP                     local_nobs, rloc_loc, hxfpert_loc, dep_loc) &
!$OMP             REDUCTION(+:n_updated)
  DO ix = 1, nx
    DO iy = 1, ny
      DO iz = 1, nz

        ! ---- 1. Filter observations for this specific grid point ----
        infl = 1.0d0
        local_nobs = 0
        DO io = 1, nobs
          dist = 0.0_r_sngl
          IF (lx > 1.0e-6) dist = dist + ((REAL(ix,r_sngl) - ox(io)) / lx)**2
          IF (ly > 1.0e-6) dist = dist + ((REAL(iy,r_sngl) - oy(io)) / ly)**2
          IF (lz > 1.0e-6) dist = dist + ((REAL(iz,r_sngl) - oz(io)) / lz)**2

          ! Keep observation if it is strictly inside the cutoff
          IF (dist <= max_dist) THEN
            local_nobs = local_nobs + 1
            ! R-localisation: inflate obs error by 1/weight
            rloc_loc(local_nobs) = oerr_dp(io) / MAX(EXP(-0.5_r_size * REAL(dist, r_size)), 1.0e-6_r_size)
            hxfpert_loc(local_nobs, :) = hxfpert(io, :)
            dep_loc(local_nobs) = dep_dp(io)
          END IF
        END DO

        ! ---- 2. Skip computation entirely if no obs are nearby ----
        IF (local_nobs == 0) THEN
          xa(ix,iy,iz,:,:) = xf(ix,iy,iz,:,:)
          CYCLE
        END IF
        n_updated = n_updated + 1
        ! ---- 3. Prior ensemble mean and perturbations -------------
        DO iv = 1, nvar
          CALL com_mean(nbv, REAL(xf(ix,iy,iz,:,iv), r_size), xfmean(iv))
          xfpert(:, iv) = REAL(xf(ix,iy,iz,:,iv), r_size) - xfmean(iv)
        END DO

        ! ---- 4. LETKF core using ONLY local obs -------------------
        CALL letkf_core(nbv, local_nobs, hxfpert_loc(1:local_nobs, :), &
                        rloc_loc(1:local_nobs), dep_loc(1:local_nobs), &
                        infl, wa, wamean, pa, 1.0d0)

        ! ---- 5. Apply weights to update state variables -----------
        DO iv = 1, nvar
          xa(ix,iy,iz,:,iv) = REAL(xfmean(iv), r_sngl)
          DO im = 1, nbv
            DO im2 = 1, nbv
              xa(ix,iy,iz,im,iv) = xa(ix,iy,iz,im,iv) &
                + REAL(xfpert(im2,iv) * (wa(im2,im) + wamean(im2)), r_sngl)
            END DO
          END DO
        END DO

      END DO
    END DO
  END DO
!$OMP END PARALLEL DO

END SUBROUTINE simple_letkf_wloc

!=======================================================================
SUBROUTINE calc_ref_ens(nx, ny, nz, nbv, qr, qs, qg, t, p, ref)
!=======================================================================
!
! Compute radar reflectivity for the full ensemble over the 3D domain.
! Vectorised wrapper around calc_ref — avoids per-point Python loop.
!
!  INPUT
!    nx, ny, nz : domain dimensions
!    nbv        : ensemble size (0 for single-member call)
!    qr(nx,ny,nz,nbv) : rain    mixing ratio [kg/kg]
!    qs(nx,ny,nz,nbv) : snow    mixing ratio [kg/kg]
!    qg(nx,ny,nz,nbv) : graupel mixing ratio [kg/kg]
!    t (nx,ny,nz,nbv) : temperature [K]
!    p (nx,ny,nz,nbv) : pressure    [Pa]
!
!  OUTPUT
!    ref(nx,ny,nz,nbv) : reflectivity [dBZ]
!
!=======================================================================
  IMPLICIT NONE
  INTEGER,      INTENT(IN)  :: nx, ny, nz, nbv
  REAL(r_size), INTENT(IN)  :: qr(nx,ny,nz,nbv)
  REAL(r_size), INTENT(IN)  :: qs(nx,ny,nz,nbv)
  REAL(r_size), INTENT(IN)  :: qg(nx,ny,nz,nbv)
  REAL(r_size), INTENT(IN)  :: t (nx,ny,nz,nbv)
  REAL(r_size), INTENT(IN)  :: p (nx,ny,nz,nbv)
  REAL(r_size), INTENT(OUT) :: ref(nx,ny,nz,nbv)

  INTEGER :: ix, iy, iz, im

!$OMP PARALLEL DO PRIVATE(ix,iy,iz,im)
  DO ix = 1, nx
    DO iy = 1, ny
      DO iz = 1, nz
        DO im = 1, nbv
          CALL calc_ref(qr(ix,iy,iz,im), qs(ix,iy,iz,im), &
                        qg(ix,iy,iz,im), t (ix,iy,iz,im), &
                        p (ix,iy,iz,im), ref(ix,iy,iz,im))
        END DO
      END DO
    END DO
  END DO
!$OMP END PARALLEL DO

END SUBROUTINE calc_ref_ens

!=======================================================================
SUBROUTINE calc_ref(qr, qs, qg, t, p, ref)
!=======================================================================
!
! Radar reflectivity forward operator (Tong & Xue 2006, 2008).
! Based on Smith et al. 1975, Marshall-Palmer distributions, S-band.
!
!  INPUT
!    qr, qs, qg : rain, snow, graupel mixing ratios [kg/kg]
!    t          : temperature [K]
!    p          : pressure    [Pa]
!
!  OUTPUT
!    ref        : equivalent reflectivity factor [dBZ], floor at -20 dBZ
!
!=======================================================================
  IMPLICIT NONE
  REAL(r_size), INTENT(IN)  :: qr, qs, qg, t, p
  REAL(r_size), INTENT(OUT) :: ref

  REAL(r_size), PARAMETER :: mindbz = -20.0d0

  ! Intercept parameters [m^-4]
  REAL(r_size), PARAMETER :: nor = 8.0d6
  REAL(r_size), PARAMETER :: nos = 2.0d6
  REAL(r_size), PARAMETER :: nog = 4.0d6

  ! Densities [kg/m^3]
  REAL(r_size), PARAMETER :: ror = 1000.0d0
  REAL(r_size), PARAMETER :: ros =  100.0d0
  REAL(r_size), PARAMETER :: rog =  913.0d0
  REAL(r_size), PARAMETER :: roi =  917.0d0

  ! Dielectric factors
  REAL(r_size), PARAMETER :: ki2 = 0.176d0
  REAL(r_size), PARAMETER :: kr2 = 0.930d0

  REAL(r_size) :: ro, pip
  REAL(r_size) :: cf, cf2, cf3, cf4
  REAL(r_size) :: zr, zs, zg

  ! Pre-compute coefficients
  pip  = pi ** 1.75d0
  cf   = 1.0d18 * 720.0d0 / (pip * (nor**0.75d0) * (ror**1.75d0))
  cf2  = 1.0d18 * 720.0d0 * ki2 * (ros**0.25d0) &
         / (pip * kr2 * (nos**0.75d0) * (roi**2.0d0))
  cf3  = 1.0d18 * 720.0d0 / (pip * (nos**0.75d0) * (ros**1.75d0))
  cf4  = (1.0d18 * 720.0d0 / (pip * (nog**0.75d0) * (rog**1.75d0)))**0.95d0

  ! Air density
  ro = p / (rd * t)

  zr = 0.0d0
  zs = 0.0d0
  zg = 0.0d0

  IF (qr > 0.0d0) zr = cf  * (ro * qr)**1.75d0
  IF (qs > 0.0d0) THEN
    IF (t <= 273.16d0) THEN
      zs = cf2 * (ro * qs)**1.75d0   ! dry snow
    ELSE
      zs = cf3 * (ro * qs)**1.75d0   ! wet snow
    END IF
  END IF
  IF (qg > 0.0d0) zg = cf4 * (ro * qg)**1.6625d0

  ref = zr + zs + zg
  IF (ref > 0.0d0) THEN
    ref = 10.0d0 * LOG10(ref)
  ELSE
    ref = mindbz
  END IF
  IF (ref < mindbz) ref = mindbz

END SUBROUTINE calc_ref

END MODULE common_da