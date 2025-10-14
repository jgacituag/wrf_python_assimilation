MODULE common_da
!=======================================================================
!
! [PURPOSE:] Simple data assimilation tools for 3D models
! With localization
! J. Ruiz - 5/2/2025
! IMPORTANT!!: This is a simple implementation of LETKF. No observation
! screening of any type is performed in this code. All observation 
! screening should be performed before calling this routine.
!
!=======================================================================
!$USE OMP_LIB
  USE common_tools
  USE common_letkf

  IMPLICIT NONE

  PUBLIC

CONTAINS

!=======================================================================
!  
!=======================================================================
SUBROUTINE simple_letkf_wloc(nx,ny,nz,nbv,nvar,nobs,hxf,xf,dep,ox,oy,oz,locs,oerr,xa)

IMPLICIT NONE
INTEGER,INTENT(IN)         :: nx,ny,nz,nbv,nvar !number of X,Y,Z, members , update variables
INTEGER,INTENT(IN)         :: nobs              !number of observations to be assimilated
REAL(r_sngl),INTENT(IN)    :: dep(nobs)         !Obs departure for each obs.
REAL(r_sngl),INTENT(IN)    :: ox(nobs),oy(nobs),oz(nobs) !Observation location
REAL(r_sngl),INTENT(IN)    :: oerr(nobs)                 !Observation error
REAL(r_sngl),INTENT(IN)    :: locs(3)                    !Localization scale in x, y and z
REAL(r_sngl),INTENT(IN)    :: xf(nx,ny,nz,nbv,nvar)      !Prior for the updated variables.
REAL(r_sngl),INTENT(IN)    :: hxf(nobs,nbv)              !Ensemble forecast in obs. space
REAL(r_sngl),INTENT(OUT)   :: xa(nx,ny,nz,nbv,nvar)      !Posterior for the updated variables.
INTEGER                    :: ix,iy,iz,im,iv,im2,io
REAL(r_size)               :: hxfmean(nobs) , xfmean(nvar) 
REAL(r_size)               :: xfpert(nbv,nvar) , hxfpert(nobs,nbv)
REAL(r_size)               :: wa(nbv,nbv) , wamean(nbv) , pa(nbv,nbv) , rloc(nobs) , infl
REAL(r_size)               :: oerr_rsize(nobs) , dep_rsize(nobs)

rloc=1.0d0
infl=1.0d0
oerr_rsize = REAL( oerr , r_size )
dep_rsize  = REAL( dep  , r_size )

DO io = 1,nobs
   CALL com_mean( nbv , REAL( hxf(io,:),r_size),hxfmean(io) )
   hxfpert( io , : ) = hxf( io , : ) - hxfmean( io ) 
ENDDO


!$OMP PARALLEL DO PRIVATE(ix,iy,iz,iv,im,im2,wa,wamean,pa,xfmean,xfpert,rloc)
DO ix = 1 , nx
  !write(*,*) ix, '/', nx
  DO iy = 1 , ny
    DO iz = 1 , nz
       !write(*,*) 'point (xi,iy,iz)' , ix, iy, iz
       !Observations will be assimilated at each grid point considering their location.
      
       !Computing the local 
       DO iv = 1,nvar
         CALL com_mean( nbv,REAL(xf(ix,iy,iz,:,iv),r_size),xfmean(iv) )
         xfpert(:,iv) = xf(ix,iy,iz,:,iv) - xfmean(iv) 
       END DO

       !Computing simple localization
       CALL simple_loc( ix , iy , iz , ox , oy , oz , locs , nobs , rloc )

       rloc(:) = oerr(:) / MAX( rloc(:), 1.0e-6_r_size )

       CALL letkf_core(nbv,nobs,hxfpert,rloc,   &
                    dep_rsize,infl,wa,wamean,pa,1.0d0)

       !Apply the weights and update the state variables. 
       DO iv=1,nvar
          xa(ix,iy,iz,:,iv) = xfmean(iv)
          DO im=1,nbv
            DO im2 = 1,nbv
               xa(ix,iy,iz,im,iv) = xa(ix,iy,iz,im,iv) &                 
               & + xfpert(im2,iv) * (wa(im2,im) + wamean(im2))    
            END DO
          END DO
       END DO

    END DO
  END DO
END DO 
!$OMP END PARALLEL DO

END SUBROUTINE simple_letkf_wloc

SUBROUTINE simple_loc( glx , gly , glz , olx , oly , olz , locs , nobs , rloc )
IMPLICIT NONE
INTEGER, INTENT(IN) :: nobs  !Number of observations.
INTEGER , INTENT(IN) :: glx , gly , glz !Grid point location
REAL(r_sngl) , INTENT(IN) :: olx(nobs) , oly(nobs) , olz(nobs) !Observation's location
REAL(r_size) , INTENT(INOUT) :: rloc(nobs)
REAL(r_sngl) , INTENT(IN)    :: locs(3)  !Localization scales
INTEGER :: iobs
REAL(r_sngl) :: dist

DO iobs = 1 , nobs 
   dist = 0.0
   !Negative localization scale means no localization in that direction.
   IF ( locs(1) > 0.0 ) dist = dist + ((REAL(glx,r_sngl)-olx(iobs))/locs(1))**2
   IF ( locs(2) > 0.0 ) dist = dist + ((REAL(gly,r_sngl)-oly(iobs))/locs(2))**2
   IF ( locs(3) > 0.0 ) dist = dist + ((REAL(glz,r_sngl)-olz(iobs))/locs(3))**2

   rloc(iobs) = exp( -0.5*dist ) 
   !WRITE(*,*)dist,rloc(iobs)
ENDDO

END SUBROUTINE simple_loc

SUBROUTINE calc_ref_ens(nx,ny,nz,nbv,qrens,qsens,qgens,tens,pens,refens)
IMPLICIT NONE
INTEGER,INTENT(IN)         :: nx,ny,nz,nbv !number of X,Y,Z, members , update variables
REAL(r_size), INTENT(IN) :: qrens(nx,ny,nz,nbv),qsens(nx,ny,nz,nbv),qgens(nx,ny,nz,nbv)
REAL(r_size), INTENT(IN) :: tens(nx,ny,nz,nbv),pens(nx,ny,nz,nbv)
REAL(r_size), INTENT(OUT):: refens(nx,ny,nz,nbv)

INTEGER :: ix,iy,iz,im

!$OMP PARALLEL DO PRIVATE(ix,iy,iz,im)
DO ix=1,nx
  DO iy=1,ny
    DO iz=1,nz
      DO im=1,nbv
         CALL calc_ref(qrens(ix,iy,iz,im),qsens(ix,iy,iz,im),qgens(ix,iy,iz,im),    &
                       tens(ix,iy,iz,im),pens(ix,iy,iz,im),refens(ix,iy,iz,im) )
      END DO
    END DO
  END DO
END DO
!$OMP END PARALLEL DO


END SUBROUTINE calc_ref_ens

!-----------------------------------------------------------------------
! Compute radar reflectivity and radial wind.
! Radial wind computations for certain methods depend on model reflectivity
! so both functions has been merged into a single one.
! First reflectivity is computed, and the the radial velocity is computed.
!-----------------------------------------------------------------------
SUBROUTINE calc_ref(qr,qs,qg,t,p,ref)
  IMPLICIT NONE
  REAL(r_size), INTENT(IN) :: qr  !Cloud and rain water
  REAL(r_size), INTENT(IN) :: qs,qg !Cloud ice, snow and graupel
  REAL(r_size), INTENT(IN) :: t,p    !Temperature and pressure.
  REAL(r_size), INTENT(OUT) :: ref   !Reflectivity
  REAL(r_size)              :: ro 
  REAL(r_size)  :: qms , qmg !Melting species concentration (method 3)
  REAL(r_size)  :: qt        !Total condensate mixing ratio (method 1)
  REAL(r_size)  :: zr , zs , zg !Rain, snow and graupel's reflectivities.
  REAL(r_size)  :: nor, nos, nog !Rain, snow and graupel's intercepting parameters.
  REAL(r_size)  :: ror, ros, rog , roi !Rain, snow and graupel, ice densities.
  REAL(r_size)  :: cf,cf2,cf3,cf4, pip , roo
  REAL(r_size)  :: ki2 , kr2
  REAL(r_size)  :: tmp_factor , rofactor
  REAL(r_size)  :: p0
  REAL(r_size),PARAMETER  :: mindbz=-20.0d0

  !Note: While equivalent reflectivity is assumed to be independent of the radar, in 
  !practice short wavelengths as those associated with K band radars migh frequently
  !experience Mie scattering. In that case, the equivalent reflectivity is not longer
  !radar independent and an appropiate relationship between the forecasted concentrations
  !and the reflectivity should be used.
  
  !Initialize reflectivities
  zr=0.0d0
  zs=0.0d0
  zg=0.0d0
  ref=mindbz

  !Compute air density (all methods use this)

  ro =  p / (rd * t)

  !Begin computation of reflectivity and vr

  !Observation operator from Tong and Xue 2006, 2008 a and b.
  !Based on Smith et al 1975.
  !Ensemble Kalman Filter Assimilation of Doppler Radar Data with a Compressible
  !Nonhydrostatic Model: OSS Experiments. MWR. 133, 1789-187.
  !It includes reflectivity contribution by all the microphisical species.
  !is assumes Marshall and Palmer distributions.
  !Based on S band radars.
    nor=8.0d6      ![m^-4]
    nos=2.0d6      ![m^-4] This value has been modified according to WRF WSM6
    nog=4.0d6      ![m^-4] This value has been modified according to WRF WSM6
    ror=1000.0d0   ![Kg/m3]
    ros=100.0d0    ![Kg/m3]
    rog=913.0d0    ![Kg/m3] 
    roi=917.0d0    ![Kg/m3]
    roo=1.0d0      ![Kg/m3] Surface air density.
    ki2=0.176d0    !Dielectric factor for ice.
    kr2=0.930d0    !Dielectric factor for water.
    pip=pi ** 1.75 !factor
    cf=1.0d18*720/(pip*(nor**0.75d0)*(ror**1.75d0))
    cf2=1.0d18*720*ki2*( ros ** 0.25 )/(pip*kr2*(nos ** 0.75)*( roi ** 2 ) )
    cf3=1.0d18*720/( pip * ( nos ** 0.75 ) * ( ros ** 1.75 ) )  
    cf4=(1.0d18*720/( pip * ( nog ** 0.75) * ( rog ** 1.75 ) ) ) ** 0.95

    zr=0.0d0
    zs=0.0d0
    zg=0.0d0
    IF( qr .GT. 0.0d0 )THEN
      !rain contribution
      zr= cf * ( ( ro * qr )**1.75 )
    ENDIF
    IF( qs .GT. 0.0d0 )THEN
      IF ( t <= 273.16 )THEN
        !Dry snow
        zs = cf2*( ( ro * qs ) ** 1.75 )
      ELSE
        !Wet snow
        zs = cf3 * ( ( ro * qs ) ** 1.75 )
      ENDIF
    ENDIF

    !Only wet graupel contribution is ussed.
    IF( qg .GT. 0.0d0 )THEN
      zg= cf4 * ( ( ro * qg ) ** 1.6625 )
    ENDIF

    ref = zr + zs + zg  

    ref = 10.0d0*log10( ref )

    IF(ref < mindbz)THEN
       ref = mindbz
    ENDIF

RETURN

END SUBROUTINE calc_ref


END MODULE common_da




