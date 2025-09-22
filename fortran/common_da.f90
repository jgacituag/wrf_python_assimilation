MODULE common_da
!=======================================================================
!
! [PURPOSE:] Data assimilation tools for 1D models
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
SUBROUTINE simple_letkf(nx,ny,nz,nbv,nvar,xfo,xf,dep,error,min_o,xa)

IMPLICIT NONE
INTEGER,INTENT(IN)         :: nx,ny,nz,nbv,nvar !number of X,Y,Z, members , update variables
REAL(r_size),INTENT(INOUT) :: xfo(nx,ny,nz,nbv)        !Prior for the observed variable.
REAL(r_sngl),INTENT(IN)    :: xf(nx,ny,nz,nbv,nvar)    !Prior for the updated variables.
REAL(r_sngl),INTENT(OUT)   :: xa(nx,ny,nz,nbv,nvar)    !Posterior for the updated variables.
!REAL(r_size)               :: dep(1) , error(1)        !Observation departure and observation error.
REAL(r_size)               :: dep(nx,ny,nz) , error(nx,ny,nz) 
INTEGER                    :: ix,iy,iz,im,iv,im2
REAL(r_size)               :: xfomean , xfmean(nvar) 
REAL(r_size)               :: xfpert(nbv,nvar) , xfopert(nbv)

REAL(r_size)               :: wa(nbv,nbv) , wamean(nbv) , pa(nbv,nbv) , rloc(1) , infl
REAL(r_size),INTENT(IN)    :: min_o !Lower threshold for obs value (for reflectivity observations)
REAL(r_size)               :: local_dep(1) , local_error(1)


rloc=1.0d0
infl=1.0d0
!$OMP PARALLEL DO PRIVATE(ix,iy,iz,iv,im,im2,wa,wamean,pa,local_dep,xfmean,xfomean,xfpert,xfopert)
DO ix = 1 , nx
  WRITE(*,*)ix
  DO iy = 1 , ny
    DO iz = 1 , nz

      !DO im = 1 , nbv
      !  IF( xfo(ix,iy,iz,im) <= min_o )THEN
      !          xfo(ix,iy,iz,im) = min_o 
      !  ENDIF
      !END DO

!An observation with the indicated dep will be assimilated at each grid point.
!The posterior is the result of assimilating each observation at each model grid point.
!Observations from distant grid points wont be considered.

      DO iv = 1,nvar
        CALL com_mean( nbv,REAL(xf(ix,iy,iz,:,iv),r_size),xfmean(iv) )
        xfpert(:,iv) = xf(ix,iy,iz,:,iv) - xfmean(iv) 
      END DO
      CALL com_mean( nbv,REAL(xfo(ix,iy,iz,:),r_size),xfomean )
      xfopert(:) = xfo(ix,iy,iz,:) - xfomean

      IF( sum( abs(xfopert) ) >  0.0d0 )THEN

         !IF ( xfomean + dep(ix,iy,iz) <= min_o )THEN
         !   local_dep = min_o - xfomean 
         !ELSE
         !   local_dep = dep(ix,iy,iz)
         !ENDIF

         local_dep   = dep(ix,iy,iz)
         local_error = error(ix,iy,iz)

         CALL letkf_core(nbv,1,xfopert,local_error,   &
                     rloc,local_dep,infl,wa,wamean,pa,1.0d0)

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

      ELSE
         xa(ix,iy,iz,:,:) = xf( ix,iy,iz,:,: )
      ENDIF

    END DO
  END DO
END DO 
!$OMP END PARALLEL DO
END SUBROUTINE simple_letkf

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




