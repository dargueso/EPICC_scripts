subroutine writeintsoil(fieldssoil,hdate,nLats,nLons,startlat,startlon,deltalon,deltalat)

!-----------------------------
!output the variables in intermediate format
!------------------------------

IMPLICIT NONE

!********** user modified variables *************
!
character (len=8), parameter :: ofname = "SOILERA5"
integer, parameter :: version = 5 ! Format version (must =5 for WPS format)
integer, parameter :: iproj = 0 ! Code for projection of data in array:
				! 0 = cylindrical equidistant
				! 1 = Mercator
				! 3 = Lambert conformal conic
				! 4 = Gaussian (global only!)
				! 5 = Polar stereographic
character (len=8), parameter :: startloc = "SWCORNER" ! Which point in array is given by
				! startlat/startlon; set either
				! to 'SWCORNER' or 'CENTER '
character (len=32), parameter :: map_source = "ERA5"! Source model / originating center
logical, parameter :: is_wind_grid_rel = .FALSE. ! Flag indicating whether winds are
				! relative to source grid (TRUE) or
				! relative to earth (FALSE)
real, parameter :: earth_radius = 6356.766 ! Earth radius, km
					!this is average not from model


!******************************************************
!From toInterData
integer :: nLons, nLats ! x- and y-dimensions of 2-d array

real :: startlat, startlon ! Lat/lon of point in array indicated by
! startloc string
real :: deltalon,deltalat ! Grid spacing, degrees
!*******************************************************




real :: xfcst ! Forecast hour of data
real :: xlvl ! Vertical level of data in 2-d array
real,dimension(nLons,nLats) :: slab ! The 2-d array holding the data
real, dimension(:,:,:) :: fieldssoil !The array holding all 3-d variables

character (len=9) :: field ! Name of the field
character (len=24) :: hdate ! Valid date for data YYYY:MM:DD_HH:00:00
character (len=25) :: units ! Units of data
character (len=46) :: desc ! Short description of data
character (len=46) :: fieldssoil_desc(9)
character (len=25) :: fieldssoil_units(9)
character (len=9)  :: fieldssoil_name(9)


integer :: ounit !output file
integer :: status,nf,nl,nfieldssoil
!local unused arguments


!f2py intent(in) ::plvs,fields3d,fields2d,hdate,nLats,nLons,startlat,startlon,deltalon

nfieldssoil=size(fieldssoil,1)

fieldssoil_name(1) = 'LANDSEA  '
fieldssoil_name(2) = 'ST000007 '
fieldssoil_name(3) = 'ST007028 '
fieldssoil_name(4) = 'ST028100 '
fieldssoil_name(5) = 'ST100289 '
fieldssoil_name(6) = 'SM000007 '
fieldssoil_name(7) = 'SM007028 '
fieldssoil_name(8) = 'SM028100 '
fieldssoil_name(9) = 'SM100289 '

fieldssoil_units(1) = '0/1 Flag                 '
fieldssoil_units(2) = 'K                        '
fieldssoil_units(3) = 'K                        '
fieldssoil_units(4) = 'K                        '
fieldssoil_units(5) = 'K                        '
fieldssoil_units(6) = 'fraction                 '
fieldssoil_units(7) = 'fraction                 '
fieldssoil_units(8) = 'fraction                 '
fieldssoil_units(9) = 'fraction                 '

fieldssoil_desc(1) = 'Land/Sea flag                               '
fieldssoil_desc(2) = 'T of 0-7 cm ground layer                    '
fieldssoil_desc(3) = 'T of 7-28 cm ground layer                   '
fieldssoil_desc(4) = 'T of 28-100 cm ground layer                 '
fieldssoil_desc(5) = 'T of 100-289 cm ground layer                '
fieldssoil_desc(6) = 'Soil moisture of 0-7 cm ground layer        '
fieldssoil_desc(7) = 'Soil moisture of 7-28 cm ground layer       '
fieldssoil_desc(8) = 'Soil moisture of 28-100 cm ground layer     '
fieldssoil_desc(9) = 'Soil moisture of 100-289 cm ground layer    '



xfcst = 0.0 ! In the case of GCM is 0 hours
ounit=11

open(ounit,file="./"//TRIM(ofname)//":"//hdate(1:13),form='unformatted',IOSTAT=status,convert="BIG_ENDIAN")
if (status /= 0) then
  print *,"could not create ./"//TRIM(ofname)//':'//hdate(1:13)
  stop
endif

print *, "./"//TRIM(ofname)//":"//hdate(1:13)


!########## SOIL FIELDS ###############

do nf = 1,nfieldssoil

  xlvl = 200100.0
  field = fieldssoil_name(nf)
  units = fieldssoil_units(nf)
  desc  = fieldssoil_desc(nf)
  slab = TRANSPOSE(fieldssoil(nf,:,:))


  write(unit=ounit) version

  write(unit=ounit) hdate, xfcst, map_source, field, &
  units, desc, xlvl, nLons, nLats, iproj
  write(unit=ounit) startloc, startlat, startlon, &
  deltalat, deltalon, earth_radius

  ! 3) WRITE WIND ROTATION FLAG
  write(unit=ounit) is_wind_grid_rel
  ! 4) WRITE 2-D ARRAY OF DATA
  write(unit=ounit) slab

end do
close(ounit)
return
END SUBROUTINE
