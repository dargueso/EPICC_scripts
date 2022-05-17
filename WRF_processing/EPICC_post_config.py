# File with EPICC postprocessing input arguments
# To be used with EPICC WRF postprocessing package based on WRF-python

path_wrfo = '/vg6/dargueso-NO-BKUP/WRF_OUT/EPICC/'
path_proc = '/vg6/dargueso-NO-BKUP/postprocessed/EPICC/'
path_unif = '/vg6/dargueso-NO-BKUP/postprocessed/unified/EPICC/'
path_geo = '/home/dargueso/share/geo_em_files/EPICC/'
file_geo= 'geo_em.d01.EPICC_2km_ERA5_HVC_GWD.nc'
file_ref = 'wrfout_d01_2020-08-01_00:00:00'
institution = 'UIB'
wruns = ['EPICC_2km_ERA5_CMIP6anom_HVC_GWD']#,'EPICC_2km_ERA5_CMIP6anom_HVC_GWD']


patt = 'wrf3hrly'
dom  = 'd01'

syear = 2013
eyear = 2020
smonth =1
emonth =12
acc_dt = 10

variables = ['CAPE2D']

#### Requested output variables (DO NOT CHANGE THIS LINE) ####

#TAS
#PRNC
#TD2
#PRNC
#PSL
#HUSS
#SST
#OLR
#CLOUDFRAC
#SWDOWN
#WDIR10
#WSPD10
#UA
#VA
#TC
#P
#Z
#SPECHUM
#ET
#GEOPOT
#CAPE2D
#TD
#RH
#Q2DIV
#SMOIS
#PBLH
#THETAE
#QVAPOR
#LH
#QCLOUD
#CTT
#QICE
#CLDFRA
#U10MET
#V10MET
#W5L
#OMEGA
#PW
#TVAVG
#PSFC
