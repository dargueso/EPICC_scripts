# File with EPICC postprocessing input arguments
# To be used with EPICC WRF postprocessing package based on WRF-python

path_wrfo = '/vg6a/dargueso/WRF_OUT/'
path_proc = '/vg5/dargueso/postprocessed/EPICC/temp'
path_unif = '/vg5/dargueso/postprocessed/EPICC/'
path_geo = '/home/dargueso/share/geo_em_files/EPICC/'
file_geo= 'geo_em.d01.EPICC_2km_ERA5_HVC_GWD.nc'
file_ref = '/home/dargueso/share/wrf_ref_files/EPICC/wrfout_d01_2011-12-22_00:00:00'
institution = 'UIB'
wruns = ['EPICC_2km_ERA5']


patt = 'wrfprec'
dom  = 'd01'

syear = 2013
eyear = 2014
smonth =12
emonth =1
acc_dt = 10
nproc_x = 12
nproc_y = 48

variables = ['PRNC']

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
