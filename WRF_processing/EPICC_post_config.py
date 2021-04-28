# File with EPICC postprocessing input arguments
# To be used with EPICC WRF postprocessing package based on WRF-python

path_in = '/vg6/dargueso-NO-BKUP/WRF_OUT/EPICC/'
path_out = '/vg5/dargueso-NO-BKUP/postprocessed/EPICC/'
file_ref = 'wrfout_d01_2020-08-01_00:00:00'
institution = 'UIB'
wruns = ['EPICC_2km_ERA5_HVC_GWD']


patt = 'wrfprec'
dom  = 'd01'

syear = 2017
eyear = 2019
smonth =1
emonth =12
acc_dt = 10

variables = ['RAIN']

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
