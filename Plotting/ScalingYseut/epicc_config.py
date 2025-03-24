# File with EPICC postprocessing input arguments
# To be used with EPICC WRF postprocessing package based on WRF-python

#/vg9b/dargueso-NOBKUP/postprocessed/EPICC/EPICC_2km_ERA5/
#path_in = "/vg9b/dargueso-NOBKUP/postprocessed/EPICC/" #in eady and mc4
path_in = "/home/dargueso/postprocessed/EPICC/" #in medicane
#path_wrfo = "/home/dargueso/WRF_OUT/"
#path_proc = "/home/dargueso/postprocessed/EPICC/temp"
#path_unif = "/home/dargueso/postprocessed/EPICC/"
#path_geo = "/home/dargueso/share/geo_em_files/EPICC/"
#file_geo = "geo_em.d01.EPICC_2km_ERA5_HVC_GWD.nc"
#file_ref = "/home/dargueso/WRF_OUT/EPICC_2km_ERA5//out/wrfout_d01_2013-01-01_00:00:00"
institution = "UIB"
wrf_runs = ["EPICC_2km_ERA5"]
seasons = ['DJF','MAM','JJA','SON']
#patt_in = "UIB_"

wet_value = 0.1 #mm
qtiles = [0.9,0.95,0.99,0.999,0.9999]
perc_name =['90','95','99','999','9999']
patt = "wrf3hrly"
dom = "d01"
syear = 2011
eyear = 2020
ds = 'wrf'
dmn = 'subdmn' # 'wdmn' 
#smonth = 1
#emonth = 1
#acc_dt = 10
#nproc_x = 12
#nproc_y = 48
freq = ['10MIN','01H','DAY']
variables = ["RAIN"]

#### Requested output variables (DO NOT CHANGE THIS LINE) ####

# TAS
# PRNC
# TD2
# PRNC
# PSL
# HUSS
# SST
# OLR
# CLOUDFRAC
# SWDOWN
# WDIR10
# WSPD10
# UA
# VA
# TC
# P
# Z
# SPECHUM
# ET
# GEOPOT
# CAPE2D
# TD
# RH
# Q2DIV
# SMOIS
# PBLH
# THETAE
# QVAPOR
# LH
# QCLOUD
# CTT
# QICE
# CLDFRA
# U10MET
# V10MET
# W5L
# OMEGA
# PW
# TVAVG
# PSFC
