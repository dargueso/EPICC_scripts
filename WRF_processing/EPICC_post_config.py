# File with EPICC postprocessing input arguments
# To be used with EPICC WRF postprocessing package based on WRF-python

path_wrfo = "/home/dargueso/WRF_OUT/"
path_proc = "/home/dargueso/postprocessed/EPICC/temp"
path_unif = "/home/dargueso/postprocessed/EPICC/"
path_geo = "/home/dargueso/share/geo_em_files/EPICC/"
file_geo = "geo_em.d01.EPICC_2km_ERA5_HVC_GWD.nc"
file_ref = "/home/dargueso/WRF_OUT/EPICC_2km_ERA5//out/wrfout_d01_2013-01-01_00:00:00"
institution = "UIB"
wruns = ["EPICC_2km_ERA5"]


patt = "wrf3hrly"
dom = "d01"

syear = 2011
eyear = 2021
smonth = 1
emonth = 1
acc_dt = 10
nproc_x = 12
nproc_y = 48

variables = ["PSL"]

# Fixed heights (metres) for the on-the-fly height interpolation of
# WAZ (vertical wind), TKZ (temperature, K) and PZ (pressure, Pa).
# zlevels_vcoord: "agl" = above ground level, "msl" = above mean sea level
zlevels_m = [500, 1000, 1500, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000, 13000]
zlevels_vcoord = "msl"

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
# WAZ
# TKZ
# PZ
# OMEGA
# PW
# TVAVG
# PSFC
