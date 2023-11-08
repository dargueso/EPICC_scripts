# File with EPICC postprocessing input arguments
# To be used with EPICC WRF postprocessing package based on WRF-python

path_wrfo = "/vg9a/dargueso/WRF_OUT"
path_proc = "/home/yseut/MetMed_work/data/WRF_atmo_variables/"
path_unif = "/home/yseut/MetMed_work/data/WRF_atmo_variables/"
path_geo = "/home/yseut/MetMed_work/data/WRF_atmo_variables/"
geofile_ref = "geo_em.d01.EPICC_2km_ERA5_HVC_GWD.nc"
file_ref = (
    "/home/yseut/MetMed_work/data/WRF_atmo_variables/wrfout_d01_2011-12-22_00:00:00"
)

# path_wrfo = "/vg6a/dargueso/WRF_OUT/"
# path_proc = "/vg6a/dargueso/postprocessed/EPICC/temp"
# path_unif = "/vg6a/dargueso/postprocessed/EPICC/"
# path_geo = "/home/dargueso/share/geo_em_files/EPICC/"
# file_geo = "geo_em.d01.EPICC_2km_ERA5_HVC_GWD.nc"
# file_ref = "/vg6a/dargueso/WRF_OUT/EPICC_2km_ERA5//out/wrfout_d01_2013-01-01_00:00:00"
institution = "UIB"
wruns = ["EPICC_2km_ERA5"]
# /home/yseut/MetMed_work/data/WRF_atmo_variables

patt = "wrf3hrly"  # (wrfprec, wrfout, wrf3hrly)
dom = "d01"

syear = 2017
eyear = 2020
smonth = 1
emonth = 12
# acc_dt = 10
nproc_x = 12
nproc_y = 48

plevs = [850]  # [1000] #[850]
variables = ["RV"] #["PSL"]  # ["RV"] # ["PVO"] #["Z"]

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
