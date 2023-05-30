#!/usr/bin/env python
"""
#####################################################################
# Author: Daniel Argueso <daniel>
# Date:   2022-03-18T10:40:13+01:00
# Email:  d.argueso@uib.es
# Last modified by:   daniel
# Last modified time: 2022-03-18T13:08:19+01:00
#
# @Project@ EPICC
# Version: 1.0
# Description: Script to extract an area from CMORPH netCDF files. It also does
# file changes to put longitudes in -180 to 180 format, and also adds a record dimension
# for concatenating.
# Example: ./extract_MEDarea_CMORPH.py -s 2011 e -2011
# Dependencies:
#
# Files:
#
#####################################################################
"""
import subprocess
import os
import optparse as opt
import glob
from tqdm import tqdm


### Options
parser = opt.OptionParser()
parser.add_option("-s", "--syear", dest="syear",help="starting year", metavar="OPTION")
parser.add_option("-e", "--eyear", dest="eyear",help="ending year", metavar="OPTION")

(opts, args) = parser.parse_args()
###
syear=int(opts.syear)
eyear=int(opts.eyear)


path_in="/home/yseut/data/CMORPH_V1.0/"
path_out="/vg6/dargueso-NO-BKUP/OBS_DATA/CMORPH/CRT/EPICC/"
if not os.path.exists(path_out):
    os.makedirs(path_out)
if not os.path.exists(f"{path_out}/rec_dmn"):
    os.makedirs(f"{path_out}/rec_dmn")
for year in range(syear,eyear+1):
    for month in range(1,13):
        print (f"{year}/{month}")
        files_ym = sorted(glob.glob(f'{path_in}/CMORPH_V1.0_ADJ_8km-30min_{year}{month:02d}*.nc'))


        for filein in tqdm(files_ym):

                short_filename = filein.split("/")[-1]
                subprocess.call(f'ncks -O --msa -d lon,181.,360. -d lon,0.,180.0 {filein} {path_out}/out.nc', shell=True)
                subprocess.call(f"ncap2 -O -s 'where(lon > 180) lon=lon-360' {path_out}/out.nc {path_out}/out.nc", shell=True)
                subprocess.call(f"ncks --mk_rec_dmn time {path_out}/out.nc {path_out}/out_time.nc", shell=True)
                subprocess.call(f'ncea -d lat,30.0,55.0 -d lon,-15.0,25.0 {path_out}/out_time.nc {path_out}/rec_dmn/MED_{short_filename}', shell=True)
                subprocess.call(f"rm {path_out}/out.nc {path_out}/out_time.nc", shell=True)
        print(f"Concatenating files: {year}-{month}")
        subprocess.call(f"ncrcat {path_out}/rec_dmn/MED_CMORPH_V1.0_ADJ_8km-30min_{year}{month:02d}* {path_out}/MED_CMORPH_V1.0_ADJ_8km-30min_{year}{month:02d}.nc",shell=True)
        subprocess.call(f"rm {path_out}/rec_dmn/MED_CMORPH_V1.0_ADJ_8km-30min_{year}{month:02d}*", shell=True)
