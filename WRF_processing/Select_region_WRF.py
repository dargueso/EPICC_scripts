import netCDF4 as nc    
import numpy as np
import wrf


bllat = 41
bllon = 0.9
urlat = 42.1
urlon = 1.7

# Open the NetCDF file
ncfile = nc.Dataset("/home/dargueso/share/wrf_ref_files/EPICC/wrfout_d01_2011-12-22_00:00:00")

# Extract the lat/lon values

blx, bly = wrf.ll_to_xy(ncfile, bllat, bllon)
urx, ury = wrf.ll_to_xy(ncfile, urlat, urlon)

print(blx, bly, urx, ury)

# Close the file

ncfile.close()

# Use cdo to extract the region

# for file in $(ls UIB_01H_TAS*); do cdo selindexbox,490,524,334,392 ${file} Segarra_for_LluisAldoma/${file}; done