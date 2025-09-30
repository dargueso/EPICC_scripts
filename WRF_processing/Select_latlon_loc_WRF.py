import netCDF4 as nc    
import numpy as np
import wrf


ilat = 39.639
ilon = 2.647

# Open the NetCDF file
ncfile = nc.Dataset("/home/dargueso/share/geo_em_files/EPICC/geo_em.d01.EPICC_2km_ERA5_HVC_GWD.nc")

# Extract the lat/lon values

ix, iy = wrf.ll_to_xy(ncfile, ilat, ilon)

print(ix.values, iy.values)

# Close the file

ncfile.close()

# Use cdo to extract the region

# for file in $(ls UIB_01H_TAS*); do cdo selindexbox,490,524,334,392 ${file} Segarra_for_LluisAldoma/${file}; done