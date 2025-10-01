#!/usr/bin/env python
'''
@File    :  select_multipleregion_WRF_shapefile.py
@Time    :  2025/09/30 15:21:17
@Author  :  Daniel Argüeso
@Version :  1.0
@Contact :  d.argueso@uib.es
@License :  (C)Copyright 2025, Daniel Argüeso
@Project :  None
@Desc    :  None
'''
import xarray as xr
import numpy as np
import netCDF4 as nc
import geopandas as gpd
import regionmask

from wrf import (to_np, getvar,get_cartopy, cartopy_xlim,GeoBounds,CoordPair,
                 cartopy_ylim, latlon_coords)

geo_file_name = "/home/dargueso/share/geo_em_files/EPICC/geo_em.d01.EPICC_2km_ERA5_HVC_GWD.nc"
geo_file = xr.open_dataset(geo_file_name)
lm_is=geo_file.LANDMASK.squeeze()

#####################################################################
#####################################################################


def get_geoinfo():

    fileref = nc.Dataset(geo_file_name)
    hgt = getvar(fileref, "HGT_M", timeidx=0)
    hgt = hgt.where(hgt>=0,0)
    lats, lons = latlon_coords(hgt)
    cart_proj = get_cartopy(hgt)

    return cart_proj,lats,lons,hgt

#####################################################################
#####################################################################

cart_proj,lats,lons,hgt = get_geoinfo()

med_mask = xr.open_dataset('/home/dargueso/postprocessed/EPICC/EPICC_2km_ERA5/coastal_med_mask.nc')


ocean_shapfile = '/home/dargueso/Misc_data/Ocean_regions_GOaS_v1_20211214/goas_v01.shp'

ocean_regions = {'MED': 6}
oc_values = list(ocean_regions.values())
ds_ocean = gpd.read_file(ocean_shapfile)
if ds_ocean.crs is None or ds_ocean.crs.to_epsg() != 4326:
    ds_ocean = ds_ocean.to_crs(epsg=4326)
if 'ocean_id' not in ds_ocean.columns:
    ds_ocean['ocean_id'] = range(len(ds_ocean))

# Create a regionmask.Regions object
oceans = regionmask.Regions(
    outlines=ds_ocean.geometry,
    numbers=ds_ocean['ocean_id'],
    names=ds_ocean['ocean_id'].astype(str)  # Use the IDs as names
)

# Generate the mask with unique numbers for each region
mask_ocean = oceans.mask(lon_or_obj=lons, lat=lats)
medsea_mask = mask_ocean.isin(oc_values)

#####################################################################
#####################################################################

mask_data = med_mask['combined_mask'].astype('int8')
mask_data[:,:193]=0

rightmost_index = mask_data.shape[1] - 1 - np.argmax(np.any(mask_data.values, axis=0)[::-1])
leftmost_index = np.argmax(np.any(mask_data.values, axis=0))
bottom_index = np.argmax(np.any(mask_data.values, axis=1))
top_index = mask_data.shape[0] - 1 - np.argmax(np.any(mask_data.values, axis=1)[::-1])
print(leftmost_index, rightmost_index, bottom_index, top_index)

medsea_mask[:, rightmost_index+1:] = 0
medsea_mask[:, :leftmost_index] = 0
medsea_mask[:bottom_index, :] = 0

mask_data.values[medsea_mask.values == 1] = 2

region_sq = np.zeros_like(med_mask['combined_mask'].values)
region_sq[bottom_index:top_index+1,leftmost_index:rightmost_index+1]=1
mask_data+= region_sq




output_ds = med_mask.copy(deep=True)
output_ds['combined_mask'] = output_ds['combined_mask'].astype('int8')

# Update the mask data
output_ds['combined_mask'].values = mask_data.values

# Write to netCDF file
output_filename = '/home/dargueso/postprocessed/EPICC/EPICC_2km_ERA5/my_coastal_med_mask.nc'
output_ds.to_netcdf(output_filename)