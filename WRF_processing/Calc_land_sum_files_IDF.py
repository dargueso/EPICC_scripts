import xarray as xr
import numpy as np
import netCDF4 as nc

import epicc_config as cfg

#####################################################################
#####################################################################

#GEO INFO
geo_file = xr.open_dataset(cfg.geofile_ref)
lats = geo_file.XLAT_M.squeeze().values
lons = geo_file.XLONG_M.squeeze().values
lm_is=geo_file.LANDMASK.squeeze()
lm_is=lm_is.rename({'west_east':'x','south_north':'y'})
lm_is.coords['y']=lats[:,0]
lm_is.coords['x']=lons[0,:]

reg ='WME'

#####################################################################
#####################################################################

wrun_pre = cfg.wrf_runs[0]
wrun_fut = wrun_pre.replace("ERA5","ERA5_CMIP6anom")

fin_pre = xr.open_dataset(f'{cfg.path_in}/{wrun_pre}/hist2d_spell_IFD_2013-2020_1.0mm.nc')
fin_fut = xr.open_dataset(f'{cfg.path_in}/{wrun_fut}/hist2d_spell_IFD_2013-2020_1.0mm.nc')
#fin_fut = xr.open_dataset(f'{cfg.path_in}/{wrun_fut}/hist2d_IFD_spell_2013-2020.nc')
#fin_fut = xr.open_dataset(f'{cfg.path_in}/{wrun_fut}/hist2d_IFD_resample_time_2013-2020.nc')


#####################################################################
#####################################################################

if reg=='':

    fin_pre_land = fin_pre.where(lm_is==1).sum(dim=('x','y'))
    fin_fut_land = fin_fut.where(lm_is==1).sum(dim=('x','y'))
    fin_pre_land.to_netcdf(f'{cfg.path_in}/{wrun_pre}/hist2d_spell_IFD_2013-2020_land_sum_1.0mm.nc')
    fin_fut_land.to_netcdf(f'{cfg.path_in}/{wrun_fut}/hist2d_spell_IFD_2013-2020_land_sum_1.0mm.nc')


else:
    fin_pre_land = fin_pre.where(lm_is==1).sel(y=slice(cfg.reg_coords[reg][0],cfg.reg_coords[reg][2]),x=slice(cfg.reg_coords[reg][1],cfg.reg_coords[reg][3])).sum(dim=('x','y'))
    fin_fut_land = fin_fut.where(lm_is==1).sel(y=slice(cfg.reg_coords[reg][0],cfg.reg_coords[reg][2]),x=slice(cfg.reg_coords[reg][1],cfg.reg_coords[reg][3])).sum(dim=('x','y'))
    fin_pre_land.to_netcdf(f'{cfg.path_in}/{wrun_pre}/hist2d_spell_IFD_2013-2020_land_sum_{reg}_1.0mm.nc')
    fin_fut_land.to_netcdf(f'{cfg.path_in}/{wrun_fut}/hist2d_spell_IFD_2013-2020_land_sum_{reg}_1.0mm.nc')
