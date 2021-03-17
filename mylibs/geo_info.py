#!/usr/bin/env python
"""
#####################################################################
# Author: Daniel Argueso <daniel>
# Date:   2021-03-17T09:48:42+01:00
# Email:  d.argueso@uib.es
# Last modified by:   daniel
# Last modified time: 2021-03-17T09:50:37+01:00
#
# @Project@ EPICC
# Version: 1.0
# Description:Retrieve geographical information fro geo_em or wrfout
#
# Dependencies:
#
# Files:
#
#####################################################################
"""



import xarray as xr
import numpy as np
import netCDF4 as nc


import config as cfg

def get_corners_region(region):
    lats = np.asarray(cfg.reg_coords[region])[np.array([0,2])]
    lons = np.asarray(cfg.reg_coords[region])[np.array([1,3])]

    return lats,lons

def get_corners_csect(csect):
    lats = np.asarray(cfg.crosssect_coords[csect])[np.array([0,2])]
    lons = np.asarray(cfg.crosssect_coords[csect])[np.array([1,3])]

    return lats,lons

def get_loc_coords(loc):
    lat= np.asarray(cfg.loc_coords[loc])[np.array([0])]
    lon = np.asarray(cfg.loc_coords[loc])[np.array([1])]

    return lat,lon

def get_res(wrun):
    return int(wrun.rpartition('km')[0].split('_')[-1])

def get_zb_res():
    all_res = np.asarray([get_res(wrun) for wrun in cfg.wrf_runs])
    zb_res = np.int(cfg.zb*np.max(all_res)/int(cfg.ref_res))

    return zb_res




class geowrf_info():

    def get_geofile(self,wrun):
       geofile = xr.open_dataset("%s/geo_em.d01.%s.nc" %(cfg.geoem_in,wrun))
       geofile.coords['south_north']=geofile.XLAT_M.values[0,:,0]
       geofile.coords['west_east']=geofile.XLONG_M.values[0,0,:]

       return geofile

    def get_latlon(self,wrun):
        geofile_wrun = self.get_geofile(wrun)
        lat_ref = geofile_wrun['XLAT_M'].isel(Time=0)
        lon_ref = geofile_wrun['XLONG_M'].isel(Time=0)

        return lat_ref,lon_ref

    def get_latlon_region(self,wrun,region):
        lats,lons = get_corners_region(region)
        lat_ref, lon_ref = self.get_latlon(wrun)
        lat_ref_region = lat_ref.isel(west_east=0).sel(south_north=slice(lats[0],lats[1]))
        lon_ref_region = lon_ref.isel(south_north=0).sel(west_east=slice(lons[0],lons[1]))

        return lat_ref_region,lon_ref_region

    def get_var(self,wrun,var):
        geofile_wrun = self.get_geofile(wrun)
        return geofile_wrun[var].isel(Time=0)

    def get_var_region(self,wrun,var,region):
        lats,lons = get_corners_region(region)
        geofile_wrun = self.get_geofile(wrun)
        var = geofile_wrun[var].isel(Time=0).sel(south_north=slice(lats[0],lats[1]),\
                                                     west_east=slice(lons[0],lons[1]))

        return var

    def get_var_loc_interp(self,wrun,var,region,lat_ref,lon_ref):
        var = self.get_var_region(wrun,var,region)
        varint = var.interp(south_north=lat_ref,west_east=lon_ref,method='nearest')

        return varint

    def get_wrfout(self,wrun):
        wrfout_ref = nc.Dataset("%s/%s/out/wrfout_d01_2020-08-01_00:00:00" %(cfg.path_wrfout,wrun))
        return wrfout_ref
