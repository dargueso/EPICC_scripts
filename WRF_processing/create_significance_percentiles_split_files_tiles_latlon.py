#!/usr/bin/env python
"""
#####################################################################
# Author: Daniel Argueso <daniel>
# Date:   2021-06-17T11:53:02+02:00
# Email:  d.argueso@uib.es
# Last modified by:   daniel
# Last modified time: 2021-06-17T11:53:43+02:00
#
# @Project@ EPICC
# Version: 1.0
# Description: Script to calculate percentiles and wet-percentiles from postprocessed files
#
# Dependencies:
#
# Files:
#
#####################################################################
"""

import xarray as xr
import numpy as np
import epicc_config as cfg
from glob import glob
from itertools import product
from joblib import Parallel, delayed
import scipy.stats as sc

wrf_runs = ['EPICC_2km_ERA5']#,'EPICC_2km_ERA5_CMIP6anom']
qtiles = np.asarray(cfg.qtiles)
mode = 'wetonly'
wet_value = 1
tile_size = 50
###########################################################
###########################################################

def main():

    """ Calculating percentiles using a loop"""
    for wrun in wrf_runs:
      for fq in ['DAY']:#'10MIN','01H']:#,'DAY']:
        filesin = sorted(glob(f'{cfg.path_in}/{wrun}/{cfg.patt_in}_{fq}_RAIN_20??-??.nc'))
        files_ref = xr.open_dataset(filesin[0])
        nlats = files_ref.sizes['y']
        nlons = files_ref.sizes['x']

        lonsteps = [f'{nn:03d}' for nn in range(nlons//tile_size+1)]
        latsteps = [f'{nn:03d}' for nn in range(nlats//tile_size+1)]

        xytiles=list(product(latsteps, lonsteps))

        filespath = f'{cfg.path_in}/{wrun}/split_files_tiles_50/{cfg.patt_in}_{fq}_RAIN_20??-??'
        print(f'Ej: {filespath}_000y-000x.nc')


        Parallel(n_jobs=20)(delayed(calc_significance)(cfg.path_in,xytile[0],xytile[1],qtiles,fq,wrun,mode) for xytile in xytiles)




#####################################################################
#####################################################################

def calc_significance(filespath,ny,nx,qtiles,fq,wrun,mode='wetonly'):
  print (f'Analyzing tile y: {ny} x: {nx}')

  filespath_p = f'{filespath}/{wrun}/split_files_tiles_50/{cfg.patt_in}_{fq}_RAIN_20??-??'
  filespath_f = filespath_p.replace('EPICC_2km_ERA5','EPICC_2km_ERA5_CMIP6anom')
  filesin_p = sorted(glob(f'{filespath_p}_{ny}y-{nx}x.nc'))
  filesin_f = sorted(glob(f'{filespath_f}_{ny}y-{nx}x.nc'))
  finp = xr.open_mfdataset(filesin_p,concat_dim="time", combine="nested").load().sel(time=slice(str(cfg.syear),str(cfg.eyear)))
  finf = xr.open_mfdataset(filesin_f,concat_dim="time", combine="nested").load().sel(time=slice(str(cfg.syear),str(cfg.eyear)))


  if mode == 'wetonly':
    finq_p = finp.RAIN.where(finp.RAIN>wet_value).quantile(qtiles,dim=['time'])
    finq_f = finf.RAIN.where(finf.RAIN>wet_value).quantile(qtiles,dim=['time'])     
  else:
    finq_p = finp.RAIN.quantile(qtiles,dim=['time'])
    finq_f = finf.RAIN.quantile(qtiles,dim=['time'])

  qtiles_p = finq_p.coords['quantile'].values
  qtiles_f = finq_f.coords['quantile'].values

  if qtiles_p.all()== qtiles_f.all():
     qtiles = qtiles_p
  else:
     raise ValueError('Percentile tiles are different in both simulations')
  
  ext_p = finp.RAIN.where(finp.RAIN>=finq_p)
  ext_f = finf.RAIN.where(finf.RAIN>=finq_f)

  sig_var = np.zeros(ext_p.shape[1:],dtype=float)
  next_p = ext_p.count('time')
  next_f = ext_f.count('time')
 

  for nq, qtile in enumerate(qtiles):
    for i in range(sig_var.shape[0]):
      print("line "+str(i)+" of "+str(sig_var.shape[0]))
      for j in range(sig_var.shape[1]):
          
          vp = ext_p.values[:,i,j,nq]
          vf = ext_f.values[:,i,j,nq]
          vp = vp[np.isfinite(vp)]
          vf = vf[np.isfinite(vf)]

          if vp.size==0 or vf.size==0:
              if vp.size==0 and vf.size==0:
                sig_var[i,j,nq]=1.0
              else:
                sig_var[i,j,nq]=0.0
          else:
            tvar,sig_var[i,j,nq]=sc.ks_2samp(vp,vf)
          # if vp.size!=0 and vf.size!=0:
          #   #tvar,sig_var[i,j,nq]=sc.mannwhitneyu(vp,vf,alternative='two-sided')
          #   tvar,sig_var[i,j,nq]=ks_2sample(vp,vf)
          # else:
          #   continue
  sig_da = xr.DataArray(sig_var,coords=[finp.y,finp.x,qtiles],dims=['y','x','quantile'])
  next_p = xr.DataArray(next_p,coords=[finp.y,finp.x,qtiles],dims=['y','x','quantile'])
  next_f = xr.DataArray(next_f,coords=[finp.y,finp.x,qtiles],dims=['y','x','quantile'])
  ptiles_p = xr.DataArray(finq_p,coords=[qtiles,finp.y,finp.x,],dims=['quantile','y','x'])
  ptiles_f = xr.DataArray(finq_f,coords=[qtiles,finf.y,finf.x],dims=['quantile','y','x'])
  #Create dataset
  output_ds = xr.Dataset({
    'significance': sig_da,
    'number_extreme_present': next_p,
    'number_extreme_future': next_f,
    'percentiles_present': ptiles_p,
    'percentiles_future': ptiles_f
})
  fout = f'{cfg.path_in}/{wrun}/{cfg.patt_in}_{fq}_RAIN_{cfg.syear}-{cfg.eyear}_{ny}y-{nx}x_qtiles_{mode}_sig.nc'
  output_ds.to_netcdf(fout)

  finp.close()
  finf.close()
###############################################################################
##### __main__  scope
###############################################################################

if __name__ == "__main__":

    main()


###############################################################################
