#!/usr/bin/env python
"""
#####################################################################
# Author: Daniel Argueso <daniel>
# Date:   2022-03-23T15:56:40+01:00
# Email:  d.argueso@uib.es
# Last modified by:   daniel
# Last modified time: 2022-03-23T15:56:45+01:00
#
# @Project@ EPICC
# Version: x.0 (Beta)
# Description: Program to imitate MODE-TD (METV10) - based on Andreas Prein code
# THis program reads EPICC postprocessed info
# The input data is smoothed in 3D and a precipitation threshold is applied
# The resulting objects are tracked over space and time
# The location and track of the objects is stored.
# Dependencies:
#
# Files:
# Postprocessed files (RAIN)
#
#####################################################################
"""

import xarray as xr
import numpy as np
import epicc_config as cfg
from glob import glob
from scipy.ndimage import gaussian_filter, uniform_filter, label, measurements
from itertools import product
import time
import pickle


from joblib import Parallel, delayed

###########################################################
###########################################################

wrf_runs = cfg.wrf_runs
lsmooth = 50000 #Smoothing filter length in meters
dx = 2000 #Resolution of imput data
lsmooth_ngrid = np.ceil(lsmooth/dx)
pr_threshold_mmhr = 2.5 #Precipitation threshold in mm/h
freq = '01H'

mmh_factor = {'10MIN': 6.,
              '01H'  : 1.,
              'DAY'  : 1/24.}

freq_dt = {'10MIN': 600,
          '01H'  : 3600,
          'DAY'  : 86400.}

pr_threshold = pr_threshold_mmhr/mmh_factor[freq]
dt = freq_dt[freq] #Step between model outputs in seconds
print(pr_threshold)

save_storm_files = False
calc_storm_properties = True

###########################################################
###########################################################

def main():


    for wrun in wrf_runs:

        filesin = sorted(glob(f'{cfg.path_in}/{wrun}/{cfg.patt_in}_{freq}_RAIN_20??-??.nc'))

        Parallel(n_jobs=6)(delayed(track_mcs)(fin_name,lsmooth_ngrid,pr_threshold,freq,wrun) for fin_name in filesin)

###########################################################
###########################################################

def track_mcs(fin_name,lsmooth_ngrid,pr_threshold,freq,wrun):

    """ This program tracks MCS from precipitation WRF outputs """

    print(f'Processing {fin_name}')

    pr = xr.open_dataset(fin_name).load()
    pr.coords['x']=range(len(pr.x))
    pr.coords['y']=range(len(pr.y))

    #GAUSSIAN
    #pr_smooth = gaussian_filter(pr_mmhr.squeeze(),(3,3),mode='nearest')

    #UNIFORM
    #pr_smooth = pr.RAIN.values.copy()
    pr_smooth = uniform_filter(pr.RAIN.squeeze().values,[1,lsmooth_ngrid,lsmooth_ngrid],mode='nearest')


    pr_thres_mask = (pr_smooth>=pr_threshold)
    pr_thres_smooth = np.where(pr_smooth>=pr_threshold,pr_smooth,0)


    prObj_Struct = np.ones((3,3,3))

    storm_id,nstorms = label(pr_thres_mask,structure=prObj_Struct)

    pr['storm_id']=(['time','y','x'],  storm_id)
    pr['pr_smooth']=(['time','y','x'],  pr_smooth)

    if save_storm_files:

        fout_name = fin_name.replace('RAIN','storms')
        fout_name = fout_name.replace('.nc',f'_thres-{pr_threshold_mmhr}mmhr_smooth-{lsmooth/1000}km.nc')
        pr.to_netcdf(fout_name,encoding={'storm_id':{'zlib': True,'complevel': 5},'RAIN':{'zlib': True,'complevel': 5},'pr_smooth':{'zlib': True,'complevel': 5}})
        for nst in range(1,nstorms+1):

            #print(f'Saving storm {nst:03d}')
            fout_name_nst = fin_name.replace('.nc',f'_STORM-{nst:03d}_thres-{pr_threshold_mmhr}mmhr_smooth-{lsmooth/1000}km.nc')
            pr_storm_id = pr.where(pr.storm_id==nst).dropna(dim='time',how='all').dropna(dim='x',how='all').dropna(dim='y',how='all')
            pr.sel(time=pr_storm_id.time,x=pr_storm_id.x,y=pr_storm_id.y).to_netcdf(fout_name_nst)

    if calc_storm_properties:
        #Only if there are storms
        if nstorms>=1:

            pr_all = pr.RAIN.values

            for nst in range(1,nstorms+1):

                #print (f'Storm #{nst}')
                start_time = time.time()

                pr_storm = pr_all.copy()

                this_storm = (storm_id == nst)
                time_storm = this_storm.any(axis=(1,2))
                valid_times = np.argwhere(time_storm).squeeze()
                if valid_times.size==1:
                    valid_times = valid_times.reshape(1)

                pr_storm[~this_storm] = 0
                #Does the storm hit the border of the domain?
                hit_border = (np.sum(this_storm[:,:,0],axis=1) +
                              np.sum(this_storm[:,:,-1],axis=1) +
                              np.sum(this_storm[:,0,:],axis=1) +
                              np.sum(this_storm[:,-1,:],axis=1)) != 0

                #Remove times when storm hit the border
                #pr_storm[hit_border,:,:]=0
                #this_storm[hit_border,:,:]=False

                # if time_storm[-1]:
                #     print(f'Storm #{nst} reaches the end of the file {file}')
                #     import pdb; pdb.set_trace()
                if valid_times.size>0:
                    #Calculate mass center for propagation metrics
                    storm_mass_center = np.ones((valid_times.size,2))*np.nan
                    storm_speed = np.ones(valid_times.size)*np.nan
                    storm_prvol = np.ones(valid_times.size)*np.nan
                    storm_prmax = np.ones(valid_times.size)*np.nan
                    storm_prmean = np.ones(valid_times.size)*np.nan
                    storm_prtile = np.ones((valid_times.size,101))*np.nan
                    storm_size = np.ones(valid_times.size)*np.nan
                    storm_size_all = np.sum(this_storm)
                    storm_prmean_all = pr_storm.mean()*mmh_factor[freq]
                    storm_prtile_all = np.percentile(pr_storm[this_storm].flatten(),range(101))*mmh_factor[freq]
                    storm_duration=valid_times.size*dt/3600.
                    storm_hitborder = hit_border

                    if valid_times.size>1:
                        storm_start = pr.isel(time=valid_times[0]).time.dt.strftime("%Y-%m-%d_%H:%M").item()
                        storm_end = pr.isel(time=valid_times[-1]).time.dt.strftime("%Y-%m-%d_%H:%M").item()
                    else:
                        storm_start = pr.isel(time=valid_times).time.dt.strftime("%Y-%m-%d_%H:%M").item()
                        storm_end = storm_start

                    if time_storm[-1]:
                        storm_hitend = True
                    else:
                        storm_hitend = False



                    for nt,tt in enumerate(valid_times):
                        storm_mass_center[nt,:] = measurements.center_of_mass(pr_storm[tt,:,:])
                        storm_prvol[nt] = np.sum(pr_storm[tt,:,:])
                        storm_prmax[nt] = np.max(pr_storm[tt,:,:])*mmh_factor[freq]
                        storm_prmean[nt] = np.mean(pr_storm[tt,:,:])*mmh_factor[freq]
                        storm_prtile[nt,:] = np.percentile(pr_storm[tt,:,:][this_storm[tt,:,:]],range(101))*mmh_factor[freq]
                        storm_size[nt] = np.sum(this_storm[tt,:,:])*(dx/1000.)**2
                    for nt,tt in enumerate(valid_times[:-1]):
                        storm_speed[nt] = (((storm_mass_center[nt,0]-storm_mass_center[nt+1,0])**2 +
                                            (storm_mass_center[nt,1]-storm_mass_center[nt+1,1])**2))**(0.5)*dx*3600./(dt*1000.)


                    storm_properties = {'storm_mass_center':storm_mass_center,
                                        'storm_speed':storm_speed,
                                        'storm_prvol': storm_prvol,
                                        'storm_prmax': storm_prmax,
                                        'storm_prmean': storm_prmean,
                                        'storm_prtile': storm_prtile,
                                        'storm_size': storm_size,
                                        'storm_size_all':storm_size_all,
                                        'storm_prmean_all': storm_prmean_all,
                                        'storm_prtile_all': storm_prtile_all,
                                        'storm_duration':storm_duration,
                                        'storm_start': storm_start,
                                        'storm_end': storm_end,
                                        'storm_hitend':storm_hitend,
                                        'storm_hitborder':storm_hitborder}



                    fout_name = f'{cfg.path_in}/{wrun}/Storm_properties_{pr.isel(time=0).time.dt.year.item()}-{pr.isel(time=0).time.dt.month.item():02d}_no{nst:03d}_thres-{pr_threshold_mmhr}mmhr_smooth-{lsmooth/1000}km.pkl'
                    pickle.dump(storm_properties,open(fout_name,'wb'))

                end_time = time.time()
                #print(f'======> DONE storm in {(end_time-start_time):.2f} seconds \n')
                #start and end
                #trajectory
                #duration



###############################################################################
##### __main__  scope
###############################################################################

if __name__ == "__main__":

    main()

###########################################################
###########################################################
