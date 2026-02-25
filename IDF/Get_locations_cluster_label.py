#!/usr/bin/env python
'''
@File    :  Untitled-1
@Time    :  2025/03/29 09:51:18
@Author  :  Daniel Argüeso
@Version :  1.0
@Contact :  d.argueso@uib.es
@License :  (C)Copyright 2022, Daniel Argüeso
@Project :  None
@Desc    :  None
This requires the data to be previously processed Calc_RFA_clusters_Wards.py
'''

import xarray as xr
import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from glob import glob
import epicc_config as cfg
from joblib import Parallel, delayed

wruns = ['EPICC_2km_ERA5','EPICC_2km_ERA5_CMIP6anom']
wrunp = wruns[0]
wrunf = wruns[1]

all_locs = ['Mallorca', 'Turis', 'Pyrenees']

xidx = {'Mallorca': 559, 'Turis': 423, 'Pyrenees': 569}
yidx = {'Mallorca': 258, 'Turis': 250, 'Pyrenees': 384}

#####################################################################
#####################################################################

def get_locations_cluster(filein,clusters_map,all_locs,cluster_locs):
    
    """
    Get the locations of the cluster from the input file
    """
    print(f'Processing {filein}')
    fin_all = xr.open_dataset(filein)
    
    for loc in all_locs:
        fout = filein.replace('.nc', f'_cluster_{loc}_{cluster_locs[loc]}.nc')
        fin = fin_all.where(clusters_map.ClusterID == cluster_locs[loc], drop=True)
        #Stack lat lon to  locations
        fin = fin.stack(locations=['y', 'x'])
        fin = fin.rename({'locations': 'loc'})
        
        # Reset the index of 'loc' to avoid MultiIndex serialization issues
        fin = fin.reset_index('loc')
        fin = fin.reset_coords(drop=True)
        # Ensure 'loc' is not a MultiIndex before saving
        if isinstance(fin.indexes['loc'], xr.core.indexes.PandasMultiIndex):
            fin = fin.reset_index('loc')
        fin.to_netcdf(fout)


def main():
    
      
    # filesinp = sorted(glob(f'{cfg.path_in}/{wrunp}/{cfg.patt_in}_10MIN_RAIN_????_daymax.nc'))
    filesinp = sorted(glob(f'{cfg.path_in}/{wrunp}/{cfg.patt_in}_MON_RAIN_????-??.nc'))
      
    clusters_map = xr.open_dataset('./ward_clusters_EPICC_2km_ERA5_301.nc')
    cluster_locs = {}
    for loc in all_locs:
        cluster_locs[loc] = clusters_map.sel(y=yidx[loc], x=xidx[loc]).ClusterID.item()
        cluster_loc_out = clusters_map.where(clusters_map.ClusterID == cluster_locs[loc], 0)
        cluster_loc_out = cluster_loc_out.rename({'ClusterID': loc})
        cluster_loc_out.to_netcdf(f'./cluster_{loc}.nc')
    Parallel(n_jobs=1)(delayed(get_locations_cluster)(filein,clusters_map,all_locs,cluster_locs) for filein in filesinp)

###############################################################################
##### __main__  scope
###############################################################################

if __name__ == "__main__":

    main()


###############################################################################
