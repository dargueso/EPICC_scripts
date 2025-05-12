#!/usr/bin/env python
'''
@File    :  optimized_cluster_extraction.py
@Time    :  2025/03/29 09:51:18
@Author  :  Daniel Argüeso
@Contact :  d.argueso@uib.es
@Desc    :  Efficient extraction of cluster-based locations using lazy loading
'''

import xarray as xr
import numpy as np
from tqdm.auto import tqdm
from glob import glob
import epicc_config as cfg
from joblib import Parallel, delayed

# Configuration
wruns = ['EPICC_2km_ERA5', 'EPICC_2km_ERA5_CMIP6anom']
wrunp = wruns[0]
#wrunf = wruns[1]  # not used in this script

all_locs = ['Mallorca', 'Turis', 'Pyrenees']
xidx = {'Mallorca': 559, 'Turis': 423, 'Pyrenees': 569}
yidx = {'Mallorca': 258, 'Turis': 250, 'Pyrenees': 384}

def main():



    filesinp = sorted(glob(f'{cfg.path_in}/{wrunp}/{cfg.patt_in}_01H_RAIN_????-??.nc'))
    clusters_map = xr.open_dataset('./ward_clusters_EPICC_2km_ERA5_301.nc')
    cluster_locs = {}
    for loc in all_locs:
        cluster_id = clusters_map.sel(y=yidx[loc], x=xidx[loc]).ClusterID.item()
        cluster_locs[loc] = cluster_id
        
    for filein in filesinp:
        fin = xr.open_dataset(filein)
        for loc in all_locs:
            print(f"Processing file: {filein}")
            fin_sel = fin.where(clusters_map.ClusterID == cluster_locs[loc]).stack(loc=('y', 'x')).reset_index('loc').dropna(dim='loc')
            # fin_sel.to_netcdf(f'{cfg.path_in}/{wrunp}/UIB_01H_RAIN_{loc}.nc')
            fout = filein.replace('.nc', f'_cluster_{loc}_{cluster_locs[loc]}.nc')
            fin_sel.to_netcdf(fout)
            fin_sel.close()


if __name__ == "__main__":

    main()
