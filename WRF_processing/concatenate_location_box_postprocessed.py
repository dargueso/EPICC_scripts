#!/usr/bin/env python
'''
@File    :  concatenate_location_box_postprocessed.py
@Time    :  2025/11/08 23:55:08
@Author  :  Daniel Argüeso
@Version :  1.0
@Contact :  d.argueso@uib.es
@License :  (C)Copyright 2022, Daniel Argüeso
@Project :  None
@Desc    :  None
'''

import xarray as xr
import epicc_config as cfg
from glob import glob
wrun = cfg.wrf_runs[1]
###########################################################
###########################################################
locs_x_idx = [559,423,569,795,638,821,1091,989]#,433,866,335]
locs_y_idx = [258,250,384,527,533,407,174,425]#,254,506,119]
locs_names = ['Mallorca','Turis','Pyrenees','Rosiglione', 'Ardeche','Corte','Catania',"L'Aquila"]#,'Valencia','Barga','Almeria']
freq = '01H'

###########################################################
###########################################################

for loc in range(len(locs_names)):

    loc_name = locs_names[loc]
    print(loc_name)
    xloc = locs_x_idx[loc]
    yloc = locs_y_idx[loc]

    filesin = sorted(glob(f'/{cfg.path_in}/{wrun}/{cfg.patt_in}_{freq}_RAIN_20??-??_{yloc:3d}y-{xloc:3d}x_010buffer.nc'))

    data_list = []
    for fin in filesin:
        ds = xr.open_dataset(fin).squeeze()
        data_list.append(ds)

    combined = xr.concat(data_list, dim='time')
    combined.to_netcdf(f'/{cfg.path_in}/{wrun}/{cfg.patt_in}_{freq}_RAIN_{yloc:3d}y-{xloc:3d}x_010buffer.nc')