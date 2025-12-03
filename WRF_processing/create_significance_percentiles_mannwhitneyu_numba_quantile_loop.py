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
# Version: 3.0 (Optimized with Quantile Loop)
# Description: Mann-Whitney U test with quantile-by-quantile processing
#              Optimized for high-RAM systems (>512 GB)
#
# Key optimization: Process quantiles sequentially to avoid 4D arrays
# This allows more parallel tiles and better cache efficiency
#
#####################################################################
"""

import xarray as xr
import numpy as np
import epicc_config as cfg
from glob import glob
import os
from itertools import product
from joblib import Parallel, delayed
from numba import njit, prange
import gc
import time

wrf_runs = ['EPICC_2km_ERA5']
qtiles = np.asarray(cfg.qtiles)
mode = 'wetonly'
wet_value = 0.1
tile_size = 50

###########################################################
# Numba-accelerated Mann-Whitney U test implementation
###########################################################

@njit
def rank_data(data):
    """
    Assign ranks to data, handling ties by averaging ranks.
    """
    n = len(data)
    order = np.argsort(data)
    ranks = np.empty(n, dtype=np.float64)
    
    i = 0
    while i < n:
        j = i
        while j < n - 1 and data[order[j]] == data[order[j + 1]]:
            j += 1
        
        rank = (i + j + 2) / 2.0
        
        for k in range(i, j + 1):
            ranks[order[k]] = rank
        
        i = j + 1
    
    return ranks


@njit
def mannwhitneyu_statistic(x, y):
    """
    Compute Mann-Whitney U statistic and approximate p-value.
    
    Parameters
    ----------
    x, y : 1D arrays
        The two samples to compare
    
    Returns
    -------
    u_statistic : float
        The U statistic
    p_value : float
        Approximate two-sided p-value
    """
    n1 = len(x)
    n2 = len(y)
    
    if n1 == 0 or n2 == 0:
        return 0.0, 1.0
    
    combined = np.concatenate((x, y))
    ranks = rank_data(combined)
    rank_sum_x = np.sum(ranks[:n1])
    u_x = rank_sum_x - n1 * (n1 + 1) / 2.0
    u_y = n1 * n2 - u_x
    u = min(u_x, u_y)
    
    mean_u = n1 * n2 / 2.0
    std_u = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12.0)
    
    if std_u == 0:
        return u, 1.0
    
    if u > mean_u:
        z = (u - mean_u - 0.5) / std_u
    else:
        z = (u - mean_u + 0.5) / std_u
    
    abs_z = abs(z)
    
    if abs_z > 6.0:
        p_value = 0.0
    elif abs_z < 0.5:
        p_value = 1.0
    else:
        p_value = 2.0 * np.exp(-0.5 * abs_z * abs_z) / np.sqrt(2.0 * np.pi) * (1.0 / abs_z)
        p_value = min(1.0, p_value)
    
    return u, p_value


@njit(parallel=True, fastmath=True)
def compute_mwu_single_quantile(data_p, data_f, min_samples=5):
    """
    Compute Mann-Whitney U test p-values for a single quantile.
    
    Parameters:
    -----------
    data_p : ndarray
        Present data, shape (time, y, x)
    data_f : ndarray
        Future data, shape (time, y, x)
    min_samples : int
        Minimum number of valid samples required for test
    
    Returns:
    --------
    sig_var : ndarray
        P-values, shape (y, x)
    """
    nt, ny, nx = data_p.shape
    sig_var = np.zeros((ny, nx), dtype=np.float64)
    
    # Parallel loop over spatial points
    for i in prange(ny):
        for j in range(nx):
            # Extract time series for this point
            vp = data_p[:, i, j]
            vf = data_f[:, i, j]
            
            # Remove NaN values
            vp_valid = vp[~np.isnan(vp)]
            vf_valid = vf[~np.isnan(vf)]
            
            # Check if we have enough samples
            if len(vp_valid) < min_samples or len(vf_valid) < min_samples:
                if len(vp_valid) == 0 and len(vf_valid) == 0:
                    sig_var[i, j] = 1.0
                else:
                    sig_var[i, j] = 0.0
            else:
                # Compute Mann-Whitney U test
                _, pval = mannwhitneyu_statistic(vp_valid, vf_valid)
                sig_var[i, j] = pval
    
    return sig_var


###########################################################
# Main processing functions
###########################################################

def main():
    """Calculating percentiles using parallel processing"""
    for wrun in wrf_runs:
        for fq in ['10MIN']:  # '10MIN','01H','DAY'
            filesin = sorted(glob(f'{cfg.path_in}/{wrun}/{cfg.patt_in}_{fq}_RAIN_20??-??.nc'))
            files_ref = xr.open_dataset(filesin[0])
            nlats = files_ref.sizes['y']
            nlons = files_ref.sizes['x']
            files_ref.close()

            lonsteps = [f'{nn:03d}' for nn in range(nlons//tile_size+1)]
            latsteps = [f'{nn:03d}' for nn in range(nlats//tile_size+1)]

            xytiles = list(product(latsteps, lonsteps))

            print(f'Processing {len(xytiles)} tiles for {fq} data')
            print(f'Using 20 parallel jobs (quantile-loop optimization)')
            print(f'Statistical test: Mann-Whitney U (Numba-accelerated)')

            Parallel(n_jobs=10)(
                delayed(calc_significance_quantile_loop)(
                    cfg.path_in, xytile[0], xytile[1], 
                    qtiles, fq, wrun, mode
                ) for xytile in xytiles
            )


def calc_significance_quantile_loop(filespath, ny, nx, qtiles, fq, wrun, mode='wetonly'):
    """
    Calculate significance using quantile-by-quantile processing.
    
    This approach:
    - Loads all data into memory (fast with high RAM)
    - Processes one quantile at a time (avoids 4D arrays)
    - Better cache efficiency
    - Allows more parallel tiles
    """
    print(f'Analyzing tile y: {ny} x: {nx}')
    start_time = time.time()
    tile_id = f"{ny}y-{nx}x"
    #Define output file
    fout = f'{cfg.path_in}/{wrun}/{cfg.patt_in}_{fq}_RAIN_{cfg.syear}-{cfg.eyear}_{ny}y-{nx}x_qtiles_{mode}_sig_mwu.nc'
    if os.path.exists(fout):
        print(tile_id, "skipped", 0.0, "Already exists")
        return 
        

    filespath_p = f'{filespath}/{wrun}/split_files_tiles_50/{cfg.patt_in}_{fq}_RAIN_20??-??'
    filespath_f = filespath_p.replace('EPICC_2km_ERA5', 'EPICC_2km_ERA5_CMIP6anom')
    filesin_p = sorted(glob(f'{filespath_p}_{ny}y-{nx}x.nc'))
    filesin_f = sorted(glob(f'{filespath_f}_{ny}y-{nx}x.nc'))
    
    # Load data fully into memory (you have plenty of RAM)
    print(f'  Loading data...')
    finp = xr.open_mfdataset(
        filesin_p, 
        concat_dim="time", 
        combine="nested"
    ).load().sel(time=slice(str(cfg.syear), str(cfg.eyear)))
    
    finf = xr.open_mfdataset(
        filesin_f, 
        concat_dim="time", 
        combine="nested"
    ).load().sel(time=slice(str(cfg.syear), str(cfg.eyear)))

    # Calculate quantiles
    print(f'  Calculating quantiles...')
    if mode == 'wetonly':
        finq_p = finp.RAIN.where(finp.RAIN > wet_value).quantile(qtiles, dim=['time'])
        finq_f = finf.RAIN.where(finf.RAIN > wet_value).quantile(qtiles, dim=['time'])
    else:
        finq_p = finp.RAIN.quantile(qtiles, dim=['time'])
        finq_f = finf.RAIN.quantile(qtiles, dim=['time'])

    qtiles_p = finq_p.coords['quantile'].values
    qtiles_f = finq_f.coords['quantile'].values

    if not np.allclose(qtiles_p, qtiles_f):
        raise ValueError('Percentile tiles are different in both simulations')
    
    qtiles_vals = qtiles_p
    
    # Get spatial dimensions
    ny_dim = finp.sizes['y']
    nx_dim = finp.sizes['x']
    nq = len(qtiles_vals)
    
    # Initialize output arrays
    sig_var = np.zeros((ny_dim, nx_dim, nq), dtype=np.float64)
    next_p_all = np.zeros((ny_dim, nx_dim, nq), dtype=np.int32)
    next_f_all = np.zeros((ny_dim, nx_dim, nq), dtype=np.int32)
    
    print(f'  Processing {nq} quantiles sequentially...')
    
    # QUANTILE LOOP - Process each quantile separately
    for q_idx, qtile in enumerate(qtiles_vals):
        print(f'    Quantile {q_idx+1}/{nq} ({qtile*100:.1f}th percentile)')
        
        # Extract extremes for this quantile only
        # This creates 3D arrays (time, y, x) instead of 4D (time, y, x, quantile)
        print(f'      Extracting extremes...')
        ext_p_q = finp.RAIN.where(finp.RAIN >= finq_p.isel(quantile=q_idx))
        ext_f_q = finf.RAIN.where(finf.RAIN >= finq_f.isel(quantile=q_idx))
        
        # Count extremes
        next_p_all[:, :, q_idx] = ext_p_q.count('time').values
        next_f_all[:, :, q_idx] = ext_f_q.count('time').values
        
        # Convert to numpy arrays
        print(f'      Converting to numpy arrays...')
        ext_p_values = np.ascontiguousarray(ext_p_q.values, dtype=np.float64)
        ext_f_values = np.ascontiguousarray(ext_f_q.values, dtype=np.float64)
        
        print(f'      Array shape: {ext_p_values.shape}')
        print(f'      Computing significance tests...')
        
        # Compute significance for this quantile
        sig_var[:, :, q_idx] = compute_mwu_single_quantile(ext_p_values, ext_f_values)
        
        # Arrays will be overwritten in next iteration, but can explicitly delete if desired
        del ext_p_q, ext_f_q, ext_p_values, ext_f_values
        
        print(f'      Quantile {q_idx+1}/{nq} complete')
    
    print(f'  Creating output dataset...')
    
    # Create output arrays
    sig_da = xr.DataArray(sig_var, coords=[finp.y, finp.x, qtiles_vals], 
                          dims=['y', 'x', 'quantile'])
    next_p = xr.DataArray(next_p_all, coords=[finp.y, finp.x, qtiles_vals], 
                          dims=['y', 'x', 'quantile'])
    next_f = xr.DataArray(next_f_all, coords=[finp.y, finp.x, qtiles_vals], 
                          dims=['y', 'x', 'quantile'])
    ptiles_p = xr.DataArray(finq_p, coords=[qtiles_vals, finp.y, finp.x], 
                           dims=['quantile', 'y', 'x'])
    ptiles_f = xr.DataArray(finq_f, coords=[qtiles_vals, finf.y, finf.x], 
                           dims=['quantile', 'y', 'x'])
    
    # Create dataset
    output_ds = xr.Dataset({
        'significance': sig_da,
        'number_extreme_present': next_p,
        'number_extreme_future': next_f,
        'percentiles_present': ptiles_p,
        'percentiles_future': ptiles_f
    })
    
    # Add metadata
    output_ds.attrs['description'] = 'Significance testing using Numba-accelerated Mann-Whitney U test'
    output_ds.attrs['statistical_test'] = 'Mann-Whitney U (Wilcoxon rank-sum test)'
    output_ds.attrs['alternative'] = 'two-sided'
    output_ds.attrs['implementation'] = 'Numba JIT-compiled, quantile-by-quantile processing'
    output_ds.attrs['optimization'] = 'Quantile loop to avoid 4D arrays and improve cache efficiency'
    output_ds['significance'].attrs['long_name'] = 'Mann-Whitney U test p-value'
    output_ds['significance'].attrs['description'] = 'Two-sided p-value from Mann-Whitney U test'
    
    print(f'  Writing output...')
    
    output_ds.to_netcdf(fout)
    
    elapsed = time.time() - start_time
    print(f'  Tile {ny}-{nx} complete in {elapsed/60:.2f} minutes!')

    # Clean up
    finp.close()
    finf.close()
    del sig_var, next_p_all, next_f_all, output_ds
    gc.collect()


###############################################################################
##### __main__  scope
###############################################################################

if __name__ == "__main__":
    overall_start = time.time()
    
    main()
    
    overall_elapsed = time.time() - overall_start
    print(f'\n{"="*60}')
    print(f'Total processing time: {overall_elapsed/60:.2f} minutes')
    print(f'{"="*60}')

###############################################################################
