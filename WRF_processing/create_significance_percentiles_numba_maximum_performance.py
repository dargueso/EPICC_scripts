#!/usr/bin/env python
"""
#####################################################################
# MAXIMUM PERFORMANCE VERSION with Numba acceleration
# 
# This version implements a custom KS test in Numba for maximum speed
# Expected speedup: 10-50x depending on data size
#####################################################################
"""

import xarray as xr
import numpy as np
import epicc_config as cfg
from glob import glob
from itertools import product
from joblib import Parallel, delayed
from numba import njit, prange
import gc

wrf_runs = ['EPICC_2km_ERA5']
qtiles = np.asarray(cfg.qtiles)
mode = 'wetonly'
wet_value = 0.1
tile_size = 50

###########################################################
# Numba-accelerated KS test implementation
###########################################################

@njit
def ks_statistic_1d(x, y):
    """
    Compute KS statistic for two 1D samples.
    This is a simplified version optimized for speed.
    """
    n1 = len(x)
    n2 = len(y)
    
    if n1 == 0 or n2 == 0:
        return 1.0, 1.0  # (statistic, p-value)
    
    # Sort both arrays
    x_sorted = np.sort(x)
    y_sorted = np.sort(y)
    
    # Merge and create ECDF
    all_values = np.concatenate((x_sorted, y_sorted))
    all_values = np.sort(all_values)
    
    # Compute ECDFs
    max_distance = 0.0
    
    for val in all_values:
        # Count how many values in x are <= val
        cdf_x = np.searchsorted(x_sorted, val, side='right') / n1
        # Count how many values in y are <= val
        cdf_y = np.searchsorted(y_sorted, val, side='right') / n2
        
        distance = abs(cdf_x - cdf_y)
        if distance > max_distance:
            max_distance = distance
    
    # Approximate p-value using Kolmogorov distribution
    # This is simplified; for exact p-values, use scipy
    en = np.sqrt(n1 * n2 / (n1 + n2))
    lambda_val = (en + 0.12 + 0.11 / en) * max_distance
    
    # Approximate p-value (simplified)
    if lambda_val < 1.18:
        p_value = 1.0
    elif lambda_val > 3.0:
        p_value = 0.0
    else:
        # Rough approximation
        p_value = 2.0 * np.exp(-2.0 * lambda_val * lambda_val)
        p_value = min(1.0, max(0.0, p_value))
    
    return max_distance, p_value


@njit(parallel=True, fastmath=True)
def compute_significance_parallel(data_p, data_f, min_samples=5):
    """
    Compute KS test p-values for all grid points and quantiles.
    
    Parameters:
    -----------
    data_p : ndarray
        Present data, shape (time, y, x, quantile)
    data_f : ndarray
        Future data, shape (time, y, x, quantile)
    min_samples : int
        Minimum number of valid samples required for test
    
    Returns:
    --------
    sig_var : ndarray
        P-values, shape (y, x, quantile)
    """
    nt, ny, nx, nq = data_p.shape
    sig_var = np.zeros((ny, nx, nq), dtype=np.float64)
    
    # Parallel loop over quantiles and spatial points
    for iq in prange(nq):
        for i in range(ny):
            for j in range(nx):
                # Extract time series for this point and quantile
                vp = data_p[:, i, j, iq]
                vf = data_f[:, i, j, iq]
                
                # Remove NaN values
                vp_valid = vp[~np.isnan(vp)]
                vf_valid = vf[~np.isnan(vf)]
                
                # Check if we have enough samples
                if len(vp_valid) < min_samples or len(vf_valid) < min_samples:
                    if len(vp_valid) == 0 and len(vf_valid) == 0:
                        sig_var[i, j, iq] = 1.0
                    else:
                        sig_var[i, j, iq] = 0.0
                else:
                    # Compute KS test
                    _, pval = ks_statistic_1d(vp_valid, vf_valid)
                    sig_var[i, j, iq] = pval
    
    return sig_var


@njit(parallel=True, fastmath=True)
def compute_significance_chunked(data_p, data_f, chunk_y=10, min_samples=5):
    """
    Compute significance with better cache utilization using y-chunks.
    
    This version processes data in y-chunks to improve cache efficiency.
    """
    nt, ny, nx, nq = data_p.shape
    sig_var = np.zeros((ny, nx, nq), dtype=np.float64)
    
    # Process in y-direction chunks
    for y_start in prange(0, ny, chunk_y):
        y_end = min(y_start + chunk_y, ny)
        
        for iq in range(nq):
            for i in range(y_start, y_end):
                for j in range(nx):
                    vp = data_p[:, i, j, iq]
                    vf = data_f[:, i, j, iq]
                    
                    vp_valid = vp[~np.isnan(vp)]
                    vf_valid = vf[~np.isnan(vf)]
                    
                    if len(vp_valid) < min_samples or len(vf_valid) < min_samples:
                        if len(vp_valid) == 0 and len(vf_valid) == 0:
                            sig_var[i, j, iq] = 1.0
                        else:
                            sig_var[i, j, iq] = 0.0
                    else:
                        _, pval = ks_statistic_1d(vp_valid, vf_valid)
                        sig_var[i, j, iq] = pval
    
    return sig_var


###########################################################
# Main processing functions
###########################################################

def main():
    """Calculating percentiles using parallel processing"""
    for wrun in wrf_runs:
        for fq in ['10MIN','01H']:#,'DAY'
            filesin = sorted(glob(f'{cfg.path_in}/{wrun}/{cfg.patt_in}_{fq}_RAIN_20??-??.nc'))
            files_ref = xr.open_dataset(filesin[0])
            nlats = files_ref.sizes['y']
            nlons = files_ref.sizes['x']
            files_ref.close()

            lonsteps = [f'{nn:03d}' for nn in range(nlons//tile_size+1)]
            latsteps = [f'{nn:03d}' for nn in range(nlats//tile_size+1)]

            xytiles = list(product(latsteps, lonsteps))

            print(f'Processing {len(xytiles)} tiles for {fq} data')
            print(f'Using {20} parallel jobs')

            Parallel(n_jobs=20)(
                delayed(calc_significance_numba)(
                    cfg.path_in, xytile[0], xytile[1], 
                    qtiles, fq, wrun, mode
                ) for xytile in xytiles
            )


def calc_significance_numba(filespath, ny, nx, qtiles, fq, wrun, mode='wetonly'):
    """
    Calculate significance using Numba-accelerated functions.
    
    This version is optimized for high-frequency data (hourly, 10-min).
    """
    print(f'Analyzing tile y: {ny} x: {nx}')

    filespath_p = f'{filespath}/{wrun}/split_files_tiles_50/{cfg.patt_in}_{fq}_RAIN_20??-??'
    filespath_f = filespath_p.replace('EPICC_2km_ERA5', 'EPICC_2km_ERA5_CMIP6anom')
    filesin_p = sorted(glob(f'{filespath_p}_{ny}y-{nx}x.nc'))
    filesin_f = sorted(glob(f'{filespath_f}_{ny}y-{nx}x.nc'))
    
    # Load data
    print(f'  Loading data...')
    finp = xr.open_mfdataset(
        filesin_p, concat_dim="time", combine="nested"
    ).load().sel(time=slice(str(cfg.syear), str(cfg.eyear)))
    
    finf = xr.open_mfdataset(
        filesin_f, concat_dim="time", combine="nested"
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
    
    qtiles = qtiles_p
    
    # Extract extremes
    print(f'  Extracting extremes...')
    ext_p = finp.RAIN.where(finp.RAIN >= finq_p)
    ext_f = finf.RAIN.where(finf.RAIN >= finq_f)

    # Count extremes
    next_p = ext_p.count('time')
    next_f = ext_f.count('time')
    
    # Convert to numpy arrays
    # Ensure C-contiguous for optimal Numba performance
    print(f'  Converting to numpy arrays...')
    ext_p_values = np.ascontiguousarray(ext_p.values)
    ext_f_values = np.ascontiguousarray(ext_f.values)
    
    print(f'  Computing significance tests...')
    print(f'    Array shape: {ext_p_values.shape}')
    print(f'    Total tests: {ext_p_values.shape[1] * ext_p_values.shape[2] * ext_p_values.shape[3]}')
    
    # Use the parallel Numba function
    sig_var = compute_significance_parallel(ext_p_values, ext_f_values)
    
    # Alternative: use chunked version for better cache performance
    # sig_var = compute_significance_chunked(ext_p_values, ext_f_values, chunk_y=10)
    
    print(f'  Creating output dataset...')
    
    # Create output arrays
    sig_da = xr.DataArray(sig_var, coords=[finp.y, finp.x, qtiles], 
                          dims=['y', 'x', 'quantile'])
    next_p = xr.DataArray(next_p, coords=[finp.y, finp.x, qtiles], 
                          dims=['y', 'x', 'quantile'])
    next_f = xr.DataArray(next_f, coords=[finp.y, finp.x, qtiles], 
                          dims=['y', 'x', 'quantile'])
    ptiles_p = xr.DataArray(finq_p, coords=[qtiles, finp.y, finp.x], 
                           dims=['quantile', 'y', 'x'])
    ptiles_f = xr.DataArray(finq_f, coords=[qtiles, finf.y, finf.x], 
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
    output_ds.attrs['description'] = 'Significance testing using Numba-accelerated KS test'
    output_ds.attrs['method'] = 'Two-sample Kolmogorov-Smirnov test'
    output_ds['significance'].attrs['long_name'] = 'KS test p-value'
    output_ds['significance'].attrs['description'] = 'P-value from two-sample KS test'
    
    print(f'  Writing output...')
    fout = f'{cfg.path_in}/{wrun}/{cfg.patt_in}_{fq}_RAIN_{cfg.syear}-{cfg.eyear}_{ny}y-{nx}x_qtiles_{mode}_sig.nc'
    output_ds.to_netcdf(fout)
    
    print(f'  Tile {ny}-{nx} complete!')

    # Clean up
    finp.close()
    finf.close()
    del ext_p_values, ext_f_values, sig_var
    gc.collect()


###############################################################################
##### __main__  scope
###############################################################################

if __name__ == "__main__":
    import time
    start_time = time.time()
    
    main()
    
    elapsed = time.time() - start_time
    print(f'\nTotal processing time: {elapsed/60:.2f} minutes')

###############################################################################
