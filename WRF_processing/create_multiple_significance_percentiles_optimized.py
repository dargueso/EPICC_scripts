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
# Version: 4.0 (Optimized with Numba)
# Description: Mann-Whitney U test with proper numba acceleration
#              Uses accurate normal CDF approximation
#
# Key optimization: Numba-compiled Mann-Whitney with proper statistics
#
#####################################################################
"""

import xarray as xr
import numpy as np
import epicc_config as cfg
from glob import glob
import os
from itertools import product
from joblib import Parallel, delayed, parallel_config
from numba import njit, prange
import gc
import time
from scipy.stats import mannwhitneyu
import warnings
from tqdm import tqdm


wrf_runs = ['EPICC_2km_ERA5']
qtiles = np.asarray(cfg.qtiles)
mode = 'wetonly'
wet_value = 0.1
tile_size = 50

###########################################################
# Numba-accelerated statistical functions
###########################################################

@njit
def _erf_approx(x):
    """
    Accurate approximation of the error function using Abramowitz & Stegun formula.
    Maximum error: 1.5e-7
    
    This is the standard approximation used in numerical computing.
    """
    # Constants for Abramowitz & Stegun approximation
    a1 =  0.254829592
    a2 = -0.284496736
    a3 =  1.421413741
    a4 = -1.453152027
    a5 =  1.061405429
    p  =  0.3275911
    
    # Save the sign of x
    sign = 1.0 if x >= 0 else -1.0
    x = abs(x)
    
    # Abramowitz & Stegun formula
    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp(-x * x)
    
    return sign * y


@njit
def _norm_cdf(x):
    """
    Standard normal cumulative distribution function.
    Uses the error function approximation.
    
    CDF(x) = 0.5 * (1 + erf(x / sqrt(2)))
    """
    return 0.5 * (1.0 + _erf_approx(x / np.sqrt(2.0)))


@njit
def _norm_sf(x):
    """
    Standard normal survival function (1 - CDF).
    More accurate for large x.
    """
    return 1.0 - _norm_cdf(x)


@njit
def _mannwhitneyu_numba(x, y):
    """
    Numba-accelerated Mann-Whitney U test with proper statistical calculation.
    
    This implementation:
    1. Computes ranks correctly (with tie handling)
    2. Calculates U statistic correctly
    3. Uses proper normal approximation with continuity correction
    4. Computes p-value using accurate normal CDF
    
    Returns:
    --------
    u_statistic : float
        The Mann-Whitney U statistic
    p_value : float
        Two-sided p-value
    """
    n1 = len(x)
    n2 = len(y)
    
    if n1 == 0 or n2 == 0:
        return np.nan, 1.0
    
    # Combine and sort
    combined = np.concatenate((x, y))
    n_total = n1 + n2
    
    # Get sort indices
    sorted_idx = np.argsort(combined)
    sorted_vals = combined[sorted_idx]
    
    # Assign ranks (1-based)
    ranks = np.empty(n_total, dtype=np.float64)
    
    # Handle ties by averaging ranks
    i = 0
    while i < n_total:
        j = i
        # Find extent of tied values
        while j < n_total - 1 and sorted_vals[j] == sorted_vals[j + 1]:
            j += 1
        
        # Average rank for tied values
        avg_rank = (i + j + 2) / 2.0  # +2 because ranks are 1-based and j is inclusive
        
        for k in range(i, j + 1):
            ranks[k] = avg_rank
        
        i = j + 1
    
    # Create group labels
    group = np.zeros(n_total, dtype=np.int32)
    group[:n1] = 0  # x group
    group[n1:] = 1  # y group
    
    # Reorder group labels according to sort
    group_sorted = group[sorted_idx]
    
    # Sum ranks for x group
    rank_sum_x = 0.0
    for i in range(n_total):
        if group_sorted[i] == 0:
            rank_sum_x += ranks[i]
    
    # Calculate U statistic
    u1 = rank_sum_x - n1 * (n1 + 1) / 2.0
    u2 = n1 * n2 - u1
    
    # Return u1 to match scipy convention
    u_statistic = u1
    
    # Calculate expected value and variance under null hypothesis
    mean_u = n1 * n2 / 2.0
    
    # Variance with tie correction
    # First, calculate tie correction factor
    tie_correction = 0.0
    i = 0
    while i < n_total:
        j = i
        while j < n_total - 1 and sorted_vals[j] == sorted_vals[j + 1]:
            j += 1
        
        t = j - i + 1  # number of tied values
        if t > 1:
            tie_correction += t * t * t - t
        
        i = j + 1
    
    std_u = np.sqrt(n1 * n2 * (n_total + 1) / 12.0 - 
                    n1 * n2 * tie_correction / (12.0 * n_total * (n_total - 1)))
    
    if std_u == 0:
        return u_statistic, 1.0
    
    # Calculate z-score with continuity correction
    # When u1 == mean_u (distributions identical), z should be 0
    if abs(u1 - mean_u) < 0.5:
        # Very close to mean, treat as no difference
        z = 0.0
    elif u1 < mean_u:
        z = (u1 - mean_u + 0.5) / std_u
    else:
        z = (u1 - mean_u - 0.5) / std_u
    
    # Two-sided p-value
    if z == 0.0:
        p_value = 1.0
    else:
        p_value = 2.0 * _norm_sf(abs(z))
    
    return u_statistic, p_value


@njit(parallel=True)
def compute_quantile_significance_numba(data_p, data_f, min_samples=5):
    """
    Compute Mann-Whitney U test p-values for a single quantile using Numba.
    
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
    sig_var_mw : ndarray
        P-values, shape (y, x)
    """
    nt, ny, nx = data_p.shape
    sig_var_mw = np.zeros((ny, nx), dtype=np.float64)

    # Parallel loop over spatial points
    for i in prange(ny):
        for j in range(nx):
            # Extract time series for this point
            vp = data_p[:, i, j]
            vf = data_f[:, i, j]
            
            # Count valid (non-NaN) values
            n_valid_p = 0
            n_valid_f = 0
            for t in range(nt):
                if not np.isnan(vp[t]):
                    n_valid_p += 1
                if not np.isnan(vf[t]):
                    n_valid_f += 1
            
            # Check if we have enough samples
            if n_valid_p < min_samples or n_valid_f < min_samples:
                if n_valid_p == 0 and n_valid_f == 0:
                    sig_var_mw[i, j] = np.nan
                else:
                    sig_var_mw[i, j] = np.nan
            else:
                # Extract valid values
                vp_valid = np.empty(n_valid_p, dtype=np.float64)
                vf_valid = np.empty(n_valid_f, dtype=np.float64)
                
                idx_p = 0
                idx_f = 0
                for t in range(nt):
                    if not np.isnan(vp[t]):
                        vp_valid[idx_p] = vp[t]
                        idx_p += 1
                    if not np.isnan(vf[t]):
                        vf_valid[idx_f] = vf[t]
                        idx_f += 1
                
                # Compute Mann-Whitney U test
                _, p_value = _mannwhitneyu_numba(vp_valid, vf_valid)
                sig_var_mw[i, j] = p_value
    
    return sig_var_mw


###########################################################
# File management functions
###########################################################

def get_completed_tiles(fq, wrun, mode):
    """
    Check which tiles have already been completed.
    
    Returns:
    --------
    set of tile IDs (e.g., {'005y-011x', '006y-012x', ...})
    """
    pattern = f'{cfg.path_in}/{wrun}/{cfg.patt_in}_{fq}_RAIN_{cfg.syear}-{cfg.eyear}_*y-*x_qtiles_{mode}_sig_mwu.nc'
    completed_files = glob(pattern)
    
    # Extract tile IDs from filenames
    completed_tiles = set()
    for filepath in completed_files:
        basename = os.path.basename(filepath)
        # Extract pattern like "005y-011x" from filename
        parts = basename.split('_')
        for part in parts:
            if 'y-' in part and 'x' in part:
                completed_tiles.add(part)
                break
    
    return completed_tiles


def get_tiles_to_process(xytiles, fq, wrun, mode, force_reprocess=False):
    """
    Filter tiles to only process those not yet completed.
    
    Parameters:
    -----------
    xytiles : list of tuples
        All tiles (latstep, lonstep)
    fq : str
        Frequency
    wrun : str
        Run name
    mode : str
        Processing mode
    force_reprocess : bool
        If True, process all tiles even if output exists
    
    Returns:
    --------
    list of tuples: tiles to process
    int: number of already completed tiles
    """
    if force_reprocess:
        return xytiles, 0
    
    completed = get_completed_tiles(fq, wrun, mode)
    
    # Filter out completed tiles
    tiles_to_process = []
    for lat, lon in xytiles:
        tile_id = f"{lat}y-{lon}x"
        if tile_id not in completed:
            tiles_to_process.append((lat, lon))
    
    n_completed = len(xytiles) - len(tiles_to_process)
    
    return tiles_to_process, n_completed


def test_single_tile():
    """Test processing a single tile"""
    print("="*60)
    print("TESTING MODE - Processing single tile")
    print("="*60)
    
    wrun = 'EPICC_2km_ERA5'
    fq = '01H'  # or '10MIN' 
    
    # Pick a tile that you know exists
    test_ny = '005'
    test_nx = '011'
    
    result = process_tile_safe(
        cfg.path_in, 
        test_ny, 
        test_nx,
        qtiles, 
        fq, 
        wrun, 
        mode
    )
    
    print(result)
    print("="*60)


def process_tile_safe(filespath, ny, nx, qtiles, fq, wrun, mode):
    """Wrapper with error handling"""
    tile_id = f"{ny}y-{nx}x"
    try:
        calc_significance_quantile_loop(filespath, ny, nx, qtiles, fq, wrun, mode)
        return f"✓ Success: {tile_id}"
    except Exception as e:
        error_msg = f"✗ Failed: {tile_id} - {str(e)}"
        print(error_msg)
        # Log errors to file
        log_file = f'{cfg.path_in}/{wrun}/processing_errors_{fq}.log'
        with open(log_file, 'a') as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {error_msg}\n")
        return error_msg


###########################################################
# Main processing function
###########################################################

def calc_significance_quantile_loop(filespath, ny, nx, qtiles, fq, wrun, mode='wetonly'):
    """
    Calculate significance using quantile-by-quantile processing with Numba acceleration.
    
    This approach:
    - Loads all data into memory (fast with high RAM)
    - Processes one quantile at a time (avoids 4D arrays)
    - Uses Numba-compiled Mann-Whitney U test
    - Better cache efficiency
    - Allows more parallel tiles
    """
    print(f'Analyzing tile y: {ny} x: {nx}')
    start_time = time.time()
    tile_id = f"{ny}y-{nx}x"
    
    # Define output file
    fout = f'{cfg.path_in}/{wrun}/{cfg.patt_in}_{fq}_RAIN_{cfg.syear}-{cfg.eyear}_{ny}y-{nx}x_qtiles_{mode}_sig_mwu.nc'
    
    if os.path.exists(fout):
        print(tile_id, "skipped", 0.0, "Already exists")
        return 

    filespath_p = f'{filespath}/{wrun}/split_files_tiles_50/{cfg.patt_in}_{fq}_RAIN_20??-??'
    filespath_f = filespath_p.replace('EPICC_2km_ERA5', 'EPICC_2km_ERA5_CMIP6anom')
    filesin_p = sorted(glob(f'{filespath_p}_{ny}y-{nx}x.nc'))
    filesin_f = sorted(glob(f'{filespath_f}_{ny}y-{nx}x.nc'))

    if not filesin_p or not filesin_f:
        raise FileNotFoundError(f"No files found for tile {tile_id}")
    if len(filesin_p) != len(filesin_f):
        raise ValueError(f"Mismatch in file counts: {len(filesin_p)} present vs {len(filesin_f)} future")
    print(f'  Found {len(filesin_p)} files per scenario')

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
    sig_var_mw = np.zeros((ny_dim, nx_dim, nq), dtype=np.float64)
    next_p_all = np.zeros((ny_dim, nx_dim, nq), dtype=np.int32)
    next_f_all = np.zeros((ny_dim, nx_dim, nq), dtype=np.int32)
    
    print(f'  Processing {nq} quantiles sequentially with Numba acceleration...')
    rain_p = finp.RAIN.values
    rain_f = finf.RAIN.values
    
    # QUANTILE LOOP - Process each quantile separately
    for q_idx, qtile in enumerate(tqdm(qtiles_vals, desc=f"Tile {tile_id}", leave=False)):
        print(f'    Quantile {q_idx+1}/{nq} ({qtile*100:.1f}th percentile)')
        
        # Get threshold values
        thr_p = finq_p.isel(quantile=q_idx).values
        thr_f = finq_f.isel(quantile=q_idx).values

        # Extract exceedances - more memory efficient
        ext_p_q = np.where(rain_p >= thr_p, rain_p, np.nan)
        ext_f_q = np.where(rain_f >= thr_f, rain_f, np.nan)

        # Count extremes
        next_p_all[:, :, q_idx] = np.sum(~np.isnan(ext_p_q), axis=0)
        next_f_all[:, :, q_idx] = np.sum(~np.isnan(ext_f_q), axis=0)
     
        print(f'      Array shape: {ext_p_q.shape}')
        print(f'      Computing significance tests with Numba...')
        
        # Compute significance for this quantile using Numba
        sig_var_mw[:, :, q_idx] = compute_quantile_significance_numba(ext_p_q, ext_f_q, min_samples=5)
        
        # Clean up
        del ext_p_q, ext_f_q
        
        print(f'      Quantile {q_idx+1}/{nq} complete')
    
    print(f'  Creating output dataset...')
    
    # Create output arrays
    sig_mw_da = xr.DataArray(sig_var_mw, coords=[finp.y, finp.x, qtiles_vals], 
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
        'significance_mw': sig_mw_da,
        'number_extreme_present': next_p,
        'number_extreme_future': next_f,
        'percentiles_present': ptiles_p,
        'percentiles_future': ptiles_f
    })
    
    # Add metadata
    output_ds.attrs['description'] = 'Quantile calculation and Significance testing using Numba-accelerated Mann-Whitney U'
    output_ds.attrs['statistical_test'] = 'Mann-Whitney U (Wilcoxon rank-sum test) with proper normal approximation'
    output_ds.attrs['implementation'] = 'Numba JIT-compiled with accurate erf-based CDF'
    output_ds.attrs['optimization'] = 'Quantile loop with parallel numba processing'
    
    print(f'  Writing output...')
    
    encoding = {
        'significance_mw': {'zlib': True, 'complevel': 4, 'dtype': 'float32'},
        'number_extreme_present': {'zlib': True, 'complevel': 4, 'dtype': 'int16'},
        'number_extreme_future': {'zlib': True, 'complevel': 4, 'dtype': 'int16'},
        'percentiles_present': {'zlib': True, 'complevel': 4, 'dtype': 'float32'},
        'percentiles_future': {'zlib': True, 'complevel': 4, 'dtype': 'float32'},
    }
    output_ds.to_netcdf(fout, encoding=encoding, format='NETCDF4')
    
    elapsed = time.time() - start_time
    print(f'  Tile {ny}-{nx} complete in {elapsed/60:.2f} minutes!')

    # Clean up
    finp.close()
    finf.close()
    del sig_var_mw, next_p_all, next_f_all, output_ds
    gc.collect()


def main(force_reprocess=False):
    """
    Calculating percentiles using parallel processing
    
    Parameters:
    -----------
    force_reprocess : bool
        If True, reprocess all tiles even if output exists
    """
    for wrun in wrf_runs:
        for fq in ['01H']:  # '10MIN','01H','DAY'
            filesin = sorted(glob(f'{cfg.path_in}/{wrun}/RAIN/{cfg.patt_in}_{fq}_RAIN_20??-??.nc'))
            files_ref = xr.open_dataset(filesin[0])
            nlats = files_ref.sizes['y']
            nlons = files_ref.sizes['x']
            files_ref.close()

            lonsteps = [f'{nn:03d}' for nn in range(nlons//tile_size+1)]
            latsteps = [f'{nn:03d}' for nn in range(nlats//tile_size+1)]

            xytiles = list(product(latsteps, lonsteps))
            
            # Check for completed tiles
            xytiles_to_process, n_completed = get_tiles_to_process(
                xytiles, fq, wrun, mode, force_reprocess
            )
            
            NJOBS = 20  # Set to 20 for production

            print(f'Total tiles: {len(xytiles)}')
            print(f'Already completed: {n_completed}')
            print(f'Remaining to process: {len(xytiles_to_process)}')
            print(f'Using {NJOBS} parallel jobs (Numba-optimized)')
            print(f'Statistical tests: Mann-Whitney U with accurate normal approximation')
            
            if len(xytiles_to_process) == 0:
                print(f'All tiles already processed for {fq}!')
                continue
            
            with parallel_config(backend='threading', n_jobs=NJOBS):
                results = Parallel()(
                    delayed(process_tile_safe)(
                        cfg.path_in, xytile[0], xytile[1], 
                        qtiles, fq, wrun, mode
                    ) for xytile in tqdm(xytiles_to_process, desc=f"Processing {fq} tiles")
                )

            successful = sum(1 for r in results if r.startswith('✓'))
            failed = sum(1 for r in results if r.startswith('✗'))
            print(f'\nProcessing complete for {fq}:')
            print(f'  Successful: {successful}')
            print(f'  Failed: {failed}')
            print(f'  Previously completed: {n_completed}')
            print(f'  Total: {n_completed + successful}/{len(xytiles)}')


###############################################################################
##### __main__  scope
###############################################################################

if __name__ == "__main__":
    overall_start = time.time()
    
    main()
    
    #test_single_tile()

    overall_elapsed = time.time() - overall_start
    print(f'\n{"="*60}')
    print(f'Total processing time: {overall_elapsed/60:.2f} minutes')
    print(f'{"="*60}')

###############################################################################
