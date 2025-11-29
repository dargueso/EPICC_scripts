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
from joblib import Parallel, delayed,parallel_config
from numba import njit, prange
import gc
import time
from scipy.stats import mannwhitneyu, ks_2samp, anderson_ksamp, norm, chi2
from scipy.stats import genpareto
import warnings
from tqdm import tqdm


wrf_runs = ['EPICC_2km_ERA5']
qtiles = np.asarray(cfg.qtiles)
mode = 'wetonly'
wet_value = 0.1
tile_size = 50

###########################################################
# Numba-accelerated Mann-Whitney U test implementation
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


def mannwhitneyu_statistic(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    if x.size == 0 or y.size == 0:
        return np.nan, 1.0

    res = mannwhitneyu(x, y, alternative='two-sided', method='asymptotic')
    return res.statistic, float(res.pvalue)


def ks_test_pvalue(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    if x.size == 0 or y.size == 0:
        return 1.0
    stat, p = ks_2samp(x, y, alternative='two-sided', mode='asymp')
    return float(p)

def anderson_k_sample_pvalue(x, y):
    """
    Anderson-Darling k-sample test.
    Suppresses p-value precision warnings (acceptable for α=0.05 testing).
    """
    x = np.asarray(x)
    y = np.asarray(y)
    try:
        # Suppress all warnings from anderson_ksamp
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Suppress all warnings in this block
            res = anderson_ksamp([x, y])
        
        if hasattr(res, 'significance_level'):
            sl = float(res.significance_level)
            if sl > 1.0:
                p = sl / 100.0
            else:
                p = sl
            return float(min(max(p, 0.0), 1.0))
        else:
            return 1.0
    except Exception:
        return 1.0


def compute_all_significance(exceed_p, exceed_f, min_samples=10):
    """
    Input arrays should already be the exceedances (i.e., values > thr) for the given quantile.
    Returns a dict with p-values:
      - mannwhitney_p
      - ks_p
      - anderson_ad_p (approx)
      - gpd_lr_p
    """
    out = {}
    # Mann-Whitney
    _, p_mw = mannwhitneyu_statistic(exceed_p, exceed_f)
    out['mannwhitney_p'] = p_mw

    # KS
    out['ks_p'] = ks_test_pvalue(exceed_p, exceed_f)

    # Anderson-Darling (approx)
    out['anderson_ad_p'] = anderson_k_sample_pvalue(exceed_p, exceed_f)

    # GPD LRT
    #out['gpd_lr_p'], out['gpd_ll_sep'], out['gpd_ll_pool'] = gpd_lrt_pvalue(exceed_p, exceed_f, min_samples=min_samples)

    return out

def compute_quantile_and_signifcance(data_p, data_f, min_samples=5):
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
    sig_var_mw = np.zeros((ny, nx), dtype=np.float64)
    #sig_var_ks = np.zeros((ny, nx), dtype=np.float64)
    #sig_var_ad = np.zeros((ny, nx), dtype=np.float64)

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
                    sig_var_mw[i, j] = np.nan
                    # sig_var_ks[i, j] = np.nan
                    # sig_var_ad[i, j] = np.nan
                    
                else:
                    sig_var_mw[i, j] = np.nan
                    # sig_var_ks[i, j] = np.nan
                    # sig_var_ad[i, j] = np.nan
                    
            else:
                # Compute all significance tests

                signif = compute_all_significance(vp_valid, vf_valid, min_samples=min_samples)
                sig_var_mw[i, j] = signif['mannwhitney_p']
                # sig_var_ks[i, j] = signif['ks_p']
                # sig_var_ad[i, j] = signif['anderson_ad_p']
              
    
    return sig_var_mw#,sig_var_ks,sig_var_ad


###########################################################
# Main processing functions
###########################################################




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
    # sig_var_ks = np.zeros((ny_dim, nx_dim, nq), dtype=np.float64)
    # sig_var_ad = np.zeros((ny_dim, nx_dim, nq), dtype=np.float64)
    next_p_all = np.zeros((ny_dim, nx_dim, nq), dtype=np.int32)
    next_f_all = np.zeros((ny_dim, nx_dim, nq), dtype=np.int32)
    
    print(f'  Processing {nq} quantiles sequentially...')
    rain_p = finp.RAIN.values
    rain_f = finf.RAIN.values
    # QUANTILE LOOP - Process each quantile separately
    for q_idx, qtile in enumerate(tqdm(qtiles_vals, desc=f"Tile {tile_id}", leave=False)):
        print(f'    Quantile {q_idx+1}/{nq} ({qtile*100:.1f}th percentile)')
        
        # Get threshold values
        thr_p = finq_p.isel(quantile=q_idx).values
        thr_f = finq_f.isel(quantile=q_idx).values

        # Extract as numpy arrays immediately

        # This is more memory efficient - no masked arrays
        ext_p_q = np.where(rain_p >= thr_p, rain_p, np.nan)
        ext_f_q = np.where(rain_f >= thr_f, rain_f, np.nan)

        # Count extremes
        next_p_all[:, :, q_idx] = np.sum(~np.isnan(ext_p_q), axis=0)
        next_f_all[:, :, q_idx] = np.sum(~np.isnan(ext_f_q), axis=0)
     

        
        print(f'      Array shape: {ext_p_q.shape}')
        print(f'      Computing significance tests...')
        
        # Compute significance for this quantile
        sig_var_mw[:, :, q_idx] = compute_quantile_and_signifcance(ext_p_q, ext_f_q, min_samples=5)
        
        # Arrays will be overwritten in next iteration, but can explicitly delete if desired
        del ext_p_q, ext_f_q
        
        print(f'      Quantile {q_idx+1}/{nq} complete')
    
    print(f'  Creating output dataset...')
    
    # Create output arrays
    sig_mw_da = xr.DataArray(sig_var_mw, coords=[finp.y, finp.x, qtiles_vals], 
                          dims=['y', 'x', 'quantile'])
    # sig_ks_da = xr.DataArray(sig_var_ks, coords=[finp.y, finp.x, qtiles_vals],
    #                         dims=['y', 'x', 'quantile'])
    # sig_ad_da = xr.DataArray(sig_var_ad, coords=[finp.y, finp.x, qtiles_vals],
    #                         dims=['y', 'x', 'quantile'])
    
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
        # 'significance_ks': sig_ks_da,
        # 'significance_ad': sig_ad_da,
        # 'significance_gpd_lr': sig_gpd_lr_da,
        'number_extreme_present': next_p,
        'number_extreme_future': next_f,
        'percentiles_present': ptiles_p,
        'percentiles_future': ptiles_f
    })
    
    # Add metadata
    output_ds.attrs['description'] = 'Quantile calculation and Significance testing using Numba-accelerated'
    output_ds.attrs['statistical_test'] = 'Mann-Whitney U (Wilcoxon rank-sum test), KS, Anderson-Darling, GPD LRT'
    output_ds.attrs['implementation'] = 'Numba JIT-compiled, quantile-by-quantile processing'
    output_ds.attrs['optimization'] = 'Quantile loop to avoid 4D arrays and improve cache efficiency'
    
    print(f'  Writing output...')
    
    encoding = {
    'significance_mw': {'zlib': True, 'complevel': 4, 'dtype': 'float32'},
    # 'significance_ks': {'zlib': True, 'complevel': 4, 'dtype': 'float32'},
    # 'significance_ad': {'zlib': True, 'complevel': 4, 'dtype': 'float32'},
    # 'significance_gpd_lr': {'zlib': True, 'complevel': 4, 'dtype': 'float32'},
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
    # del sig_var_mw, sig_var_ks, sig_var_ad, sig_var_gpd_lr, next_p_all, next_f_all, output_ds
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
            
            NJOBS=20  # Set to 20 for production

            print(f'Total tiles: {len(xytiles)}')
            print(f'Already completed: {n_completed}')
            print(f'Remaining to process: {len(xytiles_to_process)}')
            print(f'Using {NJOBS} parallel jobs (quantile-loop optimization)')
            print(f'Statistical tests: Mann-Whitney U')#, KS, Anderson-Darling, GPD LRT')
            
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
    
    #main()
    
    test_single_tile()

    overall_elapsed = time.time() - overall_start
    print(f'\n{"="*60}')
    print(f'Total processing time: {overall_elapsed/60:.2f} minutes')
    print(f'{"="*60}')

###############################################################################
