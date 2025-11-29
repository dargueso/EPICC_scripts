#!/usr/bin/env python
"""
Optimized for slow I/O: Load tiles sequentially, parallelize computation within tile.
This avoids I/O contention while still leveraging multiple cores.
"""
import os
import sys
import time
import numpy as np
import xarray as xr
from multiprocessing import Pool
from scipy import stats
from numba import jit, prange
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

PATH_IN = '/home/dargueso/postprocessed/EPICC/'
PATH_OUT = '/home/dargueso/postprocessed/EPICC/'
WRUN_PRESENT = "EPICC_2km_ERA5"
WRUN_FUTURE = "EPICC_2km_ERA5_CMIP6anom"

FREQ = '1H'
WET_THRESHOLD = 0.1
PERCENTILES = [50, 75, 90, 95, 99, 99.9]
ALPHA = 0.05

# Test options:
# 'mann-whitney-fast' : Fast numba version
# 'mann-whitney'      : Exact scipy version
# 'none'              : Skip tests (percentiles only)
TEST_TYPE = 'mann-whitney'

TILE_SIZE = 50
N_PROCESSES = 16  # Used for within-tile parallelization

FREQ_TO_NAME = {
    '10MIN': '10MIN',
    '1H': '01H',
    '3H': '03H',
    '6H': '06H',
    '12H': '12H',
    'D': 'DAY'
}

freq_name = FREQ_TO_NAME[FREQ]

# =============================================================================
# NUMBA-OPTIMIZED FUNCTIONS
# =============================================================================

@jit(nopython=True)
def mannwhitneyu_fast(x, y):
    """Fast Mann-Whitney U test."""
    n1 = len(x)
    n2 = len(y)
    
    if n1 == 0 or n2 == 0:
        return np.nan, np.nan
    
    combined = np.concatenate((x, y))
    n = n1 + n2
    
    ranks = np.empty(n, dtype=np.float64)
    order = np.argsort(combined)
    ranks[order] = np.arange(1, n + 1)
    
    # Handle ties
    i = 0
    while i < n:
        j = i
        while j < n - 1 and combined[order[j]] == combined[order[j + 1]]:
            j += 1
        if j > i:
            avg_rank = (ranks[order[i]] + ranks[order[j]]) / 2.0
            for k in range(i, j + 1):
                ranks[order[k]] = avg_rank
        i = j + 1
    
    R1 = np.sum(ranks[:n1])
    U1 = R1 - n1 * (n1 + 1) / 2.0
    U2 = n1 * n2 - U1
    U = min(U1, U2)
    
    mu = n1 * n2 / 2.0
    
    t_correction = 0.0
    i = 0
    while i < n:
        j = i
        while j < n - 1 and combined[order[j]] == combined[order[j + 1]]:
            j += 1
        t = j - i + 1
        if t > 1:
            t_correction += (t**3 - t) / 12.0
        i = j + 1
    
    sigma = np.sqrt(n1 * n2 * (n + 1) / 12.0 - n1 * n2 * t_correction / (n * (n - 1)))
    
    if sigma == 0:
        return U, 1.0
    
    z = (U - mu) / sigma
    abs_z = abs(z)
    
    if abs_z > 6:
        pvalue = 0.0
    else:
        t = 1.0 / (1.0 + 0.2316419 * abs_z)
        d = 0.3989423 * np.exp(-abs_z * abs_z / 2.0)
        pvalue = d * t * (0.3193815 + t * (-0.3565638 + t * (1.781478 + t * (-1.821256 + t * 1.330274))))
        pvalue = 2.0 * pvalue
    
    return U, pvalue


@jit(nopython=True, parallel=True)
def process_tile_numba_percentiles_only(rain_present, rain_future, wet_threshold, percentiles):
    """Process tile - percentiles only (no statistical tests)."""
    nt, ny, nx = rain_present.shape
    n_perc = len(percentiles)
    
    perc_pres = np.full((n_perc, ny, nx), np.nan, dtype=np.float32)
    perc_fut = np.full((n_perc, ny, nx), np.nan, dtype=np.float32)
    perc_chg = np.full((n_perc, ny, nx), np.nan, dtype=np.float32)
    perc_chg_pct = np.full((n_perc, ny, nx), np.nan, dtype=np.float32)
    
    for iy in prange(ny):
        for ix in range(nx):
            pres_ts = rain_present[:, iy, ix]
            fut_ts = rain_future[:, iy, ix]
            
            pres_wet = pres_ts[pres_ts >= wet_threshold]
            fut_wet = fut_ts[fut_ts >= wet_threshold]
            
            n_pres = len(pres_wet)
            n_fut = len(fut_wet)
            
            if n_pres > 0:
                for ip in range(n_perc):
                    perc_pres[ip, iy, ix] = np.percentile(pres_wet, percentiles[ip])
            
            if n_fut > 0:
                for ip in range(n_perc):
                    perc_fut[ip, iy, ix] = np.percentile(fut_wet, percentiles[ip])
            
            if n_pres > 0 and n_fut > 0:
                for ip in range(n_perc):
                    pp = perc_pres[ip, iy, ix]
                    pf = perc_fut[ip, iy, ix]
                    perc_chg[ip, iy, ix] = pf - pp
                    if pp > 0:
                        perc_chg_pct[ip, iy, ix] = 100.0 * (pf - pp) / pp
    
    return perc_pres, perc_fut, perc_chg, perc_chg_pct


@jit(nopython=True, parallel=True)
def process_tile_numba_with_test(rain_present, rain_future, wet_threshold, percentiles, alpha):
    """Process tile with percentiles and Mann-Whitney test."""
    nt, ny, nx = rain_present.shape
    n_perc = len(percentiles)
    
    perc_pres = np.full((n_perc, ny, nx), np.nan, dtype=np.float32)
    perc_fut = np.full((n_perc, ny, nx), np.nan, dtype=np.float32)
    perc_chg = np.full((n_perc, ny, nx), np.nan, dtype=np.float32)
    perc_chg_pct = np.full((n_perc, ny, nx), np.nan, dtype=np.float32)
    pvalues = np.full((ny, nx), np.nan, dtype=np.float32)
    is_sig = np.zeros((ny, nx), dtype=np.int8)
    
    for iy in prange(ny):
        for ix in range(nx):
            pres_ts = rain_present[:, iy, ix]
            fut_ts = rain_future[:, iy, ix]
            
            pres_wet = pres_ts[pres_ts >= wet_threshold]
            fut_wet = fut_ts[fut_ts >= wet_threshold]
            
            n_pres = len(pres_wet)
            n_fut = len(fut_wet)
            
            if n_pres > 0:
                for ip in range(n_perc):
                    perc_pres[ip, iy, ix] = np.percentile(pres_wet, percentiles[ip])
            
            if n_fut > 0:
                for ip in range(n_perc):
                    perc_fut[ip, iy, ix] = np.percentile(fut_wet, percentiles[ip])
            
            if n_pres > 0 and n_fut > 0:
                for ip in range(n_perc):
                    pp = perc_pres[ip, iy, ix]
                    pf = perc_fut[ip, iy, ix]
                    perc_chg[ip, iy, ix] = pf - pp
                    if pp > 0:
                        perc_chg_pct[ip, iy, ix] = 100.0 * (pf - pp) / pp
            
            if n_pres >= 10 and n_fut >= 10:
                _, pvalue = mannwhitneyu_fast(pres_wet, fut_wet)
                pvalues[iy, ix] = pvalue
                if not np.isnan(pvalue) and pvalue < alpha:
                    is_sig[iy, ix] = 1
    
    return perc_pres, perc_fut, perc_chg, perc_chg_pct, pvalues, is_sig


def process_tile_scipy(rain_present, rain_future, wet_threshold, percentiles, alpha):
    """Process tile using scipy (no numba)."""
    nt, ny, nx = rain_present.shape
    n_perc = len(percentiles)
    percentiles_arr = np.array(percentiles)
    
    perc_pres = np.full((n_perc, ny, nx), np.nan, dtype=np.float32)
    perc_fut = np.full((n_perc, ny, nx), np.nan, dtype=np.float32)
    perc_chg = np.full((n_perc, ny, nx), np.nan, dtype=np.float32)
    perc_chg_pct = np.full((n_perc, ny, nx), np.nan, dtype=np.float32)
    pvalues = np.full((ny, nx), np.nan, dtype=np.float32)
    is_sig = np.zeros((ny, nx), dtype=np.int8)
    
    for iy in range(ny):
        for ix in range(nx):
            pres_ts = rain_present[:, iy, ix]
            fut_ts = rain_future[:, iy, ix]
            
            pres_wet = pres_ts[pres_ts >= wet_threshold]
            fut_wet = fut_ts[fut_ts >= wet_threshold]
            
            n_pres = len(pres_wet)
            n_fut = len(fut_wet)
            
            if n_pres > 0:
                perc_pres[:, iy, ix] = np.percentile(pres_wet, percentiles_arr)
            
            if n_fut > 0:
                perc_fut[:, iy, ix] = np.percentile(fut_wet, percentiles_arr)
            
            if n_pres > 0 and n_fut > 0:
                perc_chg[:, iy, ix] = perc_fut[:, iy, ix] - perc_pres[:, iy, ix]
                with np.errstate(divide='ignore', invalid='ignore'):
                    perc_chg_pct[:, iy, ix] = 100.0 * perc_chg[:, iy, ix] / perc_pres[:, iy, ix]
            
            if n_pres >= 10 and n_fut >= 10:
                try:
                    _, pvalue = stats.mannwhitneyu(pres_wet, fut_wet, alternative='two-sided')
                    pvalues[iy, ix] = pvalue
                    if not np.isnan(pvalue) and pvalue < alpha:
                        is_sig[iy, ix] = 1
                except Exception:
                    pass
    
    return perc_pres, perc_fut, perc_chg, perc_chg_pct, pvalues, is_sig


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("="*80)
    print(f"Percentiles and Significance for {FREQ} (Sequential I/O)")
    print(f"Test method: {TEST_TYPE}")
    print("="*80)
    
    total_start = time.time()
    
    zarr_path_present = f'{PATH_IN}/{WRUN_PRESENT}/UIB_{freq_name}_RAIN.zarr'
    zarr_path_future = f'{PATH_IN}/{WRUN_FUTURE}/UIB_{freq_name}_RAIN.zarr'
    
    print(f"\n1. Opening Zarr datasets...")
    t0 = time.time()
    
    try:
        ds_present = xr.open_zarr(zarr_path_present, consolidated=True)
        ds_future = xr.open_zarr(zarr_path_future, consolidated=True)
        print("   Opened with consolidated metadata")
    except KeyError:
        ds_present = xr.open_zarr(zarr_path_present, consolidated=False)
        ds_future = xr.open_zarr(zarr_path_future, consolidated=False)
        print("   Opened without consolidated metadata")
    
    print(f"   Time: {time.time()-t0:.2f}s")
    
    ny, nx = len(ds_present.y), len(ds_present.x)
    lat = ds_present.lat.isel(time=0).values
    lon = ds_present.lon.isel(time=0).values
    
    ny_tiles = (ny + TILE_SIZE - 1) // TILE_SIZE
    nx_tiles = (nx + TILE_SIZE - 1) // TILE_SIZE
    total_tiles = ny_tiles * nx_tiles
    
    print(f"\n   Domain: {ny} x {nx}")
    print(f"   Tiles: {ny_tiles} x {nx_tiles} = {total_tiles}")
    print(f"   Strategy: Sequential I/O, parallel computation ({N_PROCESSES} cores)")
    
    # Warm up numba if using fast version
    if TEST_TYPE == 'mann-whitney-fast':
        print(f"\n   Compiling numba functions...", end='', flush=True)
        t_compile = time.time()
        test_data = np.random.rand(100, 5, 5).astype(np.float32)
        test_perc = np.array([50.0, 90.0], dtype=np.float32)
        _ = process_tile_numba_with_test(test_data, test_data, 0.1, test_perc, 0.05)
        print(f" done ({time.time()-t_compile:.1f}s)")
    elif TEST_TYPE == 'none':
        print(f"\n   Compiling numba functions...", end='', flush=True)
        t_compile = time.time()
        test_data = np.random.rand(100, 5, 5).astype(np.float32)
        test_perc = np.array([50.0, 90.0], dtype=np.float32)
        _ = process_tile_numba_percentiles_only(test_data, test_data, 0.1, test_perc)
        print(f" done ({time.time()-t_compile:.1f}s)")
    
    # Initialize outputs
    n_percentiles = len(PERCENTILES)
    percentiles_present_full = np.full((n_percentiles, ny, nx), np.nan, dtype=np.float32)
    percentiles_future_full = np.full((n_percentiles, ny, nx), np.nan, dtype=np.float32)
    percentiles_change_full = np.full((n_percentiles, ny, nx), np.nan, dtype=np.float32)
    percentiles_change_pct_full = np.full((n_percentiles, ny, nx), np.nan, dtype=np.float32)
    pvalues_full = np.full((ny, nx), np.nan, dtype=np.float32)
    is_significant_full = np.zeros((ny, nx), dtype=np.int8)
    
    print(f"\n2. Processing tiles (one at a time)...")
    sys.stdout.flush()
    
    tiles_processed = 0
    tile_times = []
    io_times = []
    compute_times = []
    
    percentiles_arr = np.array(PERCENTILES, dtype=np.float32)
    
    for iy_tile in range(ny_tiles):
        for ix_tile in range(nx_tiles):
            tile_start = time.time()
            
            # Calculate bounds
            y_start = iy_tile * TILE_SIZE
            y_end = min(y_start + TILE_SIZE, ny)
            x_start = ix_tile * TILE_SIZE
            x_end = min(x_start + TILE_SIZE, nx)
            
            # Load tile (SEQUENTIAL - one at a time)
            t_io = time.time()
            rain_present = ds_present.RAIN.isel(
                y=slice(y_start, y_end),
                x=slice(x_start, x_end)
            ).values.astype(np.float32)
            
            rain_future = ds_future.RAIN.isel(
                y=slice(y_start, y_end),
                x=slice(x_start, x_end)
            ).values.astype(np.float32)
            io_time = time.time() - t_io
            io_times.append(io_time)
            
            # Process tile (PARALLEL within tile)
            t_compute = time.time()
            
            if TEST_TYPE == 'none':
                perc_pres, perc_fut, perc_chg, perc_chg_pct = process_tile_numba_percentiles_only(
                    rain_present, rain_future, WET_THRESHOLD, percentiles_arr
                )
                pvalues = np.full((y_end - y_start, x_end - x_start), np.nan, dtype=np.float32)
                is_sig = np.zeros((y_end - y_start, x_end - x_start), dtype=np.int8)
            
            elif TEST_TYPE == 'mann-whitney-fast':
                perc_pres, perc_fut, perc_chg, perc_chg_pct, pvalues, is_sig = process_tile_numba_with_test(
                    rain_present, rain_future, WET_THRESHOLD, percentiles_arr, ALPHA
                )
            
            else:  # scipy version
                perc_pres, perc_fut, perc_chg, perc_chg_pct, pvalues, is_sig = process_tile_scipy(
                    rain_present, rain_future, WET_THRESHOLD, PERCENTILES, ALPHA
                )
            
            compute_time = time.time() - t_compute
            compute_times.append(compute_time)
            
            # Store results
            percentiles_present_full[:, y_start:y_end, x_start:x_end] = perc_pres
            percentiles_future_full[:, y_start:y_end, x_start:x_end] = perc_fut
            percentiles_change_full[:, y_start:y_end, x_start:x_end] = perc_chg
            percentiles_change_pct_full[:, y_start:y_end, x_start:x_end] = perc_chg_pct
            pvalues_full[y_start:y_end, x_start:x_end] = pvalues
            is_significant_full[y_start:y_end, x_start:x_end] = is_sig
            
            tiles_processed += 1
            tile_time = time.time() - tile_start
            tile_times.append(tile_time)
            
            # Progress every 10 tiles
            if tiles_processed % 10 == 0 or tiles_processed == total_tiles:
                elapsed = time.time() - total_start
                avg_io = np.mean(io_times[-10:])
                avg_compute = np.mean(compute_times[-10:])
                avg_tile = np.mean(tile_times[-10:])
                eta_sec = (total_tiles - tiles_processed) * avg_tile
                
                print(f"   Tile {tiles_processed:3d}/{total_tiles} "
                      f"({100*tiles_processed/total_tiles:5.1f}%) | "
                      f"I/O: {avg_io:5.2f}s, compute: {avg_compute:5.2f}s | "
                      f"ETA: {eta_sec/60:5.1f} min")
                sys.stdout.flush()
    
    ds_present.close()
    ds_future.close()
    
    # Save
    print(f"\n3. Saving output...")
    t0 = time.time()
    
    ds_output = xr.Dataset(
        data_vars=dict(
            percentiles_present=(("percentile", "y", "x"), percentiles_present_full),
            percentiles_future=(("percentile", "y", "x"), percentiles_future_full),
            percentiles_change=(("percentile", "y", "x"), percentiles_change_full),
            percentiles_change_pct=(("percentile", "y", "x"), percentiles_change_pct_full),
            pvalue=(("y", "x"), pvalues_full),
            is_significant=(("y", "x"), is_significant_full),
            lat=(("y", "x"), lat),
            lon=(("y", "x"), lon),
        ),
        coords=dict(
            percentile=PERCENTILES,
            y=np.arange(ny),
            x=np.arange(nx),
        ),
        attrs={
            'description': f'Percentiles and statistical significance for {FREQ} rainfall',
            'units': f'mm/{FREQ}',
            'wet_threshold': WET_THRESHOLD,
            'statistical_test': TEST_TYPE,
            'significance_level': ALPHA,
            'present_period': WRUN_PRESENT,
            'future_period': WRUN_FUTURE,
            'processing': 'sequential I/O, parallel computation'
        }
    )
    
    ds_output['percentiles_present'].attrs = {'long_name': 'Percentiles present', 'units': f'mm/{FREQ}'}
    ds_output['percentiles_future'].attrs = {'long_name': 'Percentiles future', 'units': f'mm/{FREQ}'}
    ds_output['percentiles_change'].attrs = {'long_name': 'Change', 'units': f'mm/{FREQ}'}
    ds_output['percentiles_change_pct'].attrs = {'long_name': 'Percent change', 'units': '%'}
    ds_output['pvalue'].attrs = {'long_name': 'P-value', 'method': TEST_TYPE}
    ds_output['is_significant'].attrs = {'long_name': 'Significant', 'alpha': ALPHA}
    
    suffix = TEST_TYPE.replace('-', '_')
    output_file = f'{PATH_OUT}/{WRUN_PRESENT}/percentiles_and_significance_{FREQ}_{suffix}_seqio.nc'
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    ds_output.to_netcdf(output_file)
    
    print(f"   Saved: {output_file}")
    print(f"   Save time: {time.time()-t0:.2f}s")
    
    # Summary
    n_significant = np.sum(is_significant_full)
    n_total = np.sum(~np.isnan(pvalues_full))
    
    print(f"\n4. Summary:")
    print(f"   Grid points: {ny * nx:,}")
    if n_total > 0:
        print(f"   Significant: {n_significant:,}/{n_total:,} ({100*n_significant/n_total:.1f}%)")
    print(f"\n   Mean percentile changes:")
    for i, p in enumerate(PERCENTILES):
        print(f"      P{p:5.1f}: {np.nanmean(percentiles_change_pct_full[i]):+7.1f}%")
    
    total_time = time.time() - total_start
    avg_io = np.mean(io_times)
    avg_compute = np.mean(compute_times)
    
    print(f"\n{'='*80}")
    print(f"COMPLETE!")
    print(f"  Total time: {total_time/60:.2f} minutes ({total_time:.1f}s)")
    print(f"  Avg I/O: {avg_io:.2f}s/tile")
    print(f"  Avg compute: {avg_compute:.2f}s/tile")
    print(f"  I/O percentage: {100*avg_io/(avg_io+avg_compute):.1f}%")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
