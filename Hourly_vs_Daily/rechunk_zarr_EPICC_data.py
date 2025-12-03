import os
import time
import xarray as xr
import dask

# Configure dask for better performance
dask.config.set(scheduler='threads', num_workers=32)

PATH_IN = '/home/dargueso/postprocessed/EPICC/'
PATH_OUT = '/home/dargueso/postprocessed/EPICC/'
WRUN = "EPICC_2km_ERA5"
fq = '10MIN'
# Optimal chunking parameters
#OPTIMAL_TIME_CHUNK = 100
OPTIMAL_TIME_CHUNK = 8760  # ~2 months of 10-min data
OPTIMAL_SPATIAL_CHUNK = 50  # Tile size

# Years to process
START_YEAR = 2011
END_YEAR = 2020

print("="*60)
print("Creating Zarr store from original files")
print("="*60)

total_start = time.time()

# File pattern for all years
files = f'{PATH_IN}/{WRUN}/RAIN/UIB_{fq}_RAIN_20??-??.nc'

print("\n1. Opening all files...")
t0 = time.time()
ds = xr.open_mfdataset(
    files, 
    combine='by_coords',
    chunks=None  # Don't specify chunks on open - let it use file chunks
)
t1 = time.time()
print(f"   Opened in {t1-t0:.2f}s")
print(f"   Shape: {ds.RAIN.shape}")
print(f"   Original chunks: {ds.RAIN.chunks}")

# Now rechunk to uniform sizes
print("\n2. Rechunking to uniform sizes...")
t0 = time.time()
ds_rechunked = ds.chunk({
    'time': OPTIMAL_TIME_CHUNK,
    'y': OPTIMAL_SPATIAL_CHUNK,
    'x': OPTIMAL_SPATIAL_CHUNK
})
t1 = time.time()
print(f"   Rechunked in {t1-t0:.2f}s")
print(f"   New chunks: {ds_rechunked.RAIN.chunks}")

# Create output directory
zarr_path = f'{PATH_OUT}/{WRUN}/UIB_{fq}_RAIN.zarr'
print(f"\n3. Writing to Zarr: {zarr_path}")

# Encoding for zarr
encoding = {
    'RAIN': {
        'compressor': None,  # No compression for speed
        'dtype': 'float32'
    }
}

# Write to zarr (this does the actual computation)
t0 = time.time()
ds_rechunked.to_zarr(zarr_path, mode='w', encoding=encoding, consolidated=True)
t1 = time.time()

print(f"   Written in {t1-t0:.2f}s ({(t1-t0)/60:.2f} minutes)")

total_time = time.time() - total_start
print("\n" + "="*60)
print(f"Zarr store created! Total time: {total_time/60:.2f} minutes")
print("="*60)
