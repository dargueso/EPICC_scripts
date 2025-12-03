import os 
import time
import numpy as np
import xarray as xr
import dask

# Configure dask
dask.config.set(scheduler='threads', num_workers=64)

PATH_IN = '/home/dargueso/postprocessed/EPICC/'
PATH_OUT = '/home/dargueso/postprocessed/EPICC/'
WRUN = "EPICC_2km_ERA5"

WET_VALUE_HIFREQ = 0.1  # mm/10min
WET_VALUE_LOFREQ = 0.1  # mm/h

LORAIN_BINS = np.arange(0, 105, 5)  # Hourly bins: [0, 5, 10, ..., 100]
HIRAIN_BINS = np.arange(0, 105, 5)  # 10-min bins: [0, 5, 10, ..., 100]

# Add infinity as the last bin edge to catch all values > 100
LORAIN_BINS = np.append(LORAIN_BINS, np.inf)
HIRAIN_BINS = np.append(HIRAIN_BINS, np.inf)

tile_size = 50

#####################################################################
print("="*60)
print("Creating 2D Histograms - 10min vs Hourly")
print("="*60)

ny = '005'
nx = '011'
tile_id = f"{ny}y-{nx}x"

# Define tile bounds
y_start = int(ny) * tile_size
y_end = y_start + tile_size
x_start = int(nx) * tile_size
x_end = x_start + tile_size

# Zarr file path
zarr_path = f'{PATH_IN}/{WRUN}/UIB_10MIN_RAIN.zarr'

print(f"\nProcessing tile: {tile_id}")
print(f"Tile bounds: y=[{y_start}:{y_end}], x=[{x_start}:{x_end}]")
print(f"Bins (10-min): {HIRAIN_BINS}")
print(f"Bins (hourly): {LORAIN_BINS}")

# Open and extract tile
print("\n1. Loading data...")
t0 = time.time()
fin_hf = xr.open_zarr(zarr_path, consolidated=True)
fin_hf = fin_hf.isel(y=slice(y_start, y_end), x=slice(x_start, x_end))
fin_hf_loaded = fin_hf.load()
t1 = time.time()
print(f"   Loaded in {t1-t0:.2f}s")
print(f"   Shape: {fin_hf_loaded.RAIN.shape}")

# Manual resample
print("\n2. Resampling to hourly...")
t0 = time.time()
n_timesteps_hourly = len(fin_hf_loaded.time) // 6
rain_10min = fin_hf_loaded.RAIN.values[:n_timesteps_hourly * 6]
rain_reshaped = rain_10min.reshape(n_timesteps_hourly, 6, 50, 50)
rain_hourly = rain_reshaped.sum(axis=1)
t1 = time.time()
print(f"   Resampled in {t1-t0:.2f}s")
print(f"   Hourly shape: {rain_hourly.shape}")

# Create 2D histogram for each grid point
print("\n3. Creating 2D histograms...")
t0 = time.time()

# Initialize 3D array to store histograms: (ny, nx, nbins_10min, nbins_hourly)
nbins_10min = len(HIRAIN_BINS) - 1
nbins_hourly = len(LORAIN_BINS) - 1
hist_2d = np.zeros((tile_size, tile_size, nbins_10min, nbins_hourly), dtype=np.int32)

# For each grid point, create 2D histogram
for iy in range(tile_size):
    for ix in range(tile_size):
        # Get timeseries for this grid point
        # Need to align 10-min with hourly: repeat hourly 6 times
        rain_10min_point = rain_reshaped[:, :, iy, ix].flatten()  # All 10-min values
        rain_hourly_point = np.repeat(rain_hourly[:, iy, ix], 6)  # Repeat each hourly value 6 times
        
        # Create 2D histogram
        hist_2d[iy, ix, :, :], _, _ = np.histogram2d(
            rain_10min_point,
            rain_hourly_point,
            bins=[HIRAIN_BINS, LORAIN_BINS]
        )

t1 = time.time()
print(f"   Histograms created in {t1-t0:.2f}s")

# Create bin labels for coordinate (use bin centers, except for last bin which is ">100")
bin_centers_10min = (HIRAIN_BINS[:-2] + HIRAIN_BINS[1:-1]) / 2  # Centers for bins 0-100
bin_centers_10min = np.append(bin_centers_10min, 105)  # Use 105 as label for ">100" bin

bin_centers_hourly = (LORAIN_BINS[:-2] + LORAIN_BINS[1:-1]) / 2
bin_centers_hourly = np.append(bin_centers_hourly, 105)

# Save to netCDF
print("\n4. Saving histograms...")
t0 = time.time()

# Create xarray dataset
ds_hist = xr.Dataset(
    {
        'hist_2d': (['y', 'x', 'bin_10min', 'bin_hourly'], hist_2d)
    },
    coords={
        'y': fin_hf_loaded.y,
        'x': fin_hf_loaded.x,
        'bin_10min': bin_centers_10min,
        'bin_hourly': bin_centers_hourly
    },
    attrs={
        'description': '2D histograms of 10-min vs hourly precipitation',
        'tile_id': tile_id,
        'wet_threshold_10min': WET_VALUE_HIFREQ,
        'wet_threshold_hourly': WET_VALUE_LOFREQ,
        'bin_edges_10min': HIRAIN_BINS.tolist(),
        'bin_edges_hourly': LORAIN_BINS.tolist(),
        'note': 'Last bin includes all values > 100 mm'
    }
)

output_file = f'{PATH_OUT}/{WRUN}/histograms/hist_10min_hourly_{tile_id}.nc'
os.makedirs(os.path.dirname(output_file), exist_ok=True)
ds_hist.to_netcdf(output_file)

t1 = time.time()
print(f"   Saved to: {output_file}")
print(f"   Save time: {t1-t0:.2f}s")

print("\n" + "="*60)
print("Complete!")
print("="*60)