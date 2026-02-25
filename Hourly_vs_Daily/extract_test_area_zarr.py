#!/usr/bin/env python
"""
Extract a 100x100 area around a specific lat/lon location for testing purposes.
Creates a new Zarr file with the same format as the original.
"""
import os
import time
import numpy as np
import xarray as xr
import dask

# Configure dask
dask.config.set(scheduler='threads', num_workers=16)

# =============================================================================
# CONFIGURATION
# =============================================================================

PATH_IN = '/home/dargueso/postprocessed/EPICC/'
PATH_OUT = '/home/dargueso/postprocessed/EPICC/'
WRUN = "EPICC_2km_ERA5_CMIP6anom"

# Frequency to extract (change as needed)
FREQ = '01H'  # Options: '10MIN', '01H', 'DAY'

loc_lats = {'Mallorca': 39.639, 'Barcelona': 41.385, 'Valencia': 39.469,'Rosiglione': 44.55, 'Catania': 37.51 }
loc_lons = {'Mallorca': 2.647, 'Barcelona': 2.173, 'Valencia': -0.376,'Rosiglione': 8.64, 'Catania': 15.08}

# Target location
TARGET_LAT = loc_lats['Catania']
TARGET_LON = loc_lons['Catania']
# Extraction size
GRID_SIZE = 21  # 101x101 grid (use odd number for true centering)
                 # Odd sizes (101, 99, etc.) can be perfectly centered on the target point
                 # Even sizes (100, 102, etc.) will be offset by half a grid cell

# Chunking for output (same as original)
TIME_CHUNK = 8760
SPATIAL_CHUNK = 50

# =============================================================================
# MAIN
# =============================================================================

print("="*80)
print(f"Extracting {GRID_SIZE}x{GRID_SIZE} test area around lat={TARGET_LAT}, lon={TARGET_LON}")
print("="*80)

total_start = time.time()

# Input and output paths
zarr_path_in = f'{PATH_IN}/{WRUN}/UIB_{FREQ}_RAIN.zarr'
zarr_path_out = f'{PATH_OUT}/{WRUN}/UIB_{FREQ}_RAIN_test_{GRID_SIZE}x{GRID_SIZE}.zarr'

print(f"\nInput:  {zarr_path_in}")
print(f"Output: {zarr_path_out}")

# Open the zarr file
print(f"\n1. Opening Zarr dataset...")
t0 = time.time()
try:
    ds = xr.open_zarr(zarr_path_in, consolidated=True)
    print("   Opened with consolidated metadata")
except KeyError:
    ds = xr.open_zarr(zarr_path_in, consolidated=False)
    print("   Opened without consolidated metadata")

t1 = time.time()
print(f"   Time: {t1-t0:.2f}s")
print(f"   Original shape: {ds.RAIN.shape}")
print(f"   Dimensions: time={len(ds.time)}, y={len(ds.y)}, x={len(ds.x)}")

# Get lat/lon arrays (they should be 2D: y, x)
print(f"\n2. Finding closest grid point to target location...")
lat = ds.lat.isel(time=0).values  # Assuming lat doesn't vary with time
lon = ds.lon.isel(time=0).values  # Assuming lon doesn't vary with time

# Calculate distances from target point
# Using simple Euclidean distance (fine for small areas)
distances = np.sqrt((lat - TARGET_LAT)**2 + (lon - TARGET_LON)**2)

# Find the closest grid point
min_idx = np.unravel_index(np.argmin(distances), distances.shape)
center_y, center_x = min_idx

print(f"   Target: lat={TARGET_LAT:.4f}, lon={TARGET_LON:.4f}")
print(f"   Closest grid point: y={center_y}, x={center_x}")
print(f"   Actual: lat={lat[center_y, center_x]:.4f}, lon={lon[center_y, center_x]:.4f}")
print(f"   Distance: {distances[center_y, center_x]:.6f} degrees")

# Check if grid size is odd or even
if GRID_SIZE % 2 == 0:
    print(f"\n   NOTE: Even grid size ({GRID_SIZE}x{GRID_SIZE}) cannot be truly centered.")
    print(f"         For true centering, use an odd grid size (e.g., {GRID_SIZE-1} or {GRID_SIZE+1})")

# Calculate extraction bounds (centered on target point)
half_size = GRID_SIZE // 2

if GRID_SIZE % 2 == 1:
    # Odd grid size - can be truly centered
    y_start = max(0, center_y - half_size)
    y_end = min(len(ds.y), center_y + half_size + 1)  # +1 to include center + half_size
    x_start = max(0, center_x - half_size)
    x_end = min(len(ds.x), center_x + half_size + 1)
else:
    # Even grid size - cannot be truly centered (center falls between grid points)
    y_start = max(0, center_y - half_size)
    y_end = min(len(ds.y), center_y + half_size)
    x_start = max(0, center_x - half_size)
    x_end = min(len(ds.x), center_x + half_size)

# Adjust to ensure we get exactly GRID_SIZE if possible
actual_y_size = y_end - y_start
actual_x_size = x_end - x_start

if actual_y_size < GRID_SIZE:
    print(f"   WARNING: Domain boundary limits y-dimension to {actual_y_size} (requested {GRID_SIZE})")
if actual_x_size < GRID_SIZE:
    print(f"   WARNING: Domain boundary limits x-dimension to {actual_x_size} (requested {GRID_SIZE})")

print(f"\n3. Extracting subset...")
print(f"   y range: [{y_start}, {y_end}) = {actual_y_size} points")
print(f"   x range: [{x_start}, {x_end}) = {actual_x_size} points")

# Calculate where the center point will be in the extracted grid
center_y_in_subset = center_y - y_start
center_x_in_subset = center_x - x_start
print(f"   Center point in extracted grid: y={center_y_in_subset}, x={center_x_in_subset}")

if GRID_SIZE % 2 == 1:
    expected_center = GRID_SIZE // 2
    if center_y_in_subset == expected_center and center_x_in_subset == expected_center:
        print(f"   ✓ Perfectly centered at [{expected_center}, {expected_center}]")
    else:
        print(f"   ⚠ Center offset from expected [{expected_center}, {expected_center}] due to domain edge")

# Extract the subset
ds_subset = ds.isel(
    y=slice(y_start, y_end),
    x=slice(x_start, x_end)
)

# Update the y and x coordinates to be 0-indexed for the subset
ds_subset = ds_subset.assign_coords({
    'y': np.arange(actual_y_size),
    'x': np.arange(actual_x_size)
})

print(f"   Subset shape: {ds_subset.RAIN.shape}")
print(f"   Subset chunks: {ds_subset.RAIN.chunks}")

# Rechunk to match original chunking scheme
print(f"\n4. Rechunking for optimal performance...")
t0 = time.time()
ds_subset_rechunked = ds_subset.chunk({
    'time': TIME_CHUNK,
    'y': SPATIAL_CHUNK,
    'x': SPATIAL_CHUNK
})
t1 = time.time()
print(f"   Rechunked in {t1-t0:.2f}s")
print(f"   New chunks: {ds_subset_rechunked.RAIN.chunks}")

# Add metadata about extraction
ds_subset_rechunked.attrs['extraction_info'] = (
    f"Extracted from full domain: center at lat={TARGET_LAT}, lon={TARGET_LON}, "
    f"grid indices y={center_y}, x={center_x}, "
    f"extracted region: y=[{y_start}:{y_end}), x=[{x_start}:{x_end})"
)
ds_subset_rechunked.attrs['original_file'] = zarr_path_in
ds_subset_rechunked.attrs['extraction_date'] = time.strftime('%Y-%m-%d %H:%M:%S')

# Encoding for zarr (same as original)
encoding = {
    'RAIN': {
        'compressor': None,  # No compression for speed
        'dtype': 'float32'
    }
}

# Write to zarr
print(f"\n5. Writing to Zarr...")
print(f"   Output: {zarr_path_out}")
t0 = time.time()

# Remove existing if present
if os.path.exists(zarr_path_out):
    import shutil
    print(f"   Removing existing zarr store...")
    shutil.rmtree(zarr_path_out)

ds_subset_rechunked.to_zarr(
    zarr_path_out, 
    mode='w', 
    encoding=encoding, 
    consolidated=True
)
t1 = time.time()
print(f"   Written in {t1-t0:.2f}s ({(t1-t0)/60:.2f} minutes)")

# Verify the output
print(f"\n6. Verifying output...")
ds_check = xr.open_zarr(zarr_path_out, consolidated=True)
print(f"   Shape: {ds_check.RAIN.shape}")
print(f"   Chunks: {ds_check.RAIN.chunks}")
print(f"   Variables: {list(ds_check.data_vars)}")
print(f"   Coordinates: {list(ds_check.coords)}")
print(f"   Lat range: [{ds_check.lat.min().values:.4f}, {ds_check.lat.max().values:.4f}]")
print(f"   Lon range: [{ds_check.lon.min().values:.4f}, {ds_check.lon.max().values:.4f}]")
ds_check.close()

# Close original dataset
ds.close()

total_time = time.time() - total_start
print("\n" + "="*80)
print(f"Extraction complete! Total time: {total_time/60:.2f} minutes")
print(f"\nTest area saved to: {zarr_path_out}")
print("="*80)
print("\nYou can now use this test file with your processing scripts by updating:")
print("  PATH_IN or PATH_OUT to point to the test file")
print("  Or create a test version: WRUN = 'EPICC_2km_ERA5' -> WRUN_TEST = 'EPICC_2km_ERA5'")
print("="*80)
