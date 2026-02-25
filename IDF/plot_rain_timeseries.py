import pickle
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import wrf
import glob
import joblib 

# ------------------- CONFIG ------------------- #
# Paths
geopath_in = "/home/yseut/data/WRF_data/EPICC_2km_ERA5"
geofile_ref = nc.Dataset(f"{geopath_in}/geo_em.d01.EPICC_2km.nc")

pickle_file = glob.glob("rainfall_data_2011_2020_*x_*y.pkl")
pickle_file_name = pickle_file[0]

# Extract x and y from the filename
parts = pickle_file_name.split("_")
selected_x = int(parts[-2][:-1])  # Extract number before 'x'
selected_y = int(parts[-1][:-5])  # Extract number before 'y.pkl'

print(f"Loading file: {pickle_file_name} with indices x={selected_x}, y={selected_y}")

## Selected indices
#selected_x = 730
#selected_y = 508

# ------------------- LOAD GEO DATA ------------------- #
# Get projection, lat/lon, terrain data
geo_proj = wrf.get_cartopy(wrfin=geofile_ref)
lats, lons = wrf.latlon_coords(wrf.getvar(geofile_ref, "ter"))
terrain = wrf.getvar(geofile_ref, "ter").values

# Extract selected location (for plotting verification)
selected_lon = lons[selected_y, selected_x]
selected_lat = lats[selected_y, selected_x]

# ------------------- 1️⃣ PLOT MAP WITH SELECTED POINT ------------------- #
def plot_map():
    """Plot the map and mark the selected point to verify location."""
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': geo_proj})
    ax.pcolormesh(lons, lats, terrain, transform=ccrs.PlateCarree(), cmap='terrain', alpha=0.5)
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    
    # Mark the selected point
    ax.scatter(selected_lon, selected_lat, color='red', s=100, edgecolor='black',
               transform=ccrs.PlateCarree(), label="Selected Point")
    
    ax.set_title("Selected Point Verification on Map")
    ax.legend(loc='upper right')
    plt.show()

# ------------------- 2️⃣ OPEN PICKLE AND PLOT TIME SERIES ------------------- #
def plot_rainfall_timeseries():
    """Load the pickle file and plot the time series of rainfall."""

    rainfall_data = joblib.load(pickle_file_name)  # Much faster than pickle

    #with open(pickle_file_name, "rb") as f:
    #    rainfall_data = pickle.load(f)

    # Ensure the dataset is an xarray object
    if not isinstance(rainfall_data, xr.DataArray):
        print("Error: Loaded pickle file is not an xarray DataArray!")
        return

    # Plot time series
    plt.figure(figsize=(12, 5))
    plt.plot(rainfall_data.time, rainfall_data, color='blue', linewidth=0.8)
    plt.xlabel("Time")
    plt.ylabel("10-Minute Rainfall (mm)")
    plt.title(f"Rainfall Time Series (2011-2020) at (x={selected_x}, y={selected_y})")
    plt.grid(True)
    plt.show()

# ------------------- RUN FUNCTIONS ------------------- #
plot_map()  # Step 1: Plot map with selected point
plot_rainfall_timeseries()  # Step 2: Plot time series

