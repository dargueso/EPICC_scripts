import os
import xarray as xr
import numpy as np
import pickle
import netCDF4 as nc
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import wrf
from tqdm import tqdm
import matplotlib
matplotlib.use("TkAgg")

# ------------------- PATHS ------------------- #
# path to monthly rain files
#data_dir = "/scratch0/yseut/postprocessed/EPICC/EPICC_2km_ERA5/bymonth/"
data_dir = "/scratch2/yseut/postprocessed/EPICC/EPICC_2km_ERA5_CMIP6anom/bymonth/"

# path to geoem file
geopath_in = "/home/yseut/data/WRF_data/EPICC_2km_ERA5"

# ------------------- LOAD GEO DATA ------------------- #
geofile_ref = nc.Dataset(f"{geopath_in}/geo_em.d01.EPICC_2km.nc")
geo_proj = wrf.get_cartopy(wrfin=geofile_ref)
#lats, lons = wrf.latlon_coords(wrf.getvar(geofile_ref, "ter"))
terrain = wrf.getvar(geofile_ref, "ter").values
land_mask = wrf.getvar(geofile_ref, "LANDMASK").values

# ------------------- PLOT MAP ------------------- #
fig, ax = plt.subplots(figsize=(10, 8))#, subplot_kw={'projection': geo_proj})
ax.pcolormesh(terrain, cmap='terrain', alpha=0.5)#, transform=ccrs.PlateCarree())
ax.pcolormesh(land_mask, cmap='gray', alpha=0.3)
ax.set_title("Select a Point for 10MIN Rainfall Extraction (2011-2020)")

# ------------------- FUNCTION: Extract Data ------------------- #
def extract_rainfall(x_idx, y_idx):
    """ Extract 10-minute rainfall data from 2011 to 2020 for selected grid point. """
    years = range(2011, 2021)  # Years to process
    all_data = []  # Store data across all years

    for year in years:
        for month in range(1, 13):  # Loop over months
            file_path = os.path.join(data_dir, f"UIB_10MIN_RAIN_{year}-{month:02d}.nc")

            if not os.path.exists(file_path):
                print(f"WARNING: File {file_path} not found, skipping...")
                continue

            # Open dataset
            ds = xr.open_dataset(file_path)
            
             # Select rainfall at the given point
            rain_point = ds["RAIN"].isel(y=y_idx, x=x_idx)

            # Convert time units (if needed)
            if not np.issubdtype(rain_point.time.dtype, np.datetime64):
                ds = xr.decode_cf(ds)  # Ensure proper time format
                rain_point = ds["RAIN"].isel(y=y_idx, x=x_idx)

            all_data.append(rain_point)

    # Concatenate all monthly data along time
    if all_data:
        print('concatenating monthly data...')
        full_dataset = xr.concat(all_data, dim="time")
        print(f"Extracted {full_dataset.sizes['time']} time steps.")
        return full_dataset
    else:
        print("No data extracted!")
        return None

# ------------------- FUNCTION: Click Select and extract ------------------- #

def onclick(event):
    """ Select grid point, extract data, and save as pickle. """
    if event.xdata is not None and event.ydata is not None:
        lon_sel, lat_sel = event.xdata, event.ydata
        #import pdb; pdb.set_trace()

        x_idx = int(round(lon_sel))
        y_idx = int(round(lat_sel))
        print(f"Selected indices: x={x_idx}, y={y_idx}")

        # Extract data
        rainfall_data = extract_rainfall(x_idx, y_idx)

        # Output pickle file
        #output_pickle = f"rainfall_data_2011_2020_{x_idx}x_{y_idx}y.pkl"
        output_pickle = f"rainfall_data_2011_2020_{x_idx}x_{y_idx}y_future.pkl"
        # Save to pickle if data exists
        if rainfall_data is not None:
            with open(output_pickle, "wb") as f:
                pickle.dump(rainfall_data, f)
            print(f"Saved rainfall data to {output_pickle}")

# ------------------- START INTERACTIVE SELECTION ------------------- #
fig.canvas.mpl_connect('button_press_event', onclick)
plt.show()


