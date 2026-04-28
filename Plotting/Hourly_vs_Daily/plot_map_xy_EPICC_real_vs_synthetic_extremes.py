#!/usr/bin/env python
'''
@File    :  plot_map_EPICC_real_vs_synthetic_extremes.py
@Time    :  2025/10/27 11:20:49
@Author  :  Daniel Argüeso
@Version :  1.0
@Contact :  d.argueso@uib.es
@License :  (C)Copyright 2025, Daniel Argüeso
@Project :  EPICC
@Desc    :  None
'''

from matplotlib.ticker import ScalarFormatter
import os
import xarray as xr
import numpy as np
import netCDF4 as nc
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors
from matplotlib.colors import BoundaryNorm
from matplotlib import colormaps as cmaps
import string
import seaborn as sns
import matplotlib as mpl
from matplotlib.gridspec import GridSpec

from wrf import (to_np, getvar,get_cartopy, cartopy_xlim,GeoBounds,CoordPair,
                 cartopy_ylim, latlon_coords)

mpl.rcParams["font.size"] = 14
mpl.rcParams["hatch.color"] = "red"
mpl.rcParams["hatch.linewidth"] = 0.8
###########################################################
###########################################################
geo_file_name = "/home/dargueso/share/geo_em_files/EPICC/geo_em.d01.EPICC_2km_ERA5_HVC_GWD.nc"
geo_file = xr.open_dataset(geo_file_name)
lm_is=geo_file.LANDMASK.squeeze()

# Create and add gray border zone (50 grid points from edge)
border_width = 50
border_mask = np.zeros_like(lm_is.values)
border_mask[:border_width, :] = 1  # Top
border_mask[-border_width:, :] = 1  # Bottom
border_mask[:, :border_width] = 1  # Left
border_mask[:, -border_width:] = 1  # Right

med_mask = xr.open_dataset('/home/dargueso/postprocessed/EPICC/EPICC_2km_ERA5/my_coastal_med_mask.nc')

#####################################################################
#####################################################################

def get_geoinfo():

    fileref = nc.Dataset(geo_file_name)
    hgt = getvar(fileref, "HGT_M", timeidx=0)
    hgt = hgt.where(hgt>=0,0)
    lats, lons = latlon_coords(hgt)
    cart_proj = get_cartopy(hgt)

    return cart_proj,lats,lons,hgt

#####################################################################
#####################################################################

cart_proj,lats,lons,hgt = get_geoinfo()
mbounds = None
cart_proj._threshold /= 100.
subregs = np.zeros_like(lm_is.values)
#####################################################################
#####################################################################


qtile = 99
confidence_level = 0.995
buffer = 5 

WET_VALUE_LOW  = 0.1   # mm/d  — wet-day threshold (must match pipeline)
WET_VALUE_HIGH = 0.1   # mm/h  — wet-hour threshold

mylevels=np.arange(0, 45, 5)
cmap = sns.color_palette("icefire", as_cmap=True)
norm = BoundaryNorm(mylevels, ncolors=cmap.N, extend="max")

cmap_diff = cmaps["BrBG"]
norm_diff = BoundaryNorm(np.arange(-10, 11, 1), ncolors=cmap_diff.N, extend="both")


sig_levs = np.array([-0.5, 0.5, 1.5])
# mpl.rcParams['hatch.linewidth'] = 0.25



#Load data

#Loading percentiles

filein = f'/home/dargueso/postprocessed/EPICC//EPICC_2km_ERA5/percentiles_and_significance_1H_mann_whitney_seqio.nc'
fin = xr.open_dataset(filein).sel(percentile=qtile).squeeze()

#Loading SYN confidence intervals

filein_syn = f'/home/dargueso/postprocessed/EPICC/EPICC_2km_ERA5/synthetic_fut_buf{buffer}.nc'
fin_syn = xr.open_dataset(filein_syn).sel(plot_q=qtile/100.).squeeze()

fin_syn_all = xr.open_dataset(filein_syn).squeeze()
qtiles = fin_syn_all['plot_q'].values
cl_hi = round(confidence_level, 3)
cl_lo = round(1 - cl_hi, 3)

data = [fin['percentiles_present'].values, fin['percentiles_future'].values, fin_syn.syn_h_C.sel(bootstrap_q=confidence_level, method='nearest').squeeze()]

diff = data[1]- data[0]
sig = data[1]- data[2]
sig_var= sig > 0
sig_var = sig_var.astype(int)

#####################################################################
#####################################################################

locs_x_idx = [559,423,569,795,638,821,1091,989]#,433,866,335]
locs_y_idx = [258,250,384,527,533,407,174,425]#,254,506,119]
locs_names = ['Mallorca','Turis','Pyrenees','Rosiglione', 'Ardeche','Corte','Catania',"L'Aquila"]#,'Valencia','Barga','Almeria']

fig = plt.figure(figsize=(15, 20), constrained_layout=False)
# Create main GridSpec with separate spacing control
gs_main = GridSpec(3, 1, height_ratios=[1, 1, 3], figure=fig,
                   left=0.05, bottom=0.15, right=0.95, top=0.9,
                   hspace=0.25)

# Create nested GridSpecs for the first two rows (4 columns each)
gs_row0 = gs_main[0].subgridspec(1, 4, wspace=0.1)
gs_row1 = gs_main[1].subgridspec(1, 4, wspace=0.1)
# Third row will use gs_main[2] directly for the map


#####################################################################
#####################################################################

zarr_path_present = f'/home/dargueso/postprocessed/EPICC/EPICC_2km_ERA5/UIB_01H_RAIN.zarr'
zarr_path_future  = f'/home/dargueso/postprocessed/EPICC/EPICC_2km_ERA5_CMIP6anom/UIB_01H_RAIN.zarr'

try:
    ds_present = xr.open_zarr(zarr_path_present, consolidated=True)
    ds_future = xr.open_zarr(zarr_path_future, consolidated=True)
    print("   Opened with consolidated metadata")
except KeyError:
    ds_present = xr.open_zarr(zarr_path_present, consolidated=False)
    ds_future = xr.open_zarr(zarr_path_future, consolidated=False)
    print("   Opened without consolidated metadata")

for loc in range(len(locs_names)):

    loc_name = locs_names[loc]
    print(loc_name)
    xloc = locs_x_idx[loc]
    yloc = locs_y_idx[loc]

    # filein_pres = f'/home/dargueso/postprocessed/EPICC//EPICC_2km_ERA5/UIB_01H_RAIN_{yloc:3d}y-{xloc:3d}x_{buffer:03d}buffer.nc'
    # fin_pres = xr.open_dataset(filein_pres).squeeze()

    # filein_fut = f'/home/dargueso/postprocessed/EPICC/EPICC_2km_ERA5_CMIP6anom/UIB_01H_RAIN_{yloc:3d}y-{xloc:3d}x_{buffer:03d}buffer.nc'
    # fin_fut = xr.open_dataset(filein_fut).squeeze()

    fin_pres = ds_present.RAIN.isel(
            y=slice(yloc-buffer,yloc+buffer+1),
            x=slice(xloc-buffer,xloc+buffer+1),
        ).astype(np.float32)
    
    fin_fut = ds_future.RAIN.isel(
        y=slice(yloc-buffer,yloc+buffer+1),
        x=slice(xloc-buffer,xloc+buffer+1),
    ).astype(np.float32)

    fin_syn = fin_syn_all.isel(y=yloc,x=xloc).squeeze()

    locx = lons[locs_y_idx[loc],locs_x_idx[loc]].values
    locy = lats[locs_y_idx[loc],locs_x_idx[loc]].values


    subregs[yloc-buffer:yloc+buffer+1,xloc-buffer:xloc+buffer+1]=1



    #fin_pres_loc = fin_pres.where(fin_pres.RAIN>0.1).isel(y=25,x=25).dropna(dim="time")
    #fin_fut_loc = fin_fut.where(fin_fut.RAIN>0.1).isel(y=25,x=25).dropna(dim="time")

    # Load to numpy — buffer region is small (2*buf+1)^2 pixels
    pres_np = fin_pres.values.astype(np.float32)   # (nt, ny_buf, nx_buf)
    fut_np  = fin_fut.values.astype(np.float32)

    # Daily totals via reshape (assumes N_INTERVAL=24 hourly steps per day)
    N_INTERVAL = 24
    n_days_p = pres_np.shape[0] // N_INTERVAL
    n_days_f = fut_np.shape[0]  // N_INTERVAL
    nt_p = n_days_p * N_INTERVAL
    nt_f = n_days_f * N_INTERVAL

    pres_daily_np = pres_np[:nt_p].reshape(n_days_p, N_INTERVAL,
                                            pres_np.shape[1], pres_np.shape[2]).sum(axis=1)
    fut_daily_np  = fut_np[:nt_f].reshape(n_days_f, N_INTERVAL,
                                           fut_np.shape[1], fut_np.shape[2]).sum(axis=1)

    # Wet-day mask broadcast back to hourly by repeating each day 24 times
    pres_wet_mask_h = np.repeat(pres_daily_np >= WET_VALUE_LOW, N_INTERVAL, axis=0)
    fut_wet_mask_h  = np.repeat(fut_daily_np  >= WET_VALUE_LOW, N_INTERVAL, axis=0)

    # Apply wet-hour and wet-day filters, flatten, drop NaN
    pres_filtered = np.where((pres_np[:nt_p] > WET_VALUE_HIGH) & pres_wet_mask_h,
                             pres_np[:nt_p], np.nan).reshape(-1)
    fut_filtered  = np.where((fut_np[:nt_f]  > WET_VALUE_HIGH) & fut_wet_mask_h,
                             fut_np[:nt_f],  np.nan).reshape(-1)
    pres_wet_1d = pres_filtered[np.isfinite(pres_filtered)]
    fut_wet_1d  = fut_filtered[np.isfinite(fut_filtered)]

    fin_pres_qtiles = np.quantile(pres_wet_1d, qtiles) if len(pres_wet_1d) > 0 else np.full(len(qtiles), np.nan)
    fin_fut_qtiles  = np.quantile(fut_wet_1d,  qtiles) if len(fut_wet_1d)  > 0 else np.full(len(qtiles), np.nan)

    # ------------------------------------------------------------------
    # CROSS-CHECK vs pipeline_multi_location NPZ
    # ------------------------------------------------------------------
    npz_path = f'/home/dargueso/Analyses/EPICC/Hourly_vs_Daily/testing_data/{loc_name}_buf{buffer}.npz'
    if os.path.exists(npz_path):
        h = np.load(npz_path)
        npz_q          = h['plot_quantiles']           # (n_q,)
        npz_obs_pres   = h['obs_pres_h_buf']           # (n_q,)
        npz_obs_fut    = h['obs_fut_h_buf']            # (n_q,)
        npz_syn_med    = np.nanquantile(h['syn_fut_c_h_boot_buf'], 0.5, axis=0)  # (n_q,)
        map_syn_med    = fin_syn.sel(bootstrap_q=0.5, method='nearest').syn_h_C.values  # (n_q,)

        print(f"\n  [{loc_name}] Cross-check full_domain vs multi_location (buf={buffer})")
        print(f"  {'Quantile':>10}  {'ObsPres_map':>12}  {'ObsPres_npz':>12}  {'diff%':>7}"
              f"  {'ObsFut_map':>11}  {'ObsFut_npz':>11}  {'diff%':>7}"
              f"  {'SynFut_map':>11}  {'SynFut_npz':>11}  {'diff%':>7}")
        for i, q in enumerate(qtiles):
            # find matching index in npz_q
            j = np.argmin(np.abs(npz_q - q))
            if not np.isclose(npz_q[j], q, atol=0.001):
                continue
            op_m = fin_pres_qtiles[i];  op_n = npz_obs_pres[j]
            of_m = fin_fut_qtiles[i];   of_n = npz_obs_fut[j]
            sf_m = map_syn_med[i];      sf_n = npz_syn_med[j]
            dp = 100*(op_m - op_n)/op_n if op_n != 0 else np.nan
            df = 100*(of_m - of_n)/of_n if of_n != 0 else np.nan
            ds = 100*(sf_m - sf_n)/sf_n if sf_n != 0 else np.nan
            print(f"  {q:>10.4f}  {op_m:>12.4f}  {op_n:>12.4f}  {dp:>+7.2f}%"
                  f"  {of_m:>11.4f}  {of_n:>11.4f}  {df:>+7.2f}%"
                  f"  {sf_m:>11.4f}  {sf_n:>11.4f}  {ds:>+7.2f}%")
    else:
        print(f"\n  [{loc_name}] No NPZ found at {npz_path} — skipping cross-check")

    nw = loc%4
    nr = loc//4
    nletter = nw + nr*4

    # if nr == 1:
    #     nr=nr+1
    #     nletter = nletter + 1


    # Select the appropriate GridSpec based on row
    if nr == 0:
        ax = fig.add_subplot(gs_row0[0, nw])
    elif nr == 1:
        ax = fig.add_subplot(gs_row1[0, nw])
    else:
        continue  # Skip row 2 as it's for the map
    ax.set_title(f"{string.ascii_lowercase[nletter]}", size='x-large', weight='bold',loc="left")
    ax.text(0.02, 0.98, loc_name, color='black', fontsize=10, 
        transform=ax.transAxes, zorder=103, 
        verticalalignment='top', horizontalalignment='left')
    # Plot with improved styling
    ax.plot(qtiles, fin_pres_qtiles, label='Present-day observations', 
            color='#2E86AB', linewidth=1, marker='o', markersize=2)
    ax.plot(qtiles, fin_fut_qtiles, label='Future observations', 
            color="#E50C0C", linewidth=1, marker='s', markersize=2)
    ax.plot(qtiles, fin_syn.sel(bootstrap_q=cl_hi, method='nearest').syn_h_C.squeeze(),
            label='Future synthetic', color='#F18F01', linewidth=0.5,
            linestyle='--', marker=None)
    ax.plot(qtiles, fin_syn.sel(bootstrap_q=cl_lo, method='nearest').syn_h_C.squeeze(),
            color='#F18F01', linewidth=0.5,
            linestyle='--', marker=None)
    ax.fill_between(qtiles, fin_syn.sel(bootstrap_q=cl_lo, method='nearest').syn_h_C.squeeze(),
                    fin_syn.sel(bootstrap_q=cl_hi, method='nearest').syn_h_C.squeeze(),
                    color='#F18F01', alpha=0.2)


    # Log scale for y-axis
    ax.set_yscale('log')

    # Labels and title with better formatting
    if nr == 1:
        ax.set_xlabel('Quantiles', fontsize=12, fontweight='bold')
    if nw == 0:
        ax.set_ylabel('1-hour precipitation (mm)', fontsize=12, fontweight='bold')
    
    if nw != 0:
        ax.yaxis.set_tick_params(labelleft=False)

    #ax.set_title(f'1-hour Precipitation Quantiles at {loc_name}', 
    #            fontsize=14, fontweight='bold', pad=20)

    # Improved legend
    #if loc == 7:
        # ax.legend(frameon=True, fancybox=True, shadow=True, 
        #   fontsize=9, loc='upper left', bbox_to_anchor=(1, 1))

    # Enhanced grid
    ax.grid(True, which='both', linestyle=':', linewidth=0.5, alpha=0.7)
    ax.set_axisbelow(True)

    yticks = [5, 10, 20, 50]  # Adjust to your data range
    ax.set_yticks(yticks)
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.get_major_formatter().set_scientific(False)

    # Adjust tick label sizes
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.minorticks_on()


axs= fig.add_subplot(gs_main[2],projection=cart_proj)
axs.set_title(f"{string.ascii_lowercase[8]}", size='x-large', weight='bold',loc="left")
axs.add_feature(cfeature.COASTLINE,linewidth=0.5,zorder=102)

dtrain = axs.pcolormesh(
            lons,
            lats,
            diff,
            cmap=cmap_diff,
            norm=norm_diff,
            transform=ccrs.PlateCarree(),
            zorder = 101
        )

axs.contourf(to_np(lons), to_np(lats), border_mask,
                levels=[0.5, 1.5],
                colors=['gray'],
                alpha=0.5,
                transform=ccrs.PlateCarree(),
                zorder=103)

axs.contour(
    to_np(lons), to_np(lats), to_np(subregs),
    levels=[0.5],                 # boundary between 0 and 1
    colors='red',
    linewidths=2,
    transform=ccrs.PlateCarree(),
    zorder=105,
)

axs.set_xlim(cartopy_xlim(hgt,geobounds=mbounds))
axs.set_ylim(cartopy_ylim(hgt,geobounds=mbounds))
gl=axs.gridlines(color="black", linestyle="dotted",linewidth=0.5,draw_labels=True,x_inline=False, y_inline=False,zorder=103)
gl.right_labels=False
gl.top_labels=False

cs = axs.contourf(to_np(lons),
                to_np(lats),
                sig_var, 
                sig_levs, colors='none',
                hatches=["","....."],
                transform=ccrs.PlateCarree(),
                zorder=105)


cbar = plt.colorbar(dtrain, ax=axs, location='right', orientation="vertical", 
                    shrink=0.65, label="Precipitation (mm)")

print(('Total significant pixels:', np.sum(sig_var*to_np(med_mask['combined_mask'].values==2))))
print(('Percentage of significant pixels:', 
       np.sum(sig_var*to_np(med_mask['combined_mask'].values==2))/
       np.sum(to_np(med_mask['combined_mask'].values==2))*100))

plt.savefig(f'/home/dargueso/Analyses/EPICC/Hourly_vs_Daily/map_and_quantiles_01H_precipitation_q{qtile}th_cl{confidence_level}_{buffer:03d}buffer.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
