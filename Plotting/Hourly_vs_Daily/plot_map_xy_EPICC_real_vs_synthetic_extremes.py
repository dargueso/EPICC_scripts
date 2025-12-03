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


qtile = 0.99
confidence_level = 0.975
buffer = 10

mylevels=np.arange(0, 45, 5)
cmap = sns.color_palette("icefire", as_cmap=True)
norm = BoundaryNorm(mylevels, ncolors=cmap.N, extend="max")

cmap_diff = cmaps["BrBG"]
norm_diff = BoundaryNorm(np.arange(-10, 11, 1), ncolors=cmap_diff.N, extend="both")


sig_levs = np.array([-0.5, 0.5, 1.5])
# mpl.rcParams['hatch.linewidth'] = 0.25



#Load data

#Loading CTL percentiles

filein_pres = '/home/dargueso/postprocessed/EPICC//EPICC_2km_ERA5/UIB_01H_RAIN_2011-2020_qtiles_wetonly.nc'
fin_pres = xr.open_dataset(filein_pres).sel(quantile=qtile).squeeze()
#Loading PGW percentiles

filein_fut = '/home/dargueso/postprocessed/EPICC/EPICC_2km_ERA5_CMIP6anom/UIB_01H_RAIN_2011-2020_qtiles_wetonly.nc'
fin_fut = xr.open_dataset(filein_fut).sel(quantile=qtile).squeeze()

#Loading SYN confidence intervals

filein_syn = f'/home/dargueso/postprocessed/EPICC/EPICC_2km_ERA5_CMIP6anom/future_synthetic_quant_confidence_{buffer:03d}buffer.nc'
fin_syn = xr.open_dataset(filein_syn).sel(qs_time = qtile).squeeze()

fin_syn_all = xr.open_dataset(filein_syn).squeeze()
fin_syn_all = fin_syn_all.isel(qs_time=slice(10,-2))
qtiles= fin_syn_all.qs_time.values
cl_hi = round(confidence_level,3)
cl_lo = round(1 - cl_hi, 3)

data = [fin_pres.RAIN, fin_fut.RAIN, fin_syn.precipitation.sel(quantile=confidence_level).squeeze()]

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

for loc in range(len(locs_names)):



    loc_name = locs_names[loc]
    print(loc_name)
    xloc = locs_x_idx[loc]
    yloc = locs_y_idx[loc]

    filein_pres = f'/home/dargueso/postprocessed/EPICC//EPICC_2km_ERA5/UIB_01H_RAIN_{yloc:3d}y-{xloc:3d}x_{buffer:03d}buffer.nc'
    fin_pres = xr.open_dataset(filein_pres).squeeze()

    filein_fut = f'/home/dargueso/postprocessed/EPICC/EPICC_2km_ERA5_CMIP6anom/UIB_01H_RAIN_{yloc:3d}y-{xloc:3d}x_{buffer:03d}buffer.nc'
    fin_fut = xr.open_dataset(filein_fut).squeeze()

    fin_syn = fin_syn_all.isel(y=yloc,x=xloc).squeeze()

    locx = lons[locs_y_idx[loc],locs_x_idx[loc]].values
    locy = lats[locs_y_idx[loc],locs_x_idx[loc]].values


    subregs[yloc-buffer:yloc+buffer+1,xloc-buffer:xloc+buffer+1]=1



    #fin_pres_loc = fin_pres.where(fin_pres.RAIN>0.1).isel(y=25,x=25).dropna(dim="time")
    #fin_fut_loc = fin_fut.where(fin_fut.RAIN>0.1).isel(y=25,x=25).dropna(dim="time")

    fin_pres_loc = fin_pres.where(fin_pres.RAIN>0.1).stack(xyt=("time","y","x")).dropna(dim="xyt")
    fin_fut_loc = fin_fut.where(fin_fut.RAIN>0.1).stack(xyt=("time","y","x")).dropna(dim="xyt")

    fin_pres_qtiles = fin_pres_loc.quantile(qtiles, dim='xyt', skipna=True)
    fin_fut_qtiles = fin_fut_loc.quantile(qtiles, dim='xyt', skipna=True)


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
    ax.plot(qtiles, fin_pres_qtiles.RAIN, label='Present-day observations', 
            color='#2E86AB', linewidth=1, marker='o', markersize=2)
    ax.plot(qtiles, fin_fut_qtiles.RAIN, label='Future observations', 
            color="#E50C0C", linewidth=1, marker='s', markersize=2)
    ax.plot(qtiles, fin_syn.sel(quantile=cl_hi).precipitation.squeeze(), 
            label='Future synthetic', color='#F18F01', linewidth=0.5, 
            linestyle='--', marker=None)
    ax.plot(qtiles, fin_syn.sel(quantile=cl_lo).precipitation.squeeze(), 
            color='#F18F01', linewidth=0.5, 
            linestyle='--', marker=None)
    ax.fill_between(qtiles, fin_syn.sel(quantile=cl_lo).precipitation.squeeze(), 
                    fin_syn.sel(quantile=cl_hi).precipitation.squeeze(), 
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

plt.savefig(f'/home/dargueso/Analyses/EPICC/Hourly_vs_Daily/map_and_quantiles_1H_precipitation_q{qtile}th_cl{confidence_level}_{buffer:03d}buffer.png', 
                dpi=300, bbox_inches='tight', facecolor='white')