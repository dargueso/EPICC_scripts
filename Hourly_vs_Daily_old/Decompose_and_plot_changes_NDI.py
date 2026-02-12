#!/usr/bin/env python
'''
@File    :  Decompose_changes_NDI.py
@Time    :  2025/07/24 12:33:07
@Author  :  Daniel Argüeso
@Version :  1.0
@Contact :  d.argueso@uib.es
@License :  (C)Copyright 2025, Daniel Argüeso
@Project :  None
@Desc    :  None
'''

import xarray as xr
import numpy as np
import time as ttime
import config as cfg
import netCDF4 as nc
import matplotlib.pyplot as plt
import cartopy.crs as crs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from matplotlib.colors import BoundaryNorm
from matplotlib import colormaps as cmaps
from wrf import (
    to_np,
    getvar,
    get_cartopy,
    cartopy_xlim,
    GeoBounds,
    CoordPair,
    cartopy_ylim,
    latlon_coords,
)
import cmocean

for seas in ['MAM', 'JJA', 'SON', 'DJF']:

    output_filename = f"{seas}.png"
    finp = xr.open_dataset(f"{cfg.path_out}/EPICC_2km_ERA5/Hourly_decomposition_NDI_{seas}.nc")
    finf = xr.open_dataset(f"{cfg.path_out}/EPICC_2km_ERA5_CMIP6anom/Hourly_decomposition_NDI_{seas}.nc")

    Np = finp.n_events
    Ip = finp.mean_intensity
    Dp = finp.mean_duration
    Nf = finf.n_events
    If = finf.mean_intensity
    Df = finf.mean_duration

    Dhours = Df*Nf- Dp*Np


    Dtotpr = finf.total_precipitation - finp.total_precipitation
    DN = finf.n_events - finp.n_events
    DI = finf.mean_intensity - finp.mean_intensity
    DD = finf.mean_duration - finp.mean_duration

    DPN = DN*Ip*Dp
    DPI = DI*Np*Dp
    DPD = DD*Np*Ip

    Res = Ip*DN*DD+ Np*DI*DD + Dp*DN*DI + DD*DI*DN

    Epsilon_rel =  (Dtotpr - DPN - DPI - DPD - Res)*100/Dtotpr
    #np.sum(np.abs((Dtotpr - DPN - DPI - DPD - Res)*100/Dtotpr)>1
    #np.max(np.abs((Dtotpr - DPN - DPI - DPD - Res)*100/Dtotpr))

    print(np.nanmean(np.abs(DPN/Dtotpr)))
    print(np.nanmean(np.abs(DPI/Dtotpr)))
    print(np.nanmean(np.abs(DPD/Dtotpr)))
    print(np.nanmean(np.abs(Res/Dtotpr)))
    print(np.nanmean(np.abs(Epsilon_rel)))

    print(np.nanmean((DPN/Dtotpr)))
    print(np.nanmean((DPI/Dtotpr)))
    print(np.nanmean((DPD/Dtotpr)))
    print(np.nanmean((Res/Dtotpr)))



    ## PLOTTING ##


    geo_filename = "/home/dargueso/share/geo_em_files/EPICC/geo_em.d01.EPICC_2km_ERA5_HVC_GWD.nc"
    print(f"Loading geo file: {geo_filename}")
    geo_file = xr.open_dataset(geo_filename)

    with nc.Dataset(geo_filename) as fileref:
        ds = getvar(fileref, "LU_INDEX")
        lats, lons = latlon_coords(ds)
        cart_proj = get_cartopy(ds)
        xbounds = cartopy_xlim(ds)
        ybounds = cartopy_ylim(ds)
        terrain = getvar(fileref, "ter")
        terrain = xr.where(terrain < 0, 0, terrain)

    def create_border_mask(data_array, border_width=5):
        """Create a mask that sets border pixels to NaN"""
        mask = np.ones_like(data_array, dtype=bool)
        mask[border_width:-border_width, border_width:-border_width] = False
        return mask

    fig = plt.figure(figsize=(15, 15))
    ax1 = fig.add_subplot(3, 2, 1, projection=cart_proj)
    ax1.set_title("Total Precipitation Change", fontsize=14, fontweight='bold')
    ax1.coastlines("10m", linewidth=0.8, zorder=102)
    mylevels_diff = np.arange(-200, 220, 20)
    cmap_diff = cmaps['BrBG']#cmocean.cm.curl_r  # , len(mylevels_diff)+1)
    norm_diff = BoundaryNorm(mylevels_diff, ncolors=cmap_diff.N, extend="both")

    m0 = ax1.pcolormesh(
        to_np(lons), to_np(lats),
        Dtotpr.squeeze()/10.,
        transform=crs.PlateCarree(),
        cmap=cmap_diff,
        norm=norm_diff,
        shading="auto",
        rasterized=True
    )

    # Add gridlines
    gl1 = ax1.gridlines(color="black", linestyle="dotted", draw_labels=True, 
                        x_inline=False, y_inline=False, zorder=102)
    gl1.top_labels = False
    gl1.right_labels = False
    gl1.xformatter = LONGITUDE_FORMATTER
    gl1.yformatter = LATITUDE_FORMATTER
    gl1.xpadding = 10

    # Set bounds
    ax1.set_xlim(xbounds)
    ax1.set_ylim(ybounds)

    #####################################################################
    #####################################################################
    
    ax6 = fig.add_subplot(3, 2, 2, projection=cart_proj)
    ax6.set_title("Num. Hours", fontsize=14, fontweight='bold')
    ax6.coastlines("10m", linewidth=0.8, zorder=102)
    mylevels_hours = np.arange(-200, 220, 20)
    cmap_hours = cmaps['BrBG']#cmocean.cm.curl_r  # , len(mylevels_diff)+1)
    norm_hours = BoundaryNorm(mylevels_hours, ncolors=cmap_hours.N, extend="both")

    m0 = ax6.pcolormesh(
        to_np(lons), to_np(lats),
        Dhours/10.,
        transform=crs.PlateCarree(),
        cmap=cmap_diff,
        norm=norm_diff,
        shading="auto",
        rasterized=True
    )

    # Add gridlines
    gl1 = ax6.gridlines(color="black", linestyle="dotted", draw_labels=True, 
                        x_inline=False, y_inline=False, zorder=102)
    gl1.top_labels = False
    gl1.right_labels = False
    gl1.xformatter = LONGITUDE_FORMATTER
    gl1.yformatter = LATITUDE_FORMATTER
    gl1.xpadding = 10

    # Set bounds
    ax6.set_xlim(xbounds)
    ax6.set_ylim(ybounds)

    #####################################################################
    #####################################################################

    ax2 = fig.add_subplot(3, 2, 3, projection=cart_proj)
    ax2.set_title("Duration contribution", fontsize=14, fontweight='bold')
    ax2.coastlines("10m", linewidth=0.8, zorder=102)

    m0 = ax2.pcolormesh(
        to_np(lons), to_np(lats),
        DPD/10.,
        transform=crs.PlateCarree(),
        cmap=cmap_diff,
        norm=norm_diff,
        shading="auto",
        rasterized=True
    )


    # Add gridlines
    gl1 = ax2.gridlines(color="black", linestyle="dotted", draw_labels=True, 
                        x_inline=False, y_inline=False, zorder=102)
    gl1.top_labels = False
    gl1.right_labels = False
    gl1.xformatter = LONGITUDE_FORMATTER
    gl1.yformatter = LATITUDE_FORMATTER
    gl1.xpadding = 10

    # Set bounds
    ax2.set_xlim(xbounds)
    ax2.set_ylim(ybounds)


    #####################################################################
    #####################################################################
    ax3 = fig.add_subplot(3, 2, 4, projection=cart_proj)
    ax3.set_title("Intensity contribution", fontsize=14, fontweight='bold')
    ax3.coastlines("10m", linewidth=0.8, zorder=102)


    m0 = ax3.pcolormesh(
        to_np(lons), to_np(lats),
        DPI/10.,
        transform=crs.PlateCarree(),
        cmap=cmap_diff,
        norm=norm_diff,
        shading="auto",
        rasterized=True
    )


    # Add gridlines
    gl1 = ax3.gridlines(color="black", linestyle="dotted", draw_labels=True, 
                        x_inline=False, y_inline=False, zorder=102)
    gl1.top_labels = False
    gl1.right_labels = False
    gl1.xformatter = LONGITUDE_FORMATTER
    gl1.yformatter = LATITUDE_FORMATTER
    gl1.xpadding = 10

    # Set bounds
    ax3.set_xlim(xbounds)
    ax3.set_ylim(ybounds)

    #####################################################################
    #####################################################################

    ax4 = fig.add_subplot(3, 2, 5, projection=cart_proj)
    ax4.set_title("Num. Events contribution", fontsize=14, fontweight='bold')
    ax4.coastlines("10m", linewidth=0.8, zorder=102)


    m0 = ax4.pcolormesh(
        to_np(lons), to_np(lats),
        DPN/10.,
        transform=crs.PlateCarree(),
        cmap=cmap_diff,
        norm=norm_diff,
        shading="auto",
        rasterized=True
    )


    # Add gridlines
    gl1 = ax4.gridlines(color="black", linestyle="dotted", draw_labels=True, 
                        x_inline=False, y_inline=False, zorder=102)
    gl1.top_labels = False
    gl1.right_labels = False
    gl1.xformatter = LONGITUDE_FORMATTER
    gl1.yformatter = LATITUDE_FORMATTER
    gl1.xpadding = 10

    # Set bounds
    ax4.set_xlim(xbounds)
    ax4.set_ylim(ybounds)

    #####################################################################
    #####################################################################

    ax5 = fig.add_subplot(3, 2, 6, projection=cart_proj)
    ax5.set_title("Residual", fontsize=14, fontweight='bold')
    ax5.coastlines("10m", linewidth=0.8, zorder=102)



    m0 = ax5.pcolormesh(
        to_np(lons), to_np(lats),
        Res/10.,
        transform=crs.PlateCarree(),
        cmap=cmap_diff,
        norm=norm_diff,
        shading="auto",
        rasterized=True
    )


    # Add gridlines
    gl1 = ax5.gridlines(color="black", linestyle="dotted", draw_labels=True, 
                        x_inline=False, y_inline=False, zorder=102)
    gl1.top_labels = False
    gl1.right_labels = False
    gl1.xformatter = LONGITUDE_FORMATTER
    gl1.yformatter = LATITUDE_FORMATTER
    gl1.xpadding = 10

    # Set bounds
    ax5.set_xlim(xbounds)
    ax5.set_ylim(ybounds)
    # Add colorbar with more space and label



    cbar_ax = fig.add_axes([0.25, 0.05, 0.5, 0.03])
    cb1 = plt.colorbar(
        m0, ticks=np.arange(-200, 240, 40), cax=cbar_ax, orientation="horizontal", shrink=0.62
    )
    cb1.set_label('mm/year', fontsize=12)

    plt.tight_layout()
    fig.subplots_adjust(left=0.05, right=0.95, top=0.93, bottom=0.20, wspace=0.1)

    print(f"Saving figure to: {output_filename}")
    fig.savefig(output_filename, bbox_inches="tight", dpi=300)
    print("Visualization completed successfully!")