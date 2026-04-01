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

def add_subplot_axes(ax,rect):
    fig = plt.gcf()
    box = ax.get_position()
    width = box.width
    height = box.height
    inax_position  = ax.transAxes.transform(rect[0:2])
    transFigure = fig.transFigure.inverted()
    infig_position = transFigure.transform(inax_position)    
    x = infig_position[0]
    y = infig_position[1]
    width *= rect[2]
    height *= rect[3]  # <= Typo was here
    subax = fig.add_axes([x,y,width,height])
    x_labelsize = subax.get_xticklabels()[0].get_size()
    y_labelsize = subax.get_yticklabels()[0].get_size()
    x_labelsize *= rect[2]**0.5
    y_labelsize *= rect[3]**0.5
    subax.xaxis.set_tick_params(labelsize=x_labelsize)
    subax.yaxis.set_tick_params(labelsize=y_labelsize)
    return subax

mpl.rcParams["font.size"] = 14
mpl.rcParams["hatch.color"] = "purple"
mpl.rcParams["hatch.linewidth"] = 1
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

mylevels=np.asarray([0,2,4,6,8,10,15,20,25,30,35,40,50,60,70,80,90,100])
cmap = sns.color_palette("icefire", as_cmap=True)
norm = BoundaryNorm(mylevels, ncolors=cmap.N, extend="max")

cmap_diff = cmaps["BrBG"]
norm_diff = BoundaryNorm(np.arange(-60, 70, 10), ncolors=cmap_diff.N, extend="both")
# seasons = ['DJF','MAM','JJA','SON']
seasons = ['ALL']
mode = 'wetonly'

mpl.rcParams["font.size"] = 14
mpl.rcParams["hatch.color"] = "purple"
mpl.rcParams["hatch.linewidth"] = 0.8


#####################################################################
#####################################################################

for season in seasons: 

    if season == 'ALL':
        suffix_season = ''
    else:
        suffix_season = f'_{season}'


    filein_d = f'/home/dargueso/postprocessed/EPICC//EPICC_2km_ERA5/percentiles_and_significance_DAY_mann_whitney_seqio.nc'
    fin_d = xr.open_dataset(filein_d).sel(percentile=qtile).squeeze()

    filein_h = f'/home/dargueso/postprocessed/EPICC//EPICC_2km_ERA5/percentiles_and_significance_1H_mann_whitney_seqio.nc'
    fin_h = xr.open_dataset(filein_h).sel(percentile=qtile).squeeze()

    filein_m = f'/home/dargueso/postprocessed/EPICC//EPICC_2km_ERA5/percentiles_and_significance_10MIN_mann_whitney_seqio.nc'
    fin_m = xr.open_dataset(filein_m).sel(percentile=qtile).squeeze()



    data_p={'daily':fin_d['percentiles_present'].values,'hourly':fin_h['percentiles_present'].values,'10min':fin_m['percentiles_present'].values}
    data_f={'daily':fin_d['percentiles_future'].values,'hourly':fin_h['percentiles_future'].values,'10min':fin_m['percentiles_future'].values}
    data_diff={'daily':(fin_d['percentiles_future'].values-fin_d['percentiles_present'].values)*100/fin_d['percentiles_present'].values,
            'hourly':(fin_h['percentiles_future'].values-fin_h['percentiles_present'].values)*100/fin_h['percentiles_present'].values,
            '10min':(fin_m['percentiles_future'].values-fin_m['percentiles_present'].values)*100/fin_m['percentiles_present'].values}

    data_sig = {'daily':fin_d['pvalue'].values,
                'hourly':fin_h['pvalue'].values,
                '10min':fin_m['pvalue'].values}
    
                          

    #####################################################################
    #####################################################################

    fig = plt.figure(figsize=(20, 15), constrained_layout=False)
    # Create main GridSpec with separate spacing control
    gs_main = GridSpec(3, 3, figure=fig,
                    left=0.08, bottom=0.12, right=0.95, top=0.93,
                    hspace=0.05, wspace=0.05)

    #####################################################################
    # TRANSPOSED: Now columns represent daily/hourly/10min
    # Row 0: Present data, Row 1: Future data, Row 2: Difference
    #####################################################################

    # Row 0: Present data (daily, hourly, 10min in columns)
    for row, datatype in enumerate(['daily', 'hourly', '10min']):
        axs = fig.add_subplot(gs_main[row, 0], projection=cart_proj)
        
        # Title based on position

        axs.set_title(f"{string.ascii_lowercase[row * 3 + 0]}", 
                    size='x-large', weight='bold', loc="left")
        axs.add_feature(cfeature.COASTLINE, linewidth=0.5, zorder=102)

        dtrain = axs.pcolormesh(
                    lons,
                    lats,
                    data_p[datatype],
                    cmap=cmap,
                    norm=norm,
                    transform=ccrs.PlateCarree(),
                    zorder=101
                )

        axs.contourf(to_np(lons), to_np(lats), border_mask,
                        levels=[0.5, 1.5],
                        colors=['gray'],
                        alpha=0.5,
                        transform=ccrs.PlateCarree(),
                        zorder=103)

        axs.set_xlim(cartopy_xlim(hgt, geobounds=mbounds))
        axs.set_ylim(cartopy_ylim(hgt, geobounds=mbounds))
        gl = axs.gridlines(color="black", linestyle="dotted", linewidth=0.5, 
                        draw_labels=True, x_inline=False, y_inline=False, zorder=103)
        gl.right_labels = False
        gl.top_labels = False
        if row != 2:
            gl.bottom_labels = False

    # Row 1: Future data (daily, hourly, 10min in columns)
    for row, datatype in enumerate(['daily', 'hourly', '10min']):
        axs = fig.add_subplot(gs_main[row,1], projection=cart_proj)
        axs.set_title(f"{string.ascii_lowercase[row * 3 + 1]} ", 
                    size='x-large', weight='bold', loc="left")
        axs.add_feature(cfeature.COASTLINE, linewidth=0.5, zorder=102)

        dtrain = axs.pcolormesh(
                    lons,
                    lats,
                    data_f[datatype],
                    cmap=cmap,
                    norm=norm,
                    transform=ccrs.PlateCarree(),
                    zorder=101
                )

        axs.contourf(to_np(lons), to_np(lats), border_mask,
                        levels=[0.5, 1.5],
                        colors=['gray'],
                        alpha=0.5,
                        transform=ccrs.PlateCarree(),
                        zorder=103)

        axs.set_xlim(cartopy_xlim(hgt, geobounds=mbounds))
        axs.set_ylim(cartopy_ylim(hgt, geobounds=mbounds))
        gl = axs.gridlines(color="black", linestyle="dotted", linewidth=0.5, 
                        draw_labels=True, x_inline=False, y_inline=False, zorder=103)
        gl.right_labels = False
        gl.top_labels = False
        gl.left_labels = False
        if row != 2:
            gl.bottom_labels = False

    
        

        
    for row, datatype in enumerate(['daily', 'hourly', '10min']):
 
        sig_var = data_sig[datatype]
        sig_levs = np.array([-0.5, 0.5, 1.5])
        sig = sig_var<0.05

        axs = fig.add_subplot(gs_main[row, 2], projection=cart_proj)
        axs.set_title(f"{string.ascii_lowercase[row * 3 + 2]}", 
                    size='x-large', weight='bold', loc="left")
        axs.add_feature(cfeature.COASTLINE, linewidth=0.5, zorder=102)
        
        dtrain_diff = axs.pcolormesh(
                    lons,
                    lats,
                    data_diff[datatype],
                    cmap=cmap_diff,
                    norm=norm_diff,
                    transform=ccrs.PlateCarree(),
                    zorder=101
                )
        
        cs = axs.contourf(to_np(lons),
                        to_np(lats),
                        sig_var, 
                        sig_levs, colors='none',
                        hatches=["","//////"],
                        transform=ccrs.PlateCarree(),
                        zorder=105)        

        axs.contourf(to_np(lons), to_np(lats), border_mask,
                        levels=[0.5, 1.5],
                        colors=['gray'],
                        alpha=0.5,
                        transform=ccrs.PlateCarree(),
                        zorder=103)

        axs.set_xlim(cartopy_xlim(hgt, geobounds=mbounds))
        axs.set_ylim(cartopy_ylim(hgt, geobounds=mbounds))
        gl = axs.gridlines(color="black", linestyle="dotted", linewidth=0.5, 
                        draw_labels=True, x_inline=False, y_inline=False, zorder=103)
        gl.right_labels = False
        gl.top_labels = False
        gl.left_labels = False
        if row != 2:
            gl.bottom_labels = False    

    cbar_ax = fig.add_axes([0.10, 0.07, 0.50, 0.025])
    cbar_exp = plt.colorbar(
        dtrain, cax=cbar_ax, orientation="horizontal", shrink=0.9
    )
    cbar_exp.set_label("Precipitation (mm)")

    cbar_axd = fig.add_axes([0.70, 0.07, 0.20, 0.025])
    cbar_del = plt.colorbar(
        dtrain_diff, cax=cbar_axd, orientation="horizontal", shrink=0.9
    )
    cbar_del.set_label("Precipitation change (%)")

    #####################################################################
    # Add row and column labels
    #####################################################################
    
    # Column labels (top, horizontal)
    # Calculate column positions based on GridSpec (left=0.08, right=0.95, 3 columns)
    col_width = (0.95 - 0.08) / 3
    col_positions = [0.08 + col_width * 0.5, 
                     0.08 + col_width * 1.5, 
                     0.08 + col_width * 2.5]
    col_labels = ['Present', 'Future', 'Change']
    
    for col_pos, col_label in zip(col_positions, col_labels):
        fig.text(col_pos, 0.93, col_label, 
                ha='center', va='bottom', 
                fontsize=18, 
                transform=fig.transFigure)
    
    # Row labels (left side, rotated 90 degrees)
    # Calculate row positions based on GridSpec (bottom=0.12, top=0.93, 3 rows)
    row_height = (0.93 - 0.12) / 3
    row_positions = [0.12 + row_height * 2.5,  # Row 0 (Daily) - top
                     0.12 + row_height * 1.5,  # Row 1 (Hourly) - middle
                     0.12 + row_height * 0.5]  # Row 2 (10-min) - bottom
    row_labels = ['Daily', 'Hourly', '10-min']
    
    for row_pos, row_label in zip(row_positions, row_labels):
        fig.text(0.03, row_pos, row_label, 
                ha='center', va='center', 
                fontsize=18, 
                rotation=90,
                transform=fig.transFigure)

    # Add shared colorbar for columns 1 and 2 (present and future - rows 0 and 1)
    # cbar1 = fig.colorbar(dtrain, ax=axes_col12, location='bottom', 
    #                      orientation="horizontal", shrink=0.6, pad=0.05, 
    #                      aspect=30, label="Precipitation (mm)")

    # # Add separate colorbar for column 3 (difference - row 2)
    # cbar2 = fig.colorbar(dtrain_diff, ax=axes_col3, location='bottom', 
    #                      orientation="horizontal", shrink=0.6, pad=0.05, 
    #                      aspect=30, label="Precipitation change (%)")


    plt.savefig(f'/home/dargueso/Analyses/EPICC/Hourly_vs_Daily/map_quantiles_daily_hourly_10min_precipitation_q{qtile}th{suffix_season}.png', 
                    dpi=300, bbox_inches='tight', facecolor='white')

    #####################################################################
    # Create violin plot of changes across the domain
    #####################################################################
    
    # Prepare data for violin plot - flatten the 2D arrays and remove NaN values
    violin_data = []
    labels = []
    
    for datatype in ['daily', 'hourly', '10min']:
        # Flatten the 2D array and remove NaN/inf values
        values = data_diff[datatype]
        #values[med_mask.combined_mask.values!=3]=np.nan  # Apply mask to consider only valid land points
        values = values[border_width:-border_width, border_width:-border_width].flatten()
        values = values[np.isfinite(values)]
        violin_data.append(values)
        labels.append(datatype.capitalize())
    
    # Create violin plot
    fig_violin = plt.figure(figsize=(10, 8))
    ax_violin = fig_violin.add_subplot(111)
    
    # Create violin plot without default min/max lines
    parts = ax_violin.violinplot(violin_data, positions=[1, 2, 3], 
                                   showmeans=False, showmedians=True, 
                                   showextrema=False,  # Don't show min/max
                                   widths=0.7)
    
    # Customize violin colors
    colors = ['#ED6A5A', '#F4F1BB', '#9BC1BC']
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.7)
        pc.set_edgecolor('black')
        pc.set_linewidth(1) 
    
    # Customize mean and median lines
    for partname in ['cmedians']:#, 'cmeans']:
        if partname in parts:
            vp = parts[partname]
            vp.set_edgecolor('black')
            vp.set_linewidth(1)
    
    # Add custom percentile lines (e.g., 5th and 95th percentiles)
    percentile_low = 5  # Change this value (e.g., 5, 10, 25)
    percentile_high = 95  # Change this value (e.g., 75, 90, 95)
    
    for i, values in enumerate(violin_data):
        pos = i + 1
        p_low = np.percentile(values, percentile_low)
        p_high = np.percentile(values, percentile_high)
        
        # Draw vertical line from low to high percentile
        ax_violin.plot([pos, pos], [p_low, p_high], 
                      color='black', linewidth=1, zorder=3)
        
        # Draw horizontal caps at percentiles
        cap_width = 0.15
        ax_violin.plot([pos - cap_width, pos + cap_width], [p_low, p_low], 
                      color='black', linewidth=1, zorder=3)
        ax_violin.plot([pos - cap_width, pos + cap_width], [p_high, p_high], 
                      color='black', linewidth=1, zorder=3)
    
    # Add horizontal line at 0
    ax_violin.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    
    # Customize plot
    ax_violin.set_xticks([1, 2, 3])
    ax_violin.set_xticklabels(labels, fontsize=14)
    ax_violin.set_ylabel('Precipitation change (%)', fontsize=14)
    # ax_violin.set_title(f'Distribution of precipitation changes (Q{int(qtile*100)}th percentile)\nWhiskers show {percentile_low}th-{percentile_high}th percentile range', 
    #                     fontsize=16, weight='bold')
    ax_violin.grid(axis='y', alpha=0.3, linestyle=':')
    ax_violin.set_xlim(0.5, 3.5)
    
    # Add statistics text
    for i, (datatype, values) in enumerate(zip(labels, violin_data)):
        median_val = np.median(values)
        mean_val = np.mean(values)
        p_low = np.percentile(values, percentile_low)
        p_high = np.percentile(values, percentile_high)
        ax_violin.text(i+1, ax_violin.get_ylim()[1]*0.95, 
                      f'Median: {median_val:.1f}%\nP{percentile_low}: {p_low:.1f}%\nP{percentile_high}: {p_high:.1f}%',
                      ha='center', va='top', fontsize=9, 
                      bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(f'/home/dargueso/Analyses/EPICC/Hourly_vs_Daily/violin_plot_precipitation_changes_q{qtile}th_{mode}{suffix_season}.png',
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig_violin)

    #####################################################################
    # Combined figure: maps + violin next to each change panel
    #####################################################################

    violin_colors   = {'daily': '#ED6A5A', 'hourly': '#F4F1BB', '10min': '#9BC1BC'}
    percentile_low  = 5
    percentile_high = 95

    fig3 = plt.figure(figsize=(28, 15), constrained_layout=False)
    gs3 = GridSpec(3, 4, figure=fig3,
                   width_ratios=[1, 1, 1, 0.40],
                   left=0.06, bottom=0.11, right=0.95, top=0.93,
                   hspace=0.06, wspace=0.06)

    ax_violins = []   # collect for shared y-axis

    for row, datatype in enumerate(['daily', 'hourly', '10min']):

        values_viol = data_diff[datatype]
        values_viol = values_viol[border_width:-border_width, border_width:-border_width].flatten()
        values_viol = values_viol[np.isfinite(values_viol)]

        median_val = np.median(values_viol)
        p_lo       = np.percentile(values_viol, percentile_low)
        p_hi       = np.percentile(values_viol, percentile_high)

        # --- Column 0: Present ---
        ax0 = fig3.add_subplot(gs3[row, 0], projection=cart_proj)
        ax0.set_title(f"{string.ascii_lowercase[row * 3 + 0]}",
                      size='x-large', weight='bold', loc='left')
        ax0.add_feature(cfeature.COASTLINE, linewidth=0.5, zorder=102)
        pm0 = ax0.pcolormesh(lons, lats, data_p[datatype],
                             cmap=cmap, norm=norm,
                             transform=ccrs.PlateCarree(), zorder=101)
        ax0.contourf(to_np(lons), to_np(lats), border_mask,
                     levels=[0.5, 1.5], colors=['gray'], alpha=0.5,
                     transform=ccrs.PlateCarree(), zorder=103)
        ax0.set_xlim(cartopy_xlim(hgt, geobounds=mbounds))
        ax0.set_ylim(cartopy_ylim(hgt, geobounds=mbounds))
        gl0 = ax0.gridlines(color='black', linestyle='dotted', linewidth=0.5,
                            draw_labels=True, x_inline=False, y_inline=False, zorder=103)
        gl0.right_labels  = False
        gl0.top_labels    = False
        if row != 2:
            gl0.bottom_labels = False

        # --- Column 1: Future ---
        ax1 = fig3.add_subplot(gs3[row, 1], projection=cart_proj)
        ax1.set_title(f"{string.ascii_lowercase[row * 3 + 1]}",
                      size='x-large', weight='bold', loc='left')
        ax1.add_feature(cfeature.COASTLINE, linewidth=0.5, zorder=102)
        pm1 = ax1.pcolormesh(lons, lats, data_f[datatype],
                             cmap=cmap, norm=norm,
                             transform=ccrs.PlateCarree(), zorder=101)
        ax1.contourf(to_np(lons), to_np(lats), border_mask,
                     levels=[0.5, 1.5], colors=['gray'], alpha=0.5,
                     transform=ccrs.PlateCarree(), zorder=103)
        ax1.set_xlim(cartopy_xlim(hgt, geobounds=mbounds))
        ax1.set_ylim(cartopy_ylim(hgt, geobounds=mbounds))
        gl1 = ax1.gridlines(color='black', linestyle='dotted', linewidth=0.5,
                            draw_labels=True, x_inline=False, y_inline=False, zorder=103)
        gl1.right_labels  = False
        gl1.top_labels    = False
        gl1.left_labels   = False
        if row != 2:
            gl1.bottom_labels = False

        # --- Column 2: Change map ---
        ax2 = fig3.add_subplot(gs3[row, 2], projection=cart_proj)
        ax2.set_title(f"{string.ascii_lowercase[row * 3 + 2]}",
                      size='x-large', weight='bold', loc='left')
        ax2.add_feature(cfeature.COASTLINE, linewidth=0.5, zorder=102)
        pm2 = ax2.pcolormesh(lons, lats, data_diff[datatype],
                             cmap=cmap_diff, norm=norm_diff,
                             transform=ccrs.PlateCarree(), zorder=101)
        ax2.contourf(to_np(lons), to_np(lats), data_sig[datatype],
                     np.array([-0.5, 0.5, 1.5]), colors='none',
                     hatches=['', '//////'],
                     transform=ccrs.PlateCarree(), zorder=105)
        ax2.contourf(to_np(lons), to_np(lats), border_mask,
                     levels=[0.5, 1.5], colors=['gray'], alpha=0.5,
                     transform=ccrs.PlateCarree(), zorder=103)
        ax2.set_xlim(cartopy_xlim(hgt, geobounds=mbounds))
        ax2.set_ylim(cartopy_ylim(hgt, geobounds=mbounds))
        gl2 = ax2.gridlines(color='black', linestyle='dotted', linewidth=0.5,
                            draw_labels=True, x_inline=False, y_inline=False, zorder=103)
        gl2.right_labels  = False
        gl2.top_labels    = False
        gl2.left_labels   = False
        if row != 2:
            gl2.bottom_labels = False

        # --- Column 3: Violin ---
        ax3 = fig3.add_subplot(gs3[row, 3])

        parts = ax3.violinplot([values_viol], positions=[0], showmeans=False,
                               showmedians=True, showextrema=False, widths=0.45)
        parts['bodies'][0].set_facecolor(violin_colors[datatype])
        parts['bodies'][0].set_edgecolor('black')
        parts['bodies'][0].set_alpha(0.7)
        parts['bodies'][0].set_linewidth(1)
        parts['cmedians'].set_edgecolor('black')
        parts['cmedians'].set_linewidth(1.5)

        cap_w = 0.12
        ax3.plot([0, 0], [p_lo, p_hi], color='black', linewidth=1, zorder=3)
        ax3.plot([-cap_w, cap_w], [p_lo, p_lo], color='black', linewidth=1, zorder=3)
        ax3.plot([-cap_w, cap_w], [p_hi, p_hi], color='black', linewidth=1, zorder=3)

        ax3.axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.7)

        # Stat labels next to each horizontal line
        tx = cap_w + 0.05
        ax3.text(tx, p_hi,      f'P{percentile_high}: {p_hi:.1f}%',
                 ha='left', va='center', fontsize=7.5)
        ax3.text(tx, median_val, f'Md: {median_val:.1f}%',
                 ha='left', va='center', fontsize=7.5)
        ax3.text(tx, p_lo,      f'P{percentile_low}: {p_lo:.1f}%',
                 ha='left', va='center', fontsize=7.5)

        ax3.set_xticks([])
        ax3.yaxis.tick_right()
        ax3.grid(axis='y', alpha=0.3, linestyle=':')
        ax3.set_xlim(-0.40, 1.05)

        ax_violins.append(ax3)

    # Shared y-limits across all violin panels
    vmin = min(ax.get_ylim()[0] for ax in ax_violins)
    vmax = max(ax.get_ylim()[1] for ax in ax_violins)
    for ax in ax_violins:
        ax.set_ylim(vmin, vmax)

    # Shared y-label for violins (right edge of figure)
    fig3.text(0.975, (0.11 + 0.93) / 2, 'Change (%)',
              rotation=90, ha='center', va='center', fontsize=11,
              transform=fig3.transFigure)

    # Column labels — account for unequal width_ratios=[1,1,1,0.40]
    _uw = (0.95 - 0.06) / 3.40   # figure-width per ratio unit (ignoring wspace)
    col_positions3 = [
        0.06 + _uw * 0.50,
        0.06 + _uw * 1.50,
        0.06 + _uw * 2.50,
        0.06 + _uw * (3.0 + 0.20),
    ]
    for col_pos, col_label in zip(col_positions3,
                                  ['Present', 'Future', 'Change', 'Distribution']):
        fig3.text(col_pos, 0.945, col_label,
                  ha='center', va='bottom', fontsize=14,
                  transform=fig3.transFigure)

    row_height3 = (0.93 - 0.11) / 3
    row_positions3 = [0.11 + row_height3 * (2.5 - i) for i in range(3)]
    for row_pos, row_label in zip(row_positions3, ['Daily', 'Hourly', '10-min']):
        fig3.text(0.02, row_pos, row_label,
                  ha='center', va='center', fontsize=14, rotation=90,
                  transform=fig3.transFigure)

    # Colorbars
    cbar_ax3 = fig3.add_axes([0.06, 0.055, 0.48, 0.020])
    plt.colorbar(pm0, cax=cbar_ax3, orientation='horizontal').set_label('Precipitation (mm)')

    cbar_ax3d = fig3.add_axes([0.60, 0.055, 0.20, 0.020])
    plt.colorbar(pm2, cax=cbar_ax3d, orientation='horizontal').set_label('Precipitation change (%)')

    plt.savefig(f'/home/dargueso/Analyses/EPICC/Hourly_vs_Daily/combined_map_violin_precipitation_changes_q{qtile}th{suffix_season}.png',
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig3)