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
import cmocean

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

qtile = 95

# --- Colormap selection: 'custom' | 'rain' | 'haline' (cmocean) or 'icefire' (seaborn) ---
PRECIP_CMAP = 'custom'

# Custom precipitation colormap (12 colours: white = under, 11 intervals)
_CUSTOM_RGB = np.array([
    [255, 255, 255],   # under  (no rain / below first level)
    [237, 250, 194],
    [205, 255, 205],
    [153, 240, 178],
    [ 83, 189, 159],
    [ 50, 166, 150],
    [ 50, 150, 180],
    [  5, 112, 176],
    [  5,  80, 140],
    [ 10,  31, 150],
    [ 44,   2,  70],
    [106,  44,  90],
], dtype=float) / 255.0

# --- Unit conversion: set to True to express intensities in mm hr-1
#     daily  -> divide by 24   (mm day-1 -> mm hr-1)
#     hourly -> unchanged      (already mm hr-1)
#     10 min -> multiply by 6  (mm (10 min)-1 -> mm hr-1)
CONVERT_TO_MM_HR = False

if PRECIP_CMAP == 'custom':
    # 12 colours: colour[0] (white) → 0–2 mm interval,
    #             colour[11] (dark maroon) → >100 mm extension (required by extend="max")
    cmap = matplotlib.colors.ListedColormap(_CUSTOM_RGB)        # 12 colours
    cmap.set_under(_CUSTOM_RGB[0])                              # white for < first level
    mylevels = np.array([0, 2, 4, 6, 10, 15, 20, 30, 40, 60, 80, 100])  # 12 bounds → 11 intervals
elif PRECIP_CMAP == 'rain':
    cmap = cmocean.cm.rain
    mylevels = np.asarray([0, 2, 4, 6, 8, 10, 15, 20, 25, 30, 35, 40, 50, 60, 70, 80, 90, 100])
elif PRECIP_CMAP == 'haline':
    cmap = cmocean.cm.haline
    mylevels = np.asarray([0, 2, 4, 6, 8, 10, 15, 20, 25, 30, 35, 40, 50, 60, 70, 80, 90, 100])
else:
    cmap = sns.color_palette("icefire", as_cmap=True)
    mylevels = np.asarray([0, 2, 4, 6, 8, 10, 15, 20, 25, 30, 35, 40, 50, 60, 70, 80, 90, 100])
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



    # Unit conversion factors (mm hr-1 mode)
    if CONVERT_TO_MM_HR:
        unit_factors = {'daily': 1.0/24.0, 'hourly': 1.0, '10min': 6.0}
        unit_label   = 'mm hr$^{-1}$'
        unit_suffix  = '_mmhr'
    else:
        unit_factors = {'daily': 1.0, 'hourly': 1.0, '10min': 1.0}
        unit_label   = 'mm'
        unit_suffix  = ''

    data_p={'daily':fin_d['percentiles_present'].values * unit_factors['daily'],
            'hourly':fin_h['percentiles_present'].values * unit_factors['hourly'],
            '10min':fin_m['percentiles_present'].values * unit_factors['10min']}
    data_f={'daily':fin_d['percentiles_future'].values * unit_factors['daily'],
            'hourly':fin_h['percentiles_future'].values * unit_factors['hourly'],
            '10min':fin_m['percentiles_future'].values * unit_factors['10min']}
    data_diff={'daily':(fin_d['percentiles_future'].values-fin_d['percentiles_present'].values)*100/fin_d['percentiles_present'].values,
            'hourly':(fin_h['percentiles_future'].values-fin_h['percentiles_present'].values)*100/fin_h['percentiles_present'].values,
            '10min':(fin_m['percentiles_future'].values-fin_m['percentiles_present'].values)*100/fin_m['percentiles_present'].values}

    data_sig = {'daily':fin_d['pvalue'].values,
                'hourly':fin_h['pvalue'].values,
                '10min':fin_m['pvalue'].values}

    # --- Masks for statistics and violin plots ---
    interior      = border_mask == 0                                        # full interior domain
    coastal_mask  = (med_mask.combined_mask.values == 3) & interior         # coastal region only
    n_interior = int(interior.sum())
    n_coastal  = int(coastal_mask.sum())
    print(f"\nSignificant changes at Q{qtile} (p < 0.05){' — ' + season if season != 'ALL' else ''}:")
    print(f"  {'':6s}  {'Interior':>20s}  {'Coastal':>20s}")
    print(f"  {'':6s}  {'('+str(n_interior)+' pts)':>20s}  {'('+str(n_coastal)+' pts)':>20s}")
    for datatype, pvals in data_sig.items():
        diff  = data_diff[datatype]
        sig_int = (pvals < 0.05) & interior
        sig_cst = (pvals < 0.05) & coastal_mask
        n_sig_int  = int(sig_int.sum())
        n_sig_cst  = int(sig_cst.sum())
        pct_int    = 100.0 * n_sig_int / n_interior if n_interior > 0 else np.nan
        pct_cst    = 100.0 * n_sig_cst / n_coastal  if n_coastal  > 0 else np.nan
        # total positive/negative points in each region
        pos_int = interior & (diff > 0)
        neg_int = interior & (diff < 0)
        pos_cst = coastal_mask & (diff > 0)
        neg_cst = coastal_mask & (diff < 0)
        n_pos_all_int = int(pos_int.sum());  n_neg_all_int = int(neg_int.sum())
        n_pos_all_cst = int(pos_cst.sum());  n_neg_all_cst = int(neg_cst.sum())
        # significant among positive / negative
        n_sig_pos_int = int((sig_int & (diff > 0)).sum())
        n_sig_neg_int = int((sig_int & (diff < 0)).sum())
        n_sig_pos_cst = int((sig_cst & (diff > 0)).sum())
        n_sig_neg_cst = int((sig_cst & (diff < 0)).sum())
        pct_pos_int      = 100.0 * n_pos_all_int / n_interior    if n_interior    > 0 else np.nan
        pct_neg_int      = 100.0 * n_neg_all_int / n_interior    if n_interior    > 0 else np.nan
        pct_pos_cst      = 100.0 * n_pos_all_cst / n_coastal     if n_coastal     > 0 else np.nan
        pct_neg_cst      = 100.0 * n_neg_all_cst / n_coastal     if n_coastal     > 0 else np.nan
        pct_sig_pos_int  = 100.0 * n_sig_pos_int / n_pos_all_int if n_pos_all_int > 0 else np.nan
        pct_sig_neg_int  = 100.0 * n_sig_neg_int / n_neg_all_int if n_neg_all_int > 0 else np.nan
        pct_sig_pos_cst  = 100.0 * n_sig_pos_cst / n_pos_all_cst if n_pos_all_cst > 0 else np.nan
        pct_sig_neg_cst  = 100.0 * n_sig_neg_cst / n_neg_all_cst if n_neg_all_cst > 0 else np.nan
        # of significant points, what fraction is positive / negative
        pct_ofsig_pos_int = 100.0 * n_sig_pos_int / n_sig_int if n_sig_int > 0 else np.nan
        pct_ofsig_neg_int = 100.0 * n_sig_neg_int / n_sig_int if n_sig_int > 0 else np.nan
        pct_ofsig_pos_cst = 100.0 * n_sig_pos_cst / n_sig_cst if n_sig_cst > 0 else np.nan
        pct_ofsig_neg_cst = 100.0 * n_sig_neg_cst / n_sig_cst if n_sig_cst > 0 else np.nan
        print(f"  {datatype:>6s}:  sig {n_sig_int:>6d}/{n_interior:>6d} ({pct_int:5.1f}%)"
              f"  [of sig: pos {pct_ofsig_pos_int:5.1f}% / neg {pct_ofsig_neg_int:5.1f}%]"
              f"  | pos {n_pos_all_int:>6d}/{n_interior:>6d} ({pct_pos_int:5.1f}%) of which sig ({pct_sig_pos_int:5.1f}%)"
              f"  | neg {n_neg_all_int:>6d}/{n_interior:>6d} ({pct_neg_int:5.1f}%) of which sig ({pct_sig_neg_int:5.1f}%)")
        print(f"  {'':6s}   {'Coastal':}"
              f"  sig {n_sig_cst:>6d}/{n_coastal:>6d} ({pct_cst:5.1f}%)"
              f"  [of sig: pos {pct_ofsig_pos_cst:5.1f}% / neg {pct_ofsig_neg_cst:5.1f}%]"
              f"  | pos {n_pos_all_cst:>6d}/{n_coastal:>6d} ({pct_pos_cst:5.1f}%) of which sig ({pct_sig_pos_cst:5.1f}%)"
              f"  | neg {n_neg_all_cst:>6d}/{n_coastal:>6d} ({pct_neg_cst:5.1f}%) of which sig ({pct_sig_neg_cst:5.1f}%)")
    print()

    #####################################################################
    #####################################################################

    fig = plt.figure(figsize=(20, 15), constrained_layout=False)
    # Create main GridSpec with separate spacing control
    gs_main = GridSpec(3, 3, figure=fig,
                    left=0.08, bottom=0.15, right=0.95, top=0.93,
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
 
        sig_var = np.where(np.isfinite(data_sig[datatype]), data_sig[datatype], 1.0)  # NaN → 1.0 (non-significant)
        sig_levs = np.array([-0.5, 0.05, 1.05])   # hatch where p >= 0.05 (not significant)
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
    cbar_exp.set_label(f"Precipitation ({unit_label})")

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
    # Calculate row positions based on GridSpec (bottom=0.15, top=0.93, 3 rows)
    row_height = (0.93 - 0.15) / 3
    row_positions = [0.15 + row_height * 2.5,  # Row 0 (Daily) - top
                     0.15 + row_height * 1.5,  # Row 1 (Hourly) - middle
                     0.15 + row_height * 0.5]  # Row 2 (10-min) - bottom
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


    plt.savefig(f'/home/dargueso/Analyses/EPICC/Hourly_vs_Daily/map_quantiles_daily_hourly_10min_precipitation_q{qtile}th{suffix_season}{unit_suffix}.png',
                    dpi=300, bbox_inches='tight', facecolor='white')

    #####################################################################
    # Create violin plot of changes across the domain
    #####################################################################

    percentile_low  = 5
    percentile_high = 95
    _dtypes  = ['daily', 'hourly', '10min']
    _colors  = {'daily': '#ED6A5A', 'hourly': '#F4F1BB', '10min': '#9BC1BC'}
    _offset  = 0.27   # half-distance between domain and coastal violin

    fig_violin = plt.figure(figsize=(12, 8))
    ax_violin  = fig_violin.add_subplot(111)

    group_centers = [1, 2, 3]
    for i, datatype in enumerate(_dtypes):
        pos_dom  = group_centers[i] - _offset
        pos_cst  = group_centers[i] + _offset
        col      = _colors[datatype]

        pvals_arr = data_sig[datatype]
        vals_dom = data_diff[datatype][interior]
        vals_dom = vals_dom[np.isfinite(vals_dom)]
        vals_dom_sig = data_diff[datatype][(pvals_arr < 0.05) & interior]
        vals_dom_sig = vals_dom_sig[np.isfinite(vals_dom_sig)]

        vals_cst = data_diff[datatype][coastal_mask]
        vals_cst = vals_cst[np.isfinite(vals_cst)]
        vals_cst_sig = data_diff[datatype][(pvals_arr < 0.05) & coastal_mask]
        vals_cst_sig = vals_cst_sig[np.isfinite(vals_cst_sig)]

        cap_w = 0.08
        sig_col = 'crimson'
        for pos, vals, vals_sig, alpha, ls in [(pos_dom, vals_dom, vals_dom_sig, 0.5, '-'),
                                               (pos_cst, vals_cst, vals_cst_sig, 0.9, '-')]:
            parts = ax_violin.violinplot([vals], positions=[pos],
                                         showmeans=False, showmedians=True,
                                         showextrema=False, widths=0.30)
            parts['bodies'][0].set_facecolor(col)
            parts['bodies'][0].set_alpha(alpha)
            parts['bodies'][0].set_edgecolor('black')
            parts['bodies'][0].set_linewidth(1)
            parts['cmedians'].set_edgecolor('black')
            parts['cmedians'].set_linewidth(1.5)

            p_lo = np.percentile(vals, percentile_low)
            p_hi = np.percentile(vals, percentile_high)
            ax_violin.plot([pos, pos], [p_lo, p_hi], color='black', linewidth=1, zorder=3)
            ax_violin.plot([pos - cap_w, pos + cap_w], [p_lo, p_lo], color='black', linewidth=1, zorder=3)
            ax_violin.plot([pos - cap_w, pos + cap_w], [p_hi, p_hi], color='black', linewidth=1, zorder=3)

            # Significant-only whiskers and median
            if len(vals_sig) > 1:
                p_lo_sig = np.percentile(vals_sig, percentile_low)
                p_hi_sig = np.percentile(vals_sig, percentile_high)
                med_sig  = np.median(vals_sig)
                ax_violin.plot([pos, pos], [p_lo_sig, p_hi_sig], color=sig_col, linewidth=2, zorder=4)
                ax_violin.plot([pos - cap_w*0.7, pos + cap_w*0.7], [p_lo_sig, p_lo_sig], color=sig_col, linewidth=2, zorder=4)
                ax_violin.plot([pos - cap_w*0.7, pos + cap_w*0.7], [p_hi_sig, p_hi_sig], color=sig_col, linewidth=2, zorder=4)
                ax_violin.plot([pos - cap_w*0.5, pos + cap_w*0.5], [med_sig,  med_sig],  color=sig_col, linewidth=2.5, zorder=4)

            # Black labels (all-data) to the left; crimson labels (sig-only) to the right
            tx_blk = pos - cap_w - 0.03
            ax_violin.text(tx_blk, p_hi, f'{p_hi:.1f}%', ha='right', va='center', fontsize=7, color='black')
            ax_violin.text(tx_blk, np.median(vals), f'{np.median(vals):.1f}%', ha='right', va='center', fontsize=7, color='black')
            ax_violin.text(tx_blk, p_lo, f'{p_lo:.1f}%', ha='right', va='center', fontsize=7, color='black')
            if len(vals_sig) > 1:
                tx_crim = pos + cap_w*0.7 + 0.03
                ax_violin.text(tx_crim, p_hi_sig, f'{p_hi_sig:.1f}%', ha='left', va='center', fontsize=7, color=sig_col)
                ax_violin.text(tx_crim, med_sig,  f'{med_sig:.1f}%',  ha='left', va='center', fontsize=7, color=sig_col)
                ax_violin.text(tx_crim, p_lo_sig, f'{p_lo_sig:.1f}%', ha='left', va='center', fontsize=7, color=sig_col)

    ax_violin.axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    ax_violin.set_xticks(group_centers)
    ax_violin.set_xticklabels([d.capitalize() for d in _dtypes], fontsize=14)
    ax_violin.set_ylabel('Precipitation change (%)', fontsize=14)
    ax_violin.grid(axis='y', alpha=0.3, linestyle=':')
    ax_violin.set_xlim(0.35, 3.65)

    # Legend
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    ax_violin.legend(
        [Patch(facecolor='gray', alpha=0.5, edgecolor='black'),
         Patch(facecolor='gray', alpha=0.9, edgecolor='black'),
         Line2D([0], [0], color='crimson', linewidth=2)],
        ['Full domain', 'Coastal region', 'Significant only'],
        fontsize=11, loc='upper left')

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

    # 4 cols: 3 maps + 1 narrow violin; letters a–l across all 12 panels (row×4)
    fig3 = plt.figure(figsize=(24, 15), constrained_layout=False)
    gs3 = GridSpec(3, 4, figure=fig3,
                   width_ratios=[1, 1, 1, 0.42],
                   left=0.06, bottom=0.13, right=0.96, top=0.91,
                   hspace=0.10, wspace=0.08)

    ax_violins = []   # collect for shared y-axis
    _pl = string.ascii_lowercase  # panel letter shorthand

    for row, datatype in enumerate(['daily', 'hourly', '10min']):

        pvals_arr = data_sig[datatype]
        vals_dom = data_diff[datatype][interior]
        vals_dom = vals_dom[np.isfinite(vals_dom)]
        vals_dom_sig = data_diff[datatype][(pvals_arr < 0.05) & interior]
        vals_dom_sig = vals_dom_sig[np.isfinite(vals_dom_sig)]

        vals_cst = data_diff[datatype][coastal_mask]
        vals_cst = vals_cst[np.isfinite(vals_cst)]
        vals_cst_sig = data_diff[datatype][(pvals_arr < 0.05) & coastal_mask]
        vals_cst_sig = vals_cst_sig[np.isfinite(vals_cst_sig)]

        # --- Column 0: Present ---
        ax0 = fig3.add_subplot(gs3[row, 0], projection=cart_proj)
        ax0.text(0.0, 1.02, _pl[row * 4 + 0],
                 transform=ax0.transAxes, size='x-large', weight='bold',
                 ha='left', va='bottom')
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
        ax1.text(0.0, 1.02, _pl[row * 4 + 1],
                 transform=ax1.transAxes, size='x-large', weight='bold',
                 ha='left', va='bottom')
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
        ax2.text(0.0, 1.02, _pl[row * 4 + 2],
                 transform=ax2.transAxes, size='x-large', weight='bold',
                 ha='left', va='bottom')
        ax2.add_feature(cfeature.COASTLINE, linewidth=0.5, zorder=102)
        pm2 = ax2.pcolormesh(lons, lats, data_diff[datatype],
                             cmap=cmap_diff, norm=norm_diff,
                             transform=ccrs.PlateCarree(), zorder=101)
        ax2.contourf(to_np(lons), to_np(lats),
                     np.where(np.isfinite(data_sig[datatype]), data_sig[datatype], 1.0),
                     np.array([-0.5, 0.05, 1.05]), colors='none',   # hatch where p >= 0.05 (NaN → non-sig)
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
        ax3.text(0.0, 1.02, _pl[row * 4 + 3],
                 transform=ax3.transAxes, size='x-large', weight='bold',
                 ha='left', va='bottom')

        col = violin_colors[datatype]
        cap_w = 0.10
        pos_dom, pos_cst = -0.30, 0.30
        sig_col = 'crimson'

        for pos, vals, vals_sig, alpha in [(pos_dom, vals_dom, vals_dom_sig, 0.5),
                                           (pos_cst, vals_cst, vals_cst_sig, 0.9)]:
            parts = ax3.violinplot([vals], positions=[pos], showmeans=False,
                                   showmedians=True, showextrema=False, widths=0.30)
            parts['bodies'][0].set_facecolor(col)
            parts['bodies'][0].set_edgecolor('black')
            parts['bodies'][0].set_alpha(alpha)
            parts['bodies'][0].set_linewidth(1)
            parts['cmedians'].set_edgecolor('black')
            parts['cmedians'].set_linewidth(1.5)

            p_lo = np.percentile(vals, percentile_low)
            p_hi = np.percentile(vals, percentile_high)
            med  = np.median(vals)
            ax3.plot([pos, pos], [p_lo, p_hi], color='black', linewidth=1, zorder=3)
            ax3.plot([pos - cap_w, pos + cap_w], [p_lo, p_lo], color='black', linewidth=1, zorder=3)
            ax3.plot([pos - cap_w, pos + cap_w], [p_hi, p_hi], color='black', linewidth=1, zorder=3)

            # Significant-only whiskers and median
            if len(vals_sig) > 1:
                p_lo_sig = np.percentile(vals_sig, percentile_low)
                p_hi_sig = np.percentile(vals_sig, percentile_high)
                med_sig  = np.median(vals_sig)
                ax3.plot([pos, pos], [p_lo_sig, p_hi_sig], color=sig_col, linewidth=2, zorder=4)
                ax3.plot([pos - cap_w*0.7, pos + cap_w*0.7], [p_lo_sig, p_lo_sig], color=sig_col, linewidth=2, zorder=4)
                ax3.plot([pos - cap_w*0.7, pos + cap_w*0.7], [p_hi_sig, p_hi_sig], color=sig_col, linewidth=2, zorder=4)
                ax3.plot([pos - cap_w*0.5, pos + cap_w*0.5], [med_sig,  med_sig],  color=sig_col, linewidth=2.5, zorder=4)

            # Black labels (all-data) to the left; crimson labels (sig-only) to the right
            tx_blk  = pos - cap_w - 0.03
            tx_crim = pos + cap_w*0.7 + 0.03
            ax3.text(tx_blk, p_hi, f'{p_hi:.1f}%', ha='right', va='center', fontsize=7, color='black')
            ax3.text(tx_blk, med,  f'{med:.1f}%',  ha='right', va='center', fontsize=7, color='black')
            ax3.text(tx_blk, p_lo, f'{p_lo:.1f}%', ha='right', va='center', fontsize=7, color='black')
            if len(vals_sig) > 1:
                ax3.text(tx_crim, p_hi_sig, f'{p_hi_sig:.1f}%', ha='left', va='center', fontsize=7, color=sig_col)
                ax3.text(tx_crim, med_sig,  f'{med_sig:.1f}%',  ha='left', va='center', fontsize=7, color=sig_col)
                ax3.text(tx_crim, p_lo_sig, f'{p_lo_sig:.1f}%', ha='left', va='center', fontsize=7, color=sig_col)

        ax3.axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.7)

        # x-tick labels
        ax3.set_xticks([pos_dom, pos_cst])
        ax3.set_xticklabels(['Dom', 'Coast'], fontsize=8)
        ax3.yaxis.tick_right()
        ax3.tick_params(axis='y', labelsize=8)
        ax3.spines['left'].set_visible(False)
        ax3.spines['top'].set_visible(False)
        ax3.grid(axis='y', alpha=0.3, linestyle=':')
        ax3.set_xlim(-0.75, 0.75)

        ax_violins.append(ax3)

    # Shared y-limits across all violin panels
    vmin = min(ax.get_ylim()[0] for ax in ax_violins)
    vmax = max(ax.get_ylim()[1] for ax in ax_violins)
    for ax in ax_violins:
        ax.set_ylim(vmin, vmax)

    # Shared y-label for violins (right edge)
    fig3.text(0.988, (0.13 + 0.91) / 2, 'Change (%)',
              rotation=90, ha='center', va='center', fontsize=11,
              transform=fig3.transFigure)

    # Column labels for the 3 map columns only (no "Distribution" label)
    _uw = (0.96 - 0.06) / 3.22   # figure-width per ratio unit (ignoring wspace)
    col_positions3 = [
        0.06 + _uw * 0.50,
        0.06 + _uw * 1.50,
        0.06 + _uw * 2.50,
    ]
    for col_pos, col_label in zip(col_positions3, ['Present', 'Future', 'Change']):
        fig3.text(col_pos, 0.925, col_label,
                  ha='center', va='bottom', fontsize=14,
                  transform=fig3.transFigure)

    row_height3 = (0.91 - 0.13) / 3
    row_positions3 = [0.13 + row_height3 * (2.5 - i) for i in range(3)]
    for row_pos, row_label in zip(row_positions3, ['Daily', 'Hourly', '10-min']):
        fig3.text(0.02, row_pos, row_label,
                  ha='center', va='center', fontsize=14, rotation=90,
                  transform=fig3.transFigure)

    # Colorbars — derived from actual axes positions so alignment is exact
    # ax0/ax1/ax2 still reference the last-row panels after the loop
    fig3.canvas.draw()   # force layout so get_position() is reliable
    _p0 = ax0.get_position()
    _p1 = ax1.get_position()
    _p2 = ax2.get_position()
    _cbar_y, _cbar_h, _pad = 0.045, 0.018, 0.01

    # Colorbar 1: centered under cols 0 + 1
    cbar_ax3 = fig3.add_axes([_p0.x0 + _pad, _cbar_y,
                               _p1.x1 - _p0.x0 - 2 * _pad, _cbar_h])
    plt.colorbar(pm0, cax=cbar_ax3, orientation='horizontal').set_label(f'Precipitation ({unit_label})')

    # Colorbar 2: centered under col 2
    cbar_ax3d = fig3.add_axes([_p2.x0 + _pad, _cbar_y,
                                _p2.x1 - _p2.x0 - 2 * _pad, _cbar_h])
    plt.colorbar(pm2, cax=cbar_ax3d, orientation='horizontal').set_label('Precipitation change (%)')

    plt.savefig(f'/home/dargueso/Analyses/EPICC/Hourly_vs_Daily/combined_map_violin_precipitation_changes_q{qtile}th{suffix_season}{unit_suffix}.png',
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig3)
