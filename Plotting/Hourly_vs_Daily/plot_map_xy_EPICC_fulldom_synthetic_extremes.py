#!/usr/bin/env python
'''
@File    :  plot_map_xy_EPICC_fulldom_synthetic_extremes.py
@Time    :  2026/03/23
@Author  :  Daniel Argüeso
@Version :  1.0
@Contact :  d.argueso@uib.es
@License :  (C)Copyright 2025, Daniel Argüeso
@Project :  EPICC
@Desc    :  Map + XY quantile plots using output from pipeline_full_domain.py.
            Produces TWO figures, one per method:
              Fig 1 (Method C — hourly intensity):
                Rows 0-1: 8 XY panels (4 locations each row)
                Row  2:   map of future−present difference at QTILE_IDX
              Fig 2 (Method B — daily-max):
                same layout
            Both present synthetic CI (validation) and future synthetic CI
            (attribution) are shown in each XY panel.
            Significance hatching: future obs > synthetic future upper CI
            (change not explained by daily totals alone).
            All data loaded from pipeline_full_domain.py NC outputs — no
            zarr re-reading needed.
'''

import numpy as np
import xarray as xr
import netCDF4 as nc
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import BoundaryNorm
from matplotlib import colormaps as cmaps
from matplotlib.gridspec import GridSpec
import seaborn as sns
import string

from wrf import (to_np, getvar, get_cartopy, cartopy_xlim,
                 cartopy_ylim, latlon_coords, GeoBounds)

mpl.rcParams['font.size'] = 13
mpl.rcParams['hatch.color'] = 'red'
mpl.rcParams['hatch.linewidth'] = 0.8

# =============================================================================
# CONFIGURATION
# =============================================================================

PATH_IN  = '/home/dargueso/postprocessed/EPICC/EPICC_2km_ERA5'
PATH_OUT = '/home/dargueso/Analyses/EPICC/Hourly_vs_Daily'

WRUN_PRESENT = 'EPICC_2km_ERA5'
WRUN_FUTURE  = 'EPICC_2km_ERA5_CMIP6anom'

BUFFER = 3          # must match the buffer used in pipeline_full_domain.py

# Quantile level used for the map panels (index into plot_q dimension)
# [0.90, 0.95, 0.98, 0.99, 0.995, 0.999] → index 3 = P99
QTILE_IDX = 3

CI_LO = 0.025       # lower bootstrap quantile
CI_HI = 0.975       # upper bootstrap quantile

# Locations: (grid-y index, grid-x index, display name)
LOCATIONS = [
    (258, 559, 'Mallorca'),
    (250, 423, 'Turis'),
    (384, 569, 'Pyrenees'),
    (527, 795, 'Rosiglione'),
    (533, 638, 'Ardeche'),
    (407, 821, 'Corte'),
    (174,1091, 'Catania'),
    (425, 989, "L'Aquila"),
]

GEO_FILE = ('/home/dargueso/share/geo_em_files/EPICC/'
            'geo_em.d01.EPICC_2km_ERA5_HVC_GWD.nc')
BORDER_WIDTH = 50

# Colour levels for the difference maps
DIFF_LEVELS = np.arange(-10, 11, 1)

# =============================================================================
# WRF GEOMETRY
# =============================================================================

geo_file  = xr.open_dataset(GEO_FILE)
lm_is     = geo_file.LANDMASK.squeeze()

border_mask          = np.zeros_like(lm_is.values)
border_mask[:BORDER_WIDTH, :]  = 1
border_mask[-BORDER_WIDTH:, :] = 1
border_mask[:, :BORDER_WIDTH]  = 1
border_mask[:, -BORDER_WIDTH:] = 1

fileref    = nc.Dataset(GEO_FILE)
hgt_wrf    = getvar(fileref, 'HGT_M', timeidx=0)
hgt_wrf    = hgt_wrf.where(hgt_wrf >= 0, 0)
lats_wrf, lons_wrf = latlon_coords(hgt_wrf)
cart_proj  = get_cartopy(hgt_wrf)
cart_proj._threshold /= 100.

# =============================================================================
# LOAD PIPELINE OUTPUTS
# =============================================================================

fname_pres = f'{PATH_IN}/synthetic_pres_buf{BUFFER}.nc'
fname_fut  = f'{PATH_IN}/synthetic_fut_buf{BUFFER}.nc'

print(f'Loading {fname_pres} …')
fin_pres = xr.open_dataset(fname_pres)
print(f'Loading {fname_fut} …')
fin_fut  = xr.open_dataset(fname_fut)

plot_q = fin_pres['plot_q'].values   # e.g. [0.90, 0.95, 0.98, 0.99, 0.995, 0.999]
qtile_label = f'P{int(plot_q[QTILE_IDX]*100)}'

# ---- Map data ----------------------------------------------------------------

# Observed difference at chosen quantile
diff_h  = (fin_fut['obs_h'].isel(plot_q=QTILE_IDX)
           - fin_pres['obs_h'].isel(plot_q=QTILE_IDX)).values
diff_dm = (fin_fut['obs_dm'].isel(plot_q=QTILE_IDX)
           - fin_pres['obs_dm'].isel(plot_q=QTILE_IDX)).values

# Significance: future obs > synthetic future upper CI
#   (future extremes larger than what daily-total changes alone predict)
sig_h  = (fin_fut['obs_h'].isel(plot_q=QTILE_IDX).values
          > fin_fut['syn_h_C'].sel(bootstrap_q=CI_HI).isel(plot_q=QTILE_IDX).values
          ).astype(int)
sig_dm = (fin_fut['obs_dm'].isel(plot_q=QTILE_IDX).values
          > fin_fut['syn_dm_B'].sel(bootstrap_q=CI_HI).isel(plot_q=QTILE_IDX).values
          ).astype(int)

# lat / lon from the NC file (saved by pipeline_full_domain)
lat2d = fin_pres['lat'].values
lon2d = fin_pres['lon'].values

sig_levs = np.array([-0.5, 0.5, 1.5])

# =============================================================================
# COLOUR MAPS
# =============================================================================

cmap_diff = cmaps['BrBG']
norm_diff  = BoundaryNorm(DIFF_LEVELS, ncolors=cmap_diff.N, extend='both')

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def draw_xy_panel(ax, yloc, xloc, name, letter_idx,
                  obs_pres_da, obs_fut_da,
                  syn_pres_da, syn_fut_da,
                  ylabel, first_col, last_row):
    """Draw one XY quantile panel for a single location."""
    obs_p     = obs_pres_da.isel(y=yloc, x=xloc).values
    obs_f     = obs_fut_da.isel(y=yloc, x=xloc).values
    syn_p_lo  = syn_pres_da.sel(bootstrap_q=CI_LO).isel(y=yloc, x=xloc).values
    syn_p_med = syn_pres_da.sel(bootstrap_q=0.5  ).isel(y=yloc, x=xloc).values
    syn_p_hi  = syn_pres_da.sel(bootstrap_q=CI_HI).isel(y=yloc, x=xloc).values
    syn_f_lo  = syn_fut_da.sel(bootstrap_q=CI_LO ).isel(y=yloc, x=xloc).values
    syn_f_med = syn_fut_da.sel(bootstrap_q=0.5   ).isel(y=yloc, x=xloc).values
    syn_f_hi  = syn_fut_da.sel(bootstrap_q=CI_HI ).isel(y=yloc, x=xloc).values

    # Synthetic present CI — blue shaded (validation)
    ax.fill_between(plot_q, syn_p_lo, syn_p_hi,
                    color='#2E86AB', alpha=0.18, zorder=1)
    ax.plot(plot_q, syn_p_med, color='#2E86AB', linewidth=0.8,
            linestyle='--', alpha=0.7, zorder=2)

    # Synthetic future CI — orange shaded (attribution)
    ax.fill_between(plot_q, syn_f_lo, syn_f_hi,
                    color='#F18F01', alpha=0.22, zorder=1)
    ax.plot(plot_q, syn_f_med, color='#F18F01', linewidth=0.8,
            linestyle='--', alpha=0.8, zorder=2, label='Synthetic future (median)')

    # Observed lines
    ax.plot(plot_q, obs_p, color='#2E86AB', linewidth=1.4,
            marker='o', markersize=3, label='Present obs', zorder=3)
    ax.plot(plot_q, obs_f, color='#E50C0C', linewidth=1.4,
            marker='s', markersize=3, label='Future obs', zorder=3)

    ax.set_yscale('log')
    ax.set_title(string.ascii_lowercase[letter_idx],
                 size='large', weight='bold', loc='left')
    ax.text(0.03, 0.97, name, transform=ax.transAxes,
            fontsize=10, va='top', ha='left')

    ax.xaxis.set_major_locator(mticker.FixedLocator(plot_q))
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda v, _: f'P{int(v*100)}' if v*100 == int(v*100) else f'P{v*100:.1f}'))
    ax.tick_params(axis='x', labelsize=8, rotation=30)

    ax.set_yticks([1, 5, 10, 20, 50, 100])
    ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.yaxis.get_major_formatter().set_scientific(False)
    ax.tick_params(axis='y', which='major', labelsize=9)
    ax.minorticks_on()

    ax.grid(True, which='both', linestyle=':', linewidth=0.4, alpha=0.6)
    ax.set_axisbelow(True)

    if last_row:
        ax.set_xlabel('Quantile', fontsize=10, fontweight='bold')
    if first_col:
        ax.set_ylabel(ylabel, fontsize=10, fontweight='bold')
    else:
        ax.yaxis.set_tick_params(labelleft=False)


def draw_map(axm, diff_data, sig_data, letter_idx, title):
    """Draw one difference map with significance hatching and location markers."""
    axm.set_title(f'{string.ascii_lowercase[letter_idx]}  {title}',
                  size='medium', weight='bold', loc='left', pad=6)
    axm.add_feature(cfeature.COASTLINE, linewidth=0.5, zorder=102)

    pm = axm.pcolormesh(
        to_np(lons_wrf), to_np(lats_wrf), diff_data,
        cmap=cmap_diff, norm=norm_diff,
        transform=ccrs.PlateCarree(), zorder=101,
    )
    axm.contourf(
        to_np(lons_wrf), to_np(lats_wrf), border_mask,
        levels=[0.5, 1.5], colors=['gray'], alpha=0.5,
        transform=ccrs.PlateCarree(), zorder=103,
    )
    axm.contourf(
        to_np(lons_wrf), to_np(lats_wrf), sig_data,
        sig_levs, colors='none', hatches=['', '.....'],
        transform=ccrs.PlateCarree(), zorder=105,
    )
    for yloc, xloc, lname in LOCATIONS:
        locx = float(lons_wrf[yloc, xloc])
        locy = float(lats_wrf[yloc, xloc])
        axm.plot(locx, locy, 'r^', markersize=6,
                 transform=ccrs.PlateCarree(), zorder=106)
        axm.text(locx + 0.15, locy + 0.15, lname, fontsize=6,
                 color='darkred', transform=ccrs.PlateCarree(), zorder=106)

    axm.set_xlim(cartopy_xlim(hgt_wrf))
    axm.set_ylim(cartopy_ylim(hgt_wrf))
    gl = axm.gridlines(color='black', linestyle='dotted', linewidth=0.4,
                       draw_labels=True, x_inline=False, y_inline=False,
                       zorder=103)
    gl.right_labels = False
    gl.top_labels   = False
    gl.xlabel_style = {'size': 8}
    gl.ylabel_style = {'size': 8}

    cbar = plt.colorbar(pm, ax=axm, orientation='vertical',
                        shrink=0.7, pad=0.02,
                        label='Precipitation change (mm)')
    cbar.ax.tick_params(labelsize=8)


# =============================================================================
# BUILD AND SAVE ONE FIGURE PER METHOD
# =============================================================================
#
#  Layout per figure (same as original script):
#   Row 0:  4 XY panels (locations 0–3)
#   Row 1:  4 XY panels (locations 4–7)
#   Row 2:  1 map
#

FIGURE_CONFIGS = [
    dict(
        obs_pres_da = fin_pres['obs_h'],
        obs_fut_da  = fin_fut['obs_h'],
        syn_pres_da = fin_pres['syn_h_C'],
        syn_fut_da  = fin_fut['syn_h_C'],
        diff_data   = diff_h,
        sig_data    = sig_h,
        ylabel      = '1-hour precipitation (mm)',
        method_label= 'Method C — Hourly intensity',
        map_title   = f'Future − present {qtile_label} hourly intensity (mm)',
        suffix      = 'hourly_C',
    ),
    dict(
        obs_pres_da = fin_pres['obs_dm'],
        obs_fut_da  = fin_fut['obs_dm'],
        syn_pres_da = fin_pres['syn_dm_B'],
        syn_fut_da  = fin_fut['syn_dm_B'],
        diff_data   = diff_dm,
        sig_data    = sig_dm,
        ylabel      = 'Daily-max 1-hour precip (mm)',
        method_label= 'Method B — Daily-max',
        map_title   = f'Future − present {qtile_label} daily-max (mm)',
        suffix      = 'dmax_B',
    ),
]

suptitle_base = (
    f'Blue shading: synthetic present CI (validation)   '
    f'Orange shading: synthetic future CI (attribution)\n'
    f'Hatching on map: future obs > synthetic future upper CI '
    f'(structural change significant)'
)

for cfg in FIGURE_CONFIGS:
    fig = plt.figure(figsize=(16, 20), constrained_layout=False)
    gs_main = GridSpec(
        3, 1,
        height_ratios=[1, 1, 3],
        figure=fig,
        left=0.06, bottom=0.06, right=0.96, top=0.93,
        hspace=0.30,
    )
    gs_row0 = gs_main[0].subgridspec(1, 4, wspace=0.08)
    gs_row1 = gs_main[1].subgridspec(1, 4, wspace=0.08)

    # XY panels
    for iloc, (yloc, xloc, name) in enumerate(LOCATIONS):
        col     = iloc % 4
        row_rel = iloc // 4
        gs_row  = gs_row0 if row_rel == 0 else gs_row1
        letter  = iloc

        ax = fig.add_subplot(gs_row[0, col])
        draw_xy_panel(
            ax, yloc, xloc, name, letter,
            cfg['obs_pres_da'], cfg['obs_fut_da'],
            cfg['syn_pres_da'], cfg['syn_fut_da'],
            cfg['ylabel'],
            first_col=(col == 0),
            last_row=(row_rel == 1),
        )
        if iloc == 0:
            ax.legend(fontsize=7, loc='upper left', frameon=True,
                      fancybox=True, framealpha=0.8)

    # Map
    axm = fig.add_subplot(gs_main[2], projection=cart_proj)
    draw_map(axm, cfg['diff_data'], cfg['sig_data'],
             letter_idx=8, title=cfg['map_title'])

    fig.suptitle(
        f"{cfg['method_label']} — buffer {BUFFER} — {qtile_label}\n{suptitle_base}",
        fontsize=11, y=0.965,
    )

    outfile = (f'{PATH_OUT}/map_xy_fulldom_synthetic_extremes'
               f'_buf{BUFFER}_{qtile_label}_{cfg["suffix"]}.png')
    print(f'Saving {outfile} …')
    fig.savefig(outfile, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  Done.')

print('All figures saved.')
