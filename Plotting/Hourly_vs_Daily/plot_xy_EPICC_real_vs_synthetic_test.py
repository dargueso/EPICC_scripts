#!/usr/bin/env python
'''
@File    :  plot_xy_EPICC_real_vs_synthetic_test.py
@Time    :  2026/02/23
@Author  :  Daniel Argüeso
@Version :  1.0
@Contact :  d.argueso@uib.es
@License :  (C)Copyright 2025, Daniel Argüeso
@Project :  EPICC
@Desc    :  XY quantile plots (no map) for 21x21 test area around Catania.
            Validates the hourly-from-daily synthetic future pipeline.
            Observed quantiles use the CENTER PIXEL only (not 21x21 pooled)
            so the comparison with the center-pixel synthetic is fair.
            Produces two plots:
              1. Hourly intensity quantiles
              2. Daily max hourly rainfall quantiles
'''

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import ScalarFormatter
import os

mpl.rcParams["font.size"] = 14

###########################################################################
# Configuration
###########################################################################

PATH_IN = '/home/dargueso/postprocessed/EPICC/'
PATH_OUT = '/home/dargueso/Analyses/EPICC/Hourly_vs_Daily/'

WRUN_PRESENT = 'EPICC_2km_ERA5'
WRUN_FUTURE  = 'EPICC_2km_ERA5_CMIP6anom'

test_suffix = '_test_21x21'

WET_VALUE_HIGH = 0.1   # mm  — threshold for wet hourly values
WET_VALUE_LOW  = 1.0   # mm  — threshold for wet days (daily total)

cl_hi = 0.975
cl_lo = 0.025

CENTER = 10  # centre pixel in 21x21 grid (0-based index, centre = 10)

os.makedirs(PATH_OUT, exist_ok=True)

###########################################################################
# File paths
###########################################################################

zarr_path_present = f'{PATH_IN}/{WRUN_PRESENT}/UIB_01H_RAIN{test_suffix}.zarr'
zarr_path_future  = f'{PATH_IN}/{WRUN_FUTURE}/UIB_01H_RAIN{test_suffix}.zarr'
# Synthetic built from present daily totals + present conditional probabilities
# → bootstrap CI should envelop the present observed quantiles (validation)
filein_syn        = f'{PATH_IN}/{WRUN_PRESENT}/synthetic_future_01H_from_DAY_confidence{test_suffix}.nc'

###########################################################################
# Load synthetic data — derive quantile axis
###########################################################################

fin_syn_all = xr.open_dataset(filein_syn).squeeze()
fin_syn_all = fin_syn_all.isel(quantile=slice(10, -2))
qtiles = fin_syn_all['quantile'].values

# Centre pixel for XY plots
fin_syn = fin_syn_all.isel(y=CENTER, x=CENTER).squeeze()

###########################################################################
# Load observed zarr data
###########################################################################

try:
    ds_present = xr.open_zarr(zarr_path_present, consolidated=True)
    ds_future  = xr.open_zarr(zarr_path_future,  consolidated=True)
    print("   Opened with consolidated metadata")
except KeyError:
    ds_present = xr.open_zarr(zarr_path_present, consolidated=False)
    ds_future  = xr.open_zarr(zarr_path_future,  consolidated=False)
    print("   Opened without consolidated metadata")

rain_pres = ds_present.RAIN.astype(np.float32)
rain_fut  = ds_future.RAIN.astype(np.float32)

###########################################################################
# Plot 1 — Hourly Intensity Quantiles
###########################################################################

print("Computing hourly intensity quantiles (centre pixel only) …")

# Centre pixel time series — comparable with centre-pixel synthetic
rain_pres_ctr = rain_pres.isel(y=CENTER, x=CENTER)
rain_fut_ctr  = rain_fut.isel(y=CENTER, x=CENTER)

pres_wet = rain_pres_ctr.where(rain_pres_ctr > WET_VALUE_HIGH).dropna(dim="time")
fut_wet  = rain_fut_ctr.where(rain_fut_ctr   > WET_VALUE_HIGH).dropna(dim="time")

pres_qtiles_1h = pres_wet.chunk(dict(time=-1)).quantile(qtiles, dim='time', skipna=True)
fut_qtiles_1h  = fut_wet.chunk(dict(time=-1)).quantile(qtiles, dim='time', skipna=True)

fig1, ax1 = plt.subplots(figsize=(8, 6))

ax1.plot(qtiles, pres_qtiles_1h, label='Present-day observations (centre pixel)',
         color='#2E86AB', linewidth=1.5, marker='o', markersize=4, zorder=4)
# Future observed — reference only (no CI)
ax1.plot(qtiles, fut_qtiles_1h, label='Future observations (reference, centre pixel)',
         color='#E50C0C', linewidth=1.5, marker='s', markersize=4,
         linestyle='--', zorder=3)

# Synthetic present CI (present condprob × present daily totals)
# Present observed should fall within this band
syn_lo = fin_syn.sel(bootstrap_quantile=cl_lo).hourly_intensity.squeeze()
syn_hi = fin_syn.sel(bootstrap_quantile=cl_hi).hourly_intensity.squeeze()
syn_med = fin_syn.sel(bootstrap_quantile=0.5).hourly_intensity.squeeze()
ax1.plot(qtiles, syn_med, label='Present synthetic (median)',
         color='#F18F01', linewidth=1.5, linestyle='-', marker=None, zorder=5)
ax1.plot(qtiles, syn_lo, color='#F18F01', linewidth=0.8, linestyle=':', marker=None)
ax1.plot(qtiles, syn_hi, color='#F18F01', linewidth=0.8, linestyle=':', marker=None)
ax1.fill_between(qtiles, syn_lo, syn_hi,
                 color='#F18F01', alpha=0.2, label=f'Present synthetic CI ({cl_lo}–{cl_hi})')

ax1.set_yscale('log')
ax1.set_xlabel('Quantiles', fontsize=12, fontweight='bold')
ax1.set_ylabel('1-hour precipitation (mm)', fontsize=12, fontweight='bold')
ax1.set_title('Hourly Intensity Quantiles — validation (Catania, centre pixel)',
              fontsize=13, fontweight='bold')

yticks = [1, 2, 5, 10, 20, 50]
ax1.set_yticks(yticks)
ax1.yaxis.set_major_formatter(ScalarFormatter())
ax1.yaxis.get_major_formatter().set_scientific(False)

ax1.grid(True, which='both', linestyle=':', linewidth=0.5, alpha=0.7)
ax1.set_axisbelow(True)
ax1.minorticks_on()
ax1.tick_params(axis='both', which='major', labelsize=11)

handles, labels = ax1.get_legend_handles_labels()
# Deduplicate legend (two CI boundary lines share colour; keep unique entries)
seen = {}
unique_handles, unique_labels = [], []
for h, l in zip(handles, labels):
    if l not in seen:
        seen[l] = True
        unique_handles.append(h)
        unique_labels.append(l)
ax1.legend(unique_handles, unique_labels, frameon=True, fancybox=True,
           shadow=True, fontsize=10, loc='upper left')

fig1.tight_layout()
outfile1 = f'{PATH_OUT}/quantiles_1H_intensity_test_{cl_hi}.png'
fig1.savefig(outfile1, dpi=150, bbox_inches='tight', facecolor='white')
print(f"Saved: {outfile1}")
plt.close(fig1)

###########################################################################
# Plot 2 — Daily Maximum Hourly Rainfall Quantiles
###########################################################################

print("Computing daily-max hourly quantiles (centre pixel only) …")

# Centre pixel daily max
pres_dmax = rain_pres_ctr.resample(time='1D').max()
fut_dmax  = rain_fut_ctr.resample(time='1D').max()

# Daily total — used to filter wet days (total >= WET_VALUE_LOW)
pres_dtot = rain_pres_ctr.resample(time='1D').sum()
fut_dtot  = rain_fut_ctr.resample(time='1D').sum()

# Keep only wet days
pres_dmax_wet = pres_dmax.where(pres_dtot >= WET_VALUE_LOW).dropna(dim="time")
fut_dmax_wet  = fut_dmax.where(fut_dtot  >= WET_VALUE_LOW).dropna(dim="time")

pres_qtiles_dmax = pres_dmax_wet.chunk(dict(time=-1)).quantile(qtiles, dim='time', skipna=True)
fut_qtiles_dmax  = fut_dmax_wet.chunk(dict(time=-1)).quantile(qtiles, dim='time', skipna=True)

fig2, ax2 = plt.subplots(figsize=(8, 6))

ax2.plot(qtiles, pres_qtiles_dmax, label='Present-day observations (centre pixel)',
         color='#2E86AB', linewidth=1.5, marker='o', markersize=4, zorder=4)
# Future observed — reference only (no CI)
ax2.plot(qtiles, fut_qtiles_dmax, label='Future observations (reference, centre pixel)',
         color='#E50C0C', linewidth=1.5, marker='s', markersize=4,
         linestyle='--', zorder=3)

# Synthetic present CI (present condprob × present daily totals)
syn_lo2 = fin_syn.sel(bootstrap_quantile=cl_lo).max_hourly_intensity.squeeze()
syn_hi2 = fin_syn.sel(bootstrap_quantile=cl_hi).max_hourly_intensity.squeeze()
syn_med2 = fin_syn.sel(bootstrap_quantile=0.5).max_hourly_intensity.squeeze()
ax2.plot(qtiles, syn_med2, label='Present synthetic (median)',
         color='#F18F01', linewidth=1.5, linestyle='-', marker=None, zorder=5)
ax2.plot(qtiles, syn_lo2, color='#F18F01', linewidth=0.8, linestyle=':', marker=None)
ax2.plot(qtiles, syn_hi2, color='#F18F01', linewidth=0.8, linestyle=':', marker=None)
ax2.fill_between(qtiles, syn_lo2, syn_hi2,
                 color='#F18F01', alpha=0.2, label=f'Present synthetic CI ({cl_lo}–{cl_hi})')

ax2.set_yscale('log')
ax2.set_xlabel('Quantiles', fontsize=12, fontweight='bold')
ax2.set_ylabel('Daily max 1-hour precipitation (mm)', fontsize=12, fontweight='bold')
ax2.set_title('Daily-Max Hourly Quantiles — validation (Catania, centre pixel)',
              fontsize=13, fontweight='bold')

yticks2 = [1, 2, 5, 10, 20, 50, 100]
ax2.set_yticks(yticks2)
ax2.yaxis.set_major_formatter(ScalarFormatter())
ax2.yaxis.get_major_formatter().set_scientific(False)

ax2.grid(True, which='both', linestyle=':', linewidth=0.5, alpha=0.7)
ax2.set_axisbelow(True)
ax2.minorticks_on()
ax2.tick_params(axis='both', which='major', labelsize=11)

handles2, labels2 = ax2.get_legend_handles_labels()
seen2 = {}
unique_handles2, unique_labels2 = [], []
for h, l in zip(handles2, labels2):
    if l not in seen2:
        seen2[l] = True
        unique_handles2.append(h)
        unique_labels2.append(l)
ax2.legend(unique_handles2, unique_labels2, frameon=True, fancybox=True,
           shadow=True, fontsize=10, loc='upper left')

fig2.tight_layout()
outfile2 = f'{PATH_OUT}/quantiles_1H_daily_max_test_{cl_hi}.png'
fig2.savefig(outfile2, dpi=150, bbox_inches='tight', facecolor='white')
print(f"Saved: {outfile2}")
plt.close(fig2)

print("Done.")
