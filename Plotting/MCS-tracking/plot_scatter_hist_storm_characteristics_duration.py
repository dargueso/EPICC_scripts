#!/usr/bin/env python
"""
#####################################################################
# Author: Daniel Argueso <daniel>
# Date:   2022-04-28T13:08:07+02:00
# Email:  d.argueso@uib.es
# Last modified by:   daniel
# Last modified time: 2022-04-28T13:08:08+02:00
#
# @Project@
# Version: x.0 (Beta)
# Description:
#
# Dependencies:
#
# Files:
#
#####################################################################
"""

import pickle
import numpy as np
import xarray as xr
import netCDF4 as nc
import epicc_config as cfg
import pandas as pd
import matplotlib.pyplot as plt
import wrf
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from glob import glob
import datetime as dt

###########################################################
###########################################################

wrf_runs = cfg.wrf_runs
lsmooth = 50000 #Smoothing filter length in meters
dx = 2000 #Resolution of imput data
lsmooth_ngrid = np.ceil(lsmooth/dx)
pr_threshold_mmhr = 2.5 #Precipitation threshold in mm/h
freq = '01H'

mmh_factor = {'10MIN': 6.,
              '01H'  : 1.,
              'DAY'  : 1/24.}

freq_dt = {'10MIN': 600,
          '01H'  : 3600,
          'DAY'  : 86400.}

pr_threshold = pr_threshold_mmhr/mmh_factor[freq]
dt = freq_dt[freq] #Step between model outputs in seconds
wrun = wrf_runs[0]
reg = 'WME'
syear = 2013
eyear = 2020
smonth = 1
emonth = 12
allmonths = [8,9,10]

pr_max = 250
dur_max = 96
exp = 'exp3'
###########################################################
###########################################################

#Loading geo info from WRF
geofile_ref = nc.Dataset(f'{cfg.geofile_ref}')
xbounds = wrf.cartopy_xlim(wrfin=geofile_ref)
ybounds = wrf.cartopy_ylim(wrfin=geofile_ref)
geo_proj = wrf.get_cartopy(wrfin=geofile_ref)


hgt = wrf.getvar(geofile_ref, "ter")
lats, lons = wrf.latlon_coords(hgt)



###########################################################
##########################################################
storms_pres_summary = pd.read_pickle(f"storms_summary_{syear}-{eyear}_{allmonths[0]:02d}-{allmonths[-1]:02d}_{reg}_{exp}.pkl")
storms_fut_summary = pd.read_pickle(f"storms_fut_summary_{syear}-{eyear}_{allmonths[0]:02d}-{allmonths[-1]:02d}_{reg}_{exp}.pkl")
###########################################################
###########################################################

# https://matplotlib.org/3.5.0/gallery/lines_bars_and_markers/scatter_hist.html
fig = plt.figure(figsize=(25, 20))

# Add a gridspec with two rows and two columns and a ratio of 2 to 7 between
# the size of the marginal axes and the main axes in both directions.
# Also adjust the subplot parameters for a square plot.
gs = fig.add_gridspec(4, 5,  width_ratios=(2,7,2,7,0.5), height_ratios=(2, 7,2,7),
                      left=0.05, right=0.95, bottom=0.05, top=0.95,
                      wspace=0.15, hspace=0.05)

ax = fig.add_subplot(gs[1, 1])
ax_histx = fig.add_subplot(gs[0, 1], sharex=ax)
ax_histy = fig.add_subplot(gs[1, 0], sharey=ax)

ax.set_ylim(0,dur_max)
ax.set_xlim(0,pr_max)
ax.text(0.95,0.95,f"Present",fontsize='xx-large',fontweight='bold',color="deepskyblue",horizontalalignment='right',transform=ax.transAxes)

ax_histx.tick_params(axis="x", labelbottom=False)
ax_histy.tick_params(axis="y", labelleft=False)
ax_histy.invert_xaxis()
ax_histy.yaxis.tick_right()

ax.set_ylabel("Duration (hr)")
ax.set_xlabel("Max. precip. rate (mm/hr)")
# adjust_spines2(ax, ['left','bottom'])
# adjust_spines2(ax_histy, ['right','bottom'])
# adjust_spines2(ax_histx, ['left','bottom'])

sct = ax.scatter(storms_pres_summary.prmax,storms_pres_summary.duration,
           s=(storms_pres_summary.max_size/100)*2,
           c=storms_pres_summary.tot_vol/1000,cmap='Spectral',
           vmin=1, vmax=1000,
           alpha=0.5)

ax_histx.hist(storms_pres_summary.prmax,bins = np.linspace(0,pr_max,20),color = "lightgray",alpha=0.8,edgecolor='gray',rwidth=0.8,weights=np.ones_like(storms_pres_summary.prmax) / len(storms_pres_summary.prmax))
ax_histy.hist(storms_pres_summary.duration,bins = np.linspace(0,dur_max,20),color = "lightgray",alpha=0.8,edgecolor='gray', orientation='horizontal',rwidth=0.8,weights=np.ones_like(storms_pres_summary.duration) / len(storms_pres_summary.duration))

# cbar_ax = fig.add_subplot(gs[1, 2])
# cbar=plt.colorbar(sct, cax=cbar_ax,orientation="vertical")
# cbar.set_label ('Total rain (m3)')
kw = dict(prop="sizes", num=5, fmt="{x:.0f}",
          func=lambda s: s*100/2000)
legend = ax.legend(*sct.legend_elements(**kw),
                    loc="lower center", title="Storm size\n ($10^4$ $km^2$)",frameon=False,labelspacing = 2,borderaxespad=2,ncol = 7)
plt.setp(legend.get_title(), multialignment='center')
############################################################
###########################################################

ax2 = fig.add_subplot(gs[1, 3])
ax2_histx = fig.add_subplot(gs[0, 3], sharex=ax2)
ax2_histy = fig.add_subplot(gs[1, 2], sharey=ax2)

ax2.set_ylim(0,dur_max)
ax2.set_xlim(0,pr_max)
ax2.text(0.95,0.95,f"Future",fontsize='xx-large',fontweight='bold',color="crimson",horizontalalignment='right',transform=ax2.transAxes)

#ax2.text(0.98,0.98,f'{len(storms_fut_summary)}', fontsize='large', horizontalalignment='right', transform=ax2.transAxes)

ax2_histx.tick_params(axis="x", labelbottom=False)
ax2_histy.tick_params(axis="y", labelleft=False)
ax2_histy.invert_xaxis()
ax2_histy.yaxis.tick_right()

ax2.set_ylabel("Duration (hr)")
ax2.set_xlabel("Max. precip. rate (mm/hr)")

sct2 = ax2.scatter(storms_fut_summary.prmax,storms_fut_summary.duration,
           s=(storms_fut_summary.max_size/100)*2,
           c=storms_fut_summary.tot_vol/1000,cmap='Spectral',
           vmin=1, vmax=1000,
           alpha=0.5)

ax2_histx.hist(storms_fut_summary.prmax,bins = np.linspace(0,pr_max,20),color = "lightgray",alpha=0.8,edgecolor='gray',rwidth=0.8,weights =np.ones_like(storms_fut_summary.prmax) / len(storms_fut_summary.prmax))
ax2_histy.hist(storms_fut_summary.duration,bins = np.linspace(0,dur_max,20),color = "lightgray",alpha=0.8,edgecolor='gray', orientation='horizontal',rwidth=0.8,weights=np.ones_like(storms_fut_summary.duration) / len(storms_fut_summary.duration))

cbar_ax = fig.add_subplot(gs[1, 4])
cbar=plt.colorbar(sct2, cax=cbar_ax,orientation="vertical")
cbar.set_label ('Total rain (m3)')
kw2 = dict(prop="sizes", num=5, fmt="{x:.0f}",
          func=lambda s: s*100/2000)
legend2 = ax2.legend(*sct.legend_elements(**kw2),
                    loc="lower center", title="Storm size\n ($10^4$ $km^2$)",frameon=False,labelspacing = 2,borderaxespad=2,ncol = 7)
plt.setp(legend2.get_title(), multialignment='center')

###########################################################
###########################################################


ax3 = fig.add_subplot(gs[3, 1])
ax3_histx = fig.add_subplot(gs[2, 1], sharex=ax3)
ax3_histy = fig.add_subplot(gs[3, 0], sharey=ax3)

ax3.set_ylim(0,dur_max)
ax3.set_xlim(0,pr_max)

ax3_histx.tick_params(axis="x", labelbottom=False)
ax3_histy.tick_params(axis="y", labelleft=False)
ax3_histy.invert_xaxis()
ax3_histy.yaxis.tick_right()

ax3.set_ylabel("Duration (hr)")
ax3.set_xlabel("Max. precip. rate (mm/hr)")
# adjust_spines2(ax, ['left','bottom'])
# adjust_spines2(ax_histy, ['right','bottom'])
# adjust_spines2(ax_histx, ['left','bottom'])

sct3p = ax3.scatter(storms_pres_summary.prmax,storms_pres_summary.duration,
           s=(storms_pres_summary.max_size/100)*2,
           c='deepskyblue',
           alpha=0.5)

sct3f = ax3.scatter(storms_fut_summary.prmax,storms_fut_summary.duration,
           s=(storms_fut_summary.max_size/100)*2,
           c='crimson',
           alpha=0.5)

ax3_histx.hist(storms_pres_summary.prmax,bins = np.linspace(0,pr_max,20),color = "deepskyblue",alpha=0.8,edgecolor='deepskyblue',rwidth=0.8, weights = np.ones_like(storms_pres_summary.prmax) / len(storms_pres_summary.prmax))
ax3_histy.hist(storms_pres_summary.duration,bins = np.linspace(0,dur_max,20),color = "deepskyblue",alpha=0.8,edgecolor='deepskyblue', orientation='horizontal',rwidth=0.8,weights=np.ones_like(storms_pres_summary.duration) / len(storms_pres_summary.duration))

ax3_histx.hist(storms_fut_summary.prmax,bins = np.linspace(0,pr_max,20),color = "crimson",alpha=0.8,edgecolor='crimson',rwidth=0.8,weights =np.ones_like(storms_fut_summary.prmax) / len(storms_fut_summary.prmax))
ax3_histy.hist(storms_fut_summary.duration,bins = np.linspace(0,dur_max,20),color = "crimson",alpha=0.8,edgecolor='crimson', orientation='horizontal',rwidth=0.8,weights=np.ones_like(storms_fut_summary.duration) / len(storms_fut_summary.duration))

ax3.text(0.88,0.9,f"{len(storms_pres_summary)}",fontsize='xx-large',fontweight='bold',color="deepskyblue",transform=ax3.transAxes)
ax3.text(0.88,0.85,f"{len(storms_fut_summary)}",fontsize='xx-large',fontweight='bold',color="crimson",transform=ax3.transAxes)

fig.suptitle("Storm characteristics in Western Mediterranean (Aug-Oct)", fontsize='30',fontweight='bold')
fig.savefig(f'{cfg.path_out}/MCS-tracking/Scatter_storm_characteristics_duration_both_{reg}_{exp}.png', bbox_inches='tight',dpi=300)
