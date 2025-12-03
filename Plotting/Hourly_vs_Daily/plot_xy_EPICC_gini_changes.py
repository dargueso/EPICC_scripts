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

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D

mpl.rcParams["font.size"] = 14
mpl.rcParams["hatch.color"] = "red"
mpl.rcParams["hatch.linewidth"] = 0.8
###########################################################
###########################################################
med_mask = xr.open_dataset('/home/dargueso/postprocessed/EPICC/EPICC_2km_ERA5/my_coastal_med_mask.nc')
#####################################################################
#####################################################################


buffer = 10



#Load data

filein_pres = f'/home/dargueso/postprocessed/EPICC/EPICC_2km_ERA5/rainfall_probability_optimized_conditional.nc'
fin_pres= xr.open_dataset(filein_pres)

filein_fut = f'/home/dargueso/postprocessed/EPICC/EPICC_2km_ERA5_CMIP6anom/rainfall_probability_optimized_conditional.nc'
fin_fut = xr.open_dataset(filein_fut)

drain_edges = fin_pres.drain_bin.values
labels = [f"{lo}–{hi}" for lo, hi in zip(drain_edges[:-1], drain_edges[1:])]
labels.append(f'>{drain_edges[-1]}')

#####################################################################
#####################################################################

locs_x_idx = [559,423,569,795,638,821,1091,989]#,433,866,335]
locs_y_idx = [258,250,384,527,533,407,174,425]#,254,506,119]
locs_names = ['Mallorca','Turis','Pyrenees','Rosiglione', 'Ardeche','Corte','Catania',"L'Aquila"]#,'Valencia','Barga','Almeria']
gini_dict = {}
for loc, loc_name in enumerate(locs_names):
    print(loc_name)
    xloc = locs_x_idx[loc]
    yloc = locs_y_idx[loc]

    fin_pres_loc = fin_pres.isel(y=slice(yloc-buffer,yloc+buffer+1),x=slice(xloc-buffer,xloc+buffer+1))
    fin_fut_loc = fin_fut.isel(y=slice(yloc-buffer,yloc+buffer+1),x=slice(xloc-buffer,xloc+buffer+1))

    weights_pres = fin_pres_loc['samples_per_bin']
    gini_weighted_mean_pres = fin_pres_loc['gini_coefficient'].weighted(weights_pres).mean(dim=['y', 'x'])
    weights_fut = fin_fut_loc['samples_per_bin']
    gini_weighted_mean_fut = fin_fut_loc['gini_coefficient'].weighted(weights_fut).mean(dim=['y', 'x'])

    gini_dict[loc_name]=np.array([gini_weighted_mean_pres.values, gini_weighted_mean_fut.values])


#Coastal med
weights_all_pres = fin_pres['samples_per_bin']
gini_weighted_mean_all_pres = fin_pres['gini_coefficient'].where(med_mask['combined_mask'].values==2).weighted(weights_all_pres).mean(dim=['y', 'x'])
weights_all_fut = fin_fut['samples_per_bin']
gini_weighted_mean_all_fut = fin_fut['gini_coefficient'].where(med_mask['combined_mask'].values==2).weighted(weights_all_fut).mean(dim=['y', 'x'])

gini_dict['Coastal Med']=np.array([gini_weighted_mean_all_pres.values, gini_weighted_mean_all_fut.values])

# Bootstrap confidence intervals for Coastal Med
def bootstrap_weighted_mean(data, weights, n_bootstrap=1000):
    """Calculate bootstrap confidence intervals for weighted mean"""
    boot_means = []

    for _ in range(n_bootstrap):
        # Resample indices with replacement
        indices = np.random.choice(len(data), size=len(data), replace=True)
        boot_data = data[indices]
        boot_weights = weights[indices]

        # Calculate weighted mean for this bootstrap sample
        boot_mean = np.average(boot_data, weights=boot_weights)
        boot_means.append(boot_mean)

    # Calculate 95% confidence interval
    ci_lower = np.percentile(boot_means, 2.5)
    ci_upper = np.percentile(boot_means, 97.5)

    return ci_lower, ci_upper

# Calculate bootstrap CIs for Coastal Med across all bins
coastal_ci_pres_lower = []
coastal_ci_pres_upper = []
coastal_ci_fut_lower = []
coastal_ci_fut_upper = []

for bin_idx in range(len(drain_edges)):
    # Present climate
    gini_pres_bin = fin_pres['gini_coefficient'].isel(drain_bin=bin_idx).where(med_mask['combined_mask'].values==2)
    weights_pres_bin = fin_pres['samples_per_bin'].isel(drain_bin=bin_idx).where(med_mask['combined_mask'].values==2)

    # Flatten and remove NaNs
    gini_pres_flat = gini_pres_bin.values.flatten()
    weights_pres_flat = weights_pres_bin.values.flatten()
    valid_mask_pres = ~np.isnan(gini_pres_flat) & ~np.isnan(weights_pres_flat)

    if valid_mask_pres.sum() > 0:
        ci_lower, ci_upper = bootstrap_weighted_mean(
            gini_pres_flat[valid_mask_pres],
            weights_pres_flat[valid_mask_pres]
        )
        coastal_ci_pres_lower.append(ci_lower)
        coastal_ci_pres_upper.append(ci_upper)
    else:
        coastal_ci_pres_lower.append(np.nan)
        coastal_ci_pres_upper.append(np.nan)

    # Future climate
    gini_fut_bin = fin_fut['gini_coefficient'].isel(drain_bin=bin_idx).where(med_mask['combined_mask'].values==2)
    weights_fut_bin = fin_fut['samples_per_bin'].isel(drain_bin=bin_idx).where(med_mask['combined_mask'].values==2)

    # Flatten and remove NaNs
    gini_fut_flat = gini_fut_bin.values.flatten()
    weights_fut_flat = weights_fut_bin.values.flatten()
    valid_mask_fut = ~np.isnan(gini_fut_flat) & ~np.isnan(weights_fut_flat)

    if valid_mask_fut.sum() > 0:
        ci_lower, ci_upper = bootstrap_weighted_mean(
            gini_fut_flat[valid_mask_fut],
            weights_fut_flat[valid_mask_fut]
        )
        coastal_ci_fut_lower.append(ci_lower)
        coastal_ci_fut_upper.append(ci_upper)
    else:
        coastal_ci_fut_lower.append(np.nan)
        coastal_ci_fut_upper.append(np.nan)

# Store CIs in dictionary
coastal_ci = {
    'pres_lower': np.array(coastal_ci_pres_lower),
    'pres_upper': np.array(coastal_ci_pres_upper),
    'fut_lower': np.array(coastal_ci_fut_lower),
    'fut_upper': np.array(coastal_ci_fut_upper)
}

# Debug: Print CI ranges to check if they're being calculated
print("\nBootstrap CI summary:")
print(f"Number of bins: {len(coastal_ci['pres_lower'])}")
print(f"Present mean values: {gini_dict['Coastal Med'][0, :]}")
print(f"Present CI lower:    {coastal_ci['pres_lower']}")
print(f"Present CI upper:    {coastal_ci['pres_upper']}")
print(f"\nFuture mean values:  {gini_dict['Coastal Med'][1, :]}")
print(f"Future CI lower:     {coastal_ci['fut_lower']}")
print(f"Future CI upper:     {coastal_ci['fut_upper']}")
print(f"\nCI widths (Present): {coastal_ci['pres_upper'] - coastal_ci['pres_lower']}")
print(f"CI widths (Future):  {coastal_ci['fut_upper'] - coastal_ci['fut_lower']}")

# Define colors
color_present = '#2E86AB'
color_future = '#E50C0C'
color_diff = '#6B4E71'  # Purple for difference

# Define different markers for each location
markers = ['o', 's', '^', 'v', 'D', 'P', '*', 'X', 'p', 'h', '<', '>','8']

# Create figure with two subplots sharing x-axis
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10),
                                sharex=True,
                                gridspec_kw={'height_ratios': [2, 1], 'hspace': 0.15})

# ============= TOP PANEL: Gini Coefficients =============
# Add shaded confidence intervals FIRST (so they appear behind the lines)
x_vals = np.arange(len(drain_edges))
ax1.fill_between(x_vals, coastal_ci['pres_lower'], coastal_ci['pres_upper'],
                  color='#1B4F72', alpha=0.25, linewidth=0, label='95% CI (Present)')
ax1.fill_between(x_vals, coastal_ci['fut_lower'], coastal_ci['fut_upper'],
                  color='#CB4335', alpha=0.25, linewidth=0, label='95% CI (Future)')

# Now plot the lines on top
for idx, (location, gini_values) in enumerate(gini_dict.items()):
    marker = markers[idx % len(markers)]
    
    if location == 'Coastal Med':
        markersize = 10
        linewidth = 2
        linestyle = '-'
        color_present = '#1B4F72'
        color_future = '#CB4335'
    else:
        markersize = 6
        linewidth = 0.75
        linestyle = '--'
        color_present = '#2E86AB'
        color_future = '#E50C0C'


    # Present (first row)
    ax1.plot(range(len(drain_edges)), gini_values[0, :], 
             marker=marker, 
             color=color_present, 
             linestyle=linestyle,
             linewidth=linewidth,
             markersize=markersize,
             alpha=0.7)
    
    # Future (second row)
    ax1.plot(range(len(drain_edges)), gini_values[1, :], 
             marker=marker, 
             color=color_future, 
             linestyle=linestyle,
             linewidth=linewidth,
             markersize=markersize,
             alpha=0.7)

# Add panel label (a)
ax1.set_title("a", size='x-large', weight='bold',loc="left")
# ax1.text(0.02, 0.98, 'a', transform=ax1.transAxes,
#          fontsize=16, fontweight='bold', va='top', ha='left')

# Customize top panel
ax1.set_ylabel('Gini Coefficient', fontsize=12, fontweight='bold')
# ax1.set_title('Gini Coefficient by Daily Rainfall Amount: Present vs Future', 
#               fontsize=14, fontweight='bold', pad=20)
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.set_ylim(0.4, 1.0)

# Create custom legend
# Location handles (black markers with size/width matching actual plot)
location_handles = [Line2D([0], [0], marker=markers[idx % len(markers)], color='black',
                           linestyle='-',
                           markersize=10 if location == 'Coastal Med' else 8,
                           linewidth=2 if location == 'Coastal Med' else 1,
                           label=location)
                   for idx, location in enumerate(gini_dict.keys())]

# Experiment handles (colored lines)
experiment_handles = [
    Line2D([0], [0], color=color_present, linewidth=2, label='Present'),
    Line2D([0], [0], color=color_future, linewidth=2, label='Future')
]

# Combine handles - locations in first column, experiments in second
all_handles = location_handles + experiment_handles
all_labels = [h.get_label() for h in all_handles]

ax1.legend(handles=all_handles, labels=all_labels,
           loc='lower left', frameon=False, ncol=2)

# ============= BOTTOM PANEL: Difference (Future - Present) =============
# Add shaded confidence interval FIRST (so it appears behind the lines)
coastal_diff_lower = coastal_ci['fut_lower'] - coastal_ci['pres_upper']
coastal_diff_upper = coastal_ci['fut_upper'] - coastal_ci['pres_lower']
ax2.fill_between(x_vals, coastal_diff_lower, coastal_diff_upper,
                  color='#4A2C4E', alpha=0.25, linewidth=0, label='95% CI')

# Now plot the lines on top
for idx, (location, gini_values) in enumerate(gini_dict.items()):
    marker = markers[idx % len(markers)]

    # Set location-specific styling
    if location == 'Coastal Med':
        markersize = 10
        linewidth = 2
        linestyle = '-'
        color_diff_loc = '#4A2C4E'  # Darker purple for Coastal Med
    else:
        markersize = 6
        linewidth = 0.75
        linestyle = '--'
        color_diff_loc = '#6B4E71'  # Standard purple for other locations

    # Calculate difference: Future - Present
    difference = gini_values[1, :] - gini_values[0, :]

    ax2.plot(range(len(drain_edges)), difference,
             marker=marker,
             color=color_diff_loc,
             linestyle=linestyle,
             linewidth=linewidth,
             markersize=markersize,
             alpha=0.7)

# Add horizontal line at zero
ax2.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)

# Highlight statistically significant changes for Coastal Med
# Significant if CI doesn't include zero
coastal_med_diff = gini_dict['Coastal Med'][1, :] - gini_dict['Coastal Med'][0, :]
significant_bins = []
for idx in range(len(drain_edges)):
    # Check if zero is outside the confidence interval
    if coastal_diff_lower[idx] > 0 or coastal_diff_upper[idx] < 0:
        significant_bins.append(idx)
        # Significant - add hatching
        ax2.fill_between([idx - 0.4, idx + 0.4],
                         [coastal_diff_lower[idx], coastal_diff_lower[idx]],
                         [coastal_diff_upper[idx], coastal_diff_upper[idx]],
                         color='none', edgecolor='#4A2C4E',
                         hatch='///', linewidth=0, alpha=0.8)

print(f"\nSignificant bins (p<0.05): {significant_bins}")
print(f"Total significant bins: {len(significant_bins)} out of {len(drain_edges)}")

# Add panel label (b)
ax2.set_title("b", size='x-large', weight='bold',loc="left")
# ax2.text(0.02, 0.98, 'b', transform=ax2.transAxes,
#          fontsize=16, fontweight='bold', va='top', ha='left')

# Customize bottom panel
ax2.set_xlabel('Daily Rainfall Bins (mm)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Δ Gini Coefficient\n(Future - Present)', fontsize=12, fontweight='bold')
ax2.set_xticks(range(len(drain_edges)))
ax2.set_xticklabels(labels, rotation=90, fontsize=10)
ax2.grid(True, alpha=0.3, linestyle='--')

# Set symmetric y-limits around zero for better visualization
max_diff = np.max([np.abs(gini_values[1, :] - gini_values[0, :])
                   for gini_values in gini_dict.values()])
ax2.set_ylim(-max_diff * 1.1, max_diff * 1.1)

# Tight layout
plt.tight_layout()

# Save figure
plt.savefig('/home/dargueso/Analyses/EPICC/Hourly_vs_Daily/gini_coefficient_comparison.png', 
            dpi=300, bbox_inches='tight', facecolor='white')

plt.close()