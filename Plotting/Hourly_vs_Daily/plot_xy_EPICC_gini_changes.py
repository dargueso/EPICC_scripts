#!/usr/bin/env python
'''
@File    :  plot_xy_EPICC_gini_changes.py
@Time    :  2025/10/27 11:20:49
@Author  :  Daniel Argüeso
@Version :  2.0
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
med_mask = xr.open_dataset('/home/dargueso/postprocessed/EPICC/EPICC_2km_ERA5/my_coastal_med_mask.nc')
###########################################################

buffer = 10


def compute_gini_from_hist(hist_prob, bin_centers):
    """
    Compute Gini coefficient from a conditional probability histogram via the Lorenz curve.

    Parameters
    ----------
    hist_prob   : ndarray (n_bins_01H, ...) — probability distribution along axis 0
    bin_centers : ndarray (n_bins_01H,)    — bin centre values

    Returns
    -------
    ndarray of shape hist_prob.shape[1:] with Gini values in [0, 1].
    """
    ndim = hist_prob.ndim
    x = bin_centers.reshape((-1,) + (1,) * (ndim - 1))   # broadcast over trailing dims

    mu = np.nansum(hist_prob * x, axis=0)                  # total income (mean intensity)
    valid = mu > 0

    cum_income = np.cumsum(hist_prob * x, axis=0)
    mu_safe = np.where(valid, mu, 1.0)[None]               # avoid div-by-zero
    lorenz = np.where(valid[None], cum_income / mu_safe, 0.0)

    cum_prob = np.cumsum(hist_prob, axis=0)

    # Prepend origin (0, 0) for trapezoidal integration
    zeros = np.zeros_like(lorenz[:1])
    lorenz_ext   = np.concatenate([zeros, lorenz],   axis=0)  # (n_bins+1, ...)
    cum_prob_ext = np.concatenate([zeros, cum_prob], axis=0)  # (n_bins+1, ...)

    dF          = np.diff(cum_prob_ext, axis=0)
    lorenz_mid  = (lorenz_ext[:-1] + lorenz_ext[1:]) / 2
    area_lorenz = np.nansum(dF * lorenz_mid, axis=0)

    return np.where(valid, 1.0 - 2.0 * area_lorenz, np.nan).astype(np.float32)


# ─── Load datasets ────────────────────────────────────────────────────────────
fin_pres = xr.open_dataset(
    '/home/dargueso/postprocessed/EPICC/EPICC_2km_ERA5/condprob_buf5.nc')
fin_fut = xr.open_dataset(
    '/home/dargueso/postprocessed/EPICC/EPICC_2km_ERA5_CMIP6anom/condprob_01H_given_DAY.nc')

# Align on the daily bins that are common to both datasets.
# All 20 future bins (2.5–100 mm) are present in the 27 present bins.
common_bins = np.intersect1d(fin_pres.bin_DAY.values, fin_fut.bin_DAY.values)
fin_pres_common = fin_pres.sel(bin_DAY=common_bins)
fin_fut_common  = fin_fut.sel(bin_DAY=common_bins)

bin_01H = fin_pres_common.bin_01H.values   # same for both (37 bins)
n_bins  = len(common_bins)

# Construct x-axis labels from bin_DAY centres
bin_centers = common_bins
labels = [f"{v:.0f}" for v in bin_centers]

# ─── Compute Gini from histograms, bin by bin to keep memory manageable ───────
print("Computing Gini coefficients from histograms …")

ny = fin_pres_common.dims['y']
nx = fin_pres_common.dims['x']

gini_pres_vals = np.full((n_bins, ny, nx), np.nan, dtype=np.float32)
gini_fut_vals  = np.full((n_bins, ny, nx), np.nan, dtype=np.float32)

for i, b in enumerate(common_bins):
    p_pres = fin_pres_common['hist_intensity'].sel(bin_DAY=b).values       # (37, y, x)
    p_fut  = fin_fut_common['cond_prob_intensity'].sel(bin_DAY=b).values   # (37, y, x)
    gini_pres_vals[i] = compute_gini_from_hist(p_pres, bin_01H)
    gini_fut_vals[i]  = compute_gini_from_hist(p_fut,  bin_01H)
    print(f"  bin_DAY = {b:.1f} mm  ({i+1}/{n_bins})")

# Wrap as DataArrays so xarray weighted averaging works
coords = {'bin_DAY': common_bins}
gini_pres_da = xr.DataArray(gini_pres_vals, dims=['bin_DAY', 'y', 'x'], coords=coords)
gini_fut_da  = xr.DataArray(gini_fut_vals,  dims=['bin_DAY', 'y', 'x'], coords=coords)

n_events_pres = fin_pres_common['n_events']   # (bin_DAY, y, x)
n_events_fut  = fin_fut_common['n_events']    # (bin_DAY, y, x)

# ─── Per-location weighted-mean Gini ──────────────────────────────────────────
locs_x_idx = [559, 423, 569, 795, 638, 821, 1091, 989]
locs_y_idx = [258, 250, 384, 527, 533, 407, 174,  425]
locs_names = ['Mallorca', 'Turis', 'Pyrenees', 'Rosiglione',
              'Ardeche', 'Corte', 'Catania', "L'Aquila"]

gini_dict = {}
for loc, loc_name in enumerate(locs_names):
    print(loc_name)
    xloc = locs_x_idx[loc]
    yloc = locs_y_idx[loc]
    ysl = slice(yloc - buffer, yloc + buffer + 1)
    xsl = slice(xloc - buffer, xloc + buffer + 1)

    gini_pres_loc = gini_pres_da.isel(y=ysl, x=xsl)
    gini_fut_loc  = gini_fut_da.isel(y=ysl, x=xsl)
    wt_pres       = n_events_pres.isel(y=ysl, x=xsl)
    wt_fut        = n_events_fut.isel(y=ysl, x=xsl)

    gini_wmean_pres = gini_pres_loc.weighted(wt_pres).mean(dim=['y', 'x'])
    gini_wmean_fut  = gini_fut_loc.weighted(wt_fut).mean(dim=['y', 'x'])

    gini_dict[loc_name] = np.array([gini_wmean_pres.values, gini_wmean_fut.values])

# Coastal Med
coastal_mask = med_mask['combined_mask'].values == 2
gini_wmean_all_pres = (gini_pres_da.where(coastal_mask)
                       .weighted(n_events_pres).mean(dim=['y', 'x']))
gini_wmean_all_fut  = (gini_fut_da.where(coastal_mask)
                       .weighted(n_events_fut).mean(dim=['y', 'x']))
gini_dict['Coastal Med'] = np.array([gini_wmean_all_pres.values, gini_wmean_all_fut.values])

# ─── Bootstrap confidence intervals for Coastal Med ───────────────────────────
def bootstrap_weighted_mean(data, weights, n_bootstrap=1000):
    boot_means = []
    for _ in range(n_bootstrap):
        idx = np.random.choice(len(data), size=len(data), replace=True)
        boot_means.append(np.average(data[idx], weights=weights[idx]))
    return np.percentile(boot_means, 2.5), np.percentile(boot_means, 97.5)


coastal_ci_pres_lower, coastal_ci_pres_upper = [], []
coastal_ci_fut_lower,  coastal_ci_fut_upper  = [], []

for bin_idx in range(n_bins):
    for (gini_da, n_ev_da, ci_lower_list, ci_upper_list) in [
        (gini_pres_da, n_events_pres, coastal_ci_pres_lower, coastal_ci_pres_upper),
        (gini_fut_da,  n_events_fut,  coastal_ci_fut_lower,  coastal_ci_fut_upper),
    ]:
        g_bin = gini_da.isel(bin_DAY=bin_idx).where(coastal_mask)
        w_bin = n_ev_da.isel(bin_DAY=bin_idx).where(coastal_mask)

        g_flat = g_bin.values.flatten()
        w_flat = w_bin.values.flatten()
        valid  = ~np.isnan(g_flat) & ~np.isnan(w_flat) & (w_flat > 0)

        if valid.sum() > 0:
            lo, hi = bootstrap_weighted_mean(g_flat[valid], w_flat[valid])
        else:
            lo, hi = np.nan, np.nan
        ci_lower_list.append(lo)
        ci_upper_list.append(hi)

coastal_ci = {
    'pres_lower': np.array(coastal_ci_pres_lower),
    'pres_upper': np.array(coastal_ci_pres_upper),
    'fut_lower':  np.array(coastal_ci_fut_lower),
    'fut_upper':  np.array(coastal_ci_fut_upper),
}

print("\nBootstrap CI summary:")
print(f"Number of bins: {n_bins}")
print(f"Present mean values: {gini_dict['Coastal Med'][0, :]}")
print(f"Present CI lower:    {coastal_ci['pres_lower']}")
print(f"Present CI upper:    {coastal_ci['pres_upper']}")
print(f"\nFuture mean values:  {gini_dict['Coastal Med'][1, :]}")
print(f"Future CI lower:     {coastal_ci['fut_lower']}")
print(f"Future CI upper:     {coastal_ci['fut_upper']}")
print(f"\nCI widths (Present): {coastal_ci['pres_upper'] - coastal_ci['pres_lower']}")
print(f"CI widths (Future):  {coastal_ci['fut_upper'] - coastal_ci['fut_lower']}")

# ─── Plotting ─────────────────────────────────────────────────────────────────
markers = ['o', 's', '^', 'v', 'D', 'P', '*', 'X', 'p', 'h', '<', '>', '8']

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10),
                                sharex=True,
                                gridspec_kw={'height_ratios': [2, 1], 'hspace': 0.15})

# ── Top panel: Gini coefficients ──────────────────────────────────────────────
x_vals = np.arange(n_bins)
ax1.fill_between(x_vals, coastal_ci['pres_lower'], coastal_ci['pres_upper'],
                 color='#1B4F72', alpha=0.25, linewidth=0, label='95% CI (Present)')
ax1.fill_between(x_vals, coastal_ci['fut_lower'], coastal_ci['fut_upper'],
                 color='#CB4335', alpha=0.25, linewidth=0, label='95% CI (Future)')

for idx, (location, gini_values) in enumerate(gini_dict.items()):
    marker = markers[idx % len(markers)]
    if location == 'Coastal Med':
        markersize, linewidth, linestyle = 10, 2, '-'
        color_present, color_future = '#1B4F72', '#CB4335'
    else:
        markersize, linewidth, linestyle = 6, 0.75, '--'
        color_present, color_future = '#2E86AB', '#E50C0C'

    ax1.plot(x_vals, gini_values[0, :],
             marker=marker, color=color_present, linestyle=linestyle,
             linewidth=linewidth, markersize=markersize, alpha=0.7)
    ax1.plot(x_vals, gini_values[1, :],
             marker=marker, color=color_future, linestyle=linestyle,
             linewidth=linewidth, markersize=markersize, alpha=0.7)

ax1.set_title("a", size='x-large', weight='bold', loc="left")
ax1.set_ylabel('Gini Coefficient', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.set_ylim(0.0, 1.0)

location_handles = [
    Line2D([0], [0], marker=markers[idx % len(markers)], color='black',
           linestyle='-',
           markersize=10 if loc == 'Coastal Med' else 8,
           linewidth=2 if loc == 'Coastal Med' else 1,
           label=loc)
    for idx, loc in enumerate(gini_dict.keys())
]
experiment_handles = [
    Line2D([0], [0], color='#1B4F72', linewidth=2, label='Present'),
    Line2D([0], [0], color='#CB4335', linewidth=2, label='Future'),
]
all_handles = location_handles + experiment_handles
ax1.legend(handles=all_handles, labels=[h.get_label() for h in all_handles],
           loc='lower left', frameon=False, ncol=2)

# ── Bottom panel: Future − Present ────────────────────────────────────────────
coastal_diff_lower = coastal_ci['fut_lower'] - coastal_ci['pres_upper']
coastal_diff_upper = coastal_ci['fut_upper'] - coastal_ci['pres_lower']
ax2.fill_between(x_vals, coastal_diff_lower, coastal_diff_upper,
                 color='#4A2C4E', alpha=0.25, linewidth=0, label='95% CI')

for idx, (location, gini_values) in enumerate(gini_dict.items()):
    marker = markers[idx % len(markers)]
    if location == 'Coastal Med':
        markersize, linewidth, linestyle = 10, 2, '-'
        color_diff_loc = '#4A2C4E'
    else:
        markersize, linewidth, linestyle = 6, 0.75, '--'
        color_diff_loc = '#6B4E71'

    ax2.plot(x_vals, gini_values[1, :] - gini_values[0, :],
             marker=marker, color=color_diff_loc, linestyle=linestyle,
             linewidth=linewidth, markersize=markersize, alpha=0.7)

ax2.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)

significant_bins = []
for idx in range(n_bins):
    if coastal_diff_lower[idx] > 0 or coastal_diff_upper[idx] < 0:
        significant_bins.append(idx)
        ax2.fill_between([idx - 0.4, idx + 0.4],
                         [coastal_diff_lower[idx], coastal_diff_lower[idx]],
                         [coastal_diff_upper[idx], coastal_diff_upper[idx]],
                         color='none', edgecolor='#4A2C4E',
                         hatch='///', linewidth=0, alpha=0.8)

print(f"\nSignificant bins (p<0.05): {significant_bins}")
print(f"Total significant bins: {len(significant_bins)} out of {n_bins}")

ax2.set_title("b", size='x-large', weight='bold', loc="left")
ax2.set_xlabel('Daily Rainfall Bin Centre (mm)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Δ Gini Coefficient\n(Future − Present)', fontsize=12, fontweight='bold')
ax2.set_xticks(x_vals)
ax2.set_xticklabels(labels, rotation=90, fontsize=10)
ax2.grid(True, alpha=0.3, linestyle='--')

max_diff = np.nanmax([np.abs(gv[1] - gv[0]) for gv in gini_dict.values()])
ax2.set_ylim(-max_diff * 1.1, max_diff * 1.1)

plt.tight_layout()
plt.savefig('/home/dargueso/Analyses/EPICC/Hourly_vs_Daily/gini_coefficient_comparison.png',
            dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
