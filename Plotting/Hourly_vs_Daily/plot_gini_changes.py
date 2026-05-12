#!/usr/bin/env python
"""
plot_gini_changes.py

Plot Gini coefficient vs. low-frequency rainfall bin for present and future
climate runs, for a set of locations and the Coastal Mediterranean domain.

Reads output from compute_gini.py:
  {PATH_IN}/{WRUN_PRESENT}/gini_{FREQ_HIGH}_given_{FREQ_LOW}.nc
  {PATH_IN}/{WRUN_FUTURE}/gini_{FREQ_HIGH}_given_{FREQ_LOW}.nc

Produces a two-panel figure:
  Top    — Gini coefficient (present + future) with 95% bootstrap CI for
           the Coastal Mediterranean mean
  Bottom — Future − Present Gini difference; hatching marks bins where the
           CI does not include zero (statistically significant)
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D

mpl.rcParams['font.size']        = 14
mpl.rcParams['hatch.color']      = 'red'
mpl.rcParams['hatch.linewidth']  = 0.8

# =============================================================================
# CONFIGURATION
# =============================================================================

PATH_IN      = '/home/dargueso/postprocessed/EPICC/'
WRUN_PRESENT = 'EPICC_2km_ERA5'
WRUN_FUTURE  = 'EPICC_2km_ERA5_CMIP6anom'

FREQ_HIGH = '01H'
FREQ_LOW  = 'DAY'

# Mediterranean mask (pixel value == 2 → coastal Mediterranean)
MASK_FILE  = '/home/dargueso/postprocessed/EPICC/EPICC_2km_ERA5/my_coastal_med_mask.nc'
MASK_VALUE = 2

# Spatial buffer around each point location (half-width in grid cells)
BUFFER = 10

# Number of bootstrap resamples for confidence intervals
N_BOOTSTRAP = 1000

# Output figure path
FIG_OUT = f'/home/dargueso/Analyses/EPICC/Hourly_vs_Daily/gini_changes_buf{BUFFER:02d}.png'

# Point locations  (y-index, x-index, label)
LOCATIONS = [
    (258,  559, 'Mallorca'),
    (250,  423, 'Turis'),
    (384,  569, 'Pyrenees'),
    (527,  795, 'Rosiglione'),
    (533,  638, 'Ardeche'),
    (407,  821, 'Corte'),
    (174, 1091, 'Catania'),
    (425,  989, "L'Aquila"),
]

# Colours and markers
COLOR_PRESENT = '#2E86AB'
COLOR_FUTURE  = '#E50C0C'
COLOR_COASTMED_PRES = '#1B4F72'
COLOR_COASTMED_FUT  = '#CB4335'
COLOR_DIFF          = '#4A2C4E'
COLOR_DIFF_LOC      = '#6B4E71'

MARKERS = ['o', 's', '^', 'v', 'D', 'P', '*', 'X', 'p', 'h', '<', '>', '8']

# =============================================================================
# HELPERS
# =============================================================================

def load_gini(wrun):
    path = f'{PATH_IN}/{wrun}/gini_{FREQ_HIGH}_given_{FREQ_LOW}_buf{BUFFER:02d}.nc'
    return xr.open_dataset(path)


def weighted_mean_gini(ds, mask=None):
    """
    Weighted spatial mean of gini_coefficient, using n_events as weights.

    Parameters
    ----------
    ds   : xr.Dataset  with 'gini_coefficient' and 'n_events'
    mask : 2-D boolean array (ny, nx), or None to use all points

    Returns
    -------
    xr.DataArray  (nbins,)
    """
    gini = ds['gini_coefficient']
    w    = ds['n_events']
    if mask is not None:
        gini = gini.where(mask)
        w    = w.where(mask)
    return gini.weighted(w.fillna(0)).mean(dim=['y', 'x'])


def bootstrap_weighted_mean(data_flat, weights_flat, n_bootstrap=N_BOOTSTRAP):
    """
    Bootstrap 95 % CI for a weighted spatial mean.

    Parameters
    ----------
    data_flat    : 1-D array  (valid pixels only)
    weights_flat : 1-D array  (same length)
    n_bootstrap  : int

    Returns
    -------
    ci_lower, ci_upper : float
    """
    rng = np.random.default_rng(42)
    n = len(data_flat)
    boot_means = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        w_b = weights_flat[idx]
        w_sum = w_b.sum()
        boot_means[i] = (np.dot(data_flat[idx], w_b) / w_sum
                         if w_sum > 0 else np.nan)
    return np.nanpercentile(boot_means, 2.5), np.nanpercentile(boot_means, 97.5)


def coastal_bootstrap_cis(ds_pres, ds_fut, coast_mask):
    """
    Compute per-bin bootstrap CIs for Coastal Med weighted-mean Gini.

    Returns
    -------
    dict with arrays: pres_lower, pres_upper, fut_lower, fut_upper
    """
    bin_coord = f'bin_{FREQ_LOW}'
    nbins = len(ds_pres[bin_coord])

    ci = {k: np.full(nbins, np.nan) for k in
          ('pres_lower', 'pres_upper', 'fut_lower', 'fut_upper')}

    for b in range(nbins):
        for prefix, ds in [('pres', ds_pres), ('fut', ds_fut)]:
            gini_b = ds['gini_coefficient'].isel({bin_coord: b}).where(coast_mask)
            w_b    = ds['n_events'].isel({bin_coord: b}).where(coast_mask)

            g_flat = gini_b.values.flatten()
            w_flat = w_b.values.flatten()
            valid  = ~np.isnan(g_flat) & ~np.isnan(w_flat) & (w_flat > 0)

            if valid.sum() > 0:
                lo, hi = bootstrap_weighted_mean(g_flat[valid], w_flat[valid])
                ci[f'{prefix}_lower'][b] = lo
                ci[f'{prefix}_upper'][b] = hi

    return ci


# =============================================================================
# MAIN
# =============================================================================

def main():
    # -------------------------------------------------------------------------
    # Load data
    # -------------------------------------------------------------------------
    print("Loading gini files …", flush=True)
    ds_pres = load_gini(WRUN_PRESENT)
    ds_fut  = load_gini(WRUN_FUTURE)

    bin_coord  = f'bin_{FREQ_LOW}'
    bin_edges  = ds_pres.attrs.get('bin_edges_low', ds_pres[bin_coord].values.tolist())
    bin_edges  = np.array(bin_edges)
    nbins      = len(bin_edges)
    x_vals     = np.arange(nbins)

    # Bin labels for x-axis
    labels = [f'{lo:.4g}–{hi:.4g}'
              for lo, hi in zip(bin_edges[:-1], bin_edges[1:])]
    labels.append(f'>{bin_edges[-1]:.4g}')

    # Mediterranean mask
    med_mask_ds = xr.open_dataset(MASK_FILE)
    coast_mask  = med_mask_ds['combined_mask'].values == MASK_VALUE

    # -------------------------------------------------------------------------
    # Compute weighted-mean Gini per location
    # -------------------------------------------------------------------------
    print("Computing location means …", flush=True)
    gini_dict = {}

    for yloc, xloc, name in LOCATIONS:
        fin_p = ds_pres.isel(y=slice(yloc - BUFFER, yloc + BUFFER + 1),
                              x=slice(xloc - BUFFER, xloc + BUFFER + 1))
        fin_f = ds_fut.isel( y=slice(yloc - BUFFER, yloc + BUFFER + 1),
                              x=slice(xloc - BUFFER, xloc + BUFFER + 1))
        gini_dict[name] = np.stack([
            weighted_mean_gini(fin_p).values,
            weighted_mean_gini(fin_f).values,
        ])   # (2, nbins)

    # Coastal Mediterranean
    gini_dict['Coastal Med'] = np.stack([
        weighted_mean_gini(ds_pres, mask=coast_mask).values,
        weighted_mean_gini(ds_fut,  mask=coast_mask).values,
    ])

    # -------------------------------------------------------------------------
    # Bootstrap CIs for Coastal Med
    # -------------------------------------------------------------------------
    print(f"Bootstrap CIs ({N_BOOTSTRAP} resamples) …", flush=True)
    ci = coastal_bootstrap_cis(ds_pres, ds_fut, coast_mask)

    print("\nBootstrap CI summary:")
    print(f"  Present mean:  {gini_dict['Coastal Med'][0, :]}")
    print(f"  Present CI lo: {ci['pres_lower']}")
    print(f"  Present CI hi: {ci['pres_upper']}")
    print(f"  Future  mean:  {gini_dict['Coastal Med'][1, :]}")
    print(f"  Future  CI lo: {ci['fut_lower']}")
    print(f"  Future  CI hi: {ci['fut_upper']}")

    # -------------------------------------------------------------------------
    # Figure
    # -------------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12, 10), sharex=True,
        gridspec_kw={'height_ratios': [2, 1], 'hspace': 0.15}
    )

    # === TOP PANEL: Gini coefficient =========================================
    # Shaded CI bands (behind lines)
    ax1.fill_between(x_vals, ci['pres_lower'], ci['pres_upper'],
                     color=COLOR_COASTMED_PRES, alpha=0.25, linewidth=0,
                     label='95% CI (Present)')
    ax1.fill_between(x_vals, ci['fut_lower'], ci['fut_upper'],
                     color=COLOR_COASTMED_FUT, alpha=0.25, linewidth=0,
                     label='95% CI (Future)')

    for idx, (name, gini_vals) in enumerate(gini_dict.items()):
        marker = MARKERS[idx % len(MARKERS)]
        if name == 'Coastal Med':
            kw_pres = dict(color=COLOR_COASTMED_PRES, linewidth=2,
                           linestyle='-', markersize=10)
            kw_fut  = dict(color=COLOR_COASTMED_FUT,  linewidth=2,
                           linestyle='-', markersize=10)
        else:
            kw_pres = dict(color=COLOR_PRESENT, linewidth=0.75,
                           linestyle='--', markersize=6)
            kw_fut  = dict(color=COLOR_FUTURE,  linewidth=0.75,
                           linestyle='--', markersize=6)

        ax1.plot(x_vals, gini_vals[0, :], marker=marker, alpha=0.7, **kw_pres)
        ax1.plot(x_vals, gini_vals[1, :], marker=marker, alpha=0.7, **kw_fut)

    ax1.set_title('a', size='x-large', weight='bold', loc='left')
    ax1.set_ylabel('Gini Coefficient', fontsize=12, fontweight='bold')
    ax1.set_ylim(0.4, 1.0)
    ax1.grid(True, alpha=0.3, linestyle='--')

    # Legend: locations (black markers) + experiments (coloured lines)
    loc_handles = [
        Line2D([0], [0], marker=MARKERS[i % len(MARKERS)], color='black',
               linestyle='-',
               markersize=10 if name == 'Coastal Med' else 8,
               linewidth=2  if name == 'Coastal Med' else 1,
               label=name)
        for i, name in enumerate(gini_dict)
    ]
    exp_handles = [
        Line2D([0], [0], color=COLOR_COASTMED_PRES, linewidth=2, label='Present'),
        Line2D([0], [0], color=COLOR_COASTMED_FUT,  linewidth=2, label='Future'),
    ]
    ax1.legend(handles=loc_handles + exp_handles,
               loc='lower left', frameon=False, ncol=2)

    # === BOTTOM PANEL: Future − Present difference ===========================
    diff_lower = ci['fut_lower'] - ci['pres_upper']
    diff_upper = ci['fut_upper'] - ci['pres_lower']

    # Shaded CI band (behind lines)
    ax2.fill_between(x_vals, diff_lower, diff_upper,
                     color=COLOR_DIFF, alpha=0.25, linewidth=0, label='95% CI')

    for idx, (name, gini_vals) in enumerate(gini_dict.items()):
        marker = MARKERS[idx % len(MARKERS)]
        diff   = gini_vals[1, :] - gini_vals[0, :]
        if name == 'Coastal Med':
            kw = dict(color=COLOR_DIFF, linewidth=2,
                      linestyle='-', markersize=10)
        else:
            kw = dict(color=COLOR_DIFF_LOC, linewidth=0.75,
                      linestyle='--', markersize=6)
        ax2.plot(x_vals, diff, marker=marker, alpha=0.7, **kw)

    ax2.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)

    # Hatching where CI excludes zero (statistically significant)
    significant_bins = []
    for b in range(nbins):
        if diff_lower[b] > 0 or diff_upper[b] < 0:
            significant_bins.append(b)
            ax2.fill_between(
                [b - 0.4, b + 0.4],
                [diff_lower[b], diff_lower[b]],
                [diff_upper[b], diff_upper[b]],
                color='none', edgecolor=COLOR_DIFF,
                hatch='///', linewidth=0, alpha=0.8
            )
    print(f"\nSignificant bins (CI excludes 0): {significant_bins} "
          f"({len(significant_bins)}/{nbins})")

    ax2.set_title('b', size='x-large', weight='bold', loc='left')
    ax2.set_xlabel(f'{FREQ_LOW} Rainfall Bins (mm)', fontsize=12, fontweight='bold')
    ax2.set_ylabel(f'Δ Gini Coefficient\n(Future − Present)',
                   fontsize=12, fontweight='bold')
    ax2.set_xticks(x_vals)
    ax2.set_xticklabels(labels, rotation=90, fontsize=9)
    ax2.grid(True, alpha=0.3, linestyle='--')

    max_diff = max(np.nanmax(np.abs(gini_vals[1, :] - gini_vals[0, :]))
                   for gini_vals in gini_dict.values())
    ax2.set_ylim(-max_diff * 1.1, max_diff * 1.1)

    # -------------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------------
    plt.tight_layout()
    import os
    os.makedirs(os.path.dirname(FIG_OUT), exist_ok=True)
    plt.savefig(FIG_OUT, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"\nFigure saved → {FIG_OUT}")

    ds_pres.close()
    ds_fut.close()
    med_mask_ds.close()


if __name__ == '__main__':
    main()
