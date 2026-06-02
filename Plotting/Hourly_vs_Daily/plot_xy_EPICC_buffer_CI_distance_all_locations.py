#!/usr/bin/env python
'''
@File    :  plot_xy_EPICC_buffer_CI_distance_all_locations.py
@Time    :  2026/04/30
@Author  :  Daniel Argüeso
@Version :  1.0
@Contact :  d.argueso@uib.es
@License :  (C)Copyright 2025, Daniel Argüeso
@Project :  EPICC
@Desc    :  CI-distance diagnostic across all locations simultaneously.
            Produces two figures, each 2 rows × N_locations columns:

              Figure 1: Hourly intensity     (Method C, {loc}_buf{N}.npz)
              Figure 2: 10-min from daily    (Method D, {loc}_buf{N}_10min.npz)

            Rows: Row 0 = present, Row 1 = future.
            Columns: one per location.
            Error bars span obs − CI_hi to obs − CI_lo; crossing y=0 means
            the observed value lies within the synthetic CI.
'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os

mpl.rcParams['font.size'] = 12

###########################################################################
# Configuration
###########################################################################

LOCATIONS = ['Mallorca', 'Catania', 'Turis', 'Rosiglione',
             'Ardeche', 'Corte', "L'Aquila", 'Pyrenees']

BUFFERS = [0, 1, 3, 5, 10, 15, 20]

PATH_NPZ = '/home/dargueso/Analyses/EPICC/Hourly_vs_Daily/testing_data/'
PATH_OUT = '/home/dargueso/Analyses/EPICC/Hourly_vs_Daily/'

os.makedirs(PATH_OUT, exist_ok=True)

# CI level recomputed on the fly from raw bootstrap samples
BOOTSTRAP_QUANTILES = np.array([0.005, 0.5, 0.995])

MARKERS     = {0: 'D', 1: 'o', 3: 's', 5: '^', 10: 'P', 15: '*', 20: 'd'}
BUF_OFFSETS = {0: -0.24, 1: -0.16, 3: -0.08, 5: 0.0, 10: 0.08, 15: 0.16, 20: 0.24}
CMAP        = plt.get_cmap('plasma')
BUF_COLORS  = {buf: CMAP(i / (len(BUFFERS) - 1)) for i, buf in enumerate(BUFFERS)}

###########################################################################
# Helpers
###########################################################################

def ci_from_boot(boot, bq):
    """boot: (N_SAMPLES, n_q) → (3, n_q) rows [lo, median, hi]."""
    return np.nanquantile(boot, bq, axis=0)


def diff_and_errs(obs, ci):
    """Signed difference obs−median and asymmetric CI error magnitudes."""
    diff   = obs - ci[1]
    err_lo = np.maximum(ci[2] - ci[1], 0.0)   # hi − median  (upper bar)
    err_hi = np.maximum(ci[1] - ci[0], 0.0)   # median − lo  (lower bar)
    return diff, err_lo, err_hi


###########################################################################
# Load data for all locations
###########################################################################

all_data     = {}   # all_data[loc] = {'C': {'pres'/{buf:...}, 'fut'/{buf:...}},
                    #                   'D': {...}}
plot_quantiles = None

for location in LOCATIONS:
    print(f'\nLoading: {location}')
    loc_data = {k: {'pres': {}, 'fut': {}} for k in ('C', 'D')}

    for buf in BUFFERS:
        fname_h   = os.path.join(PATH_NPZ, f'{location}_buf{buf}.npz')
        fname_10m = os.path.join(PATH_NPZ, f'{location}_buf{buf}_10min.npz')

        # Method C — hourly
        if os.path.exists(fname_h):
            d = np.load(fname_h)
            if plot_quantiles is None:
                plot_quantiles = d['plot_quantiles']
            ci_pres = ci_from_boot(d['syn_pres_c_h_boot_buf'], BOOTSTRAP_QUANTILES)
            ci_fut  = ci_from_boot(d['syn_fut_c_h_boot_buf'],  BOOTSTRAP_QUANTILES)
            loc_data['C']['pres'][buf] = diff_and_errs(d['obs_pres_h_buf'], ci_pres)
            loc_data['C']['fut'][buf]  = diff_and_errs(d['obs_fut_h_buf'],  ci_fut)
        else:
            print(f'  Missing hourly NPZ for buf={buf}')

        # Method D — 10-min from daily
        if os.path.exists(fname_10m):
            m = np.load(fname_10m)
            if plot_quantiles is None:
                plot_quantiles = m['plot_quantiles']
            ci_pres = ci_from_boot(m['D_pres_boot_buf'], BOOTSTRAP_QUANTILES)
            ci_fut  = ci_from_boot(m['D_fut_boot_buf'],  BOOTSTRAP_QUANTILES)
            loc_data['D']['pres'][buf] = diff_and_errs(m['obs_pres_10m_buf'], ci_pres)
            loc_data['D']['fut'][buf]  = diff_and_errs(m['obs_fut_10m_buf'],  ci_fut)
        else:
            print(f'  Missing 10min NPZ for buf={buf}')

    all_data[location] = loc_data

if plot_quantiles is None:
    raise RuntimeError('No data found — check PATH_NPZ.')

x_labels = [f'P{int(q * 100)}' if q * 100 == int(q * 100)
            else f'P{q * 100:.1f}'
            for q in plot_quantiles]
x_idx  = np.arange(len(plot_quantiles))
cl_lo  = BOOTSTRAP_QUANTILES[0]
cl_hi  = BOOTSTRAP_QUANTILES[2]
n_locs = len(LOCATIONS)

###########################################################################
# Helper: populate one panel
###########################################################################

def draw_panel(ax, data_dict, show_legend=False):
    ax.axhline(0, color='black', linewidth=1.6, linestyle='--', zorder=3)
    for buf in BUFFERS:
        if buf not in data_dict:
            continue
        diff, err_lo, err_hi = data_dict[buf]
        xpos = x_idx + BUF_OFFSETS[buf]
        ax.errorbar(xpos, diff,
                    yerr=[err_lo, err_hi],
                    fmt=MARKERS[buf],
                    color=BUF_COLORS[buf],
                    markersize=5, linewidth=1.2,
                    elinewidth=1.2, capsize=3, capthick=1.2,
                    label=f'buf={buf}', zorder=4)
    ax.set_xticks(x_idx)
    ax.set_xticklabels(x_labels, fontsize=8, rotation=45, ha='right')
    ax.set_xlabel('Quantile', fontsize=9)
    ax.grid(True, linestyle=':', linewidth=0.5, alpha=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(axis='both', which='major', labelsize=8)
    if show_legend:
        ax.legend(frameon=True, fontsize=7, loc='upper left',
                  ncol=1, handlelength=1.0, borderpad=0.4)


###########################################################################
# Figure 1: Method C — Hourly intensity
###########################################################################

fig_C, axes_C = plt.subplots(2, n_locs,
                              figsize=(4.2 * n_locs, 8),
                              sharey='row')
fig_C.suptitle(
    f'Hourly intensity (Method C) — CI: {cl_lo}–{cl_hi}',
    fontsize=14, fontweight='bold', y=1.01)

for col, loc in enumerate(LOCATIONS):
    axes_C[0, col].set_title(loc, fontsize=10, fontweight='bold')
    draw_panel(axes_C[0, col], all_data[loc]['C']['pres'],
               show_legend=(col == 0))
    draw_panel(axes_C[1, col], all_data[loc]['C']['fut'],
               show_legend=False)

axes_C[0, 0].set_ylabel('Present\nObs − syn median (mm/h)', fontsize=9, fontweight='bold')
axes_C[1, 0].set_ylabel('Future\nObs − syn median (mm/h)',  fontsize=9, fontweight='bold')

fig_C.tight_layout()
out_C = os.path.join(PATH_OUT, 'buffer_CI_distance_hourly_all_locations.png')
fig_C.savefig(out_C, dpi=150, bbox_inches='tight', facecolor='white')
plt.close(fig_C)
print(f'\nSaved: {out_C}')

###########################################################################
# Figure 2: Method D — 10-min from daily
###########################################################################

fig_D, axes_D = plt.subplots(2, n_locs,
                              figsize=(4.2 * n_locs, 8),
                              sharey='row')
fig_D.suptitle(
    f'10-min from daily (Method D) — CI: {cl_lo}–{cl_hi}',
    fontsize=14, fontweight='bold', y=1.01)

for col, loc in enumerate(LOCATIONS):
    axes_D[0, col].set_title(loc, fontsize=10, fontweight='bold')
    draw_panel(axes_D[0, col], all_data[loc]['D']['pres'],
               show_legend=(col == 0))
    draw_panel(axes_D[1, col], all_data[loc]['D']['fut'],
               show_legend=False)

axes_D[0, 0].set_ylabel('Present\nObs − syn median (mm/h)', fontsize=9, fontweight='bold')
axes_D[1, 0].set_ylabel('Future\nObs − syn median (mm/h)',  fontsize=9, fontweight='bold')

fig_D.tight_layout()
out_D = os.path.join(PATH_OUT, 'buffer_CI_distance_10min_daily_all_locations.png')
fig_D.savefig(out_D, dpi=150, bbox_inches='tight', facecolor='white')
plt.close(fig_D)
print(f'\nSaved: {out_D}')

print('\nDone.')
