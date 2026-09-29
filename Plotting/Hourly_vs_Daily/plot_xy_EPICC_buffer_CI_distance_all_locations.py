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
            Produces two figures, each 4 rows × 2 columns (one panel per location).

              Figure 1: Hourly intensity     (Method C, {loc}_buf{N}.npz)
              Figure 2: 10-min from daily    (Method D, {loc}_buf{N}_10min.npz)

            Each panel shows present (filled markers, left group) and future
            (open markers, right group) side by side at each quantile.
            Error bars span obs − CI_hi to obs − CI_lo; crossing y=0 means
            the observed value lies within the synthetic CI.
'''

import string
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import os

mpl.rcParams['font.size'] = 14

###########################################################################
# Configuration
###########################################################################

LOCATIONS = ['Mallorca', 'Catania', 'Turis', 'Rossiglione',
             'Ardeche', 'Corte', "L'Aquila", 'Pyrenees']

BUFFERS = [0, 1, 3, 5, 10, 15, 20]

PATH_NPZ = '/home/dargueso/Analyses/EPICC/Hourly_vs_Daily/testing_data/'
PATH_OUT = '/home/dargueso/Analyses/EPICC/Hourly_vs_Daily/'

os.makedirs(PATH_OUT, exist_ok=True)

# CI level recomputed on the fly from raw bootstrap samples
BOOTSTRAP_QUANTILES = np.array([0.005, 0.5, 0.995])

MARKERS        = {0: 'D', 1: 'o', 3: 's', 5: '^', 10: 'P', 15: '*', 20: 'd'}
BUF_OFFSETS    = {0: -0.24, 1: -0.16, 3: -0.08, 5: 0.0, 10: 0.08, 15: 0.16, 20: 0.24}
CMAP           = plt.get_cmap('plasma')
BUF_COLORS     = {buf: CMAP(i / (len(BUFFERS) - 1)) for i, buf in enumerate(BUFFERS)}

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

def draw_half(ax, data_dict, period_label='', show_xlabel=False,
              drop_bottom=False, drop_top=False):
    """Plot one period (present or future) into a single axis, centred at y=0."""
    extents = []
    for diff, err_lo, err_hi in data_dict.values():
        extents.append(float(np.nanmax(np.abs(diff) + np.maximum(err_lo, err_hi))))

    ax.axhline(0.0, color='black', linewidth=1.4, linestyle='--', zorder=3, alpha=0.7)

    for buf in BUFFERS:
        if buf not in data_dict:
            continue
        diff, err_lo, err_hi = data_dict[buf]
        ax.errorbar(x_idx + BUF_OFFSETS[buf], diff,
                    yerr=[err_lo, err_hi],
                    fmt=MARKERS[buf], color=BUF_COLORS[buf],
                    markersize=5, linewidth=1.2,
                    elinewidth=1.2, capsize=3, capthick=1.2,
                    zorder=4)

    ax.set_xticks(x_idx)
    if show_xlabel:
        ax.set_xticklabels(x_labels, fontsize=10, rotation=45, ha='right')
        ax.set_xlabel('Quantile', fontsize=11)
    else:
        ax.set_xticklabels([])
    ax.set_xlim(x_idx[0] - 0.4, x_idx[-1] + 0.4)

    max_ext = max(extents) if extents else 2.0
    for step in [0.5, 1.0, 2.0, 5.0, 10.0]:
        if max_ext / step <= 3:
            break
    n = int(np.ceil(max_ext / step))
    yticks = np.arange(-n * step, n * step + step / 2, step)
    if drop_bottom:
        yticks = yticks[1:]
    if drop_top:
        yticks = yticks[:-1]
    ax.set_yticks(yticks)
    ax.set_yticklabels([f'{int(v)}' for v in yticks], fontsize=10)

    ax.grid(True, axis='x', linestyle=':', linewidth=0.5, alpha=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(axis='both', which='major', labelsize=10)

    if period_label:
        ax.text(0.99, 0.97, period_label, transform=ax.transAxes,
                fontsize=11, color='gray', va='top', ha='right', fontstyle='italic')


def draw_panel(ax_pres, ax_fut, pres_dict, fut_dict,
               loc_name='', panel_letter='', show_xlabel=True):
    draw_half(ax_pres, pres_dict, period_label='Present (CTL)', show_xlabel=False,
              drop_bottom=True)
    draw_half(ax_fut,  fut_dict,  period_label='Future (PGW)',  show_xlabel=show_xlabel,
              drop_top=True)

    if panel_letter:
        ax_pres.set_title(panel_letter, size='x-large', weight='bold', loc='left')
    if loc_name:
        ax_pres.text(0.02, 0.95, loc_name, transform=ax_pres.transAxes,
                     fontsize=13, fontweight='bold', va='top', ha='left')

    # Grey separator line at the boundary between the two halves
    ax_pres.spines['bottom'].set_color('#888888')
    ax_pres.spines['bottom'].set_linewidth(2.0)
    ax_fut.spines['top'].set_visible(False)


###########################################################################
# Figure 1: Hourly intensity
###########################################################################

fig_C = plt.figure(figsize=(12, 5.0 * 4))

outer_gs_C = GridSpec(4, 2, figure=fig_C,
                      hspace=0.18, wspace=0.30,
                      left=0.09, right=0.97, top=0.97, bottom=0.10)

for loc_idx, loc in enumerate(LOCATIONS):
    row, col = divmod(loc_idx, 2)
    inner_gs = GridSpecFromSubplotSpec(2, 1,
                                       subplot_spec=outer_gs_C[row, col],
                                       hspace=0)
    ax_pres = fig_C.add_subplot(inner_gs[0])
    ax_fut  = fig_C.add_subplot(inner_gs[1], sharex=ax_pres)

    draw_panel(ax_pres, ax_fut,
               all_data[loc]['C']['pres'], all_data[loc]['C']['fut'],
               loc_name=loc,
               panel_letter=string.ascii_lowercase[loc_idx],
               show_xlabel=(row == 3))

    if col == 0:
        ax_pres.set_ylabel('WRF - Synthetic (mm hour$^{-1}$)',
                           fontsize=11, fontweight='bold', y=0, va='center')

legend_handles = [
    Line2D([0], [0], marker=MARKERS[buf], color=BUF_COLORS[buf],
           linestyle='none', markersize=7, label=f'buf={buf}')
    for buf in BUFFERS
]
fig_C.legend(handles=legend_handles, loc='lower center', ncol=7,
             fontsize=11, frameon=False, bbox_to_anchor=(0.5, 0.01))

out_C = os.path.join(PATH_OUT, 'buffer_CI_distance_hourly_all_locations.png')
fig_C.savefig(out_C, dpi=150, bbox_inches='tight', facecolor='white')
plt.close(fig_C)
print(f'\nSaved: {out_C}')

###########################################################################
# Figure 2: 10-min from daily
###########################################################################

fig_D = plt.figure(figsize=(12, 5.0 * 4))

outer_gs_D = GridSpec(4, 2, figure=fig_D,
                      hspace=0.18, wspace=0.30,
                      left=0.09, right=0.97, top=0.97, bottom=0.10)

for loc_idx, loc in enumerate(LOCATIONS):
    row, col = divmod(loc_idx, 2)
    inner_gs = GridSpecFromSubplotSpec(2, 1,
                                       subplot_spec=outer_gs_D[row, col],
                                       hspace=0)
    ax_pres = fig_D.add_subplot(inner_gs[0])
    ax_fut  = fig_D.add_subplot(inner_gs[1], sharex=ax_pres)

    draw_panel(ax_pres, ax_fut,
               all_data[loc]['D']['pres'], all_data[loc]['D']['fut'],
               loc_name=loc,
               panel_letter=string.ascii_lowercase[loc_idx],
               show_xlabel=(row == 3))

    if col == 0:
        ax_pres.set_ylabel('WRF - Synthetic (mm hour$^{-1}$)',
                           fontsize=11, fontweight='bold', y=0, va='center')

fig_D.legend(handles=legend_handles, loc='lower center', ncol=7,
             fontsize=11, frameon=False, bbox_to_anchor=(0.5, 0.01))

out_D = os.path.join(PATH_OUT, 'buffer_CI_distance_10min_daily_all_locations.png')
fig_D.savefig(out_D, dpi=150, bbox_inches='tight', facecolor='white')
plt.close(fig_D)
print(f'\nSaved: {out_D}')

print('\nDone.')
