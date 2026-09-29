#!/usr/bin/env python
'''
@File    :  plot_xy_EPICC_buffer_CI_distance_ctr_vs_syn.py
@Time    :  2026/06/02
@Author  :  Daniel Argüeso
@Version :  1.0
@Contact :  d.argueso@uib.es
@License :  (C)Copyright 2025, Daniel Argüeso
@Project :  EPICC
@Desc    :  CI-distance diagnostic for present (CTL) only. Two figures:

            Figure 1 — WRF central pixel vs Synthetic(buffer):
              obs from buf=0 is fixed; synthetic CI varies by buffer size.
              Tests how much spatial pooling is needed to recover the
              observed centre-pixel value.

            Figure 2 — WRF(buffer) vs Synthetic(buffer):
              Both WRF observation and synthetic CI use the same buffer.
              Self-consistency check at each spatial scale.

            Layout: 4 rows × 2 columns (one panel per location, 8 total).
'''

import string
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec
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

BOOTSTRAP_QUANTILES = np.array([0.005, 0.5, 0.995])

MARKERS     = {0: 'D', 1: 'o', 3: 's', 5: '^', 10: 'P', 15: '*', 20: 'd'}
BUF_OFFSETS = {0: -0.24, 1: -0.16, 3: -0.08, 5: 0.0, 10: 0.08, 15: 0.16, 20: 0.24}
CMAP        = plt.get_cmap('plasma')
BUF_COLORS  = {buf: CMAP(i / (len(BUFFERS) - 1)) for i, buf in enumerate(BUFFERS)}

###########################################################################
# Helpers
###########################################################################

def ci_from_boot(boot, bq=BOOTSTRAP_QUANTILES):
    """boot: (N_SAMPLES, n_q) → (3, n_q) rows [lo, median, hi]."""
    return np.nanquantile(boot, bq, axis=0)


def make_diff(obs, boot):
    ci     = ci_from_boot(boot)
    diff   = obs - ci[1]
    err_lo = np.maximum(ci[2] - ci[1], 0.0)
    err_hi = np.maximum(ci[1] - ci[0], 0.0)
    return diff, err_lo, err_hi


###########################################################################
# Load data
###########################################################################

# ctr_data[loc][buf]    = (diff, err_lo, err_hi)  obs=central pixel, syn=buf
# paired_data[loc][buf] = (diff, err_lo, err_hi)  obs=buf,           syn=buf

ctr_data    = {}
paired_data = {}
plot_quantiles = None

for location in LOCATIONS:
    print(f'\nLoading: {location}')

    fname_ctr = os.path.join(PATH_NPZ, f'{location}_buf0.npz')
    if not os.path.exists(fname_ctr):
        print(f'  Missing central-pixel NPZ — skipping location')
        ctr_data[location] = {}
        paired_data[location] = {}
        continue

    d0 = np.load(fname_ctr)
    obs_ctr = d0['obs_pres_h_buf']
    if plot_quantiles is None:
        plot_quantiles = d0['plot_quantiles']

    ctr_buf  = {}
    pair_buf = {}
    for buf in BUFFERS:
        fname = os.path.join(PATH_NPZ, f'{location}_buf{buf}.npz')
        if not os.path.exists(fname):
            print(f'  Missing NPZ for buf={buf}')
            continue
        d = np.load(fname)
        boot = d['syn_pres_c_h_boot_buf']
        obs_b = d['obs_pres_h_buf']

        ctr_buf[buf]  = make_diff(obs_ctr, boot)
        pair_buf[buf] = make_diff(obs_b,   boot)

    ctr_data[location]    = ctr_buf
    paired_data[location] = pair_buf

if plot_quantiles is None:
    raise RuntimeError('No data found — check PATH_NPZ.')

x_labels = [f'P{int(q * 100)}' if q * 100 == int(q * 100)
            else f'P{q * 100:.1f}'
            for q in plot_quantiles]
x_idx = np.arange(len(plot_quantiles))

###########################################################################
# Draw figure helper
###########################################################################

def draw_figure(dataset, ylabel, outfile):
    fig = plt.figure(figsize=(12, 4.5 * 4))
    gs  = GridSpec(4, 2, figure=fig,
                   hspace=0.30, wspace=0.30,
                   left=0.09, right=0.97, top=0.97, bottom=0.10)

    for loc_idx, loc in enumerate(LOCATIONS):
        row, col = divmod(loc_idx, 2)
        ax = fig.add_subplot(gs[row, col])

        buf_data = dataset.get(loc, {})
        if not buf_data:
            ax.set_visible(False)
            continue

        extents = []
        for diff, err_lo, err_hi in buf_data.values():
            extents.append(float(np.nanmax(np.abs(diff) + np.maximum(err_lo, err_hi))))

        ax.axhline(0.0, color='black', linewidth=1.4, linestyle='--', zorder=3, alpha=0.7)

        for buf in BUFFERS:
            if buf not in buf_data:
                continue
            diff, err_lo, err_hi = buf_data[buf]
            ax.errorbar(x_idx + BUF_OFFSETS[buf], diff,
                        yerr=[err_lo, err_hi],
                        fmt=MARKERS[buf], color=BUF_COLORS[buf],
                        markersize=5, linewidth=1.2,
                        elinewidth=1.2, capsize=3, capthick=1.2,
                        zorder=4)

        ax.set_xticks(x_idx)
        if row == 3:
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
        ax.set_yticks(yticks)
        ax.set_yticklabels([f'{int(v)}' for v in yticks], fontsize=10)

        ax.grid(True, axis='x', linestyle=':', linewidth=0.5, alpha=0.7)
        ax.set_axisbelow(True)
        ax.tick_params(axis='both', which='major', labelsize=10)

        ax.set_title(string.ascii_lowercase[loc_idx],
                     size='x-large', weight='bold', loc='left')
        ax.text(0.02, 0.97, loc, transform=ax.transAxes,
                fontsize=13, fontweight='bold', va='top', ha='left')

        if col == 0:
            ax.set_ylabel(ylabel, fontsize=11, fontweight='bold')

    legend_handles = [
        Line2D([0], [0], marker=MARKERS[buf], color=BUF_COLORS[buf],
               linestyle='none', markersize=7, label=f'buf={buf}')
        for buf in BUFFERS
    ]
    fig.legend(handles=legend_handles, loc='lower center', ncol=7,
               fontsize=11, frameon=False, bbox_to_anchor=(0.5, 0.01))

    fig.savefig(outfile, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'Saved: {outfile}')


###########################################################################
# Produce figures
###########################################################################

draw_figure(
    ctr_data,
    ylabel  = 'WRF − Synthetic (mm hour$^{-1}$)',
    outfile = os.path.join(PATH_OUT, 'buffer_CI_distance_ctr_vs_syn_present.png'),
)

draw_figure(
    paired_data,
    ylabel  = 'WRF − Synthetic (mm hour$^{-1}$)',
    outfile = os.path.join(PATH_OUT, 'buffer_CI_distance_buf_vs_syn_present.png'),
)

print('Done.')
