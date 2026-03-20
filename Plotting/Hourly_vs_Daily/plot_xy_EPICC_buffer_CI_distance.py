#!/usr/bin/env python
'''
@File    :  plot_xy_EPICC_buffer_CI_distance.py
@Time    :  2026/03/20
@Author  :  Daniel Argüeso
@Version :  1.0
@Contact :  d.argueso@uib.es
@License :  (C)Copyright 2025, Daniel Argüeso
@Project :  EPICC
@Desc    :  Diagnostic plot showing how far observed quantiles are from the
            centre (median) of the synthetic bootstrap distribution, with
            vertical error bars that span obs−CI_hi to obs−CI_lo.
            If the error bar crosses y=0 the difference is within the CI and
            cannot be considered significant.
            One figure per location, 2×2 subplots (present/future × hourly/daily-max).
'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os

mpl.rcParams["font.size"] = 12

###########################################################################
# Configuration
###########################################################################

LOCATIONS = ['Mallorca', 'Catania', 'Turis', 'Rosiglione',
             'Ardeche', 'Corte', "L'Aquila", 'Pyrenees']

# buf=0 will be skipped gracefully if the npz has not been generated yet
BUFFERS = [0, 1, 3, 5, 10, 15, 20]

PATH_NPZ = '/home/dargueso/Analyses/EPICC/Hourly_vs_Daily/testing_data/'
PATH_OUT = '/home/dargueso/Analyses/EPICC/Hourly_vs_Daily/'

os.makedirs(PATH_OUT, exist_ok=True)

# Visual properties per buffer — fixed offsets so bars don't overlap
MARKERS     = {0: 'D', 1: 'o', 3: 's', 5: '^', 10: 'P', 15:'*',20:'d'}
BUF_OFFSETS = {0: -0.24, 1: -0.12, 3: 0.0, 5: 0.12, 10: 0.24, 15:0.36, 20:0.48}

# Sequential colormap — one colour per buffer
CMAP = plt.get_cmap('plasma')
BUF_COLORS = {buf: CMAP(i / (len(BUFFERS) - 1)) for i, buf in enumerate(BUFFERS)}

###########################################################################
# Helper
###########################################################################

def ci_from_boot(boot, bq):
    """boot: (N_SAMPLES, n_q) → (3, n_q) with rows [lo, median, hi]."""
    out = np.full((3, boot.shape[1]), np.nan)
    for iq in range(boot.shape[1]):
        v = boot[:, iq]
        v = v[~np.isnan(v)]
        if len(v):
            out[:, iq] = np.quantile(v, bq)
    return out   # rows: [lo, median, hi]


def diff_and_errs(obs, ci):
    """
    Returns the signed difference from the bootstrap median and the
    asymmetric error magnitudes that span the full CI.

    diff   = obs − ci_median
    err_lo = ci_hi − ci_median   (magnitude below diff; keeps bar bottom = obs−ci_hi)
    err_hi = ci_median − ci_lo   (magnitude above diff; keeps bar top   = obs−ci_lo)

    If the error bar [diff−err_lo, diff+err_hi] crosses 0, obs is within the CI.
    """
    diff   = obs - ci[1]                 # obs − median
    err_lo = np.maximum(ci[2] - ci[1], 0.0)   # hi − median  (≥ 0)
    err_hi = np.maximum(ci[1] - ci[0], 0.0)   # median − lo  (≥ 0)
    return diff, err_lo, err_hi


###########################################################################
# Main loop over locations
###########################################################################

for location in LOCATIONS:
    print(f"\nProcessing: {location}")

    # Per-buffer: store (diff, err_lo, err_hi) tuples
    data_pres_h  = {}
    data_fut_h   = {}
    data_pres_dm = {}
    data_fut_dm  = {}

    plot_quantiles      = None
    bootstrap_quantiles = None

    for buf in BUFFERS:
        fname = os.path.join(PATH_NPZ, f'{location}_buf{buf}.npz')
        if not os.path.exists(fname):
            print(f"  Missing: {fname} — skipping buffer {buf}")
            continue

        d = np.load(fname)

        plot_quantiles      = d['plot_quantiles']       # (6,)
        bootstrap_quantiles = d['bootstrap_quantiles']  # (3,) [0.025, 0.5, 0.975]

        obs_pres_h_buf  = d['obs_pres_h_buf']
        obs_fut_h_buf   = d['obs_fut_h_buf']
        obs_pres_dm_buf = d['obs_pres_dm_buf']
        obs_fut_dm_buf  = d['obs_fut_dm_buf']

        bq = bootstrap_quantiles

        ci_pres_h  = ci_from_boot(d['syn_pres_c_h_boot_buf'], bq)
        ci_fut_h   = ci_from_boot(d['syn_fut_c_h_boot_buf'],  bq)
        ci_pres_dm = ci_from_boot(d['syn_pres_mx_boot_buf'],  bq)
        ci_fut_dm  = ci_from_boot(d['syn_fut_mx_boot_buf'],   bq)

        data_pres_h[buf]  = diff_and_errs(obs_pres_h_buf,  ci_pres_h)
        data_fut_h[buf]   = diff_and_errs(obs_fut_h_buf,   ci_fut_h)
        data_pres_dm[buf] = diff_and_errs(obs_pres_dm_buf, ci_pres_dm)
        data_fut_dm[buf]  = diff_and_errs(obs_fut_dm_buf,  ci_fut_dm)

    if plot_quantiles is None:
        print(f"  No data found for {location} — skipping.")
        continue

    # X-axis: integer indices with per-buffer offsets
    x_labels = [f"P{int(q * 100)}" if q * 100 == int(q * 100)
                else f"P{q * 100:.1f}"
                for q in plot_quantiles]
    x_idx = np.arange(len(plot_quantiles))

    ###########################################################################
    # Build figure
    ###########################################################################

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharey=False)
    fig.suptitle(location, fontsize=15, fontweight='bold', y=1.01)

    panel_info = [
        (axes[0, 0], data_pres_h,  'Present — Hourly intensity (Method C)'),
        (axes[0, 1], data_fut_h,   'Future — Hourly intensity (Method C)'),
        (axes[1, 0], data_pres_dm, 'Present — Daily-max (Method B)'),
        (axes[1, 1], data_fut_dm,  'Future — Daily-max (Method B)'),
    ]

    for ax, data_dict, title in panel_info:
        # y=0 reference — crosses here mean obs is within the CI
        ax.axhline(0, color='black', linewidth=1.8, linestyle='--', zorder=3)

        for buf in BUFFERS:
            if buf not in data_dict:
                continue

            diff, err_lo, err_hi = data_dict[buf]
            xpos = x_idx + BUF_OFFSETS[buf]

            ax.errorbar(xpos, diff,
                        yerr=[err_lo, err_hi],
                        fmt=MARKERS[buf],
                        color=BUF_COLORS[buf],
                        markersize=6,
                        linewidth=1.5,
                        elinewidth=1.5,
                        capsize=4,
                        capthick=1.5,
                        label=f'Buffer {buf}',
                        zorder=4)

        ax.set_xticks(x_idx)
        ax.set_xticklabels(x_labels, fontsize=10)
        ax.set_xlabel('Quantile', fontsize=11, fontweight='bold')
        ax.set_ylabel('Obs − synthetic median (mm)', fontsize=11, fontweight='bold')
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.grid(True, linestyle=':', linewidth=0.5, alpha=0.7)
        ax.set_axisbelow(True)
        ax.tick_params(axis='both', which='major', labelsize=10)

        ax.legend(frameon=True, fancybox=True, shadow=True,
                  fontsize=9, loc='upper left')

    fig.tight_layout()
    outfile = os.path.join(PATH_OUT, f'buffer_CI_distance_{location}.png')
    fig.savefig(outfile, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"  Saved: {outfile}")
    plt.close(fig)

print("\nDone.")
