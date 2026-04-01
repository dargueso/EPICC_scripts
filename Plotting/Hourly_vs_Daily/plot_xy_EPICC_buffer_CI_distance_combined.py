#!/usr/bin/env python
'''
@File    :  plot_xy_EPICC_buffer_CI_distance_combined.py
@Time    :  2026/03/31
@Author  :  Daniel Argüeso
@Version :  1.0
@Contact :  d.argueso@uib.es
@License :  (C)Copyright 2025, Daniel Argüeso
@Project :  EPICC
@Desc    :  Combined CI-distance diagnostic plot merging hourly and 10-min
            pipeline outputs.  One figure per location, 2 rows × 3 columns:

              Row 0 (present) | Row 1 (future)
              Col 0 — Hourly intensity      (Method C, from {loc}_buf{N}.npz)
              Col 1 — 10-min from daily     (Method D, from {loc}_buf{N}_10min.npz)
              Col 2 — 10-min from hourly    (Method E, from {loc}_buf{N}_10min.npz)

            Error bars span obs − CI_hi to obs − CI_lo.
            If a bar crosses y=0 the observed change is within the synthetic CI.
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

BUFFERS = [0, 1, 3, 5, 10, 15, 20]

PATH_NPZ = '/home/dargueso/Analyses/EPICC/Hourly_vs_Daily/testing_data/'
PATH_OUT = '/home/dargueso/Analyses/EPICC/Hourly_vs_Daily/'

os.makedirs(PATH_OUT, exist_ok=True)

# CI level — recomputed on the fly from raw bootstrap samples.
# e.g. [0.025, 0.5, 0.975] = 95%,  [0.005, 0.5, 0.995] = 99%
BOOTSTRAP_QUANTILES = np.array([0.005, 0.5, 0.995])

# Visual properties per buffer
MARKERS     = {0: 'D', 1: 'o', 3: 's', 5: '^', 10: 'P', 15: '*', 20: 'd'}
BUF_OFFSETS = {0: -0.24, 1: -0.12, 3: 0.0, 5: 0.12, 10: 0.24, 15: 0.36, 20: 0.48}
CMAP        = plt.get_cmap('plasma')
BUF_COLORS  = {buf: CMAP(i / (len(BUFFERS) - 1)) for i, buf in enumerate(BUFFERS)}

###########################################################################
# Helpers
###########################################################################

def ci_from_boot(boot, bq):
    """boot: (N_SAMPLES, n_q) → (3, n_q) with rows [lo, median, hi]."""
    return np.nanquantile(boot, bq, axis=0)


def diff_and_errs(obs, ci):
    """
    Signed difference from bootstrap median + asymmetric CI error magnitudes.
    If the error bar [diff−err_lo, diff+err_hi] crosses 0, obs is within the CI.
    """
    diff   = obs - ci[1]
    err_lo = np.maximum(ci[2] - ci[1], 0.0)   # hi − median
    err_hi = np.maximum(ci[1] - ci[0], 0.0)   # median − lo
    return diff, err_lo, err_hi


###########################################################################
# Main loop
###########################################################################

for location in LOCATIONS:
    print(f"\nProcessing: {location}")

    # data[col][period][buf] = (diff, err_lo, err_hi)
    # col: 'C' = hourly, 'D' = 10min/daily, 'E' = 10min/hourly
    # period: 'pres' / 'fut'
    data = {k: {'pres': {}, 'fut': {}} for k in ('C', 'D', 'E')}
    plot_quantiles = None

    for buf in BUFFERS:
        fname_h   = os.path.join(PATH_NPZ, f'{location}_buf{buf}.npz')
        fname_10m = os.path.join(PATH_NPZ, f'{location}_buf{buf}_10min.npz')

        # --- Hourly (Method C) ---
        if os.path.exists(fname_h):
            d = np.load(fname_h)
            if plot_quantiles is None:
                plot_quantiles = d['plot_quantiles']
            ci_C_pres = ci_from_boot(d['syn_pres_c_h_boot_buf'], BOOTSTRAP_QUANTILES)
            ci_C_fut  = ci_from_boot(d['syn_fut_c_h_boot_buf'],  BOOTSTRAP_QUANTILES)
            data['C']['pres'][buf] = diff_and_errs(d['obs_pres_h_buf'], ci_C_pres)
            data['C']['fut'][buf]  = diff_and_errs(d['obs_fut_h_buf'],  ci_C_fut)
        else:
            print(f"  Missing hourly NPZ for buf={buf} — skipping")

        # --- 10-min (Methods D and E) ---
        if os.path.exists(fname_10m):
            m = np.load(fname_10m)
            if plot_quantiles is None:
                plot_quantiles = m['plot_quantiles']
            ci_D_pres = ci_from_boot(m['D_pres_boot_buf'], BOOTSTRAP_QUANTILES)
            ci_D_fut  = ci_from_boot(m['D_fut_boot_buf'],  BOOTSTRAP_QUANTILES)
            ci_E_pres = ci_from_boot(m['E_pres_boot_buf'], BOOTSTRAP_QUANTILES)
            ci_E_fut  = ci_from_boot(m['E_fut_boot_buf'],  BOOTSTRAP_QUANTILES)
            data['D']['pres'][buf] = diff_and_errs(m['obs_pres_10m_buf'], ci_D_pres)
            data['D']['fut'][buf]  = diff_and_errs(m['obs_fut_10m_buf'],  ci_D_fut)
            data['E']['pres'][buf] = diff_and_errs(m['obs_pres_10m_buf'], ci_E_pres)
            data['E']['fut'][buf]  = diff_and_errs(m['obs_fut_10m_buf'],  ci_E_fut)
        else:
            print(f"  Missing 10min NPZ for buf={buf} — skipping")

    if plot_quantiles is None:
        print(f"  No data found for {location} — skipping.")
        continue

    x_labels = [f"P{int(q * 100)}" if q * 100 == int(q * 100)
                else f"P{q * 100:.1f}"
                for q in plot_quantiles]
    x_idx = np.arange(len(plot_quantiles))

    cl_lo = BOOTSTRAP_QUANTILES[0]
    cl_hi = BOOTSTRAP_QUANTILES[2]

    ###########################################################################
    # Build figure: 2 rows × 3 columns
    ###########################################################################

    fig, axes = plt.subplots(2, 3, figsize=(18, 9), sharey=False)
    fig.suptitle(
        f'{location}  —  CI: {cl_lo}–{cl_hi}',
        fontsize=14, fontweight='bold', y=1.01)

    panel_info = [
        (axes[0, 0], data['C']['pres'], 'Present — Hourly intensity (C)',        'mm/h'),
        (axes[0, 1], data['D']['pres'], 'Present — 10-min from daily (D)',        'mm/h'),
        (axes[0, 2], data['E']['pres'], 'Present — 10-min from hourly (E)',       'mm/h'),
        (axes[1, 0], data['C']['fut'],  'Future  — Hourly intensity (C)',         'mm/h'),
        (axes[1, 1], data['D']['fut'],  'Future  — 10-min from daily (D)',        'mm/h'),
        (axes[1, 2], data['E']['fut'],  'Future  — 10-min from hourly (E)',       'mm/h'),
    ]

    for ax, data_dict, title, units in panel_info:
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
                        markersize=6, linewidth=1.5,
                        elinewidth=1.5, capsize=4, capthick=1.5,
                        label=f'Buffer {buf}', zorder=4)

        ax.set_xticks(x_idx)
        ax.set_xticklabels(x_labels, fontsize=10)
        ax.set_xlabel('Quantile', fontsize=11, fontweight='bold')
        ax.set_ylabel(f'Obs − synthetic median ({units})', fontsize=11, fontweight='bold')
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.grid(True, linestyle=':', linewidth=0.5, alpha=0.7)
        ax.set_axisbelow(True)
        ax.tick_params(axis='both', which='major', labelsize=10)
        ax.legend(frameon=True, fancybox=True, shadow=True,
                  fontsize=9, loc='upper left')

    # Apply the same y-limits to all panels for cross-method comparison
    all_axes = [ax for ax, _, _, _ in panel_info]
    ymin = min(ax.get_ylim()[0] for ax in all_axes)
    ymax = max(ax.get_ylim()[1] for ax in all_axes)
    for ax in all_axes:
        ax.set_ylim(ymin, ymax)

    fig.tight_layout()
    outfile = os.path.join(PATH_OUT, f'buffer_CI_distance_combined_{location}.png')
    fig.savefig(outfile, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: {outfile}")

print("\nDone.")
