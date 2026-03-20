#!/usr/bin/env python
'''
@File    :  plot_xy_EPICC_hourly_daily_context.py
@Time    :  2026/03/20
@Author  :  Daniel Argüeso
@Version :  1.0
@Contact :  d.argueso@uib.es
@License :  (C)Copyright 2025, Daniel Argüeso
@Project :  EPICC
@Desc    :  Single-panel quantile plot per location showing both hourly intensity
            and daily-max quantiles together, so that hourly changes can be read
            in the context of the daily-max scale of change.
            Future synthetic CI (Method C, bootstrap) is drawn for hourly only.
            Data come from the per-(location, buffer) .npz files saved by
            pipeline_multi_location.py.
'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import ScalarFormatter
import os

mpl.rcParams["font.size"] = 13

###########################################################################
# Configuration
###########################################################################

LOCATIONS = ['Mallorca', 'Catania', 'Turis', 'Rosiglione',
             'Ardeche', 'Corte', "L'Aquila", 'Pyrenees']

BUFFER = 5   # buffer size to use (single value)

PATH_NPZ = '/home/dargueso/Analyses/EPICC/Hourly_vs_Daily/testing_data/'
PATH_OUT = '/home/dargueso/Analyses/EPICC/Hourly_vs_Daily/'

os.makedirs(PATH_OUT, exist_ok=True)

# Bootstrap CI levels
I_LO, I_MED, I_HI = 0, 1, 2

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
    return out


###########################################################################
# Main loop
###########################################################################

for location in LOCATIONS:
    print(f"Processing: {location}")

    fname = os.path.join(PATH_NPZ, f'{location}_buf{BUFFER}.npz')
    if not os.path.exists(fname):
        print(f"  Missing: {fname} — skipping.")
        continue

    data = np.load(fname)

    q_axis              = data['plot_quantiles']       # (6,)  e.g. [0.90 … 0.999]
    bq                  = data['bootstrap_quantiles']  # (3,)  [0.025, 0.5, 0.975]
    cl_lo, cl_hi        = bq[I_LO], bq[I_HI]

    obs_pres_h_buf      = data['obs_pres_h_buf']       # (6,) present hourly
    obs_fut_h_buf       = data['obs_fut_h_buf']        # (6,) future hourly
    obs_pres_dm_buf     = data['obs_pres_dm_buf']      # (6,) present daily-max
    obs_fut_dm_buf      = data['obs_fut_dm_buf']       # (6,) future daily-max

    syn_fut_c_h_boot    = data['syn_fut_c_h_boot_buf'] # (1000, 6) future hourly bootstrap

    ci_fut_h = ci_from_boot(syn_fut_c_h_boot, bq)     # (3, 6)

    ###########################################################################
    # Plot
    ###########################################################################

    fig, ax = plt.subplots(figsize=(9, 6))

    # --- Hourly intensity ---
    ax.plot(q_axis, obs_pres_h_buf,
            color='#2E86AB', linewidth=2.0, linestyle='-',
            marker='o', markersize=5,
            label='Present hourly intensity', zorder=4)

    ax.plot(q_axis, obs_fut_h_buf,
            color='#E50C0C', linewidth=1.8, linestyle='--',
            marker='s', markersize=5,
            label='Future hourly intensity', zorder=4)

    # Future hourly synthetic CI (Method C)
    ax.plot(q_axis, ci_fut_h[I_MED],
            color='#F18F01', linewidth=1.8, linestyle='-',
            label=f'Future synthetic hourly CI ({cl_lo}–{cl_hi})', zorder=5)
    ax.plot(q_axis, ci_fut_h[I_LO], color='#F18F01', linewidth=0.8, linestyle=':')
    ax.plot(q_axis, ci_fut_h[I_HI], color='#F18F01', linewidth=0.8, linestyle=':')
    ax.fill_between(q_axis, ci_fut_h[I_LO], ci_fut_h[I_HI],
                    color='#F18F01', alpha=0.25, zorder=2)

    # --- Daily-max (context lines) ---
    ax.plot(q_axis, obs_pres_dm_buf,
            color='#2E86AB', linewidth=1.6, linestyle=':',
            marker='^', markersize=5,
            label='Present daily-max', zorder=3)

    ax.plot(q_axis, obs_fut_dm_buf,
            color='#E50C0C', linewidth=1.4, linestyle=':',
            marker='D', markersize=5,
            label='Future daily-max', zorder=3)

    # --- Axes formatting ---
    ax.set_yscale('log')
    ax.set_xlabel('Quantile', fontsize=12, fontweight='bold')
    ax.set_ylabel('Precipitation (mm)', fontsize=12, fontweight='bold')
    ax.set_title(
        f'{location}  —  Hourly intensity & daily-max quantiles  (buffer={BUFFER})',
        fontsize=13, fontweight='bold', pad=12)

    yticks = [1, 2, 5, 10, 20, 50, 100]
    ax.set_yticks(yticks)
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.get_major_formatter().set_scientific(False)

    ax.grid(True, which='both', linestyle=':', linewidth=0.5, alpha=0.7)
    ax.set_axisbelow(True)
    ax.minorticks_on()
    ax.tick_params(axis='both', which='major', labelsize=11)

    # Deduplicate legend (CI boundary lines share colour but have no label)
    handles, labels = ax.get_legend_handles_labels()
    seen, uh, ul = {}, [], []
    for h, l in zip(handles, labels):
        if l not in seen:
            seen[l] = True
            uh.append(h)
            ul.append(l)
    ax.legend(uh, ul, frameon=True, fancybox=True, shadow=True,
              fontsize=10, loc='upper left')

    fig.tight_layout()
    outfile = os.path.join(
        PATH_OUT, f'hourly_daily_context_{location}_buf{BUFFER}.png')
    fig.savefig(outfile, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"  Saved: {outfile}")
    plt.close(fig)

print("\nDone.")
