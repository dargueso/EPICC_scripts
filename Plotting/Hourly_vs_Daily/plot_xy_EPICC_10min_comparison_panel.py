#!/usr/bin/env python
'''
@File    :  plot_xy_EPICC_10min_comparison_panel.py
@Time    :  2026/05/12
@Author  :  Daniel Argüeso
@Version :  1.0
@Contact :  d.argueso@uib.es
@License :  (C)Copyright 2025, Daniel Argüeso
@Project :  EPICC
@Desc    :  2×4 panel plot of 10-min extremes inferred from daily (Method D)
            and from hourly (Method E) for 8 locations.
            Layout mirrors plot_map_xy_EPICC_real_vs_synthetic_extremes.py.
            Each panel overlays both synthetic future CIs on the observed lines.
            Yellow annotation box shows P99 total Δ once, with explained and
            structural decomposition for both daily→10min and hourly→10min.
'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FuncFormatter
import string
import os

mpl.rcParams['font.size']       = 12
mpl.rcParams['hatch.color']     = 'red'
mpl.rcParams['hatch.linewidth'] = 0.8

# =============================================================================
# Configuration
# =============================================================================

LOCATIONS = ['Mallorca', 'Turis', 'Pyrenees', 'Rosiglione',
             'Ardeche', 'Corte', 'Catania', "L'Aquila"]

BUFFER = 5

PATH_NPZ = '/home/dargueso/Analyses/EPICC/Hourly_vs_Daily/testing_data/'
PATH_OUT = '/home/dargueso/Analyses/EPICC/Hourly_vs_Daily/'

BOOTSTRAP_QUANTILES = np.array([0.005, 0.5, 0.995])
I_LO, I_MED, I_HI  = 0, 1, 2
cl_lo, cl_hi        = BOOTSTRAP_QUANTILES[I_LO], BOOTSTRAP_QUANTILES[I_HI]

C_PRES_OBS = '#2E86AB'   # blue  — present observed
C_FUT_OBS  = '#E50C0C'   # red   — future observed
C_SYN_D    = '#F18F01'   # orange — synthetic future from daily
C_SYN_E    = '#9B59B6'   # purple — synthetic future from hourly

os.makedirs(PATH_OUT, exist_ok=True)

# =============================================================================
# Helpers
# =============================================================================

def ci_from_boot(boot, bq=BOOTSTRAP_QUANTILES):
    """(N_SAMPLES, n_q) → (3, n_q) rows: [lo, median, hi]."""
    out = np.full((3, boot.shape[1]), np.nan)
    for iq in range(boot.shape[1]):
        v = boot[:, iq]
        v = v[~np.isnan(v)]
        if len(v):
            out[:, iq] = np.quantile(v, bq)
    return out


def _setup_ax(ax, nrow, ncol, q_axis, all_vals):
    ax.set_yscale('log')
    ax.grid(True, which='both', linestyle=':', linewidth=0.5, alpha=0.7)
    ax.set_axisbelow(True)
    ax.minorticks_on()
    ax.tick_params(axis='both', which='major', labelsize=9)
    ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())

    vals = all_vals[np.isfinite(all_vals) & (all_vals > 0)]
    if len(vals):
        vmin, vmax = float(vals.min()), float(vals.max())
        candidates = [1, 2, 5, 10, 20, 50, 100, 200]
        nice_ticks = [t for t in candidates if vmin * 0.7 <= t <= vmax * 1.4]
        if nice_ticks:
            ax.set_yticks(nice_ticks)
            ax.set_ylim(nice_ticks[0] * 0.75, nice_ticks[-1] * 1.3)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:g}'))

    span = q_axis[-1] - q_axis[0]
    ax.set_xlim(q_axis[0] - 0.05 * span, q_axis[-1] + 0.05 * span)

    if nrow == 1:
        ax.set_xlabel('Quantile', fontsize=10, fontweight='bold')
    if ncol == 0:
        ax.set_ylabel('10-min precipitation (mm/h)', fontsize=10, fontweight='bold')


def _combined_p99_annotation(ax, obs_pres_buf, obs_fut_buf,
                              ci_D_fut, ci_E_fut, q_axis):
    idx = np.searchsorted(q_axis, 0.99)
    if idx >= len(q_axis) or not np.isclose(q_axis[idx], 0.99, atol=0.001):
        return

    op    = float(obs_pres_buf[idx])
    of    = float(obs_fut_buf[idx])
    total = of - op

    def _parts(ci_fut):
        expl   = float(ci_fut[I_MED, idx]) - op
        struct = of - float(ci_fut[I_MED, idx])
        pct_e  = 100.0 * expl   / abs(total) if total != 0 else 0.0
        pct_s  = 100.0 * struct / abs(total) if total != 0 else 0.0
        return expl, struct, pct_e, pct_s

    d_e, d_s, d_pe, d_ps = _parts(ci_D_fut)
    e_e, e_s, e_pe, e_ps = _parts(ci_E_fut)

    ax.text(0.97, 0.04,
            f'P99 total Δ : {total:+.2f} mm/h\n'
            f'Daily →10min :  Expld {d_e:+.2f} ({d_pe:.0f}%)   Strct {d_s:+.2f} ({d_ps:.0f}%)\n'
            f'Hourly→10min :  Expld {e_e:+.2f} ({e_pe:.0f}%)   Strct {e_s:+.2f} ({e_ps:.0f}%)',
            transform=ax.transAxes, fontsize=7,
            verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='square,pad=0.4', facecolor='wheat', alpha=0.75))


# =============================================================================
# Build figure
# =============================================================================

fig = plt.figure(figsize=(20, 10), constrained_layout=False)
gs_main = GridSpec(2, 1, figure=fig,
                   left=0.06, bottom=0.10, right=0.98, top=0.93,
                   hspace=0.30)
gs_row0 = gs_main[0].subgridspec(1, 4, wspace=0.12)
gs_row1 = gs_main[1].subgridspec(1, 4, wspace=0.12)

for loc_idx, loc_name in enumerate(LOCATIONS):
    nrow = loc_idx // 4
    ncol = loc_idx  % 4

    fname_h   = os.path.join(PATH_NPZ, f'{loc_name}_buf{BUFFER}.npz')
    fname_10m = os.path.join(PATH_NPZ, f'{loc_name}_buf{BUFFER}_10min.npz')

    if not os.path.exists(fname_h):
        print(f'  Missing hourly NPZ: {fname_h} — skipping')
        continue
    if not os.path.exists(fname_10m):
        print(f'  Missing 10min NPZ: {fname_10m} — skipping')
        continue

    h = np.load(fname_h)
    m = np.load(fname_10m)

    q_axis       = h['plot_quantiles']
    obs_pres_buf = m['obs_pres_10m_buf']
    obs_fut_buf  = m['obs_fut_10m_buf']
    ci_D_fut     = ci_from_boot(m['D_fut_boot_buf'])
    ci_E_fut     = ci_from_boot(m['E_fut_boot_buf'])

    gs_row = gs_row0 if nrow == 0 else gs_row1
    ax = fig.add_subplot(gs_row[0, ncol])

    # Panel label + location name
    ax.set_title(string.ascii_lowercase[loc_idx], size='x-large',
                 weight='bold', loc='left')
    ax.text(0.03, 0.98, loc_name, fontsize=9, transform=ax.transAxes,
            va='top', ha='left')

    # Observed lines (buffer pooled, no center pixel)
    ax.plot(q_axis, obs_pres_buf,
            color=C_PRES_OBS, linewidth=1.8, linestyle='-',
            marker='o', markersize=4, label='Present observed', zorder=4)
    ax.plot(q_axis, obs_fut_buf,
            color=C_FUT_OBS, linewidth=1.8, linestyle='--',
            marker='s', markersize=4, label='Future observed', zorder=4)

    # Synthetic future CIs
    ax.fill_between(q_axis, ci_D_fut[I_LO], ci_D_fut[I_HI],
                    color=C_SYN_D, alpha=0.25)
    ax.plot(q_axis, ci_D_fut[I_MED], color=C_SYN_D, linewidth=1.8,
            label=f'Syn future — daily→10min  ({cl_lo}–{cl_hi})')

    ax.fill_between(q_axis, ci_E_fut[I_LO], ci_E_fut[I_HI],
                    color=C_SYN_E, alpha=0.25)
    ax.plot(q_axis, ci_E_fut[I_MED], color=C_SYN_E, linewidth=1.8,
            label=f'Syn future — hourly→10min ({cl_lo}–{cl_hi})')

    # Combined P99 annotation box
    _combined_p99_annotation(ax, obs_pres_buf, obs_fut_buf,
                             ci_D_fut, ci_E_fut, q_axis)

    # Axis cosmetics — data-driven y range
    all_vals = np.concatenate([obs_pres_buf, obs_fut_buf,
                                ci_D_fut.ravel(), ci_E_fut.ravel()])
    _setup_ax(ax, nrow, ncol, q_axis, all_vals)

    if loc_idx == 0:
        ax.legend(fontsize=7, loc='upper left', frameon=True,
                  fancybox=True, ncol=1)

fig.suptitle(
    f'10-min extreme precipitation — buffer={BUFFER}\n'
    f'Orange: synthetic future from daily   '
    f'Purple: synthetic future from hourly',
    fontsize=12, fontweight='bold')

outfile = os.path.join(PATH_OUT, f'10min_comparison_panel_buf{BUFFER}.png')
fig.savefig(outfile, dpi=150, bbox_inches='tight', facecolor='white')
plt.close(fig)
print(f'Saved: {outfile}')
