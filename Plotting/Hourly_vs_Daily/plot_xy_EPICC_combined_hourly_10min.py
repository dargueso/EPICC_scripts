#!/usr/bin/env python
'''
@File    :  plot_xy_EPICC_combined_hourly_10min.py
@Time    :  2026/03/26
@Author  :  Daniel Argüeso
@Version :  1.0
@Contact :  d.argueso@uib.es
@License :  (C)Copyright 2025, Daniel Argüeso
@Project :  EPICC
@Desc    :  Combined plot merging outputs from pipeline_multi_location.py
            (hourly scale) and pipeline_multi_location_10min.py (10-min scale).

            One figure per (location, buffer), 1 row × 4 panels:
              Panel 0  — Hourly intensity (buffer pooled, Method C)
                         Bottom-left of validation_*.png + P99 attribution annotation
              Panel 1  — 10-min from daily (Method D, buffer pooled)
              Panel 2  — 10-min from hourly (Method E, buffer pooled)
              Panel 3  — 10-min comparison: both future CIs overlaid

            All panels show center-pixel observed lines as dotted reference
            (like the buffer rows in pipeline_multi_location.py do).

            Requires:
              testing_data/{loc}_buf{N}.npz         (pipeline_multi_location)
              testing_data/{loc}_buf{N}_10min.npz   (pipeline_multi_location_10min)
'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os

mpl.rcParams["font.size"] = 12

# =============================================================================
# Configuration
# =============================================================================

LOCATIONS = ['Mallorca', 'Catania', 'Turis', 'Rosiglione',
             'Ardeche', 'Corte', "L'Aquila", 'Pyrenees']

BUFFERS = [3]

PATH_NPZ = '/home/dargueso/Analyses/EPICC/Hourly_vs_Daily/testing_data/'
PATH_OUT = '/home/dargueso/Analyses/EPICC/Hourly_vs_Daily/'

os.makedirs(PATH_OUT, exist_ok=True)

BOOTSTRAP_QUANTILES = np.array([0.005, 0.5, 0.995])
I_LO, I_MED, I_HI  = 0, 1, 2
cl_lo, cl_hi        = BOOTSTRAP_QUANTILES[I_LO], BOOTSTRAP_QUANTILES[I_HI]

# Colours
C_PRES_OBS = '#2E86AB'   # blue  — present observed
C_FUT_OBS  = '#E50C0C'   # red   — future observed
C_SYN_PRES = '#2E86AB'   # blue shading — synthetic present CI
C_SYN_H    = '#F18F01'   # orange — hourly synthetic future CI
C_SYN_D    = '#F18F01'   # orange — 10-min from daily future CI
C_SYN_E    = '#9B59B6'   # purple — 10-min from hourly future CI


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


def _setup_ax(ax, ylabel, title, ylim_bottom=None):
    ax.set_yscale('log')
    ax.set_xlabel('Quantile', fontsize=10, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=10, fontweight='bold')
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.grid(True, which='both', linestyle=':', linewidth=0.5, alpha=0.7)
    ax.set_axisbelow(True)
    ax.minorticks_on()
    ax.set_yticks([1, 2, 5, 10, 20, 50, 100])
    ax.yaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
    ax.yaxis.get_major_formatter().set_scientific(False)
    ax.tick_params(axis='both', which='major', labelsize=9)
    if ylim_bottom is not None:
        ax.set_ylim(bottom=ylim_bottom)


def _dedup_legend(ax, **kwargs):
    handles, labels = ax.get_legend_handles_labels()
    seen, uh, ul = {}, [], []
    for h, l in zip(handles, labels):
        if l not in seen:
            seen[l] = True; uh.append(h); ul.append(l)
    ax.legend(uh, ul, **kwargs)


def _add_obs(ax, q_axis, obs_buf, obs_ctr, period):
    """Plot buffer-pooled obs (solid) and center-pixel obs (dotted reference)."""
    color = C_PRES_OBS if period == 'pres' else C_FUT_OBS
    marker_buf = 'o' if period == 'pres' else 's'
    marker_ctr = '^' if period == 'pres' else 'v'
    lw_ctr = 1.0
    label_buf = 'Present observed' if period == 'pres' else 'Future observed'
    label_ctr = ('Present obs (center pixel)' if period == 'pres'
                 else 'Future obs (center pixel)')
    ls_buf = '-' if period == 'pres' else '--'

    ax.plot(q_axis, obs_buf, color=color, linewidth=1.8,
            linestyle=ls_buf, marker=marker_buf, markersize=5,
            label=label_buf, zorder=4)
    ax.plot(q_axis, obs_ctr, color=color, linewidth=lw_ctr,
            linestyle=':', marker=marker_ctr, markersize=4, alpha=0.7,
            label=label_ctr, zorder=3)


def _add_ci(ax, ci, color, label, alpha=0.20):
    ax.plot(q_axis_ref[0], ci[I_MED], color=color, linewidth=1.8,
            label=label, zorder=5)
    ax.fill_between(q_axis_ref[0], ci[I_LO], ci[I_HI],
                    color=color, alpha=alpha)


def _p99_annotation(ax, obs_pres_buf, obs_fut_buf, ci_fut, q_axis, units='mm/h'):
    idx = np.searchsorted(q_axis, 0.99)
    if idx >= len(q_axis) or not np.isclose(q_axis[idx], 0.99, atol=0.001):
        return
    total  = float(obs_fut_buf[idx]) - float(obs_pres_buf[idx])
    expl   = float(ci_fut[I_MED, idx]) - float(obs_pres_buf[idx])
    struct = float(obs_fut_buf[idx])   - float(ci_fut[I_MED, idx])
    pct_e  = 100.0 * expl   / abs(total) if total != 0 else 0.0
    pct_s  = 100.0 * struct / abs(total) if total != 0 else 0.0
    ax.text(0.97, 0.04,
            f'P99 total Δ : {total:+.2f} {units}\n'
            f'Explained  : {expl:+.2f} {units}  ({pct_e:.0f}%)\n'
            f'Structural : {struct:+.2f} {units}  ({pct_s:.0f}%)',
            transform=ax.transAxes, fontsize=8,
            verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='square,pad=0.4', facecolor='wheat', alpha=0.75))


# Placeholder for q_axis — filled per file load
q_axis_ref = [None]


# =============================================================================
# Main loop
# =============================================================================

for location in LOCATIONS:
    for buf in BUFFERS:

        fname_h   = os.path.join(PATH_NPZ, f'{location}_buf{buf}.npz')
        fname_10m = os.path.join(PATH_NPZ, f'{location}_buf{buf}_10min.npz')

        if not os.path.exists(fname_h):
            print(f"  Missing hourly NPZ for {location} buf={buf} — skipping")
            continue
        if not os.path.exists(fname_10m):
            print(f"  Missing 10min NPZ for {location} buf={buf} — skipping")
            continue

        # ------------------------------------------------------------------
        # Load hourly NPZ
        # ------------------------------------------------------------------
        h = np.load(fname_h)
        q_axis  = h['plot_quantiles']
        q_axis_ref[0] = q_axis

        obs_pres_h     = h['obs_pres_h']       # center pixel
        obs_fut_h      = h['obs_fut_h']
        obs_pres_h_buf = h['obs_pres_h_buf']   # buffer pooled
        obs_fut_h_buf  = h['obs_fut_h_buf']

        ci_pres_h  = ci_from_boot(h['syn_pres_c_h_boot_buf'], BOOTSTRAP_QUANTILES)
        ci_fut_h   = ci_from_boot(h['syn_fut_c_h_boot_buf'],  BOOTSTRAP_QUANTILES)

        ny_reg = int(h['ny_reg'])
        nx_reg = int(h['nx_reg'])

        # ------------------------------------------------------------------
        # Load 10-min NPZ
        # ------------------------------------------------------------------
        m = np.load(fname_10m)

        obs_pres_10m     = m['obs_pres_10m']       # center pixel
        obs_fut_10m      = m['obs_fut_10m']
        obs_pres_10m_buf = m['obs_pres_10m_buf']   # buffer pooled
        obs_fut_10m_buf  = m['obs_fut_10m_buf']

        ci_D_pres = ci_from_boot(m['D_pres_boot_buf'], BOOTSTRAP_QUANTILES)
        ci_D_fut  = ci_from_boot(m['D_fut_boot_buf'],  BOOTSTRAP_QUANTILES)
        ci_E_pres = ci_from_boot(m['E_pres_boot_buf'], BOOTSTRAP_QUANTILES)
        ci_E_fut  = ci_from_boot(m['E_fut_boot_buf'],  BOOTSTRAP_QUANTILES)

        buf_str  = f"buf={buf}  ({ny_reg}×{nx_reg} px)"
        show_ctr = buf > 0   # only show center-pixel reference when buffer is active

        # ------------------------------------------------------------------
        # Build figure: 1 row × 4 panels
        # ------------------------------------------------------------------
        fig, axes = plt.subplots(1, 4, figsize=(22, 6))

        # --- Panel 0: Hourly intensity (buffer pooled) ---
        ax = axes[0]
        _add_obs(ax, q_axis, obs_pres_h_buf, obs_pres_h, 'pres')
        _add_obs(ax, q_axis, obs_fut_h_buf,  obs_fut_h,  'fut')
        ax.fill_between(q_axis, ci_pres_h[I_LO], ci_pres_h[I_HI],
                        color=C_SYN_PRES, alpha=0.15,
                        label=f'Synthetic present CI ({cl_lo}–{cl_hi})')
        ax.plot(q_axis, ci_pres_h[I_MED], color=C_SYN_PRES,
                linewidth=1.4, linestyle='--')
        ax.fill_between(q_axis, ci_fut_h[I_LO], ci_fut_h[I_HI],
                        color=C_SYN_H, alpha=0.25,
                        label=f'Synthetic future CI ({cl_lo}–{cl_hi})')
        ax.plot(q_axis, ci_fut_h[I_MED], color=C_SYN_H, linewidth=1.8,
                label='Synthetic future (from daily)')
        _p99_annotation(ax, obs_pres_h_buf, obs_fut_h_buf, ci_fut_h, q_axis)
        _setup_ax(ax, '1-hour precipitation (mm/h)',
                  f'Hourly intensity\n{location}, {buf_str}',
                  ylim_bottom=5)
        _dedup_legend(ax, fontsize=8, loc='upper left', frameon=True,
                      fancybox=True, shadow=True)

        # --- Panel 1: 10-min from daily (Method D) ---
        ax = axes[1]
        _add_obs(ax, q_axis, obs_pres_10m_buf, obs_pres_10m, 'pres')
        _add_obs(ax, q_axis, obs_fut_10m_buf,  obs_fut_10m,  'fut')
        ax.fill_between(q_axis, ci_D_pres[I_LO], ci_D_pres[I_HI],
                        color=C_SYN_PRES, alpha=0.15,
                        label=f'Synthetic present CI ({cl_lo}–{cl_hi})')
        ax.plot(q_axis, ci_D_pres[I_MED], color=C_SYN_PRES,
                linewidth=1.4, linestyle='--')
        ax.fill_between(q_axis, ci_D_fut[I_LO], ci_D_fut[I_HI],
                        color=C_SYN_D, alpha=0.25,
                        label=f'Synthetic future CI ({cl_lo}–{cl_hi})')
        ax.plot(q_axis, ci_D_fut[I_MED], color=C_SYN_D, linewidth=1.8,
                label='Synthetic future (from daily)')
        _p99_annotation(ax, obs_pres_10m_buf, obs_fut_10m_buf, ci_D_fut, q_axis, units='mm/h')
        _setup_ax(ax, '10-min precipitation (mm/h)',
                  f'10-min from daily\n{location}, {buf_str}',
                  ylim_bottom=5)
        _dedup_legend(ax, fontsize=8, loc='upper left', frameon=True,
                      fancybox=True, shadow=True)

        # --- Panel 2: 10-min from hourly (Method E) ---
        ax = axes[2]
        _add_obs(ax, q_axis, obs_pres_10m_buf, obs_pres_10m, 'pres')
        _add_obs(ax, q_axis, obs_fut_10m_buf,  obs_fut_10m,  'fut')
        ax.fill_between(q_axis, ci_E_pres[I_LO], ci_E_pres[I_HI],
                        color=C_SYN_PRES, alpha=0.15,
                        label=f'Synthetic present CI ({cl_lo}–{cl_hi})')
        ax.plot(q_axis, ci_E_pres[I_MED], color=C_SYN_PRES,
                linewidth=1.4, linestyle='--')
        ax.fill_between(q_axis, ci_E_fut[I_LO], ci_E_fut[I_HI],
                        color=C_SYN_E, alpha=0.25,
                        label=f'Synthetic future CI ({cl_lo}–{cl_hi})')
        ax.plot(q_axis, ci_E_fut[I_MED], color=C_SYN_E, linewidth=1.8,
                label='Synthetic future (from hourly)')
        _p99_annotation(ax, obs_pres_10m_buf, obs_fut_10m_buf, ci_E_fut, q_axis, units='mm/h')
        _setup_ax(ax, '10-min precipitation (mm/h)',
                  f'10-min from hourly\n{location}, {buf_str}',
                  ylim_bottom=5)
        _dedup_legend(ax, fontsize=8, loc='upper left', frameon=True,
                      fancybox=True, shadow=True)

        # --- Panel 3: comparison — both future CIs overlaid ---
        ax = axes[3]
        _add_obs(ax, q_axis, obs_pres_10m_buf, obs_pres_10m, 'pres')
        _add_obs(ax, q_axis, obs_fut_10m_buf,  obs_fut_10m,  'fut')
        ax.fill_between(q_axis, ci_D_fut[I_LO], ci_D_fut[I_HI],
                        color=C_SYN_D, alpha=0.25,
                        label=f'Synthetic future (from daily)')
        ax.plot(q_axis, ci_D_fut[I_MED], color=C_SYN_D, linewidth=1.8)
        ax.fill_between(q_axis, ci_E_fut[I_LO], ci_E_fut[I_HI],
                        color=C_SYN_E, alpha=0.25,
                        label=f'Synthetic future (from hourly)')
        ax.plot(q_axis, ci_E_fut[I_MED], color=C_SYN_E, linewidth=1.8)
        _setup_ax(ax, '10-min precipitation (mm/h)',
                  f'10-min comparison\n{location}, {buf_str}',
                  ylim_bottom=5)
        _dedup_legend(ax, fontsize=8, loc='upper left', frameon=True,
                      fancybox=True, shadow=True)

        fig.suptitle(
            f'{location}  —  buffer={buf}  ({ny_reg}×{nx_reg} px)\n'
            f'Solid: buffer-pooled obs   Dotted: center-pixel obs (reference)\n'
            f'Blue shading: synthetic present   '
            f'Orange: synthetic future from daily   '
            f'Purple: synthetic future from hourly',
            fontsize=11, fontweight='bold')
        fig.tight_layout()

        outfile = os.path.join(
            PATH_OUT, f'combined_hourly_10min_{location}_buf{buf}.png')
        fig.savefig(outfile, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f"  Saved: {outfile}")

print("\nDone.")
