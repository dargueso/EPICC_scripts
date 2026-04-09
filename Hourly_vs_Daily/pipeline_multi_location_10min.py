#!/usr/bin/env python
"""
pipeline_multi_location_10min.py

10-minute rainfall attribution pipeline — analog of pipeline_multi_location.py
but operating at 10-min sub-daily scale with two conditioning strategies:

  Method D  (from daily)  : P(10-min intensity | daily total)
      Analog profiles of 144 × 10-min intervals indexed by daily total bin.
      Bootstrap feeds future DAILY totals through present conditional model.
      Question answered: "do daily-scale changes explain 10-min changes?"

  Method E  (from hourly) : P(10-min intensity | hourly total)
      Analog profiles of 6 × 10-min intervals per hour indexed by hourly total bin.
      Bootstrap feeds future HOURLY totals through present conditional model.
      Question answered: "do hourly-scale changes explain 10-min changes?"

By comparing both synthetic future 10-min CIs against observed future 10-min
quantiles we can determine at which conditioning scale (daily or hourly) the
sub-hourly temporal structure changes.

Output files  (PATH_OUT/)
---------
  validation_10min_{loc}_buf{N}.png   — 2×2 figure
      rows: center pixel / buffer pooled
      cols: from-daily CI / from-hourly CI
  attribution_10min_{loc}_buf{N}.png  — 1×3 figure
      col 0: from-daily attribution
      col 1: from-hourly attribution
      col 2: both CIs overlaid for direct comparison
  testing_data/{loc}_buf{N}_10min.npz — bootstrap arrays for re-use

Data
----
  UIB_10MIN_RAIN.zarr  (present + future)
  Units: mm accumulated per 10-min period → converted to mm/h on load (×6)
  N_INTERVAL = 144 (10-min steps per day)
"""

import time
import os
import matplotlib
matplotlib.use('Agg')   # non-interactive backend — must be set before pyplot import
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import ScalarFormatter
from joblib import Parallel, delayed

mpl.rcParams["font.size"] = 13


# =============================================================================
# CONFIGURATION
# =============================================================================

PATH_IN  = '/home/dargueso/postprocessed/EPICC/'
PATH_OUT = '/home/dargueso/Analyses/EPICC/Hourly_vs_Daily/'

WRUN_PRESENT = 'EPICC_2km_ERA5'
WRUN_FUTURE  = 'EPICC_2km_ERA5_CMIP6anom'

loc_lats = {'Mallorca':   39.6353, 'Barcelona': 41.385,  'Valencia':   39.469,
            'Rosiglione': 44.5584, 'Catania':   37.5055, 'Turis':      39.3867,
            'Pyrenees':   41.9771, 'Ardeche':   44.7585, 'Corte':      42.3002,
            "L'Aquila":   42.3577}
loc_lons = {'Mallorca':   2.6360,  'Barcelona':  2.173,  'Valencia':   -0.376,
            'Rosiglione': 8.6722,  'Catania':   15.0935, 'Turis':      -0.6195,
            'Pyrenees':   2.8245,  'Ardeche':    4.5673, 'Corte':       9.1565,
            "L'Aquila":  13.4068}

LOCATIONS = ['Mallorca', 'Catania', 'Turis', 'Rosiglione', 'Ardeche',
             'Corte', "L'Aquila", 'Pyrenees']
BUFFERS   = [0, 1,2, 3,4, 5,6,7,8,9, 10, 15, 20]

# 10-min specific
N_INTERVAL   = 144    # 10-min steps per day
N_HOUR_STEPS = 6      # 10-min steps per hour
CONV_TO_MMH  = 6.0    # raw mm/10min × 6 → mm/h

WET_VALUE_HIGH = 0.1  # mm/h — 10-min wet threshold (after conversion)
WET_VALUE_LOW  = 0.1  # mm/d — daily wet threshold
WET_HOUR_LOW   = 0.1  # mm/h — hourly wet threshold (for Method E conditioning)

PLOT_QUANTILES      = np.array([0.90, 0.95, 0.98, 0.99, 0.995, 0.999])
# Bins for daily conditioning (Method D)
BINS_LOW   = np.concatenate([
    np.arange(0, 1.0, 0.25),   # fine bins below 1 mm/d (empty when threshold = 1 mm)
    np.arange(1.0, 5.0, 1.0),  # 1 mm/d bins for 1–5 mm range
    np.arange(5.0, 100, 5),    # original 5 mm/d bins from 5 mm up (unchanged)
    [np.inf]
])
# Bins for hourly conditioning (Method E)
BINS_HOUR  = np.concatenate([
    np.arange(0, 1.0, 0.1),    # 0.1 mm/h bins below 1 mm/h  (most wet hours here)
    np.arange(1.0, 10.0, 1.0), # 1 mm/h bins for 1–10 mm/h range
    np.arange(10.0, 100, 5),   # 5 mm/h bins from 10 mm/h up
    [np.inf]
])

N_SAMPLES           = 1000
BOOTSTRAP_QUANTILES = np.array([0.025, 0.5, 0.975])
EXP_SCALE           = 5.0

N_JOBS = -1   # joblib workers (-1 = all cores, 1 = sequential)

os.makedirs(PATH_OUT, exist_ok=True)
os.makedirs(os.path.join(PATH_OUT, 'testing_data'), exist_ok=True)


# =============================================================================
# MODULE-LEVEL DERIVED CONSTANTS
# =============================================================================

q_axis    = PLOT_QUANTILES
nbins_low  = len(BINS_LOW)  - 1
nbins_hour = len(BINS_HOUR) - 1
I_LO, I_MED, I_HI = 0, 1, 2
cl_lo = BOOTSTRAP_QUANTILES[I_LO]
cl_hi = BOOTSTRAP_QUANTILES[I_HI]


# =============================================================================
# UTILITIES
# =============================================================================

def section(title):
    ts = time.strftime('%H:%M:%S')
    print(f"\n{'='*70}")
    print(f"  [{ts}]  {title}")
    print(f"{'='*70}")


def subsection(title):
    print(f"\n  --- {title} ---")


def elapsed(t0):
    return f"{time.time() - t0:.1f}s"


def _open_zarr(path):
    try:
        return xr.open_zarr(path, consolidated=True)
    except KeyError:
        return xr.open_zarr(path, consolidated=False)


def ci_across_samples(q_boot):
    n_q = q_boot.shape[1]
    out = np.full((3, n_q), np.nan, dtype=np.float32)
    for iq in range(n_q):
        valid = q_boot[:, iq]
        valid = valid[~np.isnan(valid)]
        if len(valid) > 0:
            out[:, iq] = np.quantile(valid, BOOTSTRAP_QUANTILES)
    return out


def _attr_row(op, of, sf):
    total  = of - op
    daily  = sf - op
    struct = of - sf
    pct_d  = 100.0 * daily  / abs(total) if total != 0 else 0.0
    pct_s  = 100.0 * struct / abs(total) if total != 0 else 0.0
    return total, daily, struct, pct_d, pct_s


def _setup_ax(ax, ylabel, title):
    ax.set_yscale('log')
    ax.set_xlabel('Quantile', fontsize=11, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=11, fontweight='bold')
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.grid(True, which='both', linestyle=':', linewidth=0.5, alpha=0.7)
    ax.set_axisbelow(True)
    ax.minorticks_on()
    ax.set_yticks([1, 2, 5, 10, 20, 50, 100])
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.get_major_formatter().set_scientific(False)
    ax.tick_params(axis='both', which='major', labelsize=9)


def _dedup_legend(ax, **kwargs):
    handles, labels = ax.get_legend_handles_labels()
    seen, uh, ul = {}, [], []
    for h, l in zip(handles, labels):
        if l not in seen:
            seen[l] = True
            uh.append(h)
            ul.append(l)
    ax.legend(uh, ul, **kwargs)


def _add_ci(ax, ci, color, label, alpha=0.2):
    ax.plot(q_axis, ci[I_MED], color=color, linewidth=1.8, label=label, zorder=5)
    ax.fill_between(q_axis, ci[I_LO], ci[I_HI], color=color, alpha=alpha)


def _plot_panel(ax, obs_pres, obs_fut, pres_ci, fut_ci,
                pres_syn_label, fut_syn_label, ylabel, title):
    """Plot observed lines + present CI (validation) + future CI (attribution)."""
    ax.plot(q_axis, obs_pres, color='#2E86AB', linewidth=1.8, marker='o',
            markersize=5, label='Present observed', zorder=4)
    ax.plot(q_axis, obs_fut,  color='#E50C0C',  linewidth=1.5, marker='s',
            markersize=4, linestyle='--', label='Future observed', zorder=3)
    _add_ci(ax, pres_ci, '#2E86AB', pres_syn_label, alpha=0.15)
    _add_ci(ax, fut_ci,  '#F18F01', fut_syn_label,  alpha=0.20)
    _setup_ax(ax, ylabel, title)
    _dedup_legend(ax, fontsize=8, loc='upper left', frameon=True,
                  fancybox=True, shadow=True)


# =============================================================================
# PER-(LOCATION, BUFFER) PIPELINE
# =============================================================================

def run_single(location, buffer):
    """Run the full 10-min pipeline for one (location, buffer) combination."""

    run_label  = f"{location}_buf{buffer}_10min"
    buf_str    = f"buf={buffer}"
    TARGET_LAT = loc_lats[location]
    TARGET_LON = loc_lons[location]

    section(f"{run_label}  —  lat={TARGET_LAT}, lon={TARGET_LON}")

    # ------------------------------------------------------------------
    # STEP 1 — Load 10-min zarr, locate pixel, extract buffer region
    # ------------------------------------------------------------------
    zarr_pres = f'{PATH_IN}/{WRUN_PRESENT}/UIB_10MIN_RAIN.zarr'
    zarr_fut  = f'{PATH_IN}/{WRUN_FUTURE}/UIB_10MIN_RAIN.zarr'

    ds_pres = _open_zarr(zarr_pres)
    ds_fut  = _open_zarr(zarr_fut)

    lat2d    = ds_pres.lat.isel(time=0).values
    lon2d    = ds_pres.lon.isel(time=0).values
    ny_full, nx_full = lat2d.shape

    dist = np.sqrt((lat2d - TARGET_LAT)**2 + (lon2d - TARGET_LON)**2)
    cy, cx = np.unravel_index(np.argmin(dist), dist.shape)

    y0, y1 = max(0, cy - buffer), min(ny_full, cy + buffer + 1)
    x0, x1 = max(0, cx - buffer), min(nx_full, cx + buffer + 1)
    cy_loc, cx_loc = cy - y0, cx - x0
    ny_reg, nx_reg = y1 - y0, x1 - x0

    print(f"  Nearest pixel : y={cy}, x={cx}  "
          f"({lat2d[cy,cx]:.4f} / {lon2d[cy,cx]:.4f})")
    print(f"  Region        : {ny_reg}×{nx_reg} pixels, "
          f"center=[{cy_loc},{cx_loc}]")

    t0 = time.time()
    # Load raw mm/10min and convert to mm/h
    rain_pres = (ds_pres.RAIN.isel(y=slice(y0, y1), x=slice(x0, x1))
                 .values.astype(np.float32)) * CONV_TO_MMH
    rain_fut  = (ds_fut.RAIN.isel(y=slice(y0, y1), x=slice(x0, x1))
                 .values.astype(np.float32)) * CONV_TO_MMH
    ds_pres.close()
    ds_fut.close()
    print(f"  Load time : {elapsed(t0)}")

    # ------------------------------------------------------------------
    # STEP 2 — Reshape and compute derived fields
    # ------------------------------------------------------------------
    n_days_pres = rain_pres.shape[0] // N_INTERVAL
    n_days_fut  = rain_fut.shape[0]  // N_INTERVAL

    # (n_days, 144, ny, nx) mm/h
    blk_pres = rain_pres[:n_days_pres * N_INTERVAL].reshape(
        n_days_pres, N_INTERVAL, ny_reg, nx_reg)
    blk_fut  = rain_fut[:n_days_fut * N_INTERVAL].reshape(
        n_days_fut,  N_INTERVAL, ny_reg, nx_reg)

    # Daily totals (mm):  sum(mm/h × 10min) = sum(mm/h) / 6
    daily_pres = blk_pres.sum(axis=1) / N_HOUR_STEPS   # (n_days, ny, nx)
    daily_fut  = blk_fut.sum(axis=1)  / N_HOUR_STEPS

    # Hourly aggregates (mm/h):  mean of 6 consecutive 10-min mm/h values
    # (n_days, 24, 6, ny, nx) → mean over axis=2 → (n_days, 24, ny, nx)
    blk_pres_h6 = blk_pres.reshape(n_days_pres, 24, N_HOUR_STEPS, ny_reg, nx_reg)
    blk_fut_h6  = blk_fut.reshape( n_days_fut,  24, N_HOUR_STEPS, ny_reg, nx_reg)
    hourly_pres = blk_pres_h6.mean(axis=2)   # (n_days, 24, ny, nx) mm/h
    hourly_fut  = blk_fut_h6.mean(axis=2)

    # Center-pixel scalars
    pres_day_ctr = daily_pres[:, cy_loc, cx_loc]
    fut_day_ctr  = daily_fut[:,  cy_loc, cx_loc]

    print(f"  Present : {n_days_pres} days,  "
          f"wet days: {int((pres_day_ctr>=WET_VALUE_LOW).sum())}")
    print(f"  Future  : {n_days_fut} days,  "
          f"wet days: {int((fut_day_ctr>=WET_VALUE_LOW).sum())}")

    # ------------------------------------------------------------------
    # STEP 3 — Observed 10-min percentiles
    # ------------------------------------------------------------------

    def _obs_10m_ctr(blk, day_ctr):
        """Center-pixel observed 10-min intensity percentiles."""
        wet_rep = np.repeat(day_ctr >= WET_VALUE_LOW, N_INTERVAL)
        flat    = blk[:, :, cy_loc, cx_loc].reshape(-1)
        wet     = flat[(flat > WET_VALUE_HIGH) & wet_rep]
        return (np.percentile(wet, PLOT_QUANTILES * 100)
                if len(wet) > 0 else np.full(len(PLOT_QUANTILES), np.nan))

    def _obs_10m_buf(blk, daily):
        """Buffer-pooled observed 10-min intensity percentiles."""
        mask4d = (daily >= WET_VALUE_LOW)[:, np.newaxis, :, :]
        flat   = (blk * mask4d).reshape(-1)
        wet    = flat[flat > WET_VALUE_HIGH]
        return (np.percentile(wet, PLOT_QUANTILES * 100)
                if len(wet) > 0 else np.full(len(PLOT_QUANTILES), np.nan))

    obs_pres_10m     = _obs_10m_ctr(blk_pres, pres_day_ctr)
    obs_fut_10m      = _obs_10m_ctr(blk_fut,  fut_day_ctr)
    obs_pres_10m_buf = _obs_10m_buf(blk_pres, daily_pres)
    obs_fut_10m_buf  = _obs_10m_buf(blk_fut,  daily_fut)

    print(f"  Obs center : P99 pres={obs_pres_10m[3]:.2f}  fut={obs_fut_10m[3]:.2f} mm/h")
    print(f"  Obs buffer : P99 pres={obs_pres_10m_buf[3]:.2f}  fut={obs_fut_10m_buf[3]:.2f} mm/h")

    # Observed 10-min conditioned on wet HOURS — used for Method E validation.
    # Method E conditions on hourly_pres >= WET_HOUR_LOW; the comparison must
    # use the same conditioning or the synthetic will always overestimate
    # (light hours on wet days contribute low-intensity values to the wet-day
    # observed but are absent from the synthetic).
    def _obs_10m_wethr_ctr(blk_h6, hourly):
        """Center-pixel 10-min from wet hours (hourly mean >= WET_HOUR_LOW)."""
        wet_hr  = hourly[:, :, cy_loc, cx_loc] >= WET_HOUR_LOW   # (n_days, 24)
        wet_rep = np.repeat(wet_hr, N_HOUR_STEPS, axis=1).reshape(-1)   # (n_days*144,)
        flat    = blk_h6[:, :, :, cy_loc, cx_loc].reshape(-1)
        wet     = flat[(flat > WET_VALUE_HIGH) & wet_rep]
        return (np.percentile(wet, PLOT_QUANTILES * 100)
                if len(wet) > 0 else np.full(len(PLOT_QUANTILES), np.nan))

    def _obs_10m_wethr_buf(blk_h6, hourly):
        """Buffer-pooled 10-min from wet hours (hourly mean >= WET_HOUR_LOW)."""
        mask5d = (hourly >= WET_HOUR_LOW)[:, :, np.newaxis, :, :]   # (n_days,24,1,ny,nx)
        flat   = np.where(mask5d, blk_h6, 0.0).reshape(-1)
        wet    = flat[flat > WET_VALUE_HIGH]
        return (np.percentile(wet, PLOT_QUANTILES * 100)
                if len(wet) > 0 else np.full(len(PLOT_QUANTILES), np.nan))

    obs_pres_10m_E_ctr = _obs_10m_wethr_ctr(blk_pres_h6, hourly_pres)
    obs_fut_10m_E_ctr  = _obs_10m_wethr_ctr(blk_fut_h6,  hourly_fut)
    obs_pres_10m_E_buf = _obs_10m_wethr_buf(blk_pres_h6, hourly_pres)
    obs_fut_10m_E_buf  = _obs_10m_wethr_buf(blk_fut_h6,  hourly_fut)

    print(f"  Obs-E center: P99 pres={obs_pres_10m_E_ctr[3]:.2f}  fut={obs_fut_10m_E_ctr[3]:.2f} mm/h  (wet-hour filtered)")
    print(f"  Obs-E buffer: P99 pres={obs_pres_10m_E_buf[3]:.2f}  fut={obs_fut_10m_E_buf[3]:.2f} mm/h  (wet-hour filtered)")

    # ------------------------------------------------------------------
    # STEP 4a — Library D: analog 10-min profiles indexed by daily total
    # ------------------------------------------------------------------
    subsection("Building Library D  (10-min profiles | daily total)")
    t0 = time.time()

    wet_day_mask    = daily_pres >= WET_VALUE_LOW
    wet_day_mask_4d = wet_day_mask[:, np.newaxis, :, :]

    rain_10m_clipped = np.where(
        wet_day_mask_4d & (blk_pres >= WET_VALUE_HIGH),
        blk_pres, 0.0).astype(np.float32)                   # (n_days,144,ny,nx)

    n_wet_10m     = (rain_10m_clipped > WET_VALUE_HIGH).sum(axis=1)  # (n_days,ny,nx)
    has_wet_mask  = wet_day_mask & (n_wet_10m > 0)

    # p_has_wet per daily bin
    n_days_per_bin, _ = np.histogram(
        daily_pres[wet_day_mask].astype(np.float64), bins=BINS_LOW)
    n_days_wet_per_bin, _ = np.histogram(
        daily_pres[has_wet_mask].astype(np.float64), bins=BINS_LOW)
    p_has_wet_D = np.where(
        n_days_per_bin > 0,
        n_days_wet_per_bin / n_days_per_bin, 0.0).astype(np.float32)

    # Build library D (vectorized index extraction)
    b_day_all = np.searchsorted(BINS_LOW[1:], daily_pres).clip(0, nbins_low - 1)

    profiles_D_arrays = []
    profiles_D_totals = []
    for b in range(nbins_low):
        mask = has_wet_mask & (b_day_all == b)
        if mask.sum() > 0:
            days_i, iy_i, ix_i = np.where(mask)
            profs = rain_10m_clipped[days_i, :, iy_i, ix_i]   # (n_in_bin, 144)
            tots  = daily_pres[days_i, iy_i, ix_i]
            profiles_D_arrays.append(profs.copy())
            profiles_D_totals.append(tots.astype(np.float32))
        else:
            profiles_D_arrays.append(np.empty((0, N_INTERVAL), dtype=np.float32))
            profiles_D_totals.append(np.empty(0, dtype=np.float32))

    cnt_D = sum(len(a) for a in profiles_D_arrays)
    print(f"  Library D built: {elapsed(t0)}  |  {cnt_D:,} profiles")
    print(f"  Events per daily-total bin (Method D, present period):")
    print(f"  {'Bin (mm/d)':>14}  {'Wet days':>9}  {'With wet 10min':>15}  {'Profiles':>9}")
    for b in range(nbins_low):
        lo = BINS_LOW[b]
        hi = BINS_LOW[b + 1]
        bin_str = f"{lo:.3g}–{'inf' if np.isinf(hi) else f'{hi:.3g}'}"
        print(f"  {bin_str:>14}  {n_days_per_bin[b]:>9,}  "
              f"{n_days_wet_per_bin[b]:>15,}  {len(profiles_D_arrays[b]):>9,}")

    # ------------------------------------------------------------------
    # STEP 4b — Library E: analog 10-min-within-hour profiles indexed by hourly total
    # ------------------------------------------------------------------
    subsection("Building Library E  (10-min profiles | hourly total)")
    t0 = time.time()

    wet_hour_mask = hourly_pres >= WET_HOUR_LOW    # (n_days, 24, ny, nx)

    # p_has_wet per hourly bin (fraction of wet hours with any 10-min > WET_VALUE_HIGH)
    rain_h6_clipped = np.where(
        wet_hour_mask[:, :, np.newaxis, :, :] & (blk_pres_h6 >= WET_VALUE_HIGH),
        blk_pres_h6, 0.0).astype(np.float32)              # (n_days,24,6,ny,nx)
    n_wet_in_hour = (rain_h6_clipped > WET_VALUE_HIGH).sum(axis=2)  # (n_days,24,ny,nx)
    has_wet_hour_mask = wet_hour_mask & (n_wet_in_hour > 0)

    n_hrs_per_bin, _ = np.histogram(
        hourly_pres[wet_hour_mask].astype(np.float64), bins=BINS_HOUR)
    n_hrs_wet_per_bin, _ = np.histogram(
        hourly_pres[has_wet_hour_mask].astype(np.float64), bins=BINS_HOUR)
    p_has_wet_E = np.where(
        n_hrs_per_bin > 0,
        n_hrs_wet_per_bin / n_hrs_per_bin, 0.0).astype(np.float32)

    # Build library E: extract all wet-hour profiles at once (vectorized)
    b_hour_all = np.searchsorted(BINS_HOUR[1:], hourly_pres).clip(0, nbins_hour - 1)

    profiles_E_arrays = []
    profiles_E_totals = []
    for b in range(nbins_hour):
        mask = has_wet_hour_mask & (b_hour_all == b)
        if mask.sum() > 0:
            days_i, hrs_i, iy_i, ix_i = np.where(mask)
            # profiles_E shape per bin: (n_in_bin, 6)
            profs = rain_h6_clipped[days_i, hrs_i, :, iy_i, ix_i]
            tots  = hourly_pres[days_i, hrs_i, iy_i, ix_i]
            profiles_E_arrays.append(profs.copy())
            profiles_E_totals.append(tots.astype(np.float32))
        else:
            profiles_E_arrays.append(np.empty((0, N_HOUR_STEPS), dtype=np.float32))
            profiles_E_totals.append(np.empty(0, dtype=np.float32))

    cnt_E = sum(len(a) for a in profiles_E_arrays)
    print(f"  Library E built: {elapsed(t0)}  |  {cnt_E:,} profiles")
    print(f"  Events per hourly-total bin (Method E, present period):")
    print(f"  {'Bin (mm/h)':>14}  {'Wet hours':>9}  {'With wet 10min':>15}  {'Profiles':>9}")
    for b in range(nbins_hour):
        lo = BINS_HOUR[b]
        hi = BINS_HOUR[b + 1]
        bin_str = f"{lo:.3g}–{'inf' if np.isinf(hi) else f'{hi:.3g}'}"
        n_h  = int(n_hrs_per_bin[b])
        n_hw = int(n_hrs_wet_per_bin[b])
        if n_h == 0 and n_hw == 0:
            continue
        print(f"  {bin_str:>14}  {n_h:>9,}  {n_hw:>15,}  {len(profiles_E_arrays[b]):>9,}")

    # ------------------------------------------------------------------
    # STEP 5 — Bootstrap samplers
    # ------------------------------------------------------------------
    pq100 = PLOT_QUANTILES * 100.0
    n_q   = len(pq100)

    def _sample_D(rain_daily_1d, label, seed_base):
        """Method D — analog 10-min from daily totals."""
        print(f"\n  [Method D — {label}]  Wet days: {len(rain_daily_1d)}")
        t0_loop = time.time()

        def _one(s):
            rng = np.random.default_rng(seed_base + s)
            temp = []
            for R in rain_daily_1d:
                b = min(int(np.searchsorted(BINS_LOW[1:], R)), nbins_low - 1)
                if rng.random() >= p_has_wet_D[b]:
                    continue
                n_a = len(profiles_D_arrays[b])
                if n_a == 0:
                    continue
                idx     = rng.integers(0, n_a)
                R_a     = float(profiles_D_totals[b][idx])
                if R_a <= 0.0:
                    continue
                scaled  = profiles_D_arrays[b][idx] * (R / R_a)
                temp.extend(scaled[scaled > WET_VALUE_HIGH].tolist())
            return (np.percentile(np.array(temp, np.float32), pq100).astype(np.float32)
                    if temp else np.full(n_q, np.nan, np.float32))

        results = Parallel(n_jobs=N_JOBS)(delayed(_one)(s) for s in range(N_SAMPLES))
        print(f"  Total time: {elapsed(t0_loop)}")
        return np.array(results, dtype=np.float32)

    def _sample_E(rain_hourly_1d, label, seed_base):
        """Method E — analog 10-min from hourly totals."""
        print(f"\n  [Method E — {label}]  Wet hours: {len(rain_hourly_1d)}")
        t0_loop = time.time()

        def _one(s):
            rng = np.random.default_rng(seed_base + s)
            temp = []
            for R_h in rain_hourly_1d:
                b = min(int(np.searchsorted(BINS_HOUR[1:], R_h)), nbins_hour - 1)
                if rng.random() >= p_has_wet_E[b]:
                    continue
                n_a = len(profiles_E_arrays[b])
                if n_a == 0:
                    continue
                idx    = rng.integers(0, n_a)
                R_a    = float(profiles_E_totals[b][idx])
                if R_a <= 0.0:
                    continue
                scaled = profiles_E_arrays[b][idx] * (R_h / R_a)
                temp.extend(scaled[scaled > WET_VALUE_HIGH].tolist())
            return (np.percentile(np.array(temp, np.float32), pq100).astype(np.float32)
                    if temp else np.full(n_q, np.nan, np.float32))

        results = Parallel(n_jobs=N_JOBS)(delayed(_one)(s) for s in range(N_SAMPLES))
        print(f"  Total time: {elapsed(t0_loop)}")
        return np.array(results, dtype=np.float32)

    # Input sequences (wet days / wet hours only)
    pres_day_ctr_in  = pres_day_ctr[pres_day_ctr >= WET_VALUE_LOW]
    fut_day_ctr_in   = fut_day_ctr[ fut_day_ctr  >= WET_VALUE_LOW]
    pres_day_buf_in  = daily_pres[daily_pres >= WET_VALUE_LOW].reshape(-1)
    fut_day_buf_in   = daily_fut[ daily_fut  >= WET_VALUE_LOW].reshape(-1)

    pres_hr_ctr_in   = hourly_pres[:, :, cy_loc, cx_loc].reshape(-1)
    pres_hr_ctr_in   = pres_hr_ctr_in[pres_hr_ctr_in >= WET_HOUR_LOW]
    fut_hr_ctr_in    = hourly_fut[:, :, cy_loc, cx_loc].reshape(-1)
    fut_hr_ctr_in    = fut_hr_ctr_in[fut_hr_ctr_in >= WET_HOUR_LOW]
    pres_hr_buf_in   = hourly_pres[hourly_pres >= WET_HOUR_LOW].reshape(-1)
    fut_hr_buf_in    = hourly_fut[ hourly_fut  >= WET_HOUR_LOW].reshape(-1)

    subsection("Bootstrap sampling")

    D_pres_boot     = _sample_D(pres_day_ctr_in,  "PRESENT (center)",  seed_base=100)
    D_fut_boot      = _sample_D(fut_day_ctr_in,   "FUTURE  (center)",  seed_base=200)
    D_pres_boot_buf = _sample_D(pres_day_buf_in,  "PRESENT (buffer)",  seed_base=300)
    D_fut_boot_buf  = _sample_D(fut_day_buf_in,   "FUTURE  (buffer)",  seed_base=400)

    E_pres_boot     = _sample_E(pres_hr_ctr_in,   "PRESENT (center)",  seed_base=500)
    E_fut_boot      = _sample_E(fut_hr_ctr_in,    "FUTURE  (center)",  seed_base=600)
    E_pres_boot_buf = _sample_E(pres_hr_buf_in,   "PRESENT (buffer)",  seed_base=700)
    E_fut_boot_buf  = _sample_E(fut_hr_buf_in,    "FUTURE  (buffer)",  seed_base=800)

    D_pres_ci     = ci_across_samples(D_pres_boot)
    D_fut_ci      = ci_across_samples(D_fut_boot)
    D_pres_ci_buf = ci_across_samples(D_pres_boot_buf)
    D_fut_ci_buf  = ci_across_samples(D_fut_boot_buf)

    E_pres_ci     = ci_across_samples(E_pres_boot)
    E_fut_ci      = ci_across_samples(E_fut_boot)
    E_pres_ci_buf = ci_across_samples(E_pres_boot_buf)
    E_fut_ci_buf  = ci_across_samples(E_fut_boot_buf)

    # ------------------------------------------------------------------
    # STEP 6 — Combined figure: 2 rows × 3 cols
    #   Row 0: center pixel  — Method D | Method E | comparison
    #   Row 1: buffer pooled — Method D | Method E | comparison + P99 annotation
    # ------------------------------------------------------------------
    YLABEL = '10-min precipitation (mm/h)'
    idx_99 = np.searchsorted(q_axis, 0.99)

    fig, axes = plt.subplots(2, 3, figsize=(20, 11))

    # Each row tuple: (wet-day obs pres, wet-day obs fut,
    #                  wet-hr obs pres [for E], wet-hr obs fut [for E],
    #                  D_pres_ci, D_fut_ci, E_pres_ci, E_fut_ci, label)
    rows = [
        (obs_pres_10m,     obs_fut_10m,
         obs_pres_10m_E_ctr, obs_fut_10m_E_ctr,
         D_pres_ci,     D_fut_ci,
         E_pres_ci,     E_fut_ci,
         f'center pixel'),
        (obs_pres_10m_buf, obs_fut_10m_buf,
         obs_pres_10m_E_buf, obs_fut_10m_E_buf,
         D_pres_ci_buf, D_fut_ci_buf,
         E_pres_ci_buf, E_fut_ci_buf,
         f'buffer ({ny_reg}×{nx_reg})'),
    ]

    for row, (obs_p, obs_f,
              obs_p_E, obs_f_E,
              D_pres_ci_r, D_fut_ci_r,
              E_pres_ci_r, E_fut_ci_r,
              row_lbl) in enumerate(rows):

        # Col 0 — Method D (from daily) — compared against wet-day observations
        _plot_panel(axes[row, 0], obs_p, obs_f, D_pres_ci_r, D_fut_ci_r,
                    'Synthetic present (from daily)',
                    'Synthetic future (from daily)',
                    YLABEL,
                    f'From daily — {row_lbl}\n{location}, {buf_str}')

        # Col 1 — Method E (from hourly) — compared against wet-HOUR observations
        # (consistent with Method E's conditioning: hours with mean >= WET_HOUR_LOW)
        _plot_panel(axes[row, 1], obs_p_E, obs_f_E, E_pres_ci_r, E_fut_ci_r,
                    'Synthetic present (from hourly)',
                    'Synthetic future (from hourly)',
                    YLABEL,
                    f'From hourly — {row_lbl}\n{location}, {buf_str}\n'
                    f'[obs: wet-hour filtered]')

        # Col 2 — both future CIs vs wet-day observations (full change signal)
        ax = axes[row, 2]
        ax.plot(q_axis, obs_p, color='#2E86AB', linewidth=1.8,
                marker='o', markersize=5, label='Present observed (wet-day)', zorder=4)
        ax.plot(q_axis, obs_f, color='#E50C0C', linewidth=1.5,
                marker='s', markersize=4, linestyle='--',
                label='Future observed (wet-day)', zorder=3)
        _add_ci(ax, D_fut_ci_r, '#F18F01', 'Synthetic future (from daily)',  alpha=0.25)
        _add_ci(ax, E_fut_ci_r, '#9B59B6', 'Synthetic future (from hourly)', alpha=0.25)
        _setup_ax(ax, YLABEL,
                  f'Comparison — {row_lbl}\n{location}, {buf_str}')
        _dedup_legend(ax, fontsize=8, loc='upper left', frameon=True,
                      fancybox=True, shadow=True)

        # P99 annotation on bottom row:
        #   col 0 (D) uses wet-day obs; col 1 (E) uses wet-hour obs
        if row == 1 and idx_99 < len(q_axis) and np.isclose(
                q_axis[idx_99], 0.99, atol=0.001):
            for ax_i, ci_fut, op_i, of_i in [
                    (axes[1, 0], D_fut_ci_r, obs_p,   obs_f),
                    (axes[1, 1], E_fut_ci_r, obs_p_E, obs_f_E)]:
                total, expl, struct, pct_d, pct_s = _attr_row(
                    float(op_i[idx_99]),
                    float(of_i[idx_99]),
                    float(ci_fut[I_MED, idx_99]))
                ax_i.text(0.03, 0.04,
                          f'P99 total Δ : {total:+.2f} mm/h\n'
                          f'Explained  : {expl:+.2f} mm/h  ({pct_d:.0f}%)\n'
                          f'Structural : {struct:+.2f} mm/h  ({pct_s:.0f}%)',
                          transform=ax_i.transAxes, fontsize=8.5,
                          verticalalignment='bottom',
                          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.75))

    fig.suptitle(
        f'10-min attribution — {location}  |  {buf_str}\n'
        f'Rows: center pixel / buffer pooled   '
        f'Cols: from daily / from hourly / comparison\n'
        f'Blue shading: synthetic present   '
        f'Orange: synthetic future from daily   '
        f'Purple: synthetic future from hourly',
        fontsize=11, fontweight='bold')
    fig.tight_layout()
    out_fig = f'{PATH_OUT}/attribution_10min_{run_label}.png'
    fig.savefig(out_fig, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"\n  Saved: {out_fig}")

    # ------------------------------------------------------------------
    # STEP 7 — Attribution summary tables
    # ------------------------------------------------------------------
    subsection("Attribution summary (buffer pooled)")
    # Method D uses wet-day observed; Method E uses wet-hour observed
    # (consistent with each method's conditioning population)
    attr_rows = [
        ('Method D (from daily) ', D_fut_ci_buf, obs_pres_10m_buf,   obs_fut_10m_buf,   '(wet-day obs)'),
        ('Method E (from hourly)', E_fut_ci_buf, obs_pres_10m_E_buf, obs_fut_10m_E_buf, '(wet-hour obs)'),
    ]
    for method, ci_fut, op_buf, of_buf, obs_note in attr_rows:
        print(f"\n  [{method}]  {obs_note}")
        hdr = (f"  {'Quantile':>10}  {'Obs_Pres':>10}  {'Obs_Fut':>10}  "
               f"{'Syn_Fut':>10}  {'Total Δ':>10}  {'Explained':>10}  "
               f"{'Structural':>11}  {'Expl%':>8}  {'Struct%':>8}")
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))
        for i, q in enumerate(PLOT_QUANTILES):
            total, expl, struct, pct_d, pct_s = _attr_row(
                float(op_buf[i]),
                float(of_buf[i]),
                float(ci_fut[I_MED, i]))
            print(f"  {q:>10.4f}  {op_buf[i]:>10.3f}  "
                  f"{of_buf[i]:>10.3f}  {ci_fut[I_MED,i]:>10.3f}  "
                  f"{total:>+10.3f}  {expl:>+10.3f}  {struct:>+11.3f}  "
                  f"{pct_d:>+8.1f}%  {pct_s:>+8.1f}%")

    # ------------------------------------------------------------------
    # SAVE BOOTSTRAP DATA
    # ------------------------------------------------------------------
    npz_path = os.path.join(PATH_OUT, 'testing_data', f'{run_label}.npz')
    np.savez_compressed(npz_path,
        plot_quantiles      = PLOT_QUANTILES,
        bootstrap_quantiles = BOOTSTRAP_QUANTILES,
        buffer              = np.array(buffer),
        n_samples           = np.array(N_SAMPLES),
        wet_value_high      = np.array(WET_VALUE_HIGH),
        wet_value_low       = np.array(WET_VALUE_LOW),
        wet_hour_low        = np.array(WET_HOUR_LOW),
        n_interval          = np.array(N_INTERVAL),
        n_days_pres         = np.array(n_days_pres),
        n_days_fut          = np.array(n_days_fut),
        ny_reg              = np.array(ny_reg),
        nx_reg              = np.array(nx_reg),
        bins_low            = BINS_LOW,
        bins_hour           = BINS_HOUR,
        n_days_per_bin      = n_days_per_bin.astype(np.int32),
        n_days_wet_per_bin  = n_days_wet_per_bin.astype(np.int32),
        n_hrs_per_bin       = n_hrs_per_bin.astype(np.int32),
        n_hrs_wet_per_bin   = n_hrs_wet_per_bin.astype(np.int32),
        # observed — wet-day filtered (used by Method D)
        obs_pres_10m        = obs_pres_10m,
        obs_fut_10m         = obs_fut_10m,
        obs_pres_10m_buf    = obs_pres_10m_buf,
        obs_fut_10m_buf     = obs_fut_10m_buf,
        # observed — wet-hour filtered (used by Method E)
        obs_pres_10m_E_ctr  = obs_pres_10m_E_ctr,
        obs_fut_10m_E_ctr   = obs_fut_10m_E_ctr,
        obs_pres_10m_E_buf  = obs_pres_10m_E_buf,
        obs_fut_10m_E_buf   = obs_fut_10m_E_buf,
        # Method D bootstrap (N_SAMPLES, n_q)
        D_pres_boot         = D_pres_boot,
        D_fut_boot          = D_fut_boot,
        D_pres_boot_buf     = D_pres_boot_buf,
        D_fut_boot_buf      = D_fut_boot_buf,
        # Method E bootstrap (N_SAMPLES, n_q)
        E_pres_boot         = E_pres_boot,
        E_fut_boot          = E_fut_boot,
        E_pres_boot_buf     = E_pres_boot_buf,
        E_fut_boot_buf      = E_fut_boot_buf,
    )
    print(f"\n  Saved bootstrap data : {npz_path}")

    return dict(location=location, buffer=buffer,
                obs_pres_10m_buf=obs_pres_10m_buf,
                obs_fut_10m_buf=obs_fut_10m_buf,
                D_fut_ci_buf=D_fut_ci_buf,
                E_fut_ci_buf=E_fut_ci_buf)


# =============================================================================
# MAIN LOOP
# =============================================================================

t_total = time.time()
combos  = [(loc, buf) for loc in LOCATIONS for buf in BUFFERS]
print(f"\nRunning {len(combos)} combinations (10-min pipeline):")
for i, (loc, buf) in enumerate(combos):
    print(f"  [{i+1}/{len(combos)}]  {loc}  buffer={buf}")

all_results = []
for loc, buf in combos:
    try:
        all_results.append(run_single(loc, buf))
    except Exception as exc:
        print(f"\n  ERROR in {loc} buf={buf}: {exc}")
        import traceback
        traceback.print_exc()

print(f"\n{'='*70}")
print(f"  All done.  Total: {(time.time()-t_total)/3600:.2f} h")
print(f"{'='*70}")
