#!/usr/bin/env python
"""
Multi-location, multi-buffer testing pipeline for synthetic hourly rainfall.

Runs the single-location pipeline logic for every combination of
LOCATIONS × BUFFERS, generates the same plots as pipeline_single_location.py,
and saves bootstrap data (both methods, all runs) to
  PATH_OUT/testing_data/{location}_buf{buffer}.npz

Bootstrap arrays have shape (N_SAMPLES, n_plot_quantiles) and can be reloaded
with np.load() for further analysis without re-running the pipeline.
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
# CONFIGURATION  —  edit only this section
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

LOCATIONS = ['Mallorca', 'Catania', 'Turis','Rosiglione','Ardeche','Corte',"L'Aquila", 'Pyrenees']   # locations to run
BUFFERS   = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]              # buffer sizes (grid cells; 0 = center pixel only)

WET_VALUE_HIGH = 0.1   # mm/h
WET_VALUE_LOW  = 1.0   # mm/d

PLOT_QUANTILES      = np.array([0.90, 0.95, 0.98, 0.99, 0.995, 0.999])
BINS_HIGH           = np.append(np.arange(0, 100, 1), np.inf)
BINS_LOW            = np.append(np.arange(0, 100, 5), np.inf)

N_SAMPLES           = 1000
BOOTSTRAP_QUANTILES = np.array([0.025, 0.5, 0.975])
EXP_SCALE           = 5.0
N_INTERVAL          = 24

# Set True to also run Method A (slow, overestimates, kept for diagnostics only)
RUN_METHOD_A = False

N_JOBS = -1   # parallel workers for bootstrap (-1 = all cores, 1 = sequential)

os.makedirs(PATH_OUT, exist_ok=True)
os.makedirs(os.path.join(PATH_OUT, 'testing_data'), exist_ok=True)


# =============================================================================
# MODULE-LEVEL DERIVED CONSTANTS  (depend only on config, never change per run)
# =============================================================================

q_axis    = PLOT_QUANTILES
nbins_low = len(BINS_LOW) - 1
I_LO, I_MED, I_HI = 0, 1, 2
cl_lo = BOOTSTRAP_QUANTILES[I_LO]
cl_hi = BOOTSTRAP_QUANTILES[I_HI]


# =============================================================================
# STATELESS HELPER UTILITIES
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


def _sample_from_bin_val(lo, hi, rng):
    if np.isinf(hi):
        return lo + rng.exponential(EXP_SCALE)
    return lo + (hi - lo) * rng.random()


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
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.grid(True, which='both', linestyle=':', linewidth=0.5, alpha=0.7)
    ax.set_axisbelow(True)
    ax.minorticks_on()
    ax.set_yticks([1, 2, 5, 10, 20, 50, 100])
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.get_major_formatter().set_scientific(False)
    ax.tick_params(axis='both', which='major', labelsize=10)


def _dedup_legend(ax, **kwargs):
    handles, labels = ax.get_legend_handles_labels()
    seen, uh, ul = {}, [], []
    for h, l in zip(handles, labels):
        if l not in seen:
            seen[l] = True
            uh.append(h)
            ul.append(l)
    ax.legend(uh, ul, **kwargs)


def _add_ci(ax, ci, color, label, linestyle='-', alpha=0.2):
    ax.plot(q_axis, ci[I_MED], color=color, linewidth=1.8,
            linestyle=linestyle, label=label, zorder=5)
    ax.fill_between(q_axis, ci[I_LO], ci[I_HI], color=color, alpha=alpha)


def _plot_panel(ax, obs_pres, obs_fut, ci_lo, ci_med, ci_hi,
                syn_label, ylabel, title):
    ax.plot(q_axis, obs_pres, color='#2E86AB', linewidth=1.8, marker='o',
            markersize=5, label='Present observed', zorder=4)
    ax.plot(q_axis, obs_fut,  color='#E50C0C',  linewidth=1.5, marker='s',
            markersize=4, linestyle='--', label='Future observed', zorder=3)
    ax.plot(q_axis, ci_med,   color='#F18F01',  linewidth=1.8,
            label=syn_label, zorder=5)
    ax.plot(q_axis, ci_lo,    color='#F18F01',  linewidth=0.8, linestyle=':')
    ax.plot(q_axis, ci_hi,    color='#F18F01',  linewidth=0.8, linestyle=':')
    ax.fill_between(q_axis, ci_lo, ci_hi, color='#F18F01', alpha=0.2,
                    label=f'Synthetic CI ({cl_lo}–{cl_hi})')
    _setup_ax(ax, ylabel, title)
    _dedup_legend(ax, fontsize=9, loc='upper left', frameon=True,
                  fancybox=True, shadow=True)


# =============================================================================
# PER-(LOCATION, BUFFER) PIPELINE
# =============================================================================

def run_single(location, buffer):
    """Run the full pipeline for one (location, buffer) combination."""

    run_label  = f"{location}_buf{buffer}"
    buf_str    = f"buf={buffer}"
    TARGET_LAT = loc_lats[location]
    TARGET_LON = loc_lons[location]

    section(f"{run_label}  —  lat={TARGET_LAT}, lon={TARGET_LON}")

    # ------------------------------------------------------------------
    # STEP 1 — Load data, locate pixel, extract buffer region
    # ------------------------------------------------------------------
    zarr_pres = f'{PATH_IN}/{WRUN_PRESENT}/UIB_01H_RAIN.zarr'
    zarr_fut  = f'{PATH_IN}/{WRUN_FUTURE}/UIB_01H_RAIN.zarr'

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
    ny_reg,  nx_reg  = y1 - y0, x1 - x0

    print(f"  Nearest pixel : y={cy}, x={cx}  "
          f"({lat2d[cy,cx]:.4f} / {lon2d[cy,cx]:.4f})")
    print(f"  Region        : {ny_reg}×{nx_reg} pixels, "
          f"center=[{cy_loc},{cx_loc}]")

    t0 = time.time()
    rain_pres = ds_pres.RAIN.isel(
        y=slice(y0, y1), x=slice(x0, x1)).values.astype(np.float32)
    rain_fut = ds_fut.RAIN.isel(
        y=slice(y0, y1), x=slice(x0, x1)).values.astype(np.float32)
    ds_pres.close()
    ds_fut.close()
    print(f"  Load time     : {elapsed(t0)}")

    # ------------------------------------------------------------------
    # STEP 2 — Resample hourly → daily (24-hour blocks)
    # ------------------------------------------------------------------
    n_days_pres = rain_pres.shape[0] // N_INTERVAL
    n_days_fut  = rain_fut.shape[0]  // N_INTERVAL

    blk_pres = rain_pres[:n_days_pres * N_INTERVAL].reshape(
        n_days_pres, N_INTERVAL, ny_reg, nx_reg)
    blk_fut  = rain_fut[:n_days_fut * N_INTERVAL].reshape(
        n_days_fut,  N_INTERVAL, ny_reg, nx_reg)

    daily_pres = blk_pres.sum(axis=1)
    daily_fut  = blk_fut.sum(axis=1)

    pres_1h_ctr  = rain_pres[:n_days_pres * N_INTERVAL, cy_loc, cx_loc]
    fut_1h_ctr   = rain_fut[:n_days_fut  * N_INTERVAL,  cy_loc, cx_loc]
    pres_day_ctr = daily_pres[:, cy_loc, cx_loc]
    fut_day_ctr  = daily_fut[:,  cy_loc, cx_loc]

    print(f"  Present : {n_days_pres} days,  "
          f"wet days (>={WET_VALUE_LOW}mm): {int((pres_day_ctr>=WET_VALUE_LOW).sum())}")
    print(f"  Future  : {n_days_fut} days,  "
          f"wet days (>={WET_VALUE_LOW}mm): {int((fut_day_ctr>=WET_VALUE_LOW).sum())}")

    # ------------------------------------------------------------------
    # STEP 3 — Observed percentiles (center pixel + buffer-pooled)
    # ------------------------------------------------------------------

    # Center pixel hourly — wet-day hours only
    pres_wet_day_h = np.repeat(pres_day_ctr >= WET_VALUE_LOW, N_INTERVAL)
    fut_wet_day_h  = np.repeat(fut_day_ctr  >= WET_VALUE_LOW, N_INTERVAL)
    pres_wet_1h = pres_1h_ctr[(pres_1h_ctr > WET_VALUE_HIGH) & pres_wet_day_h]
    fut_wet_1h  = fut_1h_ctr[ (fut_1h_ctr  > WET_VALUE_HIGH) & fut_wet_day_h]
    obs_pres_h  = np.percentile(pres_wet_1h, PLOT_QUANTILES * 100)
    obs_fut_h   = np.percentile(fut_wet_1h,  PLOT_QUANTILES * 100)

    # Center pixel daily-max
    pres_dmax_ctr  = blk_pres[:, :, cy_loc, cx_loc].max(axis=1)
    fut_dmax_ctr   = blk_fut[:,  :, cy_loc, cx_loc].max(axis=1)
    pres_n_wet_ctr = (blk_pres[:, :, cy_loc, cx_loc] > WET_VALUE_HIGH).sum(axis=1)
    fut_n_wet_ctr  = (blk_fut[:,  :, cy_loc, cx_loc] > WET_VALUE_HIGH).sum(axis=1)
    pres_dmax_mask = (pres_day_ctr >= WET_VALUE_LOW) & (pres_n_wet_ctr > 0)
    fut_dmax_mask  = (fut_day_ctr  >= WET_VALUE_LOW) & (fut_n_wet_ctr  > 0)
    obs_pres_dm    = np.percentile(pres_dmax_ctr[pres_dmax_mask], PLOT_QUANTILES * 100)
    obs_fut_dm     = np.percentile(fut_dmax_ctr[fut_dmax_mask],   PLOT_QUANTILES * 100)

    # Buffer hourly — wet-day hours only
    pres_wdm_buf = (daily_pres >= WET_VALUE_LOW)[:, np.newaxis, :, :]
    fut_wdm_buf  = (daily_fut  >= WET_VALUE_LOW)[:, np.newaxis, :, :]
    pres_wet_1h_buf = (blk_pres * pres_wdm_buf).reshape(-1)
    pres_wet_1h_buf = pres_wet_1h_buf[pres_wet_1h_buf > WET_VALUE_HIGH]
    fut_wet_1h_buf  = (blk_fut * fut_wdm_buf).reshape(-1)
    fut_wet_1h_buf  = fut_wet_1h_buf[fut_wet_1h_buf > WET_VALUE_HIGH]
    obs_pres_h_buf  = np.percentile(pres_wet_1h_buf, PLOT_QUANTILES * 100)
    obs_fut_h_buf   = np.percentile(fut_wet_1h_buf,  PLOT_QUANTILES * 100)

    # Buffer daily-max
    pres_dmax_all      = blk_pres.max(axis=1)
    fut_dmax_all       = blk_fut.max(axis=1)
    pres_n_wet_all     = (blk_pres > WET_VALUE_HIGH).sum(axis=1)
    fut_n_wet_all      = (blk_fut  > WET_VALUE_HIGH).sum(axis=1)
    pres_dmax_mask_buf = (daily_pres >= WET_VALUE_LOW) & (pres_n_wet_all > 0)
    fut_dmax_mask_buf  = (daily_fut  >= WET_VALUE_LOW) & (fut_n_wet_all  > 0)
    obs_pres_dm_buf    = np.percentile(
        pres_dmax_all[pres_dmax_mask_buf], PLOT_QUANTILES * 100)
    obs_fut_dm_buf     = np.percentile(
        fut_dmax_all[fut_dmax_mask_buf],   PLOT_QUANTILES * 100)

    print(f"  Obs center pix : {len(pres_wet_1h):,} pres wet hrs, "
          f"{len(fut_wet_1h):,} fut wet hrs")
    print(f"  Obs buffer     : {len(pres_wet_1h_buf):,} pres wet hrs, "
          f"{len(fut_wet_1h_buf):,} fut wet hrs")

    # ------------------------------------------------------------------
    # STEP 4 — Conditional probabilities (vectorized) + analog library
    # ------------------------------------------------------------------
    nbins_high = len(BINS_HIGH) - 1
    n_wet_bins = np.arange(0.5, N_INTERVAL + 1.5, 1)

    # --- Vectorized histogram accumulation ---
    subsection("Building conditional probability histograms (vectorized)")
    t0 = time.time()

    wet_day_mask    = daily_pres >= WET_VALUE_LOW           # (n_days, ny, nx)
    wet_day_mask_4d = wet_day_mask[:, np.newaxis, :, :]     # (n_days, 1, ny, nx)

    # Zero out dry-day hours and below-threshold hours in one shot
    rain_h_clipped = np.where(
        wet_day_mask_4d & (blk_pres >= WET_VALUE_HIGH),
        blk_pres, 0.0).astype(np.float32)                   # (n_days, 24, ny, nx)

    # n_wet per pixel-day; valid = wet day with at least one wet hour
    n_wet_all          = (rain_h_clipped > WET_VALUE_HIGH).sum(axis=1)  # (n_days, ny, nx)
    has_wet_hours_mask = wet_day_mask & (n_wet_all > 0)                  # (n_days, ny, nx)

    # Broadcast daily totals to hourly shape for histogram 1
    d_rep_4d = np.repeat(
        daily_pres[:, np.newaxis, :, :].astype(np.float64),
        N_INTERVAL, axis=1)                                 # (n_days, 24, ny, nx)

    # Histogram 1: P(hourly intensity | daily total)
    h_flat = rain_h_clipped.reshape(-1).astype(np.float64)
    d_flat = d_rep_4d.reshape(-1)
    wet_h  = h_flat >= WET_VALUE_HIGH
    hist_intensity, _, _ = np.histogram2d(
        h_flat[wet_h], d_flat[wet_h], bins=[BINS_HIGH, BINS_LOW])
    del d_rep_4d, h_flat, d_flat, wet_h

    # Histograms 2 & 3 share the same set of valid pixel-days
    d_valid = daily_pres[has_wet_hours_mask].astype(np.float64)   # 1-D

    # Histogram 2: P(n_wet | daily total)
    hist_n_wet, _, _ = np.histogram2d(
        n_wet_all[has_wet_hours_mask].astype(np.float64), d_valid,
        bins=[n_wet_bins, BINS_LOW])

    # Histogram 3: P(daily-max | daily total)
    day_max_all = rain_h_clipped.max(axis=1)                      # (n_days, ny, nx)
    hist_max_intens, _, _ = np.histogram2d(
        day_max_all[has_wet_hours_mask].astype(np.float64), d_valid,
        bins=[BINS_HIGH, BINS_LOW])

    # Count events per daily bin
    n_days_all_by_bin, _ = np.histogram(
        daily_pres[wet_day_mask].astype(np.float64), bins=BINS_LOW)
    n_days_wet_hrs_by_bin, _ = np.histogram(d_valid, bins=BINS_LOW)
    n_days_all_by_bin     = n_days_all_by_bin.astype(np.int32)
    n_days_wet_hrs_by_bin = n_days_wet_hrs_by_bin.astype(np.int32)

    print(f"  Histogram accumulation: {elapsed(t0)}")

    # Normalize → PDFs → CDFs
    intens_pdf  = np.zeros_like(hist_intensity)
    n_wet_pdf   = np.zeros_like(hist_n_wet)
    max_int_pdf = np.zeros_like(hist_max_intens)
    for j in range(nbins_low):
        si = hist_intensity[:, j].sum()
        if si > 0: intens_pdf[:, j]  = hist_intensity[:, j]  / si
        sn = hist_n_wet[:, j].sum()
        if sn > 0: n_wet_pdf[:, j]   = hist_n_wet[:, j]      / sn
        sm = hist_max_intens[:, j].sum()
        if sm > 0: max_int_pdf[:, j] = hist_max_intens[:, j] / sm

    wet_cdf  = np.cumsum(n_wet_pdf.T,   axis=1).astype(np.float32)
    hour_cdf = np.cumsum(intens_pdf.T,  axis=1).astype(np.float32)
    max_cdf  = np.cumsum(max_int_pdf.T, axis=1).astype(np.float32)

    p_has_wet_hours = np.where(
        n_days_all_by_bin > 0,
        n_days_wet_hrs_by_bin / n_days_all_by_bin, 0.0).astype(np.float32)

    # --- Analog profile library ---
    # The nested loop is kept to preserve insertion order (same random analog
    # selected per seed as in the original code → identical bootstrap results).
    subsection("Building analog profile library")
    t0 = time.time()
    profiles_by_bin = [[] for _ in range(nbins_low)]
    b_all = np.searchsorted(BINS_LOW[1:], daily_pres).clip(0, nbins_low - 1)  # (n_days, ny, nx)

    for iy in range(ny_reg):
        for ix in range(nx_reg):
            valid_mask = has_wet_hours_mask[:, iy, ix]
            if not valid_mask.any():
                continue
            for k in np.where(valid_mask)[0]:
                b = int(b_all[k, iy, ix])
                profiles_by_bin[b].append(
                    (rain_h_clipped[k, :, iy, ix].copy(),
                     float(daily_pres[k, iy, ix])))

    # Convert lists → numpy arrays for fast indexed access in bootstrap
    profiles_arrays = []
    profiles_totals = []
    for b in range(nbins_low):
        if profiles_by_bin[b]:
            profiles_arrays.append(
                np.array([p for p, _ in profiles_by_bin[b]], dtype=np.float32))
            profiles_totals.append(
                np.array([R for _, R in profiles_by_bin[b]], dtype=np.float32))
        else:
            profiles_arrays.append(np.empty((0, N_INTERVAL), dtype=np.float32))
            profiles_totals.append(np.empty(0, dtype=np.float32))
    del profiles_by_bin

    profile_counts = [len(profiles_arrays[j]) for j in range(nbins_low)]
    print(f"  Library built: {elapsed(t0)}")
    print(f"  Total analog profiles : {sum(profile_counts):,}  "
          f"(bins with ≥1 profile: "
          f"{sum(1 for c in profile_counts if c > 0)}/{nbins_low})")
    print(f"  Events per daily-total bin (present period):")
    print(f"  {'Bin (mm/d)':>14}  {'Wet days':>9}  {'With wet hrs':>13}  {'Profiles':>9}")
    for j in range(nbins_low):
        lo = BINS_LOW[j]
        hi = BINS_LOW[j + 1]
        bin_str = f"{lo:.0f}–{'inf' if np.isinf(hi) else f'{hi:.0f}'}"
        print(f"  {bin_str:>14}  {n_days_all_by_bin[j]:>9,}  "
              f"{n_days_wet_hrs_by_bin[j]:>13,}  {profile_counts[j]:>9,}")

    # ------------------------------------------------------------------
    # SAMPLER FUNCTIONS  (defined here to close over run-specific state)
    # ------------------------------------------------------------------
    hr_edges = BINS_HIGH.astype(np.float64)
    pq100    = PLOT_QUANTILES * 100.0
    n_q      = len(pq100)

    def _progress(sample, t0_loop):
        if (sample + 1) % 200 == 0 or sample == N_SAMPLES - 1:
            rate = (sample + 1) / (time.time() - t0_loop)
            print(f"    {sample+1:4d}/{N_SAMPLES}  —  {rate:.1f} samples/s")

    def generate_synthetic_quantiles(rain_daily_1d, label, seed_base):
        """Method A — parametric hourly sampler."""
        hourly_q_boot = np.full((N_SAMPLES, n_q), np.nan, dtype=np.float32)
        max_q_boot    = np.full((N_SAMPLES, n_q), np.nan, dtype=np.float32)
        print(f"\n  [Method A — {label}]  "
              f"Wet days: {int((rain_daily_1d > 0).sum())}")
        t0_loop = time.time()
        for sample in range(N_SAMPLES):
            rng = np.random.default_rng(seed_base + sample)
            temp_hourly, temp_max = [], []
            for R in rain_daily_1d:
                if R <= 0.0 or not np.isfinite(R):
                    continue
                b = min(int(np.searchsorted(BINS_LOW[1:], R)),
                        wet_cdf.shape[0] - 1)
                if rng.random() >= p_has_wet_hours[b]:
                    continue
                wet_slice = wet_cdf[b, :]
                if not np.any(wet_slice > 0):
                    continue
                Nh = max(1, min(
                    int(np.searchsorted(wet_slice, rng.random())) + 1,
                    N_INTERVAL))
                hour_slice  = hour_cdf[b, :]
                intensities = np.empty(Nh, dtype=np.float64)
                for k in range(Nh):
                    idx = min(int(np.searchsorted(hour_slice, rng.random())),
                              nbins_high - 1)
                    intensities[k] = _sample_from_bin_val(
                        hr_edges[idx], hr_edges[idx + 1], rng)
                total = float(intensities.sum())
                intensities = (intensities * (R / total) if total > 0
                               else np.full(Nh, R / Nh))
                temp_max.append(float(intensities.max()))
                temp_hourly.extend(intensities.tolist())
            if temp_hourly:
                arr_h = np.array(temp_hourly, dtype=np.float32)
                wet_h = arr_h[arr_h > WET_VALUE_HIGH]
                if len(wet_h) > 0:
                    hourly_q_boot[sample, :] = np.percentile(wet_h, pq100)
            if temp_max:
                arr_m = np.array(temp_max, dtype=np.float32)
                wet_m = arr_m[arr_m > WET_VALUE_HIGH]
                if len(wet_m) > 0:
                    max_q_boot[sample, :] = np.percentile(wet_m, pq100)
            _progress(sample, t0_loop)
        return hourly_q_boot, max_q_boot

    def generate_synthetic_max_quantiles(rain_daily_1d, label, seed_base):
        """Method B — direct daily-max sampler, parallelized over samples."""
        print(f"\n  [Method B — {label}]  Wet days: {len(rain_daily_1d)}")
        t0_loop = time.time()

        def _one_sample(sample):
            rng = np.random.default_rng(seed_base + sample)
            temp_max = []
            for R in rain_daily_1d:
                if not np.isfinite(R):
                    continue
                b = min(int(np.searchsorted(BINS_LOW[1:], R)),
                        max_cdf.shape[0] - 1)
                if rng.random() >= p_has_wet_hours[b]:
                    continue
                max_slice = max_cdf[b, :]
                if not np.any(max_slice > 0):
                    continue
                max_bin = min(int(np.searchsorted(max_slice, rng.random())),
                              nbins_high - 1)
                max_val = float(np.clip(
                    _sample_from_bin_val(hr_edges[max_bin],
                                         hr_edges[max_bin + 1], rng),
                    WET_VALUE_HIGH, R))
                temp_max.append(max_val)
            if temp_max:
                arr_m = np.array(temp_max, dtype=np.float32)
                wet_m = arr_m[arr_m > WET_VALUE_HIGH]
                if len(wet_m) > 0:
                    return np.percentile(wet_m, pq100).astype(np.float32)
            return np.full(n_q, np.nan, dtype=np.float32)

        results   = Parallel(n_jobs=N_JOBS)(
            delayed(_one_sample)(s) for s in range(N_SAMPLES))
        max_q_boot = np.array(results, dtype=np.float32)
        print(f"  Total time: {elapsed(t0_loop)}")
        return max_q_boot

    def generate_analog_quantiles(rain_daily_1d, label, seed_base):
        """Method C — analog resampling, parallelized over samples."""
        print(f"\n  [Method C — {label}]  Wet days: {len(rain_daily_1d)}")
        t0_loop = time.time()

        def _one_sample(sample):
            rng = np.random.default_rng(seed_base + sample)
            temp_hourly, temp_max = [], []
            for R in rain_daily_1d:
                b = min(int(np.searchsorted(BINS_LOW[1:], R)), nbins_low - 1)
                if rng.random() >= p_has_wet_hours[b]:
                    continue
                n_analogs = len(profiles_arrays[b])
                if n_analogs == 0:
                    continue
                idx      = rng.integers(0, n_analogs)
                R_analog = float(profiles_totals[b][idx])
                if R_analog <= 0.0:
                    continue
                scaled = profiles_arrays[b][idx] * (R / R_analog)
                temp_max.append(float(scaled.max()))
                temp_hourly.extend(scaled[scaled > WET_VALUE_HIGH].tolist())
            h_row = (np.percentile(np.array(temp_hourly, dtype=np.float32), pq100)
                     if temp_hourly else np.full(n_q, np.nan, dtype=np.float32))
            if temp_max:
                arr_m = np.array(temp_max, dtype=np.float32)
                wet_m = arr_m[arr_m > WET_VALUE_HIGH]
                m_row = (np.percentile(wet_m, pq100)
                         if len(wet_m) > 0 else np.full(n_q, np.nan, dtype=np.float32))
            else:
                m_row = np.full(n_q, np.nan, dtype=np.float32)
            return h_row.astype(np.float32), m_row.astype(np.float32)

        results       = Parallel(n_jobs=N_JOBS)(
            delayed(_one_sample)(s) for s in range(N_SAMPLES))
        hourly_q_boot = np.array([r[0] for r in results], dtype=np.float32)
        max_q_boot    = np.array([r[1] for r in results], dtype=np.float32)
        print(f"  Total time: {elapsed(t0_loop)}")
        return hourly_q_boot, max_q_boot

    # ------------------------------------------------------------------
    # STEP 5 & 6 — Bootstrap
    # ------------------------------------------------------------------
    # Pre-filter to wet days only — dry days make no rng calls in the bootstrap
    # inner loop so removing them gives identical results with fewer iterations.
    pres_daily_input     = pres_day_ctr[pres_day_ctr >= WET_VALUE_LOW]
    fut_daily_input      = fut_day_ctr[ fut_day_ctr  >= WET_VALUE_LOW]
    pres_daily_buf_input = daily_pres[daily_pres >= WET_VALUE_LOW].reshape(-1)
    fut_daily_buf_input  = daily_fut[ daily_fut  >= WET_VALUE_LOW].reshape(-1)

    subsection("Bootstrap sampling")

    if RUN_METHOD_A:
        syn_pres_h_boot,     syn_pres_dm_boot     = generate_synthetic_quantiles(
            pres_daily_input,     "PRESENT (center pixel)", seed_base=42)
        syn_fut_h_boot,      syn_fut_dm_boot      = generate_synthetic_quantiles(
            fut_daily_input,      "FUTURE  (center pixel)", seed_base=999)
        syn_pres_h_boot_buf, syn_pres_dm_boot_buf = generate_synthetic_quantiles(
            pres_daily_buf_input, "PRESENT (buffer)",       seed_base=142)
        syn_fut_h_boot_buf,  syn_fut_dm_boot_buf  = generate_synthetic_quantiles(
            fut_daily_buf_input,  "FUTURE  (buffer)",       seed_base=1099)
    else:
        print("  [Method A skipped — RUN_METHOD_A = False]")

    syn_pres_mx_boot     = generate_synthetic_max_quantiles(
        pres_daily_input,     "PRESENT (center pixel)", seed_base=242)
    syn_fut_mx_boot      = generate_synthetic_max_quantiles(
        fut_daily_input,      "FUTURE  (center pixel)", seed_base=1999)
    syn_pres_mx_boot_buf = generate_synthetic_max_quantiles(
        pres_daily_buf_input, "PRESENT (buffer)",       seed_base=342)
    syn_fut_mx_boot_buf  = generate_synthetic_max_quantiles(
        fut_daily_buf_input,  "FUTURE  (buffer)",       seed_base=2099)

    syn_pres_c_h_boot,     syn_pres_c_dm_boot     = generate_analog_quantiles(
        pres_daily_input,     "PRESENT (center pixel)", seed_base=542)
    syn_fut_c_h_boot,      syn_fut_c_dm_boot      = generate_analog_quantiles(
        fut_daily_input,      "FUTURE  (center pixel)", seed_base=2999)
    syn_pres_c_h_boot_buf, syn_pres_c_dm_boot_buf = generate_analog_quantiles(
        pres_daily_buf_input, "PRESENT (buffer)",       seed_base=642)
    syn_fut_c_h_boot_buf,  syn_fut_c_dm_boot_buf  = generate_analog_quantiles(
        fut_daily_buf_input,  "FUTURE  (buffer)",       seed_base=3099)

    # Bootstrap CIs
    pres_mx_ci      = ci_across_samples(syn_pres_mx_boot)
    fut_mx_ci       = ci_across_samples(syn_fut_mx_boot)
    pres_mx_ci_buf  = ci_across_samples(syn_pres_mx_boot_buf)
    fut_mx_ci_buf   = ci_across_samples(syn_fut_mx_boot_buf)
    pres_c_h_ci     = ci_across_samples(syn_pres_c_h_boot)
    fut_c_h_ci      = ci_across_samples(syn_fut_c_h_boot)
    pres_c_h_ci_buf = ci_across_samples(syn_pres_c_h_boot_buf)
    fut_c_h_ci_buf  = ci_across_samples(syn_fut_c_h_boot_buf)

    # ------------------------------------------------------------------
    # STEP 7 — Plots
    # ------------------------------------------------------------------
    COL_C_FUT = '#9B59B6'

    # Validation figure (2×2)
    fig1, axes1 = plt.subplots(2, 2, figsize=(14, 11))

    _plot_panel(axes1[0, 0],
                obs_pres_h, obs_fut_h,
                pres_c_h_ci[I_LO], pres_c_h_ci[I_MED], pres_c_h_ci[I_HI],
                'Method C (analog) — synthetic present',
                '1-hour precipitation (mm)',
                f'Hourly intensity — center pixel\n{location}, {buf_str}')
    _add_ci(axes1[0, 0], fut_c_h_ci, COL_C_FUT,
            'Method C (analog) — synthetic future')
    _dedup_legend(axes1[0, 0], fontsize=8, loc='upper left', frameon=True)

    _plot_panel(axes1[0, 1],
                obs_pres_dm, obs_fut_dm,
                pres_mx_ci[I_LO], pres_mx_ci[I_MED], pres_mx_ci[I_HI],
                'Method B — synthetic present',
                'Daily-max 1-hour precipitation (mm)',
                f'Daily-max hourly — center pixel\n{location}, {buf_str}')
    _add_ci(axes1[0, 1], fut_mx_ci, COL_C_FUT, 'Method B — synthetic future')
    _dedup_legend(axes1[0, 1], fontsize=8, loc='upper left', frameon=True)

    _plot_panel(axes1[1, 0],
                obs_pres_h_buf, obs_fut_h_buf,
                pres_c_h_ci_buf[I_LO], pres_c_h_ci_buf[I_MED],
                pres_c_h_ci_buf[I_HI],
                'Method C (analog) — synthetic present',
                '1-hour precipitation (mm)',
                f'Hourly intensity — buffer ({ny_reg}×{nx_reg})\n'
                f'{location}, {buf_str}')
    _add_ci(axes1[1, 0], fut_c_h_ci_buf, COL_C_FUT,
            'Method C (analog) — synthetic future')
    if buffer > 0:
        axes1[1, 0].plot(q_axis, obs_pres_h, color='#2E86AB', linewidth=1.0,
                         linestyle=':', marker='^', markersize=4, alpha=0.7,
                         label='Present observed (center pixel)')
        axes1[1, 0].plot(q_axis, obs_fut_h,  color='#E50C0C', linewidth=1.0,
                         linestyle=':', marker='v', markersize=4, alpha=0.7,
                         label='Future observed (center pixel)')
    _dedup_legend(axes1[1, 0], fontsize=8, loc='upper left', frameon=True)

    _plot_panel(axes1[1, 1],
                obs_pres_dm_buf, obs_fut_dm_buf,
                pres_mx_ci_buf[I_LO], pres_mx_ci_buf[I_MED],
                pres_mx_ci_buf[I_HI],
                'Method B — synthetic present',
                'Daily-max 1-hour precipitation (mm)',
                f'Daily-max hourly — buffer ({ny_reg}×{nx_reg})\n'
                f'{location}, {buf_str}')
    _add_ci(axes1[1, 1], fut_mx_ci_buf, COL_C_FUT,
            'Method B — synthetic future')
    if buffer > 0:
        axes1[1, 1].plot(q_axis, obs_pres_dm, color='#2E86AB', linewidth=1.0,
                         linestyle=':', marker='^', markersize=4, alpha=0.7,
                         label='Present observed (center pixel)')
        axes1[1, 1].plot(q_axis, obs_fut_dm,  color='#E50C0C', linewidth=1.0,
                         linestyle=':', marker='v', markersize=4, alpha=0.7,
                         label='Future observed (center pixel)')
    _dedup_legend(axes1[1, 1], fontsize=8, loc='upper left', frameon=True)

    fig1.suptitle(
        f'Validation: observed vs synthetic present & future — {location}\n'
        f'Top: center pixel   |   Bottom: buffer pooled '
        f'({ny_reg}×{nx_reg} pixels)',
        fontsize=13, fontweight='bold')
    fig1.tight_layout()
    out_val = f'{PATH_OUT}/validation_{run_label}.png'
    fig1.savefig(out_val, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig1)
    print(f"\n  Saved: {out_val}")

    # Attribution figure (1×2)
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 6))
    for ax, obs_pres, obs_fut, obs_pres_ctr, obs_fut_ctr, ci, ylabel in [
        (axes2[0], obs_pres_h_buf,  obs_fut_h_buf,
         obs_pres_h,  obs_fut_h,  fut_c_h_ci_buf,
         '1-hour precipitation (mm)'),
        (axes2[1], obs_pres_dm_buf, obs_fut_dm_buf,
         obs_pres_dm, obs_fut_dm, fut_mx_ci_buf,
         'Daily-max 1-hour precipitation (mm)'),
    ]:
        _plot_panel(ax, obs_pres, obs_fut,
                    ci[I_LO], ci[I_MED], ci[I_HI],
                    'Synthetic future (median)', ylabel,
                    f'{ylabel.split("(")[0].strip()} — attribution\n'
                    f'{location}, {buf_str}')
        if buffer > 0:
            ax.plot(q_axis, obs_pres_ctr, color='#2E86AB', linewidth=1.0,
                    linestyle=':', marker='^', markersize=4, alpha=0.7,
                    label='Present observed (center pixel)')
            ax.plot(q_axis, obs_fut_ctr,  color='#E50C0C', linewidth=1.0,
                    linestyle=':', marker='v', markersize=4, alpha=0.7,
                    label='Future observed (center pixel)')
            _dedup_legend(ax, fontsize=9, loc='upper left', frameon=True,
                          fancybox=True, shadow=True)
        idx_99 = np.searchsorted(q_axis, 0.99)
        if idx_99 < len(q_axis) and np.isclose(q_axis[idx_99], 0.99,
                                                 atol=0.001):
            total  = float(obs_fut[idx_99]) - float(obs_pres[idx_99])
            daily  = float(ci[I_MED, idx_99]) - float(obs_pres[idx_99])
            struct = float(obs_fut[idx_99]) - float(ci[I_MED, idx_99])
            pct_d  = 100.0 * daily  / abs(total) if total != 0 else 0.0
            pct_s  = 100.0 * struct / abs(total) if total != 0 else 0.0
            ax.text(0.03, 0.04,
                    f'P99 total Δ : {total:+.2f} mm\n'
                    f'Daily effect : {daily:+.2f} mm  ({pct_d:.0f}%)\n'
                    f'Structural   : {struct:+.2f} mm  ({pct_s:.0f}%)',
                    transform=ax.transAxes, fontsize=8.5,
                    verticalalignment='bottom',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.75))
    fig2.suptitle(
        f'Attribution: future observed vs synthetic future — {location}',
        fontsize=13, fontweight='bold')
    fig2.tight_layout()
    out_attr = f'{PATH_OUT}/attribution_{run_label}.png'
    fig2.savefig(out_attr, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig2)
    print(f"  Saved: {out_attr}")

    # ------------------------------------------------------------------
    # STEP 8 — Attribution summary table
    # ------------------------------------------------------------------
    subsection("Attribution summary (buffer pooled — Method C hourly)")
    hdr = (f"  {'Quantile':>10}  {'Obs_Pres':>10}  {'Obs_Fut':>10}  "
           f"{'Syn_Fut':>10}  {'Total Δ':>10}  {'Daily Δ':>10}  "
           f"{'Struct Δ':>10}  {'Daily%':>8}  {'Struct%':>8}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for i, q in enumerate(PLOT_QUANTILES):
        total, daily, struct, pct_d, pct_s = _attr_row(
            float(obs_pres_h_buf[i]),
            float(obs_fut_h_buf[i]),
            float(fut_c_h_ci_buf[I_MED, i]))
        print(f"  {q:>10.4f}  {obs_pres_h_buf[i]:>10.3f}  "
              f"{obs_fut_h_buf[i]:>10.3f}  {fut_c_h_ci_buf[I_MED,i]:>10.3f}  "
              f"{total:>+10.3f}  {daily:>+10.3f}  {struct:>+10.3f}  "
              f"{pct_d:>+8.1f}%  {pct_s:>+8.1f}%")

    # ------------------------------------------------------------------
    # SAVE BOOTSTRAP DATA
    # ------------------------------------------------------------------
    npz_path = os.path.join(PATH_OUT, 'testing_data', f'{run_label}.npz')
    save_dict = dict(
        # metadata
        plot_quantiles      = PLOT_QUANTILES,
        bootstrap_quantiles = BOOTSTRAP_QUANTILES,
        buffer              = np.array(buffer),
        n_samples           = np.array(N_SAMPLES),
        wet_value_high      = np.array(WET_VALUE_HIGH),
        wet_value_low       = np.array(WET_VALUE_LOW),
        n_days_pres         = np.array(n_days_pres),
        n_days_fut          = np.array(n_days_fut),
        ny_reg              = np.array(ny_reg),
        nx_reg              = np.array(nx_reg),
        bins_low            = BINS_LOW,
        n_days_per_bin      = n_days_all_by_bin,
        n_days_wet_hrs_per_bin = n_days_wet_hrs_by_bin,
        # observed percentiles — center pixel
        obs_pres_h          = obs_pres_h,
        obs_fut_h           = obs_fut_h,
        obs_pres_dm         = obs_pres_dm,
        obs_fut_dm          = obs_fut_dm,
        # observed percentiles — buffer pooled
        obs_pres_h_buf      = obs_pres_h_buf,
        obs_fut_h_buf       = obs_fut_h_buf,
        obs_pres_dm_buf     = obs_pres_dm_buf,
        obs_fut_dm_buf      = obs_fut_dm_buf,
        # Method B bootstrap arrays (N_SAMPLES, n_q)
        syn_pres_mx_boot        = syn_pres_mx_boot,
        syn_fut_mx_boot         = syn_fut_mx_boot,
        syn_pres_mx_boot_buf    = syn_pres_mx_boot_buf,
        syn_fut_mx_boot_buf     = syn_fut_mx_boot_buf,
        # Method C hourly bootstrap arrays (N_SAMPLES, n_q)
        syn_pres_c_h_boot       = syn_pres_c_h_boot,
        syn_fut_c_h_boot        = syn_fut_c_h_boot,
        syn_pres_c_h_boot_buf   = syn_pres_c_h_boot_buf,
        syn_fut_c_h_boot_buf    = syn_fut_c_h_boot_buf,
        # Method C daily-max bootstrap arrays (N_SAMPLES, n_q)
        syn_pres_c_dm_boot      = syn_pres_c_dm_boot,
        syn_fut_c_dm_boot       = syn_fut_c_dm_boot,
        syn_pres_c_dm_boot_buf  = syn_pres_c_dm_boot_buf,
        syn_fut_c_dm_boot_buf   = syn_fut_c_dm_boot_buf,
    )
    if RUN_METHOD_A:
        save_dict.update(dict(
            syn_pres_h_boot         = syn_pres_h_boot,
            syn_fut_h_boot          = syn_fut_h_boot,
            syn_pres_h_boot_buf     = syn_pres_h_boot_buf,
            syn_fut_h_boot_buf      = syn_fut_h_boot_buf,
            syn_pres_dm_boot        = syn_pres_dm_boot,
            syn_fut_dm_boot         = syn_fut_dm_boot,
            syn_pres_dm_boot_buf    = syn_pres_dm_boot_buf,
            syn_fut_dm_boot_buf     = syn_fut_dm_boot_buf,
        ))
    np.savez_compressed(npz_path, **save_dict)
    print(f"\n  Saved bootstrap data : {npz_path}")

    # Return key results for the final cross-combo summary
    return dict(
        location     = location,
        buffer       = buffer,
        obs_pres_h   = obs_pres_h,
        obs_fut_h    = obs_fut_h,
        obs_pres_h_buf = obs_pres_h_buf,
        obs_fut_h_buf  = obs_fut_h_buf,
        fut_c_h_ci     = fut_c_h_ci,
        fut_c_h_ci_buf = fut_c_h_ci_buf,
        pres_c_h_ci_buf = pres_c_h_ci_buf,
    )


# =============================================================================
# MAIN LOOP
# =============================================================================

t_total  = time.time()
combos   = [(loc, buf) for loc in LOCATIONS for buf in BUFFERS]
n_combos = len(combos)

print(f"\nRunning {n_combos} combinations:")
for i, (loc, buf) in enumerate(combos):
    print(f"  [{i+1}/{n_combos}]  {loc}  buffer={buf}")

all_results = []
for i, (loc, buf) in enumerate(combos):
    t_combo = time.time()
    print(f"\n\n[{i+1}/{n_combos}]  Starting {loc}  buffer={buf}")
    result = run_single(loc, buf)
    all_results.append(result)
    print(f"\n  Combo wall time: {elapsed(t_combo)}")

# =============================================================================
# FINAL CROSS-COMBO SUMMARY
# =============================================================================

section("FINAL SUMMARY — all (location, buffer) combinations")
print(f"  Method C hourly (buffer-pooled)   —   P99 attribution")
print()

hdr = (f"  {'Location':>12}  {'Buffer':>6}  "
       f"{'Obs_Pres':>10}  {'Obs_Fut':>10}  {'Syn_Fut_med':>12}  "
       f"{'Total%':>8}  {'Daily%':>8}  {'Struct%':>8}  "
       f"{'Validation':>12}")
print(hdr)
print("  " + "-" * (len(hdr) - 2))

i99 = np.searchsorted(PLOT_QUANTILES, 0.99)

for r in all_results:
    op  = float(r['obs_pres_h_buf'][i99])
    of  = float(r['obs_fut_h_buf'][i99])
    sf  = float(r['fut_c_h_ci_buf'][I_MED, i99])
    sp  = float(r['pres_c_h_ci_buf'][I_MED, i99])   # synthetic present median

    total_pct  = 100 * (of - op) / op if op > 0 else float('nan')
    daily_pct  = 100 * (sf - op) / op if op > 0 else float('nan')
    struct_pct = 100 * (of - sf) / op if op > 0 else float('nan')
    val_bias   = 100 * (sp - op) / op if op > 0 else float('nan')   # validation bias

    print(f"  {r['location']:>12}  {r['buffer']:>6}  "
          f"{op:>10.3f}  {of:>10.3f}  {sf:>12.3f}  "
          f"{total_pct:>+8.1f}%  {daily_pct:>+8.1f}%  {struct_pct:>+8.1f}%  "
          f"{val_bias:>+11.1f}%")

print()
print(f"  Validation bias = (SynPres_median - Obs_Pres) / Obs_Pres × 100%")
print(f"  Should be close to 0% for Method C to be reliable.")
print(f"\n  Total wall time: {elapsed(t_total)}")
