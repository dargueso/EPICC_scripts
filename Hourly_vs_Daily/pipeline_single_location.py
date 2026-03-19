#!/usr/bin/env python
"""
End-to-end single-location pipeline for synthetic hourly rainfall validation and attribution.

Steps
-----
1.  Load present + future hourly zarr, find nearest grid pixel to target location
2.  Resample hourly → daily by 24-hour blocks
3.  Observed percentiles  (center pixel, wet-only)
4.  Conditional probabilities  P(hourly | daily)  from present climate
5.  Bootstrap synthetic present  (present condprob × present daily totals) → validation
6.  Bootstrap synthetic future   (present condprob × future  daily totals) → attribution
7.  Plots: validation + attribution
8.  Attribution summary table

Methods match the full-domain pipeline:
  create_percentiles_sequential_io_zarr.py   → wet-only percentiles
  create_conditional_probabilities_zarr.py   → 3 conditional histograms
  create_synthetic_future_zarr.py            → 3-distribution sampling, shift-then-scale
"""

import time
import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import ScalarFormatter

mpl.rcParams["font.size"] = 13


# =============================================================================
# CONFIGURATION  —  edit only this section
# =============================================================================

PATH_IN  = '/home/dargueso/postprocessed/EPICC/'
PATH_OUT = '/home/dargueso/Analyses/EPICC/Hourly_vs_Daily/'

WRUN_PRESENT = 'EPICC_2km_ERA5'
WRUN_FUTURE  = 'EPICC_2km_ERA5_CMIP6anom'

# Named locations (same as extract_test_area_zarr.py)
loc_lats = {'Mallorca': 39.639, 'Barcelona': 41.385, 'Valencia':  39.469,
            'Rosiglione': 44.55, 'Catania':   37.51}
loc_lons = {'Mallorca':  2.647, 'Barcelona':  2.173, 'Valencia':  -0.376,
            'Rosiglione': 8.64, 'Catania':   15.08}

LOCATION = 'Catania'   # key from loc_lats / loc_lons

# Spatial buffer around the nearest pixel (in grid cells).
#   BUFFER = 0  →  single nearest pixel (condprob from that pixel only)
#   BUFFER = n  →  (2n+1)×(2n+1) region; all pixels pooled for condprob;
#                   observed percentiles still use center pixel only
BUFFER = 10 

# Wet thresholds — must match the full-domain pipeline
WET_VALUE_HIGH = 0.1   # mm/h  — hourly wet threshold
WET_VALUE_LOW  = 1.0   # mm/d  — daily  wet threshold

# Quantile fractions for comparison plots and summary table
PLOT_QUANTILES = np.array([0.90, 0.95, 0.98, 0.99, 0.995, 0.999])

# Bins — must match full-domain create_conditional_probabilities_zarr.py
BINS_HIGH = np.append(np.arange(0, 100, 1), np.inf)   # 100 hourly intensity bins
BINS_LOW  = np.append(np.arange(0, 100, 5), np.inf)   #  20 daily  total bins

# Bootstrap
N_SAMPLES           = 1000   # synthetic realisations per climate state
BOOTSTRAP_QUANTILES = np.array([0.025, 0.5, 0.975])   # CI lower / median / upper

# Exponential scale used when sampling from the open-ended (inf) bin
EXP_SCALE = 5.0

# Set to True to also run Method A (parametric hourly sampler).
# Method A is slow for large buffers and known to overestimate upper quantiles;
# it is kept for diagnostic comparison only.
RUN_METHOD_A = False

# Hours per low-frequency period (do not change for hourly/daily pairing)
N_INTERVAL = 24

os.makedirs(PATH_OUT, exist_ok=True)


# =============================================================================
# HELPER UTILITIES
# =============================================================================

def section(title):
    """Print a clear section banner with timestamp."""
    ts = time.strftime('%H:%M:%S')
    print(f"\n{'='*70}")
    print(f"  [{ts}]  {title}")
    print(f"{'='*70}")


def subsection(title):
    print(f"\n  --- {title} ---")


def elapsed(t0):
    return f"{time.time() - t0:.1f}s"


def _open_zarr(path):
    """Open a zarr store, falling back to non-consolidated metadata if needed."""
    try:
        ds = xr.open_zarr(path, consolidated=True)
        print(f"    Opened (consolidated metadata): {path}")
        return ds
    except KeyError:
        ds = xr.open_zarr(path, consolidated=False)
        print(f"    Opened (no consolidated metadata): {path}")
        return ds


def print_percentile_table(label, quantiles, values):
    """Print a one-line-per-quantile table of observed values."""
    print(f"\n    {label}")
    print(f"    {'Quantile':>10}  {'Value (mm)':>12}")
    for q, v in zip(quantiles, values):
        print(f"    {q:>10.4f}  {v:>12.4f}")


def print_bootstrap_table(label, quantiles, ci_lo, ci_med, ci_hi):
    """Print CI lower / median / upper for each quantile."""
    print(f"\n    {label}")
    print(f"    {'Quantile':>10}  {'CI-lo':>10}  {'Median':>10}  {'CI-hi':>10}")
    for i, q in enumerate(quantiles):
        print(f"    {q:>10.4f}  {ci_lo[i]:>10.4f}  {ci_med[i]:>10.4f}  {ci_hi[i]:>10.4f}")


# =============================================================================
# STEP 1 — Load zarr, locate nearest pixel, extract buffer region
# =============================================================================

section("STEP 1 — Load data and locate target pixel")

TARGET_LAT = loc_lats[LOCATION]
TARGET_LON = loc_lons[LOCATION]
print(f"  Location : {LOCATION}")
print(f"  Target   : lat={TARGET_LAT}, lon={TARGET_LON}")
print(f"  Buffer   : {BUFFER} grid cells "
      f"({'single pixel' if BUFFER == 0 else f'{2*BUFFER+1}x{2*BUFFER+1} region'})")

zarr_pres = f'{PATH_IN}/{WRUN_PRESENT}/UIB_01H_RAIN.zarr'
zarr_fut  = f'{PATH_IN}/{WRUN_FUTURE}/UIB_01H_RAIN.zarr'

t0 = time.time()
print("\n  Opening present zarr ...")
ds_pres = _open_zarr(zarr_pres)
print(f"    Domain shape: {ds_pres.RAIN.shape}  (time, y, x)")

print("  Opening future zarr ...")
ds_fut = _open_zarr(zarr_fut)
print(f"    Domain shape: {ds_fut.RAIN.shape}  (time, y, x)")
print(f"  Zarr open time: {elapsed(t0)}")

# Find nearest grid point using lat/lon from the present run
subsection("Finding nearest grid pixel")
lat2d = ds_pres.lat.isel(time=0).values   # (ny, nx)
lon2d = ds_pres.lon.isel(time=0).values
ny_full, nx_full = lat2d.shape

dist = np.sqrt((lat2d - TARGET_LAT)**2 + (lon2d - TARGET_LON)**2)
cy, cx = np.unravel_index(np.argmin(dist), dist.shape)

print(f"  Nearest pixel  : y={cy}, x={cx}")
print(f"  Actual lat/lon : {lat2d[cy,cx]:.4f} / {lon2d[cy,cx]:.4f}")
print(f"  Distance       : {dist[cy,cx]:.5f} degrees")

# Buffer region bounds (clamped to domain)
y0, y1 = max(0, cy - BUFFER), min(ny_full, cy + BUFFER + 1)
x0, x1 = max(0, cx - BUFFER), min(nx_full, cx + BUFFER + 1)
cy_loc, cx_loc = cy - y0, cx - x0   # center index within extracted region
ny_reg, nx_reg = y1 - y0, x1 - x0

print(f"  Extracted region : y=[{y0}:{y1}), x=[{x0}:{x1})  "
      f"→  {ny_reg}×{nx_reg} pixels")
print(f"  Center index in region : [{cy_loc}, {cx_loc}]")

subsection("Loading hourly data for the extracted region")
t0 = time.time()
print("  Loading present ...")
rain_pres = ds_pres.RAIN.isel(y=slice(y0, y1), x=slice(x0, x1)).values.astype(np.float32)
print("  Loading future ...")
rain_fut  = ds_fut.RAIN.isel(y=slice(y0, y1), x=slice(x0, x1)).values.astype(np.float32)
ds_pres.close()
ds_fut.close()
print(f"  Load time: {elapsed(t0)}")
print(f"  Present array : {rain_pres.shape}  (time, {ny_reg}, {nx_reg})")
print(f"  Future  array : {rain_fut.shape}  (time, {ny_reg}, {nx_reg})")
print(f"  Memory (present + future): "
      f"{(rain_pres.nbytes + rain_fut.nbytes) / 1e6:.1f} MB")


# =============================================================================
# STEP 2 — Resample hourly → daily by 24-hour blocks
# =============================================================================

section("STEP 2 — Resample hourly → daily (24-hour blocks)")

nt_pres = rain_pres.shape[0]
nt_fut  = rain_fut.shape[0]

n_days_pres = nt_pres // N_INTERVAL
n_days_fut  = nt_fut  // N_INTERVAL

# Dropped hours (< N_INTERVAL remainder at the end)
dropped_pres = nt_pres - n_days_pres * N_INTERVAL
dropped_fut  = nt_fut  - n_days_fut  * N_INTERVAL
if dropped_pres:
    print(f"  WARNING: {dropped_pres} trailing present hours dropped (incomplete day)")
if dropped_fut:
    print(f"  WARNING: {dropped_fut} trailing future hours dropped (incomplete day)")

# blk_* shape: (n_days, N_INTERVAL, ny_reg, nx_reg)
blk_pres = rain_pres[:n_days_pres * N_INTERVAL].reshape(
    n_days_pres, N_INTERVAL, ny_reg, nx_reg)
blk_fut  = rain_fut[:n_days_fut * N_INTERVAL].reshape(
    n_days_fut,  N_INTERVAL, ny_reg, nx_reg)

daily_pres = blk_pres.sum(axis=1)   # (n_days, ny, nx)
daily_fut  = blk_fut.sum(axis=1)

# Center-pixel 1-D time series
pres_1h_ctr  = rain_pres[:n_days_pres * N_INTERVAL, cy_loc, cx_loc]
fut_1h_ctr   = rain_fut[:n_days_fut  * N_INTERVAL,  cy_loc, cx_loc]
pres_day_ctr = daily_pres[:, cy_loc, cx_loc]
fut_day_ctr  = daily_fut[:,  cy_loc, cx_loc]

print(f"  Present : {n_days_pres} days   ({nt_pres} hourly steps)")
print(f"  Future  : {n_days_fut} days   ({nt_fut} hourly steps)")
print(f"\n  Center-pixel daily totals (present):")
print(f"    All days  — mean={pres_day_ctr.mean():.3f} mm,  "
      f"max={pres_day_ctr.max():.2f} mm,  "
      f"wet days (>={WET_VALUE_LOW}mm): {(pres_day_ctr >= WET_VALUE_LOW).sum()}")
print(f"  Center-pixel daily totals (future):")
print(f"    All days  — mean={fut_day_ctr.mean():.3f} mm,  "
      f"max={fut_day_ctr.max():.2f} mm,  "
      f"wet days (>={WET_VALUE_LOW}mm): {(fut_day_ctr >= WET_VALUE_LOW).sum()}")


# =============================================================================
# STEP 3 — Observed percentiles  (center pixel, wet-only)
# =============================================================================

section("STEP 3 — Observed percentiles (center pixel, wet-only)")
print(f"  Wet threshold hourly : > {WET_VALUE_HIGH} mm")
print(f"  Wet threshold daily  : >= {WET_VALUE_LOW} mm")

# Hourly wet values — restricted to wet days (daily >= WET_VALUE_LOW) so the
# population matches what the synthetic generates (condprob is only built for
# wet days; dry-day drizzle hours would artificially lower the observed quantiles).
pres_wet_day_h = np.repeat(pres_day_ctr >= WET_VALUE_LOW, N_INTERVAL)
fut_wet_day_h  = np.repeat(fut_day_ctr  >= WET_VALUE_LOW, N_INTERVAL)
pres_wet_1h = pres_1h_ctr[(pres_1h_ctr > WET_VALUE_HIGH) & pres_wet_day_h]
fut_wet_1h  = fut_1h_ctr[ (fut_1h_ctr  > WET_VALUE_HIGH) & fut_wet_day_h]

obs_pres_h = np.percentile(pres_wet_1h, PLOT_QUANTILES * 100)
obs_fut_h  = np.percentile(fut_wet_1h,  PLOT_QUANTILES * 100)

# Daily-max hourly — only on days that have at least one hour > WET_VALUE_HIGH.
# This matches the synthetic, which now also only generates daily-max values for
# such days (drizzle days are skipped via p_has_wet_hours in the sampler).
pres_dmax_ctr = blk_pres[:, :, cy_loc, cx_loc].max(axis=1)   # (n_days,)
fut_dmax_ctr  = blk_fut[:,  :, cy_loc, cx_loc].max(axis=1)

pres_n_wet_ctr = (blk_pres[:, :, cy_loc, cx_loc] > WET_VALUE_HIGH).sum(axis=1)
fut_n_wet_ctr  = (blk_fut[:,  :, cy_loc, cx_loc] > WET_VALUE_HIGH).sum(axis=1)

# Wet day = daily >= WET_VALUE_LOW AND at least one hour > WET_VALUE_HIGH
pres_dmax_mask = (pres_day_ctr >= WET_VALUE_LOW) & (pres_n_wet_ctr > 0)
fut_dmax_mask  = (fut_day_ctr  >= WET_VALUE_LOW) & (fut_n_wet_ctr  > 0)

pres_dmax_wet = pres_dmax_ctr[pres_dmax_mask]
fut_dmax_wet  = fut_dmax_ctr[fut_dmax_mask]

obs_pres_dm = np.percentile(pres_dmax_wet, PLOT_QUANTILES * 100)
obs_fut_dm  = np.percentile(fut_dmax_wet,  PLOT_QUANTILES * 100)

subsection("Hourly intensity")
print(f"    Present wet hours : {len(pres_wet_1h):,}  "
      f"(mean={pres_wet_1h.mean():.3f} mm,  max={pres_wet_1h.max():.2f} mm)")
print(f"    Future  wet hours : {len(fut_wet_1h):,}  "
      f"(mean={fut_wet_1h.mean():.3f} mm,  max={fut_wet_1h.max():.2f} mm)")
print(f"\n    {'Quantile':>10}  {'Present':>10}  {'Future':>10}  {'Change%':>10}")
for i, q in enumerate(PLOT_QUANTILES):
    chg = 100.0 * (obs_fut_h[i] - obs_pres_h[i]) / obs_pres_h[i] if obs_pres_h[i] > 0 else float('nan')
    print(f"    {q:>10.4f}  {obs_pres_h[i]:>10.4f}  {obs_fut_h[i]:>10.4f}  {chg:>+10.1f}%")

# Compute center-pixel E[n_wet], E[hourly intensity], E[daily-max] per daily bin.
# Stored here; printed as comparison tables after Step 4 where condprob is available.
ctr_nwet_per_bin   = np.full(len(BINS_LOW) - 1, np.nan)
ctr_intens_per_bin = np.full(len(BINS_LOW) - 1, np.nan)   # mean wet-hour intensity
ctr_max_per_bin    = np.full(len(BINS_LOW) - 1, np.nan)   # mean daily-max
ctr_ndays_per_bin  = np.zeros(len(BINS_LOW) - 1, dtype=np.int32)
n_wet_vals_ref     = np.arange(1, N_INTERVAL + 1, dtype=float)
for j in range(len(BINS_LOW) - 1):
    lo, hi = BINS_LOW[j], BINS_LOW[j + 1]
    in_bin = (pres_day_ctr >= max(lo, WET_VALUE_LOW)) & (pres_day_ctr < hi)
    ctr_ndays_per_bin[j] = int(in_bin.sum())
    if in_bin.sum() > 0:
        blk_bin = blk_pres[in_bin, :, cy_loc, cx_loc]   # (ndays_in_bin, 24)
        n_wet_in_bin = (blk_bin > WET_VALUE_HIGH).sum(axis=1)
        ctr_nwet_per_bin[j] = float(n_wet_in_bin.mean())
        # hourly intensity and daily-max: only from days with n_wet >= 1
        has_wet = n_wet_in_bin > 0
        if has_wet.sum() > 0:
            wet_h = blk_bin[has_wet, :].flatten()
            wet_h = wet_h[wet_h > WET_VALUE_HIGH]
            if len(wet_h) > 0:
                ctr_intens_per_bin[j] = float(wet_h.mean())
            day_max_bin = blk_bin[has_wet, :].max(axis=1)
            ctr_max_per_bin[j] = float(day_max_bin.mean())

subsection("Daily-max hourly")
n_pres_wet_day_total = int((pres_day_ctr >= WET_VALUE_LOW).sum())
n_fut_wet_day_total  = int((fut_day_ctr  >= WET_VALUE_LOW).sum())
print(f"    Present — all wet days (daily>={WET_VALUE_LOW}mm)  : {n_pres_wet_day_total:,}")
print(f"    Present — wet days with n_wet>=1                : {len(pres_dmax_wet):,}  "
      f"({100*len(pres_dmax_wet)/max(n_pres_wet_day_total,1):.1f}%)")
print(f"    Future  — all wet days (daily>={WET_VALUE_LOW}mm)  : {n_fut_wet_day_total:,}")
print(f"    Future  — wet days with n_wet>=1                : {len(fut_dmax_wet):,}  "
      f"({100*len(fut_dmax_wet)/max(n_fut_wet_day_total,1):.1f}%)")
print(f"\n    {'Quantile':>10}  {'Present':>10}  {'Future':>10}  {'Change%':>10}")
for i, q in enumerate(PLOT_QUANTILES):
    chg = 100.0 * (obs_fut_dm[i] - obs_pres_dm[i]) / obs_pres_dm[i] if obs_pres_dm[i] > 0 else float('nan')
    print(f"    {q:>10.4f}  {obs_pres_dm[i]:>10.4f}  {obs_fut_dm[i]:>10.4f}  {chg:>+10.1f}%")

# ---- Buffer-pooled observed percentiles (all pixels in buffer region) ----
# Used for the "like-for-like" comparison with the buffer-pooled synthetic.
subsection("Buffer-pooled observed percentiles")
print(f"  Pooling all {ny_reg}x{nx_reg} = {ny_reg*nx_reg} pixels in the buffer region.")

# Hourly wet values: all pixels, but only from wet days (daily >= WET_VALUE_LOW).
# daily_pres shape: (n_days, ny, nx); expand to (n_days, 24, ny, nx) for masking.
pres_wet_day_mask_buf = (daily_pres >= WET_VALUE_LOW)[:, np.newaxis, :, :]  # (n_days,1,ny,nx)
fut_wet_day_mask_buf  = (daily_fut  >= WET_VALUE_LOW)[:, np.newaxis, :, :]
pres_blk_masked = blk_pres * pres_wet_day_mask_buf   # zero out dry days
fut_blk_masked  = blk_fut  * fut_wet_day_mask_buf
pres_wet_1h_buf = pres_blk_masked.reshape(-1)
pres_wet_1h_buf = pres_wet_1h_buf[pres_wet_1h_buf > WET_VALUE_HIGH]
fut_wet_1h_buf  = fut_blk_masked.reshape(-1)
fut_wet_1h_buf  = fut_wet_1h_buf[fut_wet_1h_buf > WET_VALUE_HIGH]

obs_pres_h_buf = np.percentile(pres_wet_1h_buf, PLOT_QUANTILES * 100)
obs_fut_h_buf  = np.percentile(fut_wet_1h_buf,  PLOT_QUANTILES * 100)

# Daily-max: pool all pixels, filter wet days (daily >= WET_VALUE_LOW AND n_wet >= 1)
pres_dmax_all      = blk_pres.max(axis=1)                              # (n_days, ny, nx)
fut_dmax_all       = blk_fut.max(axis=1)
pres_n_wet_all     = (blk_pres > WET_VALUE_HIGH).sum(axis=1)           # (n_days, ny, nx)
fut_n_wet_all      = (blk_fut  > WET_VALUE_HIGH).sum(axis=1)
pres_dmax_mask_buf = (daily_pres >= WET_VALUE_LOW) & (pres_n_wet_all > 0)
fut_dmax_mask_buf  = (daily_fut  >= WET_VALUE_LOW) & (fut_n_wet_all  > 0)
pres_dmax_wet_buf  = pres_dmax_all[pres_dmax_mask_buf]
fut_dmax_wet_buf   = fut_dmax_all[fut_dmax_mask_buf]

obs_pres_dm_buf = np.percentile(pres_dmax_wet_buf, PLOT_QUANTILES * 100)
obs_fut_dm_buf  = np.percentile(fut_dmax_wet_buf,  PLOT_QUANTILES * 100)

print(f"  Present: {len(pres_wet_1h_buf):,} wet hours,  {len(pres_dmax_wet_buf):,} wet-day maxima")
print(f"  Future : {len(fut_wet_1h_buf):,} wet hours,  {len(fut_dmax_wet_buf):,} wet-day maxima")
print(f"\n    {'Quantile':>10}  {'Pres_buf':>10}  {'Pres_ctr':>10}  {'Fut_buf':>10}  {'Fut_ctr':>10}")
for i, q in enumerate(PLOT_QUANTILES):
    print(f"    {q:>10.4f}  {obs_pres_h_buf[i]:>10.4f}  {obs_pres_h[i]:>10.4f}"
          f"  {obs_fut_h_buf[i]:>10.4f}  {obs_fut_h[i]:>10.4f}")


# =============================================================================
# STEP 4 — Conditional probabilities  P(hourly | daily), present climate
# =============================================================================

section("STEP 4 — Conditional probabilities (present climate)")
print(f"  Source : present hourly, {ny_reg}×{nx_reg} pixels pooled")
print(f"  All three histograms use ONLY days where n_wet >= 1 AND daily >= WET_VALUE_LOW.")
print(f"  P(n_wet >= 1 | daily bin) is stored separately and used in the sampler to")
print(f"  stochastically skip days that have no wet hours (drizzle days).")
print(f"  Histograms:")
print(f"    (1) P(hourly intensity | daily total)  →  hist_intensity  [n_wet>=1 days]")
print(f"    (2) P(n_wet hours | daily total)        →  hist_n_wet      [n_wet>=1 days]")
print(f"    (3) P(daily-max hourly | daily total)   →  hist_max_intens [n_wet>=1 days]")

nbins_high = len(BINS_HIGH) - 1   # 100
nbins_low  = len(BINS_LOW)  - 1   #  20

# Raw counts, pooled over all pixels in the buffer region
hist_intensity  = np.zeros((nbins_high, nbins_low), dtype=np.float64)
hist_n_wet      = np.zeros((N_INTERVAL, nbins_low), dtype=np.float64)
hist_max_intens = np.zeros((nbins_high, nbins_low), dtype=np.float64)

# Two counters per daily bin:
#   n_days_all_by_bin     — all wet days (daily >= WET_VALUE_LOW)
#   n_days_wet_hrs_by_bin — subset with at least one hour > WET_VALUE_HIGH (n_wet >= 1)
# Ratio → P(n_wet >= 1 | daily bin), used in the sampler
n_days_all_by_bin     = np.zeros(nbins_low, dtype=np.int32)
n_days_wet_hrs_by_bin = np.zeros(nbins_low, dtype=np.int32)

n_wet_bins = np.arange(0.5, N_INTERVAL + 1.5, 1)   # edges [0.5, 1.5, ..., 24.5]

t0 = time.time()
for iy in range(ny_reg):
    for ix in range(nx_reg):
        rain_h_blk = blk_pres[:, :, iy, ix]    # (n_days, 24) hourly blocks
        rain_d     = daily_pres[:, iy, ix]      # (n_days,)   daily totals

        # Keep only wet days (daily total >= WET_VALUE_LOW)
        wet_mask = rain_d >= WET_VALUE_LOW
        if wet_mask.sum() == 0:
            continue

        rain_h_wet = rain_h_blk[wet_mask, :].copy()   # (n_wet_days, 24)
        rain_d_wet = rain_d[wet_mask]                  # (n_wet_days,)

        # Zero out hourly values below wet threshold (within wet days)
        rain_h_wet[rain_h_wet < WET_VALUE_HIGH] = 0.0

        # Identify which days have at least one hour above the hourly threshold
        n_wet         = (rain_h_wet > WET_VALUE_HIGH).sum(axis=1)   # (n_wet_days,)
        has_wet_hours = n_wet > 0   # boolean mask over wet days

        # All three histograms use only n_wet >= 1 days.
        # Days with n_wet=0 are drizzle days — their hourly max is 0, which would
        # corrupt the max_intensity histogram and dilute the intensity histogram.
        if has_wet_hours.sum() > 0:
            rain_h_sub = rain_h_wet[has_wet_hours, :]   # (n_wet_hrs_days, 24)
            rain_d_sub = rain_d_wet[has_wet_hours]       # (n_wet_hrs_days,)

            # --- HISTOGRAM 1: P(hourly intensity | daily total) ---
            h_flat     = rain_h_sub.flatten()
            d_rep      = np.repeat(rain_d_sub, N_INTERVAL)
            wet_mask_h = h_flat >= WET_VALUE_HIGH
            if wet_mask_h.sum() > 0:
                counts, _, _ = np.histogram2d(
                    h_flat[wet_mask_h], d_rep[wet_mask_h],
                    bins=[BINS_HIGH, BINS_LOW]
                )
                hist_intensity += counts

            # --- HISTOGRAM 2: P(n_wet | daily total) ---
            counts_nw, _, _ = np.histogram2d(
                n_wet[has_wet_hours], rain_d_sub,
                bins=[n_wet_bins, BINS_LOW]
            )
            hist_n_wet += counts_nw

            # --- HISTOGRAM 3: P(daily-max hourly | daily total) ---
            day_max = rain_h_sub.max(axis=1)
            counts_mx, _, _ = np.histogram2d(
                day_max, rain_d_sub, bins=[BINS_HIGH, BINS_LOW]
            )
            hist_max_intens += counts_mx

        # Count events per daily bin — separately for all wet days and days with wet hours
        for j in range(nbins_low):
            in_bin         = (rain_d_wet >= BINS_LOW[j]) & (rain_d_wet < BINS_LOW[j + 1])
            in_bin_wet_hrs = in_bin & has_wet_hours
            n_days_all_by_bin[j]     += int(in_bin.sum())
            n_days_wet_hrs_by_bin[j] += int(in_bin_wet_hrs.sum())

print(f"  Histogram accumulation time: {elapsed(t0)}")

# Normalize columns → conditional probability (each daily bin sums to 1)
intens_pdf  = np.zeros_like(hist_intensity)
n_wet_pdf   = np.zeros_like(hist_n_wet)
max_int_pdf = np.zeros_like(hist_max_intens)

for j in range(nbins_low):
    si = hist_intensity[:, j].sum()
    if si > 0:
        intens_pdf[:, j] = hist_intensity[:, j] / si
    sn = hist_n_wet[:, j].sum()
    if sn > 0:
        n_wet_pdf[:, j] = hist_n_wet[:, j] / sn
    sm = hist_max_intens[:, j].sum()
    if sm > 0:
        max_int_pdf[:, j] = hist_max_intens[:, j] / sm

# Convert PDFs → CDFs
#   wet_cdf  : (nbins_low, N_INTERVAL)   CDF over n_wet = 1..24
#   hour_cdf : (nbins_low, nbins_high)   CDF over hourly intensity bins
#   max_cdf  : (nbins_low, nbins_high)   CDF over daily-max intensity bins
wet_cdf  = np.cumsum(n_wet_pdf.T,   axis=1).astype(np.float32)
hour_cdf = np.cumsum(intens_pdf.T,  axis=1).astype(np.float32)
max_cdf  = np.cumsum(max_int_pdf.T, axis=1).astype(np.float32)

# P(n_wet >= 1 | daily bin): fraction of wet days that had at least one wet hour.
# Used in the sampler to stochastically skip drizzle days (n_wet=0).
p_has_wet_hours = np.where(
    n_days_all_by_bin > 0,
    n_days_wet_hrs_by_bin / n_days_all_by_bin,
    0.0
).astype(np.float32)

subsection("Condprob diagnostic — events and P(n_wet>=1) per daily bin")
bin_lo_vals      = BINS_LOW[:-1]
bin_hi_vals      = BINS_LOW[1:]
bin_centers_high = np.append((BINS_HIGH[:-2] + BINS_HIGH[1:-1]) / 2.0,
                               BINS_HIGH[-2] + EXP_SCALE)
n_wet_vals       = np.arange(1, N_INTERVAL + 1, dtype=float)

print(f"  {'Daily bin':>14}  {'N_all':>7}  {'N_wet_hrs':>10}  "
      f"{'P(n_wet>=1)':>12}  {'E[n_wet]':>10}  {'E[max] mm':>10}")
for j in range(nbins_low):
    if n_days_all_by_bin[j] == 0:
        continue
    hi_str   = f"{bin_hi_vals[j]:.0f}" if not np.isinf(bin_hi_vals[j]) else "inf"
    bin_lbl  = f"[{bin_lo_vals[j]:.0f},{hi_str})"
    p_wet    = float(p_has_wet_hours[j])
    mean_nwet = float(np.dot(n_wet_vals, n_wet_pdf[:, j])) if n_wet_pdf[:, j].sum() > 0 else float('nan')
    mean_max  = float(np.dot(bin_centers_high, max_int_pdf[:, j])) if max_int_pdf[:, j].sum() > 0 else float('nan')
    print(f"  {bin_lbl:>14}  {n_days_all_by_bin[j]:>7d}  {n_days_wet_hrs_by_bin[j]:>10d}  "
          f"{p_wet:>12.3f}  {mean_nwet:>10.2f}  {mean_max:>10.3f}")

subsection("Center pixel vs pooled condprob: E[n_wet] per daily bin")
print(f"  Both filtered to wet days (daily >= {WET_VALUE_LOW}mm) only.")
print(f"  Large differences indicate the center pixel is atypical within the buffer region.")
print(f"\n  {'Daily bin':>14}  {'N_ctr':>7}  {'E[n_wet]_ctr':>14}  {'E[n_wet]_pool':>14}  {'Ratio':>7}")
for j in range(len(BINS_LOW) - 1):
    if ctr_ndays_per_bin[j] == 0 and n_days_all_by_bin[j] == 0:
        continue
    lo, hi  = BINS_LOW[j], BINS_LOW[j + 1]
    hi_str  = f"{hi:.0f}" if not np.isinf(hi) else "inf"
    bin_lbl = f"[{lo:.0f},{hi_str})"
    e_ctr   = ctr_nwet_per_bin[j]
    e_pool  = float(np.dot(n_wet_vals_ref, n_wet_pdf[:, j])) if n_wet_pdf[:, j].sum() > 0 else float('nan')
    ratio   = e_ctr / e_pool if (not np.isnan(e_ctr) and e_pool > 0) else float('nan')
    flag    = "  <-- mismatch" if (not np.isnan(ratio) and (ratio < 0.7 or ratio > 1.3)) else ""
    print(f"  {bin_lbl:>14}  {ctr_ndays_per_bin[j]:>7d}  {e_ctr:>14.2f}  {e_pool:>14.2f}  {ratio:>7.2f}{flag}")

subsection("Center pixel vs pooled condprob: E[hourly intensity] per daily bin")
print(f"  Only wet hours (> {WET_VALUE_HIGH}mm) from days with n_wet>=1 are included.")
print(f"\n  {'Daily bin':>14}  {'N_ctr':>7}  {'E[intens]_ctr':>14}  {'E[intens]_pool':>15}  {'Ratio':>7}")
for j in range(len(BINS_LOW) - 1):
    if ctr_ndays_per_bin[j] == 0 and n_days_all_by_bin[j] == 0:
        continue
    lo, hi  = BINS_LOW[j], BINS_LOW[j + 1]
    hi_str  = f"{hi:.0f}" if not np.isinf(hi) else "inf"
    bin_lbl = f"[{lo:.0f},{hi_str})"
    e_ctr   = ctr_intens_per_bin[j]
    e_pool  = float(np.dot(bin_centers_high, intens_pdf[:, j])) if intens_pdf[:, j].sum() > 0 else float('nan')
    ratio   = e_ctr / e_pool if (not np.isnan(e_ctr) and e_pool > 0) else float('nan')
    flag    = "  <-- mismatch" if (not np.isnan(ratio) and (ratio < 0.7 or ratio > 1.3)) else ""
    print(f"  {bin_lbl:>14}  {ctr_ndays_per_bin[j]:>7d}  {e_ctr:>14.3f}  {e_pool:>15.3f}  {ratio:>7.2f}{flag}")

subsection("Center pixel vs pooled condprob: E[daily-max hourly] per daily bin")
print(f"\n  {'Daily bin':>14}  {'N_ctr':>7}  {'E[max]_ctr':>12}  {'E[max]_pool':>13}  {'Ratio':>7}")
for j in range(len(BINS_LOW) - 1):
    if ctr_ndays_per_bin[j] == 0 and n_days_all_by_bin[j] == 0:
        continue
    lo, hi  = BINS_LOW[j], BINS_LOW[j + 1]
    hi_str  = f"{hi:.0f}" if not np.isinf(hi) else "inf"
    bin_lbl = f"[{lo:.0f},{hi_str})"
    e_ctr   = ctr_max_per_bin[j]
    e_pool  = float(np.dot(bin_centers_high, max_int_pdf[:, j])) if max_int_pdf[:, j].sum() > 0 else float('nan')
    ratio   = e_ctr / e_pool if (not np.isnan(e_ctr) and e_pool > 0) else float('nan')
    flag    = "  <-- mismatch" if (not np.isnan(ratio) and (ratio < 0.7 or ratio > 1.3)) else ""
    print(f"  {bin_lbl:>14}  {ctr_ndays_per_bin[j]:>7d}  {e_ctr:>12.3f}  {e_pool:>13.3f}  {ratio:>7.2f}{flag}")

print(f"\n  Total wet days (daily >= {WET_VALUE_LOW}mm)         : {n_days_all_by_bin.sum():,}")
print(f"  Total wet days with wet hours (n_wet >= 1)  : {n_days_wet_hrs_by_bin.sum():,}")
print(f"  Overall P(n_wet >= 1 | daily >= {WET_VALUE_LOW}mm)   : "
      f"{n_days_wet_hrs_by_bin.sum() / max(n_days_all_by_bin.sum(), 1):.3f}")
print(f"\n  CDF array shapes:")
print(f"    wet_cdf  (nbins_low, N_INTERVAL)   : {wet_cdf.shape}")
print(f"    hour_cdf (nbins_low, nbins_high)   : {hour_cdf.shape}")
print(f"    max_cdf  (nbins_low, nbins_high)   : {max_cdf.shape}")

# --- Analog profile library (for Method C) ---
# Store the raw 24-hour block for every wet day (n_wet>=1, daily>=WET_VALUE_LOW),
# indexed by daily bin.  Method C picks a random analog from the same bin and
# scales it by R/R_analog, preserving the n_wet/intensity negative correlation.
subsection("Building analog profile library (for Method C)")
t0 = time.time()
profiles_by_bin = [[] for _ in range(nbins_low)]
for iy in range(ny_reg):
    for ix in range(nx_reg):
        rain_h_blk = blk_pres[:, :, iy, ix]   # (n_days, 24)
        rain_d     = daily_pres[:, iy, ix]      # (n_days,)
        wet_mask   = rain_d >= WET_VALUE_LOW
        if wet_mask.sum() == 0:
            continue
        rain_h_wet = rain_h_blk[wet_mask, :]
        rain_d_wet = rain_d[wet_mask]
        n_wet      = (rain_h_wet > WET_VALUE_HIGH).sum(axis=1)
        has_wet    = n_wet > 0
        if has_wet.sum() == 0:
            continue
        rain_h_sub = rain_h_wet[has_wet, :]
        rain_d_sub = rain_d_wet[has_wet]
        for k in range(len(rain_d_sub)):
            b = int(np.searchsorted(BINS_LOW[1:], float(rain_d_sub[k])))
            b = min(b, nbins_low - 1)
            profiles_by_bin[b].append((rain_h_sub[k].copy(), float(rain_d_sub[k])))
print(f"  Library built in {elapsed(t0)}")
for j in range(nbins_low):
    if profiles_by_bin[j]:
        lo, hi = BINS_LOW[j], BINS_LOW[j + 1]
        hi_str = f"{hi:.0f}" if not np.isinf(hi) else "inf"
        print(f"    Bin [{lo:.0f},{hi_str}): {len(profiles_by_bin[j])} profiles")


# =============================================================================
# STEP 5 & 6 — Bootstrap synthetic quantiles
# =============================================================================

section("STEP 5 & 6 — Bootstrap synthetic quantiles")
print("  Three sampling methods are run and compared:")
print()
print("  METHOD A — Simplified hourly sampler (hour_cdf only):")
print("    For each wet day:")
print("      0. Bernoulli(P(n_wet>=1 | daily bin)) — skip drizzle days")
print("      1. Sample n_wet from wet_cdf[bin]")
print("      2. Sample n_wet intensities from hour_cdf[bin]")
print("      3. Proportional scaling: intensities * R/sum(intensities)")
print("      Daily max = max of the n_wet scaled values.")
print("      NOTE: overestimates upper quantiles (see Method C for fix).")
print(f"      RUN_METHOD_A = {RUN_METHOD_A}  (set True in config to enable)")
print()
print("  METHOD B — Direct daily-max sampler (max_cdf only):")
print("    For each wet day:")
print("      0. Bernoulli(P(n_wet>=1 | daily bin)) — skip drizzle days")
print("      1. Sample daily max directly from max_cdf[bin]")
print("      Simpler: no n_wet step, no scaling.")
print()
print("  METHOD C — Analog resampling (NEW, fixes Method A):")
print("    For each wet day with daily total R:")
print("      0. Bernoulli(P(n_wet>=1 | daily bin)) — skip drizzle days")
print("      1. Pick a random observed 24-h profile from the same daily bin")
print("      2. Scale all hourly values by R / R_analog")
print("      Preserves the negative n_wet/intensity correlation exactly.")
print()
print(f"  N_SAMPLES = {N_SAMPLES}  (per climate state)")


def _sample_from_bin_val(lo, hi, rng):
    """Sample uniformly from [lo, hi) or exponentially from [lo, inf)."""
    if np.isinf(hi):
        return lo + rng.exponential(EXP_SCALE)
    return lo + (hi - lo) * rng.random()


def generate_synthetic_quantiles(rain_daily_1d, label, seed_base):
    """
    METHOD A — simplified hourly sampler.

    For each wet day:
      0. Bernoulli(p_has_wet_hours[b]) — skip drizzle days
      1. Sample n_wet from wet_cdf[b]
      2. Sample n_wet intensities from hour_cdf[b]
      3. Shift-then-scale: enforce min=WET_VALUE_HIGH, sum=daily_total
      Daily max = max of the n_wet scaled values.

    Returns hourly_q_boot and max_q_boot, both (N_SAMPLES, n_plot_q).
    """
    nbins_h  = len(BINS_HIGH) - 1
    hr_edges = BINS_HIGH.astype(np.float64)
    pq100    = PLOT_QUANTILES * 100.0
    n_q      = len(pq100)

    hourly_q_boot = np.full((N_SAMPLES, n_q), np.nan, dtype=np.float32)
    max_q_boot    = np.full((N_SAMPLES, n_q), np.nan, dtype=np.float32)

    n_wet_days = int((rain_daily_1d > 0).sum())
    print(f"\n  [Method A — {label}]  Wet days to sample from: {n_wet_days}")
    t0_loop = time.time()

    for sample in range(N_SAMPLES):
        rng = np.random.default_rng(seed_base + sample)
        temp_hourly = []
        temp_max    = []

        for R in rain_daily_1d:
            if R <= 0.0 or not np.isfinite(R):
                continue

            b = int(np.searchsorted(BINS_LOW[1:], R))
            b = min(b, wet_cdf.shape[0] - 1)

            # 0. Skip drizzle days
            if rng.random() >= p_has_wet_hours[b]:
                continue

            # 1. Sample number of wet hours
            wet_slice = wet_cdf[b, :]
            if not np.any(wet_slice > 0):
                continue
            Nh = int(np.searchsorted(wet_slice, rng.random())) + 1
            Nh = max(1, min(Nh, N_INTERVAL))

            # 2. Sample Nh intensities from hour_cdf
            hour_slice  = hour_cdf[b, :]
            intensities = np.empty(Nh, dtype=np.float64)
            for k in range(Nh):
                idx = min(int(np.searchsorted(hour_slice, rng.random())), nbins_h - 1)
                intensities[k] = _sample_from_bin_val(hr_edges[idx], hr_edges[idx + 1], rng)

            # 3. Scale proportionally so the Nh values sum to R.
            #    Simple multiplicative scale — preserves the sampled ratios.
            total = float(intensities.sum())
            if total > 0.0:
                intensities = intensities * (R / total)
            else:
                intensities[:] = R / Nh

            temp_max.append(float(intensities.max()))
            temp_hourly.extend(intensities.tolist())

        if len(temp_hourly) > 0:
            arr_h = np.array(temp_hourly, dtype=np.float32)
            wet_h = arr_h[arr_h > WET_VALUE_HIGH]
            if len(wet_h) > 0:
                hourly_q_boot[sample, :] = np.percentile(wet_h, pq100)

        if len(temp_max) > 0:
            arr_m = np.array(temp_max, dtype=np.float32)
            wet_m = arr_m[arr_m > WET_VALUE_HIGH]
            if len(wet_m) > 0:
                max_q_boot[sample, :] = np.percentile(wet_m, pq100)

        if (sample + 1) % 50 == 0 or sample == N_SAMPLES - 1:
            rate = (sample + 1) / (time.time() - t0_loop)
            print(f"    Sample {sample+1:4d}/{N_SAMPLES}  —  {rate:.1f} samples/s")

    valid_h = int(np.sum(~np.isnan(hourly_q_boot[:, -1])))
    valid_m = int(np.sum(~np.isnan(max_q_boot[:, -1])))
    print(f"\n  Samples with valid hourly quantiles : {valid_h}/{N_SAMPLES}")
    print(f"  Samples with valid max    quantiles : {valid_m}/{N_SAMPLES}")
    print(f"  Total time: {elapsed(t0_loop)}")
    return hourly_q_boot, max_q_boot


def generate_synthetic_max_quantiles(rain_daily_1d, label, seed_base):
    """
    METHOD B — direct daily-max sampler.

    For each wet day:
      0. Bernoulli(p_has_wet_hours[b]) — skip drizzle days
      1. Sample daily max directly from max_cdf[b]

    No n_wet sampling. No shift-then-scale. Returns max_q_boot (N_SAMPLES, n_plot_q).
    """
    nbins_h  = len(BINS_HIGH) - 1
    hr_edges = BINS_HIGH.astype(np.float64)
    pq100    = PLOT_QUANTILES * 100.0
    n_q      = len(pq100)

    max_q_boot = np.full((N_SAMPLES, n_q), np.nan, dtype=np.float32)

    n_wet_days = int((rain_daily_1d > 0).sum())
    print(f"\n  [Method B — {label}]  Wet days to sample from: {n_wet_days}")
    t0_loop = time.time()

    for sample in range(N_SAMPLES):
        rng = np.random.default_rng(seed_base + sample)
        temp_max = []

        for R in rain_daily_1d:
            if R <= 0.0 or not np.isfinite(R):
                continue

            b = int(np.searchsorted(BINS_LOW[1:], R))
            b = min(b, max_cdf.shape[0] - 1)

            # 0. Skip drizzle days
            if rng.random() >= p_has_wet_hours[b]:
                continue

            # 1. Sample daily max from max_cdf
            max_slice = max_cdf[b, :]
            if not np.any(max_slice > 0):
                continue
            max_bin = min(int(np.searchsorted(max_slice, rng.random())), nbins_h - 1)
            max_val = _sample_from_bin_val(hr_edges[max_bin], hr_edges[max_bin + 1], rng)
            max_val = float(np.clip(max_val, WET_VALUE_HIGH, R))
            temp_max.append(max_val)

        if len(temp_max) > 0:
            arr_m = np.array(temp_max, dtype=np.float32)
            wet_m = arr_m[arr_m > WET_VALUE_HIGH]
            if len(wet_m) > 0:
                max_q_boot[sample, :] = np.percentile(wet_m, pq100)

        if (sample + 1) % 50 == 0 or sample == N_SAMPLES - 1:
            rate = (sample + 1) / (time.time() - t0_loop)
            print(f"    Sample {sample+1:4d}/{N_SAMPLES}  —  {rate:.1f} samples/s")

    valid_m = int(np.sum(~np.isnan(max_q_boot[:, -1])))
    print(f"\n  Samples with valid max quantiles : {valid_m}/{N_SAMPLES}")
    print(f"  Total time: {elapsed(t0_loop)}")
    return max_q_boot


def generate_analog_quantiles(rain_daily_1d, label, seed_base):
    """
    METHOD C — Analog resampling.

    For each wet day with daily total R in daily bin b:
      0. Bernoulli(p_has_wet_hours[b]) — skip drizzle days
      1. Pick a random observed 24-h profile from profiles_by_bin[b]
      2. Scale all hourly values by R / R_analog

    This preserves the n_wet / mean-intensity negative correlation exactly,
    because n_wet and the relative distribution of intensities come from a
    real observed day in the same daily bin.  Scale factors stay near 1
    within each 5-mm bin, so no amplification of extreme intensities.

    Returns hourly_q_boot and max_q_boot, both (N_SAMPLES, n_plot_q).
    """
    pq100 = PLOT_QUANTILES * 100.0
    n_q   = len(pq100)

    hourly_q_boot = np.full((N_SAMPLES, n_q), np.nan, dtype=np.float32)
    max_q_boot    = np.full((N_SAMPLES, n_q), np.nan, dtype=np.float32)

    n_wet_days = int((rain_daily_1d > 0).sum())
    print(f"\n  [Method C — {label}]  Wet days to sample from: {n_wet_days}")
    t0_loop = time.time()

    for sample in range(N_SAMPLES):
        rng = np.random.default_rng(seed_base + sample)
        temp_hourly = []
        temp_max    = []

        for R in rain_daily_1d:
            if R <= 0.0 or not np.isfinite(R):
                continue

            b = int(np.searchsorted(BINS_LOW[1:], R))
            b = min(b, nbins_low - 1)

            # 0. Skip drizzle days
            if rng.random() >= p_has_wet_hours[b]:
                continue

            # 1. Pick a random analog profile from the same daily bin
            analogs = profiles_by_bin[b]
            if len(analogs) == 0:
                continue
            analog_profile, R_analog = analogs[rng.integers(0, len(analogs))]

            # 2. Scale analog hours so their total matches R
            if R_analog <= 0.0:
                continue
            scaled = analog_profile * (R / R_analog)   # shape (24,)

            temp_max.append(float(scaled.max()))
            wet_vals = scaled[scaled > WET_VALUE_HIGH]
            temp_hourly.extend(wet_vals.tolist())

        if len(temp_hourly) > 0:
            arr_h = np.array(temp_hourly, dtype=np.float32)
            hourly_q_boot[sample, :] = np.percentile(arr_h, pq100)

        if len(temp_max) > 0:
            arr_m = np.array(temp_max, dtype=np.float32)
            wet_m = arr_m[arr_m > WET_VALUE_HIGH]
            if len(wet_m) > 0:
                max_q_boot[sample, :] = np.percentile(wet_m, pq100)

        if (sample + 1) % 50 == 0 or sample == N_SAMPLES - 1:
            rate = (sample + 1) / (time.time() - t0_loop)
            print(f"    Sample {sample+1:4d}/{N_SAMPLES}  —  {rate:.1f} samples/s")

    valid_h = int(np.sum(~np.isnan(hourly_q_boot[:, -1])))
    valid_m = int(np.sum(~np.isnan(max_q_boot[:, -1])))
    print(f"\n  Samples with valid hourly quantiles : {valid_h}/{N_SAMPLES}")
    print(f"  Samples with valid max    quantiles : {valid_m}/{N_SAMPLES}")
    print(f"  Total time: {elapsed(t0_loop)}")
    return hourly_q_boot, max_q_boot


def ci_across_samples(q_boot):
    """
    Collapse (N_SAMPLES, n_q) → (3, n_q) using BOOTSTRAP_QUANTILES = [lo, med, hi].
    """
    n_q = q_boot.shape[1]
    out = np.full((3, n_q), np.nan, dtype=np.float32)
    for iq in range(n_q):
        valid = q_boot[:, iq]
        valid = valid[~np.isnan(valid)]
        if len(valid) > 0:
            out[:, iq] = np.quantile(valid, BOOTSTRAP_QUANTILES)
    return out   # rows: [lo, median, hi]


# Pre-compute daily inputs (dry days zeroed out)
pres_daily_input     = np.where(pres_day_ctr >= WET_VALUE_LOW, pres_day_ctr, 0.0)
fut_daily_input      = np.where(fut_day_ctr  >= WET_VALUE_LOW, fut_day_ctr,  0.0)
pres_daily_buf_input = np.where(daily_pres   >= WET_VALUE_LOW, daily_pres,   0.0).reshape(-1)
fut_daily_buf_input  = np.where(daily_fut    >= WET_VALUE_LOW, daily_fut,    0.0).reshape(-1)

# ---- Method A: simplified hourly sampler (optional) ----
if RUN_METHOD_A:
    syn_pres_h_boot,     syn_pres_dm_boot     = generate_synthetic_quantiles(
        pres_daily_input,     label="PRESENT (center pixel)", seed_base=42)
    syn_fut_h_boot,      syn_fut_dm_boot      = generate_synthetic_quantiles(
        fut_daily_input,      label="FUTURE (center pixel)",  seed_base=999)
    syn_pres_h_boot_buf, syn_pres_dm_boot_buf = generate_synthetic_quantiles(
        pres_daily_buf_input, label="PRESENT (buffer)",       seed_base=142)
    syn_fut_h_boot_buf,  syn_fut_dm_boot_buf  = generate_synthetic_quantiles(
        fut_daily_buf_input,  label="FUTURE (buffer)",        seed_base=1099)
else:
    print("\n  [Method A skipped — RUN_METHOD_A = False]")

# ---- Method B: direct daily-max sampler ----
syn_pres_mx_boot     = generate_synthetic_max_quantiles(
    pres_daily_input,     label="PRESENT (center pixel)", seed_base=242)
syn_fut_mx_boot      = generate_synthetic_max_quantiles(
    fut_daily_input,      label="FUTURE (center pixel)",  seed_base=1999)
syn_pres_mx_boot_buf = generate_synthetic_max_quantiles(
    pres_daily_buf_input, label="PRESENT (buffer)",       seed_base=342)
syn_fut_mx_boot_buf  = generate_synthetic_max_quantiles(
    fut_daily_buf_input,  label="FUTURE (buffer)",        seed_base=2099)

# ---- Method C: analog resampling ----
syn_pres_c_h_boot,     syn_pres_c_dm_boot     = generate_analog_quantiles(
    pres_daily_input,     label="PRESENT (center pixel)", seed_base=542)
syn_fut_c_h_boot,      syn_fut_c_dm_boot      = generate_analog_quantiles(
    fut_daily_input,      label="FUTURE (center pixel)",  seed_base=2999)
syn_pres_c_h_boot_buf, syn_pres_c_dm_boot_buf = generate_analog_quantiles(
    pres_daily_buf_input, label="PRESENT (buffer)",       seed_base=642)
syn_fut_c_h_boot_buf,  syn_fut_c_dm_boot_buf  = generate_analog_quantiles(
    fut_daily_buf_input,  label="FUTURE (buffer)",        seed_base=3099)

# Bootstrap CIs — rows: [lo, median, hi]
I_LO, I_MED, I_HI = 0, 1, 2
if RUN_METHOD_A:
    pres_h_ci      = ci_across_samples(syn_pres_h_boot)
    pres_dm_ci     = ci_across_samples(syn_pres_dm_boot)
    fut_h_ci       = ci_across_samples(syn_fut_h_boot)
    fut_dm_ci      = ci_across_samples(syn_fut_dm_boot)
    pres_h_ci_buf  = ci_across_samples(syn_pres_h_boot_buf)
    pres_dm_ci_buf = ci_across_samples(syn_pres_dm_boot_buf)
    fut_h_ci_buf   = ci_across_samples(syn_fut_h_boot_buf)
    fut_dm_ci_buf  = ci_across_samples(syn_fut_dm_boot_buf)
pres_mx_ci     = ci_across_samples(syn_pres_mx_boot)
fut_mx_ci      = ci_across_samples(syn_fut_mx_boot)
pres_mx_ci_buf = ci_across_samples(syn_pres_mx_boot_buf)
fut_mx_ci_buf  = ci_across_samples(syn_fut_mx_boot_buf)
pres_c_h_ci      = ci_across_samples(syn_pres_c_h_boot)
fut_c_h_ci       = ci_across_samples(syn_fut_c_h_boot)
pres_c_h_ci_buf  = ci_across_samples(syn_pres_c_h_boot_buf)
fut_c_h_ci_buf   = ci_across_samples(syn_fut_c_h_boot_buf)
pres_c_dm_ci     = ci_across_samples(syn_pres_c_dm_boot)
fut_c_dm_ci      = ci_across_samples(syn_fut_c_dm_boot)
pres_c_dm_ci_buf = ci_across_samples(syn_pres_c_dm_boot_buf)
fut_c_dm_ci_buf  = ci_across_samples(syn_fut_c_dm_boot_buf)

cl_lo_str = f"{BOOTSTRAP_QUANTILES[I_LO]:.3f}"
cl_hi_str = f"{BOOTSTRAP_QUANTILES[I_HI]:.3f}"

if RUN_METHOD_A:
    subsection("Method A — hourly intensity CI (center pixel)")
    print(f"  CI: lo={cl_lo_str}  hi={cl_hi_str}")
    print(f"\n  {'Quantile':>10}  {'Obs_Pres':>10}  "
          f"{'SynPres_lo':>12}  {'SynPres_med':>12}  {'SynPres_hi':>12}  {'SynFut_med':>12}")
    for i, q in enumerate(PLOT_QUANTILES):
        print(f"  {q:>10.4f}  {obs_pres_h[i]:>10.4f}  "
              f"{pres_h_ci[I_LO,i]:>12.4f}  {pres_h_ci[I_MED,i]:>12.4f}  "
              f"{pres_h_ci[I_HI,i]:>12.4f}  {fut_h_ci[I_MED,i]:>12.4f}")

subsection("Method C (analog) — hourly intensity CI (center pixel)")
print(f"  CI: lo={cl_lo_str}  hi={cl_hi_str}")
print(f"\n  {'Quantile':>10}  {'Obs_Pres':>10}  "
      f"{'SynPres_lo':>12}  {'SynPres_med':>12}  {'SynPres_hi':>12}  {'SynFut_med':>12}")
for i, q in enumerate(PLOT_QUANTILES):
    print(f"  {q:>10.4f}  {obs_pres_h[i]:>10.4f}  "
          f"{pres_c_h_ci[I_LO,i]:>12.4f}  {pres_c_h_ci[I_MED,i]:>12.4f}  "
          f"{pres_c_h_ci[I_HI,i]:>12.4f}  {fut_c_h_ci[I_MED,i]:>12.4f}")

subsection("Method C (analog) — hourly intensity CI (buffer)")
print(f"\n  {'Quantile':>10}  {'Obs_Pres_buf':>13}  "
      f"{'SynPres_lo':>12}  {'SynPres_med':>12}  {'SynPres_hi':>12}  {'SynFut_med':>12}")
for i, q in enumerate(PLOT_QUANTILES):
    print(f"  {q:>10.4f}  {obs_pres_h_buf[i]:>13.4f}  "
          f"{pres_c_h_ci_buf[I_LO,i]:>12.4f}  {pres_c_h_ci_buf[I_MED,i]:>12.4f}  "
          f"{pres_c_h_ci_buf[I_HI,i]:>12.4f}  {fut_c_h_ci_buf[I_MED,i]:>12.4f}")

if RUN_METHOD_A:
    subsection("Method A — daily-max CI, center pixel vs buffer")
    print(f"  {'Quantile':>10}  {'Obs_Pres_ctr':>13}  {'SynPresA_med':>13}  "
          f"{'Obs_Pres_buf':>13}  {'SynPresA_buf':>13}")
    for i, q in enumerate(PLOT_QUANTILES):
        print(f"  {q:>10.4f}  {obs_pres_dm[i]:>13.4f}  {pres_dm_ci[I_MED,i]:>13.4f}  "
              f"{obs_pres_dm_buf[i]:>13.4f}  {pres_dm_ci_buf[I_MED,i]:>13.4f}")

subsection("Method B — direct daily-max CI, center pixel vs buffer")
print(f"  {'Quantile':>10}  {'Obs_Pres_ctr':>13}  {'SynPresB_med':>13}  "
      f"{'Obs_Pres_buf':>13}  {'SynPresB_buf':>13}")
for i, q in enumerate(PLOT_QUANTILES):
    print(f"  {q:>10.4f}  {obs_pres_dm[i]:>13.4f}  {pres_mx_ci[I_MED,i]:>13.4f}  "
          f"{obs_pres_dm_buf[i]:>13.4f}  {pres_mx_ci_buf[I_MED,i]:>13.4f}")


# =============================================================================
# STEP 7 — Plots
# =============================================================================

section("STEP 7 — Plots")

q_axis  = PLOT_QUANTILES
buf_str = f"buf={BUFFER}"
cl_lo   = BOOTSTRAP_QUANTILES[I_LO]
cl_hi   = BOOTSTRAP_QUANTILES[I_HI]


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
    """Overlay a single CI band (median line + shaded envelope) on ax."""
    ax.plot(q_axis, ci[I_MED], color=color, linewidth=1.8,
            linestyle=linestyle, label=label, zorder=5)
    ax.fill_between(q_axis, ci[I_LO], ci[I_HI], color=color, alpha=alpha)


def _plot_panel(ax, obs_pres, obs_fut, ci_lo, ci_med, ci_hi, syn_label, ylabel, title):
    ax.plot(q_axis, obs_pres, color='#2E86AB', linewidth=1.8, marker='o',
            markersize=5, label='Present observed', zorder=4)
    ax.plot(q_axis, obs_fut,  color='#E50C0C',  linewidth=1.5, marker='s',
            markersize=4, linestyle='--', label='Future observed', zorder=3)
    ax.plot(q_axis, ci_med,   color='#F18F01',  linewidth=1.8, linestyle='-',
            label=syn_label, zorder=5)
    ax.plot(q_axis, ci_lo,    color='#F18F01',  linewidth=0.8, linestyle=':')
    ax.plot(q_axis, ci_hi,    color='#F18F01',  linewidth=0.8, linestyle=':')
    ax.fill_between(q_axis, ci_lo, ci_hi, color='#F18F01', alpha=0.2,
                    label=f'Synthetic CI ({cl_lo}–{cl_hi})')
    _setup_ax(ax, ylabel, title)
    _dedup_legend(ax, fontsize=9, loc='upper left', frameon=True,
                  fancybox=True, shadow=True)


# Colours for the four CI bands
# Method A: orange (present) / purple (future)
# Method B: green  (present) / teal   (future)  — daily-max panels only
COL_A_PRES = '#F18F01'
COL_A_FUT  = '#9B59B6'
COL_B_PRES = '#3BB273'
COL_B_FUT  = '#1A936F'

# ---- VALIDATION: 2×2 figure — rows: center pixel / buffer; cols: hourly / daily-max ----
print("  Generating validation figure (2x2: center pixel + buffer) ...")
fig1, axes1 = plt.subplots(2, 2, figsize=(14, 11))

# Left column  — hourly intensity,   Method C (analog) CI
# Right column — daily-max hourly,   Method B CI

COL_C_PRES = '#F18F01'   # orange — Method C present (replaces Method A in left panels)
COL_C_FUT  = '#9B59B6'   # purple — Method C future

# --- Row 0, col 0: center pixel hourly — Method C (analog) ---
_plot_panel(axes1[0, 0],
            obs_pres_h, obs_fut_h,
            pres_c_h_ci[I_LO], pres_c_h_ci[I_MED], pres_c_h_ci[I_HI],
            syn_label='Method C (analog) — synthetic present',
            ylabel='1-hour precipitation (mm)',
            title=f'Hourly intensity — center pixel\n{LOCATION}, {buf_str}')
_add_ci(axes1[0, 0], fut_c_h_ci, COL_C_FUT, 'Method C (analog) — synthetic future')
_dedup_legend(axes1[0, 0], fontsize=8, loc='upper left', frameon=True)

# --- Row 0, col 1: center pixel daily-max — Method B ---
_plot_panel(axes1[0, 1],
            obs_pres_dm, obs_fut_dm,
            pres_mx_ci[I_LO], pres_mx_ci[I_MED], pres_mx_ci[I_HI],
            syn_label='Method B — synthetic present',
            ylabel='Daily-max 1-hour precipitation (mm)',
            title=f'Daily-max hourly — center pixel\n{LOCATION}, {buf_str}')
_add_ci(axes1[0, 1], fut_mx_ci, COL_A_FUT, 'Method B — synthetic future')
_dedup_legend(axes1[0, 1], fontsize=8, loc='upper left', frameon=True)

# --- Row 1, col 0: buffer hourly — Method C (analog) ---
_plot_panel(axes1[1, 0],
            obs_pres_h_buf, obs_fut_h_buf,
            pres_c_h_ci_buf[I_LO], pres_c_h_ci_buf[I_MED], pres_c_h_ci_buf[I_HI],
            syn_label='Method C (analog) — synthetic present',
            ylabel='1-hour precipitation (mm)',
            title=f'Hourly intensity — buffer ({ny_reg}x{nx_reg})\n{LOCATION}, {buf_str}')
_add_ci(axes1[1, 0], fut_c_h_ci_buf, COL_C_FUT, 'Method C (analog) — synthetic future')
_dedup_legend(axes1[1, 0], fontsize=8, loc='upper left', frameon=True)

# --- Row 1, col 1: buffer daily-max — Method B ---
_plot_panel(axes1[1, 1],
            obs_pres_dm_buf, obs_fut_dm_buf,
            pres_mx_ci_buf[I_LO], pres_mx_ci_buf[I_MED], pres_mx_ci_buf[I_HI],
            syn_label='Method B — synthetic present',
            ylabel='Daily-max 1-hour precipitation (mm)',
            title=f'Daily-max hourly — buffer ({ny_reg}x{nx_reg})\n{LOCATION}, {buf_str}')
_add_ci(axes1[1, 1], fut_mx_ci_buf, COL_A_FUT, 'Method B — synthetic future')
_dedup_legend(axes1[1, 1], fontsize=8, loc='upper left', frameon=True)

fig1.suptitle(
    f'Validation: observed vs synthetic present & future — {LOCATION}\n'
    f'Top: center pixel   |   Bottom: buffer pooled ({ny_reg}x{nx_reg} pixels)',
    fontsize=13, fontweight='bold')
fig1.tight_layout()
out_val = f'{PATH_OUT}/validation_{LOCATION}_{buf_str}.png'
fig1.savefig(out_val, dpi=150, bbox_inches='tight', facecolor='white')
plt.close(fig1)
print(f"  Saved: {out_val}")

# ---- ATTRIBUTION: future observed vs synthetic future CI ----
print("  Generating attribution figure (future obs vs synthetic future) ...")
fig2, axes2 = plt.subplots(1, 2, figsize=(14, 6))

for ax, obs_pres, obs_fut, ci, ylabel in [
    (axes2[0], obs_pres_h,  obs_fut_h,  fut_c_h_ci,  '1-hour precipitation (mm)'),
    (axes2[1], obs_pres_dm, obs_fut_dm, fut_mx_ci,   'Daily-max 1-hour precipitation (mm)'),
]:
    _plot_panel(ax,
                obs_pres, obs_fut,
                ci[I_LO], ci[I_MED], ci[I_HI],
                syn_label='Synthetic future (median)',
                ylabel=ylabel,
                title=f'{ylabel.split("(")[0].strip()} — attribution\n{LOCATION}, {buf_str}')

    # Decomposition box at P99 (if P99 is in PLOT_QUANTILES)
    idx_99 = np.searchsorted(q_axis, 0.99)
    if idx_99 < len(q_axis) and np.isclose(q_axis[idx_99], 0.99, atol=0.001):
        total  = float(obs_fut[idx_99]) - float(obs_pres[idx_99])
        daily  = float(ci[I_MED, idx_99]) - float(obs_pres[idx_99])
        struct = float(obs_fut[idx_99])   - float(ci[I_MED, idx_99])
        pct_d  = 100.0 * daily  / abs(total) if total != 0 else 0.0
        pct_s  = 100.0 * struct / abs(total) if total != 0 else 0.0
        ax.text(
            0.03, 0.04,
            f'P99 total Δ : {total:+.2f} mm\n'
            f'Daily effect : {daily:+.2f} mm  ({pct_d:.0f}%)\n'
            f'Structural   : {struct:+.2f} mm  ({pct_s:.0f}%)',
            transform=ax.transAxes, fontsize=8.5, verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.75))

fig2.suptitle(
    f'Attribution: future observed vs synthetic future — {LOCATION}',
    fontsize=13, fontweight='bold')
fig2.tight_layout()
out_attr = f'{PATH_OUT}/attribution_{LOCATION}_{buf_str}.png'
fig2.savefig(out_attr, dpi=150, bbox_inches='tight', facecolor='white')
plt.close(fig2)
print(f"  Saved: {out_attr}")


# =============================================================================
# STEP 8 — Attribution summary table
# =============================================================================

section("STEP 8 — Attribution summary (hourly intensity)")
print(f"  Location: {LOCATION}   Buffer: {BUFFER}   N_SAMPLES: {N_SAMPLES}")
print(f"\n  Decomposition:")
print(f"    Total change    = obs_future  − obs_present")
print(f"    Daily effect    = syn_future  − obs_present   (change explained by daily totals alone)")
print(f"    Structural eff. = obs_future  − syn_future    (residual intra-daily change)")

hdr = (f"  {'Quantile':>10}  {'Obs_Pres':>10}  {'Obs_Fut':>10}  {'Syn_Fut':>10}  "
       f"{'Total Δ':>10}  {'Daily Δ':>10}  {'Struct Δ':>10}  "
       f"{'Daily%':>8}  {'Struct%':>8}")
sep = "  " + "-" * (len(hdr) - 2)

def _attr_row(op, of, sf):
    total  = of - op
    daily  = sf - op
    struct = of - sf
    pct_d  = 100.0 * daily  / abs(total) if total != 0 else 0.0
    pct_s  = 100.0 * struct / abs(total) if total != 0 else 0.0
    return total, daily, struct, pct_d, pct_s

subsection("Center pixel  (Method C synthetic, center-pixel daily input)")
print(f"  N wet days — present: {int((pres_day_ctr >= WET_VALUE_LOW).sum())}   "
      f"future: {int((fut_day_ctr >= WET_VALUE_LOW).sum())}")
print()
print(hdr)
print(sep)
for i, q in enumerate(PLOT_QUANTILES):
    total, daily, struct, pct_d, pct_s = _attr_row(
        float(obs_pres_h[i]), float(obs_fut_h[i]), float(fut_c_h_ci[I_MED, i]))
    print(f"  {q:>10.4f}  {obs_pres_h[i]:>10.3f}  {obs_fut_h[i]:>10.3f}  "
          f"{fut_c_h_ci[I_MED,i]:>10.3f}  "
          f"{total:>+10.3f}  {daily:>+10.3f}  {struct:>+10.3f}  "
          f"{pct_d:>+8.1f}%  {pct_s:>+8.1f}%")

subsection(f"Buffer pooled  (Method C synthetic, {ny_reg}x{nx_reg} buffer daily input)")
n_pres_buf = int((daily_pres >= WET_VALUE_LOW).sum())
n_fut_buf  = int((daily_fut  >= WET_VALUE_LOW).sum())
print(f"  N wet pixel-days — present: {n_pres_buf:,}   future: {n_fut_buf:,}")
print()
print(hdr)
print(sep)
for i, q in enumerate(PLOT_QUANTILES):
    total, daily, struct, pct_d, pct_s = _attr_row(
        float(obs_pres_h_buf[i]), float(obs_fut_h_buf[i]), float(fut_c_h_ci_buf[I_MED, i]))
    print(f"  {q:>10.4f}  {obs_pres_h_buf[i]:>10.3f}  {obs_fut_h_buf[i]:>10.3f}  "
          f"{fut_c_h_ci_buf[I_MED,i]:>10.3f}  "
          f"{total:>+10.3f}  {daily:>+10.3f}  {struct:>+10.3f}  "
          f"{pct_d:>+8.1f}%  {pct_s:>+8.1f}%")

print()
print("  Done.")
