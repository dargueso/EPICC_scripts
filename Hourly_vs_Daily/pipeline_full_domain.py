#!/usr/bin/env python
"""
pipeline_full_domain.py

Full-domain synthetic rainfall pipeline with spatial buffer pooling.

For each grid cell, conditional probability histograms and analog profiles are
pooled from a (2·BUFFER+1)² neighbourhood.  Bootstrap sampling then generates
present-period synthetic distributions (for validation) and future-period
synthetic distributions (for attribution).

Architecture
------------
  • Present and future DAILY totals pre-computed tile-by-tile and cached in
    RAM (~876 MB) so they are available as fast numpy slices throughout.
  • Tile loop is sequential (controls I/O load); within each tile the
    per-pixel work is parallelised with a fork-based Pool so the large tile
    data arrays are shared copy-on-write (no serialisation cost).
  • Tile halo extends BUFFER cells beyond each tile boundary so that every
    inner pixel has its full (2·BUFFER+1)² neighbourhood available.  At
    domain edges the halo is clipped to domain bounds; those border pixels
    simply pool fewer neighbours, which is consistent with the single-
    location pipeline behaviour.

Methods
-------
  Method C (analog resampling) — hourly intensity quantiles
  Method B (direct dmax CDF)   — daily-max intensity quantiles
  Both run for present (condprob from present, daily totals from present)
  and for future   (condprob from present, daily totals from future).

Outputs  (written to PATH_OUT / WRUN_PRESENT/)
---------
  synthetic_pres_buf{B}.nc   present synthetic CIs + observed percentiles
  synthetic_fut_buf{B}.nc    future  synthetic CIs + observed percentiles
  condprob_buf{B}.nc         buffer-pooled conditional-probability histograms
                             (optional, controlled by SAVE_CONDPROB)
"""

import os
import sys
import time
import warnings
import numpy as np
import xarray as xr
from multiprocessing import get_context

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

PATH_IN      = '/home/dargueso/postprocessed/EPICC/'
PATH_OUT     = '/home/dargueso/postprocessed/EPICC/'
WRUN_PRESENT = 'EPICC_2km_ERA5'
WRUN_FUTURE  = 'EPICC_2km_ERA5_CMIP6anom'

BUFFER      = 3    # spatial pooling radius (grid cells)
TILE_SIZE   = 50   # inner tile dimension (pixels)
N_SAMPLES   = 200  # bootstrap iterations per pixel
N_PROCESSES = 128   # parallel workers (pixels within each tile)

WET_VALUE_HIGH = 0.1   # mm/h — hourly wet threshold
WET_VALUE_LOW  = 1.0   # mm/d — daily wet threshold
N_INTERVAL     = 24    # hourly steps per day

BINS_HIGH = np.append(np.arange(0, 100, 1), np.inf).astype(np.float64)
BINS_LOW  = np.append(np.arange(0, 100, 5), np.inf).astype(np.float64)
EXP_SCALE = 5.0

PLOT_QUANTILES      = np.array([0.90, 0.95, 0.98, 0.99, 0.995, 0.999])
BOOTSTRAP_QUANTILES = np.array([0.025, 0.5, 0.975])

SAVE_CONDPROB = True   # write buffer-pooled condprob histograms

# =============================================================================
# MODULE-LEVEL GLOBALS  (set per tile before fork; read-only in workers)
# =============================================================================
_BLK_PRES   = None   # (n_days_p, N_INTERVAL, ny_h, nx_h)  float32
_BLK_FUT    = None   # (n_days_f, N_INTERVAL, ny_h, nx_h)  float32
_DAILY_PRES = None   # (n_days_p, ny_h, nx_h)               float32
_DAILY_FUT  = None   # (n_days_f, ny_h, nx_h)               float32
_TILE_META  = None   # dict: tile geometry + config

# =============================================================================
# UTILITIES
# =============================================================================

def _open_zarr(path):
    try:
        return xr.open_zarr(path, consolidated=True)
    except Exception:
        return xr.open_zarr(path, consolidated=False)


def section(msg):
    ts = time.strftime('%H:%M:%S')
    print(f"\n{'='*65}\n  [{ts}]  {msg}\n{'='*65}", flush=True)


def elapsed(t0):
    return f"{time.time() - t0:.1f}s"


def ci_from_boot(q_boot, bq):
    """(N_SAMPLES, n_q) → (3, n_q)  rows: [lo, median, hi]"""
    n_q = q_boot.shape[1]
    out = np.full((3, n_q), np.nan, dtype=np.float32)
    for iq in range(n_q):
        v = q_boot[:, iq]
        v = v[~np.isnan(v)]
        if len(v):
            out[:, iq] = np.quantile(v, bq)
    return out


def _sample_bin(lo, hi, rng):
    """Uniform sample from [lo, hi); exponential tail when hi = inf."""
    if np.isinf(hi):
        return lo + rng.exponential(EXP_SCALE)
    return lo + (hi - lo) * rng.random()


# =============================================================================
# HISTOGRAM + CDF BUILDER
# =============================================================================

def build_histograms(blk_nbr, daily_nbr):
    """
    Build conditional-probability histograms from a pooled neighbourhood.

    Parameters
    ----------
    blk_nbr   : (n_days, N_INTERVAL, nbr_ny, nbr_nx)  float32
    daily_nbr : (n_days, nbr_ny, nbr_nx)               float32

    Returns
    -------
    wet_cdf   : (nbins_low, N_INTERVAL)  cumulative P(n_wet | daily bin)
    hour_cdf  : (nbins_low, nbins_high)  cumulative P(hourly | daily bin)
    max_cdf   : (nbins_low, nbins_high)  cumulative P(dmax   | daily bin)
    p_has_wet : (nbins_low,)             P(at least one wet hour | daily bin)
    n_events  : (nbins_low,)             count of wet days per daily bin
    intens_pdf: (nbins_high, nbins_low)  raw PDF (for saving)
    n_wet_pdf : (N_INTERVAL, nbins_low)  raw PDF (for saving)
    max_pdf   : (nbins_high, nbins_low)  raw PDF (for saving)
    """
    nbins_high = len(BINS_HIGH) - 1
    nbins_low  = len(BINS_LOW)  - 1
    n_wet_bins = np.arange(0.5, N_INTERVAL + 1.5, 1)

    wet_day_mask = (daily_nbr >= WET_VALUE_LOW)                    # (n_days, ny, nx)
    rain_clipped = np.where(
        wet_day_mask[:, np.newaxis, :, :] & (blk_nbr >= WET_VALUE_HIGH),
        blk_nbr, 0.0).astype(np.float64)                          # (n_days, 24, ny, nx)

    n_wet_all     = (rain_clipped > WET_VALUE_HIGH).sum(axis=1)   # (n_days, ny, nx)
    has_wet_hours = wet_day_mask & (n_wet_all > 0)

    # Histogram 1: P(hourly intensity | daily total)
    d_rep  = np.repeat(daily_nbr[:, np.newaxis, :, :].astype(np.float64),
                       N_INTERVAL, axis=1)
    h_flat = rain_clipped.reshape(-1)
    d_flat = d_rep.reshape(-1)
    wet_h  = h_flat >= WET_VALUE_HIGH
    hist_intensity, _, _ = np.histogram2d(h_flat[wet_h], d_flat[wet_h],
                                          bins=[BINS_HIGH, BINS_LOW])
    del d_rep, h_flat, d_flat, wet_h

    # Histogram 2: P(n_wet | daily total)
    d_valid = daily_nbr[has_wet_hours].astype(np.float64)
    hist_n_wet, _, _ = np.histogram2d(
        n_wet_all[has_wet_hours].astype(np.float64), d_valid,
        bins=[n_wet_bins, BINS_LOW])

    # Histogram 3: P(daily-max | daily total)
    dmax_all = rain_clipped.max(axis=1)
    hist_max, _, _ = np.histogram2d(
        dmax_all[has_wet_hours].astype(np.float64), d_valid,
        bins=[BINS_HIGH, BINS_LOW])

    # P(has wet hours | daily bin)
    n_all_by_bin, _     = np.histogram(daily_nbr[wet_day_mask].astype(np.float64),
                                        bins=BINS_LOW)
    n_wet_hrs_by_bin, _ = np.histogram(d_valid, bins=BINS_LOW)
    p_has_wet = np.where(n_all_by_bin > 0,
                         n_wet_hrs_by_bin / n_all_by_bin, 0.0).astype(np.float32)
    n_events = n_all_by_bin.astype(np.int32)

    # Normalize columns → PDFs
    intens_pdf = np.zeros_like(hist_intensity, dtype=np.float32)
    n_wet_pdf  = np.zeros_like(hist_n_wet,     dtype=np.float32)
    max_pdf    = np.zeros_like(hist_max,        dtype=np.float32)
    for j in range(nbins_low):
        si = hist_intensity[:, j].sum()
        if si > 0:
            intens_pdf[:, j] = (hist_intensity[:, j] / si).astype(np.float32)
        sn = hist_n_wet[:, j].sum()
        if sn > 0:
            n_wet_pdf[:, j]  = (hist_n_wet[:, j]     / sn).astype(np.float32)
        sm = hist_max[:, j].sum()
        if sm > 0:
            max_pdf[:, j]    = (hist_max[:, j]        / sm).astype(np.float32)

    # PDFs → CDFs
    wet_cdf  = np.cumsum(n_wet_pdf.T,  axis=1).astype(np.float32)   # (nbins_low, 24)
    hour_cdf = np.cumsum(intens_pdf.T, axis=1).astype(np.float32)   # (nbins_low, 100)
    max_cdf  = np.cumsum(max_pdf.T,    axis=1).astype(np.float32)   # (nbins_low, 100)

    return wet_cdf, hour_cdf, max_cdf, p_has_wet, n_events, intens_pdf, n_wet_pdf, max_pdf


# =============================================================================
# ANALOG LIBRARY BUILDER  (Method C)
# =============================================================================

def build_analog_library(blk_nbr, daily_nbr):
    """
    Build the Method C analog profile library from a pooled neighbourhood.

    Returns
    -------
    profiles_arrays : list[nbins_low]  each element (n_profiles, N_INTERVAL)
    profiles_totals : list[nbins_low]  each element (n_profiles,)
    """
    nbins_low  = len(BINS_LOW) - 1
    wet_day_mask = daily_nbr >= WET_VALUE_LOW
    rain_clipped = np.where(
        wet_day_mask[:, np.newaxis, :, :] & (blk_nbr >= WET_VALUE_HIGH),
        blk_nbr, 0.0).astype(np.float32)

    n_wet_all     = (rain_clipped > WET_VALUE_HIGH).sum(axis=1)
    has_wet_hours = wet_day_mask & (n_wet_all > 0)
    b_all         = np.searchsorted(BINS_LOW[1:], daily_nbr).clip(0, nbins_low - 1)

    # Vectorized extraction: find all (day, iy, ix) triples with valid wet days
    days_i, iy_i, ix_i = np.where(has_wet_hours)           # each shape (N_valid,)
    profs  = rain_clipped[days_i, :, iy_i, ix_i]           # (N_valid, N_INTERVAL)
    tots   = daily_nbr[days_i, iy_i, ix_i].astype(np.float32)
    b_vec  = b_all[days_i, iy_i, ix_i]

    profiles_arrays, profiles_totals = [], []
    for b in range(nbins_low):
        mask = b_vec == b
        profiles_arrays.append(profs[mask].copy())
        profiles_totals.append(tots[mask].copy())
    return profiles_arrays, profiles_totals


# =============================================================================
# SAMPLERS
# =============================================================================

def _method_c_one_sample(rain_daily_1d, profiles_arrays, profiles_totals,
                          nbins_low, p_has_wet, rng):
    """One Method C bootstrap sample → (hourly_q, dmax_q)."""
    pq100 = PLOT_QUANTILES * 100.0
    temp_hourly, temp_max = [], []
    for R in rain_daily_1d:
        b = min(int(np.searchsorted(BINS_LOW[1:], R)), nbins_low - 1)
        if rng.random() >= p_has_wet[b]:
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

    h_row = (np.percentile(np.array(temp_hourly, np.float32), pq100).astype(np.float32)
             if temp_hourly else np.full(len(pq100), np.nan, np.float32))
    if temp_max:
        wet_m = np.array(temp_max, np.float32)
        wet_m = wet_m[wet_m > WET_VALUE_HIGH]
        m_row = (np.percentile(wet_m, pq100).astype(np.float32)
                 if len(wet_m) > 0 else np.full(len(pq100), np.nan, np.float32))
    else:
        m_row = np.full(len(pq100), np.nan, np.float32)
    return h_row, m_row


def _method_b_one_sample(rain_daily_1d, max_cdf, p_has_wet, nbins_low, rng):
    """One Method B bootstrap sample → dmax_q."""
    pq100     = PLOT_QUANTILES * 100.0
    nbins_high = len(BINS_HIGH) - 1
    temp_max  = []
    for R in rain_daily_1d:
        b = min(int(np.searchsorted(BINS_LOW[1:], R)), nbins_low - 1)
        if rng.random() >= p_has_wet[b]:
            continue
        max_slice = max_cdf[b, :]
        if not np.any(max_slice > 0):
            continue
        max_bin = min(int(np.searchsorted(max_slice, rng.random())), nbins_high - 1)
        max_val = float(np.clip(
            _sample_bin(BINS_HIGH[max_bin], BINS_HIGH[max_bin + 1], rng),
            WET_VALUE_HIGH, R))
        temp_max.append(max_val)
    if temp_max:
        arr_m = np.array(temp_max, np.float32)
        wet_m = arr_m[arr_m > WET_VALUE_HIGH]
        if len(wet_m) > 0:
            return np.percentile(wet_m, pq100).astype(np.float32)
    return np.full(len(PLOT_QUANTILES), np.nan, np.float32)


# =============================================================================
# PER-PIXEL WORKER  (executed inside fork-Pool workers)
# =============================================================================

def process_pixel(args):
    """
    Process one grid cell.  Reads module globals set by the parent process
    (shared copy-on-write under fork, zero serialisation cost).

    Returns a dict with CI arrays and histogram data for this pixel.
    """
    iy_inner, ix_inner = args

    meta    = _TILE_META
    buf     = meta['buffer']
    y0      = meta['y0'];   x0  = meta['x0']
    y0h     = meta['y0h'];  x0h = meta['x0h']

    blk_p    = _BLK_PRES    # (n_days_p, 24, ny_h, nx_h)
    blk_f    = _BLK_FUT     # (n_days_f, 24, ny_h, nx_h)
    daily_p  = _DAILY_PRES  # (n_days_p, ny_h, nx_h)
    daily_f  = _DAILY_FUT   # (n_days_f, ny_h, nx_h)

    ny_h    = blk_p.shape[2]
    nx_h    = blk_p.shape[3]
    nbins_low  = len(BINS_LOW)  - 1
    n_q        = len(PLOT_QUANTILES)
    pq100      = PLOT_QUANTILES * 100.0

    # Pixel position in halo array space
    iy_h = (y0 - y0h) + iy_inner
    ix_h = (x0 - x0h) + ix_inner

    # Neighbourhood bounds in halo space (clamped to available data)
    yb0 = max(0, iy_h - buf);  yb1 = min(ny_h, iy_h + buf + 1)
    xb0 = max(0, ix_h - buf);  xb1 = min(nx_h, ix_h + buf + 1)

    blk_nbr   = blk_p[:, :, yb0:yb1, xb0:xb1]
    daily_nbr = daily_p[:, yb0:yb1, xb0:xb1]

    # ------------------------------------------------------------------
    # OBSERVED PERCENTILES  (buffer-pooled neighbourhood)
    # ------------------------------------------------------------------
    blk_f_nbr   = blk_f[:, :, yb0:yb1, xb0:xb1]
    daily_f_nbr = daily_f[:, yb0:yb1, xb0:xb1]

    # Present hourly (buffer-pooled)
    wet_day_mask_p  = (daily_nbr >= WET_VALUE_LOW)[:, np.newaxis, :, :]
    pres_wet_1h_buf = (blk_nbr * wet_day_mask_p).reshape(-1)
    pres_wet_1h_buf = pres_wet_1h_buf[pres_wet_1h_buf > WET_VALUE_HIGH]
    obs_pres_h = (np.percentile(pres_wet_1h_buf, pq100).astype(np.float32)
                  if len(pres_wet_1h_buf) > 0 else np.full(n_q, np.nan, np.float32))

    # Future hourly (buffer-pooled)
    wet_day_mask_f  = (daily_f_nbr >= WET_VALUE_LOW)[:, np.newaxis, :, :]
    fut_wet_1h_buf  = (blk_f_nbr * wet_day_mask_f).reshape(-1)
    fut_wet_1h_buf  = fut_wet_1h_buf[fut_wet_1h_buf > WET_VALUE_HIGH]
    obs_fut_h = (np.percentile(fut_wet_1h_buf, pq100).astype(np.float32)
                 if len(fut_wet_1h_buf) > 0 else np.full(n_q, np.nan, np.float32))

    # Present daily-max (buffer-pooled)
    dmax_pres_all = blk_nbr.max(axis=1)
    nwet_pres_all = (blk_nbr > WET_VALUE_HIGH).sum(axis=1)
    dmax_mask_p   = (daily_nbr >= WET_VALUE_LOW) & (nwet_pres_all > 0)
    obs_pres_dm   = (np.percentile(dmax_pres_all[dmax_mask_p], pq100).astype(np.float32)
                     if dmax_mask_p.sum() > 0 else np.full(n_q, np.nan, np.float32))

    # Future daily-max (buffer-pooled)
    dmax_fut_all = blk_f_nbr.max(axis=1)
    nwet_fut_all = (blk_f_nbr > WET_VALUE_HIGH).sum(axis=1)
    dmax_mask_f  = (daily_f_nbr >= WET_VALUE_LOW) & (nwet_fut_all > 0)
    obs_fut_dm   = (np.percentile(dmax_fut_all[dmax_mask_f], pq100).astype(np.float32)
                    if dmax_mask_f.sum() > 0 else np.full(n_q, np.nan, np.float32))

    # ------------------------------------------------------------------
    # CONDITIONAL PROBABILITY HISTOGRAMS + CDFs
    # ------------------------------------------------------------------
    (wet_cdf, hour_cdf, max_cdf, p_has_wet, n_events,
     intens_pdf, n_wet_pdf, max_pdf) = build_histograms(blk_nbr, daily_nbr)

    # ------------------------------------------------------------------
    # ANALOG PROFILE LIBRARY  (Method C — built from present period only)
    # ------------------------------------------------------------------
    profiles_arrays, profiles_totals = build_analog_library(blk_nbr, daily_nbr)

    # ------------------------------------------------------------------
    # BOOTSTRAP SAMPLING
    # ------------------------------------------------------------------
    # Buffer-pooled wet-day sequences as inputs
    pres_in = daily_nbr[daily_nbr >= WET_VALUE_LOW].reshape(-1)
    fut_in  = daily_f_nbr[daily_f_nbr >= WET_VALUE_LOW].reshape(-1)

    # Deterministic per-pixel seed to ensure reproducibility
    seed_base = int(42 + (y0 + iy_inner) * 100003 + (x0 + ix_inner))

    c_pres_h_boot  = np.full((N_SAMPLES, n_q), np.nan, np.float32)
    c_fut_h_boot   = np.full((N_SAMPLES, n_q), np.nan, np.float32)
    b_pres_dm_boot = np.full((N_SAMPLES, n_q), np.nan, np.float32)
    b_fut_dm_boot  = np.full((N_SAMPLES, n_q), np.nan, np.float32)

    for s in range(N_SAMPLES):
        # Method C — present
        rng = np.random.default_rng(seed_base + s)
        h, _ = _method_c_one_sample(pres_in, profiles_arrays, profiles_totals,
                                     nbins_low, p_has_wet, rng)
        c_pres_h_boot[s] = h

        # Method C — future (present condprob × future daily totals)
        rng = np.random.default_rng(seed_base + 1_000_000 + s)
        h, _ = _method_c_one_sample(fut_in, profiles_arrays, profiles_totals,
                                     nbins_low, p_has_wet, rng)
        c_fut_h_boot[s] = h

        # Method B — present
        rng = np.random.default_rng(seed_base + 2_000_000 + s)
        b_pres_dm_boot[s] = _method_b_one_sample(pres_in, max_cdf, p_has_wet,
                                                  nbins_low, rng)

        # Method B — future
        rng = np.random.default_rng(seed_base + 3_000_000 + s)
        b_fut_dm_boot[s] = _method_b_one_sample(fut_in, max_cdf, p_has_wet,
                                                 nbins_low, rng)

    # Bootstrap CIs
    bq = BOOTSTRAP_QUANTILES
    ci_C_pres_h  = ci_from_boot(c_pres_h_boot,  bq)   # (3, n_q)
    ci_C_fut_h   = ci_from_boot(c_fut_h_boot,   bq)
    ci_B_pres_dm = ci_from_boot(b_pres_dm_boot, bq)
    ci_B_fut_dm  = ci_from_boot(b_fut_dm_boot,  bq)

    return {
        'iy': iy_inner, 'ix': ix_inner,
        'obs_pres_h':   obs_pres_h,    'obs_fut_h':    obs_fut_h,
        'obs_pres_dm':  obs_pres_dm,   'obs_fut_dm':   obs_fut_dm,
        'ci_C_pres_h':  ci_C_pres_h,   'ci_C_fut_h':   ci_C_fut_h,
        'ci_B_pres_dm': ci_B_pres_dm,  'ci_B_fut_dm':  ci_B_fut_dm,
        'hist_intens':  intens_pdf,     # (nbins_high, nbins_low)
        'hist_max':     max_pdf,        # (nbins_high, nbins_low)
        'hist_nwet':    n_wet_pdf,      # (N_INTERVAL, nbins_low)
        'n_events':     n_events,       # (nbins_low,)
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    global _BLK_PRES, _BLK_FUT, _DAILY_PRES, _DAILY_FUT, _TILE_META

    total_start = time.time()
    nbins_high  = len(BINS_HIGH) - 1
    nbins_low   = len(BINS_LOW)  - 1
    n_q         = len(PLOT_QUANTILES)
    n_bq        = len(BOOTSTRAP_QUANTILES)

    zarr_pres = f'{PATH_IN}/{WRUN_PRESENT}/UIB_01H_RAIN.zarr'
    zarr_fut  = f'{PATH_IN}/{WRUN_FUTURE}/UIB_01H_RAIN.zarr'

    # -------------------------------------------------------------------------
    # PHASE 1 — Open zarr, get domain dimensions, pre-compute daily totals
    # -------------------------------------------------------------------------
    section("PHASE 1 — Domain dimensions + daily totals")

    print("  Opening zarr files …", flush=True)
    ds_pres = _open_zarr(zarr_pres)
    ds_fut  = _open_zarr(zarr_fut)

    lat2d = ds_pres.lat.isel(time=0).values.copy()
    lon2d = ds_pres.lon.isel(time=0).values.copy()
    ny, nx = lat2d.shape

    nt_pres     = len(ds_pres.time)
    nt_fut      = len(ds_fut.time)
    n_days_pres = nt_pres // N_INTERVAL
    n_days_fut  = nt_fut  // N_INTERVAL

    print(f"  Domain   : {ny} × {nx} grid cells", flush=True)
    print(f"  Present  : {nt_pres} h → {n_days_pres} days", flush=True)
    print(f"  Future   : {nt_fut} h → {n_days_fut} days", flush=True)

    ny_tiles    = int(np.ceil(ny / TILE_SIZE))
    nx_tiles    = int(np.ceil(nx / TILE_SIZE))
    total_tiles = ny_tiles * nx_tiles
    print(f"  Tiles    : {ny_tiles} × {nx_tiles} = {total_tiles}  "
          f"(inner {TILE_SIZE}×{TILE_SIZE}, halo {BUFFER} px)", flush=True)

    # Pre-compute full-domain daily totals tile-by-tile (memory-efficient)
    print("\n  Computing daily totals (tile-by-tile) …", flush=True)
    t0 = time.time()
    daily_pres_full = np.zeros((n_days_pres, ny, nx), dtype=np.float32)
    daily_fut_full  = np.zeros((n_days_fut,  ny, nx), dtype=np.float32)

    for iy_t in range(ny_tiles):
        for ix_t in range(nx_tiles):
            y0 = iy_t * TILE_SIZE;  y1 = min(y0 + TILE_SIZE, ny)
            x0 = ix_t * TILE_SIZE;  x1 = min(x0 + TILE_SIZE, nx)

            # Present
            r = ds_pres.RAIN.isel(y=slice(y0, y1), x=slice(x0, x1)).values
            r = r[:n_days_pres * N_INTERVAL].reshape(n_days_pres, N_INTERVAL,
                                                      y1 - y0, x1 - x0)
            daily_pres_full[:, y0:y1, x0:x1] = r.sum(axis=1).astype(np.float32)

            # Future
            r = ds_fut.RAIN.isel(y=slice(y0, y1), x=slice(x0, x1)).values
            r = r[:n_days_fut * N_INTERVAL].reshape(n_days_fut, N_INTERVAL,
                                                     y1 - y0, x1 - x0)
            daily_fut_full[:, y0:y1, x0:x1] = r.sum(axis=1).astype(np.float32)

        pct = 100 * (iy_t + 1) / ny_tiles
        print(f"    row {iy_t+1}/{ny_tiles}  ({pct:.0f}%)  {elapsed(t0)}",
              flush=True)

    print(f"  Daily totals ready  ({elapsed(t0)})", flush=True)

    # -------------------------------------------------------------------------
    # PHASE 2 — Pre-allocate output arrays
    # -------------------------------------------------------------------------
    section("PHASE 2 — Allocating output arrays")

    obs_pres_h_out   = np.full((n_q,  ny, nx), np.nan, dtype=np.float32)
    obs_pres_dm_out  = np.full((n_q,  ny, nx), np.nan, dtype=np.float32)
    obs_fut_h_out    = np.full((n_q,  ny, nx), np.nan, dtype=np.float32)
    obs_fut_dm_out   = np.full((n_q,  ny, nx), np.nan, dtype=np.float32)
    ci_C_pres_h_out  = np.full((n_bq, n_q, ny, nx), np.nan, dtype=np.float32)
    ci_C_fut_h_out   = np.full((n_bq, n_q, ny, nx), np.nan, dtype=np.float32)
    ci_B_pres_dm_out = np.full((n_bq, n_q, ny, nx), np.nan, dtype=np.float32)
    ci_B_fut_dm_out  = np.full((n_bq, n_q, ny, nx), np.nan, dtype=np.float32)

    if SAVE_CONDPROB:
        hist_intens_out = np.full((nbins_high, nbins_low, ny, nx),
                                  np.nan, dtype=np.float32)
        hist_max_out    = np.full((nbins_high, nbins_low, ny, nx),
                                  np.nan, dtype=np.float32)
        hist_nwet_out   = np.full((N_INTERVAL, nbins_low, ny, nx),
                                  np.nan, dtype=np.float32)
        n_events_out    = np.zeros((nbins_low, ny, nx), dtype=np.int32)

    total_mem = (obs_pres_h_out.nbytes * 4 +
                 ci_C_pres_h_out.nbytes * 4) / 1e9
    print(f"  Output arrays: {total_mem:.2f} GB allocated", flush=True)

    # -------------------------------------------------------------------------
    # PHASE 3 — Tile loop
    # -------------------------------------------------------------------------
    section("PHASE 3 — Processing tiles")

    ctx = get_context('fork')

    for iy_tile in range(ny_tiles):
        for ix_tile in range(nx_tiles):
            t_tile = time.time()
            tile_id = f"{iy_tile:03d}y_{ix_tile:03d}x"

            # Inner tile bounds
            y0 = iy_tile * TILE_SIZE;  y1 = min(y0 + TILE_SIZE, ny)
            x0 = ix_tile * TILE_SIZE;  x1 = min(x0 + TILE_SIZE, nx)
            ny_inner = y1 - y0;  nx_inner = x1 - x0

            # Halo bounds (clamped to domain — handles domain-edge tiles)
            y0h = max(0, y0 - BUFFER);  y1h = min(ny, y1 + BUFFER)
            x0h = max(0, x0 - BUFFER);  x1h = min(nx, x1 + BUFFER)

            print(f"\n  Tile {tile_id}  "
                  f"inner=[{y0}:{y1}, {x0}:{x1}]  "
                  f"halo=[{y0h}:{y1h}, {x0h}:{x1h}]  "
                  f"{ny_inner}×{nx_inner} px", flush=True)

            # Load hourly tile+halo (present and future)
            t_io = time.time()
            rain_p_tile = ds_pres.RAIN.isel(
                y=slice(y0h, y1h), x=slice(x0h, x1h)).values.astype(np.float32)
            rain_f_tile = ds_fut.RAIN.isel(
                y=slice(y0h, y1h), x=slice(x0h, x1h)).values.astype(np.float32)
            ny_h = y1h - y0h;  nx_h = x1h - x0h

            blk_p_tile = (rain_p_tile[:n_days_pres * N_INTERVAL]
                          .reshape(n_days_pres, N_INTERVAL, ny_h, nx_h))
            blk_f_tile = (rain_f_tile[:n_days_fut * N_INTERVAL]
                          .reshape(n_days_fut,  N_INTERVAL, ny_h, nx_h))
            print(f"    IO: {elapsed(t_io)}  "
                  f"({rain_p_tile.nbytes/1e6:.0f}+{rain_f_tile.nbytes/1e6:.0f} MB)",
                  flush=True)
            del rain_p_tile, rain_f_tile

            # Set module globals (contiguous copies → workers read via CoW fork)
            _BLK_PRES   = np.ascontiguousarray(blk_p_tile)
            _BLK_FUT    = np.ascontiguousarray(blk_f_tile)
            _DAILY_PRES = np.ascontiguousarray(
                daily_pres_full[:, y0h:y1h, x0h:x1h])
            _DAILY_FUT  = np.ascontiguousarray(
                daily_fut_full[:,  y0h:y1h, x0h:x1h])
            _TILE_META  = {
                'buffer': BUFFER,
                'y0': y0, 'y1': y1, 'x0': x0, 'x1': x1,
                'y0h': y0h, 'y1h': y1h, 'x0h': x0h, 'x1h': x1h,
            }
            del blk_p_tile, blk_f_tile

            # Dispatch per-pixel jobs
            pixel_jobs = [(iy, ix)
                          for iy in range(ny_inner)
                          for ix in range(nx_inner)]
            n_px = len(pixel_jobs)

            t_work = time.time()
            with ctx.Pool(processes=N_PROCESSES) as pool:
                results = pool.map(process_pixel, pixel_jobs, chunksize=1)
            print(f"    Work: {elapsed(t_work)}  ({n_px/( time.time()-t_work):.0f} px/s)",
                  flush=True)

            # Collect results into output arrays
            for r in results:
                iy_i = r['iy'];  ix_i = r['ix']
                gy   = y0 + iy_i;  gx  = x0 + ix_i

                obs_pres_h_out[:, gy, gx]    = r['obs_pres_h']
                obs_fut_h_out[:, gy, gx]     = r['obs_fut_h']
                obs_pres_dm_out[:, gy, gx]   = r['obs_pres_dm']
                obs_fut_dm_out[:, gy, gx]    = r['obs_fut_dm']
                ci_C_pres_h_out[:, :, gy, gx]  = r['ci_C_pres_h']
                ci_C_fut_h_out[:, :, gy, gx]   = r['ci_C_fut_h']
                ci_B_pres_dm_out[:, :, gy, gx] = r['ci_B_pres_dm']
                ci_B_fut_dm_out[:, :, gy, gx]  = r['ci_B_fut_dm']
                if SAVE_CONDPROB:
                    hist_intens_out[:, :, gy, gx] = r['hist_intens']
                    hist_max_out[:, :, gy, gx]    = r['hist_max']
                    hist_nwet_out[:, :, gy, gx]   = r['hist_nwet']
                    n_events_out[:, gy, gx]        = r['n_events']

            done = iy_tile * nx_tiles + ix_tile + 1
            eta  = (total_tiles - done) * (time.time() - t_tile) / 60
            print(f"    Tile done: {elapsed(t_tile)}  "
                  f"[{done}/{total_tiles}]  ETA {eta:.0f} min",
                  flush=True)

    ds_pres.close()
    ds_fut.close()

    # -------------------------------------------------------------------------
    # PHASE 4 — Write output NetCDF files
    # -------------------------------------------------------------------------
    section("PHASE 4 — Writing outputs")

    out_dir = f'{PATH_OUT}/{WRUN_PRESENT}'
    os.makedirs(out_dir, exist_ok=True)

    enc = {'dtype': 'float32', 'zlib': True, 'complevel': 4}

    # Coordinate arrays for condprob
    bin_centers_high = (BINS_HIGH[:-2] + BINS_HIGH[1:-1]) / 2
    bin_centers_high = np.append(bin_centers_high, BINS_HIGH[-2] + 5)
    bin_centers_low  = (BINS_LOW[:-2]  + BINS_LOW[1:-1])  / 2
    bin_centers_low  = np.append(bin_centers_low, BINS_LOW[-2] + 5)

    def _write_nc(fname, ds_vars, ds_coords, attrs):
        print(f"  Writing {fname} …", flush=True)
        t0 = time.time()
        enc_map = {k: enc for k in ds_vars if np.issubdtype(
            ds_vars[k][1].dtype, np.floating)}
        xr.Dataset(
            {k: (dims, data, vattrs) for k, (dims, data, vattrs) in ds_vars.items()},
            coords=ds_coords,
            attrs=attrs
        ).to_netcdf(fname, encoding=enc_map)
        print(f"  Saved ({elapsed(t0)})", flush=True)

    common_attrs = dict(
        buffer=BUFFER, n_samples=N_SAMPLES,
        wet_threshold_high=WET_VALUE_HIGH,
        wet_threshold_low=WET_VALUE_LOW,
        wrun_present=WRUN_PRESENT,
        wrun_future=WRUN_FUTURE,
    )
    spatial_coords = dict(plot_q=PLOT_QUANTILES, bootstrap_q=BOOTSTRAP_QUANTILES)

    # --- Present ---
    _write_nc(
        f'{out_dir}/synthetic_pres_buf{BUFFER}.nc',
        {
            'obs_h':    (['plot_q', 'y', 'x'], obs_pres_h_out,
                         {'long_name': 'Observed present hourly quantiles (buffer-pooled)',
                          'units': 'mm/h'}),
            'obs_dm':   (['plot_q', 'y', 'x'], obs_pres_dm_out,
                         {'long_name': 'Observed present daily-max quantiles (buffer-pooled)',
                          'units': 'mm/h'}),
            'syn_h_C':  (['bootstrap_q', 'plot_q', 'y', 'x'], ci_C_pres_h_out,
                         {'long_name': 'Synthetic present hourly CI — Method C',
                          'units': 'mm/h'}),
            'syn_dm_B': (['bootstrap_q', 'plot_q', 'y', 'x'], ci_B_pres_dm_out,
                         {'long_name': 'Synthetic present daily-max CI — Method B',
                          'units': 'mm/h'}),
            'lat':      (['y', 'x'], lat2d, {'long_name': 'Latitude',  'units': 'degrees_north'}),
            'lon':      (['y', 'x'], lon2d, {'long_name': 'Longitude', 'units': 'degrees_east'}),
        },
        spatial_coords,
        {**common_attrs, 'description':
         'Present synthetic rainfall quantiles (condprob from present × present daily totals)'},
    )

    # --- Future ---
    _write_nc(
        f'{out_dir}/synthetic_fut_buf{BUFFER}.nc',
        {
            'obs_h':    (['plot_q', 'y', 'x'], obs_fut_h_out,
                         {'long_name': 'Observed future hourly quantiles (buffer-pooled)',
                          'units': 'mm/h'}),
            'obs_dm':   (['plot_q', 'y', 'x'], obs_fut_dm_out,
                         {'long_name': 'Observed future daily-max quantiles (buffer-pooled)',
                          'units': 'mm/h'}),
            'syn_h_C':  (['bootstrap_q', 'plot_q', 'y', 'x'], ci_C_fut_h_out,
                         {'long_name': 'Synthetic future hourly CI — Method C',
                          'units': 'mm/h'}),
            'syn_dm_B': (['bootstrap_q', 'plot_q', 'y', 'x'], ci_B_fut_dm_out,
                         {'long_name': 'Synthetic future daily-max CI — Method B',
                          'units': 'mm/h'}),
            'lat':      (['y', 'x'], lat2d, {'long_name': 'Latitude',  'units': 'degrees_north'}),
            'lon':      (['y', 'x'], lon2d, {'long_name': 'Longitude', 'units': 'degrees_east'}),
        },
        spatial_coords,
        {**common_attrs, 'description':
         'Future synthetic rainfall quantiles (condprob from present × future daily totals)'},
    )

    # --- Condprob ---
    if SAVE_CONDPROB:
        _write_nc(
            f'{out_dir}/condprob_buf{BUFFER}.nc',
            {
                'hist_intensity':     (['bin_01H', 'bin_DAY', 'y', 'x'],
                                        hist_intens_out,
                                        {'long_name': 'P(hourly | daily)',
                                         'units': 'probability'}),
                'hist_max_intensity': (['bin_01H', 'bin_DAY', 'y', 'x'],
                                        hist_max_out,
                                        {'long_name': 'P(daily-max | daily)',
                                         'units': 'probability'}),
                'hist_n_wet':         (['n_wet_timesteps', 'bin_DAY', 'y', 'x'],
                                        hist_nwet_out,
                                        {'long_name': 'P(n_wet | daily)',
                                         'units': 'probability'}),
                'n_events':           (['bin_DAY', 'y', 'x'], n_events_out,
                                        {'long_name': 'Wet event count per daily bin',
                                         'units': 'count'}),
                'lat':                (['y', 'x'], lat2d, {}),
                'lon':                (['y', 'x'], lon2d, {}),
            },
            {
                'bin_01H':          bin_centers_high,
                'bin_DAY':          bin_centers_low,
                'n_wet_timesteps':  np.arange(1, N_INTERVAL + 1),
            },
            {**common_attrs, 'description':
             f'Buffer-pooled conditional probability histograms (buffer={BUFFER})'},
        )

    section(f"COMPLETE — Total time: {(time.time()-total_start)/3600:.2f} h")


if __name__ == '__main__':
    main()
