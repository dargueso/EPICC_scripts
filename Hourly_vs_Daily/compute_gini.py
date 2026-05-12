#!/usr/bin/env python
"""
compute_gini.py

Compute Gini coefficients of high-frequency rainfall conditioned on
low-frequency totals, for both present and future climate runs.

For each grid cell and each low-frequency (e.g. daily) rainfall bin,
the Gini coefficient measures how unequally rainfall is distributed
across the high-frequency (e.g. hourly) timesteps within each period.
A value of 0 means perfectly uniform distribution; 1 means all rainfall
falls in a single timestep.

Architecture follows pipeline_full_domain.py:
  - Full-domain low-frequency totals pre-computed tile-by-tile and cached
  - Tile loop sequential (controls I/O); per-pixel work parallelised with
    a fork-based Pool (large tile arrays shared copy-on-write, no serialisation)

Outputs (one file per experiment):
  {PATH_OUT}/{WRUN_PRESENT}/gini_{FREQ_HIGH}_given_{FREQ_LOW}.nc
  {PATH_OUT}/{WRUN_FUTURE}/gini_{FREQ_HIGH}_given_{FREQ_LOW}.nc
"""

import os
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

FREQ_HIGH = '01H'   # Options: '10MIN', '01H'
FREQ_LOW  = 'DAY'   # Options: '01H', 'DAY'

# Number of high-frequency intervals per low-frequency period
_N_INTERVAL_MAP = {
    ('10MIN', '01H'): 6,
    ('10MIN', 'DAY'): 144,
    ('01H',   'DAY'): 24,
}
if (FREQ_HIGH, FREQ_LOW) not in _N_INTERVAL_MAP:
    raise ValueError(
        f"Unsupported frequency pair ({FREQ_HIGH}, {FREQ_LOW}). "
        f"Valid pairs: {list(_N_INTERVAL_MAP.keys())}"
    )
N_INTERVAL = _N_INTERVAL_MAP[(FREQ_HIGH, FREQ_LOW)]

WET_VALUE_HIGH = 0.1   # mm — high-frequency wet threshold
WET_VALUE_LOW  = 0.1   # mm — low-frequency wet threshold

BUFFER      = 10    # spatial pooling radius (grid cells) — matches pipeline_full_domain.py

TILE_SIZE   = 50    # inner tile dimension (pixels)
N_PROCESSES = 32    # parallel workers per tile (pixels)

# Rainfall bins — from pipeline_full_domain.py
BINS_HIGH = np.concatenate([
    np.arange(0,    1.0,  0.1),   # 0.1 mm bins below 1 mm
    np.arange(1.0,  10.0, 1.0),   # 1 mm bins for 1–10 mm
    np.arange(10.0, 100,  5),     # 5 mm bins from 10 mm up
    [np.inf]
]).astype(np.float64)

BINS_LOW = np.concatenate([
    np.arange(0,   1.0, 0.25),    # 0.25 mm bins below 1 mm
    np.arange(1.0, 5.0, 1.0),     # 1 mm bins for 1–5 mm
    np.arange(5.0, 100, 5),       # 5 mm bins from 5 mm up
    [np.inf]
]).astype(np.float64)

# =============================================================================
# MODULE-LEVEL GLOBALS  (set per tile before fork; read-only in workers)
# =============================================================================
_BLK_PRES   = None   # (n_days_p, N_INTERVAL, ny_inner, nx_inner)  float32
_BLK_FUT    = None   # (n_days_f, N_INTERVAL, ny_inner, nx_inner)  float32
_DAILY_PRES = None   # (n_days_p, ny_inner, nx_inner)               float32
_DAILY_FUT  = None   # (n_days_f, ny_inner, nx_inner)               float32

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


# =============================================================================
# GINI COEFFICIENT
# =============================================================================

def gini_coefficient(x):
    """
    Gini coefficient for a 1-D array (including zeros).

    Parameters
    ----------
    x : array-like
        Values (e.g. hourly rainfall within one day). Zeros are included.

    Returns
    -------
    float
        Gini coefficient in [0, 1]. Returns NaN if x is empty or all-NaN.
        Returns 0.0 if all values are zero (perfect equality).
    """
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return np.nan
    if np.all(x == 0):
        return 0.0
    sorted_x = np.sort(x)
    n = len(sorted_x)
    idx = np.arange(1, n + 1)
    return (2.0 * np.sum(idx * sorted_x)) / (n * sorted_x.sum()) - (n + 1.0) / n


# =============================================================================
# PER-PIXEL WORKER  (executed inside fork-Pool workers)
# =============================================================================

def process_pixel(args):
    """
    Compute per-bin Gini coefficients for one grid cell (present + future).

    Reads module globals set by the parent process (shared copy-on-write
    under fork — zero serialisation cost).

    Parameters
    ----------
    args : tuple
        (iy, ix) — pixel coordinates within the current tile.

    Returns
    -------
    dict with keys:
        iy, ix,
        gini_pres     : (nbins_low,)  float32  — mean Gini per bin, present
        gini_fut      : (nbins_low,)  float32  — mean Gini per bin, future
        n_events_pres : (nbins_low,)  int32    — wet event count per bin, present
        n_events_fut  : (nbins_low,)  int32    — wet event count per bin, future
    """
    iy, ix = args
    nbins_low = len(BINS_LOW) - 1

    def _gini_for_run(blk, daily):
        """
        blk   : (n_days, N_INTERVAL)  — high-frequency values
        daily : (n_days,)             — low-frequency totals
        """
        wet_mask = daily >= WET_VALUE_LOW
        if not wet_mask.any():
            return (np.full(nbins_low, np.nan, dtype=np.float32),
                    np.zeros(nbins_low, dtype=np.int32))

        blk_wet   = blk[wet_mask]      # (n_wet, N_INTERVAL)
        daily_wet = daily[wet_mask]    # (n_wet,)

        gini_per_period = np.array(
            [gini_coefficient(blk_wet[i, :]) for i in range(len(daily_wet))],
            dtype=np.float32
        )

        # Assign each wet period to a low-frequency rainfall bin
        bin_idx = np.searchsorted(BINS_LOW[1:], daily_wet).clip(0, nbins_low - 1)

        gini_bins = np.full(nbins_low, np.nan, dtype=np.float32)
        n_bins    = np.zeros(nbins_low, dtype=np.int32)

        for b in range(nbins_low):
            mask_b = bin_idx == b
            n_b = mask_b.sum()
            if n_b > 0:
                gini_bins[b] = np.nanmean(gini_per_period[mask_b])
                n_bins[b]    = n_b

        return gini_bins, n_bins

    gini_p, n_p = _gini_for_run(_BLK_PRES[:, :, iy, ix], _DAILY_PRES[:, iy, ix])
    gini_f, n_f = _gini_for_run(_BLK_FUT[:, :, iy, ix],  _DAILY_FUT[:, iy, ix])

    return {
        'iy': iy, 'ix': ix,
        'gini_pres':     gini_p,
        'gini_fut':      gini_f,
        'n_events_pres': n_p,
        'n_events_fut':  n_f,
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    global _BLK_PRES, _BLK_FUT, _DAILY_PRES, _DAILY_FUT

    total_start = time.time()
    nbins_low   = len(BINS_LOW) - 1

    zarr_pres = f'{PATH_IN}/{WRUN_PRESENT}/UIB_{FREQ_HIGH}_RAIN.zarr'
    zarr_fut  = f'{PATH_IN}/{WRUN_FUTURE}/UIB_{FREQ_HIGH}_RAIN.zarr'

    # -------------------------------------------------------------------------
    # PHASE 1 — Open zarr, get domain dimensions, pre-compute low-freq totals
    # -------------------------------------------------------------------------
    section("PHASE 1 — Domain dimensions + low-frequency totals")

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
    print(f"  Present  : {nt_pres} {FREQ_HIGH} → {n_days_pres} {FREQ_LOW} periods", flush=True)
    print(f"  Future   : {nt_fut} {FREQ_HIGH} → {n_days_fut} {FREQ_LOW} periods", flush=True)

    ny_tiles    = int(np.ceil(ny / TILE_SIZE))
    nx_tiles    = int(np.ceil(nx / TILE_SIZE))
    total_tiles = ny_tiles * nx_tiles
    print(f"  Tiles    : {ny_tiles} × {nx_tiles} = {total_tiles}  "
          f"(inner {TILE_SIZE}×{TILE_SIZE} px)", flush=True)

    print("\n  Computing low-frequency totals (tile-by-tile) …", flush=True)
    t0 = time.time()
    daily_pres_full = np.zeros((n_days_pres, ny, nx), dtype=np.float32)
    daily_fut_full  = np.zeros((n_days_fut,  ny, nx), dtype=np.float32)

    for iy_t in range(ny_tiles):
        for ix_t in range(nx_tiles):
            y0 = iy_t * TILE_SIZE;  y1 = min(y0 + TILE_SIZE, ny)
            x0 = ix_t * TILE_SIZE;  x1 = min(x0 + TILE_SIZE, nx)

            r = ds_pres.RAIN.isel(y=slice(y0, y1), x=slice(x0, x1)).values
            r = r[:n_days_pres * N_INTERVAL].reshape(
                n_days_pres, N_INTERVAL, y1 - y0, x1 - x0)
            daily_pres_full[:, y0:y1, x0:x1] = r.sum(axis=1).astype(np.float32)

            r = ds_fut.RAIN.isel(y=slice(y0, y1), x=slice(x0, x1)).values
            r = r[:n_days_fut * N_INTERVAL].reshape(
                n_days_fut, N_INTERVAL, y1 - y0, x1 - x0)
            daily_fut_full[:, y0:y1, x0:x1] = r.sum(axis=1).astype(np.float32)

        pct = 100 * (iy_t + 1) / ny_tiles
        print(f"    row {iy_t+1}/{ny_tiles}  ({pct:.0f}%)  {elapsed(t0)}", flush=True)

    print(f"  Totals ready  ({elapsed(t0)})", flush=True)

    # -------------------------------------------------------------------------
    # PHASE 2 — Allocate output arrays
    # -------------------------------------------------------------------------
    section("PHASE 2 — Allocating output arrays")

    gini_pres_out     = np.full((nbins_low, ny, nx), np.nan, dtype=np.float32)
    gini_fut_out      = np.full((nbins_low, ny, nx), np.nan, dtype=np.float32)
    n_events_pres_out = np.zeros((nbins_low, ny, nx), dtype=np.int32)
    n_events_fut_out  = np.zeros((nbins_low, ny, nx), dtype=np.int32)

    mem_mb = (gini_pres_out.nbytes * 4) / 1e6
    print(f"  Output arrays: {mem_mb:.1f} MB allocated", flush=True)

    # -------------------------------------------------------------------------
    # PHASE 3 — Tile loop
    # -------------------------------------------------------------------------
    section("PHASE 3 — Processing tiles")

    ctx = get_context('fork')

    for iy_tile in range(ny_tiles):
        for ix_tile in range(nx_tiles):
            t_tile  = time.time()
            tile_id = f"{iy_tile:03d}y_{ix_tile:03d}x"

            y0 = iy_tile * TILE_SIZE;  y1 = min(y0 + TILE_SIZE, ny)
            x0 = ix_tile * TILE_SIZE;  x1 = min(x0 + TILE_SIZE, nx)
            ny_inner = y1 - y0;  nx_inner = x1 - x0

            print(f"\n  Tile {tile_id}  [{y0}:{y1}, {x0}:{x1}]  "
                  f"{ny_inner}×{nx_inner} px", flush=True)

            # Load hourly tile data
            t_io = time.time()
            rain_p = (ds_pres.RAIN
                      .isel(y=slice(y0, y1), x=slice(x0, x1))
                      .values.astype(np.float32))
            rain_f = (ds_fut.RAIN
                      .isel(y=slice(y0, y1), x=slice(x0, x1))
                      .values.astype(np.float32))
            print(f"    IO: {elapsed(t_io)}  "
                  f"({rain_p.nbytes/1e6:.0f}+{rain_f.nbytes/1e6:.0f} MB)",
                  flush=True)

            # Set module globals (shared copy-on-write via fork)
            _BLK_PRES = np.ascontiguousarray(
                rain_p[:n_days_pres * N_INTERVAL]
                .reshape(n_days_pres, N_INTERVAL, ny_inner, nx_inner))
            _BLK_FUT = np.ascontiguousarray(
                rain_f[:n_days_fut * N_INTERVAL]
                .reshape(n_days_fut, N_INTERVAL, ny_inner, nx_inner))
            _DAILY_PRES = np.ascontiguousarray(daily_pres_full[:, y0:y1, x0:x1])
            _DAILY_FUT  = np.ascontiguousarray(daily_fut_full[:,  y0:y1, x0:x1])
            del rain_p, rain_f

            # Dispatch per-pixel jobs
            pixel_jobs = [(iy, ix)
                          for iy in range(ny_inner)
                          for ix in range(nx_inner)]

            t_work = time.time()
            with ctx.Pool(processes=N_PROCESSES) as pool:
                results = pool.map(process_pixel, pixel_jobs, chunksize=4)
            n_px = len(pixel_jobs)
            print(f"    Work: {elapsed(t_work)}  "
                  f"({n_px / (time.time() - t_work):.0f} px/s)", flush=True)

            # Collect results into output arrays
            for res in results:
                iy_i = res['iy'];  ix_i = res['ix']
                gy   = y0 + iy_i;  gx  = x0 + ix_i

                gini_pres_out[:, gy, gx]     = res['gini_pres']
                gini_fut_out[:, gy, gx]      = res['gini_fut']
                n_events_pres_out[:, gy, gx] = res['n_events_pres']
                n_events_fut_out[:, gy, gx]  = res['n_events_fut']

            done = iy_tile * nx_tiles + ix_tile + 1
            eta  = (total_tiles - done) * (time.time() - t_tile) / 60
            print(f"    Tile done: {elapsed(t_tile)}  "
                  f"[{done}/{total_tiles}]  ETA {eta:.0f} min", flush=True)

    ds_pres.close()
    ds_fut.close()

    # -------------------------------------------------------------------------
    # PHASE 4 — Write output NetCDF files
    # -------------------------------------------------------------------------
    section("PHASE 4 — Writing outputs")

    # Bin coordinate: lower edge of each bin (clear and unambiguous)
    bin_edges_low = BINS_LOW[:-1]   # drop the trailing inf
    bin_coord_name = f'bin_{FREQ_LOW}'

    enc = {'dtype': 'float32', 'zlib': True, 'complevel': 4}

    common_attrs = dict(
        freq_high=FREQ_HIGH,
        freq_low=FREQ_LOW,
        n_interval=N_INTERVAL,
        wet_threshold_high=WET_VALUE_HIGH,
        wet_threshold_low=WET_VALUE_LOW,
        bin_edges_low=BINS_LOW[:-1].tolist(),   # store edges for plot labels
        description=(
            f'Gini coefficient of {FREQ_HIGH} rainfall distribution within '
            f'{FREQ_LOW} periods, binned by {FREQ_LOW} rainfall amount.'
        ),
    )

    y_coord = ds_pres.y.values if 'y' in ds_pres.coords else np.arange(ny)
    x_coord = ds_pres.x.values if 'x' in ds_pres.coords else np.arange(nx)

    for wrun, gini_out, n_events_out in [
        (WRUN_PRESENT, gini_pres_out, n_events_pres_out),
        (WRUN_FUTURE,  gini_fut_out,  n_events_fut_out),
    ]:
        out_dir  = f'{PATH_OUT}/{wrun}'
        out_file = f'{out_dir}/gini_{FREQ_HIGH}_given_{FREQ_LOW}_buf{BUFFER:02d}.nc'
        os.makedirs(out_dir, exist_ok=True)

        ds_out = xr.Dataset(
            {
                'gini_coefficient': (
                    [bin_coord_name, 'y', 'x'],
                    gini_out,
                    {
                        'long_name': (
                            f'Mean Gini coefficient of {FREQ_HIGH} rainfall '
                            f'within {FREQ_LOW} periods'
                        ),
                        'description': (
                            f'For each wet {FREQ_LOW} period (>={WET_VALUE_LOW} mm), '
                            f'the Gini coefficient of the {N_INTERVAL} {FREQ_HIGH} '
                            f'values within that period is computed (including dry '
                            f'timesteps). Values are then averaged across all periods '
                            f'falling within each {FREQ_LOW} rainfall bin. '
                            f'0 = perfectly uniform, 1 = fully concentrated.'
                        ),
                        'units': 'dimensionless (0-1)',
                    }
                ),
                'n_events': (
                    [bin_coord_name, 'y', 'x'],
                    n_events_out,
                    {
                        'long_name': f'Number of wet {FREQ_LOW} periods per bin',
                        'description': (
                            f'Count of wet {FREQ_LOW} periods (>={WET_VALUE_LOW} mm) '
                            f'in each rainfall bin. Use as weight when averaging '
                            f'gini_coefficient over space.'
                        ),
                        'units': 'count',
                    }
                ),
                'lat': (['y', 'x'], lat2d,
                        {'long_name': 'Latitude',  'units': 'degrees_north'}),
                'lon': (['y', 'x'], lon2d,
                        {'long_name': 'Longitude', 'units': 'degrees_east'}),
            },
            coords={
                bin_coord_name: (
                    bin_coord_name,
                    bin_edges_low,
                    {
                        'long_name': f'{FREQ_LOW} rainfall bin lower edge',
                        'units': 'mm',
                        'description': 'Lower edge of each rainfall bin. Last bin is open-ended (to inf).',
                    }
                ),
                'y': y_coord,
                'x': x_coord,
            },
            attrs={**common_attrs, 'wrun': wrun},
        )

        print(f"  Writing {out_file} …", flush=True)
        t0 = time.time()
        ds_out.to_netcdf(out_file, encoding={'gini_coefficient': enc})
        print(f"  Saved  ({elapsed(t0)})", flush=True)

    section(f"COMPLETE — Total time: {(time.time() - total_start) / 3600:.2f} h")


if __name__ == '__main__':
    main()
