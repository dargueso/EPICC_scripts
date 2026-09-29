#!/usr/bin/env python
"""
Rebuild EPICC hourly rainfall on true clock hours from the 10-min zarr store.

WHY. The 10-min rainfall is WRF's PREC_ACC_NC from the wrfprec files: rain
accumulated over the LAST prec_acc_dt = 10 minutes, keeping the wrfprec time
stamp. So the value stamped 13:00 is rain from 12:50 to 13:00. The original
hourly files (create_multiple_freq_files_parallel.py, `cdo hoursum`) group the
stamps 13:00..13:50, i.e. rain from 12:50 to 13:50: every hour is 10 minutes
early. Verified against RAINNC in the hourly wrfout of 2020-01-10: summing the
stamps HH:10..HH+1:00 reproduces the exact RAINNC hourly total (median error
3e-6 mm); summing HH:00..HH:50 does not (0.07 mm).

WHAT. Hour HH (HH:00-HH+1:00) = sum of the 10-min values stamped HH:10 ..
HH+1:00. The last hour of the record (2020-12-31 23:00) needs 2021-01-01 00:00,
which does not exist, so it is NaN.

HOW. Two stages, because the 10-min store is chunked as 50x50-cell tiles x
8760 steps and uncompressed (1.9 TB per run): reading it month by month
re-reads every chunk twice and takes days.
  1  tiles: each 50x50 tile is read once over the whole record, summed to
     clock hours and written to an intermediate tile-chunked hourly store.
  2  months: the intermediate store is cut into
       - UIB_01H_RAIN.zarr, one field per chunk, zstd, like the original
         UIB_01H_RAIN.zarr (lat/lon stored once, 2D, not per time step)
       - monthly UIB_01H_RAIN_YYYY-MM.nc, same names as the original files
     Both are stamped at the centre of the hour (HH:30) with
     time_bnds = [HH:00, HH+1:00], so code that floors the stamp gets HH:00,
     exactly as with the old HH:25 stamps.
Nothing existing is modified or deleted: the original files were used in
another manuscript. Output goes to
  /scratch3/dargueso/postprocessed/EPICC/<run>/RAIN/   (first written as RAIN_clockhour/)

Needs zarr >= 3: run with the ExtremeRainML environment.

    python fix_hourly_rain_clockhour.py                 # both runs, both stages
    python fix_hourly_rain_clockhour.py EPICC_2km_ERA5  # one run
"""

import os
import sys
import time
import logging
import multiprocessing as mp

# Workers are SPAWNED, not forked. zarr 3 runs an asyncio I/O thread; a child
# forked after the parent has used zarr (or xarray/netCDF) inherits that
# thread's locks but not the thread, and hangs on its first zarr/netCDF call.
# That is exactly what happened to stage 2 on the first run.
CTX = mp.get_context("spawn")

import numpy as np
import pandas as pd
import xarray as xr
import dask.array as da
import zarr
from zarr.codecs import ZstdCodec

PATH_IN = "/home/dargueso/postprocessed/EPICC"
PATH_OUT = "/scratch3/dargueso/postprocessed/EPICC"
RUNS = ["EPICC_2km_ERA5", "EPICC_2km_ERA5_CMIP6anom"]
T0 = pd.Timestamp("2011-01-01")          # first 10-min stamp of the store
TILE = 50
NWORK_TILES = 24                          # ~6 GB per worker
NWORK_MONTHS = 12                         # ~3 GB per worker
TCHUNK = 720                              # hours per chunk in the intermediate store


def paths(wrun):
    # written as RAIN_clockhour on 2026-09-29, then renamed to RAIN: it is the
    # definitive hourly rain on /scratch3 (the originals stay on /scratch1)
    d = f"{PATH_OUT}/{wrun}/RAIN"
    return {"dir": d, "src": f"{PATH_IN}/{wrun}/UIB_10MIN_RAIN.zarr",
            "tmp": f"{d}/intermediate_tiles_01H.zarr",
            "zarr": f"{d}/UIB_01H_RAIN.zarr",
            "done1": f"{d}/.stage1_tiles_done"}


###########################################################
# Stage 1: tiles -> clock-hour sums
###########################################################

def stage1_tile(args):
    wrun, y0, x0 = args
    p = paths(wrun)
    src = zarr.open_array(f"{p['src']}/RAIN", mode="r")
    dst = zarr.open_array(p["tmp"], mode="r+")
    ys, xs = slice(y0, y0 + TILE), slice(x0, x0 + TILE)
    # stamps from 00:10 on the first day: hour k = stamps 6k+1 .. 6k+6
    v = src[1:, ys, xs]
    nh = v.shape[0] // 6
    out = np.full((dst.shape[0],) + v.shape[1:], np.nan, dtype="float32")
    out[:nh] = v[:nh * 6].reshape(nh, 6, *v.shape[1:]).sum(axis=1)
    dst[:, ys, xs] = out
    return y0, x0


def stage1(wrun):
    p = paths(wrun)
    if os.path.exists(p["done1"]):
        logging.info("%s: stage 1 already done", wrun)
        return
    src = zarr.open_array(f"{p['src']}/RAIN", mode="r")
    nt10, ny, nx = src.shape
    nh = nt10 // 6                        # 87672 hours; the last one is incomplete
    os.makedirs(p["dir"], exist_ok=True)
    zarr.create_array(store=p["tmp"], shape=(nh, ny, nx), chunks=(TCHUNK, TILE, TILE),
                      dtype="float32", fill_value=np.nan, compressors=ZstdCodec(level=3),
                      overwrite=True)
    tiles = [(wrun, y0, x0) for y0 in range(0, ny, TILE) for x0 in range(0, nx, TILE)]
    t0 = time.time()
    with CTX.Pool(NWORK_TILES) as pool:
        for n, _ in enumerate(pool.imap_unordered(stage1_tile, tiles), 1):
            if n % 25 == 0 or n == len(tiles):
                logging.info("%s stage 1: %d/%d tiles, %.0f s", wrun, n, len(tiles),
                             time.time() - t0)
    open(p["done1"], "w").close()


###########################################################
# Stage 2: months -> final zarr + monthly netCDF
###########################################################

def hours_axis(nh):
    start = pd.date_range(T0, periods=nh, freq="h")
    return start


def first_field(v):
    """2D lat/lon, without loading a time-dimensioned copy of it."""
    return (v.isel(time=0) if "time" in v.dims else v).values


def make_final_store(wrun, nh, ny, nx):
    """Empty final zarr with coordinates; RAIN filled month by month."""
    p = paths(wrun)
    if os.path.exists(f"{p['zarr']}/zarr.json"):
        return
    z10 = xr.open_zarr(p["src"])
    start = hours_axis(nh)
    ds = xr.Dataset(
        {"RAIN": (("time", "y", "x"),
                  da.full((nh, ny, nx), np.nan, chunks=(1, ny, nx), dtype="float32"),
                  {"standard_name": "Accumulated rainfall",
                   "long_name": "Accumulated rainfall over the clock hour",
                   "units": "mm", "cell_methods": "time: sum",
                   "comment": "sum of the 10-min PREC_ACC_NC values stamped "
                              "HH:10..HH+1:00, i.e. rain from HH:00 to HH+1:00"}),
         "time_bnds": (("time", "bnds"),
                       np.stack([start.values,
                                 (start + pd.Timedelta(hours=1)).values], axis=1)),
         "lat": (("y", "x"), first_field(z10.lat)),
         "lon": (("y", "x"), first_field(z10.lon))},
        coords={"time": ("time", (start + pd.Timedelta(minutes=30)).values,
                         {"bounds": "time_bnds"})},
        attrs=global_attrs(wrun))
    enc = {"RAIN": {"chunks": (1, ny, nx), "compressors": ZstdCodec(level=3)}}
    ds.to_zarr(p["zarr"], mode="w", compute=False, encoding=enc, zarr_format=3)


def global_attrs(wrun):
    return {"title": f"{wrun} hourly rainfall on clock hours",
            "source": f"{PATH_IN}/{wrun}/UIB_10MIN_RAIN.zarr",
            "history": f"{pd.Timestamp.now():%Y-%m-%d} fix_hourly_rain_clockhour.py "
                       "(EPICC_scripts/WRF_processing)",
            "correction": "the original UIB_01H_RAIN files (cdo hoursum of end-stamped "
                          "10-min values) cover HH-1:50 to HH:50; these cover HH:00 to "
                          "HH+1:00, verified against WRF RAINNC",
            "author": "Daniel Argueso @UIB", "contact": "d.argueso@uib.es"}


_COORDS = {}


def stage2_init(coords):
    """Coordinates read once by the parent, handed to each spawned worker."""
    _COORDS.update(coords)


def stage2_month(args):
    wrun, tag = args
    p = paths(wrun)
    fout = f"{p['dir']}/UIB_01H_RAIN_{tag}.nc"
    tmp = zarr.open_array(p["tmp"], mode="r")
    start = hours_axis(tmp.shape[0])
    m0 = pd.Timestamp(f"{tag}-01")
    i0 = start.get_loc(m0)
    i1 = i0 + pd.Period(tag, "M").days_in_month * 24
    data = tmp[i0:i1]
    # final zarr: this month's chunks belong to this worker only
    zarr.open_array(f"{p['zarr']}/RAIN", mode="r+")[i0:i1] = data
    if not os.path.exists(fout):
        c = _COORDS
        ds = xr.Dataset(
            {"RAIN": (("time", "y", "x"), data, c["rain_attrs"]),
             "time_bnds": (("time", "bnds"), c["time_bnds"][i0:i1]),
             "lat": (("y", "x"), c["lat"]), "lon": (("y", "x"), c["lon"])},
            coords={"time": ("time", c["time"][i0:i1], {"bounds": "time_bnds"})},
            attrs=global_attrs(wrun))
        # one units string for time and its bounds, as CF asks
        units = "minutes since 2011-01-01 00:00:00"
        enc = {"RAIN": {"zlib": True, "complevel": 4, "chunksizes": (24, 250, 250)},
               "time": {"units": units, "dtype": "float64"},
               "time_bnds": {"units": units, "dtype": "float64"}}
        ds.to_netcdf(f"{fout}.tmp", encoding=enc)
        os.replace(f"{fout}.tmp", fout)
    return tag


def stage2(wrun):
    p = paths(wrun)
    tmp = zarr.open_array(p["tmp"], mode="r")
    nh, ny, nx = tmp.shape
    make_final_store(wrun, nh, ny, nx)
    start = hours_axis(nh)
    tags = sorted({t.strftime("%Y-%m") for t in start})
    with xr.open_zarr(p["zarr"]) as z:
        coords = {"time": z.time.values, "time_bnds": z.time_bnds.values,
                  "lat": z.lat.values, "lon": z.lon.values,
                  "rain_attrs": dict(z.RAIN.attrs)}
    t0 = time.time()
    with CTX.Pool(NWORK_MONTHS, initializer=stage2_init, initargs=(coords,)) as pool:
        for n, tag in enumerate(pool.imap_unordered(stage2_month,
                                                    [(wrun, t) for t in tags]), 1):
            if n % 12 == 0 or n == len(tags):
                logging.info("%s stage 2: %d/%d months, %.0f s", wrun, n, len(tags),
                             time.time() - t0)


def main():
    logging.basicConfig(format="%(asctime)s | %(message)s", level=logging.INFO)
    runs = sys.argv[1:] or RUNS
    for wrun in runs:
        stage1(wrun)
        stage2(wrun)
        logging.info("%s done", wrun)


if __name__ == "__main__":
    main()
