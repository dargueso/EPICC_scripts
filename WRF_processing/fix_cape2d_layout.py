#!/usr/bin/env python
"""
Rewrite the CAPE2D files produced before the compute_CAPE2D fix of 2026-09-30
into the intended layout, in place.

Old layout (UIB_03H_CAPE2D_YYYY-MM.nc, made 2024-11): dims (time = 8*ndays,
lev = 8, y, x). wrf.cape_2d returned (4, 8, y, x) per day and the writer took
the first axis as time and the second as lev, so within each day the records
8d+0..3 of `time` hold MCAPE, MCIN, LCL, LFC and 8d+4..7 are fill (the time
variable, written with 8 stamps, stretched the record dimension), while `lev`
holds the eight 3-hourly steps of the day. The time stamps themselves are
right: record 8d+t is stamped at step t of day d.

New layout: (time = 8*ndays, lev = 4, y, x), new[8d+t, v] = old[8d+v, t].
Same time stamps, lat, lon, attributes; `levels` = 0..3 with the variable
names; half the size. Every file is verified against its original before the
original is moved aside (to CAPE2D_oldlayout/, same filesystem, so the move
is instant); nothing is deleted here.

    python fix_cape2d_layout.py            # both runs, all months
    python fix_cape2d_layout.py 2011-08    # just these months
"""

import os
import sys
import time
import logging
from multiprocessing import Pool

import numpy as np
import netCDF4 as nc

ROOT = "/scratch3/dargueso/postprocessed/EPICC"
RUNS = ["EPICC_2km_ERA5", "EPICC_2km_ERA5_CMIP6anom"]
NPERDAY = 8
NVAR = 4
NAMES = "0: MCAPE (J kg-1), 1: MCIN (J kg-1), 2: LCL (m), 3: LFC (m)"
NWORKERS = 6


def convert(fin):
    """Write the corrected file next to fin, verify it, move the original aside."""
    d = os.path.dirname(fin)
    old_dir = f"{d}_oldlayout"
    fout = fin + ".fixed.tmp"
    t0 = time.time()
    with nc.Dataset(fin) as src:
        v = src.variables["CAPE2D"]
        v.set_auto_mask(False)
        nt, nl, ny, nx = v.shape
        assert nl == NPERDAY and nt % NPERDAY == 0, f"{fin}: unexpected shape {v.shape}"
        fill = v._FillValue
        first = v[:NPERDAY, 0]                       # (8, y, x): records of day 0 at step 0
        used = [bool((first[i] != fill).any()) for i in range(NPERDAY)]
        assert used == [True] * NVAR + [False] * (NPERDAY - NVAR), f"{fin}: record pattern {used}"

        if os.path.exists(fout):
            os.remove(fout)
        out = nc.Dataset(fout, "w", format="NETCDF4_CLASSIC")
        out.createDimension("time", None)
        out.createDimension("lev", NVAR)
        out.createDimension("y", ny)
        out.createDimension("x", nx)
        ov = out.createVariable("CAPE2D", "f4", ("time", "lev", "y", "x"), zlib=True, complevel=5,
                                shuffle=True, chunksizes=(1, NVAR, min(375, ny), min(625, nx)),
                                fill_value=fill)
        for a in v.ncattrs():
            if a != "_FillValue":
                ov.setncattr(a, v.getncattr(a))
        ov.setncattr("level_description", NAMES)
        for name in ("time", "lat", "lon"):
            sv = src.variables[name]
            kw = {"fill_value": sv._FillValue} if "_FillValue" in sv.ncattrs() else {}
            o = out.createVariable(name, sv.dtype, sv.dimensions, zlib=True, complevel=5, **kw)
            for a in sv.ncattrs():
                if a != "_FillValue":
                    o.setncattr(a, sv.getncattr(a))
            o[:] = sv[:]
        ol = out.createVariable("levels", "f4", ("lev",), zlib=True, complevel=5)
        ol[:] = np.arange(NVAR)
        ol.setncattr("standard_name", "cape2d_variable")
        ol.setncattr("long_name", NAMES)
        ol.setncattr("units", "")
        ol.setncattr("_CoordinateAxisType", "z")
        for a in src.ncattrs():
            out.setncattr(a, src.getncattr(a))
        out.setncattr("history", f"{time.strftime('%a %b %d %H:%M:%S %Y')}: fix_cape2d_layout.py: axes put in the "
                      f"order (time, variable, y, x); the file written in 2024-11 had the four variables along "
                      f"time and the times of day along lev. Previous history: {src.getncattr('history')}")
        # day by day: old block (8 records, 8 steps, y, x) -> new (8 steps, 4 vars, y, x)
        for day in range(nt // NPERDAY):
            blk = v[day * NPERDAY: day * NPERDAY + NVAR]          # (4, 8, y, x)
            ov[day * NPERDAY: (day + 1) * NPERDAY] = np.transpose(blk, (1, 0, 2, 3))
        out.close()

    # verify: every value of three days, all days at a set of points, finite counts
    with nc.Dataset(fin) as src, nc.Dataset(fout) as new:
        a, b = src.variables["CAPE2D"], new.variables["CAPE2D"]
        a.set_auto_mask(False); b.set_auto_mask(False)
        assert b.shape == (nt, NVAR, ny, nx), b.shape
        assert np.array_equal(src.variables["time"][:], new.variables["time"][:])
        ndays = nt // NPERDAY
        for day in (0, ndays // 2, ndays - 1):
            old = a[day * NPERDAY: day * NPERDAY + NVAR]                 # (4, 8, y, x)
            got = b[day * NPERDAY: (day + 1) * NPERDAY]                  # (8, 4, y, x)
            assert np.array_equal(np.transpose(old, (1, 0, 2, 3)), got), f"{fin}: day {day} mismatch"
        rng = np.random.default_rng(0)
        for _ in range(20):
            j, i = rng.integers(ny), rng.integers(nx)
            old_pt = a[:, :, j, i].reshape(ndays, NPERDAY, NPERDAY)[:, :NVAR, :]   # (ndays, var, step)
            new_pt = b[:, :, j, i].reshape(ndays, NPERDAY, NVAR)                   # (ndays, step, var)
            assert np.array_equal(np.transpose(old_pt, (0, 2, 1)), new_pt), f"{fin}: point {j},{i} mismatch"
    os.makedirs(old_dir, exist_ok=True)
    os.replace(fin, f"{old_dir}/{os.path.basename(fin)}")
    os.replace(fout, fin)
    return f"{os.path.basename(fin)}: {nt} steps, {os.path.getsize(fin) / 1e9:.2f} GB, {time.time() - t0:.0f} s"


def main():
    logging.basicConfig(format="%(asctime)s | %(message)s", datefmt="%H:%M:%S", level=logging.INFO)
    months = sys.argv[1:]
    files = []
    for run in RUNS:
        d = f"{ROOT}/{run}/CAPE2D"
        for f in sorted(os.listdir(d)):
            if f.startswith("UIB_03H_CAPE2D_") and f.endswith(".nc") and (not months or f[15:22] in months):
                files.append(f"{d}/{f}")
    logging.info("%d files", len(files))
    with Pool(NWORKERS) as pool:
        for msg in pool.imap_unordered(convert, files):
            logging.info(msg)
    logging.info("done")


if __name__ == "__main__":
    main()
