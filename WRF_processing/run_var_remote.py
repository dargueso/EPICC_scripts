#!/usr/bin/env python
"""
Lean, restart-safe driver to postprocess one 3-D diagnostic (default: WAZ) from
daily wrf3hrly files, one output file per day: <path_out>/<wrun>/UIB_<var>_<date>.nc

Meant to run next to the raw WRF output (e.g. on eady3) so that only the small
postprocessed files need to be copied elsewhere. It does not need ncks nor the
geo_em file: fields that wrf3hrly lacks (HGT, PSFC, XLAT, XLONG) are taken
from the same-day wrfout file, subsampled to the wrf3hrly time steps.

Usage:
  python run_var_remote.py --wrun EPICC_2km_ERA5 --syear 2011 --eyear 2020 \
      --var WAZ --njobs 2

Days whose output file already exists (and opens) are skipped, so the script can
be relaunched after an interruption.
"""

import argparse
import calendar
import datetime as dt
import os
import sys
import traceback
from glob import glob

import netCDF4 as nc
import numpy as np
from joblib import Parallel, delayed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import compute_vars as cvars  # noqa: E402
from wrf_utils import wrftime2date  # noqa: E402

INSTITUTION = "UIB"
# wrf3hrly files carry only the 3-D fields; these are read from the same-day wrfout
FROM_WRFOUT = ["HGT", "PSFC", "XLAT", "XLONG"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--wrun", default="EPICC_2km_ERA5")
    p.add_argument("--var", default="WAZ")
    p.add_argument("--syear", type=int, default=2011)
    p.add_argument("--eyear", type=int, default=2020)
    p.add_argument("--smonth", type=int, default=1)
    p.add_argument("--emonth", type=int, default=12)
    p.add_argument("--njobs", type=int, default=2)
    p.add_argument("--patt", default="wrf3hrly")
    p.add_argument("--dom", default="d01")
    p.add_argument("--path_wrfo", default=os.path.expanduser("~/WRF_OUT"))
    p.add_argument("--path_out", default=os.path.expanduser("~/postprocessed/EPICC/temp"))
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--months", nargs="+", type=int, metavar="M", help="keep only these calendar months (e.g. 8 9 10 11)")
    p.add_argument("--stop-at", metavar="HH:MM", help="submit no new day after this local time (in-flight days finish); days left over are reported as deferred")
    p.add_argument("--dates", nargs="+", metavar="YYYY-MM-DD", help="process only these days (ignores --syear/--eyear)")
    return p.parse_args()


def output_ok(fileout, varn):
    try:
        with nc.Dataset(fileout) as f:
            return varn in f.variables and f.variables["time"].size > 0
    except Exception:
        return False


def process_day(args, date):
    sdate = date.strftime("%Y-%m-%d")
    fullpathin = f"{args.path_wrfo}/{args.wrun}/out"
    fullpathout = f"{args.path_out}/{args.wrun}"
    fileout = f"{fullpathout}/{INSTITUTION}_{args.var}_{sdate}.nc"

    if not args.overwrite and output_ok(fileout, args.var):
        return (sdate, "skip")
    if args.deadline is not None and dt.datetime.now() >= args.deadline:
        return (sdate, "deferred (past stop time)")

    filesin = sorted(glob(f"{fullpathin}/{args.patt}_{args.dom}_{sdate}*"))
    if not filesin:
        return (sdate, "missing input")

    t0 = dt.datetime.now()
    try:
        x, t = [], []
        for filename in filesin:
            ncfile = nc.Dataset(filename)
            ntimes = ncfile.dimensions["Time"].size
            fwrfout = None
            for sname in FROM_WRFOUT:
                if sname in ncfile.variables:
                    continue
                if fwrfout is None:
                    fwrfout = nc.Dataset(filename.replace(args.patt, "wrfout"))
                    nt_out = fwrfout.dimensions["Time"].size
                    if nt_out % ntimes != 0:
                        raise ValueError(
                            f"{filename}: {ntimes} steps vs {nt_out} in wrfout, cannot subsample"
                        )
                    step = nt_out // ntimes
                # subsample the (hourly) wrfout to the wrf3hrly time steps so that
                # wrf-python sees a consistent time dimension (same as ncks -d Time,0,23,3)
                ncfile.variables[sname] = np.asarray(fwrfout.variables[sname][::step])

            xFragment, atts = cvars.compute_WRFvar(ncfile, args.var)
            tFragment = wrftime2date([filename])
            # wrf-python squeezes the time axis of single-step files: restore it
            if xFragment.shape[0] != len(tFragment):
                xFragment = np.expand_dims(xFragment, axis=0)
            lat = ncfile.variables["XLAT"][0, :]
            lon = ncfile.variables["XLONG"][0, :]
            ncfile.close()
            if fwrfout is not None:
                fwrfout.close()
            x.append(xFragment)
            t.append(tFragment)

        varinfo = {
            "values": np.concatenate(x, axis=0),
            "varname": args.var,
            "atts": atts,
            "lat": lat,
            "lon": lon,
            "times": np.concatenate(t, axis=0),
        }
        tmpout = fileout + ".tmp"
        cvars.create_netcdf(varinfo, tmpout)
        os.replace(tmpout, fileout)
    except Exception:
        return (sdate, "ERROR\n" + traceback.format_exc())

    return (sdate, "done in %.0f s" % (dt.datetime.now() - t0).total_seconds())


def main():
    args = parse_args()
    os.makedirs(f"{args.path_out}/{args.wrun}", exist_ok=True)

    if args.dates:
        date_list = [dt.datetime.strptime(d, "%Y-%m-%d") for d in args.dates]
    else:
        d1 = dt.datetime(args.syear, args.smonth, 1)
        d2 = dt.datetime(args.eyear, args.emonth, calendar.monthrange(args.eyear, args.emonth)[1]) + dt.timedelta(days=1)
        date_list = [d1 + dt.timedelta(days=i) for i in range((d2 - d1).days)]
    if args.months:
        date_list = [d for d in date_list if d.month in args.months]
    args.deadline = None
    if args.stop_at:
        h, m = (int(x) for x in args.stop_at.split(":"))
        args.deadline = dt.datetime.now().replace(hour=h, minute=m, second=0, microsecond=0)
        if args.deadline <= dt.datetime.now():
            args.deadline += dt.timedelta(days=1)
    print(f"{args.var} {args.wrun}: {len(date_list)} days, {args.njobs} workers"
          + (f", no new day after {args.deadline:%Y-%m-%d %H:%M}" if args.deadline else ""), flush=True)

    results = Parallel(n_jobs=args.njobs, verbose=0, return_as="generator")(
        delayed(process_day)(args, date) for date in date_list
    )
    nerr = ndef = 0
    for sdate, status in results:
        nerr += status.startswith("ERROR")
        ndef += status.startswith("deferred")
        print(f"{dt.datetime.now():%Y-%m-%d %H:%M:%S} {sdate}: {status}", flush=True)
    print(f"finished, {nerr} errors, {ndef} deferred", flush=True)


if __name__ == "__main__":
    main()
