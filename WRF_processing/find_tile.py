#!/usr/bin/env python
"""
find_tile_anygrid.py
====================

Return the tile indices (nnlat, nnlon) — formatted “000y-002x” — for an
arbitrary (lat, lon) point, whether your grid is regular (1-D coords) or
curvilinear/irregular (2-D coords).

Example
-------
$ python find_tile_anygrid.py   --lat  37.25   --lon -3.80 \
                                --grid-file ERA5_full_2000.nc \
                                --tile-size 200 \
                                --base-name ERA5_full_2000
→ 000y-002x
→ ERA5_full_2000_000y-002x.nc
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import xarray as xr

# SciPy accelerates look-ups on irregular grids.
try:
    from scipy.spatial import cKDTree
    _HAVE_SCIPY = True
except ModuleNotFoundError:       # graceful, but slower, fallback
    _HAVE_SCIPY = False

EARTH_RADIUS = 6_371_000.0        # in metres


# ----------------------------------------------------------------------
# utilities
# ----------------------------------------------------------------------
def _wrap_lon_to_match_grid(lon_val: float, lon_grid: np.ndarray) -> float:
    """
    Wrap *lon_val* so that it lives in the same longitude convention as
    *lon_grid* (0–360 or –180–180).
    """
    if lon_grid.min() >= 0 and lon_val < 0:
        return lon_val % 360.0
    if lon_grid.max() <= 180 and lon_val > 180:
        return ((lon_val + 180.0) % 360.0) - 180.0
    return lon_val


def _nearest_1d(coord: np.ndarray, value: float) -> int:
    """Return argmin index along a 1-D coordinate array."""
    return int(np.abs(coord - value).argmin())


def _latlon_to_unit_xyz(lat_deg: np.ndarray, lon_deg: np.ndarray) -> np.ndarray:
    """Convert lat/lon arrays (deg) to unit vectors on the sphere (x, y, z)."""
    lat_rad = np.deg2rad(lat_deg)
    lon_rad = np.deg2rad(lon_deg)
    cos_lat = np.cos(lat_rad)
    x = cos_lat * np.cos(lon_rad)
    y = cos_lat * np.sin(lon_rad)
    z = np.sin(lat_rad)
    return np.column_stack((x.ravel(), y.ravel(), z.ravel()))


# ----------------------------------------------------------------------
# core logic
# ----------------------------------------------------------------------
def _nearest_index_2d(lat2d: np.ndarray, lon2d: np.ndarray,
                      lat_val: float, lon_val: float) -> tuple[int, int]:
    """
    Return (j, i) indices of grid point nearest to (lat_val, lon_val)
    for a 2-D curvilinear grid.  Uses cKDTree if SciPy is available,
    else falls back to brute-force NumPy.
    """
    lon_val = _wrap_lon_to_match_grid(lon_val, lon2d)

    if _HAVE_SCIPY:
        xyz = _latlon_to_unit_xyz(lat2d, lon2d)
        tree = cKDTree(xyz)                       # O(N log N)
        xq, yq, zq = _latlon_to_unit_xyz(np.asarray(lat_val),
                                         np.asarray(lon_val))[0]
        _, idx = tree.query([xq, yq, zq])         # O(log N)
    else:
        # fallback: haversine distance to every grid cell (slow for big grids)
        lat_rad = np.deg2rad(lat2d)
        lon_rad = np.deg2rad(lon2d)
        q_lat = np.deg2rad(lat_val)
        q_lon = np.deg2rad(lon_val)
        dlat = lat_rad - q_lat
        dlon = lon_rad - q_lon
        a = (np.sin(dlat / 2.0)**2 +
             np.cos(lat_rad) * np.cos(q_lat) * np.sin(dlon / 2.0)**2)
        idx = np.argmin(a)                        # haversine proxy

    j, i = np.unravel_index(idx, lat2d.shape)
    return int(j), int(i)


def find_tile_indices(ds: xr.Dataset,
                      lat_val: float,
                      lon_val: float,
                      tile_size: int,
                      lat_name: str = "lat",
                      lon_name: str = "lon") -> tuple[int, int]:
    """
    Return (nnlat, nnlon) tile counters for *any* grid topology.
    """
    lat = ds[lat_name].values
    lon = ds[lon_name].values

    if lat.ndim == 1 and lon.ndim == 1:
        # regular grid — fast path
        lon_val = _wrap_lon_to_match_grid(lon_val, lon)
        j = _nearest_1d(lat, lat_val)
        i = _nearest_1d(lon, lon_val)
        nnlat = j // tile_size
        nnlon = i // tile_size
        return nnlat, nnlon

    if lat.ndim == 2 and lon.ndim == 2:
        j, i = _nearest_index_2d(lat, lon, lat_val, lon_val)
        nnlat = j // tile_size
        nnlon = i // tile_size
        return nnlat, nnlon

    raise ValueError("Unsupported lat/lon dimensionality: "
                     f"lat.ndim={lat.ndim}, lon.ndim={lon.ndim}")


def tile_suffix(nnlat: int, nnlon: int) -> str:
    """Format indices exactly like the splitter: 000y-002x."""
    return f"{nnlat:03d}y-{nnlon:03d}x"


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def cli():
    p = argparse.ArgumentParser(
        description="Locate the tile covering an arbitrary lat/lon point.")
    p.add_argument("--lat", type=float, required=True, help="Latitude (°)")
    p.add_argument("--lon", type=float, required=True, help="Longitude (°)")
    p.add_argument("--grid-file", type=Path, required=True,
                   help="Any NetCDF file that carries the *full* grid")
    p.add_argument("--tile-size", type=int, required=True,
                   help="Tile edge-length in index space (same as splitter)")
    p.add_argument("--lat-name", default="lat",
                   help="Variable name holding latitude values")
    p.add_argument("--lon-name", default="lon",
                   help="Variable name holding longitude values")
    p.add_argument("--base-name",
                   help="If given, also print BASE_{suffix}.nc")
    args = p.parse_args()

    ds = xr.open_dataset(args.grid_file, decode_times=False)

    nnlat, nnlon = find_tile_indices(ds,
                                     args.lat, args.lon,
                                     args.tile_size,
                                     args.lat_name, args.lon_name)

    suffix = tile_suffix(nnlat, nnlon)
    print(suffix)
    if args.base_name:
        print(f"{args.base_name}_{suffix}.nc")


if __name__ == "__main__":
    cli()
