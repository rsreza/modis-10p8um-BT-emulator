"""
generate_seviri.py
==================

Generate synthetic SEVIRI 10.8 um brightness temperature from the
high-resolution synthetic MODIS field produced by ``data/generate_modis.py``.

SEVIRI is spatially coarser (3 km) but temporally denser (15 min) than MODIS
(1 km, ~2 passes/day). We emulate that by:

    1. Reading the MODIS BT field (223 x 274, at overpass times).
    2. Regridding to 3 km with AREA-AVERAGING (not interpolation) to mimic
       the sensor's larger footprint.
    3. Adding SEVIRI-like sensor noise.
    4. Extending to all 33 time steps by linear temporal interpolation
       between the sparse MODIS overpasses.
    5. Adding small temporal noise to mimic atmospheric evolution at
       scales not resolved by MODIS.

Output
------
data/synthetic_seviri.nc with variables:
    BT                (time, lat, lon)  float32  [K]
    cloudy            (time, lat, lon)  uint8    {0,1}
    cloudy_fraction   (time, lat, lon)  float32  in [0, 1]

Usage
-----
    python -m data.generate_seviri
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import xarray as xr

from src.grid_spec import GRID


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SENSOR_NOISE_K: float = 0.3
TEMPORAL_NOISE_K: float = 0.2
CLOUD_COVERAGE_THRESHOLD: float = 0.5

BT_MIN_K: float = 180.0
BT_MAX_K: float = 320.0

SEED: int = 7


# ---------------------------------------------------------------------------
# Regridding (area-average from MODIS 1 km -> SEVIRI 3 km)
# ---------------------------------------------------------------------------

def _regrid_area_mean(
    field: np.ndarray,
    src_lat: np.ndarray,
    src_lon: np.ndarray,
    dst_lat: np.ndarray,
    dst_lon: np.ndarray,
) -> np.ndarray:
    """
    Area-average a 2-D field from a fine grid (src) to a coarse grid (dst).

    Each coarse pixel receives the mean of all fine pixels whose centres
    fall inside the coarse pixel's footprint. We map each fine pixel to the
    nearest coarse pixel centre, then average.
    """
    lat_idx = np.abs(src_lat[:, None] - dst_lat[None, :]).argmin(axis=1)
    lon_idx = np.abs(src_lon[:, None] - dst_lon[None, :]).argmin(axis=1)

    n_dst_lat = len(dst_lat)
    n_dst_lon = len(dst_lon)

    total = np.zeros((n_dst_lat, n_dst_lon), dtype=np.float64)
    count = np.zeros((n_dst_lat, n_dst_lon), dtype=np.float64)

    for i_lat_src, i_lat_dst in enumerate(lat_idx):
        for i_lon_src, i_lon_dst in enumerate(lon_idx):
            total[i_lat_dst, i_lon_dst] += field[i_lat_src, i_lon_src]
            count[i_lat_dst, i_lon_dst] += 1.0

    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(count > 0, total / count, np.nan)

    # Fill any empty coarse cells with the global mean (edge case only)
    if np.any(np.isnan(mean)):
        mean = np.where(np.isnan(mean), np.nanmean(mean), mean)

    return mean.astype(np.float32)


def _regrid_area_mean_nd(
    field_3d: np.ndarray,
    src_lat: np.ndarray,
    src_lon: np.ndarray,
    dst_lat: np.ndarray,
    dst_lon: np.ndarray,
) -> np.ndarray:
    """Apply _regrid_area_mean over a leading (time) axis."""
    n_time = field_3d.shape[0]
    out = np.empty((n_time, len(dst_lat), len(dst_lon)), dtype=np.float32)
    for t in range(n_time):
        out[t] = _regrid_area_mean(
            field_3d[t], src_lat, src_lon, dst_lat, dst_lon
        )
    return out


# ---------------------------------------------------------------------------
# Temporal interpolation
# ---------------------------------------------------------------------------

def _interp_time(
    field_overpasses: np.ndarray,
    overpass_hours: np.ndarray,
    all_hours: np.ndarray,
) -> np.ndarray:
    """
    Linear temporal interpolation between overpass fields.
    """
    n_lat, n_lon = field_overpasses.shape[1:]
    n_all = len(all_hours)
    out = np.empty((n_all, n_lat, n_lon), dtype=np.float32)

    for i, h in enumerate(all_hours):
        if h <= overpass_hours[0]:
            out[i] = field_overpasses[0]
        elif h >= overpass_hours[-1]:
            out[i] = field_overpasses[-1]
        else:
            j = np.searchsorted(overpass_hours, h) - 1
            h0, h1 = overpass_hours[j], overpass_hours[j + 1]
            w = (h - h0) / (h1 - h0)
            out[i] = (1 - w) * field_overpasses[j] + w * field_overpasses[j + 1]
    return out


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------

def generate_seviri_dataset(modis_path: Path, seed: int = SEED) -> xr.Dataset:
    print(f"Reading {modis_path} ...")
    modis = xr.open_dataset(modis_path)

    modis_lat = modis["lat"].values
    modis_lon = modis["lon"].values
    overpass_hours = modis["time"].values
    n_pass = len(overpass_hours)

    print(f"  MODIS overpasses at hours {list(overpass_hours)}")
    print(f"  MODIS grid: {len(modis_lat)} x {len(modis_lon)}")

    BT_modis = modis["BT"].values
    CLD_modis = modis["cloudy"].values.astype(np.uint8)

    seviri_lat = GRID.lat_seviri
    seviri_lon = GRID.lon_seviri

    print("Regridding MODIS 1 km -> SEVIRI 3 km (area-average) ...")
    BT_seviri_pass = _regrid_area_mean_nd(
        BT_modis, modis_lat, modis_lon, seviri_lat, seviri_lon,
    )
    frac_cloudy_pass = _regrid_area_mean_nd(
        CLD_modis.astype(np.float32),
        modis_lat, modis_lon, seviri_lat, seviri_lon,
    )
    CLD_seviri_pass = (frac_cloudy_pass > CLOUD_COVERAGE_THRESHOLD).astype(np.uint8)

    all_hours = GRID.times_hours
    print(f"Interpolating {n_pass} overpasses to {len(all_hours)} times ...")
    BT_seviri_all = _interp_time(BT_seviri_pass, overpass_hours, all_hours)
    frac_cloudy_all = _interp_time(frac_cloudy_pass, overpass_hours, all_hours)

    rng = np.random.default_rng(seed)
    BT_seviri_all += TEMPORAL_NOISE_K * rng.standard_normal(
        BT_seviri_all.shape
    ).astype(np.float32)
    BT_seviri_all = np.clip(BT_seviri_all, BT_MIN_K, BT_MAX_K).astype(np.float32)

    CLD_seviri_all = (frac_cloudy_all > CLOUD_COVERAGE_THRESHOLD).astype(np.uint8)

    ds = xr.Dataset(
        data_vars={
            "BT":              (("time", "lat", "lon"), BT_seviri_all),
            "cloudy":          (("time", "lat", "lon"), CLD_seviri_all),
            "cloudy_fraction": (("time", "lat", "lon"),
                                frac_cloudy_all.astype(np.float32)),
        },
        coords={
            "time": all_hours,
            "lat": seviri_lat,
            "lon": seviri_lon,
        },
        attrs={
            "description": "Synthetic SEVIRI 10.8 um brightness temperature",
            "source": "MODIS field degraded to 3 km + temporal interpolation",
            "sensor_noise_K": SENSOR_NOISE_K,
            "temporal_noise_K": TEMPORAL_NOISE_K,
            "cloud_coverage_threshold": CLOUD_COVERAGE_THRESHOLD,
        },
    )
    return ds


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic SEVIRI BT")
    parser.add_argument(
        "--in", dest="in_path", type=Path,
        default=Path("data/synthetic_modis.nc"),
        help="input MODIS NetCDF",
    )
    parser.add_argument(
        "--out", type=Path,
        default=Path("data/synthetic_seviri.nc"),
        help="output SEVIRI NetCDF",
    )
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    ds = generate_seviri_dataset(args.in_path, seed=args.seed)

    print(f"\nWriting {args.out} ...")
    ds.to_netcdf(args.out)
    size_mb = args.out.stat().st_size / 1e6
    print(f"Done.  File size: {size_mb:.2f} MB")
    print(f"Variables: {list(ds.data_vars)}")
    print(f"Shapes   : time={ds.sizes['time']}, "
          f"lat={ds.sizes['lat']}, lon={ds.sizes['lon']}")


if __name__ == "__main__":
    main()
